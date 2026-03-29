from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

from datasets.wiki_face_dataset import WikiFaceDataset


def _default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root / "wiki_face_112_fin" / "wiki_face_112_fin.test.csv",
        repo_root / "wiki_face_112_fin",
    )


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_parser() -> argparse.ArgumentParser:
    default_csv, default_images = _default_paths()
    parser = argparse.ArgumentParser(
        description="Evaluate a face-embedding model on WikiFace with gallery/query identification."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=default_csv,
        help="Path to WikiFace CSV metadata file.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=default_images,
        help="Root directory containing WikiFace images.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3",
        help="timm model identifier.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--device",
        type=str,
        default=_default_device(),
        help="Inference device (cpu/cuda/mps).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If >0, evaluate only first N samples for a quick smoke run.",
    )
    parser.add_argument(
        "--show-misclassified-top1",
        type=int,
        default=0,
        help="If >0, save up to N Top-1 mistakes (query vs predicted gallery).",
    )
    parser.add_argument(
        "--misclassified-output-dir",
        type=Path,
        default=Path("evaluation_artifacts"),
        help="Directory where misclassified Top-1 image pairs are saved.",
    )
    parser.add_argument(
        "--top1-boxplot-path",
        type=Path,
        default=Path("evaluation_artifacts") / "top1_cosine_boxplot.png",
        help="Where to save boxplot of Top-1 cosine scores (correct vs misclassified).",
    )
    return parser


def _resolve_dataset_indices(dataset: WikiFaceDataset | Subset) -> list[int]:
    if isinstance(dataset, Subset):
        parent_indices = _resolve_dataset_indices(dataset.dataset)
        return [parent_indices[int(i)] for i in dataset.indices]
    return list(range(len(dataset)))


def _extract_embeddings(
    dataset: WikiFaceDataset | Subset,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    # Optional dataset truncation for fast smoke tests.
    if max_samples > 0:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))

    sample_indices = _resolve_dataset_indices(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    emb_list: list[torch.Tensor] = []
    label_list: list[torch.Tensor] = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            embs = model(images)
            # The selected timm model returns raw embeddings; enforce unit-length vectors
            # so cosine similarity is equivalent to a dot product.
            embs = F.normalize(embs, dim=1)
            emb_list.append(embs.cpu())
            label_list.append(labels.cpu())

    if not emb_list:
        raise ValueError("Dataset is empty, no embeddings were generated.")

    return torch.cat(emb_list, dim=0), torch.cat(label_list, dim=0), sample_indices


def _gallery_query_topk(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    ks: tuple[int, ...] = (1, 5),
) -> tuple[dict[int, float], list[dict[str, int | float]], list[float], list[float]]:
    # Build a closed-set protocol from a single split:
    # - gallery: first sample per identity
    # - query: remaining samples of the same identities
    unique_labels = labels.unique(sorted=True)
    gallery_idx: list[int] = []
    query_idx: list[int] = []

    for label_value in unique_labels.tolist():
        cls_idx = torch.nonzero(labels == label_value, as_tuple=False).flatten()
        # Single-image identities cannot produce a query and are skipped.
        if cls_idx.numel() < 2:
            continue
        gallery_idx.append(int(cls_idx[0].item()))
        query_idx.extend(int(x.item()) for x in cls_idx[1:])

    if not query_idx:
        raise ValueError(
            "Need at least one identity with >=2 images to build query samples."
        )

    gallery_emb = embeddings[gallery_idx]
    gallery_labels = labels[gallery_idx]
    query_emb = embeddings[query_idx]
    query_labels = labels[query_idx]

    # Because embeddings are L2-normalized, dot product gives cosine similarity.
    sims = query_emb @ gallery_emb.T
    max_k = min(max(ks), sims.shape[1])
    topk_indices = sims.topk(k=max_k, dim=1).indices
    topk_labels = gallery_labels[topk_indices]

    metrics: dict[int, float] = {}
    for k in tuple(ks):
        k_eff = min(k, topk_labels.shape[1])
        hits = (topk_labels[:, :k_eff] == query_labels.unsqueeze(1)).any(dim=1)
        metrics[k] = float(hits.float().mean().item())

    label_to_gallery_embedding_idx = {
        int(label.item()): gallery_idx[pos] for pos, label in enumerate(gallery_labels)
    }

    top1_gallery_idx = topk_indices[:, 0]
    top1_pred_labels = gallery_labels[top1_gallery_idx]
    top1_scores = sims.gather(1, top1_gallery_idx.unsqueeze(1)).squeeze(1)
    top1_correct_mask = top1_pred_labels == query_labels
    correct_scores = [float(x) for x in top1_scores[top1_correct_mask].tolist()]
    wrong_scores = [float(x) for x in top1_scores[~top1_correct_mask].tolist()]

    wrong_query_local_idx = torch.nonzero(
        top1_pred_labels != query_labels, as_tuple=False
    ).flatten()

    misclassified: list[dict[str, int | float]] = []
    for wrong_pos in wrong_query_local_idx:
        wrong_idx = int(wrong_pos.item())
        gallery_local_idx = int(top1_gallery_idx[wrong_idx].item())
        query_embedding_idx = query_idx[wrong_idx]
        gallery_embedding_idx = int(gallery_idx[gallery_local_idx])
        true_label = int(query_labels[wrong_idx].item())
        true_gallery_embedding_idx = int(label_to_gallery_embedding_idx[true_label])
        misclassified.append(
            {
                "query_embedding_idx": query_embedding_idx,
                "gallery_embedding_idx": gallery_embedding_idx,
                "true_gallery_embedding_idx": true_gallery_embedding_idx,
                "true_label": true_label,
                "pred_label": int(top1_pred_labels[wrong_idx].item()),
                "score": float(sims[wrong_idx, gallery_local_idx].item()),
            }
        )
    return metrics, misclassified, correct_scores, wrong_scores


def _describe_scores(scores: list[float]) -> tuple[int, float, float]:
    if not scores:
        return 0, float("nan"), float("nan")
    return len(scores), float(statistics.mean(scores)), float(statistics.median(scores))


def _first_quartile(scores: list[float]) -> float:
    if not scores:
        return float("nan")
    if len(scores) == 1:
        return float(scores[0])
    quartiles = statistics.quantiles(scores, n=4, method="inclusive")
    return float(quartiles[0])


def _save_top1_score_boxplot(
    correct_scores: list[float], wrong_scores: list[float], output_path: Path
) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        [correct_scores if correct_scores else [float("nan")], wrong_scores if wrong_scores else [float("nan")]],
        tick_labels=["Top-1 correct", "Top-1 wrong"],
        patch_artist=True,
    )
    ax.set_ylabel("Cosine score")
    ax.set_title("Top-1 cosine score distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _save_top1_misclassified_previews(
    dataset: WikiFaceDataset,
    sample_indices: list[int],
    misclassified: list[dict[str, int | float]],
    limit: int,
    output_dir: Path,
) -> int:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for item in misclassified[:limit]:
        query_emb_idx = int(item["query_embedding_idx"])
        pred_gallery_emb_idx = int(item["gallery_embedding_idx"])
        true_gallery_emb_idx = int(item["true_gallery_embedding_idx"])
        true_label_idx = int(item["true_label"])
        pred_label_idx = int(item["pred_label"])
        score = float(item["score"])

        query_ds_idx = sample_indices[query_emb_idx]
        pred_gallery_ds_idx = sample_indices[pred_gallery_emb_idx]
        true_gallery_ds_idx = sample_indices[true_gallery_emb_idx]

        query_rel = Path(dataset.df.iloc[query_ds_idx][dataset.image_col])
        pred_gallery_rel = Path(dataset.df.iloc[pred_gallery_ds_idx][dataset.image_col])
        true_gallery_rel = Path(dataset.df.iloc[true_gallery_ds_idx][dataset.image_col])
        query_path = dataset.images_root / query_rel
        pred_gallery_path = dataset.images_root / pred_gallery_rel
        true_gallery_path = dataset.images_root / true_gallery_rel

        if (
            not query_path.exists()
            or not pred_gallery_path.exists()
            or not true_gallery_path.exists()
        ):
            continue

        query_img = Image.open(query_path).convert("RGB").resize((224, 224))
        pred_gallery_img = Image.open(pred_gallery_path).convert("RGB").resize((224, 224))
        true_gallery_img = Image.open(true_gallery_path).convert("RGB").resize((224, 224))

        canvas = Image.new("RGB", (224 * 3 + 40, 300), color=(250, 250, 250))
        draw = ImageDraw.Draw(canvas)
        canvas.paste(query_img, (10, 10))
        canvas.paste(pred_gallery_img, (244, 10))
        canvas.paste(true_gallery_img, (478, 10))

        true_name = dataset.idx_to_class.get(true_label_idx, str(true_label_idx))
        pred_name = dataset.idx_to_class.get(pred_label_idx, str(pred_label_idx))
        draw.text((10, 240), f"Query: {true_name}", fill=(0, 0, 0))
        draw.text((244, 240), f"Predicted gallery: {pred_name}", fill=(0, 0, 0))
        draw.text((478, 240), f"Correct gallery: {true_name}", fill=(0, 0, 0))
        draw.text((10, 270), f"Top-1 cosine score: {score:.4f}", fill=(0, 0, 0))

        out_path = output_dir / f"top1_miss_{saved:03d}.jpg"
        canvas.save(out_path)
        saved += 1

    return saved


def main() -> int:
    args = _build_parser().parse_args()

    # Dataset returns (image_tensor, class_index) where image tensor is normalized
    # to [-1, 1] in the dataset preprocessing pipeline.
    dataset = WikiFaceDataset(csv_path=args.csv_path, images_root=args.images_root)

    if len(dataset) == 0:
        print("Dataset is empty, nothing to evaluate.")
        return 1

    device = torch.device(args.device)
    # num_classes=0 removes the classification head and returns embedding vectors.
    model = timm.create_model(args.model_id, pretrained=True, num_classes=0).to(device)
    model.eval()

    embeddings, labels, sample_indices = _extract_embeddings(
        dataset=dataset,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    metrics, misclassified, correct_scores, wrong_scores = _gallery_query_topk(
        embeddings, labels, ks=(1, 5)
    )

    boxplot_path = _save_top1_score_boxplot(
        correct_scores=correct_scores,
        wrong_scores=wrong_scores,
        output_path=args.top1_boxplot_path,
    )

    saved_misses = 0

    print("WikiFace evaluation finished.")
    print(f"CSV: {args.csv_path.resolve()}")
    print(f"Images root: {args.images_root.resolve()}")
    print(f"Samples used: {embeddings.shape[0]}")
    print(f"Classes in samples: {labels.unique().numel()}")
    print(f"Device: {device}")
    print(f"Model: {args.model_id}")
    print(f"Top-1 (gallery/query): {metrics[1] * 100:.2f}%")
    print(f"Top-5 (gallery/query): {metrics[5] * 100:.2f}%")

    correct_n, correct_mean, correct_median = _describe_scores(correct_scores)
    wrong_n, wrong_mean, wrong_median = _describe_scores(wrong_scores)
    correct_q1 = _first_quartile(correct_scores)
    high_score_misclassified = [
        item for item in misclassified if float(item["score"]) > correct_q1
    ]

    if args.show_misclassified_top1 > 0:
        saved_misses = _save_top1_misclassified_previews(
            dataset=dataset,
            sample_indices=sample_indices,
            misclassified=high_score_misclassified,
            limit=args.show_misclassified_top1,
            output_dir=args.misclassified_output_dir,
        )

    print(
        "Top-1 cosine (correct) -> "
        f"count: {correct_n}, mean: {correct_mean:.4f}, median: {correct_median:.4f}"
    )
    print(f"Top-1 cosine (correct) -> Q1: {correct_q1:.4f}")
    print(
        "Top-1 cosine (wrong) -> "
        f"count: {wrong_n}, mean: {wrong_mean:.4f}, median: {wrong_median:.4f}"
    )
    print(
        "Top-1 wrong with score > Q1(correct) -> "
        f"count: {len(high_score_misclassified)}"
    )
    print(f"Saved Top-1 cosine boxplot: {boxplot_path}")

    if args.show_misclassified_top1 > 0:
        print(
            f"Saved Top-1 misclassified previews (score > Q1(correct)): {saved_misses} -> "
            f"{args.misclassified_output_dir.resolve()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

