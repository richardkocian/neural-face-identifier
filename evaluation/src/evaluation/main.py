from __future__ import annotations

import argparse
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
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
    return parser


def _extract_embeddings(
    dataset: WikiFaceDataset | Subset,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Optional dataset truncation for fast smoke tests.
    if max_samples > 0:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))

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

    return torch.cat(emb_list, dim=0), torch.cat(label_list, dim=0)


def _gallery_query_topk(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    ks: tuple[int, ...] = (1, 5),
) -> dict[int, float]:
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
    for k in ks:
        k_eff = min(k, topk_labels.shape[1])
        hits = (topk_labels[:, :k_eff] == query_labels.unsqueeze(1)).any(dim=1)
        metrics[k] = float(hits.float().mean().item())
    return metrics


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

    embeddings, labels = _extract_embeddings(
        dataset=dataset,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    metrics = _gallery_query_topk(embeddings, labels, ks=(1, 5))

    print("WikiFace evaluation finished.")
    print(f"CSV: {args.csv_path.resolve()}")
    print(f"Images root: {args.images_root.resolve()}")
    print(f"Samples used: {embeddings.shape[0]}")
    print(f"Classes in samples: {labels.unique().numel()}")
    print(f"Device: {device}")
    print(f"Model: {args.model_id}")
    print(f"Top-1 (gallery/query): {metrics[1] * 100:.2f}%")
    print(f"Top-5 (gallery/query): {metrics[5] * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

