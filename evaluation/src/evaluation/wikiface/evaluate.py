from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import timm
import torch

from datasets.wiki_face_dataset import WikiFaceDataset

from ..core.artifacts import save_top1_misclassified_previews, save_top1_score_boxplot
from ..core.checkpoints import load_finetuned_state_dict
from ..core.embeddings import extract_embeddings
from ..core.metrics import (
    describe_scores,
    first_quartile,
    gallery_query_topk_with_pair_labels_scores,
)
from ..people_gator.retrieval.det_curve import compute_det, save_det_csv, save_det_plot

DEFAULT_TIMM_MODEL_ID = "hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3"


def _default_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[4]
    return {
        "wiki_face_csv": repo_root / "wiki_face_112_fin" / "wiki_face_112_fin.test.csv",
        "wiki_face_images": repo_root / "wiki_face_112_fin",
    }


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate WikiFace embeddings with gallery/query Top-1/Top-5 metrics and save Top-1 boxplot + DET curve."
        )
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=defaults["wiki_face_csv"],
        help="Path to WikiFace CSV metadata file.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=defaults["wiki_face_images"],
        help="Root directory containing WikiFace images.",
    )
    model_group.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="timm model identifier for pretrained evaluation.",
    )
    model_group.add_argument(
        "--finetuned-model",
        type=Path,
        default=None,
        help="Path to a finetuned .pth checkpoint (for example step_20.pth).",
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
        default=10,
        help="Save up to N Top-1 mistakes (query vs predicted gallery).",
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
        default=Path("evaluation_artifacts") / "top1_cosine_boxplot_wikiface.png",
        help="Where to save boxplot of Top-1 cosine scores (correct vs misclassified).",
    )
    parser.add_argument(
        "--det-image-path",
        type=Path,
        default=Path("evaluation_artifacts") / "top1_det_wikiface.png",
        help="Where to save DET plot image (.png).",
    )
    parser.add_argument(
        "--det-csv-path",
        type=Path,
        default=Path("evaluation_artifacts") / "top1_det_wikiface.csv",
        help="Where to save DET CSV with threshold,FPR,FNR.",
    )
    parser.add_argument(
        "--metrics-csv-path",
        type=Path,
        default=Path("evaluation_artifacts") / "wikiface.metrics.csv",
        help="Where to save metrics CSV with Top-1/Top-5 accuracy.",
    )
    parser.add_argument(
        "--det-title",
        type=str,
        default="DET curve - WikiFace",
        help="Plot title for DET curve output.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    dataset = WikiFaceDataset(csv_path=args.csv_path, images_root=args.images_root)
    if len(dataset) == 0:
        print("Dataset is empty, nothing to evaluate.")
        return 1

    device = torch.device(args.device)
    if args.finetuned_model is not None:
        model = timm.create_model(DEFAULT_TIMM_MODEL_ID, pretrained=False, num_classes=0).to(device)
        finetuned_state = load_finetuned_state_dict(args.finetuned_model)
        load_result = model.load_state_dict(finetuned_state, strict=False)
        unexpected = [
            key for key in load_result.unexpected_keys if key not in {"head.weight", "head.bias"}
        ]
        if load_result.missing_keys or unexpected:
            raise ValueError(
                "Failed to load finetuned checkpoint. "
                f"Missing keys: {load_result.missing_keys}. Unexpected keys: {unexpected}."
            )
        model_source = f"finetuned checkpoint: {args.finetuned_model.resolve()}"
    else:
        assert args.model_id is not None
        model = timm.create_model(args.model_id, pretrained=True, num_classes=0).to(device)
        model_source = f"pretrained model-id: {args.model_id}"
    model.eval()

    embeddings, labels, sample_indices = extract_embeddings(
        dataset=cast(Any, dataset),
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    metrics, misclassified, correct_scores, wrong_scores, y_true_t, y_score_t = (
        gallery_query_topk_with_pair_labels_scores(embeddings, labels, ks=(1, 5, 10))
    )

    boxplot_path = save_top1_score_boxplot(
        correct_scores=correct_scores,
        wrong_scores=wrong_scores,
        output_path=args.top1_boxplot_path,
    )

    saved_misses = 0
    correct_n, correct_mean, correct_median = describe_scores(correct_scores)
    wrong_n, wrong_mean, wrong_median = describe_scores(wrong_scores)
    correct_q1 = first_quartile(correct_scores)
    high_score_misclassified = [item for item in misclassified if float(item["score"]) > correct_q1]
    top_score_misclassified = sorted(
        misclassified,
        key=lambda item: float(item["score"]),
        reverse=True,
    )

    if args.show_misclassified_top1 > 0:
        saved_misses = save_top1_misclassified_previews(
            dataset=cast(Any, dataset),
            sample_indices=sample_indices,
            misclassified=top_score_misclassified,
            limit=args.show_misclassified_top1,
            output_dir=args.misclassified_output_dir,
            dataset_suffix="wiki_face",
        )

    thresholds, fpr, fnr = compute_det(
        y_true=y_true_t.cpu().numpy().astype(np.int8),
        y_score=y_score_t.cpu().numpy().astype(np.float64),
    )
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    det_eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    det_image_path = args.det_image_path.resolve()
    det_csv_path = args.det_csv_path.resolve()
    save_det_csv(output_csv=det_csv_path, thresholds=thresholds, fpr=fpr, fnr=fnr)
    save_det_plot(
        output_image=det_image_path,
        fpr=fpr,
        fnr=fnr,
        title=args.det_title,
        eer=det_eer,
    )

    # Save metrics CSV
    metrics_csv_path = args.metrics_csv_path.resolve()
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_csv_path.open("w", encoding="utf-8") as f:
        f.write("top_k,accuracy\n")
        f.write(f"1,{metrics[1]}\n")
        f.write(f"5,{metrics[5]}\n")
        f.write(f"10,{metrics[10]}\n")

    print("WikiFace evaluation finished.")
    print(f"CSV: {args.csv_path.resolve()}")
    print(f"Images root: {args.images_root.resolve()}")
    print(f"Samples used: {embeddings.shape[0]}")
    print(f"Classes in samples: {labels.unique().numel()}")
    print(f"Device: {device}")
    print(f"Model: {model_source}")
    print(f"Top-1 (gallery/query): {metrics[1] * 100:.2f}%")
    print(f"Top-5 (gallery/query): {metrics[5] * 100:.2f}%")
    print(f"Top-10 (gallery/query): {metrics[10] * 100:.2f}%")
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
    print(f"DET EER: {det_eer:.6f}")
    print(f"Saved DET CSV: {det_csv_path}")
    print(f"Saved DET plot: {det_image_path}")

    if args.show_misclassified_top1 > 0:
        print(
            f"Saved Top-1 misclassified previews (top {args.show_misclassified_top1} highest-scoring mismatches): {saved_misses} -> "
            f"{args.misclassified_output_dir.resolve()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
