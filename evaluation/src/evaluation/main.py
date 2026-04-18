from __future__ import annotations

from typing import Any, cast

import numpy as np
import timm
import torch

from datasets.people_gator_dataset import PeopleGatorDataset
from datasets.wiki_face_dataset import WikiFaceDataset

from .artifacts import save_top1_misclassified_previews, save_top1_score_boxplot
from .cli import build_parser
from .det_curve import compute_det, save_det_csv, save_det_plot
from .embeddings import extract_embeddings
from .metrics import (
    describe_scores,
    first_quartile,
    gallery_query_pair_labels_scores,
    gallery_query_topk,
)


def _build_dataset(default_dataset: str, args) -> WikiFaceDataset | PeopleGatorDataset:
    if default_dataset == "wiki_face":
        return WikiFaceDataset(csv_path=args.csv_path, images_root=args.images_root)
    if default_dataset == "people_gator":
        return PeopleGatorDataset(jsonl_path=args.jsonl_path, images_root=args.images_root)
    raise ValueError(f"Unsupported dataset backend: {default_dataset}")


def _main_with_default_dataset(default_dataset: str) -> int:
    args = build_parser(default_dataset=default_dataset).parse_args()
    dataset_name = args.dataset

    # Dataset returns (image_tensor, class_index) where image tensor is normalized
    # to [-1, 1] in the dataset preprocessing pipeline.
    dataset = _build_dataset(dataset_name, args)

    if len(dataset) == 0:
        print("Dataset is empty, nothing to evaluate.")
        return 1

    device = torch.device(args.device)
    # num_classes=0 removes the classification head and returns embedding vectors.
    model = timm.create_model(args.model_id, pretrained=True, num_classes=0).to(device)
    model.eval()

    embeddings, labels, sample_indices = extract_embeddings(
        dataset=cast(Any, dataset),
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    metrics, misclassified, correct_scores, wrong_scores = gallery_query_topk(
        embeddings, labels, ks=(1, 5)
    )

    boxplot_path = save_top1_score_boxplot(
        correct_scores=correct_scores,
        wrong_scores=wrong_scores,
        output_path=args.top1_boxplot_path,
    )

    saved_misses = 0

    print("Evaluation finished.")
    print(f"Dataset backend: {dataset_name}")
    if dataset_name == "wiki_face":
        print(f"CSV: {args.csv_path.resolve()}")
    else:
        print(f"JSONL: {args.jsonl_path.resolve()}")
    print(f"Images root: {args.images_root.resolve()}")
    print(f"Samples used: {embeddings.shape[0]}")
    print(f"Classes in samples: {labels.unique().numel()}")
    print(f"Device: {device}")
    print(f"Model: {args.model_id}")
    print(f"Top-1 (gallery/query): {metrics[1] * 100:.2f}%")
    print(f"Top-5 (gallery/query): {metrics[5] * 100:.2f}%")

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
            dataset_suffix=dataset_name,
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

    if args.det_enabled:
        y_true_t, y_score_t = gallery_query_pair_labels_scores(embeddings=embeddings, labels=labels)
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
        print(f"DET EER: {det_eer:.6f}")
        print(f"Saved DET CSV: {det_csv_path}")
        print(f"Saved DET plot: {det_image_path}")

    if args.show_misclassified_top1 > 0:
        print(
            f"Saved Top-1 misclassified previews (top {args.show_misclassified_top1} highest-scoring mismatches): {saved_misses} -> "
            f"{args.misclassified_output_dir.resolve()}"
        )
    return 0


def main_wikiface() -> int:
    return _main_with_default_dataset(default_dataset="wiki_face")


def main_people_gator() -> int:
    return _main_with_default_dataset(default_dataset="people_gator")

