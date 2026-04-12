from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from peoplegator_namedfaces.retrieval.evaluate import (
    load_dataset,
    load_ground_truth,
    load_predictions,
)
from peoplegator_namedfaces.retrieval.models import QueryType


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build DET curve (FPR vs FNR) from retrieval predictions and ground truth."
    )
    parser.add_argument("--predictions", required=True, help="Path to retrieval predictions .pkl")
    parser.add_argument("--ground-truth", required=True, help="Path to retrieval ground-truth .jsonl")
    parser.add_argument("--dataset", required=True, help="Path to retrieval dataset config .json")
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=-1,
        help="Ignore label used for image self-match (default: -1). Use no value to disable.",
    )
    parser.add_argument(
        "--output-image",
        default="evaluation_artifacts/retrieval.det.png",
        help="Output DET image path (.png)",
    )
    parser.add_argument(
        "--output-csv",
        default="evaluation_artifacts/retrieval.det.csv",
        help="Output CSV with threshold,FPR,FNR",
    )
    parser.add_argument(
        "--title",
        default="DET curve - PeopleGator retrieval",
        help="Plot title",
    )
    return parser


def _collect_labels_and_scores(
    scores: np.ndarray,
    ground_truth,
    dataset,
    ignore_index: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_score: list[float] = []

    for query_idx, gt in enumerate(ground_truth):
        relevant = set()
        for face_path in gt.faces:
            try:
                relevant.add(dataset.image_index(face_path))
            except KeyError:
                continue

        ignore_gallery_idx: int | None = None
        if ignore_index is not None and gt.query_type == QueryType.IMAGE:
            try:
                ignore_gallery_idx = dataset.image_index(gt.query)
            except KeyError:
                ignore_gallery_idx = None

        for gallery_idx in range(scores.shape[1]):
            if ignore_gallery_idx is not None and gallery_idx == ignore_gallery_idx:
                continue
            y_true.append(1 if gallery_idx in relevant else 0)
            y_score.append(float(scores[query_idx, gallery_idx]))

    return np.asarray(y_true, dtype=np.int8), np.asarray(y_score, dtype=np.float64)


def compute_det(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if y_true.size == 0:
        raise ValueError("No pairs available for DET computation.")

    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("DET requires both positive and negative pairs.")

    order = np.argsort(y_score)[::-1]
    sorted_scores = y_score[order]
    sorted_true = y_true[order]

    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)

    change_points = np.where(np.diff(sorted_scores) != 0)[0]
    idx = np.r_[change_points, sorted_scores.size - 1]

    thresholds = sorted_scores[idx]
    tps = tp[idx]
    fps = fp[idx]

    fnr = (positives - tps) / positives
    fpr = fps / negatives

    # Add boundary points for complete curve.
    thresholds = np.r_[sorted_scores[0] + 1e-12, thresholds, sorted_scores[-1] - 1e-12]
    fpr = np.r_[0.0, fpr, 1.0]
    fnr = np.r_[1.0, fnr, 0.0]

    return thresholds, fpr, fnr


def save_det_csv(output_csv: Path, thresholds: np.ndarray, fpr: np.ndarray, fnr: np.ndarray) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "fpr", "fnr"])
        for thr, x_fpr, x_fnr in zip(thresholds, fpr, fnr):
            writer.writerow([f"{thr:.8f}", f"{x_fpr:.8f}", f"{x_fnr:.8f}"])


def save_det_plot(output_image: Path, fpr: np.ndarray, fnr: np.ndarray, title: str, eer: float) -> None:
    output_image.parent.mkdir(parents=True, exist_ok=True)

    # DET uses normal deviate (probit) scale on both axes.
    # Use practical clipping so axes do not explode from exact 0/1 endpoints.
    det_clip = 1e-3
    fpr_safe = np.clip(fpr, det_clip, 1.0 - det_clip)
    fnr_safe = np.clip(fnr, det_clip, 1.0 - det_clip)
    x = norm.ppf(fpr_safe)
    y = norm.ppf(fnr_safe)

    tick_perc = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40, 60, 80, 90, 95, 98, 99, 99.5, 99.8, 99.9])
    tick_pos = norm.ppf(tick_perc / 100.0)

    plt.figure(figsize=(6, 5))
    plt.plot(x, y, label=f"DET (EER={eer:.4f})")
    # Keep labels readable by showing a sparser subset of ticks.
    show_tick_perc = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 40, 60, 80, 90, 95, 99, 99.9])
    show_tick_pos = norm.ppf(show_tick_perc / 100.0)
    plt.xticks(show_tick_pos, [f"{t:g}" for t in show_tick_perc], rotation=45)
    plt.yticks(show_tick_pos, [f"{t:g}" for t in show_tick_perc])
    plt.xlim(tick_pos[0], tick_pos[-1])
    plt.ylim(tick_pos[0], tick_pos[-1])
    plt.xlabel("False Positive Rate (%) [probit scale]")
    plt.ylabel("False Negative Rate (%) [probit scale]")
    plt.title(f"{title} (probit)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    plt.close()


def main() -> int:
    args = build_parser().parse_args()

    predictions = load_predictions(args.predictions)
    ground_truth = load_ground_truth(args.ground_truth)
    dataset = load_dataset(args.dataset)

    scores = np.asarray(predictions.scores)
    y_true, y_score = _collect_labels_and_scores(
        scores=scores,
        ground_truth=ground_truth,
        dataset=dataset,
        ignore_index=args.ignore_index,
    )

    thresholds, fpr, fnr = compute_det(y_true=y_true, y_score=y_score)
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)

    output_image = Path(args.output_image)
    output_csv = Path(args.output_csv)

    save_det_csv(output_csv=output_csv, thresholds=thresholds, fpr=fpr, fnr=fnr)
    save_det_plot(output_image=output_image, fpr=fpr, fnr=fnr, title=args.title, eer=eer)

    print(f"Pairs used: {len(y_true)} (positives={int(np.sum(y_true == 1))}, negatives={int(np.sum(y_true == 0))})")
    print(f"EER: {eer:.6f}")
    print(f"Saved DET CSV: {output_csv}")
    print(f"Saved DET plot: {output_image}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

