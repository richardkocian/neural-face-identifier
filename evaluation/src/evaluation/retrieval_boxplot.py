from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np

from evaluation.artifacts import save_top1_score_boxplot
from evaluation.metrics import first_quartile
from peoplegator_namedfaces.retrieval.evaluate import (
    load_dataset,
    load_ground_truth,
    load_predictions,
)
from peoplegator_namedfaces.retrieval.models import QueryType


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Top-1 cosine boxplot (correct vs wrong) from retrieval predictions."
    )
    parser.add_argument("--predictions", required=True, help="Path to retrieval predictions .pkl")
    parser.add_argument("--ground-truth", required=True, help="Path to retrieval ground-truth .jsonl")
    parser.add_argument("--dataset", required=True, help="Path to retrieval dataset config .json")
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=-1,
        help="Ignore self-match for image queries when selecting Top-1 (default: -1).",
    )
    parser.add_argument(
        "--output-image",
        default="evaluation_artifacts/retrieval.top1_cosine_boxplot.png",
        help="Output boxplot image path (.png)",
    )
    return parser


def _top1_scores(
    scores: np.ndarray,
    ground_truth,
    dataset,
    ignore_index: int | None,
) -> tuple[list[float], list[float], int]:
    correct_scores: list[float] = []
    wrong_scores: list[float] = []
    skipped_queries = 0

    for query_idx, gt in enumerate(ground_truth):
        relevant: set[int] = set()
        for face_path in gt.faces:
            try:
                relevant.add(dataset.image_index(face_path))
            except KeyError:
                continue

        query_scores = np.asarray(scores[query_idx], dtype=np.float64)

        ignore_gallery_idx: int | None = None
        if ignore_index is not None and gt.query_type == QueryType.IMAGE:
            try:
                ignore_gallery_idx = dataset.image_index(gt.query)
            except KeyError:
                ignore_gallery_idx = None

        if ignore_gallery_idx is not None:
            query_scores = query_scores.copy()
            query_scores[ignore_gallery_idx] = -np.inf

        if not np.isfinite(query_scores).any():
            skipped_queries += 1
            continue

        top1_idx = int(np.argmax(query_scores))
        top1_score = float(query_scores[top1_idx])

        if top1_idx in relevant:
            correct_scores.append(top1_score)
        else:
            wrong_scores.append(top1_score)

    return correct_scores, wrong_scores, skipped_queries


def _distribution_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": float("nan"),
            "q1": float("nan"),
            "median": float("nan"),
            "q3": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "q1": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q3": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
    }


def _print_distribution_stats(label: str, stats: dict[str, float | int]) -> None:
    print(
        f"{label} -> "
        f"count: {stats['count']}, "
        f"min: {float(stats['min']):.4f}, "
        f"Q1: {float(stats['q1']):.4f}, "
        f"median: {float(stats['median']):.4f}, "
        f"Q3: {float(stats['q3']):.4f}, "
        f"max: {float(stats['max']):.4f}, "
        f"mean: {float(stats['mean']):.4f}, "
        f"std: {float(stats['std']):.4f}"
    )


def main() -> int:
    args = build_parser().parse_args()

    predictions = load_predictions(args.predictions)
    ground_truth = load_ground_truth(args.ground_truth)
    dataset = load_dataset(args.dataset)

    scores = np.asarray(predictions.scores)
    if scores.ndim != 2:
        raise ValueError(f"Expected score matrix with 2 dimensions, got shape: {scores.shape}")

    if len(ground_truth) != scores.shape[0]:
        raise ValueError(
            "Ground-truth query count does not match prediction rows: "
            f"{len(ground_truth)} vs {scores.shape[0]}"
        )

    correct_scores, wrong_scores, skipped_queries = _top1_scores(
        scores=scores,
        ground_truth=ground_truth,
        dataset=dataset,
        ignore_index=args.ignore_index,
    )

    output_image = Path(args.output_image)
    saved_path = save_top1_score_boxplot(
        correct_scores=correct_scores,
        wrong_scores=wrong_scores,
        output_path=output_image,
    )

    correct_stats = _distribution_stats(correct_scores)
    wrong_stats = _distribution_stats(wrong_scores)
    correct_q1 = first_quartile(correct_scores)
    high_score_wrong = [x for x in wrong_scores if x > correct_q1] if np.isfinite(correct_q1) else []

    print("Top-1 retrieval boxplot finished.")
    print(f"Queries used: {len(ground_truth) - skipped_queries} / {len(ground_truth)}")
    print(f"Skipped queries: {skipped_queries}")
    _print_distribution_stats("Top-1 cosine (correct)", correct_stats)
    _print_distribution_stats("Top-1 cosine (wrong)", wrong_stats)
    if high_score_wrong:
        print(
            "Top-1 wrong with score > Q1(correct) -> "
            f"count: {len(high_score_wrong)}, mean: {statistics.mean(high_score_wrong):.4f}"
        )
    else:
        print("Top-1 wrong with score > Q1(correct) -> count: 0")
    print(f"Saved Top-1 cosine boxplot: {saved_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

