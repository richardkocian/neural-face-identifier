from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from evaluation.core.artifacts import save_top1_score_boxplot
from evaluation.core.metrics import first_quartile
from peoplegator_namedfaces.retrieval.evaluate import (
    load_dataset,
    load_ground_truth,
    load_predictions,
)
from peoplegator_namedfaces.retrieval.models import QueryType


@dataclass(frozen=True)
class MisclassifiedPreviewItem:
    query_path: str
    predicted_gallery_path: str
    correct_gallery_path: str
    score: float


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
    parser.add_argument(
        "--show-misclassified-top1",
        type=int,
        default=10,
        help="Save up to N highest-scoring wrong Top-1 retrieval previews.",
    )
    parser.add_argument(
        "--misclassified-output-dir",
        type=Path,
        default=Path("evaluation_artifacts"),
        help="Directory where Top-1 misclassified previews are saved.",
    )
    return parser


def _dataset_image_path(dataset, index: int) -> str:
    sample = dataset[index]
    image_path = sample.get("image_path")
    if not isinstance(image_path, str) or not image_path:
        raise ValueError(f"Dataset sample at index {index} has invalid image_path.")
    return image_path


def _top1_scores_with_misclassified(
    scores: np.ndarray,
    ground_truth,
    dataset,
    ignore_index: int | None,
) -> tuple[list[float], list[float], int, list[MisclassifiedPreviewItem]]:
    correct_scores: list[float] = []
    wrong_scores: list[float] = []
    misclassified: list[MisclassifiedPreviewItem] = []
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
            if gt.query_type != QueryType.IMAGE:
                continue

            relevant_candidates = [
                idx
                for idx in relevant
                if (ignore_gallery_idx is None or idx != ignore_gallery_idx)
                and np.isfinite(float(query_scores[idx]))
            ]
            if not relevant_candidates:
                continue

            try:
                query_path = _dataset_image_path(dataset, dataset.image_index(gt.query))
                predicted_path = _dataset_image_path(dataset, top1_idx)
                correct_gallery_idx = int(max(relevant_candidates, key=lambda idx: float(query_scores[idx])))
                correct_path = _dataset_image_path(dataset, correct_gallery_idx)
            except (KeyError, ValueError, IndexError):
                continue

            misclassified.append(
                MisclassifiedPreviewItem(
                    query_path=query_path,
                    predicted_gallery_path=predicted_path,
                    correct_gallery_path=correct_path,
                    score=top1_score,
                )
            )

    return correct_scores, wrong_scores, skipped_queries, misclassified


def _save_top1_misclassified_previews(
    misclassified: list[MisclassifiedPreviewItem],
    limit: int,
    output_dir: Path,
    dataset_suffix: str,
) -> int:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for item in sorted(misclassified, key=lambda x: float(x.score), reverse=True)[:limit]:
        query_path = Path(item.query_path)
        predicted_path = Path(item.predicted_gallery_path)
        correct_path = Path(item.correct_gallery_path)
        if not query_path.exists() or not predicted_path.exists() or not correct_path.exists():
            continue

        query_img = Image.open(query_path).convert("RGB").resize((224, 224))
        predicted_img = Image.open(predicted_path).convert("RGB").resize((224, 224))
        correct_img = Image.open(correct_path).convert("RGB").resize((224, 224))

        canvas = Image.new("RGB", (224 * 3 + 40, 300), color=(250, 250, 250))
        draw = ImageDraw.Draw(canvas)
        canvas.paste(query_img, (10, 10))
        canvas.paste(predicted_img, (244, 10))
        canvas.paste(correct_img, (478, 10))
        draw.text((10, 240), "Query", fill=(0, 0, 0))
        draw.text((244, 240), "Predicted gallery", fill=(0, 0, 0))
        draw.text((478, 240), "Correct gallery", fill=(0, 0, 0))
        draw.text((10, 270), f"Top-1 cosine score: {item.score:.4f}", fill=(0, 0, 0))

        out_path = output_dir / f"top1_miss_{saved:03d}_{dataset_suffix}.png"
        canvas.save(out_path)

        metadata_path = out_path.with_suffix(".txt")
        metadata_path.write_text(
            "\n".join(
                [
                    f"top1_cosine_score: {item.score:.6f}",
                    f"query_path: {query_path}",
                    f"predicted_gallery_path: {predicted_path}",
                    f"correct_gallery_path: {correct_path}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        saved += 1

    return saved


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

    correct_scores, wrong_scores, skipped_queries, misclassified = _top1_scores_with_misclassified(
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
    saved_misses = 0
    if args.show_misclassified_top1 > 0:
        saved_misses = _save_top1_misclassified_previews(
            misclassified=misclassified,
            limit=args.show_misclassified_top1,
            output_dir=args.misclassified_output_dir,
            dataset_suffix="people_gator_retrieval",
        )

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
    if args.show_misclassified_top1 > 0:
        print(
            f"Saved Top-1 misclassified previews (top {args.show_misclassified_top1} highest-scoring mismatches): {saved_misses} -> "
            f"{args.misclassified_output_dir.resolve()}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
