from __future__ import annotations

from pathlib import Path
from typing import Protocol

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from .metrics import MisclassifiedItem


class ArtifactDatasetProtocol(Protocol):
    df: pd.DataFrame
    image_col: str
    images_root: Path
    idx_to_class: dict[int, str]


def save_top1_score_boxplot(
    correct_scores: list[float], wrong_scores: list[float], output_path: Path
) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        [
            correct_scores if correct_scores else [float("nan")],
            wrong_scores if wrong_scores else [float("nan")],
        ],
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


def save_top1_misclassified_previews(
    dataset: ArtifactDatasetProtocol,
    sample_indices: list[int],
    misclassified: list[MisclassifiedItem],
    limit: int,
    output_dir: Path,
    dataset_suffix: str,
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

        if not query_path.exists() or not pred_gallery_path.exists() or not true_gallery_path.exists():
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

        out_path = output_dir / f"top1_miss_{saved:03d}_{dataset_suffix}.jpg"
        canvas.save(out_path)

        metadata_path = out_path.with_suffix(".txt")
        metadata_path.write_text(
            "\n".join(
                [
                    f"query_name: {true_name}",
                    f"predicted_gallery_name: {pred_name}",
                    f"correct_gallery_name: {true_name}",
                    f"top1_cosine_score: {score:.6f}",
                    f"query_path: {query_path}",
                    f"predicted_gallery_path: {pred_gallery_path}",
                    f"correct_gallery_path: {true_gallery_path}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        saved += 1

    return saved

