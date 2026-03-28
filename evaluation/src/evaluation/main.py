from __future__ import annotations

import argparse
from pathlib import Path

from datasets.wiki_face_dataset import WikiFaceDataset


def _default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root / "wiki_face_112_fin" / "wiki_face_112_fin.csv",
        repo_root / "wiki_face_112_fin",
    )


def _build_parser() -> argparse.ArgumentParser:
    default_csv, default_images = _default_paths()
    parser = argparse.ArgumentParser(description="Load and inspect WikiFace dataset.")
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
        "--sample-index",
        type=int,
        default=0,
        help="Dataset sample index to preview.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    dataset = WikiFaceDataset(csv_path=args.csv_path, images_root=args.images_root)

    print("WikiFace dataset loaded successfully.")
    print(f"CSV: {args.csv_path.resolve()}")
    print(f"Images root: {args.images_root.resolve()}")
    print(f"Samples: {len(dataset)}")
    print(f"Classes: {len(dataset.class_to_idx)}")

    if len(dataset) == 0:
        print("Dataset is empty, nothing to preview.")
        return 0

    sample_index = max(0, min(args.sample_index, len(dataset) - 1))
    image, label = dataset[sample_index]
    label_idx = int(label.item())
    label_name = dataset.idx_to_class[label_idx]
    print(
        "First preview sample -> "
        f"index: {sample_index}, image shape: {tuple(image.shape)}, "
        f"label idx: {label_idx}, label name: {label_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

