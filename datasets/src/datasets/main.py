import argparse
from pathlib import Path

from .wiki_face_dataset import make_dataset_split


def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic identity-based train/test split for WikiFace CSV."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the source WikiFace CSV file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic splitting (default: 42).",
    )
    parser.add_argument(
        "--dataset-split",
        type=float,
        default=80.0,
        help="Train split as percent or ratio. Examples: 80 or 0.8 (default: 80).",
    )
    return parser.parse_args()


def main()->int:
    args = parse_args()
    make_dataset_split(args.dataset_path,args.seed,args.dataset_split)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())