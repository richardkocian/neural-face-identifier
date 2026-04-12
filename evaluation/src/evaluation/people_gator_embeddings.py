from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import timm
import torch

from datasets.people_gator_dataset import PeopleGatorDataset

from .embeddings import extract_embeddings


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _slugify_model_id(model_id: str) -> str:
    return (
        model_id.replace("hf_hub:", "")
        .replace("/", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate face embeddings for PeopleGator using PeopleGatorDataset and a timm model."
        )
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=Path("people_gator/people_gator__corresponding_faces__2026-02-11.test.cleaned.jsonl"),
        help="PeopleGator JSONL metadata file.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("people_gator/people_gator__data"),
        help="Root directory containing images referenced by JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for embeddings.npy and image_paths.txt. "
            "If omitted, .embeddings/<model_slug> is used."
        ),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3",
        help="timm model id to use for embedding extraction.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", type=str, default=_default_device())
    parser.add_argument("--image-col", type=str, default="face")
    parser.add_argument("--label-col", type=str, default="person_name")
    parser.add_argument(
        "--no-deduplicate-rows",
        action="store_true",
        help="Disable metadata row deduplication in PeopleGatorDataset.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(".embeddings") / _slugify_model_id(args.model_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PeopleGatorDataset(
        jsonl_path=args.jsonl_path,
        images_root=args.images_root,
        image_col=args.image_col,
        label_col=args.label_col,
        deduplicate_rows=not args.no_deduplicate_rows,
    )
    if len(dataset) == 0:
        raise ValueError("PeopleGatorDataset is empty, cannot generate embeddings.")

    device = torch.device(args.device)
    model = timm.create_model(args.model_id, pretrained=True, num_classes=0).to(device)
    model.eval()

    embeddings, _, sample_indices = extract_embeddings(
        dataset=dataset,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )

    sampled_df = dataset.df.iloc[sample_indices]
    image_paths = sampled_df[args.image_col].astype(str).tolist()

    embeddings_path = output_dir / "embeddings.npy"
    image_paths_path = output_dir / "image_paths.txt"
    np.save(embeddings_path, embeddings.numpy())
    image_paths_path.write_text("\n".join(image_paths) + "\n")

    print(f"Saved embeddings: {embeddings_path}")
    print(f"Saved image paths: {image_paths_path}")
    print(f"Samples: {len(image_paths)}")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print("Use these files in retrieval dataset config as image_embeddings and image_paths.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

