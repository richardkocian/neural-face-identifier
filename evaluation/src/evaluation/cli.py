from __future__ import annotations

import argparse
from pathlib import Path

import torch


def default_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    return {
        "wiki_face_csv": repo_root / "wiki_face_112_fin" / "wiki_face_112_fin.test.csv",
        "wiki_face_images": repo_root / "wiki_face_112_fin",
        "people_gator_jsonl": repo_root
        / "people_gator" / "people_gator__corresponding_faces__2026-02-11.test.cleaned.jsonl",
        "people_gator_images": repo_root / "people_gator" / "people_gator__data",
    }


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser(default_dataset: str = "wiki_face") -> argparse.ArgumentParser:
    defaults = default_paths()
    if default_dataset not in {"wiki_face", "people_gator"}:
        raise ValueError(f"Unsupported default dataset: {default_dataset}")
    default_images_root = (
        defaults["people_gator_images"]
        if default_dataset == "people_gator"
        else defaults["wiki_face_images"]
    )
    default_boxplot_path = (
        Path("evaluation_artifacts")
        / f"top1_cosine_boxplot_{default_dataset.replace('wiki_face', 'wikiface')}.png"
    )
    parser = argparse.ArgumentParser(
        description="Evaluate a face-embedding model with gallery/query identification."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("wiki_face", "people_gator"),
        default=default_dataset,
        help="Dataset backend to evaluate.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=defaults["wiki_face_csv"],
        help="Path to WikiFace CSV metadata file (used when --dataset=wiki_face).",
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=defaults["people_gator_jsonl"],
        help="Path to PeopleGator JSONL metadata file (used when --dataset=people_gator).",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=default_images_root,
        help=(
            "Root directory containing image files. "
            "For --dataset=people_gator default can be overridden with --images-root."
        ),
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
        default=default_device(),
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
        default=default_boxplot_path,
        help="Where to save boxplot of Top-1 cosine scores (correct vs misclassified).",
    )
    return parser
