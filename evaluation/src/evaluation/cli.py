from __future__ import annotations

import argparse
from pathlib import Path

import torch


def default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root / "wiki_face_112_fin" / "wiki_face_112_fin.test.csv",
        repo_root / "wiki_face_112_fin",
    )


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    default_csv, default_images = default_paths()
    parser = argparse.ArgumentParser(
        description="Evaluate a face-embedding model on WikiFace with gallery/query identification."
    )
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
        default=0,
        help="If >0, save up to N Top-1 mistakes (query vs predicted gallery).",
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
        default=Path("evaluation_artifacts") / "top1_cosine_boxplot.png",
        help="Where to save boxplot of Top-1 cosine scores (correct vs misclassified).",
    )
    return parser

