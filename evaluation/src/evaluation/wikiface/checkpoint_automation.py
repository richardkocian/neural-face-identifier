from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def build_parser() -> argparse.ArgumentParser:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run full WikiFace evaluation pipeline for every .pth checkpoint "
            "found recursively under --checkpoints-root."
        )
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=root.parent / "checkpoints",
        help="Root directory scanned recursively for .pth checkpoints (default: ../checkpoints).",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=root / "wiki_face_112_fin" / "wiki_face_112_fin.test.csv",
        help="Path to WikiFace CSV metadata file.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=root / "wiki_face_112_fin",
        help="Root directory containing WikiFace images.",
    )
    parser.add_argument(
        "--summary-output-dir",
        type=Path,
        default=root / "summary-graphs-wikiface",
        help="Output directory for generated PDF summary charts (default: root/summary-graphs-wikiface).",
    )
    return parser


def _run_command(command: list[str]) -> None:
    print(f"$ {shlex.join(command)}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {shlex.join(command)}")


def _prefix_for_checkpoint(checkpoint_path: Path) -> str:
    # Use parent directory name (e.g. run name) and checkpoint stem (e.g. step400)
    return f"{checkpoint_path.parent.name}_{checkpoint_path.stem}"


def _process_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> None:
    checkpoint_dir = checkpoint_path.parent
    prefix = _prefix_for_checkpoint(checkpoint_path)

    eval_dir = checkpoint_dir / "evaluation_artifacts_wikiface"
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = eval_dir / f"{prefix}.wikiface.metrics.csv"
    det_csv = eval_dir / f"{prefix}.top1_det_wikiface.csv"
    det_png = eval_dir / f"{prefix}.top1_det_wikiface.png"
    boxplot_png = eval_dir / f"{prefix}.top1_cosine_boxplot_wikiface.png"
    misclassified_dir = eval_dir / f"{prefix}_misclassified"

    print(f"\n=== Processing checkpoint: {checkpoint_path} ===")

    # Skip only if all artifacts exist
    if metrics_csv.exists() and det_csv.exists() and det_png.exists() and boxplot_png.exists():
        print(f"WikiFace evaluation for this model exist, skipping. Proceeding to next step.")
    else:
        _run_command(
            [
                "uv",
                "run",
                "--package",
                "evaluation",
                "run-wikiface",
                "--csv-path",
                str(args.csv_path.resolve()),
                "--images-root",
                str(args.images_root.resolve()),
                "--finetuned-model",
                str(checkpoint_path.resolve()),
                "--metrics-csv-path",
                str(metrics_csv.resolve()),
                "--det-csv-path",
                str(det_csv.resolve()),
                "--det-image-path",
                str(det_png.resolve()),
                "--top1-boxplot-path",
                str(boxplot_png.resolve()),
                "--misclassified-output-dir",
                str(misclassified_dir.resolve()),
                "--show-misclassified-top1",
                "5",
            ]
        )


def main() -> int:
    args = build_parser().parse_args()

    checkpoints_root = args.checkpoints_root.resolve()
    if not checkpoints_root.exists():
        raise FileNotFoundError(f"Checkpoints root does not exist: {checkpoints_root}")

    checkpoints = sorted(checkpoints_root.rglob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pth files found under: {checkpoints_root}")

    print(f"Found {len(checkpoints)} checkpoints under {checkpoints_root}")
    for checkpoint_path in checkpoints:
        _process_checkpoint(
            checkpoint_path=checkpoint_path,
            args=args,
        )

    print("\nAll checkpoints processed successfully.")

    # Generate summary metrics report
    print("\n=== Generating WikiFace Summary Metrics Report ===")
    args.summary_output_dir.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            "uv",
            "run",
            "--package",
            "evaluation",
            "run-wikiface-metrics-summary-pdf",
            str(checkpoints_root),
            "--output-dir",
            str(args.summary_output_dir.resolve()),
        ]
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
