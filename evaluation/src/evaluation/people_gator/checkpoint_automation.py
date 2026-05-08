from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def build_parser() -> argparse.ArgumentParser:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run full People Gator retrieval evaluation pipeline for every .pth checkpoint "
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
        "--jsonl-path",
        type=Path,
        default=root
        / "people_gator"
        / "people_gator__corresponding_faces__2026-02-11.test.cleaned.jsonl",
        help="People Gator JSONL metadata file for embeddings.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=root / "people_gator" / "people_gator__data",
        help="Root directory with image files referenced by JSONL.",
    )
    parser.add_argument(
        "--dataset-template",
        type=Path,
        default=root
        / "evaluation"
        / "src"
        / "peoplegator_namedfaces"
        / "retrieval"
        / "configs"
        / "dataset.template.json",
        help="Template dataset config used to generate per-checkpoint dataset JSON files.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=root
        / "evaluation"
        / "src"
        / "peoplegator_namedfaces"
        / "retrieval"
        / "configs"
        / "image_queries.union.tst.jsonl",
        help="Query JSONL for retrieval and shared ground truth generation.",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=root
        / "evaluation"
        / "src"
        / "peoplegator_namedfaces"
        / "retrieval"
        / "configs"
        / "engine.image_embedding.json",
        help="Retrieval engine config JSON.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=root
        / "people_gator"
        / "people_gator__corresponding_faces__2026-02-11.test.cleaned.jsonl",
        help="People Gator annotations JSONL for shared ground truth and boxplot labels.",
    )
    parser.add_argument(
        "--shared-ground-truth",
        type=Path,
        default=root / "evaluation_artifacts" / "retrieval.union.tst.ground_truth.jsonl",
        help="Shared ground-truth JSONL path reused across all checkpoints.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Top-k values used by retrieval evaluation.",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=-1,
        help="Ignore index for self-match in retrieval evaluation commands.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=10000,
        help="Bootstrap iterations for retrieval-evaluate bootstrap CSV.",
    )
    parser.add_argument(
        "--summary-output-dir",
        type=Path,
        default=root / "summary-graphs",
        help="Output directory for generated PDF summary charts (default: root/summary-graphs).",
    )
    parser.add_argument(
        "--force-regenerate-shared-ground-truth",
        action="store_true",
        help="Regenerate shared ground-truth JSONL even if an existing file is present.",
    )
    return parser


def _run_command(command: list[str]) -> None:
    print(f"$ {shlex.join(command)}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {shlex.join(command)}")


def _prefix_for_checkpoint(checkpoint_path: Path) -> str:
    return f"{checkpoint_path.parent.name}_{checkpoint_path.stem}"


def _load_dataset_template(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        template = json.load(f)
    if not isinstance(template, dict):
        raise ValueError(f"Dataset template must be a JSON object: {path}")
    return template


def _write_dataset_config(template: dict[str, Any], output_path: Path, embeddings_dir: Path) -> None:
    if "image_paths" not in template or "image_embeddings" not in template:
        raise KeyError("Dataset template JSON must contain 'image_paths' and 'image_embeddings'.")
    dataset_cfg = dict(template)
    dataset_cfg["image_paths"] = str((embeddings_dir / "image_paths.txt").resolve())
    dataset_cfg["image_embeddings"] = str((embeddings_dir / "embeddings.npy").resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset_cfg, indent=2) + "\n", encoding="utf-8")


def _ensure_shared_ground_truth(args: argparse.Namespace) -> Path:
    def _source_fingerprint(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    shared_gt = args.shared_ground_truth.resolve()
    metadata_path = shared_gt.with_suffix(shared_gt.suffix + ".meta.json")
    source_meta = {
        "queries": _source_fingerprint(args.queries.resolve()),
        "annotations": _source_fingerprint(args.annotations.resolve()),
    }

    if shared_gt.exists() and not args.force_regenerate_shared_ground_truth:
        if not metadata_path.exists():
            raise RuntimeError(
                f"Existing shared ground truth has no metadata: {shared_gt}. "
                "Run with --force-regenerate-shared-ground-truth once to initialize metadata."
            )

        stored_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if stored_meta == source_meta:
            print(f"Reusing shared ground truth: {shared_gt}")
            return shared_gt
        print("Shared ground truth metadata changed; regenerating.")

    shared_gt.parent.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            "uv",
            "run",
            "--package",
            "evaluation",
            "run-people-gator-retrieval-gt",
            "--queries",
            str(args.queries.resolve()),
            "--annotations",
            str(args.annotations.resolve()),
            "--output",
            str(shared_gt),
        ]
    )
    metadata_path.write_text(json.dumps(source_meta, indent=2) + "\n", encoding="utf-8")
    return shared_gt


def _process_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    template: dict[str, Any],
    shared_gt: Path,
) -> None:
    checkpoint_dir = checkpoint_path.parent
    prefix = _prefix_for_checkpoint(checkpoint_path)

    embeddings_dir = checkpoint_dir / ".embeddings" / prefix
    eval_dir = checkpoint_dir / "evaluation_artifacts"
    eval_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = eval_dir / f"{prefix}.dataset.json"
    predictions = eval_dir / f"{prefix}.retrieval.union.tst.pkl"
    metrics_csv = eval_dir / f"{prefix}.retrieval.union.tst.metrics.csv"
    bootstrap_csv = eval_dir / f"{prefix}.retrieval.union.tst.metrics.bootstrap.csv"
    det_png = eval_dir / f"{prefix}.retrieval.union.tst.det.png"
    det_csv = eval_dir / f"{prefix}.retrieval.union.tst.det.csv"
    boxplot_png = eval_dir / f"{prefix}.retrieval.union.tst.top1_cosine_boxplot.png"

    print(f"\n=== Processing checkpoint: {checkpoint_path} ===")

    embeddings_file = embeddings_dir / "embeddings.npy"
    image_paths_file = embeddings_dir / "image_paths.txt"

    if embeddings_file.exists() and image_paths_file.exists():
        print(f"Embeddings for this model exist, skipping with embeddings generation. Proceeding to next step.")
    else:
        _run_command(
            [
                "uv",
                "run",
                "--package",
                "evaluation",
                "run-people-gator-embeddings",
                "--jsonl-path",
                str(args.jsonl_path.resolve()),
                "--images-root",
                str(args.images_root.resolve()),
                "--finetuned-model",
                str(checkpoint_path.resolve()),
                "--output-dir",
                str(embeddings_dir.resolve()),
            ]
        )

    _write_dataset_config(template=template, output_path=dataset_cfg, embeddings_dir=embeddings_dir)

    _run_command(
        [
            "uv",
            "run",
            "--package",
            "evaluation",
            "run-people-gator-retrieval",
            "--dataset",
            str(dataset_cfg.resolve()),
            "--queries",
            str(args.queries.resolve()),
            "--engine",
            str(args.engine.resolve()),
            "--output",
            str(predictions.resolve()),
        ]
    )

    top_k_args = [str(x) for x in args.top_k]
    _run_command(
        [
            "uv",
            "run",
            "--package",
            "evaluation",
            "run-people-gator-retrieval-evaluate",
            "--predictions",
            str(predictions.resolve()),
            "--ground-truth",
            str(shared_gt),
            "--dataset",
            str(dataset_cfg.resolve()),
            "--top-k",
            *top_k_args,
            "--ignore-index",
            str(args.ignore_index),
            "--output-file",
            str(metrics_csv.resolve()),
        ]
    )

    _run_command(
        [
            "uv",
            "run",
            "--package",
            "evaluation",
            "run-people-gator-retrieval-evaluate",
            "--predictions",
            str(predictions.resolve()),
            "--ground-truth",
            str(shared_gt),
            "--dataset",
            str(dataset_cfg.resolve()),
            "--top-k",
            *top_k_args,
            "--ignore-index",
            str(args.ignore_index),
            "--bootstrap-iters",
            str(args.bootstrap_iters),
            "--output-file",
            str(bootstrap_csv.resolve()),
        ]
    )

    _run_command(
        [
            "uv",
            "run",
            "--package",
            "evaluation",
            "run-people-gator-retrieval-det",
            "--predictions",
            str(predictions.resolve()),
            "--ground-truth",
            str(shared_gt),
            "--dataset",
            str(dataset_cfg.resolve()),
            "--ignore-index",
            str(args.ignore_index),
            "--output-image",
            str(det_png.resolve()),
            "--output-csv",
            str(det_csv.resolve()),
        ]
    )

    _run_command(
        [
            "uv",
            "run",
            "--package",
            "evaluation",
            "run-people-gator-retrieval-boxplot",
            "--predictions",
            str(predictions.resolve()),
            "--ground-truth",
            str(shared_gt),
            "--dataset",
            str(dataset_cfg.resolve()),
            "--images-root",
            str(args.images_root.resolve()),
            "--annotations-jsonl",
            str(args.annotations.resolve()),
            "--ignore-index",
            str(args.ignore_index),
            "--output-image",
            str(boxplot_png.resolve()),
            "--show-misclassified-top1",
            "0",
        ]
    )


def main() -> int:
    args = build_parser().parse_args()

    checkpoints_root = args.checkpoints_root.resolve()
    if not checkpoints_root.exists():
        raise FileNotFoundError(f"Checkpoints root does not exist: {checkpoints_root}")
    if not checkpoints_root.is_dir():
        raise NotADirectoryError(f"Checkpoints root is not a directory: {checkpoints_root}")

    template = _load_dataset_template(args.dataset_template.resolve())
    shared_gt = _ensure_shared_ground_truth(args)

    checkpoints = sorted(checkpoints_root.rglob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pth files found under: {checkpoints_root}")

    print(f"Found {len(checkpoints)} checkpoints under {checkpoints_root}")
    for checkpoint_path in checkpoints:
        _process_checkpoint(
            checkpoint_path=checkpoint_path,
            args=args,
            template=template,
            shared_gt=shared_gt,
        )

    print("\nAll checkpoints processed successfully.")

    # Generate summary metrics report
    print("\n=== Generating Summary Metrics Report ===")
    args.summary_output_dir.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            "uv",
            "run",
            "--package",
            "evaluation",
            "run-people-gator-retrieval-metrics-summary-pdf",
            str(checkpoints_root),
            "--output-dir",
            str(args.summary_output_dir.resolve()),
        ]
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
