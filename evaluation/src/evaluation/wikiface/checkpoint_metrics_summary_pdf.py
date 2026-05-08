from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.stats import norm

# Pattern to extract step number from filename
STEP_RE = re.compile(r"_step_(\d+)\.(?:wikiface\.metrics\.csv|top1_det_wikiface\.csv)$")
METRICS_SUFFIX = ".wikiface.metrics.csv"
DET_SUFFIX = ".top1_det_wikiface.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find all *.wikiface.metrics.csv and *.top1_det_wikiface.csv files under one or more roots "
            "and generate summary line charts and DET curves as PDF files."
        )
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="One or more root directories to scan recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory for generated PDF charts (default: current directory).",
    )
    return parser.parse_args()


def _find_files(roots: list[Path], suffix: str) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if not resolved.exists() or not resolved.is_dir():
            print(f"WARNING: root is not an existing directory, skipping: {resolved}", file=sys.stderr)
            continue
        files.extend(
            p for p in resolved.rglob(f"*{suffix}") if p.is_file() and p.name.endswith(suffix)
        )
    return sorted(files)


def _extract_step(path: Path) -> int | None:
    match = STEP_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def _extract_config_name(path: Path) -> str:
    # Example: peoplegator_augmented_lr_1e-5_steps_2000_20260426_232940_step400.wikiface.metrics.csv
    # Config name is the part before _stepN
    name = path.name
    match = STEP_RE.search(name)
    if match:
        return name[: match.start()]
    return name.split(".")[0]


def _sanitize_for_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", text)


def _plot_metric_series(
    metric_name: str,
    top_k: int,
    config_map: dict[str, list[tuple[int, float]]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Training Step")
    ax.set_ylabel(f"{metric_name.capitalize()} (top_k={top_k})")
    ax.set_title(f"WikiFace {metric_name.capitalize()} vs Step (Top-K={top_k})")

    for config_name, points in sorted(config_map.items()):
        points.sort(key=lambda x: x[0])  # Sort by step
        steps, values = zip(*points)
        ax.plot(steps, values, marker="o", label=config_name, markersize=4)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _plot_det_curves(
    det_series: dict[str, dict[int, tuple[np.ndarray, np.ndarray, float]]],
    output_dir: Path,
) -> None:
    # Summary Best DET plot
    fig, ax = plt.subplots(figsize=(8, 7))
    _setup_det_ax(ax, "WikiFace Summary Best DET Curves")

    for config_name in sorted(det_series.keys()):
        steps_map = det_series[config_name]
        # Find step with minimum EER
        best_step = min(steps_map.keys(), key=lambda s: steps_map[s][2])
        fpr, fnr, eer = steps_map[best_step]
        _draw_det_line(ax, fpr, fnr, eer, label=f"{config_name} (step {best_step})")

    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    out_path = output_dir / "det_summary_best_wikiface.pdf"
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _setup_det_ax(ax: plt.Axes, title: str) -> None:
    det_clip = 1e-4
    show_tick_perc = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 40, 60, 80, 90, 95, 99, 99.9])
    show_tick_pos = norm.ppf(show_tick_perc / 100.0)

    ax.set_xticks(show_tick_pos)
    ax.set_xticklabels([f"{t:g}" for t in show_tick_perc], rotation=45)
    ax.set_yticks(show_tick_pos)
    ax.set_yticklabels([f"{t:g}" for t in show_tick_perc])

    ax.set_xlim(norm.ppf(det_clip), norm.ppf(1 - det_clip))
    ax.set_ylim(norm.ppf(det_clip), norm.ppf(1 - det_clip))

    ax.set_xlabel("False Positive Rate (%) [probit scale]")
    ax.set_ylabel("False Negative Rate (%) [probit scale]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def _draw_det_line(
    ax: plt.Axes,
    fpr: np.ndarray,
    fnr: np.ndarray,
    eer: float,
    label: str,
) -> None:
    det_clip = 1e-7
    fpr_safe = np.clip(fpr, det_clip, 1.0 - det_clip)
    fnr_safe = np.clip(fnr, det_clip, 1.0 - det_clip)
    ax.plot(
        norm.ppf(fpr_safe),
        norm.ppf(fnr_safe),
        label=f"{label} (EER={eer:.4f})",
        linewidth=1.5,
    )


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_files = _find_files(args.roots, METRICS_SUFFIX)
    det_files = _find_files(args.roots, DET_SUFFIX)

    if not metric_files and not det_files:
        print(f"No WikiFace results found in {args.roots}")
        return 0

    # series[(metric_name, top_k)][config_name] = [(step, value), ...]
    series: dict[tuple[str, int], dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for metric_file in metric_files:
        step = _extract_step(metric_file)
        if step is None:
            continue
        config_name = _extract_config_name(metric_file)

        with metric_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    top_k = int(row["top_k"])
                    accuracy = float(row["accuracy"])
                    series[("accuracy", top_k)][config_name].append((step, accuracy))
                except (KeyError, TypeError, ValueError):
                    continue

    det_series: dict[str, dict[int, tuple[np.ndarray, np.ndarray, float]]] = defaultdict(dict)
    for det_file in det_files:
        step = _extract_step(det_file)
        if step is None:
            continue
        config_name = _extract_config_name(det_file)

        with det_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fpr, fnr = [], []
            for row in reader:
                try:
                    fpr.append(float(row["fpr"]))
                    fnr.append(float(row["fnr"]))
                except (KeyError, ValueError):
                    continue
            if fpr and fnr:
                fpr_arr = np.array(fpr)
                fnr_arr = np.array(fnr)
                eer_idx = int(np.argmin(np.abs(fpr_arr - fnr_arr)))
                eer = float((fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2.0)
                det_series[config_name][step] = (fpr_arr, fnr_arr, eer)

    # Plot accuracies
    for (metric_name, top_k), config_map in sorted(series.items()):
        out_name = f"wikiface_{_sanitize_for_filename(metric_name)}_topk_{top_k}.pdf"
        _plot_metric_series(metric_name, top_k, config_map, output_dir / out_name)

    # Plot DET curves
    if det_series:
        _plot_det_curves(det_series, output_dir)

    print("\nWikiFace summary report generated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
