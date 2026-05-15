from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.stats import norm

STEP_RE = re.compile(r"_step_(\d+)\.(?:retrieval\.union\.tst\.metrics\.csv|retrieval\.union\.tst\.det\.csv)$")
METRICS_SUFFIX = ".retrieval.union.tst.metrics.csv"
DET_SUFFIX = ".retrieval.union.tst.det.csv"
NON_METRIC_COLUMNS = {"top_k", "ignore_index", "count"}
MINIMIZING_METRICS = {"fallout", "error_rate", "fnr", "fpr", "eer"}


def _get_styles() -> itertools.cycle:
    # Use tab20 colors and 4 different linestyles for maximum variety
    colors = plt.get_cmap("tab20").colors
    linestyles = ["-", "--", "-.", ":"]
    # Cycle through (linestyle, color) pairs
    return itertools.cycle(itertools.product(linestyles, colors))


def _is_minimizing(metric_name: str) -> bool:
    return any(m in metric_name.lower() for m in MINIMIZING_METRICS)


def _get_closeup_ylim(
    metric_name: str, data_values: list[float], baseline: float | None
) -> tuple[float, float] | None:
    if baseline is None:
        return None

    if _is_minimizing(metric_name):
        # Target is 0.0, closeup focus from 0.0 to baseline + 0.01
        return (None, baseline + 0.01)
    else:
        # Target is 1.0, closeup focus from baseline - 0.01 to current max (usually 1.0)
        return (baseline - 0.01, None)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find all *.retrieval.union.tst.det.csv files under one or more roots "
            "and generate DET summary PDFs."
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
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Optional directory containing baseline retrieval.union.tst.det.csv.",
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
    if path.parent.name == "evaluation_artifacts":
        return path.parent.parent.name
    return path.parent.name


def _sanitize_for_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "metric"


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline if provided
    baseline_det: tuple[np.ndarray, np.ndarray, float] | None = None

    if args.baseline_dir:
        b_det_file = args.baseline_dir / "retrieval.union.tst.det.csv"
        if b_det_file.exists():
            with b_det_file.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                b_fpr, b_fnr = [], []
                for row in reader:
                    try:
                        b_fpr.append(float(row["fpr"]))
                        b_fnr.append(float(row["fnr"]))
                    except (KeyError, ValueError):
                        continue
                if b_fpr and b_fnr:
                    fpr_arr = np.array(b_fpr)
                    fnr_arr = np.array(b_fnr)
                    eer_idx = int(np.argmin(np.abs(fpr_arr - fnr_arr)))
                    eer = float((fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2.0)
                    baseline_det = (fpr_arr, fnr_arr, eer)

    det_files = _find_files(args.roots, DET_SUFFIX)

    if not det_files:
        raise FileNotFoundError("No *.retrieval.union.tst.det.csv files found.")

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

    # Plot DET curves
    if not det_series:
        raise RuntimeError("No plottable DET data found after parsing.")
    _plot_det_curves(det_series, baseline_det, output_dir)

    return 0


def _plot_metric_series(
    metric_name: str,
    top_k: int,
    config_map: dict[str, list[tuple[int, float]]],
    baseline: float | None,
    output_path: Path,
    ylim: tuple[float, float] | None = None,
    xlim_max: float | None = None,
    title_suffix: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted_any = False
    styles = _get_styles()

    for config_name in sorted(config_map):
        points = sorted(config_map[config_name], key=lambda x: x[0])
        if not points:
            continue
        steps = [p[0] for p in points]
        values = [p[1] for p in points]
        linestyle, color = next(styles)
        ax.plot(
            steps,
            values,
            marker="o",
            label=config_name,
            color=color,
            linestyle=linestyle,
            markersize=4,
        )
        plotted_any = True

    if baseline is not None:
        ax.axhline(
            y=baseline,
            color="red",
            linestyle="-",
            linewidth=2.5,
            zorder=10,
            label=f"baseline ({baseline:.4f})",
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_title(f"{metric_name} @ top_k={top_k}{title_suffix}")
    ax.set_xlabel("step")
    ax.set_ylabel(metric_name)
    if ylim:
        cur_min, cur_max = ax.get_ylim()
        new_min = ylim[0] if ylim[0] is not None else cur_min
        new_max = ylim[1] if ylim[1] is not None else cur_max
        ax.set_ylim(new_min, new_max)
    if xlim_max:
        ax.set_xlim(None, xlim_max)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    # fig.tight_layout()

    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _plot_det_curves(
    det_series: dict[str, dict[int, tuple[np.ndarray, np.ndarray, float]]],
    baseline_det: tuple[np.ndarray, np.ndarray, float] | None,
    output_dir: Path,
) -> None:
    # 1. Per-config DET plots
    for config_name, steps_map in det_series.items():
        fig, ax = plt.subplots(figsize=(8, 7))
        _setup_det_ax(ax, f"DET Curves: {config_name}")

        if baseline_det:
            _draw_det_line(
                ax, *baseline_det, label="baseline", color="red", linestyle="-", linewidth=2.5, zorder=10
            )

        sorted_steps = sorted(steps_map.keys())
        colors = cm.viridis(np.linspace(0, 1, len(sorted_steps)))

        for step, color in zip(sorted_steps, colors):
            fpr, fnr, eer = steps_map[step]
            _draw_det_line(ax, fpr, fnr, eer, label=f"step {step}", color=color)

        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize="small")
        # fig.tight_layout()
        out_path = output_dir / f"det_{_sanitize_for_filename(config_name)}.pdf"
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    # 2. Summary Best DET plot
    fig, ax = plt.subplots(figsize=(8, 7))
    _setup_det_ax(ax, "Summary Best DET Curves")

    if baseline_det:
        _draw_det_line(
            ax, *baseline_det, label="baseline", color="red", linestyle="-", linewidth=2.5, zorder=10
        )

    styles = _get_styles()
    for config_name in sorted(det_series.keys()):
        steps_map = det_series[config_name]
        best_step = min(steps_map.keys(), key=lambda s: steps_map[s][2])
        fpr, fnr, eer = steps_map[best_step]
        linestyle, color = next(styles)
        _draw_det_line(
            ax,
            fpr,
            fnr,
            eer,
            label=f"{config_name} (step {best_step})",
            color=color,
            linestyle=linestyle,
        )

    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize="small")
    # fig.tight_layout()
    out_path = output_dir / "det_summary_best.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # 3. Summary Best DET plot on log-log axes (often clearer for very close curves)
    fig, ax = plt.subplots(figsize=(8, 7))
    _setup_det_log_ax(ax, "Summary Best DET Curves (log-log)")

    if baseline_det:
        _draw_det_line_log(
            ax, *baseline_det, label="baseline", color="red", linestyle="-", linewidth=2.5, zorder=10
        )

    styles = _get_styles()
    for config_name in sorted(det_series.keys()):
        steps_map = det_series[config_name]
        best_step = min(steps_map.keys(), key=lambda s: steps_map[s][2])
        fpr, fnr, eer = steps_map[best_step]
        linestyle, color = next(styles)
        _draw_det_line_log(
            ax,
            fpr,
            fnr,
            eer,
            label=f"{config_name} (step {best_step})",
            color=color,
            linestyle=linestyle,
        )

    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize="small")
    out_path = output_dir / "det_summary_best_loglog.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _setup_det_ax(ax: plt.Axes, title: str) -> None:
    det_clip = 1e-4
    tick_perc = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40, 60, 80, 90, 95, 98, 99, 99.5, 99.8, 99.9])
    tick_pos = norm.ppf(tick_perc / 100.0)

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


def _setup_det_log_ax(ax: plt.Axes, title: str) -> None:
    det_clip = 1e-5
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(det_clip, 1.0)
    ax.set_ylim(det_clip, 1.0)
    ax.set_xlabel("False Positive Rate (fraction, log scale)")
    ax.set_ylabel("False Negative Rate (fraction, log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)


def _draw_det_line(
    ax: plt.Axes,
    fpr: np.ndarray,
    fnr: np.ndarray,
    eer: float,
    label: str,
    color: str | np.ndarray | None = None,
    linestyle: str = "-",
    linewidth: float = 1.5,
    zorder: int = 2,
) -> None:
    det_clip = 1e-7
    fpr_safe = np.clip(fpr, det_clip, 1.0 - det_clip)
    fnr_safe = np.clip(fnr, det_clip, 1.0 - det_clip)
    ax.plot(
        norm.ppf(fpr_safe),
        norm.ppf(fnr_safe),
        label=f"{label} (EER={eer:.4f})",
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        zorder=zorder,
    )


def _draw_det_line_log(
    ax: plt.Axes,
    fpr: np.ndarray,
    fnr: np.ndarray,
    eer: float,
    label: str,
    color: str | np.ndarray | None = None,
    linestyle: str = "-",
    linewidth: float = 1.5,
    zorder: int = 2,
) -> None:
    det_clip = 1e-7
    fpr_safe = np.clip(fpr, det_clip, 1.0 - det_clip)
    fnr_safe = np.clip(fnr, det_clip, 1.0 - det_clip)
    ax.plot(
        fpr_safe,
        fnr_safe,
        label=f"{label} (EER={eer:.4f})",
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        zorder=zorder,
    )


if __name__ == "__main__":
    raise SystemExit(main())
