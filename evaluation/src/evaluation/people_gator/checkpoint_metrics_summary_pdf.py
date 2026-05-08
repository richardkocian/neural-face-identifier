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

STEP_RE = re.compile(r"_step_(\d+)\.(?:retrieval\.union\.tst\.metrics\.csv|retrieval\.union\.tst\.det\.csv)$")
METRICS_SUFFIX = ".retrieval.union.tst.metrics.csv"
DET_SUFFIX = ".retrieval.union.tst.det.csv"
NON_METRIC_COLUMNS = {"top_k", "ignore_index", "count"}
MINIMIZING_METRICS = {"fallout", "error_rate", "fnr", "fpr", "eer"}


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
            "Find all *.retrieval.union.tst.metrics.csv and *.retrieval.union.tst.det.csv files under one or more roots "
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
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Optional directory containing baseline results (retrieval.union.tst.metrics.csv and retrieval.union.tst.det.csv).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional allowlist of metric names (e.g. hitrate precision mrr).",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=None,
        help="Optional allowlist of top_k values.",
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

    metrics_filter = set(args.metrics) if args.metrics else None
    top_k_filter = set(args.top_k) if args.top_k else None

    # Load baseline if provided
    baseline_metrics: dict[tuple[str, int], float] = {}
    baseline_det: tuple[np.ndarray, np.ndarray, float] | None = None

    if args.baseline_dir:
        b_metrics_file = args.baseline_dir / "retrieval.union.tst.metrics.csv"
        if b_metrics_file.exists():
            with b_metrics_file.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        top_k = int(row["top_k"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    for col, val in row.items():
                        if col in NON_METRIC_COLUMNS or val is None or val == "":
                            continue
                        try:
                            baseline_metrics[(col, top_k)] = float(val)
                        except ValueError:
                            continue

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

    metric_files = _find_files(args.roots, METRICS_SUFFIX)
    det_files = _find_files(args.roots, DET_SUFFIX)

    if not metric_files and not det_files:
        raise FileNotFoundError("No *.retrieval.union.tst.metrics.csv or *.retrieval.union.tst.det.csv files found.")

    series: dict[tuple[str, int], dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for metric_file in metric_files:
        step = _extract_step(metric_file)
        if step is None:
            print(f"WARNING: could not parse step from file name, skipping: {metric_file}", file=sys.stderr)
            continue
        config_name = _extract_config_name(metric_file)

        with metric_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue

            for row in reader:
                try:
                    top_k = int(row["top_k"])
                except (KeyError, TypeError, ValueError):
                    continue

                if top_k_filter is not None and top_k not in top_k_filter:
                    continue

                for column, raw_value in row.items():
                    if column in NON_METRIC_COLUMNS:
                        continue
                    if metrics_filter is not None and column not in metrics_filter:
                        continue
                    if raw_value is None or raw_value == "":
                        continue
                    try:
                        value = float(raw_value)
                    except ValueError:
                        continue
                    series[(column, top_k)][config_name].append((step, value))

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

    if not series:
        raise RuntimeError("No plottable metric data found after filtering/parsing.")

    for (metric_name, top_k), config_map in sorted(series.items(), key=lambda x: (x[0][0], x[0][1])):
        baseline = baseline_metrics.get((metric_name, top_k))

        # 1. Standard Plot
        out_name = f"{_sanitize_for_filename(metric_name)}_topk_{top_k}.pdf"
        _plot_metric_series(metric_name, top_k, config_map, baseline, output_dir / out_name)

        # 2. Closeup Plot
        all_values = [v for points in config_map.values() for _, v in points]
        ylim = _get_closeup_ylim(metric_name, all_values, baseline)
        if ylim:
            closeup_name = f"{_sanitize_for_filename(metric_name)}_topk_{top_k}_closeup.pdf"
            _plot_metric_series(
                metric_name,
                top_k,
                config_map,
                baseline,
                output_dir / closeup_name,
                ylim=ylim,
                title_suffix=" (closeup)",
            )

    # Plot DET curves
    if det_series:
        _plot_det_curves(det_series, baseline_det, output_dir)

    return 0


def _plot_metric_series(
    metric_name: str,
    top_k: int,
    config_map: dict[str, list[tuple[int, float]]],
    baseline: float | None,
    output_path: Path,
    ylim: tuple[float, float] | None = None,
    title_suffix: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted_any = False

    for config_name in sorted(config_map):
        points = sorted(config_map[config_name], key=lambda x: x[0])
        if not points:
            continue
        steps = [p[0] for p in points]
        values = [p[1] for p in points]
        ax.plot(steps, values, marker="o", label=config_name)
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

    for config_name in sorted(det_series.keys()):
        steps_map = det_series[config_name]
        best_step = min(steps_map.keys(), key=lambda s: steps_map[s][2])
        fpr, fnr, eer = steps_map[best_step]
        _draw_det_line(ax, fpr, fnr, eer, label=f"{config_name} (step {best_step})")

    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize="small")
    # fig.tight_layout()
    out_path = output_dir / "det_summary_best.pdf"
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


if __name__ == "__main__":
    raise SystemExit(main())
