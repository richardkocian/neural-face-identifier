from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

STEP_RE = re.compile(r"_step_(\d+)\.retrieval\.union\.tst\.metrics\.csv$")
METRICS_SUFFIX = ".retrieval.union.tst.metrics.csv"
NON_METRIC_COLUMNS = {"top_k", "ignore_index", "count"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find all *.retrieval.union.tst.metrics.csv files under one or more roots "
            "and generate summary line charts as PDF files (one PDF per metric+top_k)."
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


def _find_metric_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if not resolved.exists() or not resolved.is_dir():
            print(f"WARNING: root is not an existing directory, skipping: {resolved}", file=sys.stderr)
            continue
        files.extend(
            p for p in resolved.rglob(f"*{METRICS_SUFFIX}") if p.is_file() and p.name.endswith(METRICS_SUFFIX)
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

    metric_files = _find_metric_files(args.roots)
    if not metric_files:
        raise FileNotFoundError("No *.retrieval.union.tst.metrics.csv files were found in the provided roots.")

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
                print(f"WARNING: empty CSV, skipping: {metric_file}", file=sys.stderr)
                continue

            for row in reader:
                try:
                    top_k = int(row["top_k"])
                except (KeyError, TypeError, ValueError):
                    print(f"WARNING: invalid top_k in {metric_file}, row skipped.", file=sys.stderr)
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

    if not series:
        raise RuntimeError("No plottable metric data found after filtering/parsing.")

    for (metric_name, top_k), config_map in sorted(series.items(), key=lambda x: (x[0][0], x[0][1])):
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

        if not plotted_any:
            plt.close(fig)
            continue

        ax.set_title(f"{metric_name} @ top_k={top_k}")
        ax.set_xlabel("step")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()

        out_name = f"{_sanitize_for_filename(metric_name)}_topk_{top_k}.pdf"
        out_path = output_dir / out_name
        fig.savefig(out_path, format="pdf")
        plt.close(fig)
        print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
