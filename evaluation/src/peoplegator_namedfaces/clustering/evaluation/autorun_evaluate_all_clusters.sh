#!/usr/bin/env bash
set -euo pipefail

# Usage: evaluate_all_clusters.sh <root_dir> <ground_truth.jsonl>
# Finds files matching clusters.*.jsonl under <root_dir> and evaluates each
# against the provided ground truth file. Writes a JSONL named
# cluster_evaluations.jsonl in the current directory and also prints each
# JSON line to stdout.

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <root_dir> <ground_truth.jsonl> <output_dir>" >&2
  exit 2
fi

ROOT_DIR="$1"
GT_FILE="$2"
OUTPUT_DIR="$3"

mkdir -p "$OUTPUT_DIR"

# Ensure PYTHONPATH points to the project's src/ so the CLI can import the package
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src"

find "$ROOT_DIR" -type f -name 'clusters.*.jsonl' -print0 | while IFS= read -r -d '' cluster_path; do
  parent_dir_name="$(basename "$(dirname "$cluster_path")")"
  filename="$(basename "$cluster_path")"
  algorithm="${filename#clusters.}"
  algorithm="${algorithm%.jsonl}"
  uv run people-gator-evaluate-clustering-v3 -g "$GT_FILE" -p "$cluster_path" -o "$OUTPUT_DIR/cluster_evaluations.${parent_dir_name}.${algorithm}.jsonl" -s 4 -b 3000
done


echo "Wrote metadata files to $OUTPUT_DIR"
