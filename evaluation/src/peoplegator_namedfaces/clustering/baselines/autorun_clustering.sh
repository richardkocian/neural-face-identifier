#!/usr/bin/env bash
# Auto-run clustering for all models with embeddings under .embeddings
# Usage: ./autorun_clustering.sh [-f] [-v] [-M metric]
#  -f : force overwrite existing outputs
#  -v : verbose (passes -v to python script)
#  -M metric : override metric (default: euclidean)

set -euo pipefail

FORCE=0
VERBOSE=0
METRIC=cosine

while getopts "+fvm:M:" opt; do
  case "$opt" in
    f) FORCE=1 ;;
    v) VERBOSE=1 ;;
    M) METRIC="$OPTARG" ;;
    *) echo "Usage: $0 [-f] [-v] [-M metric]"; exit 2 ;;
  esac
done

PY_MODULE="cluster_embeddings"
PY_CMD=(python -m "$PY_MODULE")
ALGS=(agglomerative dbscan kmeans spectral)
EMBED_DIR=".embeddings"

if [ ! -d "$EMBED_DIR" ]; then
  echo "Directory $EMBED_DIR not found" >&2
  exit 1
fi

for d in "$EMBED_DIR"/*; do
  if [ -f "$d/embeddings.npy" ]; then
    echo "Processing model: $(basename "$d")"
    for alg in "${ALGS[@]}"; do
      out="$d/clusters.${alg}.jsonl"
      if [ -f "$out" ] && [ "$FORCE" -eq 0 ]; then
        echo "  Skipping existing $out"
        continue
      fi

      cmd=("${PY_CMD[@]}" -e "$d/embeddings.npy" -m "$d/image_paths.txt" -a "$alg" -M "$METRIC" -c "$out")
      if [ "$VERBOSE" -ne 0 ]; then
        cmd+=( -v )
        echo "  Running: ${cmd[*]}"
      else
        echo "  Running algorithm: $alg -> $(basename "$out")"
      fi

      if "${cmd[@]}"; then
        echo "  Finished $alg"
      else
        echo "  Error running $alg on $d" >&2
      fi
    done
  fi
done

echo "All done."
