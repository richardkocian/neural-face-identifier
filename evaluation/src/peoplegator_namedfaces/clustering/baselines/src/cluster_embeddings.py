"""Cluster embeddings CLI

Usage examples:
  python -m cluster_embeddings \
    --embeddings embeddings.npy --mapping files.txt --algorithm kmeans --params '{"n_clusters":8}'

The script loads a .npy 2D float32 array of embeddings and optionally a .txt
file containing one filename per row matching each embedding. It supports
several scikit-learn clustering algorithms and distance metrics. Parameters
may be supplied via the `--params` JSON string which overrides heuristics.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger("cluster_embeddings")


def estimate_n_clusters(n_samples: int) -> int:
    # heuristic: sqrt(2*N) bounded
    k = int(max(2, min(700, np.round(np.sqrt(n_samples * 2.0)))))
    logger.debug("Estimated n_clusters=%s for %s samples", k, n_samples)
    return k


def estimate_dbscan_eps(embeddings: np.ndarray) -> float:
    # use typical nearest-neighbor heuristic: median of 2nd NN distances
    k = min(embeddings.shape[0] - 1, 5)
    if k < 2:
        return 0.5
    nbrs = NearestNeighbors(n_neighbors=2).fit(embeddings)
    dists, _ = nbrs.kneighbors(embeddings)
    # dists[:, 1] is distance to nearest neighbor excluding self
    median = float(np.median(dists[:, 1]))
    eps = max(1e-6, median * 1.5)
    logger.debug("Estimated DBSCAN eps=%s (median NN=%s)", eps, median)
    return eps


def load_embeddings(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError("embeddings .npy must be a 2D array")
    if arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float32)
    logger.info("Loaded embeddings %s with shape %s dtype=%s", path, arr.shape, arr.dtype)
    return arr


def load_mapping(path: Path) -> list[str]:
    with path.open("r", encoding="utf8") as fh:
        lines = [l.rstrip("\n") for l in fh]
    return lines


def build_model(
    embeddings: np.ndarray,
    algorithm: str,
    metric: str,
    params: dict,
):
    n_samples = embeddings.shape[0]

    # default param filling
    params = dict(params)  # copy
    if algorithm == "kmeans":
        if metric == "cosine":
            # normalize outside then run KMeans (spherical approximation)
            logger.debug("Will normalize embeddings for cosine KMeans")
        if "n_clusters" not in params:
            params["n_clusters"] = estimate_n_clusters(n_samples)
        model = KMeans(n_clusters=int(params.pop("n_clusters")), **params)

    elif algorithm == "agglomerative":
        if "n_clusters" not in params:
            params["n_clusters"] = estimate_n_clusters(n_samples)
        # newer sklearn supports metric param
        model = AgglomerativeClustering(n_clusters=int(params.pop(
            "n_clusters")), metric=metric, linkage=params.pop("linkage", "average"))

    elif algorithm == "dbscan":
        if "eps" not in params:
            params["eps"] = estimate_dbscan_eps(embeddings)
        model = DBSCAN(metric=metric, **params)

    elif algorithm == "spectral":
        if "n_clusters" not in params:
            params["n_clusters"] = estimate_n_clusters(n_samples)
        model = SpectralClustering(n_clusters=int(params.pop("n_clusters")), **params)

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return model, params


def run_clustering(
    embeddings_path: Path,
    mapping_path: Path,
    algorithm: str,
    metric: str,
    params: dict,
    out_mapping: Path,
):
    embeddings = load_embeddings(embeddings_path)
    mapping = load_mapping(mapping_path)
    if len(mapping) != embeddings.shape[0]:
        logger.warning("Mapping length %s does not match embeddings %s", len(mapping), embeddings.shape[0])

    # handle metric transformations
    transformed = embeddings
    if metric == "cosine" and algorithm != "spectral":
        transformed = normalize(embeddings, axis=1)
        used_metric = "euclidean"
    else:
        used_metric = metric

    model, leftover = build_model(transformed, algorithm, used_metric, params)

    # fit/predict depending on estimator
    if hasattr(model, "fit_predict"):
        labels = model.fit_predict(transformed)
    else:
        model.fit(transformed)
        if hasattr(model, "labels_"):
            labels = model.labels_
        elif hasattr(model, "predict"):
            assert not isinstance(model, AgglomerativeClustering), "AgglomerativeClustering does not support predict()"
            assert not isinstance(model, DBSCAN), "DBSCAN does not support predict()"
            assert not isinstance(model, SpectralClustering), "SpectralClustering does not support predict()"
            labels = model.predict(transformed)
        else:
            raise RuntimeError("Unable to obtain labels from fitted model")

    labels = np.asarray(labels)
    logger.info("Clustering produced %s clusters (labels min,max)=(%s,%s)", len(np.unique(labels)), labels.min(), labels.max())


    # write JSON Lines ordered by embedding index matching
    # PeopleGatorNamedFaces__ClusterPrediction: face, cluster, cluster_score
    # cluster_score: higher is better. For `cosine` metric we return cosine similarity,
    # otherwise we return negative distance (so higher==closer).
    unique_labels = np.unique(labels)
    centroids = {}
    for lbl in unique_labels:
        if lbl == -1:
            continue
        idxs = np.where(labels == lbl)[0]
        if idxs.size == 0:
            continue
        centroids[int(lbl)] = np.mean(transformed[idxs], axis=0)

    orig_metric = metric
    with out_mapping.open("w", encoding="utf8") as fh:
        for i in range(embeddings.shape[0]):
            fname = mapping[i] if (mapping is not None and i < len(mapping)) else str(i)
            lbl = int(labels[i]) if i < len(labels) else None
            score = None
            if lbl is not None and lbl != -1 and lbl in centroids:
                cen = centroids[lbl]
                vec = transformed[i]
                try:
                    if orig_metric == "cosine":
                        # transformed vectors are normalized when metric was cosine
                        score = float(np.dot(vec, cen))
                    else:
                        # use euclidean distance and invert to make higher better
                        dist = float(np.linalg.norm(vec - cen))
                        score = -dist
                except Exception:
                    score = None

            obj = {"face": fname, "cluster": lbl, "cluster_score": (None if score is None else float(score))}
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        logger.info("Saved JSONL labeled mapping to %s", out_mapping)


def parse_args(argv=None) -> tuple[Path, Path, str, str, dict, Path, bool]:
    p = argparse.ArgumentParser(description="Cluster embeddings (.npy) with sklearn algorithms")
    p.add_argument("-e", "--embeddings", type=Path, required=True, help="Path to embeddings .npy file")
    p.add_argument("-m", "--mapping",    type=Path, required=True, help="Optional mapping .txt file with one filename per embedding")
    p.add_argument("-a", "--algorithm",  type=str, default="kmeans", choices=("kmeans", "agglomerative", "dbscan", "spectral"))
    p.add_argument("-M", "--metric",     type=str, default="euclidean", help="Distance metric (euclidean, cosine, manhattan, precomputed, etc.)")
    p.add_argument("-p", "--params",     type=json.loads, default={}, help="JSON string with algorithm-specific parameters")
    p.add_argument("-o", "--output",     type=Path, required=True, help="Optional .jsonl path to save filename\tlabel mapping")
    p.add_argument("-v", "--verbose",    type=bool, action="store_true")
    args = p.parse_args(argv)
    embeddings_path = args.embeddings
    mapping_path = args.mapping
    algorithm = args.algorithm
    metric = args.metric
    params = args.params
    out_mapping = args.output
    verbose = args.verbose
    return embeddings_path, mapping_path, algorithm, metric, params, out_mapping, verbose


def main(argv=None):
    embeddings_path, mapping_path, algorithm, metric, params, out_mapping, verbose = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s: %(message)s")
    run_clustering(
        embeddings_path=embeddings_path,
        mapping_path=mapping_path,
        algorithm=algorithm,
        metric=metric,
        params=params,
        out_mapping=out_mapping,
    )


if __name__ == "__main__":
    main()
