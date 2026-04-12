from math import sqrt
from collections.abc import Iterable
from typing import Literal

from peoplegator_namedfaces.clustering.evaluation.src.schemas import (
    PeopleGatorNamedFaces__ClusterPrediction,
    PeopleGatorNamedFaces__GroundTruth,
)


def _safe_div(num: int | float, den: int | float) -> float:
    return float(num) / den if den else 0.0

def _comb2(n: int) -> int:
    return 0 if n < 2 else n * (n - 1) // 2


def _build_label_maps(
    ground_truth: Iterable[PeopleGatorNamedFaces__GroundTruth],
    predictions: Iterable[PeopleGatorNamedFaces__ClusterPrediction],
) -> tuple[dict[str, list[str]], dict[str, int], list[str]]:
    """Return (true_label_map, pred_cluster_map, faces_list).

    - `true_label_map`: face -> person_name or None (background / unlabeled)
    - `pred_cluster_map`: face -> cluster int (use -1 for background / missing)
    - `faces_list`: deterministic list of all faces (union)

    Convention: predicted cluster < 0 (e.g. -1) is treated as background.
    """
    gt_faces = set(gt.face for gt in ground_truth)
    pred_faces = set(p.face for p in predictions)
    extra_gt_faces = gt_faces - pred_faces
    
    true_map: dict[str, list[str]] = {gt_face: [] for gt_face in gt_faces}
    for gt in ground_truth:
        true_map[gt.face].append(gt.person_name)
    
    pred_map: dict[str, int] = {}
    for p in predictions:
        # treat any negative cluster as background marker
        if p.face in pred_map:
            raise ValueError(f"Duplicate prediction for face {p.face}")
        pred_map[p.face] = int(p.cluster)
    faces = sorted([p.face for p in predictions] + list(extra_gt_faces))
    # ensure any face that appears only in truth/pred is present in both maps
    for f in faces:
        true_map.setdefault(f, [])
        pred_map.setdefault(f, -1)
    return true_map, pred_map, faces


def evaluate_pairwise_clusters(
    ground_truth: list[PeopleGatorNamedFaces__GroundTruth],
    cluster_predictions: list[PeopleGatorNamedFaces__ClusterPrediction],
    interagreement_size: int = 4,
    interagreement_strategy: Literal["probabilistic", "majority", "union", "intersection"] = "probabilistic",
) -> dict[str, object]:
    """Evaluate pairwise clustering metrics.

    Rules and conventions:
    - Any face not present in `ground_truth` is considered background (true label = None).
    - Any predicted cluster < 0 (or missing) is considered background.
    - A predicted pair is considered "same cluster" only when both faces share the same
      non-background predicted cluster id.

    Returns a dict with:
    - confusion: TP/FP/FN/TN counts and total_pairs
    - metrics: precision, recall, f1, accuracy, jaccard, fowlkes_mallows, rand_index, adjusted_rand_index
    - support: counts of labeled faces, background faces
    """
    true_map, pred_map, faces = _build_label_maps(ground_truth, cluster_predictions)
    rows: list[int] = []
    cols: list[int] = []
    data_gt: list[float] = []
    data_pred: list[float] = []
    n = len(faces)
    try:
        interagreement_fn = {
            "probabilistic": lambda same_count: same_count / interagreement_size,
            "majority": lambda same_count: 1.0 if same_count > interagreement_size / 2 else 0.0,
            "union": lambda same_count: 1.0 if same_count > 0 else 0.0,
            "intersection": lambda same_count: 1.0 if same_count == interagreement_size else 0.0,
        }[interagreement_strategy]
    except KeyError:
        raise ValueError(f"Invalid interagreement_strategy: {interagreement_strategy}")
    
    total_pairs = 0
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(i)
            cols.append(j)
            a, b = faces[i], faces[j]
            a_clusters, b_clusters = true_map[a], true_map[b]
            num_same = sum(1 for ac in a_clusters for bc in b_clusters if ac == bc)
            predicted_same = 1 if (pred_map[a] >= 0 and pred_map[a] == pred_map[b]) else 0
            is_background_pair = (not a_clusters) and (not b_clusters)
            p_same = interagreement_fn(num_same)
            if not is_background_pair and predicted_same:
                TP += p_same
                FP += 1.0 - p_same
                FN += 0.0
                TN += 0.0
                total_pairs += 1
            elif not is_background_pair and not predicted_same:
                TP += 0.0
                FP += 0.0
                FN += p_same
                TN += 1.0 - p_same
                total_pairs += 1
            elif is_background_pair and predicted_same:
                continue
            elif is_background_pair and not predicted_same:
                continue
            else:
                raise RuntimeError(f"Unexpected case in pairwise evaluation {is_background_pair=} {predicted_same=}")

    precision: float = _safe_div(TP, TP + FP)
    recall: float = _safe_div(TP, TP + FN)
    f1: float = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    accuracy: float = _safe_div(TP + TN, total_pairs)
    jaccard: float = _safe_div(TP, TP + FP + FN)
    fowlkes: float = _safe_div(TP, int(sqrt((TP + FP) * (TP + FN)))) if (TP + FP) and (TP + FN) else 0.0
    rand_index: float = accuracy

    # Adjusted Rand Index (combinatorial formulation)
    # Build contingency between true labels and predicted clusters
    label_to_idx: dict[str, int] = {}
    pred_to_idx: dict[int, int] = {}
    true_idx = 0
    pred_idx = 0
    true_labels: list[list[int]] = []
    pred_labels: list[int] = []
    for f in faces:
        true = true_map[f]
        pl = pred_map[f]
        for tl in true:
            if tl not in label_to_idx:
                label_to_idx[tl] = true_idx
                true_idx += 1
        if pl not in pred_to_idx:
            pred_to_idx[pl] = pred_idx
            pred_idx += 1
        true_labels.append([label_to_idx[tl] for tl in true])
        pred_labels.append(pred_to_idx[pl])

    # contingency matrix as dict
    contingency: dict[tuple[int, int], int] = {}
    row_sums: dict[int, int] = {}
    col_sums: dict[int, int] = {}
    for t, p in zip(true_labels, pred_labels):
        for tl in t:
            contingency[(tl, p)] = contingency.get((tl, p), 0) + 1
            row_sums[tl] = row_sums.get(tl, 0) + 1
            col_sums[p] = col_sums.get(p, 0) + 1

    sum_comb_c = sum(_comb2(v) for v in contingency.values())
    sum_comb_rows = sum(_comb2(v) for v in row_sums.values())
    sum_comb_cols = sum(_comb2(v) for v in col_sums.values())
    total_comb = _comb2(len(faces))

    expected_index = (sum_comb_rows * sum_comb_cols) / total_comb if total_comb else 0.0
    max_index = 0.5 * (sum_comb_rows + sum_comb_cols)
    ari = (
        (sum_comb_c - expected_index) / (max_index - expected_index)
        if (max_index - expected_index)
        else 0.0
    )

    result = {
        "confusion": {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "total_pairs": total_pairs},
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "jaccard": jaccard,
            "fowlkes_mallows": fowlkes,
            "rand_index": rand_index,
            "adjusted_rand_index": ari,
        },
        "support": {
            "n_faces": len(faces),
            "n_labeled_faces": sum(1 for v in true_map.values() if v is not None),
            "n_background_faces": sum(1 for v in true_map.values() if v is None),
        },
    }
    return result

def _hungarian_minimize(cost_matrix: list[list[float]]) -> list[int]:
    """Solve assignment problem (minimize total cost).

    Returns list `assign` of length n_rows where assign[i] is the assigned column index
    or -1 if not assigned. This implementation expects n_rows <= n_cols; it will
    work for rectangular matrices by padding.
    """
    n = len(cost_matrix)
    if n == 0:
        return []
    m = max(len(row) for row in cost_matrix)
    for row in cost_matrix:
        if len(row) < m:
            row.extend([0.0] * (m - len(row)))

    # Prefer using scipy's implementation when available for speed and robustness.
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np

        _arr = np.asarray(cost_matrix, dtype=float)
        row_ind, col_ind = linear_sum_assignment(_arr)
        assign = [-1] * n
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if 0 <= r < n:
                assign[int(r)] = int(c)
            # ignore assignments outside original dimensions
        return assign
    except ImportError:
        raise RuntimeError("Hungarian algorithm requires scipy. Please install scipy to use optimal assignment evaluation.")



def evaluate_with_optimal_assignment(
    ground_truth: list[PeopleGatorNamedFaces__GroundTruth],
    cluster_predictions: list[PeopleGatorNamedFaces__ClusterPrediction],
) -> dict[str, object]:
    """Assign predicted clusters to ground-truth labels by maximizing IoU (Hungarian).

    Returns:
    - `iou_matrix`: nested list rows=true_labels, cols=pred_clusters
    - `true_labels`: list of true label strings (including "__bg__") matching iou_matrix rows
    - `pred_clusters`: list of predicted cluster ids matching iou_matrix cols
    - `assignment`: dict true_label -> assigned_pred_cluster (or None)
    - `classification`: classification metrics after relabelling predicted clusters by assignment
    """
    true_map, pred_map, faces = _build_label_maps(ground_truth, cluster_predictions)

    # build label sets
    true_to_idx: dict[str, int] = {
        v: k for k,v in enumerate(set(
            tl for tls in true_map.values() 
            for tl in tls if tl is not None
        ))
    }
    pred_to_idx: dict[int, int] = {
        v: k for k,v in enumerate(set(
            pc for pc in pred_map.values() if pc >= 0
        ))
    }
    
    true_labels: list[None | str] = [None] * len(true_to_idx)
    for k, v in true_to_idx.items():
        true_labels[v] = k
    pred_clusters: list[None | int] = [None] * len(pred_to_idx)
    for k, v in pred_to_idx.items():
        pred_clusters[v] = k

    # counts and intersection
    size_true = [0] * len(true_labels)
    size_pred = [0] * len(pred_clusters)
    intersection = [[0] * len(pred_clusters) for _ in range(len(true_labels))]
    for f in faces:
        true = true_map[f]
        tl = str(true) if true is not None else "__bg__"
        pc = pred_map[f]
        i = true_to_idx[tl]
        j = pred_to_idx[pc]
        intersection[i][j] += 1
        size_true[i] += 1
        size_pred[j] += 1

    # build IoU matrix
    iou = []
    for i in range(len(true_labels)):
        row = []
        for j in range(len(pred_clusters)):
            inter = intersection[i][j]
            union = size_true[i] + size_pred[j] - inter
            row.append(_safe_div(inter, union))
        iou.append(row)

    # convert IoU to cost (minimize). pad to ensure rows <= cols
    max_iou = max((max(row) for row in iou), default=0.0)
    cost = [[max_iou - v for v in row] for row in iou]
    # if more rows than cols, transpose handled in _hungarian_minimize
    assign = _hungarian_minimize([list(r) for r in cost])

    # build assignment mapping true_label -> pred_cluster or None
    assignment: dict[str, int | None] = {}
    for i, j in enumerate(assign):
        label = true_labels[i]
        if label is None:
            continue
        if j is None or j < 0 or j >= len(pred_clusters):
            assignment[label] = None
        else:
            assignment[label] = pred_clusters[j]

    # relabel predicted clusters per assignment
    predicted_label_of_face: dict[str, str] = {}
    # build reverse mapping pred_cluster -> assigned true label
    rev_map: dict[int, str | None] = {pc: None for pc in pred_clusters if pc is not None}
    for tlabel, pc in assignment.items():
        if pc is not None:
            rev_map[pc] = tlabel

    for f in faces:
        pc = pred_map[f]
        assigned = rev_map.get(pc)
        predicted_label_of_face[f] = assigned if assigned is not None else "__bg__"

    # compute classification metrics per label
    all_labels = list(true_labels)
    per_class: dict[str, dict[str, float | int]] = {}
    TP_sum = FP_sum = FN_sum = 0
    for label in all_labels:
        if label is None:
            continue
        TP = FP = FN = 0
        for f in faces:
            true_l = true_map[f] if true_map[f] is not None else "__bg__"
            pred_l = predicted_label_of_face[f]
            if pred_l == label and true_l == label:
                TP += 1
            elif pred_l == label and true_l != label:
                FP += 1
            elif pred_l != label and true_l == label:
                FN += 1
        TP_sum += TP
        FP_sum += FP
        FN_sum += FN
        prec = _safe_div(TP, TP + FP)
        rec = _safe_div(TP, TP + FN)
        f1v = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0
        per_class[label] = {"TP": TP, "FP": FP, "FN": FN, "precision": prec, "recall": rec, "f1": f1v}

    micro_prec = _safe_div(TP_sum, TP_sum + FP_sum)
    micro_rec = _safe_div(TP_sum, TP_sum + FN_sum)
    micro_f1 = _safe_div(2 * micro_prec * micro_rec, micro_prec + micro_rec) if (micro_prec + micro_rec) else 0.0

    macro_prec = sum(d["precision"] for d in per_class.values()) / len(per_class) if per_class else 0.0
    macro_rec = sum(d["recall"] for d in per_class.values()) / len(per_class) if per_class else 0.0
    macro_f1 = sum(d["f1"] for d in per_class.values()) / len(per_class) if per_class else 0.0

    accuracy = sum(1 for f in faces if (true_map[f] if true_map[f] is not None else "__bg__") == predicted_label_of_face[f]) / len(faces) if faces else 0.0

    return {
        "iou_matrix": iou,
        "true_labels": true_labels,
        "pred_clusters": pred_clusters,
        "assignment": assignment,
        "predicted_label_of_face": predicted_label_of_face,
        "classification": {
            "per_class": per_class,
            "micro": {"precision": micro_prec, "recall": micro_rec, "f1": micro_f1},
            "macro": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1},
            "accuracy": accuracy,
        },
    }


__all__ = [
    "evaluate_pairwise_clusters",
    "evaluate_with_optimal_assignment",
]


def main():
    import argparse
    from ..storage import load_jsonl
    import json
    parser = argparse.ArgumentParser(description="Evaluate clustering predictions against ground truth.")
    parser.add_argument("-g", "--ground_truth", type=str, required=True, help="Path to ground truth JSONL file")
    parser.add_argument("-p", "--predictions", type=str, required=True, help="Path to cluster predictions JSONL file")
    args = parser.parse_args()

    ground_truth = load_jsonl(args.ground_truth, PeopleGatorNamedFaces__GroundTruth)
    predictions = load_jsonl(args.predictions, PeopleGatorNamedFaces__ClusterPrediction)

    pairwise_results = evaluate_pairwise_clusters(ground_truth, predictions)
    #optimal_results = evaluate_with_optimal_assignment(ground_truth, predictions)

    #print("Pairwise Evaluation:")
    print(json.dumps(pairwise_results, indent=2))
    #print("\nOptimal Assignment Evaluation:")
    #print(optimal_results)
