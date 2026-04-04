from collections.abc import Callable
import pathlib
from typing import TypeVar
from collections.abc import Iterable
import numpy as np
from peoplegator_namedfaces.clustering.evaluation.src.evaluation.cluster import Literal
from peoplegator_namedfaces.clustering.evaluation.src.storage import load_jsonl, save_json
from peoplegator_namedfaces.clustering.evaluation.src.schemas import (
    PeopleGatorNamedFaces__AssignmentReport,
    PeopleGatorNamedFaces__ClusterEvaluationReport,
    PeopleGatorNamedFaces__PairwiseReport,
    PeopleGatorNamedFaces__ClusterPrediction,
    PeopleGatorNamedFaces__GroundTruth,
)

InteragreementStrategy = Literal[
    "union", "intersection", "majority", "probabilistic"
]

def build_ground_truth_mappings(
    ground_truth: list[PeopleGatorNamedFaces__GroundTruth],
    annotators: list[str],
) -> tuple[
    dict[str, dict[str, set[str]]],
    dict[str, dict[str, set[str]]], 
    set[str],
    list[str]
]:
    """Build mappings from ground-truth annotations.

    Constructs two maps:
    - `face_to_name_map`: mapping face id -> annotator -> set of assigned person names
    - `name_to_face_map`: mapping person name -> annotator -> set of faces assigned

    Args:
        ground_truth: List of `PeopleGatorNamedFaces__NamedFaceGroundTruth` records.
        annotators: List of annotator identifiers expected in the ground truth.

    Returns:
        A tuple `(face_to_name_map, name_to_face_map, names, faces)` where:
        - `face_to_name_map` is a dict mapping face -> annotator -> set of names
        - `name_to_face_map` is a dict mapping name -> annotator -> set of faces
        - `names` is the set of all person names present in `ground_truth`
        - `faces` is the list of face identifiers present in `ground_truth`

    Raises:
        No exceptions are raised by this function.
    """
    names: set[str] = set(g.person_name for g in ground_truth)
    faces: list[str] = [f.face for f in ground_truth]
    
    face_to_name_map: dict[str, dict[str, set[str]]] = {f: {
        annotator: set() for annotator in annotators
    } for f in set(faces)}
    
    for gt in ground_truth:
        face_to_name_map[gt.face][gt.annotator].add(gt.person_name)
    
    name_to_face_map: dict[str, dict[str, set[str]]] = {n: {
        annotator: set() for annotator in annotators
    } for n in names}
    
    for gt in ground_truth:
        name_to_face_map[gt.person_name][gt.annotator].add(gt.face)
    
    return face_to_name_map, name_to_face_map, names, faces

def enforce_single_cluster_per_annotator(
    face_to_name_map: dict[str, dict[str, set[str]]],
    force: bool = False,
) -> dict[str, dict[str, str]]:
    """Ensure each annotator assigns at most one name per face.

    For each face and annotator, collapse multiple assigned names into a single
    name. If `force` is False and any annotator assigns more than one name to
    the same face, a `ValueError` is raised. When `force` is True the first
    name from the set is chosen deterministically via `next(iter(...))`.

    Args:
        face_to_name_map: Mapping face -> annotator -> set of names.
        force: If True, silently choose one name when multiple are present.

    Returns:
        A mapping face -> annotator -> single name (string).

    Raises:
        ValueError: If an annotator has multiple cluster/name assignments for a
            face and `force` is False.
    """
    single_cluster_face_to_name_map: dict[str, dict[str, str]] = {}
    for face, annotator_map in face_to_name_map.items():
        single_cluster_face_to_name_map[face] = {}
        for annotator, clusters in annotator_map.items():
            if len(clusters) > 1:
                if force:
                    chosen_cluster = next(iter(clusters))
                    single_cluster_face_to_name_map[face][annotator] = chosen_cluster
                else:
                    raise ValueError(f"Face {face} has multiple cluster assignments for annotator {annotator}: {clusters}")
            elif len(clusters) == 0:
                continue
            else:
                single_cluster_face_to_name_map[face][annotator] = next(iter(clusters))
    return single_cluster_face_to_name_map

def build_prediction_mappings(
    predictions: list[PeopleGatorNamedFaces__ClusterPrediction],
) -> tuple[
    dict[str, set[int]],
    dict[int, set[str]],
    set[int],
    list[str], 
]:
    """Build mappings describing model predictions.

    Args:
        predictions: List of `PeopleGatorNamedFaces__ClusterPrediction` records.

    Returns:
        A tuple `(face_to_cluster_map, cluster_to_face_map, clusters, faces)`
        where:
        - `face_to_cluster_map` maps face id -> set of predicted cluster ids
        - `cluster_to_face_map` maps cluster id -> set of face ids contained
        - `clusters` is the set of cluster ids seen in `predictions`
        - `faces` is the list of face ids seen in `predictions`

    Raises:
        No exceptions are raised by this function.
    """
    clusters = set(p.cluster for p in predictions)
    faces = [p.face for p in predictions]
    
    face_to_cluster_map: dict[str, set[int]] = {f: set() for f in set(faces)}
    for pred in predictions:
        face_to_cluster_map[pred.face].add(pred.cluster)
    
    cluster_to_face_map: dict[int, set[str]] = {c: set() for c in clusters}
    for pred in predictions:
        cluster_to_face_map[pred.cluster].add(pred.face)
            
    return face_to_cluster_map, cluster_to_face_map, clusters, faces

def enforce_single_cluster_per_face(
    face_to_cluster_map: dict[str, set[int]],
    force: bool = False,
) -> dict[str, int]:
    """Ensure each face is assigned to at most one predicted cluster.

    If a face is present in multiple clusters and `force` is False a
    `ValueError` is raised. When `force` is True one cluster id is chosen
    deterministically via `next(iter(...))`.

    Args:
        face_to_cluster_map: Mapping face id -> set of predicted cluster ids.
        force: If True, silently choose one cluster when multiple are present.

    Returns:
        Mapping face id -> single cluster id.

    Raises:
        ValueError: If a face has multiple cluster assignments and `force` is False.
    """
    single_cluster_face_to_cluster_map: dict[str, int] = {}
    for face, clusters in face_to_cluster_map.items():
        if len(clusters) > 1:
            if force:
                chosen_cluster = next(iter(clusters))
                single_cluster_face_to_cluster_map[face] = chosen_cluster
            else:
                raise ValueError(f"Face {face} has multiple cluster assignments: {clusters}")
        elif len(clusters) == 0:
            continue
        else:
            single_cluster_face_to_cluster_map[face] = next(iter(clusters))
    return single_cluster_face_to_cluster_map


def _interagreement_set_strategy_factory(
    interagreement_strategy: InteragreementStrategy,
    interagreement_size: int
) -> Callable[[Iterable[set[str]]], dict[str, float]]:
    
    
    def _create_counts_dict(clusters: Iterable[set[str]]) -> dict[str, float]:
        """Count occurrences of names across a sequence of annotator sets.

        Args:
            clusters: Iterable of sets of names (per-annotator assignments).

        Returns:
            Dict mapping name -> count (float) of occurrences across `clusters`.
        """
        counts_dict: dict[str, float] = {}
        for c in clusters:
            for name in c:
                counts_dict[name] = counts_dict.get(name, 0.0) + 1.0
        return counts_dict
    
    def _union_interagreement(clusters: Iterable[set[str]]) -> dict[str, float]:
        """Union strategy: any name appearing in any annotator set gets weight 1."""
        union_dict: dict[str, float] = _create_counts_dict(clusters)
        return {name: 1.0 for name in union_dict.keys()}
    
    def _intersection_interagreement(clusters: Iterable[set[str]]) -> dict[str, float]:
        """Intersection strategy: names present in all annotator sets get weight 1."""
        intersection_dict: dict[str, float] = _create_counts_dict(clusters)
        return { name: 1.0 for name, count in intersection_dict.items() if count == interagreement_size }
    
    def _majority_interagreement(clusters: Iterable[set[str]]) -> dict[str, float]:
        """Majority strategy: names with > half of annotator votes get weight 1."""
        majority_dict: dict[str, float] = _create_counts_dict(clusters)
        return {name: 1.0 for name, count in majority_dict.items() if count > interagreement_size / 2}
    
    def _probabilistic_interagreement(clusters: Iterable[set[str]]) -> dict[str, float]:
        """Probabilistic strategy: assigns fractional weights = count / size."""
        probabilistic_dict: dict[str, float] = _create_counts_dict(clusters)
        return {name: count / interagreement_size for name, count in probabilistic_dict.items()}
    
    match interagreement_strategy:
        case "union":
            return _union_interagreement
        case "intersection":
            return _intersection_interagreement
        case "majority":
            return _majority_interagreement
        case "probabilistic":
            return _probabilistic_interagreement
        case _:
            raise ValueError(f"Invalid interagreement strategy: {interagreement_strategy}")

def _interagreement_strategy_factory(
    interagreement_strategy: InteragreementStrategy,
    interagreement_size: int
) -> Callable[[int], float]:
    def _union_interagreement(num_true_labels: int) -> float:
        """Union strategy on counts: returns 1.0 if any annotator agrees."""
        return 1.0 if num_true_labels > 0 else 0.0
    
    def _intersection_interagreement(num_true_labels: int) -> float:
        """Intersection strategy on counts: returns 1.0 only if all annotators agree."""
        return 1.0 if num_true_labels == interagreement_size else 0.0
    
    def _majority_interagreement(num_true_labels: int) -> float:
        """Majority strategy on counts: returns 1.0 if > half of annotators agree."""
        return 1.0 if num_true_labels > interagreement_size / 2 else 0.0
    
    def _probabilistic_interagreement(num_true_labels: int) -> float:
        """Probabilistic strategy on counts: returns fraction num_true_labels / interagreement_size."""
        return num_true_labels / interagreement_size
    
    match interagreement_strategy:
        case "union":
            return _union_interagreement
        case "intersection":
            return _intersection_interagreement
        case "majority":
            return _majority_interagreement
        case "probabilistic":
            return _probabilistic_interagreement
        case _:
            raise ValueError(f"Invalid interagreement strategy: {interagreement_strategy}")

def count_same_cluster_assignemts(
    clusters1: dict[str, str],
    clusters2: dict[str, str],
) -> int:
    """Count annotators that assigned the same cluster/name in two mappings.

    Args:
        clusters1: Mapping annotator -> cluster/name string for the first item.
        clusters2: Mapping annotator -> cluster/name string for the second item.

    Returns:
        The integer count of annotators for which both mappings exist and are equal.
    """
    count = 0
    annotators = set(clusters1.keys()) | set(clusters2.keys())
    for a in annotators:
        c1 = clusters1.get(a)
        c2 = clusters2.get(a)
        if c1 is not None and c2 is not None and c1 == c2:
            count += 1
    return count

def build_pairwise_matrix(
    true_map: dict[str, dict[str, str]],
    pred_map: dict[str, int],
    true_faces: list[str],
    pred_faces: list[str],
    interagreement_size: int,
    interagreement_strategy: InteragreementStrategy = "probabilistic",
):
    """Build pairwise lists used for pairwise clustering evaluation.

    For every unordered pair of faces present in either the ground truth or
    predictions this function computes a soft ground-truth association value
    according to the chosen interagreement strategy and a binary prediction of
    whether the two faces are in the same predicted cluster.

    Args:
        true_map: Mapping face -> annotator -> single true cluster/name.
        pred_map: Mapping face -> predicted cluster id (single cluster per face).
        true_faces: List of face ids from ground truth.
        pred_faces: List of face ids from predictions.
        interagreement_size: Number of annotators used for interagreement.
        interagreement_strategy: Strategy for combining annotator labels.

    Returns:
        A tuple `(data_gt, data_pred, row, col, all_faces)` where `data_gt` is a
        list of soft ground-truth association weights, `data_pred` is a list of
        predicted binary same-cluster values (as floats), `row` and `col` are
        index lists for the pairs, and `all_faces` is the combined face list.
    """
    interagreement_strategy_fn = _interagreement_strategy_factory(
        interagreement_strategy, interagreement_size=interagreement_size
    )
    all_faces = sorted(pred_faces + list(set(true_faces)))
    row: list[int] = []
    col: list[int] = []
    data_gt: list[float] = []
    data_pred: list[float] = []
    for i in range(len(all_faces)):
        for j in range(i + 1, len(all_faces)):
            f1 = all_faces[i]
            f2 = all_faces[j]
            if f1 not in true_map and f2 not in true_map:
                continue
            true_counts = count_same_cluster_assignemts(true_map.get(f1, {}), true_map.get(f2, {}))
            interageed_association = interagreement_strategy_fn(true_counts)
            pred_same = pred_map.get(f1) == pred_map.get(f2)
            row.append(i)
            col.append(j)
            data_gt.append(interageed_association)
            data_pred.append(float(pred_same))
    return data_gt, data_pred, row, col, all_faces

def compute_pairwise_errors(
    pairwise_gt: list[float],
    pairwise_pred: list[float],
) -> tuple[float, float, float, float]:
    """Compute soft pairwise TP/FP/TN/FN counts.

    The ground-truth values in `pairwise_gt` are soft (between 0 and 1) while
    `pairwise_pred` are treated as hard predictions (1.0 => same cluster).

    Args:
        pairwise_gt: List of ground-truth association weights (floats).
        pairwise_pred: List of predicted same-cluster floats (1.0 or 0.0).

    Returns:
        Tuple `(TP, FP, TN, FN)` of floats representing soft counts.
    """
    TP = FP = TN = FN = 0.0
    for gt, pred in zip(pairwise_gt, pairwise_pred):
        is_background_pair = gt == 0.0
        predicted_same = pred == 1.0
        if not is_background_pair and predicted_same:
            TP += gt
        elif not is_background_pair and not predicted_same:
            FN += gt
            TN += 1.0 - gt
        elif is_background_pair and predicted_same:
            FP += 1.0 - gt
        elif is_background_pair and not predicted_same:
            TN += 1.0 - gt
        else:
            raise RuntimeError(f"Unexpected case in pairwise evaluation {is_background_pair=} {predicted_same=}")
    return TP, FP, TN, FN


def _fuzzy_set_intersection(
    set1: dict[str, float],
    set2: dict[str, float],
) -> dict[str, float]:
    """Compute fuzzy intersection of two fuzzy sets.

    Args:
        set1: Mapping element -> membership weight in first fuzzy set.
        set2: Mapping element -> membership weight in second fuzzy set.

    Returns:
        Mapping element -> min(weight1, weight2) for elements present in both.
    """
    intersection: dict[str, float] = {}
    for key in set(set1.keys()) | set(set2.keys()):
        if key in set1 and key in set2:
            intersection[key] = min(set1[key], set2[key])
    return intersection

def _fuzzy_set_union(
    set1: dict[str, float],
    set2: dict[str, float],
) -> dict[str, float]:
    """Compute fuzzy union of two fuzzy sets.

    Returns the element-wise maximum of membership weights.
    """
    union: dict[str, float] = {}
    for key in set(set1.keys()) | set(set2.keys()):
        union[key] = max(set1.get(key, 0.0), set2.get(key, 0.0))
    return union

def _fuzzy_set_subtraction(
    set1: dict[str, float],
    set2: dict[str, float],
) -> dict[str, float]:
    """Compute fuzzy set subtraction (set1 - set2) in fuzzy logic terms.

    For each element returns min(weight_in_set1, 1 - weight_in_set2).
    """
    subtraction: dict[str, float] = {}
    for key in set(set1.keys()) | set(set2.keys()):
        subtraction[key] = min(set1.get(key, 0.0), 1.0 - set2.get(key, 0.0))
    return subtraction

def _fuzzy_set_size(
    fuzzy_set: dict[str, float]
) -> float:
    """Return the size (sum of membership weights) of a fuzzy set."""
    return sum(fuzzy_set.values())

def build_contingency_matrix(
    true_name_to_face_map: dict[str, dict[str, set[str]]],
    pred_cluster_to_face_map: dict[int, set[str]],
    true_clusters: set[str],
    pred_clusters: set[int],
    interagreement_size: int,
    interagreement_strategy: InteragreementStrategy = "probabilistic",
):
    """Build fuzzy contingency (intersection and union) matrices.

    Each true cluster (name) is converted into a fuzzy set across faces using
    the chosen interagreement set strategy; predicted clusters are crisp sets
    (faces belong with membership 1.0). The function computes intersection and
    union sizes for every (true_name, pred_cluster) pair.

    Args:
        true_name_to_face_map: Mapping true name -> annotator -> set of faces.
        pred_cluster_to_face_map: Mapping predicted cluster id -> set of faces.
        true_clusters: Set of true cluster (name) identifiers to include.
        pred_clusters: Set of predicted cluster ids to include.
        interagreement_size: Number of annotators.
        interagreement_strategy: Strategy for combining annotators into fuzzy sets.

    Returns:
        Tuple `(intersections_matrix, union_matrix, _true_clusters, _pred_clusters, true_cluster_face_fuzzy_sets)`
        where matrices are NumPy arrays with sizes `(len(true_clusters), len(pred_clusters))`.
    """
    _true_clusters = sorted(true_clusters)
    _pred_clusters = sorted(pred_clusters)
    true_cluster_to_idx: dict[str, int] = {
        c: i for i, c in enumerate(_true_clusters)}
    pred_cluster_to_idx: dict[int, int] = {
        c: i for i, c in enumerate(_pred_clusters)}
    intersections_matrix: np.ndarray = np.zeros(
        (len(_true_clusters), len(_pred_clusters)), dtype=int)
    union_matrix: np.ndarray = np.zeros(
        (len(_true_clusters), len(_pred_clusters)), dtype=int)
    interagreement_stategy_fn = _interagreement_set_strategy_factory(
        interagreement_strategy, interagreement_size=interagreement_size
    )
    true_cluster_face_fuzzy_sets: dict[str, dict[str, float]] = {
        true_cluster: interagreement_stategy_fn(true_annotator_faces.values())
        for true_cluster, true_annotator_faces in true_name_to_face_map.items()
    }
    for true_cluster, true_annotator_faces in true_name_to_face_map.items():
        for pred_cluster, pred_faces in pred_cluster_to_face_map.items():
            pred_face_fuzzy_set = {
                f: 1.0 for f in pred_faces
            }
            fuzzy_intersection = _fuzzy_set_intersection(true_cluster_face_fuzzy_sets[true_cluster], pred_face_fuzzy_set)
            fuzzy_union = _fuzzy_set_union(true_cluster_face_fuzzy_sets[true_cluster], pred_face_fuzzy_set)
            intersections_matrix[
                true_cluster_to_idx[true_cluster],
                pred_cluster_to_idx[pred_cluster]
            ] = _fuzzy_set_size(fuzzy_intersection)
            union_matrix[
                true_cluster_to_idx[true_cluster],
                pred_cluster_to_idx[pred_cluster]
            ] = _fuzzy_set_size(fuzzy_union)
    return intersections_matrix, union_matrix, _true_clusters, _pred_clusters, true_cluster_face_fuzzy_sets


T = TypeVar("T", bound=np.ndarray)

def _comb2(n: T) -> T:
    """Vectorized combination n choose 2 for NumPy arrays.

    Args:
        n: NumPy array of non-negative integers.

    Returns:
        NumPy array where each element is n*(n-1)/2 for n>=2, else 0.
    """
    n_lt_2 = n < 2
    n_selection = n[~n_lt_2]
    result = n.copy()
    result[n_lt_2] = 0
    result[~n_lt_2] = n_selection * (n_selection - 1) / 2
    return result

def compute_adjusted_rand_index(
    contingency_matrix: np.ndarray,
) -> float:
    """Compute Adjusted Rand Index (ARI) from a contingency matrix.

    Args:
        contingency_matrix: 2D NumPy array of contingency counts between true
            and predicted clusters (rows=true, cols=pred).

    Returns:
        Adjusted Rand Index as a float.
    """
    sum_comb_c = _comb2(contingency_matrix).sum()
    sum_comb_rows = _comb2(contingency_matrix.sum(axis=0)).sum()
    sum_comb_cols = _comb2(contingency_matrix.sum(axis=1)).sum()
    n = contingency_matrix.sum()
    index = sum_comb_c
    expected_index = sum_comb_rows * sum_comb_cols / _comb2(n)
    max_index = (sum_comb_rows + sum_comb_cols) / 2
    rand_index = (index - expected_index) / (max_index - expected_index) if max_index != expected_index else 0.0
    return float(rand_index)

def compute_pairwise_metrics(
    ground_truth: list[PeopleGatorNamedFaces__GroundTruth],
    predictions: list[PeopleGatorNamedFaces__ClusterPrediction],
    interagreement_size: int,
    interagreement_strategy: InteragreementStrategy = "probabilistic",
) -> PeopleGatorNamedFaces__PairwiseReport:
    """Compute pairwise clustering metrics between ground truth and predictions.

    This function orchestrates mapping construction, enforces single labels per
    face/annotator (using `force=True`), builds pairwise lists, computes soft
    TP/FP/TN/FN, and derives commonly used clustering metrics including ARI.

    Args:
        ground_truth: List of `PeopleGatorNamedFaces__NamedFaceGroundTruth`.
        predictions: List of `PeopleGatorNamedFaces__ClusterPrediction`.
        interagreement_size: Expected number of annotators used for ground truth.
        interagreement_strategy: How to combine annotator labels for soft truth.

    Returns:
        A `PeopleGatorNamedFaces__ClusterPairwiseReport` containing pairwise
        evaluation metrics (precision, recall, f1, jaccard, ARI, etc.).

    Raises:
        ValueError: If the provided `interagreement_size` does not match the
            number of unique annotators in `ground_truth`.
    """
    annotators = list(set(gt.annotator for gt in ground_truth))
    if len(annotators) != interagreement_size:
        raise ValueError(f"Interagreement size {interagreement_size} does not match number of annotators {len(annotators)}")
    gt_face_to_name_map, gt_name_to_face_map, gt_names, gt_faces = build_ground_truth_mappings(ground_truth, annotators)
    pred_face_to_cluster_map, pred_cluster_to_face_map, pred_clusters, pred_faces = build_prediction_mappings(predictions)
    enforced_gt_face_to_name_map = enforce_single_cluster_per_annotator(gt_face_to_name_map, force=True)
    enforced_pred_face_to_cluster_map = enforce_single_cluster_per_face(pred_face_to_cluster_map, force=True)
    pairwise_gt, pairwise_pred, *_ = build_pairwise_matrix(
        enforced_gt_face_to_name_map, 
        enforced_pred_face_to_cluster_map, 
        gt_faces, 
        pred_faces, 
        interagreement_size, 
        interagreement_strategy
    )
    contingency_matrix, *_ = build_contingency_matrix(
        gt_name_to_face_map, 
        pred_cluster_to_face_map, 
        gt_names, 
        pred_clusters, 
        interagreement_size, 
        interagreement_strategy
    )
    
    TP, FP, TN, FN = compute_pairwise_errors(pairwise_gt, pairwise_pred)
    # Soft pairwise clustering metrics
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (TP + TN) / (TP + FP + TN + FN) if TP + FP + TN + FN > 0 else 0.0
    rand_index = (TP + TN) / (TP + FP + TN + FN) if TP + FP + TN + FN > 0 else 0.0
    jaccard = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0.0
    fowlkes_mallows = np.sqrt(precision * recall) if precision > 0 and recall > 0 else 0.0
    # ARI
    adjusted_rand_index = compute_adjusted_rand_index(contingency_matrix)
    
    return PeopleGatorNamedFaces__PairwiseReport(
        true_positives=TP,
        false_positives=FP,
        true_negatives=TN,
        false_negatives=FN,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        jaccard=jaccard,
        fowlkes_mallows=fowlkes_mallows,
        rand_index=rand_index,
        adjusted_rand_index=adjusted_rand_index,
    )
    

def build_optimal_cluster_assignment(
    contingency_matrix: np.ndarray,
    names: list[str],
    clusters: list[int],
) -> list[tuple[int, str]]:
    """Find an optimal assignment between predicted clusters and true names.

    Uses the Hungarian algorithm (via `linear_sum_assignment`) to maximize
    overlap (we minimize the negative intersection/IoU by passing -matrix).

    Args:
        contingency_matrix: 2D array of pairwise scores between true names
            (rows) and predicted clusters (cols). Higher is better.
        names: List of true cluster names matching the rows of the matrix.
        clusters: List of predicted cluster ids matching the columns of the matrix.

    Returns:
        List of `(predicted_cluster_id, true_name)` tuples representing the
        optimal assignment.
    """
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    cluster_pairs = list((int(a), int(b)) for a,b in zip(row_ind, col_ind))
    return [
        (clusters[pred_idx], names[true_idx]) 
        for true_idx, pred_idx in cluster_pairs
    ]

def compute_fuzzy_set_metrics(
    pred_set: dict[str, float],
    true_set: dict[str, float],
) -> PeopleGatorNamedFaces__AssignmentReport:
    intersection = _fuzzy_set_intersection(pred_set, true_set) # fuzzy true positives
    subtraction = _fuzzy_set_subtraction(pred_set, true_set) # fuzzy false positives
    reversed_subtraction = _fuzzy_set_subtraction(true_set, pred_set) # fuzzy false negatives
    union = _fuzzy_set_union(pred_set, true_set) # fuzzy true positives + fuzzy false positives + fuzzy false negatives
    
    intersection_size = _fuzzy_set_size(intersection)
    union_size = _fuzzy_set_size(union)
    subtraction_size = _fuzzy_set_size(subtraction)
    reversed_subtraction_size = _fuzzy_set_size(reversed_subtraction)
    pred_size = _fuzzy_set_size(pred_set)
    true_size = _fuzzy_set_size(true_set)
    
    accuracy = intersection_size / true_size if true_size > 0 else 0.0
    precision = intersection_size / pred_size if pred_size > 0 else 0.0
    recall = intersection_size / true_size if true_size > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    return PeopleGatorNamedFaces__AssignmentReport(
        num_samples=int(true_size),
        true_positives=intersection_size,
        false_positives=subtraction_size,
        true_negatives=0.0,
        false_negatives=reversed_subtraction_size,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score
    )

def compute_cluster_auto_assignment_metrics(
    ground_truth: list[PeopleGatorNamedFaces__GroundTruth],
    predictions: list[PeopleGatorNamedFaces__ClusterPrediction],
    interagreement_size: int,
    interagreement_strategy: InteragreementStrategy = "probabilistic",
) -> dict[str, PeopleGatorNamedFaces__AssignmentReport | list[PeopleGatorNamedFaces__AssignmentReport] | list[str] | list[int] | float]:
    annotators = list(set(gt.annotator for gt in ground_truth))
    if len(annotators) != interagreement_size:
        raise ValueError(f"Interagreement size {interagreement_size} does not match number of annotators {len(annotators)}")
    gt_face_to_name_map, gt_name_to_face_map, gt_names, gt_faces = build_ground_truth_mappings(ground_truth, annotators)
    pred_face_to_cluster_map, pred_cluster_to_face_map, pred_clusters, pred_faces = build_prediction_mappings(predictions)
    enforced_gt_face_to_name_map = enforce_single_cluster_per_annotator(gt_face_to_name_map, force=True)
    enforced_pred_face_to_cluster_map = enforce_single_cluster_per_face(pred_face_to_cluster_map, force=True)
    (
        intersection_matrix, 
        union_matrix, 
        names_list, 
        clusters_list, 
        gt_name_to_face_fuzzy_sets
    ) = build_contingency_matrix(
        gt_name_to_face_map, 
        pred_cluster_to_face_map, 
        gt_names, 
        pred_clusters, 
        interagreement_size, 
        interagreement_strategy
    )
    interagreement_strategy_fn = _interagreement_set_strategy_factory(
        interagreement_strategy, interagreement_size=interagreement_size
    )
    fg_face_to_name_fuzzy_sets = {
        face: interagreement_strategy_fn(gt_face_to_name_map[face].values())
        for face in set(gt_faces)
    }
    nonzero_union_mask = union_matrix > 0
    iou_matrix = np.zeros_like(intersection_matrix, dtype=float)
    iou_matrix[nonzero_union_mask] = intersection_matrix[nonzero_union_mask] / union_matrix[nonzero_union_mask]
    cluster_assignments = build_optimal_cluster_assignment(iou_matrix, names_list, clusters_list)
    cluster_assignments_map = dict(cluster_assignments)
    
    per_cluster_metrics: dict[str, PeopleGatorNamedFaces__AssignmentReport] = {
        name: PeopleGatorNamedFaces__AssignmentReport(
            num_samples=0,
            true_positives=0,
            false_positives=0,
            true_negatives=0.0,
            false_negatives=0,
            accuracy=0.0, # to be computed after summing up TP, FP, TN, FN
            precision=0.0, # to be computed after summing up TP, FP TN, FN
            recall=0.0, # to be computed after summing up TP, FP TN, FN
            f1_score=0.0, # to be computed after summing up TP, FP TN, FN
        ) for _, name in cluster_assignments
    }
    
    for face in pred_faces + list(set(gt_faces) - set(pred_faces)):
        cluster = enforced_pred_face_to_cluster_map.get(face)
        assigned_name = cluster_assignments_map.get(cluster) if cluster is not None else None
        face_names_fuzzy_set = fg_face_to_name_fuzzy_sets.get(face)
        if cluster is None or face_names_fuzzy_set is None or assigned_name is None:
            continue
        for true_name, probability in face_names_fuzzy_set.items():
            if true_name == assigned_name:
                per_cluster_metrics[assigned_name].num_samples += probability
                per_cluster_metrics[assigned_name].true_positives += probability
            else:
                per_cluster_metrics[true_name].num_samples += probability
                per_cluster_metrics[true_name].false_negatives += probability
                per_cluster_metrics[assigned_name].num_samples += 1 - probability
                per_cluster_metrics[assigned_name].false_positives += 1 - probability
    
    total_samples = sum(m.num_samples for m in per_cluster_metrics.values())
    
    macro_average = PeopleGatorNamedFaces__AssignmentReport(
        num_samples=0,
        true_positives=0,
        false_positives=0,
        true_negatives=0,
        false_negatives=0,
        accuracy=0.0,  # to be computed after summing up TP, FP, TN, FN
        precision=0.0,  # to be computed after summing up TP, FP TN, FN
        recall=0.0,  # to be computed after summing up TP, FP TN, FN
        f1_score=0.0,  # to be computed after summing up TP, FP TN, FN
    )
    
    for metrics in per_cluster_metrics.values():
        if metrics.num_samples == 0 or total_samples == 0:
            continue
        name_weight = metrics.num_samples / total_samples
        metrics.accuracy = (metrics.true_positives + metrics.true_negatives) / metrics.num_samples
        metrics.precision = metrics.true_positives / (metrics.true_positives + metrics.false_positives) if metrics.true_positives + metrics.false_positives > 0 else 0.0
        metrics.recall = metrics.true_positives / (metrics.true_positives + metrics.false_negatives) if metrics.true_positives + metrics.false_negatives > 0 else 0.0
        metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall) if metrics.precision + metrics.recall > 0 else 0.0
        
        macro_average.accuracy += name_weight * metrics.accuracy
        macro_average.precision += name_weight * metrics.precision
        macro_average.recall += name_weight * metrics.recall
        macro_average.f1_score += name_weight * metrics.f1_score
    
    
    micro_average = PeopleGatorNamedFaces__AssignmentReport(
        num_samples=sum(m.num_samples for m in per_cluster_metrics.values()),
        true_positives=sum(m.true_positives for m in per_cluster_metrics.values()),
        false_positives=sum(m.false_positives for m in per_cluster_metrics.values()),
        true_negatives=sum(m.true_negatives for m in per_cluster_metrics.values()),
        false_negatives=sum(m.false_negatives for m in per_cluster_metrics.values()),
        accuracy=0.0,  # to be computed after summing up TP, FP, TN, FN
        precision=0.0,  # to be computed after summing up TP, FP TN, FN
        recall=0.0,  # to be computed after summing up TP, FP TN, FN
        f1_score=0.0,  # to be computed after summing up TP, FP TN, FN
    )
    
    micro_average.accuracy = (micro_average.true_positives + micro_average.true_negatives) / micro_average.num_samples if micro_average.num_samples > 0 else 0.0
    micro_average.precision = micro_average.true_positives / (micro_average.true_positives + micro_average.false_positives) if micro_average.true_positives + micro_average.false_positives > 0 else 0.0
    micro_average.recall = micro_average.true_positives / (micro_average.true_positives + micro_average.false_negatives) if micro_average.true_positives + micro_average.false_negatives > 0 else 0.0
    micro_average.f1_score = 2 * micro_average.precision * micro_average.recall / (micro_average.precision + micro_average.recall) if micro_average.precision + micro_average.recall > 0 else 0.0
    
    
    assigned_true_clusters = set(true_cluster for _, true_cluster in cluster_assignments)
    unassigned_clusters = [c for c in clusters_list if c not in assigned_true_clusters]
    assigned_names = set(name for _, name in cluster_assignments)
    unassigned_names = [n for n in names_list if n not in assigned_names]
    return {
        "unassigned_clusters": unassigned_clusters,
        "unassigned_names": unassigned_names,
        "assigned_clusters": [c for c, _ in cluster_assignments],
        "assigned_names": [n for _, n in cluster_assignments],
        "clusters": [per_cluster_metrics[name] for _, name in cluster_assignments],
        "micro_average": micro_average,
        "macro_average": macro_average,
    }



def report_for_predictions(
    ground_truth: list[PeopleGatorNamedFaces__GroundTruth],
    predictions: list[PeopleGatorNamedFaces__ClusterPrediction],
    interagreement_size: int,
    interagreement_strategy: InteragreementStrategy = "probabilistic",
) -> PeopleGatorNamedFaces__ClusterEvaluationReport:
    pairwise_report = compute_pairwise_metrics(
        ground_truth, predictions, interagreement_size, interagreement_strategy
    )
    auto_assignment_report = compute_cluster_auto_assignment_metrics(
        ground_truth, predictions, interagreement_size, interagreement_strategy
    )
    return PeopleGatorNamedFaces__ClusterEvaluationReport.model_validate({
        'assigned_person_names': auto_assignment_report["assigned_names"],
        'assigned_clusters': auto_assignment_report["assigned_clusters"],
        'unassigned_person_names': auto_assignment_report["unassigned_names"],
        'unassigned_clusters': auto_assignment_report["unassigned_clusters"],
        'per_cluster_assignment_reports': auto_assignment_report["clusters"],
        'macro_assignment_report': auto_assignment_report["macro_average"],
        'micro_assignment_report': auto_assignment_report["micro_average"],
        'pairwise_report': pairwise_report
    })


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate clustering predictions against ground truth.")
    parser.add_argument("-g","--ground_truth_path", type=pathlib.Path, required=True, help="Path to ground truth JSONL file.")
    parser.add_argument("-p","--predictions_path", type=pathlib.Path, required=True, help="Path to predictions JSONL file.")
    parser.add_argument("-o","--output_path", type=pathlib.Path, required=True, help="Path to output JSON file.")
    parser.add_argument("-s","--interagreement_size", type=int, required=True, help="Number of annotators for interagreement.")
    parser.add_argument("-y","--interagreement_strategy", choices=["union", "intersection", "majority", "probabilistic"], default="probabilistic", help="Strategy for interannotator agreement.")
    args = parser.parse_args()
    
    ground_truth_path: pathlib.Path = args.ground_truth_path
    predictions_path: pathlib.Path = args.predictions_path
    output_path: pathlib.Path = args.output_path
    interagreement_size: int = args.interagreement_size
    interagreement_strategy: InteragreementStrategy = args.interagreement_strategy
    
    ground_truth: list[PeopleGatorNamedFaces__GroundTruth] = load_jsonl(ground_truth_path, PeopleGatorNamedFaces__GroundTruth)
    predictions: list[PeopleGatorNamedFaces__ClusterPrediction] = load_jsonl(predictions_path, PeopleGatorNamedFaces__ClusterPrediction)
    
    report = report_for_predictions(
        ground_truth, 
        predictions, 
        interagreement_size,
        interagreement_strategy
    )
    
    save_json(output_path, report)

