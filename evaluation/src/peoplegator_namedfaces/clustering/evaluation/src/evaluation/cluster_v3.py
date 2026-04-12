try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

import dataclasses
import os
from line_profiler import profile
import numpy as np
from typing import Literal

import scipy.sparse as sp

from peoplegator_namedfaces.clustering.evaluation.src.schemas import (
    PeopleGatorNamedFaces__ClusterEvaluationReport,
    PeopleGatorNamedFaces__ClusterPrediction,
    PeopleGatorNamedFaces__GroundTruth,
    PeopleGatorNamedFaces__UniqueFace,
)

# IO processing and indexing
from peoplegator_namedfaces.clustering.evaluation.src.utils import (
    load_cluster_prediction_indexers, 
    load_clustering_files, 
    load_ground_truth_indexers,
    build_ground_truth_assignment_and_pairwise_probability_matrix,
    build_cluster_prediction_assignment_matrix
)


InteragreementStrategy = Literal["union", "intersection", "majority", "probabilistic"]

@profile
def align_matrices_to_sampled_faces(
    ground_truth_person_name_probabilities: np.ndarray,
    ground_truth_pair_probabilities: sp.csr_array,
    predicted_cluster_assignments: np.ndarray,
    predicted_pairs: sp.csr_array,
    reordered_ground_truth_person_name_probabilities: np.ndarray,
    reordered_predicted_cluster_assignments: np.ndarray,
    sampled_indices: list[int],
) -> tuple[
    np.ndarray, sp.csr_array,
    np.ndarray, sp.csr_array,
    np.ndarray, np.ndarray,
]:
    """Align prediction and ground-truth matrices to a target face list.

    The output row order follows sampled_faces. If add_missing_faces is True,
    faces that were not sampled but exist in either source matrix are appended.

    Args:
        unique_prediction_matrix: Prediction matrix indexed by
            unique_prediction_faces.
        unique_prediction_faces: Row face ids for unique_prediction_matrix.
        unique_ground_truth_matrix: Ground-truth matrix indexed by
            unique_ground_truth_faces.
        unique_ground_truth_faces: Row face ids for unique_ground_truth_matrix.
        sampled_faces: Target face sequence.
        add_missing_faces: Whether to append unsampled faces from both domains.

    Raises:
        ValueError: If duplicate face ids are passed in unique face lists.

    Returns:
        Aligned prediction and ground-truth matrices indexed by sampled_faces.
    """
    sampled_indices_np = np.asarray(sampled_indices, dtype=np.int64)
    sampled_ground_truth_person_name_probabilities = ground_truth_person_name_probabilities[sampled_indices]
    sampled_ground_truth_pair_probabilities = ground_truth_pair_probabilities[sampled_indices_np][:, sampled_indices_np].tocsr()
    sampled_predicted_cluster_assignments = predicted_cluster_assignments[sampled_indices]
    sampled_predicted_pairs = predicted_pairs[sampled_indices_np][:, sampled_indices_np].tocsr()
    sampled_reordered_ground_truth_person_name_probabilities = reordered_ground_truth_person_name_probabilities[sampled_indices]
    sampled_reordered_predicted_cluster_assignments = reordered_predicted_cluster_assignments[sampled_indices]
    return (
        sampled_ground_truth_person_name_probabilities,
        sampled_ground_truth_pair_probabilities,
        sampled_predicted_cluster_assignments,
        sampled_predicted_pairs,
        sampled_reordered_ground_truth_person_name_probabilities,
        sampled_reordered_predicted_cluster_assignments,
    )
    

@profile
def sample_prediction_assignment_matrix(
    ground_truth_person_name_probabilities: np.ndarray,
    ground_truth_pair_probabilities: sp.csr_array,
    predicted_cluster_assignments: np.ndarray,
    predicted_pairs: sp.csr_array,
    reordered_ground_truth_person_name_probabilities: np.ndarray,
    reordered_predicted_cluster_assignments: np.ndarray,
    sample_size: int,
    random_state: np.random.RandomState,
    replace: bool = True,
) -> tuple[
    np.ndarray, sp.csr_array, 
    np.ndarray, sp.csr_array,
    np.ndarray, np.ndarray,
    list[int]
]:
    """Sample faces and return aligned prediction/ground-truth submatrices.

    Args:
        ground_truth_person_name_probabilities: Ground-truth person name probabilities.
        ground_truth_pair_probabilities: Ground-truth pair probabilities.
        predicted_cluster_assignments: Predicted cluster assignments.
        predicted_pairs: Predicted pair probabilities.
        reordered_ground_truth_person_name_probabilities: Reordered ground-truth person name probabilities.
        reordered_predicted_cluster_assignments: Reordered predicted cluster assignments.
        sample_size: Number of sampled faces.
        random_state: NumPy random state used for sampling.
        replace: Whether to sample with replacement.

    Returns:
        Aligned prediction and ground-truth submatrices indexed by sampled faces, and the sampled face indices.
    """
    if len(predicted_cluster_assignments) != len(ground_truth_person_name_probabilities):
        raise ValueError("Prediction and ground truth matrices must have the same number of faces (rows) for sampling")
    num_unique_faces = len(ground_truth_person_name_probabilities)
    sampled_indices = [int(idx) for idx in random_state.choice(
        num_unique_faces,
        size=sample_size, 
        replace=replace
    )]
    (
        sampled_ground_truth_person_name_probabilities,
        sampled_ground_truth_pair_probabilities,
        sampled_predicted_cluster_assignments,
        sampled_predicted_pairs,
        sampled_reordered_ground_truth_person_name_probabilities,
        sampled_reordered_predicted_cluster_assignments,
    ) = align_matrices_to_sampled_faces(
        ground_truth_person_name_probabilities,
        ground_truth_pair_probabilities,
        predicted_cluster_assignments,
        predicted_pairs,
        reordered_ground_truth_person_name_probabilities,
        reordered_predicted_cluster_assignments,
        sampled_indices,
    )
    return (
        sampled_ground_truth_person_name_probabilities,
        sampled_ground_truth_pair_probabilities,
        sampled_predicted_cluster_assignments,
        sampled_predicted_pairs,
        sampled_reordered_ground_truth_person_name_probabilities,
        sampled_reordered_predicted_cluster_assignments,
        sampled_indices, 
    )

@profile
def build_contingency_matrices(
    predicted_cluster_assignments: np.ndarray,
    ground_truth_person_name_assignments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build intersection, union, and IoU matrices between clusters and names.

    Args:
        predicted_cluster_assignments: Prediction matrix (faces x clusters).
        ground_truth_person_name_assignments: Ground-truth matrix (faces x names).
        item_indices: Face ids corresponding to the rows of both matrices. To filter out same-item-to-same-item pairs from the pairwise evaluation, these should be unique face identifiers.
    Returns:
        A tuple of intersection sizes, union sizes, and IoU matrices.
    """
    # op([Cl, Fa, 1] x [1, Fa, Na]) -> [Cl, Fa, Na] -> sum over clusters (axis=1) -> [Cl, Na]
    # build sparse non_zero_triplets of cluster-face-name combinations to efficiently compute intersection and union sizes without explicit broadcasting of large dense matrices. This is crucial for scalability when the number of faces is large.
    pred_nz = predicted_cluster_assignments > 0   # [Fa, Cl]
    gt_nz = ground_truth_person_name_assignments > 0   # [Fa, Na]
    triplets = []
    c_indices = np.arange(pred_nz.shape[1], dtype=np.int64)  # cluster indices
    n_indices = np.arange(gt_nz.shape[1], dtype=np.int64)    # name indices
    for f in range(pred_nz.shape[0]):
        c = c_indices[pred_nz[f]]
        n = n_indices[gt_nz[f]]
        if c.size and n.size:
            indices = np.empty((len(c) * len(n), 3), dtype=np.int64)
            indices[:, 0] = np.repeat(c, n.size)
            indices[:, 2] = np.tile(n, c.size)
            indices[:, 1] = f
            triplets.append(indices)
    common_indices = np.concatenate(
        triplets, axis=0
    ) if triplets else np.empty((0, 3), dtype=np.int64)
    
    # calculate intersection and union sizes using the common indices and the original fuzzy set values, then compute IoU. This avoids materializing large intermediate tensors that would arise from naive broadcasting of the original matrices.
    common_min = np.minimum(
        predicted_cluster_assignments[common_indices[:, 1], common_indices[:, 0]],
        ground_truth_person_name_assignments[common_indices[:, 1], common_indices[:, 2]]
    )
    # For union, we can use the fact that union = sum - intersection for binary sets, but since we have fuzzy values, we need to compute the max instead of sum. We can still do this efficiently using the common indices without broadcasting the full matrices.
    common_max = np.maximum(
        predicted_cluster_assignments[common_indices[:, 1], common_indices[:, 0]],
        ground_truth_person_name_assignments[common_indices[:, 1], common_indices[:, 2]]
    )
    intersection_sizes_matrix = np.zeros((
        predicted_cluster_assignments.shape[1],
        ground_truth_person_name_assignments.shape[1]
    ), dtype=float)
    np.add.at(intersection_sizes_matrix, (common_indices[:, 0], common_indices[:, 2]), common_min)
    union_sizes_matrix = np.zeros((
        predicted_cluster_assignments.shape[1],
        ground_truth_person_name_assignments.shape[1]
    ), dtype=float)
    np.add.at(union_sizes_matrix, (common_indices[:, 0], common_indices[:, 2]), common_max)
    intersection_over_union_matrix = np.divide(
        intersection_sizes_matrix,
        union_sizes_matrix,
        out=np.zeros_like(intersection_sizes_matrix, dtype=float),
        where=union_sizes_matrix > 0
    )
    return intersection_sizes_matrix, union_sizes_matrix, intersection_over_union_matrix

@profile
def compute_adjusted_rand_index(
    intersection_sizes_matrix: np.ndarray,
) -> float:
    """Compute adjusted Rand index from an intersection contingency matrix.

    Args:
        intersection_sizes_matrix: Intersection contingency matrix.

    Returns:
        Adjusted Rand index.
    """
    sum_comb_c = np.sum(intersection_sizes_matrix.sum(axis=1) *
                        (intersection_sizes_matrix.sum(axis=1) - 1)) / 2
    sum_comb_k = np.sum(intersection_sizes_matrix.sum(axis=0) *
                        (intersection_sizes_matrix.sum(axis=0) - 1)) / 2
    sum_comb = np.sum(intersection_sizes_matrix * (intersection_sizes_matrix - 1)) / 2
    n = intersection_sizes_matrix.sum()
    if n <= 1:
        return 0.0
    expected_index = sum_comb_c * sum_comb_k / (n * (n - 1) / 2)
    max_index = (sum_comb_c + sum_comb_k) / 2
    adjusted_rand_index = (sum_comb - expected_index) / (max_index -
                                                         expected_index) if max_index != expected_index else 0.0
    return float(adjusted_rand_index)

@profile
def compute_binary_classification_metrics(
    true_positives: float,
    false_positives: float,
    true_negatives: float,
    false_negatives: float,
) -> tuple[float, float, float, float, float, float, float]:
    """Compute scalar binary metrics from confusion counts."""
    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives) if (true_positives + false_positives + true_negatives + false_negatives) > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    jaccard = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0.0
    fowlkes_mallows = ((precision * recall)**0.5) if (precision > 0 and recall > 0) else 0.0
    rand_index = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives) if (true_positives + false_positives + true_negatives + false_negatives) > 0 else 0.0
    return accuracy, precision, recall, f1_score, jaccard, fowlkes_mallows, rand_index

@profile
def compute_binary_classification_metrics_np(
    true_positives: np.ndarray,
    false_positives: np.ndarray,
    true_negatives: np.ndarray,
    false_negatives: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute vectorized binary metrics from confusion-count arrays."""
    all_counts = true_positives + false_positives + true_negatives + false_negatives
    accuracy = np.divide(
        true_positives + true_negatives,
         all_counts,
        out=np.zeros_like(true_positives, dtype=float),
        where=all_counts > 0
    )
    
    positives = true_positives + false_positives
    precision = np.divide(
        true_positives,
        positives,
        out=np.zeros_like(true_positives, dtype=float),
        where=positives > 0
    )
    predictions = true_positives + false_negatives
    recall = np.divide(
        true_positives,
        predictions,
        out=np.zeros_like(true_positives, dtype=float),
        where=predictions > 0
    )
    precision_recall_sum = precision + recall
    f1_score = np.divide(
        2 * precision * recall,
        precision_recall_sum,
        out=np.zeros_like(true_positives, dtype=float),
        where=precision_recall_sum > 0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

@profile
def compute_pairwise_metrics(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
    intersection_sizes_matrix: np.ndarray,
    ignore_background_negatives: bool = False,
) -> dict[str, float]:
    """Compute pairwise metrics from aligned pairwise probability matrices.

    Args:
        predictions: Prediction pairwise matrix (flattened to 1d).
        ground_truths: Ground-truth pairwise matrix (flattened to 1d).
        intersection_sizes_matrix: Intersection matrix used for ARI.
        ignore_background_negatives: If True, evaluate only pairs where either
            prediction or ground truth is positive.

    Returns:
        Dictionary containing confusion components and derived metrics.
    """
    if predictions.shape != ground_truths.shape:
        raise ValueError("Prediction and ground truth pairwise matrices must have the same shape")
    prediction_within_zero_one = (
        (predictions >= 0 -np.finfo(float).eps) & 
        (predictions <= 1 + np.finfo(float).eps)
    ).all()
    if not prediction_within_zero_one:
        raise ValueError("Prediction pairwise should have values between 0 and 1 inclusive")
    
    mask = (
        ((ground_truths > 0) | (predictions > 0))
        if ignore_background_negatives 
        else np.ones_like(ground_truths, dtype=bool)
    )
    scores_masked = predictions[mask]
    label_weights_masked = ground_truths[mask]
    true_positives = float(np.sum(scores_masked * label_weights_masked))
    false_positives = float(np.sum(scores_masked * (1 - label_weights_masked)))
    true_negatives = float(np.sum((1 - scores_masked) * (1 - label_weights_masked)))
    false_negatives = float(np.sum((1 - scores_masked) * label_weights_masked))
    (
        accuracy, 
        precision, 
        recall, 
        f1_score, 
        jaccard, 
        fowlkes_mallows, 
        rand_index
    ) = compute_binary_classification_metrics(
        true_positives, false_positives, true_negatives, false_negatives
    )
    adjusted_rand_index = compute_adjusted_rand_index(intersection_sizes_matrix)
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "jaccard": jaccard,
        "fowlkes_mallows": fowlkes_mallows,
        "rand_index": rand_index,
        "adjusted_rand_index": adjusted_rand_index,
    }

@profile
def automatically_assign_clusters_to_names(
    intersection_over_unions_matrix: np.ndarray,
    ground_truth_name_assignment_probabilities: np.ndarray,
    predicted_cluster_assignments: np.ndarray,
    unique_person_names: list[str],
    unique_clusters: list[int],
) -> tuple[
    np.ndarray, np.ndarray,
    list[str], list[int],
    list[str], list[int],
]:
    """Compute optimal cluster-to-name assignment based on IoU matrix.

    Args:
        intersection_over_unions_matrix: Cluster-to-name IoU matrix.
        ground_truth_person_name_assignments: Ground-truth person-name assignments.
        predicted_cluster_assignments: Predicted cluster assignments.
        unique_person_names: List of unique person names.
        unique_clusters: List of unique cluster labels.
    Returns:
        A tuple of reordered ground-truth assignments, reordered predicted
        assignments, reordered unique person names, and reordered unique
        clusters, all aligned according to the optimal assignment.
    """
    from scipy.optimize import linear_sum_assignment
    cost_matrix = -intersection_over_unions_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    reordered_ground_truth_person_name_probabilities = ground_truth_name_assignment_probabilities[
        :, col_ind]
    reordered_predicted_cluster_assignments = predicted_cluster_assignments[:, row_ind]
    assigned_person_names = [unique_person_names[i] for i in col_ind]
    assigned_clusters = [unique_clusters[i] for i in row_ind]
    missing_person_names = sorted(set(unique_person_names) - set(assigned_person_names))
    missing_clusters = sorted(set(unique_clusters) - set(assigned_clusters))
    return (
        reordered_ground_truth_person_name_probabilities,
        reordered_predicted_cluster_assignments,
        assigned_person_names,
        assigned_clusters,
        missing_person_names,
        missing_clusters
    )

@profile
def compute_automatic_assignment_metrics(
    reordered_ground_truth_person_name_probabilities: np.ndarray,
    reordered_predicted_cluster_assignments: np.ndarray,
) -> dict:
    """Compute optimal cluster-to-name assignment and assignment metrics.

    Args:
        reordered_ground_truth_person_name_probabilities: Reordered ground-truth person-name probabilities.
        reordered_predicted_cluster_assignments: Reordered predicted cluster assignments.

    Returns:
        Assignment outputs with mapped labels and macro/micro summaries.
    """
    true_positives = (reordered_ground_truth_person_name_probabilities * reordered_predicted_cluster_assignments).sum(axis=0)
    false_positives = (reordered_predicted_cluster_assignments * (1 - reordered_ground_truth_person_name_probabilities)).sum(axis=0)
    false_negatives = ((1 - reordered_predicted_cluster_assignments) * reordered_ground_truth_person_name_probabilities).sum(axis=0)
    true_negatives = ((1 - reordered_predicted_cluster_assignments) * (1 - reordered_ground_truth_person_name_probabilities)).sum(axis=0)
    num_samples = true_positives + false_positives + false_negatives + true_negatives
    multiclass_metrics = compute_binary_classification_metrics_np(true_positives, false_positives, true_negatives, false_negatives)
    results_dict = [
        {
            "num_samples": float(ns),
            "true_positives": float(tp),
            "false_positives": float(fp),
            "true_negatives": float(tn),
            "false_negatives": float(fn),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        } for (
            ns, tp, fp, 
            tn, fn, acc, prec, rec, f1
        ) in zip(
            num_samples,
            true_positives, false_positives, true_negatives, false_negatives, multiclass_metrics["accuracy"], 
            multiclass_metrics["precision"], 
            multiclass_metrics["recall"], multiclass_metrics["f1_score"]
        )
    ]
    
    micro_num_samples = float(num_samples.sum())
    micro_true_positives = float(true_positives.sum())
    micro_false_positives = float(false_positives.sum())
    micro_false_negatives = float(false_negatives.sum())
    micro_true_negatives = float(true_negatives.sum())
    micro_metrics = compute_binary_classification_metrics(
        micro_true_positives, micro_false_positives, micro_true_negatives, micro_false_negatives
    )
    micro_results_dict = {
        "num_samples": micro_num_samples,
        "true_positives": micro_true_positives,
        "false_positives": micro_false_positives,
        "true_negatives": micro_true_negatives,
        "false_negatives": micro_false_negatives,
        "accuracy": float(micro_metrics[0]),
        "precision": float(micro_metrics[1]),
        "recall": float(micro_metrics[2]),
        "f1_score": float(micro_metrics[3])
    }
    
    macro_num_samples = float(num_samples.mean())
    macro_true_positives = float(true_positives.mean())
    macro_false_positives = float(false_positives.mean())
    macro_false_negatives = float(false_negatives.mean())
    macro_true_negatives = float(true_negatives.mean())
    macro_metrics = compute_binary_classification_metrics(
        macro_true_positives, macro_false_positives, macro_true_negatives, macro_false_negatives
    )
    macro_results_dict = {
        "num_samples": macro_num_samples,
        "true_positives": macro_true_positives,
        "false_positives": macro_false_positives,
        "true_negatives": macro_true_negatives,
        "false_negatives": macro_false_negatives,
        "accuracy": float(macro_metrics[0]),
        "precision": float(macro_metrics[1]),
        "recall": float(macro_metrics[2]),
        "f1_score": float(macro_metrics[3])
    }
    
    return {
        "assignment_metrics": results_dict,
        "micro_assignment_report": micro_results_dict,
        "macro_assignment_report": macro_results_dict
    }


@profile
def evaluate_bootstrap_run(
    ground_truth_person_name_probabilities: np.ndarray,
    ground_truth_pair_probabilities: sp.csr_array,
    predicted_cluster_assignments: np.ndarray,
    predicted_pairs: sp.csr_array,
    reordered_ground_truth_person_name_probabilities: np.ndarray,
    reordered_predicted_cluster_assignments: np.ndarray,
    indices: list[int],
    ignore_background_negatives: bool = False,
) -> PeopleGatorNamedFaces__ClusterEvaluationReport:
    """Evaluate one aligned run and produce a cluster evaluation report."""
    (
        intersection_sizes_matrix,
        _,
        _
    ) = build_contingency_matrices(
        predicted_cluster_assignments,
        ground_truth_person_name_probabilities
    )
    _sampled_indices = np.array(indices)
    not_isame_indices_1, not_isame_indices_2 = np.where(
        _sampled_indices[None, :] != _sampled_indices[:, None]
    )
    pairwise_predictions = np.asarray(predicted_pairs[not_isame_indices_1, not_isame_indices_2]).ravel()
    pairwise_ground_truths = np.asarray(ground_truth_pair_probabilities[not_isame_indices_1, not_isame_indices_2]).ravel()
    pairwise_metrics = compute_pairwise_metrics(
        pairwise_predictions,
        pairwise_ground_truths,
        intersection_sizes_matrix,
        ignore_background_negatives=ignore_background_negatives
    )
    assignment_metrics = compute_automatic_assignment_metrics(
        reordered_ground_truth_person_name_probabilities,
        reordered_predicted_cluster_assignments
    )
    evaluation_report = PeopleGatorNamedFaces__ClusterEvaluationReport.model_validate({
        "per_cluster_assignment_reports": assignment_metrics["assignment_metrics"],
        "macro_assignment_report": assignment_metrics["macro_assignment_report"],
        "micro_assignment_report": assignment_metrics["micro_assignment_report"],
        "pairwise_report": pairwise_metrics
    })
    return evaluation_report

@dataclasses.dataclass
class EvaluateBootstrapRunWrapperInputs:
    ground_truth_person_name_probabilities: np.ndarray
    ground_truth_pair_probabilities: sp.csr_array
    predicted_cluster_assignments: np.ndarray
    predicted_pairs: sp.csr_array
    reordered_ground_truth_person_name_probabilities: np.ndarray
    reordered_predicted_cluster_assignments: np.ndarray
    
    ignore_background_negatives: bool
    sample_size_for_bootstrap: int
    seed: int | None
    replace: bool
    save_dir: str


@profile
def _evaluate_bootstrap_run_wrapper(
    inputs: EvaluateBootstrapRunWrapperInputs
) -> PeopleGatorNamedFaces__ClusterEvaluationReport | None:
    _seed = inputs.seed if inputs.seed is not None else np.random.randint(0, 1_000_000_000)
    random_state = np.random.RandomState(_seed)
    (
        sampled_ground_truth_person_name_probabilities,
        sampled_ground_truth_pair_probabilities,
        sampled_predicted_cluster_assignments,
        sampled_predicted_pairs,
        sampled_reordered_ground_truth_person_name_probabilities,
        sampled_reordered_predicted_cluster_assignments,
        sampled_indices,
    ) = sample_prediction_assignment_matrix(
        inputs.ground_truth_person_name_probabilities,
        inputs.ground_truth_pair_probabilities,
        inputs.predicted_cluster_assignments,
        inputs.predicted_pairs,
        inputs.reordered_ground_truth_person_name_probabilities,
        inputs.reordered_predicted_cluster_assignments,
        sample_size=inputs.sample_size_for_bootstrap,
        random_state=random_state,
        replace=inputs.replace
    )
    evaluation_report = evaluate_bootstrap_run(
        sampled_ground_truth_person_name_probabilities,
        sampled_ground_truth_pair_probabilities,
        sampled_predicted_cluster_assignments,
        sampled_predicted_pairs,
        sampled_reordered_ground_truth_person_name_probabilities,
        sampled_reordered_predicted_cluster_assignments,
        sampled_indices,
        ignore_background_negatives=inputs.ignore_background_negatives
    )
    with open(f"{inputs.save_dir}/evaluation_report_{_seed}.json", "w") as f:
        f.write(evaluation_report.model_dump_json())
    return None

@profile
def evaluate_clustering(
    ground_truths: list[PeopleGatorNamedFaces__GroundTruth],
    predictions: list[PeopleGatorNamedFaces__ClusterPrediction],
    _unique_faces: list[PeopleGatorNamedFaces__UniqueFace],
    interagreement_size: int,
    interagreement_strategy: InteragreementStrategy,
    save_dir: str,
    sample_size_for_bootstrap: int | None,
    bootstrapping_iterations: int,
    random_state: int | None,
    ignore_background_negatives: bool = False,
    enforce_max_interagreement_size: bool = False,
    enforce_correct_interagreement_size: bool = False,
    enforce_no_duplicite_annotations: bool = False,
    enforce_annotator_single_face_single_name_assignment: bool = False,
    enforce_single_face_single_cluster: bool = False,
    single_threaded_bootstrap: bool = False,
):
    """Evaluate clustering predictions against ground truth annotations.

    Args:
        ground_truths: Ground-truth rows.
        predictions: Prediction rows.
        interagreement_size: Fixed annotator denominator/threshold.
        interagreement_strategy: Vote aggregation strategy.
        ignore_background_negatives: Pairwise metric mode.
        enforce_max_interagreement_size: Enforce max agreement count.
        enforce_correct_interagreement_size: Enforce expected annotator count.
        enforce_no_duplicite_annotations: Enforce no duplicate annotations.
        enforce_annotator_single_face_single_name_assignment: Enforce one name per
            annotator per face.
        enforce_single_face_single_cluster: Enforce one cluster per face.
        sample_size_for_bootstrap: If provided, run bootstrap evaluation.
        bootstrapping_iterations: Number of bootstrap repetitions.
        random_state: Seed for bootstrap reproducibility.

    Returns:
        List of evaluation reports (length 1 without bootstrap).
    """
    (
        ground_truth_name_assignment_probabilities, 
        ground_truth_pair_probabilities, 
        ground_truth_unique_faces,
        unique_person_names
    ) = build_ground_truth_assignment_and_pairwise_probability_matrix(
        ground_truths,
        _unique_faces,
        interagreement_size=interagreement_size,
        interagreement_strategy=interagreement_strategy,
        enforce_max_interagreement_size=enforce_max_interagreement_size,
        enforce_correct_interagreement_size=enforce_correct_interagreement_size,
        enforce_no_duplicite_annotations=enforce_no_duplicite_annotations,
        enforce_annotator_single_face_single_name_assignment=enforce_annotator_single_face_single_name_assignment,
    )
    (
        predicted_cluster_assignments, predicted_pairs, predicted_unique_faces, unique_clusters

    ) = build_cluster_prediction_assignment_matrix(
        predictions,
        _unique_faces,
        enforce_single_face_single_cluster=enforce_single_face_single_cluster
    )
    if ground_truth_unique_faces != predicted_unique_faces:
        raise ValueError("Unique faces derived from ground truth and predictions do not match. Please ensure that the unique faces input is consistent with the faces referenced in both ground truth and prediction data.")
    unique_faces = ground_truth_unique_faces  # or predicted_unique_faces, they should be the same
    (
        *_,
        intersection_over_unions_matrix
    ) = build_contingency_matrices(
        predicted_cluster_assignments,
        ground_truth_name_assignment_probabilities
    )
    (
        reordered_ground_truth_person_name_assignments,
        reordered_predicted_cluster_assignments,
        assigned_person_names,
        assigned_clusters,
        missing_person_names,
        missing_clusters
    ) = automatically_assign_clusters_to_names(
        intersection_over_unions_matrix,
        ground_truth_name_assignment_probabilities,
        predicted_cluster_assignments,
        unique_person_names,
        unique_clusters
    )
    # create output directory
    os.makedirs(save_dir, exist_ok=True) if save_dir is not None else None
    if sample_size_for_bootstrap is not None:
        seeds = [
            random_state + i if random_state is not None else None
            for i in range(bootstrapping_iterations)
        ]
        _iterator = (
            EvaluateBootstrapRunWrapperInputs(
                ground_truth_person_name_probabilities=ground_truth_name_assignment_probabilities,
                ground_truth_pair_probabilities=ground_truth_pair_probabilities,
                predicted_cluster_assignments=predicted_cluster_assignments,
                predicted_pairs=predicted_pairs,
                reordered_ground_truth_person_name_probabilities=reordered_ground_truth_person_name_assignments,
                reordered_predicted_cluster_assignments=reordered_predicted_cluster_assignments,
                ignore_background_negatives=ignore_background_negatives,
                sample_size_for_bootstrap=sample_size_for_bootstrap,
                seed=seed,
                replace=True,
                save_dir=save_dir
            )
            for seed in seeds
        )
        if single_threaded_bootstrap:
            _single_threaded_iterator = map( _evaluate_bootstrap_run_wrapper, _iterator)
            for _ in tqdm(_single_threaded_iterator, total=bootstrapping_iterations):
                ...
        else:
            import multiprocessing
            with multiprocessing.Pool() as pool:
                _mutlithredding_iterator = pool.imap(
                    _evaluate_bootstrap_run_wrapper, _iterator
                )
                for _ in tqdm(_mutlithredding_iterator, total=bootstrapping_iterations):
                    ...
    else:
        evaluation_report = evaluate_bootstrap_run(
            ground_truth_person_name_probabilities=ground_truth_name_assignment_probabilities,
            ground_truth_pair_probabilities=ground_truth_pair_probabilities,
            predicted_cluster_assignments=predicted_cluster_assignments,
            predicted_pairs=predicted_pairs,
            reordered_ground_truth_person_name_probabilities=reordered_ground_truth_person_name_assignments,
            reordered_predicted_cluster_assignments=reordered_predicted_cluster_assignments,
            indices=np.arange(len(unique_faces)).tolist(),
            ignore_background_negatives=ignore_background_negatives,
        )
        with open(f"{save_dir}/evaluation_report.json", "w") as f:
            f.write(evaluation_report.model_dump_json())

def main():
    """CLI entry point for clustering evaluation."""
    import argparse
    import pathlib
    from people_gator_named_faces_evaluation.storage import load_jsonl, save_jsonl
    
    parser = argparse.ArgumentParser(
        description="Evaluate clustering predictions against ground truth.")
    parser.add_argument(
        "-g", "--ground_truth_path", type=pathlib.Path,
        required=True, help="Path to ground truth JSONL file."
    )
    parser.add_argument(
        "-a", "--unique_faces_path", type=pathlib.Path,
        required=True, help="Path to unique faces JSONL file."
    )
    parser.add_argument(
        "-p", "--predictions_path", type=pathlib.Path,
        required=True, help="Path to predictions JSONL file."
    )
    parser.add_argument(
        "-o", "--output_path", type=pathlib.Path,
        required=True, help="Path to output JSONL file."
    )
    parser.add_argument(
        "-s", "--interagreement_size", type=int,
        required=True, help="Number of annotators for interagreement."
    )
    parser.add_argument(
        "-y", "--interagreement_strategy", choices=[
            "union", "intersection", "majority", "probabilistic"
        ], 
        default="probabilistic", 
        help="Strategy for interannotator agreement."
    )
    parser.add_argument(
        "-b", "--sample_size_for_bootstrap", 
        type=int, default=None, 
        help="Sample size for bootstrap evaluation. If not provided, no bootstrapping will be performed."
    )
    parser.add_argument(
        "-i", "--bootstrapping_iterations", type=int, default=1000,
        help="Number of bootstrapping iterations."
    )
    parser.add_argument(
        "-r", "--random_state", type=int, 
        default=None, help="Random state for reproducibility in bootstrap evaluation."
    )
    parser.add_argument("-f", "--flags", nargs="*", default=[], help="Additional flags for evaluation. Supported flags: enforce_max_interagreement_size, enforce_correct_interagreement_size, enforce_single_annotator_single_face_single_names, enforce_single_face_single_cluster, add_missing_faces, ignore_background_negatives", choices=[
        "enforce_max_interagreement_size",
        "enforce_correct_interagreement_size",
        "enforce_single_annotator_single_face_single_names",
        "enforce_single_face_single_cluster",
        "ignore_background_negatives",
        "single_threaded_bootstrap"
    ])
    args = parser.parse_args()

    ground_truth_path: pathlib.Path = args.ground_truth_path
    unique_faces_path: pathlib.Path = args.unique_faces_path
    predictions_path: pathlib.Path = args.predictions_path
    output_path: pathlib.Path = args.output_path
    interagreement_size: int = args.interagreement_size
    sample_size_for_bootstrap: int | None = args.sample_size_for_bootstrap
    interagreement_strategy: InteragreementStrategy = args.interagreement_strategy
    random_state: int | None = args.random_state
    bootstrapping_iterations: int = args.bootstrapping_iterations
    flags: dict[str, bool] = {f: True for f in args.flags}
    (
        ground_truth, 
        predictions, 
        unique_faces
    ) = load_clustering_files(
        ground_truth_path, 
        predictions_path, 
        unique_faces_path
    )
    evaluate_clustering(
        ground_truth, 
        predictions, 
        unique_faces,
        interagreement_size, 
        interagreement_strategy,
        save_dir=str(output_path),
        sample_size_for_bootstrap=sample_size_for_bootstrap,
        bootstrapping_iterations=bootstrapping_iterations,
        random_state=random_state,
        **flags
    )

if __name__ == "__main__":
    import cProfile
    report = cProfile.run("main()", sort="cumtime", filename="cluster_evaluation_profile.prof")
    
