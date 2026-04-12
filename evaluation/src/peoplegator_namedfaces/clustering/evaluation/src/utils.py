import pathlib
import scipy.sparse as sp
from peoplegator_namedfaces.clustering.evaluation.src.schemas import (
    PeopleGatorNamedFaces__ClusterPrediction,
    PeopleGatorNamedFaces__GroundTruth,
    PeopleGatorNamedFaces__UniqueFace,
    InteragreementStrategy
)
from peoplegator_namedfaces.clustering.evaluation.src.storage import load_jsonl
import numpy as np

def load_ground_truth_indexers(
    ground_truths: list[PeopleGatorNamedFaces__GroundTruth],
    _unique_faces: list[PeopleGatorNamedFaces__UniqueFace],
) -> tuple[
    list[str], list[str], list[str],
    dict[str, int], dict[str, int], dict[str, int],
    list[int], list[int], list[int],
]:
    """Prepare indexers and index lists for ground-truth loading.

    Returns:
        (unique_faces, unique_person_names, unique_annotators,
         face_to_index, person_name_to_index, annotator_to_index,
         face_indices, person_name_indices, annotator_indices)
    """
    unique_faces = sorted(set(uf.face for uf in _unique_faces))
    unique_person_names = sorted(set(gt.person_name for gt in ground_truths))
    unique_annotators = sorted(set(gt.annotator for gt in ground_truths))

    face_to_index: dict[str, int] = {face: idx for idx, face in enumerate(unique_faces)}
    person_name_to_index: dict[str, int] = {name: idx for idx, name in enumerate(unique_person_names)}
    annotator_to_index: dict[str, int] = {a: idx for idx, a in enumerate(unique_annotators)}

    face_indices: list[int] = [face_to_index[gt.face] for gt in ground_truths]
    person_name_indices: list[int] = [person_name_to_index[gt.person_name] for gt in ground_truths]
    annotator_indices: list[int] = [annotator_to_index[gt.annotator] for gt in ground_truths]

    return (
        unique_faces,
        unique_person_names,
        unique_annotators,
        face_to_index,
        person_name_to_index,
        annotator_to_index,
        face_indices,
        person_name_indices,
        annotator_indices,
    )


def load_cluster_prediction_indexers(
    predictions: list[PeopleGatorNamedFaces__ClusterPrediction],
    _unique_faces: list[PeopleGatorNamedFaces__UniqueFace],
) -> tuple[
    list[str], list[int],
    dict[str, int], dict[int, int],
    list[int], list[int],
]:
    """Prepare indexers and index lists for cluster prediction loading.

    Returns:
        (unique_faces, unique_clusters, face_to_index, cluster_to_index,
         face_indices, cluster_indices)
    """
    unique_faces = sorted(set(uf.face for uf in _unique_faces))
    unique_clusters = sorted(set(pred.cluster for pred in predictions))

    face_to_index: dict[str, int] = {face: idx for idx, face in enumerate(unique_faces)}
    cluster_to_index: dict[int, int] = {c: idx for idx, c in enumerate(unique_clusters)}

    face_indices: list[int] = [face_to_index[pred.face] for pred in predictions]
    cluster_indices: list[int] = [cluster_to_index[pred.cluster] for pred in predictions]

    return (
        unique_faces,
        unique_clusters,
        face_to_index,
        cluster_to_index,
        face_indices,
        cluster_indices,
    )

def load_clustering_files(
    ground_truths_path: pathlib.Path,
    predictions_path: pathlib.Path,
    unique_faces_path: pathlib.Path,
) -> tuple[
    list[PeopleGatorNamedFaces__GroundTruth],
    list[PeopleGatorNamedFaces__ClusterPrediction],
    list[PeopleGatorNamedFaces__UniqueFace],
]:
    if not ground_truths_path.is_file():
        raise FileNotFoundError(f"Ground truths file not found: {ground_truths_path}")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    if not unique_faces_path.is_file():
        raise FileNotFoundError(f"Unique faces file not found: {unique_faces_path}")
    ground_truths = load_jsonl(ground_truths_path, PeopleGatorNamedFaces__GroundTruth)
    predictions = load_jsonl(predictions_path, PeopleGatorNamedFaces__ClusterPrediction)
    unique_faces = load_jsonl(unique_faces_path, PeopleGatorNamedFaces__UniqueFace)
    return ground_truths, predictions, unique_faces


def build_ground_truth_assignment_and_pairwise_probability_matrix(
    ground_truths: list[PeopleGatorNamedFaces__GroundTruth],
    _unique_faces: list[PeopleGatorNamedFaces__UniqueFace],
    interagreement_size: int,
    interagreement_strategy: InteragreementStrategy,
    enforce_max_interagreement_size: bool,
    enforce_correct_interagreement_size: bool,
    enforce_no_duplicite_annotations: bool,
    enforce_annotator_single_face_single_name_assignment: bool,
) -> tuple[np.ndarray, sp.csr_array, list[str], list[str]]:
    """Build a face-to-name assignment matrix from annotator labels.

    Args:
        ground_truths: Ground-truth rows containing face, person_name,
            and annotator.
        interagreement_size: Fixed annotator denominator/threshold used by
            agreement strategies.
        interagreement_strategy: Vote aggregation strategy.
        enforce_max_interagreement_size: If True, raise when any face/name
            agreement count exceeds interagreement_size.
        enforce_correct_interagreement_size: If True, require the number of
            unique annotators to equal interagreement_size.
        enforce_no_duplicite_annotations: If True, forbid multiple annotations
            for the same face, person name, and annotator combination.
        enforce_annotator_single_face_single_name_assignment: If True, forbid one
            annotator assigning multiple names to a single face.

    Raises:
        ValueError: If any configured consistency constraint is violated.

    Returns:
        A tuple of assignment matrix (faces x names), pairwise counts matrix (faces x faces),
        sorted unique faces, and sorted unique person names.
    """
    # IO processing and indexing
    # Always sort unique faces to ensure consistency
    (
        unique_faces, unique_person_names, unique_annotators,
        _, _, _,
        face_indices, person_name_indices, annotator_indices,
    ) = load_ground_truth_indexers(ground_truths, _unique_faces)
    if enforce_correct_interagreement_size and len(unique_annotators) != interagreement_size:
        raise ValueError(
            f"Number of unique annotators ({len(unique_annotators)}) does not equal the specified interagreement size ({interagreement_size})")

    # Build annotation tensor for cluster assignments
    # [Fa, Na, An] tensor
    annotator_face_to_name_assignments = sp.coo_array(
        (np.ones_like(face_indices, dtype=int),
         (face_indices, person_name_indices, annotator_indices)),
        shape=(len(unique_faces), len(
            unique_person_names), len(unique_annotators)),
        dtype=int
    )
    annotator_face_to_name_assignments.sum_duplicates()
    # enforce constraints
    if enforce_no_duplicite_annotations and np.any(annotator_face_to_name_assignments.data > 1):
        raise ValueError(
            "Multiple annotations for the same face, person name, and annotator combination found"
        )
    annotator_face_to_name_assignments = annotator_face_to_name_assignments.minimum(
        1)
    # [Fa, Na, An].transpose(2, 0, 1) -> [An, Fa, Na]
    # [Fa, Na, An].transpose(2, 1, 0) -> [An, Na, Fa]
    # [An, Fa, Na] x [An, Na, Fa] -> [An, Fa, Fa] -> number of times pair is labeled by annotator as same (max 1)
    adjencency_matrices: sp.coo_array = (
        sp.permute_dims(annotator_face_to_name_assignments,
                        (2, 0, 1), copy=True)
        @
        sp.permute_dims(annotator_face_to_name_assignments,
                        (2, 1, 0), copy=True)
    )
    if enforce_annotator_single_face_single_name_assignment and np.any(adjencency_matrices.data > 1):
        raise ValueError(
            "An annotator assigned same face to multiple person names")
    adjencency_matrices = adjencency_matrices.minimum(1)
    # [Fa, Na, An].sum(axis=-1) -> [Fa, Na]
    face_name_assignments_count = sp.coo_array(
        (annotator_face_to_name_assignments.data,
         (annotator_face_to_name_assignments.coords[0], annotator_face_to_name_assignments.coords[1])),
        shape=(len(unique_faces), len(unique_person_names)),
        dtype=int
    )
    face_name_assignments_count.sum_duplicates()
    # [An, Fa, Fa].sum(axis=0) -> [Fa, Fa]
    adjencency_counts = sp.coo_array(
        (adjencency_matrices.data, (adjencency_matrices.row, adjencency_matrices.col)),
        shape=(len(unique_faces), len(unique_faces)),
        dtype=int
    )
    adjencency_counts.sum_duplicates()
    face_name_assignments_count = face_name_assignments_count.tocsr()
    adjencency_counts = adjencency_counts.tocsr()

    # enforce constraints
    if enforce_max_interagreement_size and np.any(face_name_assignments_count.data > interagreement_size):
        raise ValueError(
            "Number of annotators for at least one face exceeds the maximum interagreement size")
    if enforce_max_interagreement_size and np.any(adjencency_counts.data > interagreement_size):
        raise ValueError(
            "Number of annotators for at least one face pair exceeds the maximum interagreement size")

    face_name_assignments_count = face_name_assignments_count.minimum(
        interagreement_size)
    adjencency_counts = adjencency_counts.minimum(interagreement_size)

    match interagreement_strategy:
        case "union":
            ground_truth_name_assignment_probabilities: sp.csr_array = face_name_assignments_count.astype(
                float)
            ground_truth_name_assignment_probabilities.data = np.ones_like(
                ground_truth_name_assignment_probabilities.data, dtype=float)
            ground_truth_pair_probabilities: sp.csr_array = adjencency_counts.astype(
                float)
            ground_truth_pair_probabilities.data = np.ones_like(
                ground_truth_pair_probabilities.data, dtype=float)
        case "intersection":
            ground_truth_name_assignment_probabilities: sp.csr_array = face_name_assignments_count.astype(
                float)
            ground_truth_name_assignment_probabilities.data = (
                ground_truth_name_assignment_probabilities.data >= interagreement_size).astype(float)
            ground_truth_name_assignment_probabilities.eliminate_zeros()
            ground_truth_pair_probabilities: sp.csr_array = adjencency_counts.astype(
                float)
            ground_truth_pair_probabilities.data = (
                ground_truth_pair_probabilities.data >= interagreement_size).astype(float)
            ground_truth_pair_probabilities.eliminate_zeros()
        case "majority":
            ground_truth_name_assignment_probabilities: sp.csr_array = face_name_assignments_count.astype(
                float)
            ground_truth_name_assignment_probabilities.data = (
                ground_truth_name_assignment_probabilities.data > interagreement_size / 2.0).astype(float)
            ground_truth_name_assignment_probabilities.eliminate_zeros()
            ground_truth_pair_probabilities: sp.csr_array = adjencency_counts.astype(
                float)
            ground_truth_pair_probabilities.data = (
                ground_truth_pair_probabilities.data > interagreement_size / 2.0).astype(float)
            ground_truth_pair_probabilities.eliminate_zeros()
        case "probabilistic":
            ground_truth_name_assignment_probabilities: sp.csr_array = face_name_assignments_count.astype(
                float)
            ground_truth_name_assignment_probabilities.data = ground_truth_name_assignment_probabilities.data / interagreement_size
            ground_truth_pair_probabilities: sp.csr_array = adjencency_counts.astype(
                float)
            ground_truth_pair_probabilities.data = ground_truth_pair_probabilities.data / \
                interagreement_size
        case _:
            raise ValueError(
                f"Invalid interagreement strategy: {interagreement_strategy}")
    return ground_truth_name_assignment_probabilities.toarray(), ground_truth_pair_probabilities.tocsr(), unique_faces, unique_person_names

def build_cluster_prediction_assignment_matrix(
    predictions: list[PeopleGatorNamedFaces__ClusterPrediction],
    _unique_faces: list[PeopleGatorNamedFaces__UniqueFace],
    enforce_single_face_single_cluster: bool,
) -> tuple[np.ndarray, sp.csr_array, list[str], list[int]]:
    """Build a face-to-cluster assignment matrix from predictions.

    Args:
        predictions: Predicted (face, cluster) assignments.
        _unique_faces: List of unique faces to index the output matrix rows.
        enforce_single_face_single_cluster: If True, raise when a face is
            assigned to more than one cluster.

    Raises:
        ValueError: If enforce_single_face_single_cluster is True and a face
            appears in multiple clusters.

    Returns:
        A tuple of assignment matrix (faces x clusters), pairwise face
        adjacency matrix (faces x faces), sorted unique faces, and sorted
        unique cluster ids.
    """
    (
        unique_faces,
        unique_clusters,
        _, _,
        face_indices,
        cluster_indices
    ) = (
        load_cluster_prediction_indexers(predictions, _unique_faces)
    )

    # Build assignment matrix and pairwise counts
    predicted_cluster_assignments = np.zeros(
        (len(unique_faces), len(unique_clusters)), dtype=np.int64
    )
    np.add.at(predicted_cluster_assignments,
              (face_indices, cluster_indices), 1)
    # enforce constraints that would lead to ambiguous cluster assignments and pairwise relationships
    if enforce_single_face_single_cluster and (predicted_cluster_assignments.sum(axis=1) > 1).any():
        raise ValueError(
            "At least one face is assigned to multiple clusters, which violates the single-face-single-cluster constraint")
    # if not raise, sanitize counts to be binary indicators
    predicted_cluster_assignments = np.minimum(
        predicted_cluster_assignments, 1)

    predicted_cluster_assignments_sparse = sp.csr_array(
        predicted_cluster_assignments)
    predicted_pairs = predicted_cluster_assignments_sparse @ predicted_cluster_assignments_sparse.T
    # enforce constraints that would lead to ambiguous pairwise relationships
    if enforce_single_face_single_cluster and predicted_pairs.max() > 1:
        raise ValueError(
            "At least one face pair is assigned to the same cluster more than once, which violates the single-face-single-cluster constraint")
    # if not raise, sanitize counts to be binary indicators
    predicted_pairs = predicted_pairs.minimum(1).tocsr()
    return predicted_cluster_assignments, predicted_pairs, unique_faces, unique_clusters
