import torch

from peoplegator_namedfaces.clustering.evaluation.src.schemas import (
    PeopleGatorNamedFaces__ClusterEvaluationReport,
    PeopleGatorNamedFaces__ClusterPrediction,
    PeopleGatorNamedFaces__GroundTruth,
    PeopleGatorNamedFaces__UniqueFace,
)

# IO processing and indexing
from peoplegator_namedfaces.clustering.evaluation.src.storage import save_jsonl
from peoplegator_namedfaces.clustering.evaluation.src.utils import (
    load_clustering_files,
)

from torchmetrics.clustering import (
    AdjustedMutualInfoScore,
    AdjustedRandScore,
    CompletenessScore,
    FowlkesMallowsIndex,
    HomogeneityScore,
    MutualInfoScore,
    NormalizedMutualInfoScore,
    RandScore,
    VMeasureScore
)

def evaluate_clustering(
    ground_truths: list[PeopleGatorNamedFaces__GroundTruth],
    predictions: list[PeopleGatorNamedFaces__ClusterPrediction],
    _unique_faces: list[PeopleGatorNamedFaces__UniqueFace],
    bootstrapping_iterations: int | None,
    random_state: int | None,
) -> list[PeopleGatorNamedFaces__ClusterEvaluationReport]:
    """Evaluate clustering predictions against ground truth annotations.

    Args:
        ground_truths: Ground-truth rows.
        predictions: Prediction rows.
        _unique_faces: Unique face rows for building indices.
        save_dir: Directory to save evaluation reports.
        bootstrapping_iterations: Number of bootstrapping iterations to perform. If None, no bootstrapping will be performed.
        random_state: Random state for reproducibility in bootstrapping. If None, bootstrapping will be non-deterministic.
    Returns:
        List of evaluation reports (length 1 without bootstrap).
    """
    targets_dict = {n:i for i, n in enumerate(sorted(set(gt.person_name for gt in ground_truths)))}
    predicted_clusters_dict = {n:i for i, n in enumerate(sorted(set(pred.cluster for pred in predictions)))}
    gt_cluster_for_face = {gt.face: targets_dict[gt.person_name] for gt in ground_truths}
    pred_cluster_for_face = {pred.face: predicted_clusters_dict[pred.cluster] for pred in predictions}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true = torch.tensor([gt_cluster_for_face.get(face.face, -1)
                          for face in _unique_faces]).to(device)
    y_pred = torch.tensor([pred_cluster_for_face.get(face.face, -1)
                          for face in _unique_faces]).to(device)
    
    adjusted_mutual_info_score = AdjustedMutualInfoScore(average_method="arithmetic")
    adjusted_rand_score = AdjustedRandScore()
    completeness_score = CompletenessScore()
    fowlkes_mallows_index = FowlkesMallowsIndex()
    homogeneity_score = HomogeneityScore()
    mutual_info_score = MutualInfoScore()
    normalized_mutual_info_score = NormalizedMutualInfoScore(average_method="arithmetic")
    rand_score = RandScore()
    v_measure_score = VMeasureScore()
    
    reports: list[PeopleGatorNamedFaces__ClusterEvaluationReport] = []

    _random_state = random_state if random_state is not None else torch.seed()
    for i in range(bootstrapping_iterations or 1):
        if bootstrapping_iterations is not None:
            torch.manual_seed(_random_state + i)
            poisson_counts = torch.poisson(torch.ones(len(y_true)))
            poisson_indices = torch.repeat_interleave(torch.arange(len(y_true)), poisson_counts)
            y_true_sample = y_true[poisson_indices]
            y_pred_sample = y_pred[poisson_indices]
        else:
            y_true_sample = y_true
            y_pred_sample = y_pred
        
        # Do not evaluate on background faces (those with -1 label in ground truth)
        y_pred_annotated = y_pred_sample[y_true_sample != -1]
        y_true_annotated = y_true_sample[y_true_sample != -1]
        adjusted_mutual_info_score.update(y_pred_annotated, y_true_annotated)
        adjusted_rand_score.update(y_pred_annotated, y_true_annotated)
        completeness_score.update(y_pred_annotated, y_true_annotated)
        fowlkes_mallows_index.update(y_pred_annotated, y_true_annotated)
        homogeneity_score.update(y_pred_annotated, y_true_annotated)
        mutual_info_score.update(y_pred_annotated, y_true_annotated)
        normalized_mutual_info_score.update(y_pred_annotated, y_true_annotated)
        rand_score.update(y_pred_annotated, y_true_annotated)
        v_measure_score.update(y_pred_annotated, y_true_annotated)
        
        reports.append(PeopleGatorNamedFaces__ClusterEvaluationReport(
            adjusted_mutual_info_score=float(adjusted_mutual_info_score.compute().cpu().item()),
            adjusted_rand_score=float(adjusted_rand_score.compute().cpu().item()),
            completeness_score=float(completeness_score.compute().cpu().item()),
            fowlkes_mallows_index=float(fowlkes_mallows_index.compute().cpu().item()),
            homogeneity_score=float(homogeneity_score.compute().cpu().item()),
            mutual_info_score=float(mutual_info_score.compute().cpu().item()),
            normalized_mutual_info_score=float(normalized_mutual_info_score.compute().cpu().item()),
            rand_score=float(rand_score.compute().cpu().item()),
            v_measure_score=float(v_measure_score.compute().cpu().item())
        ))
    
    return reports


def main():
    """CLI entry point for clustering evaluation."""
    import argparse
    import pathlib
    
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
        "-b", "--bootstrapping_iterations", type=int, default=None,
        help="Number of bootstrapping iterations."
    )
    parser.add_argument(
        "-r", "--random_state", type=int, 
        default=None, help="Random state for reproducibility in bootstrap evaluation."
    )
    args = parser.parse_args()

    ground_truth_path: pathlib.Path = args.ground_truth_path
    unique_faces_path: pathlib.Path = args.unique_faces_path
    predictions_path: pathlib.Path = args.predictions_path
    output_path: pathlib.Path = args.output_path
    bootstrapping_iterations: int | None = args.bootstrapping_iterations
    random_state: int | None = args.random_state
    (
        ground_truth, 
        predictions, 
        unique_faces
    ) = load_clustering_files(
        ground_truth_path, 
        predictions_path, 
        unique_faces_path
    )
    report = evaluate_clustering(
        ground_truth, 
        predictions, 
        unique_faces,
        bootstrapping_iterations,
        random_state
    )
    save_jsonl(output_path, report)

if __name__ == "__main__":
    main()
    
