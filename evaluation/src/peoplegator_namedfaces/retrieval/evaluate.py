import json
import torch
import einops
import pickle
import argparse
import torchmetrics
import time
import numpy as np

from scipy.stats import bootstrap as scipy_bootstrap

from peoplegator_namedfaces.retrieval.models import QueriesResult, GroundTruth, Dataset, QueryType


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", help="Path to the predictions pickle file.")
    parser.add_argument("--ground-truth", help="Path to the ground-truth JSONL file.")
    parser.add_argument("--dataset", help="Path to the dataset JSON file.")
    parser.add_argument("--top-k", type=int, default=None, nargs="+", help="Number of top predictions to consider for evaluation.")
    parser.add_argument("--ignore-index", type=int, default=None, help="Index to ignore during evaluation.")
    parser.add_argument("--output-file", help="Path to the output file (CSV).")
    parser.add_argument("--csv-separator", default=",", help="Separator to use for CSV output (default: ',').")
    parser.add_argument("--bootstrap-iters", type=int, default=None, help="Number of bootstrap iterations for confidence intervals (default: None, no bootstrapping).")

    args = parser.parse_args()
    return args


def load_dataset(path) -> Dataset:
    with open(path, "r") as file:
        config = json.load(file)

    dataset = Dataset(**config)

    return dataset


def load_predictions(path) -> QueriesResult:
    with open(path, "rb") as file:
        result = pickle.load(file)
    return result


def load_ground_truth(path):
    with open(path, "r") as file:
        result = [GroundTruth(**json.loads(line)) for line in file]
    return result


def transform_predictions(predictions: QueriesResult, dataset: Dataset):
    preds = einops.rearrange(predictions.scores, "q f -> (q f)")
    preds = torch.tensor(preds)
    return preds


def transform_ground_truth(ground_truth: list[GroundTruth], dataset: Dataset, ignore_index=None):
    target = []
    indexes = []

    for i, gt in enumerate(ground_truth):
        t = [0] * len(dataset)
        for image_path in gt.faces:
            dataset_index = dataset.image_index(image_path)
            t[dataset_index] = 1

        if ignore_index is not None and gt.query_type == QueryType.IMAGE:
            dataset_index = dataset.image_index(gt.query)
            t[dataset_index] = ignore_index

        target += t
        indexes += [i] * len(dataset)

    target = torch.tensor(target)
    indexes = torch.tensor(indexes)

    return target, indexes


def evaluate(preds, target, indexes, top_k=None, ignore_index=None):
    precision = evaluate_precision(preds, target, indexes, top_k=top_k, ignore_index=ignore_index).item()
    recall = evaluate_recall(preds, target, indexes, top_k=top_k, ignore_index=ignore_index).item()
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fallout = evaluate_fallout(preds, target, indexes, top_k=top_k, ignore_index=ignore_index).item()
    hitrate = evaluate_hitrate(preds, target, indexes, top_k=top_k, ignore_index=ignore_index).item()
    map_score = evaluate_map(preds, target, indexes, top_k=top_k, ignore_index=ignore_index).item()
    mrr = evaluate_mrr(preds, target, indexes, top_k=top_k, ignore_index=ignore_index).item()
    ndcg = evaluate_ndcg(preds, target, indexes, top_k=top_k, ignore_index=ignore_index).item()
    rprecision = evaluate_rprecision(preds, target, indexes, top_k=top_k, ignore_index=ignore_index).item()
    auroc = evaluate_auroc(preds, target, indexes, top_k=top_k, ignore_index=ignore_index).item()

    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fallout": fallout,
        "hitrate": hitrate,
        "map": map_score,
        "mrr": mrr,
        "ndcg": ndcg,
        "rprecision": rprecision,
        "auroc": auroc
    }

    return result


def evaluate_bootstrap(preds, target, indexes, top_k=None, ignore_index=None, bootstrap_iters=1000):
    precision_per_query = evaluate_per_query(evaluate_precision, preds, target, indexes, top_k=top_k, ignore_index=ignore_index)
    recall_per_query = evaluate_per_query(evaluate_recall, preds, target, indexes, top_k=top_k, ignore_index=ignore_index)
    fallout_per_query = evaluate_per_query(evaluate_fallout, preds, target, indexes, top_k=top_k, ignore_index=ignore_index)
    hitrate_per_query = evaluate_per_query(evaluate_hitrate, preds, target, indexes, top_k=top_k, ignore_index=ignore_index)
    map_per_query = evaluate_per_query(evaluate_map, preds, target, indexes, top_k=top_k, ignore_index=ignore_index)
    mrr_per_query = evaluate_per_query(evaluate_mrr, preds, target, indexes, top_k=top_k, ignore_index=ignore_index)
    ndcg_per_query = evaluate_per_query(evaluate_ndcg, preds, target, indexes, top_k=top_k, ignore_index=ignore_index)
    rprecision_per_query = evaluate_per_query(evaluate_rprecision, preds, target, indexes, top_k=top_k, ignore_index=ignore_index)
    auroc_per_query = evaluate_per_query(evaluate_auroc, preds, target, indexes, top_k=top_k, ignore_index=ignore_index)

    precision = bootstrap(precision_per_query, bootstrap_iters=bootstrap_iters)
    recall = bootstrap(recall_per_query, bootstrap_iters=bootstrap_iters)
    fallout = bootstrap(fallout_per_query, bootstrap_iters=bootstrap_iters)
    hitrate = bootstrap(hitrate_per_query, bootstrap_iters=bootstrap_iters)
    map_score = bootstrap(map_per_query, bootstrap_iters=bootstrap_iters)
    mrr = bootstrap(mrr_per_query, bootstrap_iters=bootstrap_iters)
    ndcg = bootstrap(ndcg_per_query, bootstrap_iters=bootstrap_iters)
    rprecision = bootstrap(rprecision_per_query, bootstrap_iters=bootstrap_iters)
    auroc = bootstrap(auroc_per_query, bootstrap_iters=bootstrap_iters)

    result = {
        "precision": precision,
        "recall": recall,
        "f1": "---",
        "fallout": fallout,
        "hitrate": hitrate,
        "map": map_score,
        "mrr": mrr,
        "ndcg": ndcg,
        "rprecision": rprecision,
        "auroc": auroc
    }

    return result


def evaluate_per_query(metric_fn, preds, target, indexes, top_k=None, ignore_index=None):
    unique_queries = torch.unique(indexes)
    per_query_results = []

    for query in unique_queries:
        query_mask = (indexes == query)
        query_preds = preds[query_mask]
        query_target = target[query_mask]
        query_indexes = indexes[query_mask]

        result = metric_fn(query_preds, query_target, query_indexes, top_k=top_k, ignore_index=ignore_index).item()
        per_query_results.append(result)

    return per_query_results


def bootstrap(data, bootstrap_iters=1000):
    result = scipy_bootstrap(
        data=(data,),
        statistic=np.mean,
        n_resamples=bootstrap_iters,
        method="BCa",
        confidence_level=0.95
    )

    _mean = np.mean(result.bootstrap_distribution)
    _median = np.median(result.bootstrap_distribution)
    _ci_low = result.confidence_interval.low
    _ci_high = result.confidence_interval.high
    _min = np.min(result.bootstrap_distribution)
    _max = np.max(result.bootstrap_distribution)
    _q1 = np.percentile(result.bootstrap_distribution, 25)
    _q3 = np.percentile(result.bootstrap_distribution, 75)

    return '|'.join([f"{_mean:.4f}", f"{_median:.4f}", f"{_ci_low:.4f}", f"{_ci_high:.4f}", f"{_min:.4f}", f"{_q1:.4f}", f"{_q3:.4f}", f"{_max:.4f}"])


def evaluate_precision(preds, target, indexes, top_k=None, ignore_index=None):
    metric = torchmetrics.retrieval.RetrievalPrecision(top_k=top_k, ignore_index=ignore_index)
    result = metric(preds, target, indexes)
    return result


def evaluate_recall(preds, target, indexes, top_k=None, ignore_index=None):
    metric = torchmetrics.retrieval.RetrievalRecall(top_k=top_k, ignore_index=ignore_index)
    result = metric(preds, target, indexes)
    return result


def evaluate_fallout(preds, target, indexes, top_k=None, ignore_index=None):
    metric = torchmetrics.retrieval.RetrievalFallOut(top_k=top_k, ignore_index=ignore_index)
    result = metric(preds, target, indexes)
    return result


def evaluate_hitrate(preds, target, indexes, top_k=None, ignore_index=None):
    metric = torchmetrics.retrieval.RetrievalHitRate(top_k=top_k, ignore_index=ignore_index)
    result = metric(preds, target, indexes)
    return result


def evaluate_map(preds, target, indexes, top_k=None, ignore_index=None):
    metric = torchmetrics.retrieval.RetrievalMAP(top_k=top_k, ignore_index=ignore_index)
    result = metric(preds, target, indexes)
    return result


def evaluate_mrr(preds, target, indexes, top_k=None, ignore_index=None):
    metric = torchmetrics.retrieval.RetrievalMRR(top_k=top_k, ignore_index=ignore_index)
    result = metric(preds, target, indexes)
    return result


def evaluate_ndcg(preds, target, indexes, top_k=None, ignore_index=None):
    metric = torchmetrics.retrieval.RetrievalNormalizedDCG(top_k=top_k, ignore_index=ignore_index)
    result = metric(preds, target, indexes)
    return result


def evaluate_rprecision(preds, target, indexes, top_k=None, ignore_index=None):
    metric = torchmetrics.retrieval.RetrievalRPrecision(ignore_index=ignore_index)
    result = metric(preds, target, indexes)
    return result


def evaluate_auroc(preds, target, indexes, top_k=None, ignore_index=None):
    metric = torchmetrics.retrieval.RetrievalAUROC(top_k=top_k, ignore_index=ignore_index)
    result = metric(preds, target, indexes)
    return result


def main():
    args = parse_arguments()

    start_time = time.time()

    predictions = load_predictions(args.predictions)

    predictions_time = time.time()
    print(f"Loaded predictions in {predictions_time - start_time:.2f} seconds.")

    ground_truth = load_ground_truth(args.ground_truth)

    ground_truth_time = time.time()
    print(f"Loaded ground truth in {ground_truth_time - predictions_time:.2f} seconds.")

    dataset = load_dataset(args.dataset)

    dataset_time = time.time()
    print(f"Loaded dataset in {dataset_time - ground_truth_time:.2f} seconds.")

    preds = transform_predictions(predictions, dataset)

    preds_time = time.time()
    print(f"Transformed predictions in {preds_time - dataset_time:.2f} seconds.")

    target, indexes = transform_ground_truth(ground_truth, dataset, ignore_index=args.ignore_index)

    target_time = time.time()
    print(f"Transformed ground truth in {target_time - preds_time:.2f} seconds.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = preds.to(device)
    target = target.to(device)
    indexes = indexes.to(device)

    results = {}
    for top_k in args.top_k:
        top_k = int(top_k)

        if args.bootstrap_iters is not None:
            print(f"Evaluating metrics with bootstrapping for top_k={top_k}...")
            result = evaluate_bootstrap(preds, target, indexes, top_k=top_k, ignore_index=args.ignore_index, bootstrap_iters=args.bootstrap_iters)
        else:
            result = evaluate(preds, target, indexes, top_k=top_k, ignore_index=args.ignore_index)
        results[f"{top_k}"] = result

    result_time = time.time()
    print(f"Evaluated metrics in {result_time - target_time:.2f} seconds.")

    with open(args.output_file, "w") as file:
        key = list(results.keys())[0]
        metrics = list(results[key].keys())
        file.write(f"top_k{args.csv_separator}{args.csv_separator.join(metrics)}\n")

        for top_k in results.keys():
            values = [f"{results[top_k][metric]:.4f}" if type(results[top_k][metric]) == float else results[top_k][metric] for metric in metrics]
            file.write(f"{top_k}{args.csv_separator}{args.csv_separator.join(values)}\n")

    print(f"Saved results to {args.output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
