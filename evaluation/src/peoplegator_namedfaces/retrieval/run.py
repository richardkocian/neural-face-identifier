import os
import json
import pickle
import argparse

from peoplegator_namedfaces.retrieval.models import Dataset, Query
from peoplegator_namedfaces.retrieval.engines.base import BaseRetrievalEngine


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to the dataset.", required=True)
    parser.add_argument("--queries", help="Path to JSONL file with queries.", required=True)
    parser.add_argument("--engine", help="Path to JSON file containing the retrieval engine configuration.", required=True)
    parser.add_argument("--output", help="Path to the output pickle file where results will be saved.", required=True)
    parser.add_argument("--hyper-params", help="Path to JSON file containing hyperparameters for the retrieval.", required=False, default=None)

    args = parser.parse_args()
    return args


def load_dataset(path, hyper_params=None) -> Dataset:
    with open(path, "r") as file:
        config = json.load(file)

    dataset = Dataset(**config, graph_hyper_params=hyper_params)

    return dataset


def load_engine(path, engine_hyper_params: dict) -> BaseRetrievalEngine:
    with open(path, "r") as file:
        config = json.load(file)

    config = {**config, **engine_hyper_params}

    if config["engine"] == "random":
        from peoplegator_namedfaces.retrieval.engines.random_engine import RandomRetrievalEngine
        engine = RandomRetrievalEngine(**config)
    elif config["engine"] == "image_embedding":
        from peoplegator_namedfaces.retrieval.engines.image_embedding_engine import ImageEmbeddingEngine
        engine = ImageEmbeddingEngine(**config)
    elif config["engine"] == "pagerank":
        from peoplegator_namedfaces.retrieval.engines.page_rank_engine import PageRankEngine
        engine = PageRankEngine(**config)
    elif config["engine"] == "svd":
        from peoplegator_namedfaces.retrieval.engines.svd_engine import SVDEngine
        engine = SVDEngine(**config)
    else:
        raise ValueError(f"Unknown engine type: {config['engine']}")

    return engine


def load_query(path) -> list[Query]:
    result = []
    with open(path, "r") as file:
        for line in file:
            result.append(Query(**json.loads(line)))

    return result


def save_results(results, path):
    with open(path, "wb") as file:
        pickle.dump(results, file)


def main():
    args = parse_arguments()

    hyper_params = None
    if args.hyper_params is not None:
        hyper_params = json.loads(args.hyper_params)
        print(f"Loaded hyperparameters: ({type(hyper_params)}) {hyper_params}")

    dataset_hyper_params = {}
    if hyper_params is not None and "dataset" in hyper_params:
        dataset_hyper_params = hyper_params["dataset"]
        print(f"Using dataset hyperparameters: {dataset_hyper_params}")

    dataset = load_dataset(args.dataset, dataset_hyper_params)
    print(f"Loaded dataset with {len(dataset)} samples.")

    engine_hyper_params = {}
    if hyper_params is not None and "engine" in hyper_params:
        engine_hyper_params = hyper_params["engine"]
        print(f"Using engine hyperparameters: {engine_hyper_params}")

    engine = load_engine(args.engine, engine_hyper_params)
    print(f"Loaded retrieval engine: {engine.__class__.__name__}")

    queries = load_query(args.queries)
    print(f"Loaded queries: {len(queries)} queries.")

    result = engine(queries, dataset)
    print(f"Retrieved results for {len(result.queries)} queries.")

    save_results(result, args.output)
    print(f"Saved results to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
