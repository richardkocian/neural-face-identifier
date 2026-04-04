import numpy as np

from sknetwork.ranking import PageRank

from peoplegator_namedfaces.retrieval.models import Query, QueriesResult, Dataset, QueryType
from peoplegator_namedfaces.retrieval.engines.base import BaseRetrievalEngine


class PageRankEngine(BaseRetrievalEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.damping_factor = kwargs.get("damping_factor", 0.85)
        self.iters = kwargs.get("iters", 10)

        self.pagerank = PageRank(damping_factor=self.damping_factor, n_iter=self.iters)
        print(f"Initialized PageRankEngine with damping_factor={self.damping_factor} and iters={self.iters}")

    def __call__(self, queries: list[Query], dataset: Dataset) -> QueriesResult:
        graph = dataset.graph
        num_faces = len(dataset._image_paths)
        queries_result = QueriesResult(queries=[])

        for i, query in enumerate(queries):
            if query.query_type == QueryType.TEXT:
                sample = dataset.get_sample_by_ground_truth_name(query.query)
                n = np.argsort(np.dot(dataset._graph_names_embeddings, sample["name_embedding"]))[-1] + num_faces
            else:
                n = dataset.image_index(query.query)

            scores = self.pagerank.fit_predict(graph, weights={n: 1.0})
            face_scores = scores[:num_faces]
            queries_result.add_query_scores(query, face_scores)

        return queries_result
