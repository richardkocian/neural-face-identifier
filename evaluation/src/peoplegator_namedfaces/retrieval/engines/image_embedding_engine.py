import numpy as np

from peoplegator_namedfaces.retrieval.models import Query, QueriesResult, Dataset
from peoplegator_namedfaces.retrieval.engines.base import BaseRetrievalEngine


class ImageEmbeddingEngine(BaseRetrievalEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.threshold = kwargs.get('threshold', None)
        self.binarize = kwargs.get('binarize', False)

        if self.threshold is None and self.binarize:
            raise ValueError("Binarization requires a threshold to be set.")

    def __call__(self, queries: list[Query], dataset: Dataset) -> QueriesResult:
        '''
        This retrieval engine computes the cosine similarity between the query image embeddings and the dataset image embeddings, and returns the scores for each query and each dataset image. The scores are thresholded to remove low similarity scores.
        '''
        query_embeddings = np.array([dataset.get_sample_by_image_path(q.query)["image_embedding"] for q in queries])
        dataset_embeddings = dataset.image_embeddings

        query_embeddings_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        dataset_embeddings_norm = np.linalg.norm(dataset_embeddings, axis=1, keepdims=True)

        query_embeddings_normalized = query_embeddings / query_embeddings_norm
        dataset_embeddings_normalized = dataset_embeddings / dataset_embeddings_norm

        scores = np.dot(query_embeddings_normalized, dataset_embeddings_normalized.T)

        if self.threshold is not None:
            scores[scores < self.threshold] = 0.0

            if self.binarize:
                scores = (scores > 0).astype(float)

        return QueriesResult(queries=queries, scores=scores)
