import numpy as np

from peoplegator_namedfaces.retrieval.models import Query, QueriesResult, Dataset
from peoplegator_namedfaces.retrieval.engines.base import BaseRetrievalEngine


class RandomRetrievalEngine(BaseRetrievalEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, queries: list[Query], dataset: Dataset) -> QueriesResult:
        '''
        This is a dummy retrieval engine that returns random results. It should be replaced with a real retrieval engine.
        '''

        scores = np.random.uniform(-1, 1, (len(queries), len(dataset)))
        return QueriesResult(queries=queries, scores=scores)
