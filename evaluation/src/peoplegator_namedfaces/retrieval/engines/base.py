from abc import ABC, abstractmethod

from peoplegator_namedfaces.retrieval.models import Dataset, Query, QueriesResult


class BaseRetrievalEngine(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, queries: list[Query], dataset: Dataset) -> QueriesResult:
        '''
        :param query: A list of Query objects to retrieve results for.
        :param dataset: The dataset to retrieve results from.
        :return: A QueriesResult object containing results.
        '''
        pass
