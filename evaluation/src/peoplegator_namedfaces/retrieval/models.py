import numpy as np

from enum import Enum
from scipy import sparse
from pydantic import BaseModel


class QueryType(Enum):
    TEXT = 'text'
    IMAGE = 'image'


class Query(BaseModel):
    query: str
    query_type: QueryType


class QueriesResult:
    def __init__(self, queries: list[Query], scores: None|np.ndarray = None):
        self.queries = queries
        self.scores = scores

    def add_query_scores(self, query: Query, scores: np.ndarray):
        self.queries.append(query)
        if self.scores is None:
            self.scores = scores
        else:
            self.scores = np.vstack([self.scores, scores])


class GroundTruth(Query):
    faces: list[str]

class Dataset:
    def __init__(self,
                 image_paths: str|None = None,
                 image_embeddings: str|None = None,
                 graph_names: str|None = None,
                 graph_names_embeddings: str|None = None,
                 ground_truth_names: str|None = None,
                 ground_truth_names_embeddings: str|None = None,
                 image_similarity_matrix: str|None = None,
                 graph_name_similarity_matrix: str|None = None,
                 image_name_matrix: str|None = None,
                 graph_hyper_params: dict|None = None):
        self._image_paths = self.load_file(image_paths) if image_paths is not None else None
        self._image_embeddings = self.load_embeddings(image_embeddings) if image_embeddings is not None else None
        self._graph_names = self.load_file(graph_names) if graph_names is not None else None
        self._graph_names_embeddings = self.load_embeddings(graph_names_embeddings) if graph_names_embeddings is not None else None
        self._ground_truth_names = self.load_file(ground_truth_names) if ground_truth_names is not None else None
        self._ground_truth_names_embeddings = self.load_embeddings(ground_truth_names_embeddings) if ground_truth_names_embeddings is not None else None
        self._image_similarity_matrix = self.load_similarity_matrix(image_similarity_matrix) if image_similarity_matrix is not None else None
        self._graph_name_similarity_matrix = self.load_similarity_matrix(graph_name_similarity_matrix) if graph_name_similarity_matrix is not None else None
        self._image_name_matrix = self.load_similarity_matrix(image_name_matrix, is_sparse=True) if image_name_matrix is not None else None

        self.graph_hyper_params = graph_hyper_params if graph_hyper_params is not None else {}

        if self._image_similarity_matrix is not None and self._image_name_matrix is not None:
            image_similarity_size = self._image_similarity_matrix.shape[0]
            if self._image_name_matrix.shape[1] == image_similarity_size:
                self._image_name_matrix = self._image_name_matrix.T

        self._image_index_mapping = {}
        for i, image_path in enumerate(self._image_paths):
            self._image_index_mapping[image_path] = i

        self._graph_name_index_mapping = {}
        if self._graph_names is not None:
            for i, name in enumerate(self._graph_names):
                self._graph_name_index_mapping[name] = i

        self._ground_truth_name_index_mapping = {}
        if self._ground_truth_names is not None:
            for i, name in enumerate(self._ground_truth_names):
                self._ground_truth_name_index_mapping[name] = i

    @property
    def image_embeddings(self):
        return self._image_embeddings

    @property
    def graph(self):
        if self._image_similarity_matrix is not None and self._graph_name_similarity_matrix is not None and self._image_name_matrix is not None:
            num_images = len(self._image_paths)
            num_names = len(self._graph_names)
            graph = np.zeros((num_images + num_names, num_images + num_names))

            graph[:num_images, :num_images] = self.adjust_matrix(self._image_similarity_matrix, self.graph_hyper_params.get('image'))
            graph[num_images:, num_images:] = self.adjust_matrix(self._graph_name_similarity_matrix, self.graph_hyper_params.get('name'))

            image_name_matrix = self.adjust_matrix(self._image_name_matrix, self.graph_hyper_params.get('image_name'))
            graph[:num_images, num_images:] = image_name_matrix
            graph[num_images:, :num_images] = image_name_matrix.T

            graph = sparse.csr_matrix(graph)

            return graph
        else:
            return None

    @staticmethod
    def load_file(path):
        with open(path, "r") as file:
            return [line.strip() for line in file]

    @staticmethod
    def load_similarity_matrix(path, is_sparse=False):
        if is_sparse:
            result = sparse.load_npz(path).toarray()
        else:
            result = np.load(path)
        return result

    @staticmethod
    def load_embeddings(path):
        embeddings = np.load(path)
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return normalized_embeddings

    @staticmethod
    def adjust_matrix(matrix, hyper_params=None):
        adjusted_matrix = matrix

        if hyper_params is not None:
            if "k" in hyper_params:
                k = hyper_params["k"]
                only_nonzero = hyper_params.get("only_nonzero", False)
                if only_nonzero:
                    adjusted_matrix = np.where(matrix != 0, np.exp(k * matrix), 0.0)
                    print(f"Applied exponential adjustment with k={k} to only non-zero values")
                else:
                    adjusted_matrix = np.exp(k * matrix)
                    print(f"Applied exponential adjustment with k={k}")

            if "threshold" in hyper_params:
                threshold = hyper_params["threshold"]
                adjusted_matrix[adjusted_matrix < threshold] = 0.0
                print(f"Applied thresholding with threshold={threshold}")

            if "amount" in hyper_params:
                relative_amount = hyper_params["amount"]
                target_amount = int(adjusted_matrix.shape[0] * adjusted_matrix.shape[1] * relative_amount)
                if target_amount < adjusted_matrix.size:
                    flat_indices = np.argpartition(adjusted_matrix.flatten(), -target_amount)[-target_amount:]
                    row_indices, col_indices = np.unravel_index(flat_indices, adjusted_matrix.shape)
                    mask = np.zeros_like(adjusted_matrix, dtype=bool)
                    mask[row_indices, col_indices] = True
                    adjusted_matrix[~mask] = 0.0

                print(f"Applied sparsification to keep top {relative_amount} values (absolute amount: {target_amount})")

        return adjusted_matrix

    def __len__(self):
        if self._image_paths is not None:
            return len(self._image_paths)
        elif self._image_embeddings is not None:
            return len(self._image_embeddings)
        else:
            return 0

    def __getitem__(self, idx):
        image_path = None
        image_embedding = None
        name = None
        name_embedding = None

        if idx < len(self._image_paths):
            image_path = self._image_paths[idx] if self._image_paths is not None else None
            image_embedding = self._image_embeddings[idx] if self._image_embeddings is not None else None
        elif len(self._image_paths) < idx < len(self._image_paths) + len(self._graph_names):
            name_idx = idx - len(self._image_paths)
            name = self._graph_names[name_idx] if self._graph_names is not None else None
            name_embedding = self._graph_names_embeddings[name_idx] if self._graph_names_embeddings is not None else None
        else:
            raise IndexError("Index out of range")

        return {
            "image_path": image_path,
            "image_embedding": image_embedding,
            "name": name,
            "name_embedding": name_embedding
        }

    def image_index(self, image_path):
        return self._image_index_mapping[image_path]

    def graph_name_index(self, name, use_offset=True):
        index = self._graph_name_index_mapping[name]
        if use_offset:
            index += len(self._image_paths) if self._image_paths is not None else 0

        return index

    def ground_truth_name_index(self, name):
        return self._ground_truth_name_index_mapping[name]

    def get_sample_by_image_path(self, image_path):
        idx = self.image_index(image_path)
        return self[idx]

    def get_sample_by_graph_name(self, name, use_offset=True):
        idx = self.graph_name_index(name, use_offset=use_offset)
        return self[idx]

    def get_sample_by_ground_truth_name(self, name):
        idx = self.ground_truth_name_index(name)
        ground_truth_name = self._ground_truth_names[idx] if self._ground_truth_names is not None else None
        ground_truth_name_embedding = self._ground_truth_names_embeddings[idx] if self._ground_truth_names_embeddings is not None else None
        return {
            "name": ground_truth_name,
            "name_embedding": ground_truth_name_embedding
        }

