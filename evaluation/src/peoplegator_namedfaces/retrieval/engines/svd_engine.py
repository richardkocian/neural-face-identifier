import torch
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from peoplegator_namedfaces.retrieval.models import Query, QueriesResult, Dataset
from peoplegator_namedfaces.retrieval.engines.base import BaseRetrievalEngine


class SVDEngine(BaseRetrievalEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.k = kwargs.get("k", 16)

    def __call__(self, queries: list[Query], dataset: Dataset) -> QueriesResult:
        graph = dataset.graph
        num_faces = len(dataset._image_paths)
        queries_result = QueriesResult(queries=[])

        eigen_values, eigen_vectors, keep = laplacian_generalized_eigs(graph, self.k)
        eigen_vectors = l2_normalize_numpy(eigen_vectors)

        for i, query in enumerate(queries):
            sample = dataset.get_sample_by_ground_truth_name(query.query)
            n = np.argsort(np.dot(dataset._graph_names_embeddings, sample["name_embedding"]))[-1] + num_faces

            query_vector = eigen_vectors[n]
            face_scores = np.dot(eigen_vectors[:num_faces], query_vector)

            queries_result.add_query_scores(query, face_scores)

        return queries_result


def laplacian_generalized_eigs(W, k=16, tol=1e-8, maxiter=None, sigma=1e-8):
    W = W.tocsr()
    W = (W + W.T) * 0.5

    d = np.asarray(W.sum(axis=1)).ravel()
    keep = d > 0
    if not np.all(keep):
        W = W[keep][:, keep]
        d = d[keep]

    D = sp.diags(d, format="csc")
    L = (D - W).tocsc()

    # Small positive shift to avoid singular factorization
    evals, evecs = spla.eigsh(
        A=L, k=k, M=D,
        sigma=float(sigma),
        which="LM",
        tol=tol, maxiter=maxiter
    )

    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx], keep


# def laplacian_generalized_eigs(W: sp.spmatrix, k: int = 16, tol: float = 1e-8, maxiter=None):
#     """
#     Solve (D - W) x = λ D x for the k smallest eigenpairs (near 0).
#
#     Returns:
#       evals: (k,) ascending
#       evecs: (n, k) corresponding eigenvectors (for the non-isolated nodes if any were removed)
#       keep: boolean mask of nodes kept (True for nodes with degree>0)
#     """
#     if not sp.isspmatrix(W):
#         raise TypeError("W must be a SciPy sparse matrix")
#     W = W.tocsr()
#
#     # W is symmetric per user; still enforce numerical symmetry
#     W = (W + W.T) * 0.5
#
#     # Degree
#     d = np.asarray(W.sum(axis=1)).ravel()
#
#     # Handle isolated nodes (degree==0): D is singular and the generalized problem is ill-posed for them.
#     keep = d > 0
#     if not np.all(keep):
#         W = W[keep][:, keep]
#         d = d[keep]
#
#     D = sp.diags(d, format="csc")
#     L = (D - W).tocsc()
#
#     # Shift-invert around sigma=0 to get the smallest generalized eigenvalues efficiently.
#     # eigsh will repeatedly solve (L - sigma*D) y = D x; with sigma=0 this is just L y = D x.
#     evals, evecs = spla.eigsh(
#         A=L,
#         k=k,
#         M=D,
#         sigma=0.0,
#         which="LM",   # "LM" around the shift gives eigenvalues closest to sigma
#         tol=tol,
#         maxiter=maxiter,
#     )
#
#     idx = np.argsort(evals)
#     return evals[idx], evecs[:, idx], keep



def l2_normalize(embeddings, dim: int=-1):
    norms = torch.norm(embeddings, p=2, dim=dim, keepdim=True)
    return embeddings / norms


def l2_normalize_numpy(embeddings, axis=-1):
    norms = np.linalg.norm(embeddings, ord=2, axis=axis, keepdims=True)
    return embeddings / norms


def svd_node_embeddings(U, S, k=None, scale="sqrt"):
    """
    Build node embeddings from SVD outputs for a square/symmetric graph matrix.

    Parameters
    ----------
    U : torch.Tensor, shape (n_nodes, r)
    S : torch.Tensor, shape (r,)
    k : int or None
        Number of components to keep. If None, use all returned components.
    scale : {"sqrt", "linear", "none"}
        How to scale singular vectors.

    Returns
    -------
    Z : torch.Tensor, shape (n_nodes, k)
        Node embeddings.
    """
    if k is None:
        k = S.shape[0]
    U_k = U[:, :k]
    S_k = S[:k]

    if scale == "sqrt":
        w = torch.sqrt(torch.clamp(S_k, min=0))
    elif scale == "linear":
        w = S_k
    elif scale == "none":
        w = torch.ones_like(S_k)
    else:
        raise ValueError("scale must be one of {'sqrt', 'linear', 'none'}")

    # Broadcast multiply each column of U_k by corresponding weight
    Z = U_k * w.unsqueeze(0)
    return Z


def scipy_csr_to_torch_sparse(csr_mat):
    # Convert to COO format
    coo = csr_mat.tocoo()

    # Stack row and col indices
    indices = np.vstack((coo.row, coo.col))

    # Convert to torch tensors
    indices = torch.from_numpy(indices).long()
    values = torch.from_numpy(coo.data)

    shape = coo.shape

    # Create sparse tensor
    return torch.sparse_coo_tensor(indices, values, size=shape)
