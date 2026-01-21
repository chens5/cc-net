import numpy as np
import scipy.sparse as sp
import sklearn.datasets as datasets
import torch
import graphlearning as gl

from scipy.sparse import coo_matrix, triu
from scipy.sparse.csgraph import connected_components, laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh
from torch_geometric.data import Data, Dataset
from . import dataset_utils as utils

from tqdm import trange

def two_moons(n_samples, noise=0.15, use_range=False, **kwargs):
    if use_range:
        min_ = n_samples - n_samples//2
        max_ = n_samples
        n_points = np.random.randint(low=min_, high=max_)
    else:
        n_points = n_samples
    return datasets.make_moons(n_points, noise=noise)

def images(dataset, metric='raw', **kwargs):
    """https://jwcalder.github.io/GraphLearning/datasets.html"""
    return gl.datasets.load(dataset, metric)

def single_knn(params):
    generator = globals()[params['base']]
    X, labels = generator(**params)
    W = gl.weightmatrix.knn(X, k=10, kernel='gaussian')
    W.setdiag(0); W.eliminate_zeros()
    data = utils.graphlearning_to_pyg(X, W)
    return [data]

def multiple_knn(params):
    generator = globals()[params['base']]
    try:
        n_graphs = params['n_graphs']
    except KeyError as e: 
        print(f"WARNING: 'n_graphs' not found in dataset parameters, setting n_graphs=1 automatically")
        n_graphs = 1
    dataset = []
    print("Generating dataset....")
    for _ in trange(n_graphs):
        X, labels = generator(**params, use_range=True)
        k_neighbors = np.random.randint(low=5, high=15)
        W = gl.weightmatrix.knn(X, k=k_neighbors, kernel='gaussian')
        W.setdiag(0); W.eliminate_zeros()
        data = utils.graphlearning_to_pyg(X, W)
        dataset.append(data)
    return dataset

def _sample_sbm_sparse(block_sizes, probs, rng: np.random.Generator) -> sp.csr_matrix:
    """
    Sample an undirected unweighted SBM adjacency W (symmetric, {0,1}) as sparse CSR.
    probs: (K,K) with probs[a,b] edge probability between blocks a,b
    """
    K = len(block_sizes)
    n = int(sum(block_sizes))
    offsets = np.cumsum([0] + list(block_sizes))

    rows_all, cols_all = [], []

    def ones_rvs(m: int) -> np.ndarray:
        return np.ones(m, dtype=np.float32)

    for a in range(K):
        sa, oa = block_sizes[a], offsets[a]
        for b in range(a, K):
            sb, ob = block_sizes[b], offsets[b]
            p = float(probs[a, b])
            if p <= 0.0:
                continue

            # fix a random seed
            rs = int(rng.integers(0, 2**31 - 1))

            if a == b:
                M = sp.random(sa, sa, density=p, format="coo", random_state=rs, data_rvs=ones_rvs)
                M = sp.triu(M, k=1).tocoo()
                r = M.row + oa
                c = M.col + oa
            else:
                M = sp.random(sa, sb, density=p, format="coo", random_state=rs, data_rvs=ones_rvs).tocoo()
                r = M.row + oa
                c = M.col + ob

            if r.size == 0:
                continue

            # undirected: add both directions
            rows_all.append(r); cols_all.append(c)
            rows_all.append(c); cols_all.append(r)

    if len(rows_all) == 0:
        W = sp.csr_matrix((n, n), dtype=np.float32)
    else:
        row = np.concatenate(rows_all)
        col = np.concatenate(cols_all)
        data = np.ones_like(row, dtype=np.float32)
        W = sp.coo_matrix((data, (row, col)), shape=(n, n), dtype=np.float32).tocsr()

    W.setdiag(0)
    W.eliminate_zeros()
    W.data[:] = 1.0
    return W

def _patch_to_connected(W: sp.csr_matrix) -> sp.csr_matrix:
    """If disconnected, connect components by chaining. """
    n_comps, labels = connected_components(W, directed=False)
    if n_comps <= 1:
        return W

    reps = np.array([np.flatnonzero(labels == c)[0] for c in range(n_comps)], dtype=np.int64)
    r, c = reps[:-1], reps[1:]
    rows = np.concatenate([r, c])
    cols = np.concatenate([c, r])
    patch = sp.coo_matrix(
        (np.ones(rows.shape[0], dtype=np.float32), (rows, cols)),
        shape=W.shape,
    ).tocsr()

    W2 = (W + patch).tocsr()
    W2.setdiag(0)
    W2.eliminate_zeros()
    W2.data[:] = 1.0
    return W2

def _laplacian_kappa_from_W(
    W: sp.csr_matrix,
    tol: float = 1e-3,
) -> tuple[float, float, float]:
    """
    Returns (kappa, lambda_max, lambda2) for L.
    If disconnected: (inf, nan, 0.0).
    """
    n_comps, _ = connected_components(W, directed=False)
    if n_comps > 1:
        return float("inf"), float("nan"), 0.0

    L = csgraph_laplacian(W, normed=True).astype(np.float64)

    lam_max = float(eigsh(L, k=1, which="LM", return_eigenvectors=False, tol=tol)[0])

    # second-smallest. Increase sigma if slow
    vals = eigsh(L, k=2, sigma=1e-8, which="LM", return_eigenvectors=False, tol=tol)
    vals = np.sort(vals)
    lam2 = float(vals[1])

    kappa = lam_max / max(lam2, 1e-12)
    return float(kappa), float(lam_max), float(lam2)

def _gaussian_sbm_dataset(
    n_graphs: int = 100,
    n_nodes: int = 500,
    n_clusters: int = 5,
    p_in: float = 0.10,
    p_out: float = 0.01,
    feature_dim: int = 16,
    feature_sigma: float = 0.5,
    mean_scale: float = 3.0,
    compute_kappa: bool = True,
    eigsh_tol: float = 1e-3,
    seed: int = 0,
    **kwargs,
):
    """
    Returns a list of pyG data objects.
    Each Data has: x, y, edge_index, edge_attr, and condition number info.
    """
    rng = np.random.default_rng(seed)

    # block sizes equal size
    sizes = [n_nodes // n_clusters] * n_clusters
    sizes[-1] += n_nodes - sum(sizes)

    probs = np.full((n_clusters, n_clusters), p_out, dtype=np.float64)
    np.fill_diagonal(probs, p_in)

    # labels in block order
    y0 = np.concatenate([np.full(s, c, dtype=np.int64) for c, s in enumerate(sizes)])

    dataset = []
    print("Generating dataset....")
    for gi in trange(n_graphs):
        # sample W
        W = _sample_sbm_sparse(sizes, probs, rng)

        # permute nodes
        y = y0.copy()
        perm = rng.permutation(n_nodes)
        W = W[perm, :][:, perm].tocsr()
        y = y[perm]

        # ensure connected
        W = _patch_to_connected(W)

        # gaussian means + noise features
        means = (rng.standard_normal((n_clusters, feature_dim)) * mean_scale).astype(np.float32)
        X = means[y] + (rng.standard_normal((n_nodes, feature_dim)).astype(np.float32) * feature_sigma)

        # convert to pyg
        data = utils.graphlearning_to_pyg(X, W)
        data.y = torch.from_numpy(y).long()
        data.cluster_means = torch.from_numpy(means)  # [K, d]

        if compute_kappa:
            kappa, lam_max, lam2 = _laplacian_kappa_from_W(W, tol=eigsh_tol)
            data.kappa = torch.tensor(kappa, dtype=torch.float32)
            data.lambda_max = torch.tensor(lam_max, dtype=torch.float32)
            data.lambda2 = torch.tensor(lam2, dtype=torch.float32)

        dataset.append(data)

    return dataset

def _gaussian_blob_dataset(n_graphs: int = 100,
                           min_n_nodes: int=200,
                           max_n_nodes: int = 500,
                           n_clusters: int = 5,
                           cluster_std: float = 1.0,
                           k_neighbors: float = 5,
                           feature_dim: int = 2, 
                           **kwargs):

    dataset = []
    print("Generating dataset..... ")
    for _ in trange(n_graphs):
        num_nodes = np.random.randint(low=min_n_nodes, high=max_n_nodes)
        clusters = np.random.randint(low=2, high=n_clusters)
        X, _ = datasets.make_blobs(n_samples=num_nodes, 
                                   n_features=feature_dim, 
                                   centers=clusters,
                                   cluster_std = cluster_std,)
        
        W = gl.weightmatrix.knn(X, k=k_neighbors, kernel='gaussian')
        W.setdiag(0); W.eliminate_zeros()
        W = _patch_to_connected(W)

        data = utils.graphlearning_to_pyg(X, W)
        dataset.append(data)
    print("Done!")
    return dataset

def gaussian_sbm_dataset(params):
    return _gaussian_sbm_dataset(**params)

def gaussian_blob_dataset(params):
    return _gaussian_blob_dataset(**params)