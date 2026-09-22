"""Expander convergence experiment dataset parameters.

    branching=[2, 2, 2], level_scales=[13.0, 6.0, 2.0],
    points_per_leaf=10, dim=2, degree=15,
    min_lambda2_normalized=0.4, max_kappa_normalized=4.0,
    max_rho_nontrivial=0.6, max_attempts=100
    train=(n_graphs=256, seed=1001)
    validation=(n_graphs=64, seed=2001)
    test=(n_graphs=100, seed=3001)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components, laplacian
from torch_geometric.data import Data


def generate_recursive_hierarchy(
    branching: Sequence[int],
    points_per_leaf: int = 50,
    level_scales: Sequence[float] | None = None,
    dim: int = 2,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Sample the hierarchical Gaussian features used in the pilot.

    ``level_scales[level]`` controls the displacement of child centers.  At a
    leaf, the last level scale is deliberately reused for sampling points.  That
    convention matches the completed experiment.
    """
    branching = tuple(int(value) for value in branching)
    if not branching or any(value < 1 for value in branching):
        raise ValueError("branching must contain positive integers.")
    if points_per_leaf < 1:
        raise ValueError("points_per_leaf must be positive.")
    if dim < 1:
        raise ValueError("dim must be positive.")
    if level_scales is None:
        scales = tuple(10.0 / (2**level) for level in range(len(branching)))
    else:
        scales = tuple(float(value) for value in level_scales)
    if len(scales) != len(branching):
        raise ValueError("level_scales must have one value per hierarchy level.")
    if any(not np.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("level_scales must contain finite positive values.")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    features: list[np.ndarray] = []
    labels: list[tuple[int, ...]] = []

    def recurse(center: np.ndarray, level: int, path: tuple[int, ...]) -> None:
        if level == len(branching):
            points = rng.normal(
                loc=center,
                scale=scales[-1],
                size=(points_per_leaf, dim),
            )
            features.append(points)
            labels.extend([path] * points_per_leaf)
            return

        child_centers = rng.normal(
            loc=center,
            scale=scales[level],
            size=(branching[level], dim),
        )
        for child_id, child_center in enumerate(child_centers):
            recurse(child_center, level + 1, (*path, child_id))

    recurse(np.zeros(dim), 0, ())
    return np.vstack(features), labels


def graph_spectral_diagnostics(
    adjacency: sp.spmatrix,
) -> dict[str, float | np.ndarray]:
    """Return the exact spectral diagnostics used to certify pilot graphs."""
    adjacency = sp.csr_matrix(adjacency, dtype=np.float64)
    if adjacency.shape[0] != adjacency.shape[1] or adjacency.shape[0] < 2:
        raise ValueError("adjacency must be a square matrix with at least two nodes.")
    if adjacency.nnz == 0 or not np.isfinite(adjacency.data).all():
        raise ValueError("adjacency must contain finite edges.")
    if np.any(adjacency.data < 0.0):
        raise ValueError("adjacency must not contain negative edge weights.")
    difference = adjacency - adjacency.T
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-10:
        raise ValueError("adjacency must be symmetric.")
    n_components, _ = connected_components(adjacency, directed=False)
    if n_components != 1:
        raise ValueError("Spectral diagnostics require a connected graph.")

    laplacian_combinatorial = laplacian(adjacency, normed=False).toarray()
    laplacian_normalized = laplacian(adjacency, normed=True).toarray()
    eigenvalues_combinatorial = np.linalg.eigvalsh(laplacian_combinatorial)
    eigenvalues_normalized = np.linalg.eigvalsh(laplacian_normalized)
    positive_combinatorial = eigenvalues_combinatorial[
        eigenvalues_combinatorial > 1e-8
    ]
    positive_normalized = eigenvalues_normalized[eigenvalues_normalized > 1e-8]
    if positive_combinatorial.size == 0 or positive_normalized.size == 0:
        raise ValueError("Graph Laplacian has no positive eigenvalues.")

    normalized_adjacency_eigenvalues = np.linalg.eigvalsh(
        np.eye(adjacency.shape[0]) - laplacian_normalized
    )
    adjacency_abs_descending = np.sort(
        np.abs(normalized_adjacency_eigenvalues)
    )[::-1]

    return {
        "lambda2_combinatorial": float(positive_combinatorial[0]),
        "lambda_max_combinatorial": float(positive_combinatorial[-1]),
        "kappa_combinatorial": float(
            positive_combinatorial[-1] / positive_combinatorial[0]
        ),
        "lambda2_normalized": float(positive_normalized[0]),
        "lambda_max_normalized": float(positive_normalized[-1]),
        "kappa_normalized": float(
            positive_normalized[-1] / positive_normalized[0]
        ),
        "rho_nontrivial": float(adjacency_abs_descending[1]),
        "normalized_laplacian_eigenvalues": eigenvalues_normalized.astype(
            np.float32
        ),
        "normalized_adjacency_abs_eigenvalues": (
            adjacency_abs_descending.astype(np.float32)
        ),
    }


def _to_pyg(
    features: np.ndarray,
    adjacency: sp.spmatrix,
) -> Data:
    """Store one ordered copy of each undirected edge, matching the solver API."""
    upper = sp.triu(adjacency, k=1).tocoo()
    edge_index = torch.stack(
        (
            torch.from_numpy(upper.row).long(),
            torch.from_numpy(upper.col).long(),
        )
    )
    return Data(
        x=torch.as_tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.as_tensor(upper.data, dtype=torch.float32),
    )


def _generate_hierarchical_expanders(
    n_graphs: int,
    branching: Sequence[int],
    normalize: bool = True,
    points_per_leaf: int = 20,
    level_scales: Sequence[float] | None = None,
    dim: int = 2,
    seed: int = 42,
    degree: int = 15,
    min_lambda2_normalized: float = 0.4,
    max_kappa_normalized: float = 4.0,
    max_rho_nontrivial: float = 0.6,
    max_attempts: int = 100,
    **_: Any,
) -> list[Data]:
    """Generate deterministic random-regular expanders with Gaussian features.

    A candidate is accepted only when it is connected and satisfies all three
    normalized spectral constraints.  Each graph's topology has its own seed
    derived from ``(seed, graph_id, attempt, 0xCC)``; therefore graph retries do
    not consume or otherwise perturb the feature RNG.
    """
    if n_graphs < 1:
        raise ValueError("n_graphs must be positive.")
    branching_values = tuple(int(value) for value in branching)
    if not branching_values or any(value < 1 for value in branching_values):
        raise ValueError("branching must contain positive integers.")
    n_nodes = int(points_per_leaf * np.prod(branching_values))
    if degree <= 0 or degree >= n_nodes:
        raise ValueError(f"degree must be in [1, {n_nodes - 1}], got {degree}.")
    if (degree * n_nodes) % 2:
        raise ValueError("degree * number of nodes must be even.")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    dataset: list[Data] = []
    feature_rng = np.random.default_rng(seed)
    for graph_id in range(n_graphs):
        features, labels = generate_recursive_hierarchy(
            branching=branching_values,
            points_per_leaf=points_per_leaf,
            level_scales=level_scales,
            dim=dim,
            random_state=feature_rng,
        )
        if normalize:
            features = features - features.mean(axis=0, keepdims=True)
            radius = np.linalg.norm(features, axis=1).max()
            features = features / max(radius, 1e-12)

        adjacency = None
        diagnostics = None
        accepted_attempt = None
        for attempt in range(max_attempts):
            graph_seed = int(
                np.random.SeedSequence(
                    [int(seed), graph_id, attempt, 0xCC]
                ).generate_state(1)[0]
            )
            graph = nx.random_regular_graph(degree, n_nodes, seed=graph_seed)
            candidate = sp.csr_matrix(
                nx.to_scipy_sparse_array(
                    graph,
                    format="csr",
                    dtype=np.float32,
                )
            )
            candidate.setdiag(0)
            candidate.eliminate_zeros()
            candidate_diagnostics = graph_spectral_diagnostics(candidate)
            if (
                candidate_diagnostics["lambda2_normalized"]
                >= min_lambda2_normalized
                and candidate_diagnostics["kappa_normalized"]
                <= max_kappa_normalized
                and candidate_diagnostics["rho_nontrivial"]
                <= max_rho_nontrivial
            ):
                adjacency = candidate
                diagnostics = candidate_diagnostics
                accepted_attempt = attempt
                break

        if adjacency is None or diagnostics is None or accepted_attempt is None:
            raise RuntimeError(
                f"Could not sample an accepted expander for graph {graph_id} "
                f"within {max_attempts} attempts."
            )

        data = _to_pyg(features, adjacency)
        data.labels = labels
        data.graph_id = torch.tensor(graph_id, dtype=torch.long)
        data.expander_attempt = torch.tensor(accepted_attempt, dtype=torch.long)
        data.degree = torch.tensor(degree, dtype=torch.long)
        for key, value in diagnostics.items():
            data[key] = torch.as_tensor(value)
        dataset.append(data)
    return dataset


def hierarchical_expanders(params: dict[str, Any]) -> list[Data]:
    """Legacy configuration-dictionary wrapper used by ``train.py``."""
    return _generate_hierarchical_expanders(**params)


__all__ = [
    "generate_recursive_hierarchy",
    "graph_spectral_diagnostics",
    "hierarchical_expanders",
]
