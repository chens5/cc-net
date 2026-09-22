import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from datasets.expander import (
    _generate_hierarchical_expanders,
    generate_recursive_hierarchy,
    graph_spectral_diagnostics,
    hierarchical_expanders,
)
from datasets.datasets import hierarchical_expanders as registered_hierarchical_expanders


PILOT_PARAMS = {
    "branching": [2, 2, 2],
    "points_per_leaf": 10,
    "level_scales": [13.0, 6.0, 2.0],
    "dim": 2,
    "normalize": True,
    "seed": 1001,
    "degree": 15,
    "min_lambda2_normalized": 0.4,
    "max_kappa_normalized": 4.0,
    "max_rho_nontrivial": 0.6,
    "max_attempts": 100,
}

TRAIN_GRAPH_ZERO_X_SHA256 = (
    "a2e26f097729fe9071557ff18f331e807dc12462209c83b3191300a3035acbeb"
)
TRAIN_GRAPH_ZERO_EDGE_SHA256 = (
    "0592dfcee50180882b99472506701be3d6471943c13ff617c34cf09f28e220bd"
)


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _normalized_hierarchy(rng: np.random.Generator) -> tuple[np.ndarray, list[tuple]]:
    features, labels = generate_recursive_hierarchy(
        branching=PILOT_PARAMS["branching"],
        points_per_leaf=PILOT_PARAMS["points_per_leaf"],
        level_scales=PILOT_PARAMS["level_scales"],
        dim=PILOT_PARAMS["dim"],
        random_state=rng,
    )
    features = features - features.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(features, axis=1).max()
    return (features / max(radius, 1e-12)).astype(np.float32), labels


@pytest.fixture(scope="module")
def generated_pairs():
    params = {**PILOT_PARAMS, "n_graphs": 2}
    return (
        _generate_hierarchical_expanders(**params),
        _generate_hierarchical_expanders(**params),
    )


def test_pilot_seed_is_deterministic_and_matches_golden_graph(generated_pairs):
    first, repeated = generated_pairs

    for left, right in zip(first, repeated):
        torch.testing.assert_close(left.x, right.x, rtol=0, atol=0)
        torch.testing.assert_close(left.edge_index, right.edge_index, rtol=0, atol=0)
        torch.testing.assert_close(left.edge_attr, right.edge_attr, rtol=0, atol=0)
        assert left.labels == right.labels

    assert _tensor_sha256(first[0].x) == TRAIN_GRAPH_ZERO_X_SHA256
    assert _tensor_sha256(first[0].edge_index) == TRAIN_GRAPH_ZERO_EDGE_SHA256


def test_pilot_graph_structure_features_and_metadata(generated_pairs):
    graphs, _ = generated_pairs

    for graph_id, graph in enumerate(graphs):
        assert graph.num_nodes == 80
        assert graph.x.shape == (80, 2)
        assert graph.x.dtype == torch.float32
        assert graph.edge_index.shape == (2, 600)
        assert graph.edge_index.dtype == torch.long
        assert graph.edge_attr.shape == (600,)
        assert graph.edge_attr.dtype == torch.float32
        torch.testing.assert_close(graph.edge_attr, torch.ones_like(graph.edge_attr))

        src, dst = graph.edge_index
        assert bool(torch.all(src < dst))
        degree = torch.bincount(torch.cat((src, dst)), minlength=graph.num_nodes)
        torch.testing.assert_close(degree, torch.full_like(degree, 15))

        neighbors = [set() for _ in range(graph.num_nodes)]
        for left, right in zip(src.tolist(), dst.tolist()):
            neighbors[left].add(right)
            neighbors[right].add(left)
        reached = {0}
        frontier = [0]
        while frontier:
            node = frontier.pop()
            unseen = neighbors[node] - reached
            reached.update(unseen)
            frontier.extend(unseen)
        assert len(reached) == graph.num_nodes

        torch.testing.assert_close(
            graph.x.mean(dim=0), torch.zeros(2), rtol=0, atol=2e-7
        )
        assert float(torch.linalg.vector_norm(graph.x, dim=1).max()) == pytest.approx(
            1.0, abs=2e-7
        )

        assert int(graph.graph_id) == graph_id
        assert int(graph.degree) == 15
        assert int(graph.expander_attempt) >= 0
        assert len(graph.labels) == graph.num_nodes
        assert all(isinstance(label, tuple) and len(label) == 3 for label in graph.labels)
        assert graph.normalized_laplacian_eigenvalues.shape == (80,)
        assert graph.normalized_adjacency_abs_eigenvalues.shape == (80,)
        assert bool(torch.isfinite(graph.normalized_laplacian_eigenvalues).all())
        assert bool(torch.isfinite(graph.normalized_adjacency_abs_eigenvalues).all())

        assert float(graph.lambda2_normalized) >= 0.4
        assert float(graph.kappa_normalized) <= 4.0
        assert float(graph.rho_nontrivial) <= 0.6


def test_attached_spectral_metadata_matches_recomputation(generated_pairs):
    graph = generated_pairs[0][0]
    src, dst = graph.edge_index.numpy()
    rows = np.concatenate((src, dst))
    cols = np.concatenate((dst, src))
    weights = np.ones(rows.size, dtype=np.float64)

    scipy_sparse = pytest.importorskip("scipy.sparse")
    adjacency = scipy_sparse.csr_matrix(
        (weights, (rows, cols)), shape=(graph.num_nodes, graph.num_nodes)
    )
    diagnostics = graph_spectral_diagnostics(adjacency)

    scalar_names = (
        "lambda2_combinatorial",
        "lambda_max_combinatorial",
        "kappa_combinatorial",
        "lambda2_normalized",
        "lambda_max_normalized",
        "kappa_normalized",
        "rho_nontrivial",
    )
    for name in scalar_names:
        assert float(graph[name]) == pytest.approx(float(diagnostics[name]), abs=1e-6)
    np.testing.assert_allclose(
        graph.normalized_laplacian_eigenvalues.numpy(),
        diagnostics["normalized_laplacian_eigenvalues"],
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        graph.normalized_adjacency_abs_eigenvalues.numpy(),
        diagnostics["normalized_adjacency_abs_eigenvalues"],
        rtol=0,
        atol=1e-6,
    )


def test_feature_stream_advances_and_is_independent_of_topology(generated_pairs):
    graphs = generated_pairs[0]
    assert not torch.equal(graphs[0].x, graphs[1].x)

    feature_rng = np.random.default_rng(PILOT_PARAMS["seed"])
    expected_first, labels_first = _normalized_hierarchy(feature_rng)
    expected_second, labels_second = _normalized_hierarchy(feature_rng)

    np.testing.assert_array_equal(graphs[0].x.numpy(), expected_first)
    np.testing.assert_array_equal(graphs[1].x.numpy(), expected_second)
    assert graphs[0].labels == labels_first
    assert graphs[1].labels == labels_second


def test_hierarchy_helper_and_dataset_wrappers_are_deterministic():
    first_x, first_labels = generate_recursive_hierarchy(
        branching=[2, 2, 2],
        points_per_leaf=10,
        level_scales=[13.0, 6.0, 2.0],
        dim=2,
        random_state=np.random.default_rng(1001),
    )
    second_x, second_labels = generate_recursive_hierarchy(
        branching=[2, 2, 2],
        points_per_leaf=10,
        level_scales=[13.0, 6.0, 2.0],
        dim=2,
        random_state=np.random.default_rng(1001),
    )
    np.testing.assert_array_equal(first_x, second_x)
    assert first_labels == second_labels

    params = {
        **PILOT_PARAMS,
        "n_graphs": 1,
        "max_attempts": 1,
    }
    direct = hierarchical_expanders(params)
    registered = registered_hierarchical_expanders(params)
    torch.testing.assert_close(direct[0].x, registered[0].x, rtol=0, atol=0)
    torch.testing.assert_close(
        direct[0].edge_index, registered[0].edge_index, rtol=0, atol=0
    )


@pytest.mark.parametrize("degree", [0, 80])
def test_invalid_degree_is_rejected(degree):
    with pytest.raises(ValueError, match="degree"):
        _generate_hierarchical_expanders(
            **{**PILOT_PARAMS, "n_graphs": 1, "degree": degree}
        )


def test_odd_degree_sum_is_rejected():
    with pytest.raises(ValueError, match="even"):
        _generate_hierarchical_expanders(
            n_graphs=1,
            branching=[1],
            points_per_leaf=3,
            level_scales=[1.0],
            dim=2,
            seed=9,
            degree=1,
        )


def test_sampling_exhaustion_is_reported():
    with pytest.raises(RuntimeError, match="within 1 attempts"):
        _generate_hierarchical_expanders(
            **{
                **PILOT_PARAMS,
                "n_graphs": 1,
                "min_lambda2_normalized": 3.0,
                "max_kappa_normalized": float("inf"),
                "max_rho_nontrivial": float("inf"),
                "max_attempts": 1,
            }
        )
