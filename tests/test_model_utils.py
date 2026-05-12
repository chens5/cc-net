import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from models import model_utils as mutils


def test_project_l2_scales_only_rows_outside_ball():
    q = torch.tensor([[3.0, 4.0], [1.0, 1.0]])
    r = torch.tensor([5.0, 1.0])

    out = mutils.project_l2(q, r)

    expected = torch.tensor([[3.0, 4.0], [2.0**-0.5, 2.0**-0.5]])
    torch.testing.assert_close(out, expected)


def test_project_linf_clips_coordinates():
    q = torch.tensor([[2.0, -0.5], [-3.0, 4.0]])
    r = torch.tensor([1.0, 2.0])

    out = mutils.project_linf(q, r)

    expected = torch.tensor([[1.0, -0.5], [-2.0, 2.0]])
    torch.testing.assert_close(out, expected)


def test_project_l1_projects_rows_to_l1_ball():
    q = torch.tensor([[3.0, 1.0, 0.0], [0.25, -0.25, 0.0], [1.0, -2.0, 3.0]])
    r = torch.tensor([2.0, 1.0, 3.0])

    out = mutils.project_l1(q, r)

    expected = torch.tensor([[2.0, 0.0, 0.0], [0.25, -0.25, 0.0], [0.0, -1.0, 2.0]])
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(out.abs().sum(dim=-1), torch.tensor([2.0, 0.5, 3.0]))


def test_prox_l0_hits_zero_and_identity_branches():
    q = torch.tensor([[1.0, 0.0], [3.0, 4.0]])
    r = torch.tensor([1.0, 1.0])

    out = mutils.prox_l0(q, r, tau=0.5)

    expected = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    torch.testing.assert_close(out, expected)


def test_prox_clipped_l1_hits_zero_shrink_and_identity_branches():
    q = torch.tensor([[0.5, 2.0, 5.0, -3.0]])
    r = torch.tensor([2.0])

    out = mutils.prox_clipped_l1(q, r, tau=0.5, clip_t=4.0)

    expected = torch.tensor([[0.0, 1.0, 5.0, -2.0]])
    torch.testing.assert_close(out, expected)


def test_prox_mcp_hits_zero_firm_and_identity_branches():
    q = torch.tensor([[1.0, 0.0], [2.0, 0.0], [7.0, 0.0]])
    r = torch.tensor([2.0, 2.0, 2.0])

    out = mutils.prox_mcp(q, r, tau=0.5, mcp_gamma=3.0)

    expected = torch.tensor([[0.0, 0.0], [1.2, 0.0], [7.0, 0.0]])
    torch.testing.assert_close(out, expected)


def test_prox_mcp_requires_tau_below_gamma():
    q = torch.tensor([[1.0, 0.0]])
    r = torch.tensor([1.0])

    with pytest.raises(ValueError, match="tau < mcp_gamma"):
        mutils.prox_mcp(q, r, tau=3.0, mcp_gamma=3.0)


def test_identity_returns_input_unchanged():
    q = torch.tensor([[1.0, -2.0]])

    assert mutils.identity(q, torch.tensor([1.0])) is q


def test_graph_pdhg_net_accepts_projection_kwargs():
    pytest.importorskip("graphlearning")
    pytest.importorskip("torch_geometric")

    from models.models import GraphPDHGNet

    default_model = GraphPDHGNet(in_node_dim=2, in_edge_dim=2, hidden_dim=4, num_layers=1)
    model = GraphPDHGNet(
        in_node_dim=2,
        in_edge_dim=2,
        hidden_dim=4,
        num_layers=1,
        tau=0.25,
        projection="prox_mcp",
        projection_kwargs={"mcp_gamma": 4.0},
    )

    assert default_model.layers[0].projection is mutils.project_l2
    assert model.layers[0].projection is mutils.prox_mcp
    assert model.layers[0].projection_kwargs == {"tau": 0.25, "mcp_gamma": 4.0}
