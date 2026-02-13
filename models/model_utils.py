import torch

def project_l2(q, r):
    """Project each edge-vector row of q onto an ell_2 ball with (per-edge) radius r."""
    eps = 1e-8
    nrm = q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (q * torch.minimum(torch.ones_like(nrm), r.view(-1,1) / nrm)).float()

def k_transpose_p(p, edge_index, w, num_nodes):
    src, dst = edge_index
    sqrtw = w.sqrt().view(-1, 1)
    msg = sqrtw * p
    out = p.new_zeros((num_nodes, p.size(-1)))
    out.index_add_(0, src,  msg)
    out.index_add_(0, dst, -msg)
    return out

def linear(q, **kwargs):
    raise NotImplementedError

def identity(q, **kwargs):
    raise NotImplementedError