import torch


def _edge_radius(r, q):
    if not torch.is_tensor(r):
        r = q.new_tensor(r)
    else:
        r = r.to(device=q.device, dtype=q.dtype)
    if r.ndim == 0:
        return r.reshape(1, 1).expand(q.size(0), 1)
    return r.reshape(-1, 1)


def project_l2(q, r, **kwargs):
    """Project each edge-vector row of q onto an ell_2 ball with (per-edge) radius r."""
    eps = 1e-8
    r = _edge_radius(r, q)
    nrm = q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (q * torch.minimum(torch.ones_like(nrm), r / nrm)).float()


def project_linf(q, r, **kwargs):
    """Project each row of q onto a per-edge ell_inf ball."""
    r = _edge_radius(r, q).clamp_min(0.0)
    return torch.minimum(torch.maximum(q, -r), r)


def project_l1(q, r, **kwargs):
    """Project each row of q onto a per-edge ell_1 ball."""
    r = _edge_radius(r, q).clamp_min(0.0)
    abs_q = q.abs()
    sorted_q, _ = torch.sort(abs_q, dim=-1, descending=True)
    cssv = sorted_q.cumsum(dim=-1) - r
    idx = torch.arange(1, q.size(-1) + 1, device=q.device, dtype=q.dtype).view(1, -1)
    active = sorted_q * idx > cssv
    rho = active.sum(dim=-1, keepdim=True)
    rho_safe = rho.clamp_min(1)
    theta = cssv.gather(-1, rho_safe - 1) / rho_safe.to(q.dtype)
    theta = theta.clamp_min(0.0)
    return q.sign() * (abs_q - theta).clamp_min(0.0)


def prox_l0(q, r, tau=1.0, **kwargs):
    """Group hard-thresholding prox for lambda * 1_{z != 0}."""
    r = _edge_radius(r, q).clamp_min(0.0)
    tau = q.new_tensor(tau)
    threshold = torch.sqrt(2.0 * tau * r)
    keep = q.norm(dim=-1, keepdim=True) > threshold
    return torch.where(keep, q, torch.zeros_like(q))


def prox_clipped_l1(q, r, tau=1.0, clip_t=1.0, **kwargs):
    """Coordinatewise prox for lambda * min(abs(z), clip_t)."""
    r = _edge_radius(r, q).clamp_min(0.0)
    tau = q.new_tensor(tau)
    clip_t = q.new_tensor(clip_t)
    tau_r = tau * r
    abs_q = q.abs()
    soft = q.sign() * (abs_q - tau_r).clamp_min(0.0)
    return torch.where(
        abs_q > clip_t + 0.5 * tau_r,
        q,
        torch.where(abs_q <= tau_r, torch.zeros_like(q), soft),
    )


def prox_mcp(q, r, tau=1.0, mcp_gamma=3.0, **kwargs):
    """Group firm-thresholding prox for the minimax concave penalty."""
    if tau >= mcp_gamma:
        raise ValueError("prox_mcp requires tau < mcp_gamma")
    eps = 1e-8
    r = _edge_radius(r, q).clamp_min(0.0)
    tau = q.new_tensor(tau)
    gamma = q.new_tensor(mcp_gamma)
    nrm = q.norm(dim=-1, keepdim=True)
    lower = tau * r
    upper = gamma * r
    scale = gamma / (gamma - tau) * (1.0 - lower / nrm.clamp_min(eps))
    firm = scale * q
    return torch.where(nrm <= lower, torch.zeros_like(q), torch.where(nrm > upper, q, firm))


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


def identity(q, r=None, **kwargs):
    return q
