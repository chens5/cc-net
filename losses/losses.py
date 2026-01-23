import torch 

def energy(U, X, src, dst, w, lam, **kwargs):
    """
    Objective: 0.5*||U-X||_2^2  +  lam * sum_e w_e ||U_i - U_j||_2.
    This returns the primal objective for convex clustering
    """
    data = 0.5 * (U - X).pow(2).sum()
    tv = (U[src] - U[dst]).norm(dim=-1).mul(w).sum()
    return data + lam * tv

def energy_pdg(U, X, P, src, dst, w, lam, eps=1e-4, return_parts=False, **kwargs):
    """
    Primal-dual gap loss.
    
    """
    n = U.size(0)
    sqrtw = w.sqrt()

    # primal
    G_u = 0.5 * (U - X).pow(2).sum()
    F_Ku = lam * (w * (U[src] - U[dst]).norm(dim=-1)).sum()

    # dual
    KTp = divergence(sqrtw.view(-1,1) * P, src, dst, n)
    q = -KTp
    G_star = (q * X).sum() + 0.5 * q.pow(2).sum()

    feasible = (P.norm(dim=-1) <= lam * sqrtw + eps).all().item()
    F_star = 0.0 if feasible else float("inf")
    if return_parts:
        return (G_u + F_Ku + G_star + F_star), G_u, F_Ku

    return (G_u + F_Ku + G_star + F_star)

def divergence(p, src, dst, n):
    """Graph divergence: add +p_e at node i=src[e] and -p_e at node j=dst[e]."""
    d = p.size(-1)
    out = torch.zeros(n, d, dtype=p.dtype, device=p.device) # Fixed: Initialize out with p's dtype
    out.index_add_(0, src,  p)
    out.index_add_(0, dst, -p)
    return out

def kkt_residuals(U, P, X, src, dst, w, lam, eps=1e-8):
    """
    Compute KKT residuals for convex clustering:

    - stationarity: (U - X) + div(P) = 0
    - dual feasibility: ||P_e|| <= lam * w_e
    - alignment: for active edges, P_e ≈ lam * w_e * (U_i - U_j)/||U_i - U_j||
    """
    n, d = U.shape
    m = P.shape[0]
    sqrtw = w.sqrt()

    # stationarity: (u_i - x_i) + sum_j p_ij = 0
    stat = (U - X) + divergence(sqrtw.view(-1,1) * P, src, dst, n)
    stat_abs = stat.norm()
    stat_rel = stat_abs / (X.norm() + eps)

    # dual feasibility: ||p_ij|| <= lam * w_ij
    p_norm = P.norm(dim=-1)
    feas_violation = torch.relu(p_norm - lam * sqrtw)
    feas_abs = feas_violation.norm()
    feas_rel = feas_abs / ((lam * sqrtw).norm() + eps)

    # alignment (only on "active" edges)
    diff = U[src] - U[dst]
    diff_norm = diff.norm(dim=-1)
    active = diff_norm > 1e-4

    if active.any():
        dir_vec = diff[active] / diff_norm[active].view(-1, 1)
        p_target = (lam * sqrtw[active]).view(-1, 1) * dir_vec
        align_res = P[active] - p_target
        align_abs = align_res.norm()
        align_rel = align_abs / (p_target.norm() + eps)
    else:
        align_abs = torch.tensor(0.0, dtype=U.dtype)
        align_rel = torch.tensor(0.0, dtype=U.dtype)

    # summed relative KKT residual (scalar)
    kkt_rel_total = stat_rel + feas_rel + align_rel

    return {
        "stat_abs":  float(stat_abs.item()),
        "stat_rel":  float(stat_rel.item()),
        "feas_abs":  float(feas_abs.item()),
        "feas_rel":  float(feas_rel.item()),
        "align_abs": float(align_abs.item()),
        "align_rel": float(align_rel.item()),
        "kkt_rel":   float(kkt_rel_total.item()),
    }