import argparse
import os
from collections import defaultdict
from typing import Callable

import shortuuid
import torch
import yaml
from tqdm import tqdm

from losses.losses import energy, energy_pdg, kkt_residuals, divergence
from models.model_utils import project_l2


def _graph_difference(U: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return U[src] - U[dst]


def _weighted_group_l2_shrinkage(Q: torch.Tensor, radius: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Row-wise proximal map of sum_i radius_i * ||q_i||_2:
    prox(Q)_i = max(1 - radius_i / ||q_i||, 0) * q_i.
    """
    nrm = Q.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(1.0 - radius.view(-1, 1) / nrm, min=0.0)
    return scale * Q


def vanilla_pdhg(
    X: torch.Tensor,
    edge_index: torch.Tensor,
    w: torch.Tensor,
    lam: float = 1.0,
    iters: int = 200,
    tau: float = 0.35,
    sigma: float = 0.35,
    use_extrapolation: bool = True,
    theta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Baseline PDHG solver used in notebooks/mnist_analysis.ipynb.
    """
    src, dst = edge_index
    src = src.long()
    dst = dst.long()
    w = w.to(dtype=X.dtype, device=X.device)
    sqrtw = w.sqrt()
    n, d = X.shape

    U = X.clone()
    U_bar = U.clone()
    P = torch.zeros(src.numel(), d, dtype=X.dtype, device=X.device)

    theta_eff = float(theta if use_extrapolation else 0.0)
    r = lam * sqrtw

    for _ in range(iters):
        diff_bar = U_bar[src] - U_bar[dst]
        P_new = project_l2(P + tau * (sqrtw[:, None] * diff_bar), r)

        div = divergence(sqrtw[:, None] * P_new, src, dst, n)
        U_new = (U + sigma * (X - div)) / (1.0 + sigma)

        U_prev = U
        U = U_new
        P = P_new
        U_bar = U + theta_eff * (U - U_prev)

    return U, P


def _cg_solve(
    op: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    tol: float,
    max_iters: int,
    eps: float = 1e-20,
) -> torch.Tensor:
    """
    Matrix-free conjugate-gradient solve for SPD linear operators.

    The SSNCG paper uses an absolute residual tolerance for the Newton
    equation, not a relative tolerance.
    """
    x = torch.zeros_like(b)
    r = b - op(x)
    p = r.clone()
    rs_old = (r * r).sum()

    if r.norm() <= tol:
        return x

    for _ in range(max_iters):
        Ap = op(p)
        denom = (p * Ap).sum()
        if torch.abs(denom) <= eps:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap

        if r.norm() <= tol:
            break

        rs_new = (r * r).sum()
        if rs_old.abs() <= eps:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


def _ssn_generalized_jacobian_apply(
    D: torch.Tensor,
    E: torch.Tensor,
    lam_w_over_sigma: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Apply one explicit Clarke generalized Jacobian element P in d prox_{p/sigma}(D) to E.
    Formula matches the construction in Sun-Toh-Yuan (2021), Section 5.4.
    """
    d_norm = D.norm(dim=-1, keepdim=True)
    alpha = lam_w_over_sigma.view(-1, 1) / d_norm.clamp_min(eps)
    active = alpha < 1.0

    inner = (D * E).sum(dim=-1, keepdim=True)
    proj_part = alpha * inner / d_norm.clamp_min(eps).pow(2) * D
    passthrough = (1.0 - alpha) * E
    PE_active = proj_part + passthrough

    return torch.where(active, PE_active, torch.zeros_like(E))


def _ssncg_subproblem(
    A: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    w: torch.Tensor,
    lam: float,
    sigma_k: float,
    Zk: torch.Tensor,
    X0: torch.Tensor,
    tau: float,
    max_newton_iters: int,
    cg_max_iters: int,
    stopping_tol: float | None = None,
) -> torch.Tensor:
    """
    Solve min_X phi_k(X) with semismooth Newton-CG.
    """
    X = X0.clone()
    n = X.shape[0]

    # Armijo and inexact Newton parameters.
    mu = 1e-4
    delta = 0.5
    eta_bar = 1e-1

    sigma_t = torch.tensor(float(sigma_k), dtype=X.dtype, device=X.device)
    lam_w = lam * w
    lam_w_over_sigma = lam_w / sigma_t

    def phi_and_grad(Xcur: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        BX = _graph_difference(Xcur, src, dst)
        D = BX + Zk / sigma_t
        Uprox = _weighted_group_l2_shrinkage(D, lam_w_over_sigma)
        prox_sigma_pstar = Zk + sigma_t * (BX - Uprox)
        grad = Xcur - A + divergence(prox_sigma_pstar, src, dst, n)

        phi_val = (
            0.5 * (Xcur - A).pow(2).sum()
            + (lam_w * Uprox.norm(dim=-1)).sum()
            + 0.5 * sigma_t * (Uprox - D).pow(2).sum()
        )
        return phi_val, grad

    if stopping_tol is None:
        stopping_tol = float(1e-8 * (1.0 + A.norm().item()))

    for _ in range(max_newton_iters):
        phi_x, grad = phi_and_grad(X)
        grad_norm = grad.norm()
        if grad_norm <= stopping_tol:
            break

        D = _graph_difference(X, src, dst) + Zk / sigma_t

        def hessian_op(V: torch.Tensor) -> torch.Tensor:
            BV = _graph_difference(V, src, dst)
            PBV = _ssn_generalized_jacobian_apply(D, BV, lam_w_over_sigma)
            return V + sigma_t * (divergence(BV, src, dst, n) - divergence(PBV, src, dst, n))

        cg_tol = float(min(eta_bar, grad_norm.pow(1.0 + tau).item()))
        direction = _cg_solve(
            op=hessian_op,
            b=-grad,
            tol=max(cg_tol, torch.finfo(X.dtype).eps),
            max_iters=cg_max_iters,
        )

        descent = (grad * direction).sum()
        if descent >= 0:
            direction = -grad
            descent = -(grad * grad).sum()

        # Armijo backtracking.
        step = 1.0
        accepted = False
        for _ in range(50):
            X_trial = X + step * direction
            phi_trial, _ = phi_and_grad(X_trial)
            if phi_trial <= phi_x + mu * step * descent:
                X = X_trial
                accepted = True
                break
            step *= delta

        if not accepted:
            break

    return X


def ssnal_convex_clustering(
    X: torch.Tensor,
    edge_index: torch.Tensor,
    w: torch.Tensor,
    lam: float = 1.0,
    iters: int = 200,
    tau: float = 0.35,
    sigma: float = 0.35,
    use_extrapolation: bool = True,
    theta: float = 1.0,
    U0: torch.Tensor | None = None,
    P0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Semismooth Newton augmented Lagrangian method (SSNAL) for weighted convex clustering.

    Interface intentionally matches `vanilla_pdhg`.
    Returns:
      - U: primal centroid matrix (n x d)
      - P: dual variable in the same scaled form used by PDHG
           (i.e. feasibility radius ||P_e|| <= lam * sqrt(w_e)).
      - Optional warm start:
        U0 initializes the primal variable, and P0 initializes the dual
        variable in PDHG scaling (same scaling as return value P).
    """
    src, dst = edge_index
    src = src.long()
    dst = dst.long()
    w = w.to(dtype=X.dtype, device=X.device)
    n, d = X.shape
    m = src.numel()

    if m == 0:
        U_ret = X.clone() if U0 is None else U0.to(dtype=X.dtype, device=X.device).clone()
        return U_ret, torch.zeros((0, d), dtype=X.dtype, device=X.device)

    sigma_k = max(float(sigma), 1e-6)
    ssn_tau = float(min(max(tau, 1e-6), 1.0))
    # Keep the penalty fixed by default for stability.
    # The PDHG-style extrapolation knobs are accepted only for interface compatibility.
    growth = 1.0

    Xk = X.clone() if U0 is None else U0.to(dtype=X.dtype, device=X.device).clone()
    if P0 is None:
        Zk = torch.zeros((m, d), dtype=X.dtype, device=X.device)
    else:
        if P0.shape != (m, d):
            raise ValueError(f"P0 must have shape {(m, d)}, got {tuple(P0.shape)}.")
        sqrtw = w.sqrt().clamp_min(1e-12).view(-1, 1)
        Zk = sqrtw * P0.to(dtype=X.dtype, device=X.device)
    Uk = _weighted_group_l2_shrinkage(
        _graph_difference(Xk, src, dst),
        (lam * w) / sigma_k,
    )

    max_newton_iters = 30
    cg_max_iters = min(400, max(40, 2 * n))

    for k in range(iters):
        alm_epsilon_k = min(1e-2, 1.0 / float((k + 1) ** 2))
        subproblem_tol = alm_epsilon_k / max(1.0, sigma_k**0.5)
        Xk = _ssncg_subproblem(
            A=X,
            src=src,
            dst=dst,
            w=w,
            lam=lam,
            sigma_k=sigma_k,
            Zk=Zk,
            X0=Xk,
            tau=ssn_tau,
            max_newton_iters=max_newton_iters,
            cg_max_iters=cg_max_iters,
            stopping_tol=subproblem_tol,
        )

        Dk = _graph_difference(Xk, src, dst) + Zk / sigma_k
        Uk = _weighted_group_l2_shrinkage(Dk, (lam * w) / sigma_k)
        primal_residual = _graph_difference(Xk, src, dst) - Uk
        Zk = Zk + sigma_k * primal_residual

        # Relative KKT-style residual (paper Section 6 criterion).
        eta_p = primal_residual.norm() / (1.0 + Uk.norm())
        eta_d = torch.relu(Zk.norm(dim=-1) - lam * w).sum() / (1.0 + X.norm())
        prox_consistency = Uk - _weighted_group_l2_shrinkage(Uk + Zk, lam * w)
        stationarity = Xk - X + divergence(Zk, src, dst, n)
        eta = (stationarity.norm() + prox_consistency.norm()) / (1.0 + X.norm() + Uk.norm())
        if torch.maximum(torch.maximum(eta_p, eta_d), eta).item() < 1e-6:
            break

        sigma_k = min(sigma_k * growth, 1e6)

    sqrtw = w.sqrt().clamp_min(1e-12).view(-1, 1)
    P = Zk / sqrtw
    return Xk, P


def primal_dual(
    X: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    w: torch.Tensor,
    lam: float = 1.0,
    iters: int = 200,
    tau: float = 0.35,
    sigma: float = 0.35,
    logging: bool = True,
    **kwargs,
):
    """
    Backward-compatible PDHG wrapper with optional metric logging.
    """
    sqrtw = w.sqrt()
    n, d = X.shape

    U = X.clone()
    U_bar = U.clone()
    P = torch.zeros(src.numel(), d, dtype=X.dtype, device=X.device)
    r = lam * sqrtw
    theta_eff = 1.0

    primal_objs = []
    pdgs = []
    kkt = defaultdict(list)

    for _ in range(iters):
        diff_bar = U_bar[src] - U_bar[dst]
        P_new = project_l2(P + tau * (sqrtw[:, None] * diff_bar), r)
        div = divergence(sqrtw[:, None] * P_new, src, dst, n)
        U_new = (U + sigma * (X - div)) / (1.0 + sigma)

        U_prev = U
        U = U_new
        P = P_new
        U_bar = U + theta_eff * (U - U_prev)

        if logging:
            primal_objective = energy(U, X, src, dst, w, lam).item()
            primal_dual_gap = energy_pdg(U, X, P, src, dst, w, lam, eps=1e-4)
            kkt_stats = kkt_residuals(U, P, X, src, dst, w, lam, eps=1e-8)

            primal_objs.append(primal_objective)
            pdgs.append(primal_dual_gap)
            for k, v in kkt_stats.items():
                kkt[k].append(v)

    return U, primal_objs, pdgs, kkt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, help="yaml file with experiment configs")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument(
        "--solver",
        type=str,
        default="pdhg",
        choices=["pdhg", "ssnal"],
        help="Which solver to run for each graph sample.",
    )
    args = parser.parse_args()

    with open(args.experiment, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        print(cfg)

    dataset = torch.load(cfg["dataset_pth"])
    savepth = cfg["savepth"]
    os.makedirs(savepth, exist_ok=True)
    shortid = shortuuid.uuid()
    savefile = f"{savepth}/output-{shortid}.pt"
    print("Saving experiment to:", savefile)

    U_ = []
    primal_objs_ = []
    pdgs_ = []
    kkt_ = []

    for data in tqdm(dataset):
        X = data.x.float()
        edge_index = data.edge_index.long()
        src = edge_index[0]
        dst = edge_index[1]
        w = data.edge_attr.float()

        if args.solver == "ssnal":
            U, P = ssnal_convex_clustering(
                X=X,
                edge_index=edge_index,
                w=w,
                lam=float(cfg.get("lam", 1.0)),
                iters=int(cfg.get("iters", 200)),
                tau=float(cfg.get("tau", 0.35)),
                sigma=float(cfg.get("sigma", 0.35)),
                use_extrapolation=bool(cfg.get("use_extrapolation", True)),
                theta=float(cfg.get("theta", 1.0)),
            )
            if args.logging:
                primal_objective = energy(U, X, src, dst, w, cfg.get("lam", 1.0)).item()
                primal_dual_gap = energy_pdg(U, X, P, src, dst, w, cfg.get("lam", 1.0), eps=1e-4)
                kkt_stats = kkt_residuals(U, P, X, src, dst, w, cfg.get("lam", 1.0), eps=1e-8)
                primal_objs_.append([primal_objective])
                pdgs_.append([primal_dual_gap])
                kkt_.append({k: [v] for k, v in kkt_stats.items()})
        else:
            U, primal_objs, pdgs, kkt = primal_dual(
                X=X,
                src=src,
                dst=dst,
                w=w,
                logging=args.logging,
                **cfg,
            )
            if args.logging:
                primal_objs_.append(primal_objs)
                pdgs_.append(pdgs)
                kkt_.append(kkt)

        U_.append(U)

    if args.logging:
        torch.save({"U": U_, "primal_objs": primal_objs_, "pdgs": pdgs_, "kkt": kkt_}, savefile)
    else:
        torch.save({"U": U_}, savefile)

    print(f"Saved results to:{savefile}")
