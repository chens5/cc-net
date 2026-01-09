import graphlearning as gl
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import torch
import scipy.sparse as sp
from scipy.sparse import coo_matrix, triu

from . import model_utils as mutils

def primal_dual(X, src, dst, w, lam=1.0, iters=200, tau=0.35, sigma=0.35):
    """Main iterations """
    n, d = X.shape
    m = src.numel()
    U = X.clone()
    P = torch.zeros(m, d)
    r = lam * w
    losses = []
    for _ in tqdm(range(iters)):
        # dual step (edge-wise projection onto norm ball with radius \lambda w)
        diff = U[src] - U[dst]
        P = project_l2(P + tau * diff, r)
        # primal step (node update with divergence of dual)
        U = (U + sigma * (X - divergence(P, src, dst, n))) / (1.0 + sigma)

        losses.append(energy(U, X, src, dst, w, lam).item())
    return U
