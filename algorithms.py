import graphlearning as gl
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import torch
import scipy.sparse as sp
from scipy.sparse import coo_matrix, triu

from losses.losses import energy, energy_pdg, kkt_residuals, divergence
from collections import defaultdict
from utils.globals import GLOBAL_OUTPUT, DATA_OUTPUT, ALGORITHM_OUTPUT

from tqdm import trange, tqdm
import shortuuid
import os
import argparse
import yaml
from models.model_utils import project_l2


def primal_dual(X, src, dst, w, lam=1.0, iters=200, tau=0.35, sigma=0.35, logging=True, **kwargs):
    """
    Baseline primal dual algorithm
    """

    n, d = X.shape
    m = src.numel()
    U = X.clone()
    P = torch.zeros(m, d)
    r = lam * w
    primal_objs = []
    pdgs=[]
    kkt = defaultdict(list)

    for _ in range(iters):
        # dual step (edge-wise projection onto norm ball with radius \lambda w)
        diff = U[src] - U[dst]
        P = project_l2(P + tau * diff, r)
        
        # primal step (node update with divergence of dual)
        U = (U + sigma * (X - divergence(P, src, dst, n))) / (1.0 + sigma)

        if logging: 
            primal_objective = energy(U, X, src, dst, w, lam).item()
            primal_dual_gap = energy_pdg(U, X, P, src, dst, w, lam, eps=1e-4)
            kkt_stats = kkt_residuals(U, P, X, src, dst, w, lam, eps=1e-8)

            primal_objs.append(primal_objective)
            pdgs.append(primal_dual_gap)
            for k, v in kkt_stats.items():
                kkt[k].append(v)
    return U, primal_objs, pdgs, kkt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, help='yaml file with experiment configs')
    parser.add_argument('--logging', action='store_true')
    args = parser.parse_args()

    with open(args.experiment, "r") as f:
        cfg = yaml.safe_load(f)
        print(cfg)

    
    # Simple loading and caching data 
    # Algorithm only uses train dataset
    dataset = torch.load(cfg['dataset_pth'])
    savepth = cfg['savepth']
    if not os.path.exists(savepth):
        os.makedirs(savepth)
    shortid = shortuuid.uuid()
    savefile = f'{savepth}/output-{shortid}.pt'
    print("Saving experiment to:", savefile)
    U_ = []
    primal_objs_ = []
    pdgs_ = []
    kkt_ = []
    for data in tqdm(dataset):
        X = data.x
        edge_index = data.edge_index 
        src = edge_index[0]
        dst = edge_index[1]
        w = data.edge_attr

        U, primal_objs, pdgs, kkt = primal_dual(X=X, src=src, dst=dst, w=w, logging=args.logging, **cfg)
        U_.append(U)
        primal_objs_.append(primal_objs)
        pdgs_.append(pdgs)
        kkt_.append(kkt)
    
    if args.logging:
        torch.save({'U': U_, 'primal_objs': primal_objs_, 'pdgs': pdgs_, 'kkt': kkt_}, savefile)
    else:
        torch.save({'U': U_}, savefile)

    print(f"Saved results to:{savefile}")