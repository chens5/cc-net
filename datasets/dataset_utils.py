from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse as sp
import torch
import os
from utils.globals import DATA_OUTPUT
import shortuuid

def project_l2(q, r):
    """Project each edge-vector row of q onto an ell_2 ball with (per-edge) radius r."""
    eps = 1e-8
    nrm = q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (q * torch.minimum(torch.ones_like(nrm), r.view(-1,1) / nrm)).float()

def divergence(p, src, dst, n):
    """Graph divergence: add +p_e at node i=src[e] and -p_e at node j=dst[e]."""
    d = p.size(-1)
    out = torch.zeros(n, d, dtype=p.dtype, device=p.device) # Fixed: Initialize out with p's dtype
    out.index_add_(0, src,  p)
    out.index_add_(0, dst, -p)
    return out

def graphlearning_to_pyg(X, W):
    """
    Converts from graph learning output to pytorch geometric object
    Parameters:
        X: node features
        W: edge weights 
    
    Returns: pytorch geometric data object
    """
    C = sp.triu(W, k=1).tocoo()

    src = torch.from_numpy(C.row)
    dst = torch.from_numpy(C.col)
    w   = torch.from_numpy(C.data)
    edge_index = torch.stack([src, dst])
    X = torch.tensor(X)
    return Data(x=X, edge_index=edge_index, edge_attr=w)

def convert_cfgdict_to_str(cfg):
    ds_params = cfg['params']
    config_str = "_".join(f"{k}={v}" for k, v in ds_params.items())
    return cfg['type'] + '_' + config_str


def save_dataset(cfg, dataset, which='train'):
    """
    cfg is structured as follows:
    {type: <global function for creating dataset>, params: {dataset parameters}}
    
    dataset: list of pytorch geometric data objects

    which: saving train/val/test
    """
    dataset_str = convert_cfgdict_to_str(cfg)
    datafile = os.path.join(DATA_OUTPUT, f'{dataset_str}-{which}.pt')
    if os.path.exists(datafile):
        random_short_id = shortuuid.uuid()
        datafile = os.path.join(DATA_OUTPUT, f'{dataset_str}-{which}-{random_short_id}.pt')
    torch.save(dataset,datafile)
    print("saved dataset in :", datafile)

def pdhg_algorithm(X, src, dst, w, lam=1.0, iters=200, tau=0.35, sigma=0.35, logging=True, **kwargs):
    """
    Baseline primal dual algorithm
    """
    sqrtw = w.sqrt()
    n, d = X.shape
    m = src.numel()
    U = X.clone()
    P = torch.zeros(m, d)

    #r = lam * w
    r = lam * sqrtw
    for _ in range(iters):
        # dual step (edge-wise projection onto norm ball with radius \lambda w)
        diff = U[src] - U[dst]
        #P = project_l2(P + tau * diff, r)
        P = project_l2(P + tau * (sqrtw[:, None] * diff), r)

        
        # primal step (node update with divergence of dual)
        #U = (U + sigma * (X - divergence(P, src, dst, n))) / (1.0 + sigma)
        U = (U + sigma * (X - divergence(sqrtw[:, None] * P, src, dst, n))) / (1.0 + sigma)

      
    return U, P

def get_pdhg_labels(data, lam):
    X = data.x
    edge_index = data.edge_index
    src = edge_index[0]
    dst = edge_index[1]
    w = data.edge_attr
    U, P = pdhg_algorithm(X=X, src=src, dst=dst, w=w, logging=False, lam=lam)
    return U, P 