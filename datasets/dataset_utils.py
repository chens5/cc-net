from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse as sp
import torch

DATA_OUTPUT = '/data/sam/cc-net/data'

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


def save_dataset(cfg, dataset):
    """
    cfg is structured as follows:
    {type: <global function for creating dataset>, params: {dataset parameters}}
    """
    dataset_str = save_dataset(cfg)
    datafile = os.path.join(DATA_OUTPUT, dataset_str+'.pt')
    torch.save(dataset,datafile)
    print("saved dataset in :", datafile)