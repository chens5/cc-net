from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.sparse as sp
import torch


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