import graphlearning as gl
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import torch
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
import scipy.sparse as sp
from scipy.sparse import coo_matrix, triu

from tqdm import tqdm
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from torch.nn import ReLU, LeakyReLU, Sigmoid, SiLU


from torch.optim import Adam, AdamW
from tqdm import trange

def project_l2(q, r):
    """Project each edge-vector row of q onto an ell_2 ball with (per-edge) radius r."""
    eps = 1e-8
    nrm = q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (q * torch.minimum(torch.ones_like(nrm), r.view(-1,1) / nrm)).float()

class PDHGLayer(MessagePassing):
    def __init__(self, node_dim, 
                       edge_dim, 
                       out_dim,
                       activation='SiLU',
                       lam=1.0, 
                       tau = 0.35, 
                       sigma = 0.1,
                       projection=project_l2, 
                       **kwargs):
        super().__init__(aggr='mean')
        '''functions for the equations'''
        self.f_edge_up = Linear(edge_dim, out_dim)
        self.f_edge_agg = Linear(node_dim, out_dim)
        self.activation = globals()[activation]()
        self.f_node_up = nn.Sequential(
            Linear(node_dim + out_dim, out_dim),
            self.activation,
            Linear(out_dim, out_dim)
        )
        

        self.lam = lam
        self.sigma = sigma
        self.tau = tau
        self.projection = projection
    
    def forward(self, h, e, edge_index, w):
        src, dst = edge_index
        edge_diff = h[src] - h[dst]

        '''first equation'''
        edge_up = self.f_edge_up(e)
        edge_agg = self.f_edge_agg(edge_diff)
        edge_update = edge_up + edge_agg

        r = self.lam * w
        e_proj = self.projection(edge_update, r) #Normalize
        edge_index = edge_index.long()
        agg = self.propagate(edge_index, edge_attr=e_proj)
        '''second equation'''
        node_input = torch.cat([h, agg], dim=-1)
        h_new = self.f_node_up(node_input)

        return h_new, e_proj

    def message(self, edge_attr):
        return edge_attr

class GraphPDHGNet(nn.Module):
    """
    Small network that stacks multiple GraphConv layers.

    First layer: (in_node_dim, in_edge_dim) -> hidden_dim
    Subsequent layers: (hidden_dim, hidden_dim) -> hidden_dim
    """
    def __init__(self,
                 in_node_dim,
                 in_edge_dim,
                 hidden_dim,
                 num_layers=3,
                 lam=1.0,
                 tau=0.35,
                 sigma=0.1, 
                 projection='project_l2',
                 activation='SiLU',
                 **kwargs):
        super().__init__()
        assert num_layers >= 1
        self.projection = globals()[projection]

        layers = []

        # first layer: input dims -> hidden
        layers.append(
            PDHGLayer(
                node_dim=in_node_dim,
                edge_dim=in_edge_dim,
                out_dim=hidden_dim,
                lam=lam,
                tau=tau,
                sigma=sigma,
                activation=activation,
                projection=self.projection
            )
        )

        # remaining layers: hidden -> hidden
        for _ in range(num_layers - 1):
            layers.append(
                PDHGLayer(
                    node_dim=hidden_dim,
                    edge_dim=hidden_dim,
                    out_dim=hidden_dim,
                    lam=lam,
                    tau=tau,
                    sigma=sigma,
                    activation=activation,
                    projection=self.projection
                )
            )
        layers.append(PDHGLayer(
                    node_dim=hidden_dim,
                    edge_dim=hidden_dim,
                    out_dim=in_node_dim,
                    lam=lam,
                    tau=tau,
                    sigma=sigma,
                    activation=activation,
                    projection=self.projection
                ))
        self.layers = nn.ModuleList(layers)
        self.hidden_dim = hidden_dim
    
    def forward(self, h, e, edge_index, w, **kwargs):
        """
        h: [N, in_node_dim] (or hidden_dim after first layer)
        e: [E, in_edge_dim] (or hidden_dim after first layer)
        edge_index: [2, E]
        w: [E]
        """
        for layer in self.layers:
            h, e = layer(h, e, edge_index, w)
        return h, e


class GNNBaseline(nn.Module):
    def __init__(self,
                 in_node_dim,
                 in_edge_dim,
                 hidden_dim,
                 layer_type='GATConv',
                 num_layers=3,
                 heads=2,
                 activation='SiLU',
                 **kwargs):
        super(GNNBaseline, self).__init__()
        # Initialize graph layer type
        self.layer_type = layer_type
        graph_layer = globals()[layer_type]

        # Initializing the activation function
        self.activation=globals()[activation]()

        self.initial = graph_layer(in_node_dim, hidden_dim, edge_dim=in_edge_dim, heads=heads)

        self.module_list = nn.ModuleList([graph_layer(hidden_dim, hidden_dim, edge_dim=edge_dim) for _ in range()])

        # Readout layer
        self.readout = nn.Linear(hidden_dim, in_node_dim)
        

    def forward(self, h, e, edge_index, w):
        x = self.initial(h, edge_index, edge_attr=w)
        x = self.activation(x)
        for layer in self.module_list:
            x = layer(x, edge_index, edge_attr=w)
            x = self.activation(x)
        x = self.output
        return 
