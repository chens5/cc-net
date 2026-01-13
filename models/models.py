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

import torch_geometric as pyg
from torch_geometric.nn import MLP

from torch.optim import Adam, AdamW
from tqdm import trange
from . import model_utils as mutils


class PDHGLayer(MessagePassing):
    def __init__(self, node_dim, 
                       edge_dim, 
                       out_dim,
                       residual_dim,
                       activation='SiLU',
                       lam=1.0, 
                       tau = 0.35, 
                       sigma = 0.1,
                       projection='project_l2', 
                       **kwargs):
        super().__init__(aggr='sum')
        '''functions for the equations'''
        self.f_edge_up = Linear(edge_dim, out_dim)
        self.f_edge_agg = Linear(node_dim, out_dim)
        
        self.activation = globals()[activation]()
        self.f_node_up = nn.Sequential(
            # Linear(node_dim  + out_dim , out_dim),
            # Linear(node_dim + residual_dim + out_dim , out_dim),
            Linear(out_dim, out_dim),
            self.activation,
            Linear(out_dim, out_dim)
        )
        
        self.residual_linear_layer = Linear(residual_dim, out_dim)
        self.nf_linear_layer = Linear(node_dim, out_dim)
        self.aggregation_linear_layer = Linear(out_dim, out_dim)

        self.lam = lam
        self.sigma = sigma
        self.tau = tau
        projection_fn = getattr(mutils, projection)
        self.projection = projection_fn
    
    def forward(self, h, e, edge_index, w, x):
        sqrtw = w.sqrt().view(-1,1)
        src, dst = edge_index
        edge_diff = sqrtw.float() * (h[src] - h[dst])

        '''first equation'''
        edge_up = self.f_edge_up(e)
        edge_agg = self.f_edge_agg(edge_diff)
        edge_update = edge_up + edge_agg

        r = self.lam * sqrtw
        e_proj = self.projection(edge_update, r) # Normalization 
        dual = sqrtw.float() * e_proj
        edge_index = edge_index.long()
        agg = self.propagate(edge_index, edge_attr=dual)
        '''second equation'''
        # node_input = self.nf_linear_layer(h) +  self.aggregation_linear_layer(agg)
        node_input = self.nf_linear_layer(h) + self.residual_linear_layer(x) + self.aggregation_linear_layer(agg)
        # node_input = torch.cat([h, x, agg], dim=-1)
        # node_input = torch.cat([h, agg], dim=-1)
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
        assert in_node_dim == in_edge_dim 
        
        projection_fn = getattr(mutils, projection)
        self.projection = projection_fn

        layers = []
        residual_dim = in_node_dim

        # first layer: input dims -> hidden
        layers.append(
            PDHGLayer(
                node_dim=in_node_dim,
                edge_dim=in_edge_dim,
                out_dim=hidden_dim,
                residual_dim=in_node_dim,
                lam=lam,
                tau=tau,
                sigma=sigma,
                activation=activation,
                projection=projection
            )
        )

        # remaining layers: hidden -> hidden
        for _ in range(num_layers - 1):
            layers.append(
                PDHGLayer(
                    node_dim=hidden_dim,
                    edge_dim=hidden_dim,
                    out_dim=hidden_dim,
                    residual_dim=in_node_dim,
                    lam=lam,
                    tau=tau,
                    sigma=sigma,
                    activation=activation,
                    projection=projection
                )
            )
        layers.append(PDHGLayer(
                    node_dim=hidden_dim,
                    edge_dim=hidden_dim,
                    out_dim=in_node_dim,
                    residual_dim=in_node_dim,
                    lam=lam,
                    tau=tau,
                    sigma=sigma,
                    activation=activation,
                    projection=projection
                ))
        self.layers = nn.ModuleList(layers)
        self.hidden_dim = hidden_dim
    
    def forward(self, h, e, edge_index, w, x,**kwargs):
        """
        h: [N, in_node_dim] (or hidden_dim after first layer)
        e: [E, in_edge_dim] (or hidden_dim after first layer)
        edge_index: [2, E]
        w: [E]
        """
        for layer in self.layers:
            h, e = layer(h, e, edge_index, w, x)
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
        

    def forward(self, h, e, edge_index, w, **kwargs):
        x = self.initial(h, edge_index, edge_attr=w)
        x = self.activation(x)
        for layer in self.module_list:
            x = layer(x, edge_index, edge_attr=w)
            x = self.activation(x)
        x = self.output
        return x, e

class EncodeProcessDecode(torch.nn.Module):
    def __init__(self, 
                 processor_cfg,
                 in_node_dim,
                 in_edge_dim,
                 embedding_dim,
                 lam=1.0, 
                 mlp_hidden_dim=32,
                 recurrent_steps = 1,
                 residual_stream = True,
                 load_processor_parameters: str | None = None,
                 projection='project_l2',
                ):
        """
        Recurrent encode-processor-decode model.
        Processor config can be 
        """
        super().__init__()
        
        self.node_encoder = MLP([in_node_dim, mlp_hidden_dim, embedding_dim])
        self.edge_encoder = MLP([in_edge_dim, mlp_hidden_dim, embedding_dim])

        out_dim = in_node_dim # out_dim=in_node_dim as we need to recover centroids
        if residual_stream:
            self.node_decoder = MLP([2 * embedding_dim, mlp_hidden_dim, out_dim])
            self.edge_decoder = MLP([2 * embedding_dim, mlp_hidden_dim, out_dim])
        else:
            self.node_decoder = MLP([embedding_dim, mlp_hidden_dim, out_dim])
            self.edge_decoder = MLP([embedding_dim, mlp_hidden_dim, out_dim])
        
        # Load processor
        processor_class = globals()[processor_cfg['model']]
        # Force the in_dim for processor network to be the same as the embedding dimension
        proc_in_dim = 2*embedding_dim if residual_stream else embedding_dim
        processor = processor_class(in_node_dim=proc_in_dim, 
                                    in_edge_dim=proc_in_dim,
                                    lam=lam, 
                                    **processor_cfg['cfg'])
        self.processor = processor
        if load_processor_parameters is not None:
            print("Loading model from:", load_processor_parameters)
            model_state = torch.load(load_processor_parameters)
            self.processor.load_state_dict(model_state)

        self.recurrent_steps = recurrent_steps
        self.residual_stream = residual_stream
        self.lam = lam
        projection_fn = getattr(mutils, projection)
        self.projection = projection_fn

    def forward(self, h, e, edge_index, w, x, **kwargs):
        h_input = self.node_encoder(h)
        e_input = self.edge_encoder(e)
        h_hidden = h_input
        e_hidden = e_input
        for step in range(self.recurrent_steps):
            if self.residual_stream:
                h_hidden = torch.cat([h_hidden, h_input], dim=-1)
                e_hidden = torch.cat([e_hidden, e_input], dim=-1)
            h_hidden, e_hidden = self.processor(h_hidden, e_hidden, edge_index, w, x=h_input, **kwargs)
        h_out = self.node_decoder(h_hidden)
        e_out = self.edge_decoder(e_hidden)

        sqrtw = w.sqrt().view(-1,1)
        r = self.lam * sqrtw
        e_out = self.projection(e_out, r) # Normalization 
        return h_out, e_out