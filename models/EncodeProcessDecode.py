import torch
import torch_geometric as pyg
from torch_geometric.nn import MLP

class EncodeProcessDecode(torch.nn.Module):
    def __init__(self, 
                 processor,
                 in_node_dim,
                 in_edge_dim,
                 out_dim,
                 hidden_dim,
                 mlp_hidden_dim=32,
                 recurrent_steps = 1,
                 residual_stream = True
                ):
        super().__init__()
        
        self.node_encoder = MLP([in_node_dim, mlp_hidden_dim, hidden_dim])
        self.edge_encoder = MLP([in_edge_dim, mlp_hidden_dim, hidden_dim])

        if residual_stream:
            self.node_decoder = MLP([2 * hidden_dim, mlp_hidden_dim, out_dim])
            self.edge_decoder = MLP([2 * hidden_dim, mlp_hidden_dim, out_dim])
        else:
            self.node_decoder = MLP([hidden_dim, mlp_hidden_dim, out_dim])
            self.edge_decoder = MLP([hidden_dim, mlp_hidden_dim, out_dim])
        self.processor = processor
        self.recurrent_steps = recurrent_steps
        self.residual_stream = residual_stream

    def forward(self, h, e, edge_index, w, **kwargs):
        h_input = self.node_encoder(h)
        e_input = self.edge_encoder(e)
        h_hidden = h_input
        e_hidden = e_input
        for step in range(self.recurrent_steps):
            if self.residual_stream:
                h_hidden = torch.cat([h_hidden, h_input], dim=-1)
                e_hidden = torch.cat([e_hidden, e_input], dim=-1)
            h_hidden, e_hidden = self.processor(h_hidden, e_hidden, edge_index, w, **kwargs)
        h_out = self.node_decoder(h_hidden)
        e_out = self.edge_decoder(e_hidden)
        return h_out, e_out