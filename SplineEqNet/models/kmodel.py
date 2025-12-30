import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from models.gcl import E_GCL_AT, E_GCL, GCL
from models.layer import AGLTSA
from models.transformer.Models import Encoder
from einops import rearrange
from torch_dct import dct



#Non-equivariant STAG
class PositionalEncoding(nn.Module):
    def __init__(self, d_model,dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, None, :]
        return self.dropout(x)
 
class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(SpatialBlock, self).__init__()

        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels, out_channels))

        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv1, stdv1)


    def forward(self, X, A_hat):
        lfs1 = torch.einsum("ij,kjlm->kilm", [A_hat, X])
        t1 = F.relu(torch.matmul(lfs1, self.Theta1))

        return self.batch_norm(t1)

class kESTAG(nn.Module):
    def __init__(self,  num_past, num_future, in_node_nf, hidden_nf, fft, eat, device, n_layers, n_nodes, nodes_att_dim=0,
                act_fn=nn.SiLU(), coords_weight=1.0, with_mask=False, tempo=True, filter=True):
        super(kESTAG, self).__init__()
        self.hidden_nf = hidden_nf
        self.fft = fft
        self.eat = eat
        self.device = device
        self.n_layers = n_layers
        self.n_nodes=n_nodes
        self.num_past = num_past
        self.tempo = tempo
        self.filter = filter
        self.horizon = num_future
        
        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)

        edges_in_d = 0
        edges_in_d = num_past-1
        nodes_att_dim = num_past-1


        for i in range(n_layers):
            self.add_module("egcl_0_%d" % (i*2+1), E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, norm_diff=True, clamp=True))
            self.add_module("egcl_1_%d" % (i*2+1), E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, norm_diff=True, clamp=True))
            self.add_module("egcl_2_%d" % (i*2+1), E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, norm_diff=True, clamp=True))

            if self.eat:
                self.add_module("egcl_at_%d" % (i*2+2), E_GCL_AT(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edges_in_d,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, with_mask=with_mask))
        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))

        self.attn_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())
        
        self.temporal_weights = nn.Parameter(torch.ones(self.num_past)/self.num_past)  # shape: [num_past]

        self.reset_parameters()
        self.to(self.device)
        
    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)
        #O init
        self.theta.data*=0
         
    def FFT(self, h, x, n_nodes, edges):
        x_ = rearrange(x, 't (b n) d -> t b n d', n=n_nodes)
        x_bar = torch.mean(x_, dim=-2, keepdim=True)
        x_norm = x_ - x_bar
        x_norm = rearrange(x_norm, 't b n d -> (b n) d t')
        
        ### (b*n_node, 3, num_past)
        F = torch.fft.fftn(x_norm, dim=-1)

        ### (b*n_node, num_past-1)
        if self.filter:
            attn_val = self.attn_mlp(h[1:]).squeeze(-1).transpose(0, 1)
        else:
            # (b*n_node,), broadcast
            attn_val = torch.ones(h.shape[1], device=h.device).unsqueeze(-1)

        F = F[..., 1:]
        F_i = F[edges[0]]
        F_j = F[edges[1]]


        ## (n_egde, num_past-1)
        edge_attr = torch.abs(torch.sum(torch.conj(F_i) * F_j, dim=-2))

        edge_attr = edge_attr * (attn_val[edges[0]] * attn_val[edges[1]])

        edge_attr_norm = edge_attr / (torch.sum(edge_attr, dim=-1, keepdim=True)+1e-9)

        ### (b*n_node, num_past-1)
        Fs = torch.abs(torch.sum(F**2, dim=-2))
        
        Fs = Fs * attn_val

        Fs_norm = Fs / (torch.sum(Fs, dim=-1, keepdim=True)+1e-9)
        return edge_attr_norm, Fs_norm
    
    def FCT(self, h, x, n_nodes, edges):
        x_ = rearrange(x, 't (b n) d -> t b n d', n=n_nodes)
        x_bar = torch.mean(x_, dim=-2, keepdim=True)
        x_norm = x_ - x_bar
        x_norm = rearrange(x_norm, 't b n d -> (b n) d t')
        
        ### (b*n_node, 3, num_past)
        # F = torch.fft.fftn(x_norm, dim=-1)
        F = dct(x_norm)

        ### (b*n_node, num_past-1)
        if self.filter:
            attn_val = self.attn_mlp(h[1:]).squeeze(-1).transpose(0, 1)
        else:
            # (b*n_node,), broadcast
            attn_val = torch.ones(h.shape[1], device=h.device).unsqueeze(-1)

        F = F[..., 1:]
        F_i = F[edges[0]]
        F_j = F[edges[1]]

        # Edge features via cosine similarity in DCT domain
        edge_attr = torch.abs(torch.sum(F_i * F_j, dim=-2))

        edge_attr = edge_attr * (attn_val[edges[0]] * attn_val[edges[1]])
        edge_attr_norm = edge_attr / (torch.sum(edge_attr, dim=-1, keepdim=True) + 1e-9)

        Fs = torch.abs(torch.sum(F ** 2, dim=-2))
        Fs = Fs * attn_val
        Fs_norm = Fs / (torch.sum(Fs, dim=-1, keepdim=True) + 1e-9)

        return edge_attr_norm, Fs_norm

    def get_topk_adjacency(self, edge_attr_for_adj, edge_attr, k_percent = 0.15):
        num_edges = edge_attr_for_adj.shape[0]
        N = int(math.sqrt(num_edges))
        T = edge_attr_for_adj.shape[1]
        k = int(k_percent * num_edges)

        row_idx = torch.arange(N).repeat_interleave(N)  # source
        col_idx = torch.arange(N).repeat(N)             # target
        all_edges = torch.stack([row_idx, col_idx], dim=0).to(self.device)  # shape: (2, N^2)

        adjacency_list = []
        edge_attr_samples = torch.zeros(k, T, device=self.device)

        for t in range(T):
            edge_weights = edge_attr_for_adj[:, t]  # shape: (N^2,)
            _, topk_indices = torch.topk(edge_weights, k)

            # Select corresponding edge indices
            topk_edge_indices = all_edges[:, topk_indices]  # shape: (2, k)
            adjacency_list.append(topk_edge_indices)

            edge_attr_samples[:, t] = edge_attr[topk_indices, t]


        return torch.stack(adjacency_list), edge_attr_samples  # Each entry is (2, M) for a timestep

    def forward(self, h, x, edges, edge_attr = None):
        """parameters
            h: (b*n_node, 1)
            x: (num_past, b*n_node, 3)
            edges: (2, n_edge)
            edge_attr: (n_edge, 3)
        """ 

        ### (num_past, b*n_node, hidden_nf)
        h = self.embedding(h.unsqueeze(0).repeat(x.shape[0],1,1))
        
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding
        h = self.PosEmbedding(h)

        Fs=None
        if self.fft:
            ### (n_edge, num_past-1), (b*n_node, num_past-1)
            edge_attr_full, Fs = self.FFT(h, x, self.n_nodes, edges=edges)
            edge_attr_for_adj, _ = self.FCT(h, x, self.n_nodes, edges=edges)
        
        permute = lambda x: x.permute(1, 0, 2)
        h, x = map(permute, [h, x])
        if Fs is not None: Fs = Fs.unsqueeze(1).repeat(1,h.shape[1],1)
        if edge_attr_full is not None: 
            adjacency_list_fft, _ = self.get_topk_adjacency(edge_attr_full, edge_attr_full, k_percent=0.2)
            adjacency_list_cos, edge_attr = self.get_topk_adjacency(edge_attr_for_adj, edge_attr_full, k_percent=0.2)
            edge_attr = edge_attr.unsqueeze(1).repeat(1,h.shape[1],1)
            edge_attr_full = edge_attr_full.unsqueeze(1).repeat(1,h.shape[1],1)

            adjacency_list_cos = (adjacency_list_cos.to(torch.float32).mean(axis=0) > 0.8).to(torch.int64)
            adjacency_list_fft = (adjacency_list_fft.to(torch.float32).mean(axis=0) > 0.8).to(torch.int64)



        for i in range(self.n_layers):
            # h, x = self._modules["egcl_0_%d" % (i*2+1)](h, x, edges, edge_attr_full, Fs)
            h, x = self._modules["egcl_1_%d" % (i*2+1)](h, x, adjacency_list_cos, edge_attr, Fs)
            h, x = self._modules["egcl_2_%d" % (i*2+1)](h, x, adjacency_list_fft, edge_attr, Fs)
            if self.eat:
                h, x = self._modules["egcl_at_%d" % (i*2+2)](h, x)

        x = permute(x)
        if self.tempo:
            # x_hat=torch.einsum("ij,jkt->ikt", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]
            w = self.temporal_weights
            delta = (x * w.view(self.num_past, 1, 1)).sum(dim=0)
            x_hat=torch.einsum("ij,jkt->ikt", self.theta,x-delta.unsqueeze(0)).squeeze(0)+delta
        else:
            x_hat=torch.einsum("ij,jkt->ikt", torch.softmax(self.theta,dim=1),x).squeeze(0)

        return x_hat


class GNN(nn.Module):
    def __init__(self, num_future, num_past, input_dim, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, attention=0, recurrent=False):
        super(GNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.num_past = num_past

        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf = in_edge_nf ,
                                              act_fn=act_fn, attention=attention, recurrent=recurrent))

        self.decoder = nn.Sequential(nn.Linear(hidden_nf, hidden_nf),
                              act_fn,
                              nn.Linear(hidden_nf,3))
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))

        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))
        self.reset_parameters()
        self.to(self.device)
        

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)


    def forward(self, nodes, edges, edge_attr=None):
        h = self.embedding(nodes)
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding
        # h = self.PosEmbedding(h)
        
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)

        x = self.decoder(h)
        x_hat=torch.einsum("ij,jkt->ikt", torch.softmax(self.theta,dim=1),x).squeeze(0)

        return x_hat



