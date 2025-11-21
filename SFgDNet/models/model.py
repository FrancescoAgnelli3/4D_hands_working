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
from torch_geometric.utils import dense_to_sparse, to_dense_adj


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

class STAG(nn.Module):

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, out_dim=3):
        super(STAG, self).__init__()
        self.spatial = SpatialBlock(in_channels=num_features, out_channels=8,num_nodes=num_nodes)

        self.encoder = Encoder(n_layers=2, n_head=4, d_k=2, d_v=2,d_model=8,
                                 d_inner=12,  dropout=0.1, n_position=num_timesteps_input, scale_emb=False)
        
        self.theta= nn.Parameter(torch.FloatTensor(num_timesteps_input * 8, num_timesteps_output*out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)

    def forward(self, A, X):

        out1 = self.spatial(X, A)#[N, 245, 36, 8]
        out2 = self.encoder(src_seq=out1.reshape(-1,out1.shape[2],out1.shape[3]), src_mask=None, return_attns=False)[0]
        out3=torch.matmul(out2.reshape(out2.shape[0],-1), self.theta)
        return out3

class EGNN(nn.Module):
    def __init__(self,num_past, num_future, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.horizon = num_future

        self.num_past=num_past
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_att_dim=0,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight))
        
        self.reset_parameters()
        self.to(self.device)
        

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)



    def forward(self, h, x, edges, edge_attr=None):
        h = self.embedding(h.unsqueeze(0).repeat(x.shape[0],1,1))
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h = h + time_embedding
        # h = self.PosEmbedding(h)

        permute = lambda x: x.permute(1, 0, 2)
        h, x = map(permute, [h, x])
        if edge_attr:
            edge_attr = edge_attr.unsqueeze(1).repeat(1,h.shape[1],1)

        for i in range(0, self.n_layers):
            h, x = self._modules["gcl_%d" % i](h, x, edges, edge_attr, None)

        x = permute(x)
        ### only one frame
        if x.shape[0]==1:
            x_hat=x.squeeze(0)
        else:
            x_hat=torch.einsum("ij,jkt->ikt", torch.softmax(self.theta,dim=1),x).squeeze(0)
        return x_hat

class ETNN(nn.Module):
    def __init__(self,  num_past, num_future, in_node_nf, hidden_nf, fft, eat, device, n_layers, n_nodes, nodes_att_dim=0,
                act_fn=nn.SiLU(), coords_weight=1.0, with_mask=False, tempo=True, filter=True):
        super(ETNN, self).__init__()
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
        if self.fft:
            edges_in_d = num_past-1


        for i in range(n_layers):
            self.add_module("egcl_at_%d" % (i*2+2), E_GCL_AT(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edges_in_d,
                                            act_fn=act_fn, recurrent=True, coords_weight=coords_weight, with_mask=with_mask))
        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))

        self.attn_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

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
        # h = self.PosEmbedding(h)

        Fs=None
        if self.fft:
            ### (n_edge, num_past-1), (b*n_node, num_past-1)
            edge_attr, Fs = self.FFT(h, x, self.n_nodes, edges=edges)
        

        permute = lambda x: x.permute(1, 0, 2)
        h, x = map(permute, [h, x])
        if Fs is not None: Fs = Fs.unsqueeze(1).repeat(1,h.shape[1],1)
        if edge_attr is not None: edge_attr = edge_attr.unsqueeze(1).repeat(1,h.shape[1],1)

        for i in range(self.n_layers):
            h, x = self._modules["egcl_at_%d" % (i*2+2)](h, x)

        x = permute(x)
        if self.tempo:
            x_hat=torch.einsum("ij,jkt->ikt", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]
        else:
            x_hat=torch.einsum("ij,jkt->ikt", torch.softmax(self.theta,dim=1),x).squeeze(0)

        return x_hat

class ENN(nn.Module):
    def __init__(self,  num_past, num_future, in_node_nf, hidden_nf, fft, eat, device, n_layers, n_nodes, nodes_att_dim=0,
                act_fn=nn.SiLU(), coords_weight=1.0, with_mask=False, tempo=True, filter=True):
        super(ENN, self).__init__()
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

        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))

        self.reset_parameters()
        self.to(self.device)
        

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)
        #O init
        self.theta.data*=0

    def forward(self, h, x, edges, edge_attr = None):
        """parameters
            h: (b*n_node, 1)
            x: (num_past, b*n_node, 3)
            edges: (2, n_edge)
            edge_attr: (n_edge, 3)
        """ 
        x_hat=torch.einsum("ij,jkt->ikt", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]

        return x_hat


class SESTAG(nn.Module):
    def __init__(self, num_past, num_future, in_node_nf, hidden_nf, fft, eat, device,
                 n_layers, n_nodes, edges = True, act_fn=nn.SiLU(), coords_weight=1.0, with_mask=False,
                 tempo=True, filter=True):
        super(SESTAG, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.num_past = num_past
        self.num_future = num_future
        self.horizon = num_future
        self.coords_weight = coords_weight
        self.edges = edges

        # Temporal projection
        self.theta = nn.Parameter(torch.randn(num_future, num_past))  # (F, P)
        self.TimeEmbedding = nn.Embedding(num_past, self.hidden_nf)
        self.PosEmbedding = PositionalEncoding(hidden_nf, max_len=num_past)
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        
        # Spatial encoder from pairwise distances
        self.spatial_encoder = nn.Sequential(
            nn.Linear(1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 3)
        )

        # Feature fusion
        self.future_encoder = nn.Sequential(
            nn.Linear(num_past, int((num_past + num_future)/2)),
            act_fn,
            nn.Linear(int((num_past + num_future)/2), num_future)  # Modulate time steps
        )

        self.attn_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        self.to(device)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.theta)

        
    def FCT(self, h, x, n_nodes, edges):
        x_ = rearrange(x, 't (b n) d -> t b n d', n=n_nodes)
        x_bar = torch.mean(x_, dim=-2, keepdim=True)
        x_norm = x_ - x_bar
        x_norm = rearrange(x_norm, 't b n d -> (b n) d t')
        
        ### (b*n_node, 3, num_past)
        # F = torch.fft.fftn(x_norm, dim=-1)
        F = dct(x_norm)

        attn_val = self.attn_mlp(h[1:]).squeeze(-1).transpose(0, 1)

        F = F[..., 1:]
        F_i = F[edges[0]]
        F_j = F[edges[1]]

        # Edge features via cosine similarity in DCT domain
        edge_attr = torch.abs(torch.sum(F_i * F_j, dim=-2))

        edge_attr = edge_attr * (attn_val[edges[0]] * attn_val[edges[1]])
        edge_attr_norm = edge_attr / (torch.sum(edge_attr, dim=-1, keepdim=True) + 1e-9)

        return edge_attr_norm

    def get_topk_adjacency(self, edge_attr_for_adj, k_percent = 0.15):
        num_edges = edge_attr_for_adj.shape[0]
        N = int(math.sqrt(num_edges))
        T = edge_attr_for_adj.shape[1]
        k = int(k_percent * num_edges)

        row_idx = torch.arange(N).repeat_interleave(N)  # source
        col_idx = torch.arange(N).repeat(N)             # target
        all_edges = torch.stack([row_idx, col_idx], dim=0).to(self.device)  # shape: (2, N^2)

        adjacency_list = []

        for t in range(T):
            edge_weights = edge_attr_for_adj[:, t]  # shape: (N^2,)
            _, topk_indices = torch.topk(edge_weights, k)

            # Select corresponding edge indices
            topk_edge_indices = all_edges[:, topk_indices]  # shape: (2, k)
            adjacency_list.append(topk_edge_indices)

        return torch.stack(adjacency_list)  # Each entry is (2, M) for a timestep

    def forward(self, h, x, edges, edge_attr=None):
        """
        h: (B * n_nodes, in_node_nf)
        x: (T_past, B * n_nodes, D)
        Returns:
            x_hat: (num_future, B * n_nodes, D)
        """
        T, NB, D = x.shape
        B = NB // self.n_nodes

        h_emb = self.embedding(h.unsqueeze(0).repeat(x.shape[0],1,1))
        
        time_embedding = self.TimeEmbedding(torch.arange(self.num_past).to(self.device)).unsqueeze(1)
        h_emb = h_emb + time_embedding

        # Reshape
        x = x.reshape(T, B, self.n_nodes, D).permute(1, 0, 2, 3)  # (B, T, n_nodes, D)
        h = h.view(B, self.n_nodes, -1)  # (B, n_nodes, in_node_nf)

        if self.edges:
            edge_attr = self.FCT(h_emb, x.permute(1, 0, 2, 3).reshape(T, -1, D), self.n_nodes, edges)
            adj_cos = self.get_topk_adjacency(edge_attr, k_percent=0.2)
            adj_cos = torch.stack([to_dense_adj(adj_cos[i], max_num_nodes=self.n_nodes).squeeze(0) for i in range(adj_cos.shape[0])])
            adj_cos = adj_cos.mean(dim=0) 
            adj_cos = dense_to_sparse((adj_cos > torch.quantile(adj_cos, 0.95)).to(torch.int64)+torch.eye(self.n_nodes).to(self.device))[0]
            src, dst = adj_cos  # (E,)
            E = src.shape[0]
            B, T, N, D = x.shape

            # Get relative vectors and distances for each edge (batch-wise)
            # x[:, :, src] → (B, T, E, D), same for dst
            rel_vec = x[:, :, src] - x[:, :, dst]  # (B, T, E, D)
            rel_dist = torch.norm(rel_vec, dim=-1, keepdim=True)  # (B, T, E, 1)

            # Encode spatial geometry
            geom_feat = self.spatial_encoder(rel_dist)  # (B, T, E, 3)

            # Reshape for scatter: combine B and T into one batch
            BT = B * T
            geom_feat_flat = geom_feat.view(BT, E, 3)  # (BT, E, 3)

            # Create destination index repeated over batch*time
            dst_expand = dst.unsqueeze(0).expand(BT, -1)  # (BT, E)

            # Flatten the output tensor (BT, N, 3)
            geom_agg_flat = torch.zeros(BT, self.n_nodes, 3, device=self.device)
            counts_flat = torch.zeros(BT, self.n_nodes, 1, device=self.device) + 1e-6

            # Scatter-add messages to destination nodes
            geom_agg_flat.scatter_add_(1, dst_expand.unsqueeze(-1).expand(-1, -1, 3), geom_feat_flat)
            counts_flat.scatter_add_(1, dst_expand.unsqueeze(-1).expand(-1, -1, 1), torch.ones_like(rel_dist.view(BT, E, 1)))

            # Normalize and reshape back to (B, T, N, 3)
            geom_agg = (geom_agg_flat / counts_flat).view(B, T, self.n_nodes, 3)
        else:
            # Fall back to full pairwise modeling
            rel_vec = x.unsqueeze(3) - x.unsqueeze(2)  # (B, T, n, n, D)
            rel_dist = torch.norm(rel_vec, dim=-1, keepdim=True)  # (B, T, n, n, 1)
            geom_feat = self.spatial_encoder(rel_dist)  # (B, T, n, n, 3)
            geom_agg = geom_feat.mean(dim=3)

        # Aggregate per node
        geom_agg = self.future_encoder(geom_agg.permute(0, 2, 3, 1))  # (B, n, 3, T_fut)
        geom_agg = geom_agg.permute(3, 0, 1, 2) # (T_fut, B, n, 3)
        geom_agg = geom_agg.reshape(self.num_future, B*self.n_nodes, -1)

        # Prepare motion input (x_rel): (T, B, n, D)
        x_rel = x - x[:, -1:].clone()  # (B, T, n, D)

        # Temporal projection
        x_rel = x_rel.permute(1, 0, 2, 3)  # (T, B, n, D)
        x_rel_flat = x_rel.reshape(T, B * self.n_nodes, D)
        x_hat_rel = torch.einsum("fp,pnd->fnd", self.theta, x_rel_flat)  # (T_fut, B*n, D)

        # Add back last position
        x_last = x[:, -1]  # (B, n, D)
        x_last_flat = x_last.reshape(B * self.n_nodes, D)
        x_hat = x_hat_rel*geom_agg + x_last_flat.unsqueeze(0)  # (F, B*n, D)

        return x_hat

class ESTAG(nn.Module):
    def __init__(self,  num_past, num_future, in_node_nf, hidden_nf, fft, eat, device, n_layers, n_nodes, nodes_att_dim=0,
                act_fn=nn.SiLU(), coords_weight=1.0, tempo=True, filter=True):
        super(ESTAG, self).__init__()
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
        if self.fft:
            edges_in_d = num_past-1
            nodes_att_dim = num_past-1


        for i in range(n_layers):
            self.add_module("egcl_%d" % (i*2+1), E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_att_dim,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight, norm_diff=True, clamp=True, n_nodes=self.n_nodes))
            if self.eat:
                self.add_module("egcl_at_%d" % (i*2+2), E_GCL_AT(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edges_in_d,
                                                act_fn=act_fn, recurrent=True, coords_weight=coords_weight))
        self.theta= nn.Parameter(torch.FloatTensor(num_future, num_past))

        self.attn_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

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
        # h = self.PosEmbedding(h)

        Fs=None
        if self.fft:
            ### (n_edge, num_past-1), (b*n_node, num_past-1)
            edge_attr, Fs = self.FFT(h, x, self.n_nodes, edges=edges)
        

        permute = lambda x: x.permute(1, 0, 2)
        h, x = map(permute, [h, x])
        if Fs is not None: Fs = Fs.unsqueeze(1).repeat(1,h.shape[1],1)
        if edge_attr is not None: edge_attr = edge_attr.unsqueeze(1).repeat(1,h.shape[1],1)

        for i in range(self.n_layers):
            h, x = self._modules["egcl_%d" % (i*2+1)](h, x, edges, edge_attr, Fs)

            if self.eat:
                h, x = self._modules["egcl_at_%d" % (i*2+2)](h, x)

        x = permute(x)
        if self.tempo:
            x_hat=torch.einsum("ij,jkt->ikt", self.theta,x-x[-1].unsqueeze(0)).squeeze(0)+x[-1]
        else:
            x_hat=torch.einsum("ij,jkt->ikt", torch.softmax(self.theta,dim=1),x).squeeze(0)

        return x_hat



def cal_similarity_fourier(fourier_features):
    similarity=torch.abs(torch.mm(torch.conj(fourier_features), fourier_features.t()))
    return similarity





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




class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=2):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)#100 13 10 6 -> 100 6 13 10  #100 13 8 16 #100 13 6 64 # 100 13 4 16

        ######## + -> *
        #100 64 13 9
        temp = self.conv1(X) * torch.sigmoid(self.conv2(X))

        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """

        ### (b, n, t, c)
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output, out_dim, device):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()

        self.device = device
        self.embedding = nn.Linear(num_features, 32)
        self.block1 = STGCNBlock(in_channels=32, out_channels=64,
                                 spatial_channels=32, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=32, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        

        #### 1 * 5 = 1 * (2 * 2 + 1)   (1 is kernel_size-1,    block1 -2  |  block2 -2  | last_temporal, -1)
        self.fully = nn.Linear((num_timesteps_input - 1 * 5) * 64,
                               num_timesteps_output*out_dim)


        self.theta= nn.Parameter(torch.FloatTensor(1, num_timesteps_output))
        self.reset_parameters()
        self.to(self.device)
        

    def reset_parameters(self):
        self.theta.data.uniform_(-1, 1)
        

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        X = self.embedding(X)
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))

        # x = rearrange(out4, 'b n (t d) -> t (b n) d', d=3)
        # x_hat=torch.einsum("ij,jkt->ikt", torch.softmax(self.theta,dim=1),x).squeeze(0)

        # return x_hat

        return out4



class AGLSTAN(nn.Module):
    ### embed_dim is d_e
    def __init__(self, num_nodes, batch_size, input_dim, output_dim, window, num_layers, filter_size, embed_dim, cheb_k):
        super(AGLSTAN, self).__init__()
        self.num_node = num_nodes
        self.batch_size = batch_size
        ### K
        self.input_dim = input_dim
        
        ### F
        self.output_dim = output_dim
        
        ### alpha
        self.window = window
        self.num_layers = num_layers
        self.filter_size = filter_size

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node,embed_dim), requires_grad=True)

        self.encoder = AGLTSA(num_nodes, input_dim, output_dim, cheb_k,
                                embed_dim, num_nodes * self.output_dim, filter_size, num_layers)

        self.end_conv = nn.Conv2d(in_channels=self.window, out_channels=1, padding=(2, 2), kernel_size=(5, 5), bias=True)

    def forward(self, source):
        #source: B, T_1, N, D
        #target: B, T_2, N, D

        output = self.encoder(source, self.node_embeddings)   #B, T, N, hidden

        
        output = output.view(self.batch_size, self.window, self.num_node, -1)
        output = self.end_conv(output)

        return output.squeeze(1)