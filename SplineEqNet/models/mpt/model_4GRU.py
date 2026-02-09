import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
from typing import List, Optional, Sequence, Tuple

class Graph:
    def __init__(
        self,
        node_num: int = 21,
        adjacency: Optional[torch.Tensor] = None,
        edge_index: Optional[Sequence[Sequence[int]]] = None,
    ):
        self.node_num = int(node_num)
        self.edge: Sequence[Tuple[int, int]] = ()
        self.pre_sem_edge: Sequence[Tuple[int, int]] = ()
        self.A_ske = torch.zeros((self.node_num, self.node_num), dtype=torch.float32)
        self.A_pre_sem = torch.zeros((self.node_num, self.node_num), dtype=torch.float32)
        if adjacency is None and edge_index is None:
            self._build_default()
        else:
            self._build_from_metadata(adjacency, edge_index)

    def _build_default(self) -> None:
        if self.node_num < 42:
            raise ValueError("Default Graph expects at least 42 nodes.")
        self_link = [(i, i) for i in range(self.node_num)]
        bone_link_1 = [
            (4, 19), (3, 16), (2, 13), (1, 10), (19, 18), (16, 15), (13, 12), (10, 9),
            (18, 17), (15, 14), (12, 11), (9, 8), (17, 5), (14, 5), (11, 5), (8, 5),
            (0, 7), (7, 6), (6, 5), (20, 5), (17, 14), (14, 11), (11, 8),
        ]
        bone_link_2 = [
            (25, 40), (24, 37), (23, 34), (22, 31), (40, 39), (37, 36), (34, 33), (31, 30),
            (39, 38), (36, 35), (33, 32), (30, 29), (38, 26), (35, 26), (32, 26), (29, 26),
            (21, 28), (28, 27), (27, 26), (41, 26), (38, 35), (35, 32), (32, 29),
        ]
        edges = self_link + bone_link_1 + bone_link_2
        self.edge = edges
        self.pre_sem_edge = [(2, 7), (3, 8), (16, 21), (17, 22)]
        self.A_ske.zero_()
        for i, j in edges:
            if 0 <= i < self.node_num and 0 <= j < self.node_num:
                self.A_ske[i, j] = 1.0
                self.A_ske[j, i] = 1.0
        self.A_pre_sem.zero_()
        for p, q in self.pre_sem_edge:
            if 0 <= p < self.node_num and 0 <= q < self.node_num:
                self.A_pre_sem[p, q] = 1.0
                self.A_pre_sem[q, p] = 1.0

    def _build_from_metadata(
        self,
        adjacency: Optional[torch.Tensor],
        edge_index: Optional[Sequence[Sequence[int]]],
    ) -> None:
        if adjacency is not None:
            if torch.is_tensor(adjacency):
                A = adjacency.clone().detach().float()
            else:
                A = torch.tensor(adjacency, dtype=torch.float32)
        else:
            A = torch.zeros((self.node_num, self.node_num), dtype=torch.float32)

        if A.shape[0] != self.node_num or A.shape[1] != self.node_num:
            resized = torch.zeros((self.node_num, self.node_num), dtype=torch.float32)
            rows = min(self.node_num, A.shape[0])
            cols = min(self.node_num, A.shape[1])
            resized[:rows, :cols] = A[:rows, :cols]
            A = resized

        edges: List[Tuple[int, int]] = []
        loops_present = set()
        if edge_index:
            for pair in edge_index:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                try:
                    i, j = int(pair[0]), int(pair[1])
                except (TypeError, ValueError):
                    continue
                if not (0 <= i < self.node_num and 0 <= j < self.node_num):
                    continue
                edges.append((i, j))
                if i == j:
                    loops_present.add(i)
                A[i, j] = 1.0
                A[j, i] = max(A[j, i], 1.0)
        else:
            nz = (A > 0).nonzero(as_tuple=False)
            for i, j in nz:
                edges.append((int(i), int(j)))
                if i == j:
                    loops_present.add(int(i))

        for idx in range(self.node_num):
            if idx not in loops_present:
                edges.append((idx, idx))
                A[idx, idx] = 1.0

        self.edge = edges
        self.pre_sem_edge = []
        A = 0.5 * (A + A.transpose(0, 1))
        self.A_ske = A
        self.A_pre_sem = torch.zeros((self.node_num, self.node_num), dtype=torch.float32)

class SemskeConv(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        node_num,
        bias: bool = True,
        use_layernorm: bool = True,
        adjacency: Optional[torch.Tensor] = None,
        edge_index: Optional[Sequence[Sequence[int]]] = None,
    ):
        """Semantic + skeleton graph convolution with true feature mixing.

        Expects input of shape (B, F_in, N) and returns (B, F_out, N).
        - Per-node feature projection: Linear(F_in -> F_out)
        - Node mixing with normalized adjacency (skeleton + learnable semantic complement)
        - Optional LayerNorm over (F_out, N)
        """
        super(SemskeConv, self).__init__()
        self.node_num = int(node_num)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.use_layernorm = bool(use_layernorm)

        adj_source = adjacency.detach().cpu() if isinstance(adjacency, torch.Tensor) else adjacency
        self.graph = Graph(node_num=self.node_num, adjacency=adj_source, edge_index=edge_index)

        # Base skeleton adjacency (buffer so it moves with module to device)
        A_ske = self.graph.A_ske.clone().detach()
        # If requested node count differs from Graph's default (42), slice to fit.
        if A_ske.shape[0] != self.node_num:
            A_ske = A_ske[:self.node_num, :self.node_num]
        self.register_buffer('adj', A_ske)

        # Learnable semantic adjacency complement and gates
        self.A_sem = nn.Parameter(torch.zeros(node_num, node_num))
        # Edge weighting and post-aggregation node mixing matrices (kept for compatibility and logging)
        self.W = nn.Parameter(torch.zeros(node_num, node_num))
        self.M = nn.Parameter(torch.zeros(node_num, node_num))

        # Gating scalars for semantic strength and self-loop
        self.beta = nn.Parameter(torch.tensor(0.5))   # scales A_sem
        self.lambda_I = nn.Parameter(torch.tensor(0.1))  # scales identity

        # Feature projections: per-node linear maps
        self.lin_in = nn.Linear(self.in_features, self.out_features, bias=True)
        self.lin_out = nn.Identity()  # keep single projection in this layer; second layer will map back

        self.act = nn.GELU()
        self.ln = nn.LayerNorm([self.out_features, self.node_num]) if use_layernorm else nn.Identity()

        if bias:
            # Optional per-node bias after aggregation
            self.bias = nn.Parameter(torch.zeros(self.node_num))
        else:
            self.register_parameter('bias', None)

        # Initialization
        with torch.no_grad():
            # Start M as identity so it preserves message scale
            self.M.copy_(torch.eye(node_num))
            # Initialize W small and mask to skeleton edges
            nn.init.xavier_uniform_(self.W, gain=0.1)
            edge_mask = (A_ske > 0).float() + torch.eye(node_num)
            self.W.mul_(edge_mask)
            # Initialize Linear with Xavier
            nn.init.xavier_uniform_(self.lin_in.weight)
            if self.lin_in.bias is not None:
                self.lin_in.bias.zero_()

    def _normalize_adj(self, A: torch.Tensor) -> torch.Tensor:
        # symmetric normalization: D^{-1/2} A D^{-1/2}
        deg = torch.clamp(A.sum(dim=-1), min=1e-6)
        inv_sqrt = torch.pow(deg, -0.5)
        D_inv_sqrt = torch.diag(inv_sqrt)
        return D_inv_sqrt @ A @ D_inv_sqrt

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (B, F_in, N)
        B, F_in, N = input.shape
        device = input.device

        # Clean NaNs and build effective adjacency
        A_ske = torch.where(torch.isnan(self.adj), torch.zeros_like(self.adj), self.adj).to(device)
        A_sem = torch.where(torch.isnan(self.A_sem), torch.zeros_like(self.A_sem), self.A_sem)
        W = torch.where(torch.isnan(self.W), torch.zeros_like(self.W), self.W)
        M = torch.where(torch.isnan(self.M), torch.zeros_like(self.M), self.M)

        # A_eff = A_ske + beta * A_sem + lambda_I * I
        I = torch.eye(self.node_num, device=device)
        A_eff = A_ske + torch.clamp(self.beta, min=0).to(device) * A_sem + torch.clamp(self.lambda_I, min=0).to(device) * I
        # Symmetrize and normalize
        A_eff = 0.5 * (A_eff + A_eff.transpose(0, 1))
        A_eff = self._normalize_adj(A_eff)
        # Edge weighting (masked) stays elementwise; keep baseline flow via (1 + tanh(W))
        edge_mask = (A_ske > 0).float() + I
        A_eff = A_eff * (1.0 + torch.tanh(W) * edge_mask)

        # Feature projection per node: (B, N, F_in) -> (B, N, F_out) -> (B, F_out, N)
        x = input.transpose(1, 2)  # (B, N, F_in)
        x = self.lin_in(x)         # (B, N, F_out)
        x = x.transpose(1, 2)      # (B, F_out, N)

        # Node mixing via adjacency on the node dimension
        x = torch.matmul(x, A_eff)  # (B, F_out, N)
        x = torch.matmul(x, M)      # (B, F_out, N)

        # Optional bias per node
        if self.bias is not None:
            x = x + self.bias.view(1, 1, N)

        # Norm + nonlinearity
        x = self.ln(x)
        x = self.act(x)
        return x

class _GraphConv(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_feature,
        node_num,
        adjacency: Optional[torch.Tensor] = None,
        edge_index: Optional[Sequence[Sequence[int]]] = None,
        p_dropout: float = 0.005,
    ):
        super(_GraphConv, self).__init__()
        self.in_features = int(in_features)
        self.hidden_feature = int(hidden_feature)
        self.node_num = int(node_num)

        adj_source = adjacency.detach().cpu() if isinstance(adjacency, torch.Tensor) else adjacency
        self.gconv1 = SemskeConv(
            self.in_features,
            self.hidden_feature,
            self.node_num,
            use_layernorm=True,
            adjacency=adj_source,
            edge_index=edge_index,
        )
        self.gconv2 = SemskeConv(
            self.hidden_feature,
            self.in_features,
            self.node_num,
            use_layernorm=True,
            adjacency=adj_source,
            edge_index=edge_index,
        )

        self.act = nn.GELU()
        self.res_gate = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(p_dropout) if p_dropout is not None else None

    def forward(self, x):
        # x: (B, F_in, N)
        y = self.gconv1(x)
        if self.dropout is not None:
            y = self.dropout(y)
        y = self.gconv2(y)
        y = self.act(y)
        # Gated residual
        y = x + torch.clamp(self.res_gate, min=0.0) * y
        return y


class Generator(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        node_num,
        batch_size,
        space,
        num_layers: int = 4,
        adjacency: Optional[torch.Tensor] = None,
        edge_index: Optional[Sequence[Sequence[int]]] = None,
    ):
        super(Generator, self).__init__()

        self.space = space
        self.node_num = int(node_num)
        self.output_size = int(output_size)
        # Allow configurable number of GRU layers (default 4 for backward compatibility)
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        # Learnable initial state shared across the batch; expanded dynamically per forward call.
        self.hidden_init = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))
        

        # Rewire GRU to model time: input_size is number of nodes, sequence length is time
        self.GRU = nn.GRU(input_size=self.node_num, hidden_size=hidden_size,
                          num_layers=self.num_layers, dropout=0.05, batch_first=True)

        # Make GCN aware of the true feature dimension (time window length)
        adj_source = adjacency.detach().cpu() if isinstance(adjacency, torch.Tensor) else adjacency
        self.GCN = _GraphConv(input_size, 10, node_num, adjacency=adj_source, edge_index=edge_index)
        
        # Decoder maps final hidden state to per-node horizon outputs
        self.linear = nn.Linear(hidden_size, self.node_num * self.output_size)


    def forward(self, x, hidden_size):

        # Arrange as (B, T, N) for GRU over time (N = nodes)
        x = x.permute(0, 2, 1)
        # Optional spatial block acting on node dimension per time slice
        if self.space:
            x = self.GCN(x)

        batch_size = x.size(0)
        hidden_prev = self.hidden_init.expand(-1, batch_size, -1).contiguous().to(x.device)

        # GRU over time; take the final hidden state from top layer
        out_seq, h = self.GRU(x, hidden_prev)
        h_last = h[-1]  # (B, hidden_size)
        out = self.linear(h_last)  # (B, N * T_out)
        out = out.view(-1, self.node_num, self.output_size)  # (B, N, T_out)
        out = out.view(-1, self.output_size)  # (B*N, T_out)
        out = out.unsqueeze(dim=0)  # (1, B*N, T_out) to match existing caller view
        return out, h


# class Generator(nn.Module):

#     def __init__(self, input_size, hidden_size, output_size, node_num, batch_size, space, num_layers: int = 4):
#         super(Generator, self).__init__()

#         self.space = space
#         # Allow configurable number of GRU layers (default 4 for backward compatibility)
#         self.num_layers = int(num_layers)
#         self.hidden_prev = nn.Parameter(torch.zeros(self.num_layers, batch_size, hidden_size))
        

#         self.GRU = nn.GRU(input_size=input_size, hidden_size=hidden_size,
#                           num_layers=self.num_layers, dropout=0.05, batch_first=True)

#         self.GCN = _GraphConv(hidden_size, hidden_size, node_num)
        
#         self.linear = nn.Linear(hidden_size, output_size)


#     def forward(self, x, hidden_size):

#         out, h = self.GRU(x, self.hidden_prev)
#          # GCN block
#         if self.space:
#             x = x.permute(0, 2, 1)
#             GCN_set = self.GCN(x)
#             x = GCN_set.reshape(x.shape[0],x.shape[1],x.shape[2])
#             x = x.permute(0, 2, 1)
#         out = out.reshape(-1, hidden_size)
#         out = self.linear(out)
#         out = out.unsqueeze(dim=0)
#         return out, h
