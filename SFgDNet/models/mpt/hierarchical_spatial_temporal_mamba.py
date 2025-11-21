import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Reuse temporal pipeline exactly from dual_path_temporal_mamba
from .dual_path_temporal_mamba import (
    FastPathTemporalConv,
    SlowPathCausalMamba,
    TemporalSummarizer,
    GatedFusion,
)
from .temporal_dct import DCTTemporalFilter
from .augment import AugmentStack

# Reuse base hand adjacency definition
from . import model_4GRU


def _sym_norm_adj(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalize adjacency: D^{-1/2} A D^{-1/2}."""
    A = A.clone()
    A = 0.5 * (A + A.transpose(0, 1))
    deg = torch.clamp(A.sum(dim=-1), min=1e-6)
    inv_sqrt = torch.pow(deg, -0.5)
    D_inv_sqrt = torch.diag(inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


def _build_finger_groups_per_hand() -> Tuple[list, list]:
    """Return lists of 6 finger groups per hand (left, right).

    Indices follow the existing Graph edge layout in model_4GRU where nodes 0..20
    are the first hand and 21..41 are the second hand.
    """
    # Left hand (0..20). Five fingers + a palm cluster.
    left = [
        [0, 7, 6, 5],      # thumb
        [1, 10, 9, 8],     # index
        [2, 13, 12, 11],   # middle
        [3, 16, 15, 14],   # ring
        [4, 19, 18, 17],   # little
        [5, 8, 11, 14, 17] # palm cluster around hub (5)
    ]

    # Right hand (21..41) mirrors with +21 offset; hub is 26
    right = [
        [21, 28, 27, 26],
        [22, 31, 30, 29],
        [23, 34, 33, 32],
        [24, 37, 36, 35],
        [25, 40, 39, 38],
        [26, 29, 32, 35, 38] # palm cluster around hub (26)
    ]
    return left, right


def _membership_matrix(num_nodes: int, groups: list) -> torch.Tensor:
    """Build a binary membership matrix P of shape (num_nodes, n_groups).

    P[i, g] = 1 if node i belongs to group g else 0.
    """
    n_groups = len(groups)
    P = torch.zeros(num_nodes, n_groups)
    for g, idxs in enumerate(groups):
        for i in idxs:
            P[i, g] = 1.0
    return P


class GenericGraphConv(nn.Module):
    """Graph convolution over a fixed adjacency with learnable semantic complement.

    - Input: (B, F_in, N)
    - Output: (B, F_out, N)
    """

    def __init__(self, in_features: int, out_features: int, adj: torch.Tensor, use_layernorm: bool = True):
        super().__init__()
        assert adj.ndim == 2 and adj.size(0) == adj.size(1), "adj must be square"
        self.N = int(adj.size(0))
        self.F_in = int(in_features)
        self.F_out = int(out_features)

        # Base adjacency as buffer (normalized in forward to follow device)
        self.register_buffer('A_base', adj.clone().detach())
        # Learnable semantic adjacency complement and gates
        self.A_sem = nn.Parameter(torch.zeros(self.N, self.N))
        self.beta = nn.Parameter(torch.tensor(0.5))     # scales A_sem
        self.lambda_I = nn.Parameter(torch.tensor(0.1)) # scales identity

        # Feature projections
        self.lin_in = nn.Linear(self.F_in, self.F_out, bias=True)
        self.ln = nn.LayerNorm([self.F_out, self.N]) if use_layernorm else nn.Identity()
        self.act = nn.GELU(approximate='tanh')

        # Init
        nn.init.xavier_uniform_(self.lin_in.weight)
        if self.lin_in.bias is not None:
            nn.init.zeros_(self.lin_in.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F_in, N)
        B, _, N = x.shape
        assert N == self.N
        device = x.device

        A = torch.where(torch.isnan(self.A_base), torch.zeros_like(self.A_base), self.A_base).to(device)
        A_sem = torch.where(torch.isnan(self.A_sem), torch.zeros_like(self.A_sem), self.A_sem)
        I = torch.eye(self.N, device=device)
        A_eff = A + torch.clamp(self.beta, min=0).to(device) * A_sem + torch.clamp(self.lambda_I, min=0).to(device) * I
        A_eff = _sym_norm_adj(A_eff)

        # Linear over features per node
        y = x.transpose(1, 2)           # (B, N, F_in)
        y = self.lin_in(y)              # (B, N, F_out)
        y = y.transpose(1, 2)           # (B, F_out, N)

        # Node mixing by adjacency
        y = torch.matmul(y, A_eff)      # (B, F_out, N)

        # Norm + activation
        y = self.ln(y)
        y = self.act(y)
        return y


class GraphConvBlock(nn.Module):
    """Two-layer graph conv with gated residual, matching _GraphConv behavior."""

    def __init__(self, in_features: int, hidden_feature: int, adj: torch.Tensor, p_dropout: float = 0.005):
        super().__init__()
        self.N = int(adj.size(0))
        self.g1 = GenericGraphConv(in_features, hidden_feature, adj)
        self.g2 = GenericGraphConv(hidden_feature, in_features, adj)
        self.dropout = nn.Dropout(p_dropout) if p_dropout is not None else None
        self.act = nn.GELU()
        self.res_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.g1(x)
        if self.dropout is not None:
            y = self.dropout(y)
        y = self.g2(y)
        y = self.act(y)
        return x + torch.clamp(self.res_gate, min=0.0) * y


class HierarchicalSpatialModuleUNet(nn.Module):
    """U-Net style 3-level spatial hierarchy for the 42-node hand graph.

    Encoder: nodes (42) -> fingers (12) -> hands (2)
    Decoder: hands -> fingers (skip) -> nodes (skip) with concat-based refinement.

    Input:  (B, C, 42)
    Output: (B, C, 42)
    """

    def __init__(self, in_channels: int, hidden_feature: int = 10):
        super().__init__()
        base_graph = model_4GRU.Graph()
        A_base = base_graph.A_ske.clone().detach()
        assert A_base.size(0) == 42, "Expected 42-node hand graph"

        # Build finger groups and membership matrices
        left, right = _build_finger_groups_per_hand()
        finger_groups = left + right  # 12 groups (6 per hand)
        P_nf = _membership_matrix(42, finger_groups)
        assert P_nf.shape[1] == 12, f"Expected 12 sub-hand groups, got {P_nf.shape[1]}"

        # Hand membership (42 -> 2): left-hand (0..20), right-hand (21..41)
        P_nh = torch.zeros(42, 2)
        P_nh[:21, 0] = 1.0
        P_nh[21:, 1] = 1.0

        # Derive pooled level adjacencies via coarsening: A_G = P^T A P
        A_lvl2 = P_nf.T @ A_base @ P_nf  # (12, 12)
        P_fh = torch.zeros(12, 2)
        P_fh[:6, 0] = 1.0
        P_fh[6:, 1] = 1.0
        assert P_fh.shape == (12, 2)
        A_lvl3 = P_fh.T @ A_lvl2 @ P_fh  # (2, 2)

        # Normalize adjacencies for numerical stability in blocks
        self.register_buffer('A_lvl1', _sym_norm_adj(A_base))
        self.register_buffer('A_lvl2', _sym_norm_adj(A_lvl2))
        self.register_buffer('A_lvl3', _sym_norm_adj(A_lvl3))

        # Store membership for pooling/unpooling
        self.register_buffer('P_nf', P_nf)  # nodes -> sub-hand groups (42, 12)
        self.register_buffer('P_fh', P_fh)  # sub-hand -> hands (12, 2)
        self.register_buffer('P_nh', P_nh)  # nodes -> hands (42, 2)

        # Encoder graph convs
        self.enc_nodes = GraphConvBlock(in_channels, hidden_feature, self.A_lvl1)
        self.enc_fingers = GraphConvBlock(in_channels, hidden_feature, self.A_lvl2)
        self.bottleneck = GraphConvBlock(in_channels, hidden_feature, self.A_lvl3)

        # Decoder fusion + graph convs
        self.dec_fuse_fingers = nn.Conv1d(in_channels * 2, in_channels, kernel_size=1, bias=True)
        self.dec_fuse_nodes = nn.Conv1d(in_channels * 2, in_channels, kernel_size=1, bias=True)
        self.dec_fingers = GraphConvBlock(in_channels, hidden_feature, self.A_lvl2)
        self.dec_nodes = GraphConvBlock(in_channels, hidden_feature, self.A_lvl1)

        self.fuse_act = nn.GELU()
        self.res_gate = nn.Parameter(torch.tensor(0.1))

    def _pool_avg(self, x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Average pool features along node dimension using binary membership P (N, G)."""
        B, C, _ = x.shape
        G = P.size(1)
        device = x.device
        P = P.to(device)
        counts = torch.clamp(P.sum(dim=0), min=1.0)
        y = torch.matmul(x, P)  # (B, C, G)
        y = y / counts.view(1, 1, G)
        return y

    def _unpool_rep(self, x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Replicate pooled features back to nodes using membership P (N, G)."""
        device = x.device
        P = P.to(device)
        return torch.matmul(x, P.transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x_nodes = self.enc_nodes(x)                                       # (B, C, 42)
        x_fingers_in = self._pool_avg(x_nodes, self.P_nf)                 # (B, C, 12)
        x_fingers = self.enc_fingers(x_fingers_in)                        # (B, C, 12)
        x_hands_in = self._pool_avg(x_fingers, self.P_fh)                 # (B, C, 2)
        x_hands = self.bottleneck(x_hands_in)                             # (B, C, 2)

        # Decoder: hands -> fingers with skip fusion
        x_fingers_up = self._unpool_rep(x_hands, self.P_fh)               # (B, C, 12)
        x_fingers_cat = torch.cat([x_fingers_up, x_fingers], dim=1)       # (B, 2C, 12)
        x_fingers_refine = self.dec_fuse_fingers(x_fingers_cat)           # (B, C, 12)
        x_fingers_refine = self.fuse_act(x_fingers_refine)
        x_fingers_dec = self.dec_fingers(x_fingers_refine)                # (B, C, 12)

        # Decoder: fingers -> nodes with skip fusion
        x_nodes_up = self._unpool_rep(x_fingers_dec, self.P_nf)           # (B, C, 42)
        x_nodes_cat = torch.cat([x_nodes_up, x_nodes], dim=1)             # (B, 2C, 42)
        x_nodes_refine = self.dec_fuse_nodes(x_nodes_cat)                 # (B, C, 42)
        x_nodes_refine = self.fuse_act(x_nodes_refine)
        x_nodes_dec = self.dec_nodes(x_nodes_refine)                      # (B, C, 42)

        # Residual scaling with positive gate
        res_scale = F.softplus(self.res_gate)
        return x + res_scale * x_nodes_dec


class HierarchicalSpatialTemporalMamba(nn.Module):
    """Dual-path temporal pipeline with hierarchical spatial module.

    Spatial module differs from dual_path_temporal_mamba by using a 3-level
    hierarchical graph U-Net. Temporal processing is identical.

    forward(Z) -> (Vhat, aux)
      - Z: (B, T_in, N, C)
      - Vhat: (B, T_out, N, d)
      - aux: {'gate_entropy': tensor}
    """

    def __init__(self, in_channels: int, n_nodes: int, d_out: int, t_out: int,
                 fast_c_f: int = 64, fast_layers: int = 4, fast_kernel: int = 3,
                 slow_hidden: int = 256, slow_layers: int = 2, spatial_hidden: int = 10,
                 use_temporal_dct: bool = False, dct_window: Optional[int] = None,
                 dct_keep_coeffs: int = 8, dct_input_scale: float = 1.0,
                 use_augmentation: bool = False,
                 augment_node_dropout: float = 0.001,
                 augment_channel_dropout: float = 0.0,
                 augment_time_dropout: float = 0.0):
        super().__init__()
        if n_nodes != 42:
            raise ValueError("HierarchicalSpatialTemporalMamba currently expects n_nodes=42 (two hands)")
        self.n_nodes = int(n_nodes)
        self.in_channels = int(in_channels)
        self.d_out = int(d_out)
        self.t_out = int(t_out)

        # Optional learnable temporal filtering in the frequency domain
        self.temporal_filter = DCTTemporalFilter(
            in_channels=self.in_channels,
            window_size=dct_window,
            keep_coeffs=dct_keep_coeffs,
            input_scale=dct_input_scale,
        ) if use_temporal_dct else None

        # Hierarchical spatial processor
        self.hier_spatial = HierarchicalSpatialModuleUNet(self.in_channels, hidden_feature=spatial_hidden)

        # Temporal pipelines (identical to dual_path_temporal_mamba)
        self.fast = FastPathTemporalConv(in_channels=self.in_channels, c_f=fast_c_f, n_layers=fast_layers, t_out=self.t_out, d_out=self.d_out, kernel_size=fast_kernel)
        self.slow = SlowPathCausalMamba(in_channels=self.in_channels, n_nodes=self.n_nodes, hidden_size=slow_hidden, t_out=self.t_out, d_out=self.d_out, num_layers=slow_layers)
        self.summarizer = TemporalSummarizer()

        c_s = self.in_channels * 3
        self.fusion = GatedFusion(d_out=self.d_out, c_s=int(c_s))

        if use_augmentation:
            self.augment = AugmentStack(
                p_node=float(augment_node_dropout),
                p_ch=float(augment_channel_dropout),
                p_time=float(augment_time_dropout),
                frac_min=0.05,
                frac_max=0.2,
            )
        else:
            self.augment = None
    
    def forward(self, Z: torch.Tensor):
        if self.augment is not None and self.training:
            Z = self.augment(Z)
        B, T, N, C = Z.shape
        assert N == self.n_nodes and C == self.in_channels

        if self.temporal_filter is not None:
            Z = self.temporal_filter(Z)

        # Apply hierarchical spatial processing per time step: (B, T, N, C) -> (B*T, C, N) -> hier -> back
        Z_bt = Z.permute(0, 1, 3, 2).contiguous().view(B * T, C, N)
        Z_bt = self.hier_spatial(Z_bt)
        Z = Z_bt.view(B, T, C, N).permute(0, 1, 3, 2).contiguous()

        # Temporal branches and fusion (identical to dual_path_temporal_mamba)
        _, Y_fast = self.fast(Z)
        Y_slow = self.slow(Z)
        Z_summ = self.summarizer(Z)
        Vhat, gate_entropy = self.fusion(Y_fast, Y_slow, Z_summ)
        return Vhat, {"gate_entropy": gate_entropy}


__all__ = [
    "DCTTemporalFilter",
    "HierarchicalSpatialTemporalMamba",
    "HierarchicalSpatialModuleUNet",
]