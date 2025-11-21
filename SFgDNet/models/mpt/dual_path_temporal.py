import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from . import model_4GRU

# Optional dependency: einops. If unavailable, fall back to permute/view.
try:
    from einops import rearrange  # noqa: F401
except Exception:
    rearrange = None  # type: ignore


class CausalConv1d(nn.Module):
    """1D conv with left-only padding that enforces strict causality.

    - Supports kernel_size in {3, 5} and arbitrary dilation.
    - Output length equals input length; no future frame access.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        assert kernel_size in (3, 5), "kernel_size must be 3 or 5"
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.pad_left = self.dilation * (self.kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, dilation=self.dilation, padding=0, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        x = F.pad(x, (self.pad_left, 0))
        return self.conv(x)

    @staticmethod
    def unit_test(device: Optional[torch.device] = None):
        dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        torch.manual_seed(0)
        B, C, T = 2, 3, 17
        x = torch.randn(B, C, T, device=dev)
        for ks in (3, 5):
            for dil in (1, 2, 4):
                m = CausalConv1d(C, C, kernel_size=ks, dilation=dil, groups=1).to(dev)
                y = m(x)
                assert y.shape[-1] == T
                t_cut = T // 2
                x_masked = x.clone()
                x_masked[..., t_cut + 1 :] = 0.0
                y_full = m(x)
                y_masked = m(x_masked)
                max_diff = (y_full[..., : t_cut + 1] - y_masked[..., : t_cut + 1]).abs().max().item()
                assert max_diff < 1e-5, f"Future leakage: ks={ks}, dil={dil}, diff={max_diff}"


class _LayerNormCT(nn.Module):
    """LayerNorm over channels at each time step (causality-safe)."""

    def __init__(self, channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x_ = x.transpose(1, 2)  # (B, T, C)
        x_ = self.ln(x_)
        return x_.transpose(1, 2)


class _FastResidualBlock(nn.Module):
    """Depthwise dilated causal conv residual block for micro-gestures."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.dw = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, groups=channels, bias=True)
        self.act = nn.SiLU()
        self.pw = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = _LayerNormCT(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.act(y)
        y = self.pw(y)
        y = self.norm(y)
        return x + y


class FastPathTemporalConv(nn.Module):
    """Per-node causal dilated convs capturing ~100–250 ms micro-gestures.

    Input: Z (B, T_in, N, C)
    Output: history H (B, T_in, N, C_f), Y_fast (B, T_out, N, d)
    """

    def __init__(self, in_channels: int, c_f: int, n_layers: int, t_out: int, d_out: int, kernel_size: int = 3):
        super().__init__()
        assert kernel_size in (3, 5)
        self.c_in = int(in_channels)
        self.c_f = int(c_f)
        self.L = int(n_layers)
        self.t_out = int(t_out)
        self.d_out = int(d_out)

        self.in_proj = nn.Conv1d(self.c_in, self.c_f, kernel_size=1)
        self.blocks = nn.ModuleList([
            _FastResidualBlock(self.c_f, dilation=2 ** i, kernel_size=kernel_size)
            for i in range(self.L)
        ])
        self.fast_head = nn.Sequential(
            CausalConv1d(self.c_f, self.c_f, kernel_size=kernel_size, dilation=1, groups=self.c_f),
            nn.SiLU(),
            nn.Conv1d(self.c_f, self.d_out * self.t_out, kernel_size=1),
        )

    def forward(self, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N, C = Z.shape
        x = Z.permute(0, 2, 3, 1).contiguous().view(B * N, C, T)
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)
        H = x.view(B, N, self.c_f, T).permute(0, 3, 1, 2).contiguous()
        head_out = self.fast_head(x)  # (B*N, d*T_out, T)
        last = head_out[:, :, -1]
        Y_fast = last.view(B, N, self.t_out, self.d_out).permute(0, 2, 1, 3).contiguous()
        return H, Y_fast


class SlowPathCausalGRU(nn.Module):
    """Causal GRU over flattened nodes with long memory (~1–3 s).

    Input: Z (B, T_in, N, C)
    Output: Y_slow (B, T_out, N, d)
    """

    def __init__(self, in_channels: int, n_nodes: int, hidden_size: int, t_out: int, d_out: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.c_in = int(in_channels)
        self.hidden = int(hidden_size)
        self.t_out = int(t_out)
        self.d_out = int(d_out)
        self.num_layers = int(num_layers)
        self.gru = nn.GRU(input_size=self.n_nodes * self.c_in, hidden_size=self.hidden, num_layers=self.num_layers, batch_first=True, dropout=dropout if self.num_layers > 1 else 0.0)
        self.slow_head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, self.t_out * self.n_nodes * self.d_out),
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        B, T, N, C = Z.shape
        assert N == self.n_nodes and C == self.c_in
        x = Z.view(B, T, N * C)
        out, h = self.gru(x)
        h_final = h[-1]
        y = self.slow_head(h_final)
        return y.view(B, self.t_out, self.n_nodes, self.d_out)


class NodeTypeEmbedding(nn.Module):
    """Learnable embeddings per node type."""

    def __init__(self, num_types: int, embed_dim: int, type_to_id: Optional[dict] = None):
        super().__init__()
        self.num_types = int(num_types)
        self.embed_dim = int(embed_dim)
        self.emb = nn.Embedding(self.num_types, self.embed_dim)
        self.type_to_id = dict(type_to_id or {})

    def forward(self, node_types: torch.Tensor) -> torch.Tensor:
        return self.emb(node_types)

    @staticmethod
    def build_from_type_dict(type_to_nodes: dict, embed_dim: int = 16, type_order: Optional[List[str]] = None):
        if type_order is None:
            type_names = sorted(list(type_to_nodes.keys()))
        else:
            type_names = list(type_order)
        type_to_id = {name: i for i, name in enumerate(type_names)}
        all_idx = [idx for lst in type_to_nodes.values() for idx in lst]
        N = max(all_idx) + 1 if all_idx else 0
        node_types = torch.zeros(N, dtype=torch.long)
        for name, idxs in type_to_nodes.items():
            t_id = type_to_id[name]
            for j in idxs:
                node_types[j] = t_id
        mod = NodeTypeEmbedding(num_types=len(type_names), embed_dim=embed_dim, type_to_id=type_to_id)
        return mod, node_types, type_to_id


class TemporalSummarizer(nn.Module):
    """Concat last, running mean, running std over observed window (causal)."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        last = Z[:, -1:, :, :]
        mean = Z.mean(dim=1, keepdim=True)
        var = Z.var(dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        return torch.cat([last, mean, std], dim=-1)


class GatedFusion(nn.Module):
    """Node/time-wise gating with fingertip bias and optional entropy regularizer."""

    def __init__(self, d_out: int, c_s: int):
        super().__init__()
        self.d_out = int(d_out)
        self.c_s = int(c_s)
        fin = self.c_s + 2 * self.d_out
        self.mlp = nn.Sequential(
            nn.Linear(fin, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, self.d_out),
        )

    def forward(self, Y_fast: torch.Tensor, Y_slow: torch.Tensor, Z_summ: torch.Tensor):
        B, T_out, N, d = Y_fast.shape
        assert Y_slow.shape == Y_fast.shape
        Zb = Z_summ.expand(B, T_out, N, self.c_s)
        U = torch.cat([Zb, Y_fast, Y_slow], dim=-1)
        logits = self.mlp(U)
        g = torch.sigmoid(logits)
        Vhat = g * Y_fast + (1.0 - g) * Y_slow
        eps = 1e-6
        gc = g.clamp(min=eps, max=1.0 - eps)
        H = -(gc * torch.log(gc) + (1.0 - gc) * torch.log(1.0 - gc)).mean()
        return Vhat, H


class DualPathTemporal(nn.Module):
    """Fast-path TCN + Slow-path GRU fused with node-type-aware gating.

    forward(Z, node_types) -> (Vhat, aux)
      - Z: (B, T_in, N, C)
      - node_types: (N,)
      - Vhat: (B, T_out, N, d)
      - aux: {'gate_entropy': tensor}
    """

    def __init__(
        self,
        in_channels: int,
        n_nodes: int,
        d_out: int,
        t_out: int,
        fast_c_f: int = 64,
        fast_layers: int = 4,
        fast_kernel: int = 3,
        slow_hidden: int = 256,
        slow_layers: int = 2,
        use_spatial_gcn: bool = True,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.in_channels = int(in_channels)
        self.d_out = int(d_out)
        self.t_out = int(t_out)
        self.use_spatial_gcn = bool(use_spatial_gcn)
        # Spatial GCN applied before temporal processing (per time step)
        self.GCN = model_4GRU._GraphConv(self.in_channels, 10, self.n_nodes) if self.use_spatial_gcn else None
        self.fast = FastPathTemporalConv(in_channels=self.in_channels, c_f=fast_c_f, n_layers=fast_layers, t_out=self.t_out, d_out=self.d_out, kernel_size=fast_kernel)
        self.slow = SlowPathCausalGRU(in_channels=self.in_channels, n_nodes=self.n_nodes, hidden_size=slow_hidden, t_out=self.t_out, d_out=self.d_out, num_layers=slow_layers)
        self.summarizer = TemporalSummarizer()
        self.register_buffer('tip_mask', None)
        c_s = self.in_channels * 3
        self.fusion = GatedFusion(d_out=self.d_out, c_s=int(c_s))

    def forward(self, Z: torch.Tensor):
        B, T, N, C = Z.shape
        assert N == self.n_nodes and C == self.in_channels
        # Apply spatial GCN per time step if enabled: (B, T, N, C) -> (B*T, C, N) -> GCN -> back
        if self.use_spatial_gcn and self.GCN is not None:
            Z_btcn = Z.permute(0, 1, 3, 2).contiguous().view(B * T, C, N)
            Z_btcn = self.GCN(Z_btcn)
            Z = Z_btcn.view(B, T, C, N).permute(0, 1, 3, 2).contiguous()
        _, Y_fast = self.fast(Z)
        Y_slow = self.slow(Z)
        Z_summ = self.summarizer(Z)
        Vhat, gate_entropy = self.fusion(Y_fast, Y_slow, Z_summ)
        return Vhat, {"gate_entropy": gate_entropy}


__all__ = [
    "CausalConv1d",
    "FastPathTemporalConv",
    "SlowPathCausalGRU",
    "NodeTypeEmbedding",
    "TemporalSummarizer",
    "GatedFusion",
    "DualPathTemporal"
]
