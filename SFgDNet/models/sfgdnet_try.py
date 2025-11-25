"""Multi-rate spectral siMLPe with spline-gated splits over DCT coefficients.
Public API/signature unchanged: same class names and ctor signatures.

Padding uses an internal learnable prediction module that, by default, runs the
compact MLP fallback: a single future frame is predicted from the observed
sequence and then repeated t_out times. When explicitly enabled, the RNN
padder replaces the MLP; disabling learned padding entirely falls back to
last-frame repetition.

Slow/fast band split:
    Let positions p_k ∈ [0,1] index the kept DCT rows.
    Each gate is parameterized by a compact B-spline expansion evaluated on p_k:
        g(p_k) = σ( Σ_{m=1}^M α_m B_m(p_k) )
    where {B_m} is a fixed open-uniform basis (degree configurable) and α_m are
    learned coefficients. This makes the gating functions smooth yet expressive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import torch
from torch import nn


# ----------------------------- Norms & FCs -----------------------------------

class _SpatialNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        inv_std = torch.rsqrt(var + self.eps)
        y = (x - mean) * inv_std
        return y * self.alpha + self.beta


class _TemporalNorm(nn.Module):
    def __init__(self, seq_len: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1, 1, seq_len))
        self.beta = nn.Parameter(torch.zeros(1, 1, seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        inv_std = torch.rsqrt(var + self.eps)
        y = (x - mean) * inv_std
        return y * self.alpha + self.beta


class _SpatialFC(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.proj(x)
        return x.transpose(1, 2)


class _TemporalFC(nn.Module):
    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _MLPBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        *,
        use_norm: bool = True,
        use_spatial_fc: bool = False,
        norm_axis: Literal["spatial", "temporal", "all"] = "spatial",
    ) -> None:
        super().__init__()
        self.fc = _SpatialFC(dim) if use_spatial_fc else _TemporalFC(seq_len)
        if use_norm:
            if norm_axis == "spatial":
                self.norm = _SpatialNorm(dim)
            elif norm_axis == "temporal":
                self.norm = _TemporalNorm(seq_len)
            elif norm_axis == "all":
                self.norm = nn.LayerNorm([dim, seq_len])
            else:
                raise ValueError(f"Unknown norm_axis={norm_axis}")
        else:
            self.norm = nn.Identity()
        if hasattr(self.fc, "proj"):
            nn.init.xavier_uniform_(self.fc.proj.weight, gain=1e-8)
            nn.init.constant_(self.fc.proj.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.norm(self.fc(x))


class _TransMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        *,
        use_norm: bool,
        use_spatial_fc: bool,
        num_layers: int,
        norm_axis: Literal["spatial", "temporal", "all"],
        mix_spatial_temporal: bool = False,
    ) -> None:
        super().__init__()
        blocks = []
        for layer_idx in range(num_layers):
            if mix_spatial_temporal:
                # Alternate projections: even -> temporal, odd -> spatial.
                use_spatial_layer = (layer_idx % 2) == 1
            else:
                use_spatial_layer = use_spatial_fc
            blocks.append(
                _MLPBlock(
                    dim,
                    seq_len,
                    use_norm=use_norm,
                    use_spatial_fc=use_spatial_layer,
                    norm_axis=norm_axis,
                )
            )
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------- DCT utils -----------------------------------

def _build_dct_matrix(total_len: int) -> torch.Tensor:
    if total_len <= 0:
        raise ValueError("Sequence length for DCT must be positive.")
    n = torch.arange(total_len, dtype=torch.float32).unsqueeze(1)
    k = torch.arange(total_len, dtype=torch.float32).unsqueeze(0)
    scale = math.sqrt(2.0 / total_len)
    mat = scale * torch.cos(math.pi * (n + 0.5) * k / float(total_len))
    mat[0, :] = math.sqrt(1.0 / total_len)
    return mat


def _softplus_inverse(value: float) -> float:
    if value <= 0.0:
        raise ValueError("softplus inverse expects a strictly positive value.")
    return math.log(math.expm1(float(value)))


def _open_uniform_knots(num_basis: int, degree: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    num_knots = num_basis + degree + 1
    start = torch.zeros(degree + 1, dtype=dtype, device=device)
    end = torch.ones(degree + 1, dtype=dtype, device=device)
    inner_count = num_knots - 2 * (degree + 1)
    if inner_count > 0:
        inner = torch.linspace(0.0, 1.0, inner_count + 2, dtype=dtype, device=device)[1:-1]
    else:
        inner = torch.empty(0, dtype=dtype, device=device)
    return torch.cat([start, inner, end])


def _build_bspline_basis(freq_pos: torch.Tensor, num_basis: int, degree: int) -> torch.Tensor:
    """
    Evaluate an open-uniform B-spline basis at the given normalized frequencies.
    Returns a matrix of shape (len(freq_pos), num_basis).
    """
    if num_basis <= 0:
        raise ValueError("num_basis must be positive for spline gates.")
    degree = max(0, int(degree))
    if num_basis < degree + 1:
        raise ValueError("num_basis must be at least degree+1 for spline gates.")

    freq = freq_pos.detach()
    dtype = freq.dtype
    device = freq.device
    knots = _open_uniform_knots(num_basis, degree, dtype, device)

    # Degree-0 basis
    basis_prev = []
    for i in range(num_basis):
        left = knots[i]
        right = knots[i + 1]
        if i == num_basis - 1:
            mask = (freq >= left) & (freq <= right)
        else:
            mask = (freq >= left) & (freq < right)
        basis_prev.append(mask.to(dtype))
    basis_prev = torch.stack(basis_prev, dim=0)  # (num_basis, K)

    if degree == 0:
        return basis_prev.transpose(0, 1).contiguous()

    for deg in range(1, degree + 1):
        basis_curr = torch.zeros_like(basis_prev)
        for i in range(num_basis):
            denom_left = float(knots[i + deg] - knots[i])
            denom_right = float(knots[i + deg + 1] - knots[i + 1])

            if denom_left > 0:
                term_left = ((freq - knots[i]) / denom_left) * basis_prev[i]
            else:
                term_left = torch.zeros_like(freq)

            if denom_right > 0 and i + 1 < num_basis:
                term_right = ((knots[i + deg + 1] - freq) / denom_right) * basis_prev[i + 1]
            else:
                term_right = torch.zeros_like(freq)

            basis_curr[i] = term_left + term_right
        basis_prev = basis_curr

    return basis_prev.transpose(0, 1).contiguous()


# -------------------------------- Configs ------------------------------------

@dataclass
class SiMLPeConfig:
    input_length: int
    pred_length: int
    num_nodes: int
    hidden_dim: int
    use_norm: bool = True
    use_spatial_fc_only: bool = False
    mix_spatial_temporal: bool = False
    norm_axis: Literal["spatial", "temporal", "all"] = "spatial"
    add_last_offset: bool = True
    force_mlp_padding: bool = True
    @property
    def total_length(self) -> int:
        return self.input_length + self.pred_length

    @property
    def feature_dim(self) -> int:
        return self.num_nodes * 3


@dataclass
class SFgDNetTryConfig(SiMLPeConfig):
    """siMLPe with learned soft band partition in the DCT domain."""
    dct_components: Optional[int] = 100
    mask_temperature: float = 1
    slow_num_layers: Optional[int] = 6
    fast_num_layers: Optional[int] = 6
    mid_gate_max: float = 1  # unused, kept for signature compatibility
    gate_basis_size: Optional[int] = None
    gate_basis_degree: int = 3
    learn_lambda: bool = True
    learn_centers: bool = True
    fixed_lambda: Optional[float] = None
    fixed_centers: Optional[Tuple[float, float]] = None
    padding_mode: Optional[Literal["mlp", "repeat_last"]] = None
    padding_predictor: Literal["mlp", "rnn"] = "mlp"
    padding_full_horizon: bool = False

    @property
    def dct_length(self) -> int:
        keep = self.dct_components
        total = self.total_length
        if keep is None:
            return total
        return max(1, min(int(keep), total))


# ---------------------------- Single-frame padder ----------------------------

class SingleFramePadder(nn.Module):
    """
    Predicts the padding frames from the observed sequence, optionally producing
    a full horizon instead of repeating one predicted step.

    The default predictor is an MLP, but an RNN can be used for the single-step case.
    """
    def __init__(
        self,
        obs_dim: int,
        d_model: int,
        *,
        predictor: Literal["mlp", "rnn"] = "mlp",
        output_steps: int = 1,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.output_steps = max(1, int(output_steps))
        predictor = str(predictor).lower()
        if predictor not in {"mlp", "rnn"}:
            raise ValueError(f"Unsupported padding predictor '{predictor}'.")
        self.predictor = predictor

        self.in_proj = nn.Linear(obs_dim, d_model)
        self.act = nn.SiLU()

        # Temporal encoder shared by both predictors.
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.rnn: Optional[nn.GRU] = (
            nn.GRU(d_model, d_model, batch_first=True) if self.predictor == "rnn" else None
        )

        head_out = obs_dim * self.output_steps
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, head_out),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1e-3)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        # past: (B, T_in, F)
        x = self.act(self.in_proj(past))         # (B, T_in, d)
        h = self.encoder(x)                      # (B, T_in, d)
        if self.rnn is not None:
            out, _ = self.rnn(h)                 # (B, T_in, d)
            z = out[:, -1, :]
        else:
            z = h[:, -1, :]

        delta = self.head(z)                     # (B, F * steps)
        tau = 0.1
        delta = tau * torch.tanh(delta / max(1e-6, tau))

        if self.output_steps == 1:
            return past[:, -1, :] + delta

        delta = delta.view(delta.size(0), self.output_steps, self.obs_dim)
        base = past[:, -1:, :].expand(-1, self.output_steps, -1)
        return base + delta




# ------------------------------ Backbones ------------------------------------

class SiMLPeBackbone(nn.Module):
    def __init__(self, cfg: SiMLPeConfig, seq_len_override: Optional[int] = None, num_layers_override: Optional[int] = None) -> None:
        super().__init__()
        self.cfg = cfg
        seq_len = cfg.total_length if seq_len_override is None else int(seq_len_override)
        default_layers = getattr(cfg, "num_layers", 6)
        num_layers = default_layers if num_layers_override is None else int(num_layers_override)

        self.fc_in = nn.Linear(cfg.feature_dim, cfg.hidden_dim)
        self.mlp = _TransMLP(
            cfg.hidden_dim,
            seq_len,
            use_norm=cfg.use_norm,
            use_spatial_fc=cfg.use_spatial_fc_only,
            num_layers=num_layers,
            norm_axis=cfg.norm_axis,
            mix_spatial_temporal=bool(getattr(cfg, "mix_spatial_temporal", False)),
        )
        self.fc_out = nn.Linear(cfg.hidden_dim, cfg.feature_dim)

        nn.init.xavier_uniform_(self.fc_out.weight, gain=1e-8)
        nn.init.constant_(self.fc_out.bias, 0.0)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(seq)
        x = x.transpose(1, 2).contiguous()
        x = self.mlp(x)
        x = x.transpose(1, 2).contiguous()
        return self.fc_out(x)


class SFgDNetTryForecaster(nn.Module):
    """Two-branch siMLPe with spline-based gates over kept DCT rows.
    Signature: __init__(cfg), forward(history) -> future.

    Padding:
        By default, compute padding with the learned predictor (repeat single
        frame or full horizon) and fall back to the last frame if
        cfg.padding_mode is ``repeat_last``.
    """
    def __init__(self, cfg: "SFgDNetTryConfig") -> None:
        super().__init__()
        self.cfg = cfg
        T = cfg.total_length

        # DCT basis (keep K)
        K = cfg.dct_length
        dct = _build_dct_matrix(T)
        idct = dct.t()
        self.register_buffer("_dct_keep", dct[:K, :], persistent=False)
        self.register_buffer("_idct_keep", idct[:, :K], persistent=False)
        self.full_seq_len = T
        self.dct_len = K

        # Frequency mask over kept bands
        self.mask_logits = nn.Parameter(torch.zeros(K))
        self.mask_temperature = nn.Parameter(0.1 * torch.ones(1), requires_grad=True)

        # Frequency positions p_k ∈ [0,1]
        freq_pos = torch.linspace(0.0, 1.0, K)
        self.register_buffer("_freq_pos", freq_pos, persistent=False)

        # Spline gate parameters
        gate_degree = max(0, int(cfg.gate_basis_degree))
        gate_basis = cfg.gate_basis_size if cfg.gate_basis_size is not None else min(K, 8)
        gate_basis = max(gate_degree + 1, int(gate_basis))
        basis = _build_bspline_basis(freq_pos, gate_basis, gate_degree)
        self.register_buffer("_gate_basis", basis, persistent=False)
        self.relu = nn.ReLU()

        self.alpha_slow = nn.Parameter(torch.ones(gate_basis))
        self.alpha_fast = nn.Parameter(torch.ones(gate_basis))
        slow_mask = torch.zeros(gate_basis)
        fast_mask = torch.zeros(gate_basis)
        split_idx = int(gate_basis * 0.6)  # or whatever
        slow_mask[:split_idx] = 1.0
        fast_mask[1-split_idx:] = 1.0
        self.register_buffer("_slow_alpha_mask", slow_mask, persistent=False)
        self.register_buffer("_fast_alpha_mask", fast_mask, persistent=False)
        self.softplus = nn.Softplus()


        # Branch depths
        default_layers = getattr(cfg, "num_layers", 6)
        slow_layers = cfg.slow_num_layers if cfg.slow_num_layers is not None else default_layers
        fast_layers = cfg.fast_num_layers if cfg.fast_num_layers is not None else max(1, default_layers // 2)

        # Branches operate on K×F sequences
        self.slow_branch = SiMLPeBackbone(cfg, seq_len_override=K, num_layers_override=slow_layers)
        self.fast_branch = SiMLPeBackbone(cfg, seq_len_override=K, num_layers_override=fast_layers)

        # Padding mode
        pad_mode_raw = getattr(cfg, "padding_mode", None)
        pad_mode = "mlp" if pad_mode_raw is None else str(pad_mode_raw).lower()
        if pad_mode not in {"mlp", "repeat_last"}:
            raise ValueError(
                f"Unknown padding_mode='{pad_mode_raw}'. Expected one of ('mlp','repeat_last')."
            )
        self.padding_mode = pad_mode

        # Internal single-frame padder if needed
        self._padder: Optional["SingleFramePadder"] = None
        need_learned_padding = cfg.pred_length > 0 and self.padding_mode == "mlp"
        if need_learned_padding:
            self._padder = SingleFramePadder(
                obs_dim=cfg.feature_dim,
                d_model=int(cfg.hidden_dim / 4),
                predictor=cfg.padding_predictor,
                output_steps=cfg.pred_length if cfg.padding_full_horizon else 1,
            )

        # for inspection
        self.last_gate_slow: Optional[torch.Tensor] = None
        self.last_gate_fast: Optional[torch.Tensor] = None
        self.last_mask: Optional[torch.Tensor] = None

        # Runtime controls
        self._fixed_mask_value: Optional[float] = None
        self._fixed_mid_gate_value: Optional[float] = None
        self._use_slow_branch: bool = True
        self._use_fast_branch: bool = True
        self._skip_dct: bool = False
        self._gate_frozen: bool = False

        # Learning flags for gate parameters
        self._learn_slow_gate: bool = bool(getattr(cfg, "learn_lambda", True))
        self._learn_fast_gate: bool = bool(getattr(cfg, "learn_centers", True))
        self.alpha_slow.requires_grad_(self._learn_slow_gate and not self._gate_frozen)
        self.alpha_fast.requires_grad_(self._learn_fast_gate and not self._gate_frozen)

    def _append_with_single_prediction(self, flat_hist: torch.Tensor) -> torch.Tensor:
        """Pad by appending the predictions returned by the learned padder."""
        assert self._padder is not None
        pad_seq = self._padder(flat_hist)
        if pad_seq.ndim == 2:
            pad_seq = pad_seq.unsqueeze(1)
        pad_len = pad_seq.size(1)
        if pad_len == 1:
            pad_seq = pad_seq.expand(-1, self.cfg.pred_length, -1)
        elif pad_len != self.cfg.pred_length:
            raise ValueError(
                f"Padder returned {pad_len} frames but expected {self.cfg.pred_length}."
            )
        return torch.cat([flat_hist, pad_seq], dim=1)

    def _compute_spline_gates(self, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build per-band gates δ^f, δ^c with a spline basis expansion over frequency.
        Returns tensors of shape (1, K, 1) suitable for broadcasting over (B, K, F).
        """
        basis = self._gate_basis.to(device=device, dtype=dtype)  # (K, M)
        alpha_slow_eff = self.alpha_slow #* self._slow_alpha_mask.to(device=device, dtype=dtype)
        alpha_fast_eff = self.alpha_fast #* self._fast_alpha_mask.to(device=device, dtype=dtype)

        slow_logits = torch.matmul(basis, alpha_slow_eff)
        fast_logits = torch.matmul(basis, alpha_fast_eff)

        delta_slow = slow_logits.view(1, -1, 1)
        delta_fast = fast_logits.view(1, -1, 1)
        return delta_slow, delta_fast

    # ------------------------------------------------------------------ Forward
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 4:
            raise ValueError("history must be shaped (B, T_in, N, 3)")
        B, T_in, N, C = history.shape
        if T_in != self.cfg.input_length:
            raise ValueError(f"Expected {self.cfg.input_length} input frames, got {T_in}.")
        if N != self.cfg.num_nodes or C != 3:
            raise ValueError("Input history must have N=num_nodes and coordinate dim=3.")

        Fdim = N * 3
        flat = history.reshape(B, T_in, Fdim)

        # Padding: learned single-frame prediction or fallback
        if self.cfg.pred_length > 0 and self._padder is not None:
            padded = self._append_with_single_prediction(flat)

        dct = self._dct_keep.unsqueeze(0).to(dtype=padded.dtype, device=padded.device)  # (1, K, T)
        coeffs = torch.matmul(dct, padded)  # (B, K, F)

        # Learned gates over frequencies
        delta_f, delta_c = self._compute_spline_gates(coeffs.device, coeffs.dtype)
        self.last_gate_slow = delta_f.detach().cpu()
        self.last_gate_fast = delta_c.detach().cpu()

        # Split coefficients
        coeffs_slow = delta_f * coeffs
        coeffs_fast = delta_c * coeffs


        pred_slow = self.slow_branch(coeffs_slow)   # (B, K, F)
        pred_fast = self.fast_branch(coeffs_fast)   # (B, K, F)

        # Temperatured logistic mask to fuse branches, per-band
        temp = self.mask_temperature.to(device=coeffs.device, dtype=coeffs.dtype).clamp(min=1e-3)
        m = torch.sigmoid(self.mask_logits.to(coeffs.device, coeffs.dtype) / temp).view(1, -1, 1)


        fused_coeffs = m * pred_slow + (1 - m) * pred_fast
        self.last_mask = m.detach().cpu()

        # iDCT with kept bands
        idct = self._idct_keep.unsqueeze(0).to(dtype=fused_coeffs.dtype, device=fused_coeffs.device)  # (1, T, K)
        recon = torch.matmul(idct, fused_coeffs)    # (B, T_total, F)

        # Slice future and reshape
        future = recon[:, -self.cfg.pred_length:, :].reshape(B, self.cfg.pred_length, N, 3)

        if self.cfg.add_last_offset:
            future = future + history[:, -1:, :, :]
        return future



__all__ = ["SFgDNetTryConfig", "SFgDNetTryForecaster"]
