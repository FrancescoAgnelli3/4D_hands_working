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
from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple

import torch
from torch import nn

from config import MODEL_DEFAULT_CONFIGS


def _sfgdnet_try_default(key: str, fallback):
    defaults = MODEL_DEFAULT_CONFIGS.get("sfgdnet_try") or {}
    return defaults.get(key, fallback)


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


def _build_hard_partition_mask(num_freq: int, num_gates: int) -> torch.Tensor:
    if num_freq <= 0 or num_gates <= 0:
        raise ValueError("num_freq and num_gates must be positive for hard partition routing.")
    mask = torch.zeros(num_freq, num_gates, dtype=torch.float32)
    edges = torch.linspace(0, num_freq, steps=num_gates + 1)
    last_end = 0
    for gate_idx in range(num_gates):
        start = int(round(float(edges[gate_idx].item())))
        end = int(round(float(edges[gate_idx + 1].item())))
        start = max(last_end, min(start, num_freq - 1))
        if gate_idx == num_gates - 1:
            end = num_freq
        end = max(end, start + 1)
        if end > num_freq:
            end = num_freq
        mask[start:end, gate_idx] = 1.0
        last_end = end
    return mask


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
    num_layers: Optional[int] = field(default=_sfgdnet_try_default("num_layers", None))
    dct_components: Optional[int] = field(default=_sfgdnet_try_default("dct_keep_coeffs", None))
    mask_temperature: float = field(default=_sfgdnet_try_default("simlpe_mask_temperature", 1.0))
    slow_num_layers: Optional[int] = field(default=_sfgdnet_try_default("simlpe_slow_layers", None))
    fast_num_layers: Optional[int] = field(default=_sfgdnet_try_default("simlpe_fast_layers", None))
    mid_gate_max: float = field(
        default=_sfgdnet_try_default("simlpe_mid_gate_max", 1.0)
    )  # unused, kept for signature compatibility
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
    """Multi-gate DCT MoE forecaster with spline-based frequency gates.

    History: (B, T_in, N, 3) -> future: (B, T_pred, N, 3).

    Differences vs SFgDNetTryForecaster:
        - Uses G spline-parameterized frequency gates instead of two (slow/fast).
        - Uses G SiMLPe experts (ModuleList) on gated DCT coefficients.
        - Fuses experts with the same continuous gates (no top-k).
        - Provides a simple L1 regularizer on average gate usage per expert.
    """
    def __init__(self, cfg: "SFgDNetTryConfig") -> None:
        super().__init__()
        self.cfg = cfg
        T = cfg.total_length

        # DCT basis (keep K)
        K = cfg.dct_length
        dct = _build_dct_matrix(T)  # (T, T)
        idct = dct.t()              # (T, T)
        self.register_buffer("_dct_keep", dct[:K, :], persistent=False)   # (K, T)
        self.register_buffer("_idct_keep", idct[:, :K], persistent=False) # (T, K)
        self.full_seq_len = T
        self.dct_len = K

        # Number of experts/gates
        self.num_gates: int = int(getattr(cfg, "num_gates", 4))  # keep small (e.g. 3–4)
        routing_raw = getattr(cfg, "routing", "soft")
        routing_mode = str(routing_raw).lower()
        if routing_mode not in {"soft", "top1", "hard_partition"}:
            raise ValueError(f"Unknown routing mode '{routing_raw}'. Expected 'soft', 'top1', or 'hard_partition'.")
        self.routing_mode = routing_mode
        if routing_mode == "hard_partition":
            partition_mask = _build_hard_partition_mask(K, self.num_gates)
        else:
            partition_mask = torch.zeros(0, 0, dtype=torch.float32)
        self.register_buffer("_hard_partition_mask", partition_mask, persistent=False)

        # Frequency positions in [0,1]
        freq_pos = torch.linspace(0.0, 1.0, K)
        self.register_buffer("_freq_pos", freq_pos, persistent=False)

        # Spline basis over frequency
        gate_degree = max(0, int(cfg.gate_basis_degree))
        gate_basis = cfg.gate_basis_size if cfg.gate_basis_size is not None else min(K, 32)
        gate_basis = max(gate_degree + 1, int(gate_basis))
        basis = _build_bspline_basis(freq_pos, gate_basis, gate_degree)   # (K, M)
        self.register_buffer("_gate_basis", basis, persistent=False)
        self.gate_basis_size = gate_basis
        self.gate_degree = gate_degree

        # Spline weights: one column per expert
        # alpha: (M, G); gates_k,g = sigmoid( (basis @ alpha)_k,g )
        self.alpha = nn.Parameter(torch.ones(gate_basis, self.num_gates))

        # For inspection
        self.last_gates: Optional[torch.Tensor] = None   # (1, K, G)
        self.last_coeffs: Optional[torch.Tensor] = None  # (B, K, F)

        # Branch (expert) depths
        expert_layers = getattr(cfg, "num_layers", 6)

        # G experts over K × F sequences
        self.branches = nn.ModuleList([
            SiMLPeBackbone(cfg, seq_len_override=K, num_layers_override=expert_layers)
            for _ in range(self.num_gates)
        ])

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

        # Runtime controls / debug
        self._skip_dct: bool = False

    # ------------------------------------------------------------------ Controls
    def set_skip_dct(self, skip: bool = True) -> None:
        self._skip_dct = bool(skip)

    # ------------------------------------------------------------------ Padding
    def _append_with_single_prediction(self, flat_hist: torch.Tensor) -> torch.Tensor:
        """Pad by appending the predictions returned by the learned padder."""
        assert self._padder is not None
        pad_seq = self._padder(flat_hist)  # (B, 1 or T_pred, F)
        if pad_seq.ndim == 2:
            pad_seq = pad_seq.unsqueeze(1)
        pad_len = pad_seq.size(1)
        if pad_len == 1:
            pad_seq = pad_seq.expand(-1, self.cfg.pred_length, -1)
        elif pad_len != self.cfg.pred_length:
            raise ValueError(
                f"Padder returned {pad_len} frames but expected {self.cfg.pred_length}."
            )
        return torch.cat([flat_hist, pad_seq], dim=1)  # (B, T_total, F)

    # ----------------------------------------------------------- Spline gates
    def _compute_spline_gates(self, device, dtype) -> torch.Tensor:
        """
        Build per-band, per-expert gates G_k,g with a spline basis expansion.

        Returns:
            gates: (1, K, G) in [0, 1], suitable for broadcasting over (B, K, F).
        """
        basis = self._gate_basis.to(device=device, dtype=dtype)   # (K, M)
        alpha = self.alpha.to(device=device, dtype=dtype)         # (M, G)
        logits = torch.matmul(basis, alpha)                                    # (K, G)
        gates = torch.sigmoid(logits)                             # (K, G)
        gates = gates.view(1, self.dct_len, self.num_gates)       # (1, K, G)
        return self._apply_routing(gates)

    def _apply_routing(self, gates: torch.Tensor) -> torch.Tensor:
        if self.routing_mode == "soft":
            return gates
        if self.routing_mode == "top1":
            top_idx = gates.argmax(dim=-1, keepdim=True)
            hard = torch.zeros_like(gates)
            hard.scatter_(-1, top_idx, 1.0)
            return hard
        if self.routing_mode == "hard_partition":
            mask = self._hard_partition_mask
            if mask.numel() == 0:
                return gates
            mask = mask.to(device=gates.device, dtype=gates.dtype)  # (K, G)
            return mask.unsqueeze(0)
        return gates

    # --------------------------------------------------------- L1 regularizer
    def gate_usage_l1(self, lambda_l1: float) -> torch.Tensor:
        """
        L1 regularizer on average gate usage per expert.

        Usage:
            loss = main_loss + model.gate_usage_l1(lambda_l1=1e-3)

        Returns:
            scalar tensor on same device as alpha.
        """
        gates = self._compute_spline_gates(self.alpha.device, self.alpha.dtype)  # (1, K, G)
        usage = gates.mean(dim=(0, 1))  # (G,)
        return lambda_l1 * usage.abs().sum()

    def gate_diversity_loss(self, lambda_div: float) -> torch.Tensor:
        """
        Penalize similarity between experts' gates over frequency.

        Returns a scalar tensor; add it to your loss as:
            loss = main_loss + model.gate_diversity_loss(lambda_div)
        """
        # Use current gates over K frequencies and G experts
        gates = self._compute_spline_gates(self.alpha.device, self.alpha.dtype)  # (1, K, G)
        gates = gates.squeeze(0)   # (K, G)

        # Normalize per-expert gate vectors
        gates = gates / (gates.norm(dim=0, keepdim=True) + 1e-8)  # (K, G)

        # Gram matrix over experts: G x G, diag ~ 1, off-diag are cos-sim
        gram = gates.t() @ gates  # (G, G)

        # We want gram ≈ I -> off-diagonal entries → 0
        G = gram.shape[0]
        eye = torch.eye(G, device=gram.device, dtype=gram.dtype)
        off_diag = gram - eye

        return lambda_div * (off_diag.pow(2).sum())


    # ------------------------------------------------------------------ Forward
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        history: (B, T_in, N, 3)
        returns:
            future: (B, T_pred, N, 3)
        """
        if history.ndim != 4:
            raise ValueError("history must be shaped (B, T_in, N, 3)")
        B, T_in, N, C = history.shape
        if T_in != self.cfg.input_length:
            raise ValueError(f"Expected {self.cfg.input_length} input frames, got {T_in}.")
        if N != self.cfg.num_nodes or C != 3:
            raise ValueError("Input history must have N=num_nodes and coordinate dim=3.")

        Fdim = N * 3
        flat = history.reshape(B, T_in, Fdim)  # (B, T_in, F)

        # Padding: learned single-frame prediction or fallback
        if self.cfg.pred_length > 0 and self._padder is not None:
            padded = self._append_with_single_prediction(flat)  # (B, T_total, F)
        else:
            # repeat last frame if no MLP padder
            if self.cfg.pred_length > 0:
                last = flat[:, -1:, :].expand(-1, self.cfg.pred_length, -1)
                padded = torch.cat([flat, last], dim=1)
            else:
                padded = flat

        if self._skip_dct:
            # Optional debug path: no DCT, just operate in time domain
            # Still route through experts for consistency
            seq = padded  # (B, T_total, F)
            T_total = seq.size(1)
            if T_total != self.dct_len:
                raise ValueError("skip_dct=True requires T_total == dct_len.")
            coeffs = seq  # treat as (B, K, F)
        else:
            # DCT over time dimension
            dct = self._dct_keep.unsqueeze(0).to(dtype=padded.dtype, device=padded.device)  # (1, K, T)
            coeffs = torch.matmul(dct, padded)  # (B, K, F)

        self.last_coeffs = coeffs.detach().cpu()

        # Compute spline gates over frequencies and experts
        gates = self._compute_spline_gates(coeffs.device, coeffs.dtype)  # (1, K, G)
        self.last_gates = gates.detach().cpu()

        # Prepare for expert-wise gating
        # coeffs: (B, K, F) -> (B, K, F, 1)
        coeffs_exp = coeffs.unsqueeze(-1)               # (B, K, F, 1)
        gates_exp = gates.unsqueeze(2)                  # (1, K, 1, G)
        coeffs_gated = coeffs_exp * gates_exp          # (B, K, F, G)

        # Run each expert on its gated coefficients
        preds = []
        for g in range(self.num_gates):
            input_g = coeffs_gated[..., g]             # (B, K, F)
            preds_g = self.branches[g](input_g)        # (B, K, F)
            preds.append(preds_g)

        preds = torch.stack(preds, dim=-1)             # (B, K, F, G)

        # # Fuse experts using the same continuous gates
        fused_coeffs = (preds * gates_exp).sum(dim=-1)  # (B, K, F)

        # Fuse experts using normalized mask
        # fused_coeffs = preds * 

        # iDCT with kept bands (if not skipping DCT)
        if not self._skip_dct:
            idct = self._idct_keep.unsqueeze(0).to(
                dtype=fused_coeffs.dtype,
                device=fused_coeffs.device,
            )  # (1, T, K)
            recon = torch.matmul(idct, fused_coeffs)   # (B, T_total, F)
        else:
            recon = fused_coeffs  # already in time domain

        # Slice future and reshape
        future = recon[:, -self.cfg.pred_length:, :].reshape(B, self.cfg.pred_length, N, 3)

        if self.cfg.add_last_offset:
            future = future + history[:, -1:, :, :]

        return future



__all__ = ["SFgDNetTryConfig", "SFgDNetTryForecaster"]
