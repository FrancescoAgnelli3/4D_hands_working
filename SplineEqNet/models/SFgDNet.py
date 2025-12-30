"""Multi-rate spectral siMLPe with a LEARNED SOFTMAX split over DCT coefficients.
Public API/signature unchanged: same class names and ctor signatures.

Padding uses an internal learnable prediction module that, by default, runs the
compact MLP fallback: a single future frame is predicted from the observed
sequence and then repeated t_out times. When explicitly enabled, the RNN
padder replaces the MLP; disabling learned padding entirely falls back to
last-frame repetition.

Slow/fast band split:
    Let positions p_k ∈ [0,1] index the kept DCT rows.
    Define logits with scale λ > 0 and learned centers c_f, c_c and per-band bias b_k:
        L_f(k) = -λ * |p_k - c_f| + b_k
        L_c(k) = -λ * |p_k - c_c| - b_k
    Distributions:
        s_f = softmax(L_f), s_c = softmax(L_c)
    Complementary gates:
        δ^f_k = s_f(k) / (s_f(k) + s_c(k))
        δ^c_k = 1 - δ^f_k
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import torch
from torch import nn
import torch.nn.functional as F


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
class SFgDNetConfig(SiMLPeConfig):
    """siMLPe with learned soft band partition in the DCT domain."""
    dct_components: Optional[int] = 100
    mask_temperature: float = 1
    slow_num_layers: Optional[int] = 6
    fast_num_layers: Optional[int] = 6
    mid_gate_max: float = 1  # unused, kept for signature compatibility
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


# ------------------------------ Forecaster -----------------------------------

def _logit_eps(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x) - torch.log1p(-x)

class SFgDNetForecaster(nn.Module):
    """Two-branch siMLPe with learned SOFTMAX/INV-SOFTMAX gates over kept DCT rows.
    Signature: __init__(cfg), forward(history) -> future.

    Padding:
        By default, compute padding with the learned predictor (repeat single
        frame or full horizon) and fall back to the last frame if
        cfg.padding_mode is ``repeat_last``.
    """
    def __init__(self, cfg: "SFgDNetConfig") -> None:
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

        # Gating parameters: centers constrained to [0,1] via sigmoid of raw params, λ>0 via softplus
        self.center_slow_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5
        self.center_fast_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5
        self.band_bias = nn.Parameter(torch.zeros(1))           # reserved if you re-enable band-wise bias
        init_lambda = 1.0 / max(1e-3, float(cfg.mask_temperature))
        self.lambda_raw = nn.Parameter(torch.tensor(init_lambda))  # λ = softplus(lambda_raw)

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
        self._fixed_lambda_value: Optional[float] = None
        self._fixed_centers_value: Optional[Tuple[float, float]] = None
        self._use_slow_branch: bool = True
        self._use_fast_branch: bool = True
        self._skip_dct: bool = False
        self._gate_frozen: bool = False

        # Learning flags for gate parameters
        self._learn_lambda: bool = bool(getattr(cfg, "learn_lambda", True))
        self._learn_centers: bool = bool(getattr(cfg, "learn_centers", True))
        self.lambda_raw.requires_grad_(self._learn_lambda)
        self.center_slow_raw.requires_grad_(self._learn_centers)
        self.center_fast_raw.requires_grad_(self._learn_centers)

        # Apply any fixed values from cfg
        if cfg.fixed_lambda is not None:
            self.set_fixed_lambda(cfg.fixed_lambda, freeze=not self._learn_lambda)
        if cfg.fixed_centers is not None:
            slow_c, fast_c = cfg.fixed_centers
            self.set_fixed_centers(slow_c, fast_c, freeze=not self._learn_centers)

    # ------------------------------------------------------------------ Controls
    def set_fixed_mask(self, value: Optional[float], freeze: bool = True) -> None:
        if value is None:
            self._fixed_mask_value = None
            if not self._gate_frozen:
                self.mask_logits.requires_grad_(True)
        else:
            self._fixed_mask_value = float(value)
            if freeze:
                self.mask_logits.requires_grad_(False)

    def set_fixed_mid_gate(self, value: Optional[float], freeze: bool = True) -> None:
        if value is None:
            self._fixed_mid_gate_value = None
            if not self._gate_frozen:
                self.mask_logits.requires_grad_(True)
        else:
            self._fixed_mid_gate_value = float(value)
            if freeze:
                self.mask_logits.requires_grad_(False)

    def set_branch_usage(self, use_slow: bool, use_fast: bool) -> None:
        self._use_slow_branch = bool(use_slow)
        self._use_fast_branch = bool(use_fast)

    def set_skip_dct(self, skip: bool = True) -> None:
        self._skip_dct = bool(skip)

    def set_fixed_lambda(self, value: Optional[float], freeze: bool = True) -> None:
        if value is None:
            self._fixed_lambda_value = None
            allow_grad = self._learn_lambda and not (freeze or self._gate_frozen)
            self.lambda_raw.requires_grad_(allow_grad)
            return

        lam = max(1e-6, float(value))
        self._fixed_lambda_value = lam
        raw_value = _softplus_inverse(lam)
        tensor_value = self.lambda_raw.data.new_tensor(raw_value)
        self.lambda_raw.data.copy_(tensor_value)
        allow_grad = (not freeze) and self._learn_lambda and not self._gate_frozen
        self.lambda_raw.requires_grad_(allow_grad)

    def set_fixed_centers(self, slow: float, fast: float, freeze: bool = True) -> None:
        # values are clipped into [0,1] to avoid invalid requests
        slow = float(min(1.0, max(0.0, slow)))
        fast = float(min(1.0, max(0.0, fast)))
        self._fixed_centers_value = (slow, fast)

        # keep raw params consistent for logging or later unfreezing
        slow_raw = _logit_eps(torch.tensor(slow, dtype=self.center_slow_raw.dtype, device=self.center_slow_raw.device))
        fast_raw = _logit_eps(torch.tensor(fast, dtype=self.center_fast_raw.dtype, device=self.center_fast_raw.device))
        self.center_slow_raw.data.copy_(slow_raw)
        self.center_fast_raw.data.copy_(fast_raw)

        allow_grad = (not freeze) and self._learn_centers and not self._gate_frozen
        self.center_slow_raw.requires_grad_(allow_grad)
        self.center_fast_raw.requires_grad_(allow_grad)

    def set_gate_learning(self, *, learn_lambda: Optional[bool] = None, learn_centers: Optional[bool] = None) -> None:
        if learn_lambda is not None:
            self._learn_lambda = bool(learn_lambda)
            allow_grad = self._learn_lambda and not self._gate_frozen and self._fixed_lambda_value is None
            self.lambda_raw.requires_grad_(allow_grad)
        if learn_centers is not None:
            self._learn_centers = bool(learn_centers)
            allow_grad = self._learn_centers and not self._gate_frozen and self._fixed_centers_value is None
            self.center_slow_raw.requires_grad_(allow_grad)
            self.center_fast_raw.requires_grad_(allow_grad)

    def freeze_frequency_gate(self, freeze: bool = True) -> None:
        self._gate_frozen = bool(freeze)
        allow_grad_common = not self._gate_frozen
        self.mask_logits.requires_grad_(allow_grad_common)
        self.mask_temperature.requires_grad_(allow_grad_common)
        # self.band_bias.requires_grad_(allow_grad_common)  # keep disabled unless used

        allow_lambda_grad = allow_grad_common and self._learn_lambda and self._fixed_lambda_value is None
        allow_center_grad = allow_grad_common and self._learn_centers and self._fixed_centers_value is None
        self.lambda_raw.requires_grad_(allow_lambda_grad)
        self.center_slow_raw.requires_grad_(allow_center_grad)
        self.center_fast_raw.requires_grad_(allow_center_grad)

    # ------------------------------------------------------------------ Internals
    def _append_repeat_last(self, flat_hist: torch.Tensor) -> torch.Tensor:
        """Pad by repeating the last observed frame."""
        last_frame = flat_hist[:, -1:, :].expand(-1, self.cfg.pred_length, -1)  # (B, T_out, F)
        return torch.cat([flat_hist, last_frame], dim=1)                         # (B, T_total, F)

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

    def _compute_softmax_gates(self, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build complementary gates δ^f, δ^c over K bands using logistic functions.
        Returns tensors of shape (1, K, 1) suitable for broadcasting over (B, K, F).
        """
        p = self._freq_pos.to(device=device, dtype=dtype)  # (K,)

        # λ>0
        if self._fixed_lambda_value is not None:
            lam = torch.tensor(self._fixed_lambda_value, device=device, dtype=dtype)
        else:
            lam = F.softplus(self.lambda_raw.to(device=device, dtype=dtype)) + 1e-6

        # centers in [0,1]
        if self._fixed_centers_value is not None:
            self.center_slow = torch.tensor(self._fixed_centers_value[0], device=device, dtype=dtype)
            self.center_fast = torch.tensor(self._fixed_centers_value[1], device=device, dtype=dtype)
        else:
            self.center_slow = torch.sigmoid(self.center_slow_raw.to(device=device, dtype=dtype))
            self.center_fast = torch.sigmoid(self.center_fast_raw.to(device=device, dtype=dtype))

        x_slow = lam * (self.center_slow - p)
        x_fast = lam * (self.center_fast - p)

        L_f = torch.sigmoid(x_slow).view(1, -1, 1)      # slow gate in [0,1]
        L_c = 1.0 - torch.sigmoid(x_fast).view(1, -1, 1)  # fast gate in [0,1], complementary

        return L_f, L_c

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
        else:
            padded = self._append_repeat_last(flat)

        # DCT → keep K rows
        if self._skip_dct:
            coeffs = padded
        else:
            dct = self._dct_keep.unsqueeze(0).to(dtype=padded.dtype, device=padded.device)  # (1, K, T)
            coeffs = torch.matmul(dct, padded)  # (B, K, F)

        # Learned gates over frequencies
        delta_f, delta_c = self._compute_softmax_gates(coeffs.device, coeffs.dtype)
        self.last_gate_slow = delta_f.detach().cpu()
        self.last_gate_fast = delta_c.detach().cpu()

        # Split coefficients
        coeffs_slow = delta_f * coeffs
        coeffs_fast = delta_c * coeffs

        # Per-branch processing
        if self._use_slow_branch:
            pred_slow = self.slow_branch(coeffs_slow)   # (B, K, F)
        else:
            pred_slow = torch.zeros_like(coeffs_slow)
        if self._use_fast_branch:
            pred_fast = self.fast_branch(coeffs_fast)   # (B, K, F)
        else:
            pred_fast = torch.zeros_like(coeffs_fast)

        # Temperatured logistic mask to fuse branches, per-band
        temp = self.mask_temperature.to(device=coeffs.device, dtype=coeffs.dtype).clamp(min=1e-3)
        m = torch.sigmoid(self.mask_logits.to(coeffs.device, coeffs.dtype) / temp).view(1, -1, 1)

        if self._fixed_mask_value is not None:
            m = torch.full_like(m, float(self._fixed_mask_value))
        elif self._fixed_mid_gate_value is not None:
            m = torch.full_like(m, float(self._fixed_mid_gate_value))

        if not self._use_slow_branch and self._use_fast_branch:
            m = torch.zeros_like(m)
        elif not self._use_fast_branch and self._use_slow_branch:
            m = torch.ones_like(m)

        fused_coeffs = m * pred_slow + (1 - m) * pred_fast
        self.last_mask = m.detach().cpu()

        if self._skip_dct:
            recon = fused_coeffs
        else:
            # iDCT with kept bands
            idct = self._idct_keep.unsqueeze(0).to(dtype=fused_coeffs.dtype, device=fused_coeffs.device)  # (1, T, K)
            recon = torch.matmul(idct, fused_coeffs)    # (B, T_total, F)

        # Slice future and reshape
        future = recon[:, -self.cfg.pred_length:, :].reshape(B, self.cfg.pred_length, N, 3)

        if self.cfg.add_last_offset:
            future = future + history[:, -1:, :, :]
        return future

__all__ = ["SFgDNetConfig", "SFgDNetForecaster"]
