"""
Multi-rate spectral siMLPe with per-scale Wavelet experts (no gates).
Public API/signature unchanged: same class names and ctor signatures.

Key change vs the DCT+gates variant:
- Replace DCT/IDCT + spline gates with DWT/IDWT + hard per-scale routing.
- Use separate SiMLPeBackbone per wavelet subband (approx + each detail scale).

Dependency:
    pip install pytorch-wavelets
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple, List

import torch
from torch import nn

from config import MODEL_DEFAULT_CONFIGS

try:
    from pytorch_wavelets import DWT1DForward, DWT1DInverse
except Exception as e:
    DWT1DForward = None  # type: ignore
    DWT1DInverse = None  # type: ignore
    _PWT_IMPORT_ERROR = e
else:
    _PWT_IMPORT_ERROR = None


def _WaveNet_default(key: str, fallback):
    defaults = MODEL_DEFAULT_CONFIGS.get("WaveNet") or {}
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
class SiMLPeWaveletConfig(SiMLPeConfig):
    """
    Kept class name/signature for compatibility.
    In this variant, DCT+gates are replaced by DWT per-scale routing.
    """
    num_layers: Optional[int] = field(default=_WaveNet_default("num_layers", None))
    dct_components: Optional[int] = field(default=_WaveNet_default("dct_keep_coeffs", None))  # unused, kept
    mask_temperature: float = field(default=_WaveNet_default("simlpe_mask_temperature", 1.0))  # unused
    slow_num_layers: Optional[int] = field(default=_WaveNet_default("simlpe_slow_layers", None))  # unused
    fast_num_layers: Optional[int] = field(default=_WaveNet_default("simlpe_fast_layers", None))  # unused
    mid_gate_max: float = field(default=_WaveNet_default("simlpe_mid_gate_max", 1.0))  # unused

    gate_basis_size: Optional[int] = None          # unused
    gate_basis_degree: int = 3                     # unused
    learn_lambda: bool = True                      # unused
    learn_centers: bool = True                     # unused
    fixed_lambda: Optional[float] = None           # unused
    fixed_centers: Optional[Tuple[float, float]] = None  # unused

    padding_mode: Optional[Literal["mlp", "repeat_last"]] = None
    padding_predictor: Literal["mlp", "rnn"] = "mlp"
    padding_full_horizon: bool = False

    # New fields (optional): wavelet settings
    num_scales: Optional[int] = None
    wavelet: str = "db2"
    wavelet_mode: str = "symmetric"

    @property
    def dct_length(self) -> int:
        # kept for compatibility; not used by the DWT variant
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
    def __init__(
        self,
        cfg: SiMLPeConfig,
        seq_len_override: Optional[int] = None,
        num_layers_override: Optional[int] = None,
    ) -> None:
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
        # seq: (B, T, F)
        x = self.fc_in(seq)
        x = x.transpose(1, 2).contiguous()  # (B, C, T)
        x = self.mlp(x)
        x = x.transpose(1, 2).contiguous()  # (B, T, C)
        return self.fc_out(x)


# ------------------------------ DWT Forecaster -------------------------------

class SiMLPeWaveletForecaster(nn.Module):
    """
    Wavelet multi-scale forecaster (no gates).

    History: (B, T_in, N, 3) -> future: (B, T_pred, N, 3).

    Pipeline:
      1) (optional) learned padding in time domain -> padded sequence length T_total
      2) DWT along time for each feature channel (treat F as channels)
      3) One SiMLPe backbone per subband (cA_J and each cD_j)
      4) IDWT -> reconstructed sequence
      5) slice future, reshape, add last offset (optional)
    """

    def __init__(self, cfg: "SiMLPeWaveletConfig") -> None:
        super().__init__()
        self.cfg = cfg
        self.full_seq_len = int(cfg.total_length)

        if DWT1DForward is None or DWT1DInverse is None:
            raise ImportError(
                "pytorch-wavelets is required for the DWT variant. "
                "Install with: pip install pytorch-wavelets"
            ) from _PWT_IMPORT_ERROR

        # Wavelet parameters
        J = getattr(cfg, "num_scales", None)
        if J is None:
            # conservative default: 3 scales unless very short sequences
            J = 3 if self.full_seq_len >= 16 else max(1, int(math.log2(max(2, self.full_seq_len))) - 2)
        self.num_scales = int(max(1, J))
        self.wavelet = str(getattr(cfg, "wavelet", "db2"))
        self.wavelet_mode = str(getattr(cfg, "wavelet_mode", "symmetric"))

        # DWT modules (operate on (B, C, L))
        self.dwt = DWT1DForward(J=self.num_scales, wave=self.wavelet, mode=self.wavelet_mode)
        self.idwt = DWT1DInverse(wave=self.wavelet, mode=self.wavelet_mode)

        # Infer subband lengths once (CPU) for constructing per-band backbones
        band_lens = self._infer_band_lengths(
            T=self.full_seq_len,
            F=int(cfg.feature_dim),
            J=self.num_scales,
        )
        # band_lens: [len(cA_J), len(cD_1), ..., len(cD_J)] as produced by pytorch_wavelets
        # We will build branches aligned with [cA_J] + details list order.
        self._band_lens: List[int] = band_lens
        self.num_bands: int = 1 + self.num_scales  # approx + J details

        # Experts/backbones per band
        expert_layers = getattr(cfg, "num_layers", 6)
        self.branches = nn.ModuleList([
            SiMLPeBackbone(cfg, seq_len_override=self._band_lens[band_idx], num_layers_override=expert_layers)
            for band_idx in range(self.num_bands)
        ])

        # Padding mode
        pad_mode_raw = getattr(cfg, "padding_mode", None)
        pad_mode = "mlp" if pad_mode_raw is None else str(pad_mode_raw).lower()
        if pad_mode not in {"mlp", "repeat_last"}:
            raise ValueError(f"Unknown padding_mode='{pad_mode_raw}'. Expected one of ('mlp','repeat_last').")
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

        # Optional debug path: skip wavelet transform (expects T_total == len(cA_J), not recommended)
        self._skip_dwt: bool = False

        # For inspection/debug
        self.last_bands: Optional[List[torch.Tensor]] = None  # list of (B, band_len, F) on CPU

    def set_skip_dwt(self, skip: bool = True) -> None:
        self._skip_dwt = bool(skip)

    def _infer_band_lengths(self, T: int, F: int, J: int) -> List[int]:
        with torch.no_grad():
            x = torch.zeros(1, F, T, dtype=torch.float32)  # (B, C, L)
            yl, yh = self.dwt(x)
            # yl: (1, F, L_a), yh: list length J, each (1, F, L_d_j)
            lens = [int(yl.shape[-1])]
            for j in range(J):
                lens.append(int(yh[j].shape[-1]))
            return lens

    def _append_with_single_prediction(self, flat_hist: torch.Tensor) -> torch.Tensor:
        """Pad by appending the predictions returned by the learned padder."""
        assert self._padder is not None
        pad_seq = self._padder(flat_hist)  # (B, 1 or T_pred, F) or (B, F) for single-step
        if pad_seq.ndim == 2:
            pad_seq = pad_seq.unsqueeze(1)
        pad_len = pad_seq.size(1)
        if pad_len == 1:
            pad_seq = pad_seq.expand(-1, self.cfg.pred_length, -1)
        elif pad_len != self.cfg.pred_length:
            raise ValueError(f"Padder returned {pad_len} frames but expected {self.cfg.pred_length}.")
        return torch.cat([flat_hist, pad_seq], dim=1)  # (B, T_total, F)

    def _check_lengths_match(self, yl: torch.Tensor, yh: List[torch.Tensor]) -> None:
        # Ensure runtime lengths are consistent with branches
        if yl.shape[-1] != self._band_lens[0]:
            raise ValueError(
                f"Approx band length mismatch: got {yl.shape[-1]} expected {self._band_lens[0]}. "
                f"Check wavelet mode or sequence length."
            )
        if len(yh) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} detail bands, got {len(yh)}.")
        for j in range(self.num_scales):
            exp = self._band_lens[1 + j]
            got = int(yh[j].shape[-1])
            if got != exp:
                raise ValueError(
                    f"Detail band {j} length mismatch: got {got} expected {exp}. "
                    f"Check wavelet mode or sequence length."
                )

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

        # Padding in time domain
        if self.cfg.pred_length > 0 and self._padder is not None:
            padded = self._append_with_single_prediction(flat)  # (B, T_total, F)
        else:
            if self.cfg.pred_length > 0:
                last = flat[:, -1:, :].expand(-1, self.cfg.pred_length, -1)
                padded = torch.cat([flat, last], dim=1)
            else:
                padded = flat

        if padded.size(1) != self.full_seq_len:
            raise ValueError(
                f"Internal error: padded length {padded.size(1)} != cfg.total_length {self.full_seq_len}."
            )

        if self._skip_dwt:
            # Debug path: treat padded as a single band and apply only branch[0]
            # Requires branch[0] seq_len_override == full_seq_len; typically false.
            if self._band_lens[0] != self.full_seq_len:
                raise ValueError(
                    "skip_dwt=True requires approx band length == full_seq_len. "
                    "This is not generally true for DWT. Disable skip_dwt."
                )
            recon = self.branches[0](padded)  # (B, T_total, F)
        else:
            # Prepare for DWT: (B, T, F) -> (B, F, T)
            x = padded.transpose(1, 2).contiguous()

            # DWT: yl approx, yh list of details
            yl, yh = self.dwt(x)
            self._check_lengths_match(yl, yh)

            # Process per band with separate backbones.
            # Backbone expects (B, band_len, F), so transpose each band (B, F, L) -> (B, L, F).
            # Branch index mapping:
            #   0: approx (yl)
            #   1..J: details in the order returned by pytorch_wavelets (yh[0], ..., yh[J-1])
            yl_t = yl.transpose(1, 2).contiguous()           # (B, L_a, F)
            yl_t = self.branches[0](yl_t)                    # (B, L_a, F)
            yl_p = yl_t.transpose(1, 2).contiguous()         # (B, F, L_a)

            yh_p: List[torch.Tensor] = []
            for j in range(self.num_scales):
                yj = yh[j].transpose(1, 2).contiguous()      # (B, L_d, F)
                yj = self.branches[1 + j](yj)                # (B, L_d, F)
                yj = yj.transpose(1, 2).contiguous()         # (B, F, L_d)
                yh_p.append(yj)

            # IDWT: reconstruct (B, F, T_total) then transpose back
            recon_x = self.idwt((yl_p, yh_p))                # (B, F, T_total) (should match padded length)
            if recon_x.shape[-1] != self.full_seq_len:
                raise ValueError(
                    f"IDWT length mismatch: got {recon_x.shape[-1]} expected {self.full_seq_len}. "
                    f"Try a different wavelet_mode (e.g., 'periodization' or 'symmetric')."
                )
            recon = recon_x.transpose(1, 2).contiguous()     # (B, T_total, F)

            # Save bands for inspection (CPU) if needed
            # Note: store time-major (B, L, F)
            self.last_bands = [yl_t.detach().cpu()] + [t.detach().cpu() for t in
                                                       [yh[j].transpose(1, 2).contiguous() for j in range(self.num_scales)]]

        # Slice future and reshape
        future = recon[:, -self.cfg.pred_length:, :].reshape(B, self.cfg.pred_length, N, 3)

        if self.cfg.add_last_offset:
            future = future + history[:, -1:, :, :]

        return future


__all__ = [
    "SiMLPeWaveletConfig",
    "SiMLPeWaveletForecaster",
]
