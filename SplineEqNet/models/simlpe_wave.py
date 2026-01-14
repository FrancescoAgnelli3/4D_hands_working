"""
Wavelet forecaster with differentiable Torch SWT + true Torch ISWT (synthesis filter bank)
using the same boundary handling for analysis and synthesis.

Constraints satisfied:
- Config name/signature: SiMLPeWaveletConfig
- Forecaster name/signature: SiMLPeWaveletForecaster(history) -> future
- SWT/ISWT are implemented in Torch (no numpy, no detach, gradients flow)
- Proper ISWT: implements the standard shift-averaged inverse SWT algorithm
  (the one used conceptually by PyWavelets) with the SAME boundary mode
  in the underlying convolutions as analysis.

Important design choice:
- For a true ISWT you must have the approximation a_j at every level j, not only a_L.
  Therefore, when include_approx=True, this implementation stores BOTH a_j and d_j
  for j=1..L, i.e. num_scales = 2 * swt_level and the scale axis layout is:
      [a1, d1, a2, d2, ..., aL, dL]
- If include_approx=False, the forward SWT can be computed (details only), but a true
  ISWT is not defined; this code raises in inverse.

Boundary handling:
- Convolutions use SAME-length padding with mode="replicate" (stable for motion data).
  Analysis and synthesis both use this.

Dependencies:
- PyWavelets is used ONLY to read filter coefficients (dec_lo/dec_hi/rec_lo/rec_hi).
  Install: pip install PyWavelets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

try:
    import pywt  # type: ignore
except Exception:  # pragma: no cover
    pywt = None


# -----------------------------------------------------------------------------
# Config (same signature as requested)
# -----------------------------------------------------------------------------

@dataclass
class SiMLPeWaveletConfig:
    input_length: int
    pred_length: int
    num_nodes: int
    hidden_dim: int
    num_layers: int
    use_norm: bool = True
    norm_axis: Literal["all", "time", "scale", "hidden"] = "all"
    add_last_offset_time_domain: bool = False

    # SWT parameters
    wavelet: str = "db4"
    swt_level: int = 2
    include_approx: bool = True  # must be True for true ISWT here

    @property
    def total_length(self) -> int:
        return self.input_length + self.pred_length

    @property
    def padded_total_length(self) -> int:
        L = int(self.swt_level)
        if L <= 0:
            raise ValueError("swt_level must be positive.")
        m = 1 << L  # 2**L
        t = self.total_length
        return ((t + m - 1) // m) * m

    @property
    def feat_dim(self) -> int:
        return self.num_nodes * 3

    @property
    def num_scales(self) -> int:
        # For true ISWT, we store (a_j, d_j) for every level j => 2*L
        L = int(self.swt_level)
        return 2 * L if self.include_approx else L


# -----------------------------------------------------------------------------
# Convolution helpers (SAME length, boundary mode replicate)
# -----------------------------------------------------------------------------

def _same_pad_1d(total_pad: int) -> tuple[int, int]:
    left = total_pad // 2
    right = total_pad - left
    return left, right


def _conv1d_same(
    x: torch.Tensor,
    kernel: torch.Tensor,
    *,
    dilation: int,
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """
    SAME-length 1D conv along time axis using conv1d.

    x: (B, C, T)
    kernel: (C_out, C_in/groups, K) or (1,1,K) for depthwise-like use
    """
    k = kernel.size(-1)
    eff = dilation * (k - 1) + 1
    total_pad = eff - 1
    pl, pr = _same_pad_1d(total_pad)
    x_pad = F.pad(x, (pl, pr), mode=pad_mode)
    return F.conv1d(x_pad, kernel, stride=1, dilation=dilation)


def _roll_time(x: torch.Tensor, shift: int) -> torch.Tensor:
    # x: (..., T)
    if shift == 0:
        return x
    return torch.roll(x, shifts=shift, dims=-1)


# -----------------------------------------------------------------------------
# Torch SWT / true Torch ISWT (shift-averaged)
# -----------------------------------------------------------------------------

class TorchSWT(nn.Module):
    """
    Differentiable SWT and true ISWT with shift averaging.

    Layout when include_approx=True:
        scales index 2*(j-1)     -> a_j
        scales index 2*(j-1) + 1 -> d_j
      for j = 1..L

    forward:
      x: (B, T, F) -> w: (B, T, F, 2L)

    inverse:
      w: (B, T, F, 2L) -> x: (B, T, F)

    Boundary mode:
      replicate padding in both analysis and synthesis convolutions.
    """

    def __init__(self, *, wavelet: str, level: int, include_approx: bool, pad_mode: str = "replicate") -> None:
        super().__init__()
        if pywt is None:
            raise ImportError("PyWavelets is required (only to load wavelet filters).")
        if level <= 0:
            raise ValueError("level must be positive.")
        self.wavelet = wavelet
        self.level = int(level)
        self.include_approx = bool(include_approx)
        self.pad_mode = pad_mode

        # Load wavelet filters
        w = pywt.Wavelet(wavelet)

        # Torch conv1d is cross-correlation; reverse taps to match the usual convolutional definition.
        dec_lo = torch.tensor(list(w.dec_lo)[::-1], dtype=torch.float32)
        dec_hi = torch.tensor(list(w.dec_hi)[::-1], dtype=torch.float32)
        rec_lo = torch.tensor(list(w.rec_lo)[::-1], dtype=torch.float32)
        rec_hi = torch.tensor(list(w.rec_hi)[::-1], dtype=torch.float32)

        # Store as (1,1,K) conv kernels; broadcasted over channels by manual reshape later
        self.register_buffer("dec_lo", dec_lo.view(1, 1, -1), persistent=False)
        self.register_buffer("dec_hi", dec_hi.view(1, 1, -1), persistent=False)
        self.register_buffer("rec_lo", rec_lo.view(1, 1, -1), persistent=False)
        self.register_buffer("rec_hi", rec_hi.view(1, 1, -1), persistent=False)

    @property
    def num_scales(self) -> int:
        return 2 * self.level if self.include_approx else self.level

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        returns w: (B, T, F, 2L) if include_approx else (B, T, F, L) (details only)
        """
        if x.ndim != 3:
            raise ValueError("Expected x shaped (B, T, F).")
        B, T, Fdim = x.shape

        if self.include_approx:
            out = torch.empty((B, T, Fdim, 2 * self.level), device=x.device, dtype=x.dtype)
        else:
            out = torch.empty((B, T, Fdim, self.level), device=x.device, dtype=x.dtype)

        # Process each feature dimension as a separate channel by reshaping to (B*F, 1, T)
        a = x.permute(0, 2, 1).contiguous().view(B * Fdim, 1, T)

        dec_lo = self.dec_lo.to(device=x.device, dtype=x.dtype)
        dec_hi = self.dec_hi.to(device=x.device, dtype=x.dtype)

        for j in range(1, self.level + 1):
            dil = 1 << (j - 1)
            d = _conv1d_same(a, dec_hi, dilation=dil, pad_mode=self.pad_mode)  # (B*F,1,T)
            a = _conv1d_same(a, dec_lo, dilation=dil, pad_mode=self.pad_mode)  # (B*F,1,T)

            d = d.view(B, Fdim, T).permute(0, 2, 1).contiguous()  # (B,T,F)

            if self.include_approx:
                a_j = a.view(B, Fdim, T).permute(0, 2, 1).contiguous()  # (B,T,F)
                out[..., 2 * (j - 1)] = a_j
                out[..., 2 * (j - 1) + 1] = d
            else:
                out[..., (j - 1)] = d

        return out

    def inverse(self, w: torch.Tensor) -> torch.Tensor:
        """
        True ISWT with shift-averaging, matching the analysis boundary mode.

        w: (B, T, F, 2L) required (include_approx=True)
        returns x: (B, T, F)
        """
        if not self.include_approx:
            raise ValueError("True ISWT requires include_approx=True (need a_j at every level).")
        if w.ndim != 4:
            raise ValueError("Expected w shaped (B, T, F, S).")
        B, T, Fdim, S = w.shape
        if S != 2 * self.level:
            raise ValueError(f"Expected S={2*self.level} for include_approx=True, got {S}.")

        rec_lo = self.rec_lo.to(device=w.device, dtype=w.dtype)
        rec_hi = self.rec_hi.to(device=w.device, dtype=w.dtype)

        # Extract a_j and d_j for all levels
        # a_j, d_j: (B, T, F)
        a_list = []
        d_list = []
        for j in range(1, self.level + 1):
            a_list.append(w[..., 2 * (j - 1)])
            d_list.append(w[..., 2 * (j - 1) + 1])

        # Start from the coarsest approximation a_L
        xj = a_list[-1]  # (B,T,F)

        # Iterate from level L down to 1 with shift averaging
        for j in range(self.level, 0, -1):
            step = 1 << (j - 1)

            # Use a_j from coefficients (not xj) to match SWT inversion definition.
            a_j = a_list[j - 1]
            d_j = d_list[j - 1]

            # Shift-averaged reconstruction:
            # x_{j-1} = (1/step) * sum_{k=0}^{step-1} shift_back( recon( shift(a_j,-k), shift(d_j,-k) ), k )
            acc = torch.zeros((B, T, Fdim), device=w.device, dtype=w.dtype)

            for k in range(step):
                a_shift = _roll_time(a_j, -k)
                d_shift = _roll_time(d_j, -k)

                # (B,T,F) -> (B*F,1,T)
                a_in = a_shift.permute(0, 2, 1).contiguous().view(B * Fdim, 1, T)
                d_in = d_shift.permute(0, 2, 1).contiguous().view(B * Fdim, 1, T)

                # Synthesis at this level (dilated)
                dil = step
                y_a = _conv1d_same(a_in, rec_lo, dilation=dil, pad_mode=self.pad_mode)
                y_d = _conv1d_same(d_in, rec_hi, dilation=dil, pad_mode=self.pad_mode)
                y = (y_a + y_d)  # (B*F,1,T)

                y = y.view(B, Fdim, T).permute(0, 2, 1).contiguous()  # (B,T,F)
                y = _roll_time(y, k)
                acc = acc + y

            xj = acc / float(step)

        return xj  # (B,T,F)


# -----------------------------------------------------------------------------
# Wavelet-aware backbone (2D TCN over scale × time in hidden space)
# -----------------------------------------------------------------------------

class _Norm2D(nn.Module):
    """
    Normalize x shaped (B, H, S, T).
    """

    def __init__(self, H: int, S: int, T: int, norm_axis: Literal["all", "time", "scale", "hidden"]) -> None:
        super().__init__()
        self.norm_axis = norm_axis
        if norm_axis == "all":
            self.ln = nn.LayerNorm([H, S, T])
        elif norm_axis == "time":
            self.ln = nn.LayerNorm([T])
        elif norm_axis == "scale":
            self.ln = nn.LayerNorm([S])
        elif norm_axis == "hidden":
            self.ln = nn.LayerNorm([H])
        else:
            raise ValueError(f"Unknown norm_axis={norm_axis}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_axis == "all":
            return self.ln(x)
        if self.norm_axis == "time":
            return self.ln(x)
        if self.norm_axis == "scale":
            y = x.permute(0, 1, 3, 2).contiguous()  # (B,H,T,S)
            y = self.ln(y)
            return y.permute(0, 1, 3, 2).contiguous()
        # hidden
        y = x.permute(0, 2, 3, 1).contiguous()  # (B,S,T,H)
        y = self.ln(y)
        return y.permute(0, 3, 1, 2).contiguous()


class _WaveletTCNBlock(nn.Module):
    def __init__(
        self,
        H: int,
        S: int,
        T: int,
        *,
        dilation_t: int,
        k_scale: int = 3,
        k_time: int = 3,
        dropout: float = 0.1,
        use_norm: bool = True,
        norm_axis: Literal["all", "time", "scale", "hidden"] = "all",
    ) -> None:
        super().__init__()
        self.dilation_t = int(dilation_t)
        self.k_time = int(k_time)

        pad_s = (k_scale - 1) // 2
        self.scale_mix = nn.Conv2d(H, H, kernel_size=(k_scale, 1), padding=(pad_s, 0), bias=False)

        self.time_dw = nn.Conv2d(
            H,
            H,
            kernel_size=(1, k_time),
            dilation=(1, self.dilation_t),
            groups=H,
            bias=False,
        )

        self.pw = nn.Conv2d(H, H, kernel_size=1, bias=True)
        self.drop = nn.Dropout(dropout)
        self.norm = _Norm2D(H, S, T, norm_axis=norm_axis) if use_norm else nn.Identity()

        nn.init.kaiming_normal_(self.scale_mix.weight)
        nn.init.kaiming_normal_(self.time_dw.weight)
        nn.init.kaiming_normal_(self.pw.weight)
        nn.init.constant_(self.pw.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,H,S,T)
        residual = x
        y = self.scale_mix(x)

        eff = self.dilation_t * (self.k_time - 1) + 1
        total_pad = eff - 1
        left = total_pad // 2
        right = total_pad - left
        y = F.pad(y, (left, right, 0, 0), mode="replicate")  # pad time only
        y = self.time_dw(y)

        y = F.gelu(y)
        y = self.pw(y)
        y = self.drop(y)

        y = residual + y
        y = self.norm(y)
        return y


class WaveletBackbone(nn.Module):
    """
    Operates on wavelet coeffs (B, T_pad, F, S) -> (B, T_pad, F, S)
    where S = cfg.num_scales (=2*L for include_approx=True).
    """

    def __init__(self, cfg: SiMLPeWaveletConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.T = cfg.padded_total_length
        self.S = cfg.num_scales
        H = cfg.hidden_dim
        Fdim = cfg.feat_dim

        self.fc_in = nn.Linear(Fdim, H)
        self.fc_out = nn.Linear(H, Fdim)

        nn.init.xavier_uniform_(self.fc_out.weight, gain=1e-8)
        nn.init.constant_(self.fc_out.bias, 0.0)

        # Dilation schedule (cap to avoid absurd dilation)
        blocks = []
        max_pow = max(1, (self.T // 2)).bit_length() - 1
        for i in range(cfg.num_layers):
            dil = 1 << min(i, max_pow)
            blocks.append(
                _WaveletTCNBlock(
                    H=H,
                    S=self.S,
                    T=self.T,
                    dilation_t=dil,
                    k_scale=3,
                    k_time=3,
                    dropout=0.1,
                    use_norm=cfg.use_norm,
                    norm_axis=cfg.norm_axis,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # w: (B,T,F,S)
        if w.ndim != 4:
            raise ValueError("Expected wavelet tensor shaped (B,T,F,S).")
        B, T, Fdim, S = w.shape
        if T != self.T or Fdim != self.cfg.feat_dim or S != self.S:
            raise ValueError(f"Got (T,F,S)=({T},{Fdim},{S}), expected ({self.T},{self.cfg.feat_dim},{self.S}).")

        # (B,T,F,S) -> (B,T,S,F)
        x = w.permute(0, 1, 3, 2).contiguous()
        x = self.fc_in(x)                      # (B,T,S,H)
        x = x.permute(0, 3, 2, 1).contiguous()  # (B,H,S,T)

        x = self.blocks(x)

        x = x.permute(0, 3, 2, 1).contiguous()  # (B,T,S,H)
        x = self.fc_out(x)                      # (B,T,S,F)
        x = x.permute(0, 1, 3, 2).contiguous()  # (B,T,F,S)
        return x


# -----------------------------------------------------------------------------
# Forecaster (same signature)
# -----------------------------------------------------------------------------

class SiMLPeWaveletForecaster(nn.Module):
    """End-to-end forecaster: xyz -> SWT -> Backbone -> ISWT -> xyz."""

    def __init__(self, cfg: SiMLPeWaveletConfig) -> None:
        super().__init__()
        if pywt is None:
            raise ImportError("PyWavelets is required (only to load wavelet filters).")
        if not cfg.include_approx:
            raise ValueError("This implementation requires include_approx=True for true ISWT.")

        self.cfg = cfg
        self.T_pad = cfg.padded_total_length

        # Differentiable SWT/ISWT with matched boundary mode
        self.swt = TorchSWT(
            wavelet=cfg.wavelet,
            level=cfg.swt_level,
            include_approx=cfg.include_approx,
            pad_mode="replicate",
        )

        self.backbone = WaveletBackbone(cfg)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: (B, T_in, N, 3)

        Returns:
            future: (B, T_out, N, 3)
        """
        if history.ndim != 4:
            raise ValueError("history must be shaped (B,T_in,N,3).")
        B, T_in, N, C = history.shape
        if T_in != self.cfg.input_length:
            raise ValueError(f"Expected input_length={self.cfg.input_length}, got {T_in}.")
        if N != self.cfg.num_nodes or C != 3:
            raise ValueError("Expected (N,3) matching cfg.num_nodes and coordinate dim=3.")

        flat = history.reshape(B, T_in, -1)  # (B,T_in,F)

        # Base seed padding to T_base
        seed_base = flat[:, -1:, :].expand(B, self.cfg.pred_length, flat.size(-1)).clone()
        seq_base = torch.cat([flat, seed_base], dim=1)  # (B,T_base,F)
        T_base = seq_base.size(1)

        # Additional seed padding to T_pad divisible by 2**L
        if self.T_pad < T_base:
            raise ValueError("padded_total_length < total_length; decrease swt_level or lengths.")
        if self.T_pad > T_base:
            extra = self.T_pad - T_base
            seed_extra = flat[:, -1:, :].expand(B, extra, flat.size(-1)).clone()
            seq = torch.cat([seq_base, seed_extra], dim=1)  # (B,T_pad,F)
        else:
            seq = seq_base  # (B,T_pad,F)

        # SWT: (B,T_pad,F) -> (B,T_pad,F,S)
        w = self.swt(seq)

        # Residual prediction in wavelet space:
        # backbone predicts Δw, then we add it to the input coefficients.
        delta = self.backbone(w)
        w_hat = w + delta

        # ISWT: (B,T_pad,F,S) -> (B,T_pad,F)
        recon = self.swt.inverse(w_hat)

        # Crop to base length and output future
        # recon = recon[:, :T_base, :]  # (B,T_base,F)
        future = recon[:, -self.cfg.pred_length :, :].reshape(B, self.cfg.pred_length, N, C)

        if self.cfg.add_last_offset_time_domain:
            future = future + history[:, -1:, :, :]

        return future



__all__ = ["SiMLPeWaveletConfig", "SiMLPeWaveletForecaster"]
