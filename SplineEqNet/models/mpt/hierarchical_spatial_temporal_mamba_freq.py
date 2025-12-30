import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse your existing spatial module
from .hierarchical_spatial_temporal_mamba import HierarchicalSpatialModuleUNet

_DCT_MATRIX_CACHE = {}

def _build_dct_matrix(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    n = torch.arange(length, device=device, dtype=dtype).unsqueeze(0)  # (1, L)
    k = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)  # (L, 1)
    mat = torch.cos((n + 0.5) * k * (math.pi / length))               # (L, L)
    weights = torch.full((length, 1), math.sqrt(2.0 / length), device=device, dtype=dtype)
    weights[0, 0] = math.sqrt(1.0 / length)
    return mat * weights

def _get_dct_matrix(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # cache in fp32 to improve numerical stability; cast on return
    key = (length, str(device), "fp32")
    mat = _DCT_MATRIX_CACHE.get(key)
    if mat is None:
        with torch.no_grad():
            mat = _build_dct_matrix(length, device, torch.float32)
        _DCT_MATRIX_CACHE[key] = mat
    return mat.to(device=device, dtype=dtype)

def _apply_dct_matrix(x: torch.Tensor, dim: int, transpose: bool) -> torch.Tensor:
    length = x.size(dim)
    mat = _get_dct_matrix(length, x.device, x.dtype)
    last_dim = x.ndim - 1

    if dim != last_dim:
        permute_dims = list(range(x.ndim))
        permute_dims[dim], permute_dims[last_dim] = permute_dims[last_dim], permute_dims[dim]
        x_perm = x.permute(*permute_dims)
    else:
        permute_dims = None
        x_perm = x

    flat = x_perm.reshape(-1, length)
    proj = flat @ (mat.transpose(0, 1) if transpose else mat)
    proj = proj.view_as(x_perm)

    if permute_dims is None:
        return proj

    inverse_permute = [0] * len(permute_dims)
    for i, d in enumerate(permute_dims):
        inverse_permute[d] = i
    return proj.permute(*inverse_permute)

def _dct_ortho(x: torch.Tensor, dim: int) -> torch.Tensor:
    if hasattr(torch.fft, "dct"):
        return torch.fft.dct(x, type=2, dim=dim, norm="ortho")
    return _apply_dct_matrix(x, dim, transpose=True)

def _idct_ortho(x: torch.Tensor, dim: int) -> torch.Tensor:
    if hasattr(torch.fft, "idct"):
        return torch.fft.idct(x, type=2, dim=dim, norm="ortho")
    return _apply_dct_matrix(x, dim, transpose=False)


class DepthwiseSeparable1D(nn.Module):
    """Lightweight 1D conv over coefficient index."""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.dw = nn.Conv1d(channels, channels, kernel_size, padding=padding,
                            dilation=dilation, groups=channels, bias=False)
        self.pw = nn.Conv1d(channels, channels, 1, bias=True)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):  # x: (B, C, K)
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        return F.gelu(x)


class SpectralForecaster(nn.Module):
    """
    Maps low-frequency DCT coefficients from length T_in basis to low-frequency
    coefficients in length T_out basis.

    Input:  (B, K_in, N, C_in)
    Output: (B, K_out, N, C_out)
    """
    def __init__(self, c_in: int, c_out: int, k_in: int, k_out: int, width: int = 128, layers: int = 4):
        super().__init__()
        self.k_in = k_in
        self.k_out = k_out
        self.c_in = c_in
        self.c_out = c_out

        # Work in (B, N, C, K) -> treat (N,C) as channels for 1D conv over K
        ch_in = c_in
        ch_mid = width
        ch_out = c_out

        self.in_proj = nn.Conv1d(ch_in, ch_mid, 1)
        blocks = []
        for i in range(layers):
            blocks.append(DepthwiseSeparable1D(ch_mid, kernel_size=3, dilation=2**(i % 3)))
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Conv1d(ch_mid, ch_out, 1)

        # Learned resampler across coefficient index: K_in -> K_out
        self.register_parameter(
            "resample", nn.Parameter(torch.empty(k_out, k_in))
        )
        nn.init.xavier_uniform_(self.resample)

    def forward(self, Xcin):  # (B, K_in, N, C_in)
        B, K_in, N, C_in = Xcin.shape
        assert K_in == self.k_in and C_in == self.c_in

        # Linear resampling along coefficient axis to target K_out
        # Result shape: (B, K_out, N, C_in)
        Xres = torch.einsum("bkni,ok->boni", Xcin, self.resample)  # (B, N, C_in, K_out)
        Xres = Xres.permute(0, 1, 2, 3)  # (B, N, C_in, K_out)

        # Conv over coefficient axis works on (B*N, C_in, K_out)
        X = Xres.reshape(B * N, C_in, self.k_out)
        X = self.in_proj(X)
        X = self.blocks(X)
        X = self.out_proj(X)
        X = X.reshape(B, N, self.c_out, self.k_out).permute(0, 3, 1, 2)  # (B, K_out, N, C_out)
        return X


class HierarchicalSpatialTemporalMambaDCT(nn.Module):
    """
    Frequency-enforcing, length-consistent variant.

    - Spatial processing happens per frame in the time domain with a hierarchical U-Net.
    - Temporal forecasting is done strictly in the DCT domain:
        * Keep only the first K_in coefficients from T_in.
        * Predict the first K_out coefficients for T_out.
        * Zero-pad higher frequencies, then IDCT of length T_out.
    - Length mismatch is explicit and principled: input basis = T_in, output basis = T_out.

    forward(Z) -> (Vhat, aux)
      - Z:    (B, T_in, N, C_in)
      - Vhat: (B, T_out, N, C_out)
      - aux:  {'coeff_pred': (B, K_out, N, C_out)}
    """

    def __init__(self, in_channels: int, n_nodes: int, d_out: int, t_out: int,
                 fast_c_f: int = 64, fast_layers: int = 4, fast_kernel: int = 3,
                 slow_hidden: int = 256, slow_layers: int = 2,
                 spatial_hidden: Optional[int] = None,
                 spectral_width: int = 128, spectral_layers: int = 4,
                 use_temporal_dct: bool = False, dct_window: Optional[int] = None,
                 dct_keep_coeffs: int = 8, dct_input_scale: float = 1.0,
                 use_augmentation: bool = False,
                 augment_node_dropout: float = 0.001,
                 augment_channel_dropout: float = 0.0,
                 augment_time_dropout: float = 0.0):
        super().__init__()
        if n_nodes != 42:
            raise ValueError("This implementation currently expects n_nodes=42 (two hands).")
        self.n_nodes = int(n_nodes)
        self.in_channels = int(in_channels)
        self.d_out = int(d_out)
        self.t_out = int(t_out)

        spatial_hidden_size = int(spatial_hidden) if spatial_hidden is not None else 10
        self.spectral_width = int(spectral_width)
        self.spectral_layers = int(spectral_layers)

        # K values (defaults: retain up to Nyquist-ish)
        self.k_in =  None  # determined at forward by T_in if None
        self.k_out = None

        # Hierarchical spatial processor (per-frame)
        self.hier_spatial = HierarchicalSpatialModuleUNet(self.in_channels, hidden_feature=spatial_hidden_size)

        # Spectral forecaster (configured lazily once K_in/K_out are known)
        self.spectral: Optional[SpectralForecaster] = None

    def _maybe_build_spectral(self, k_in: int, k_out: int, device: torch.device, dtype: torch.dtype):
        need_new = (
            self.spectral is None
            or self.spectral.k_in != k_in
            or self.spectral.k_out != k_out
        )
        if need_new:
            spectral = SpectralForecaster(
                c_in=self.in_channels, c_out=self.d_out, k_in=k_in, k_out=k_out,
                width=self.spectral_width, layers=self.spectral_layers
            )
            # move to match the incoming batch
            spectral.to(device=device, dtype=dtype)
            self.spectral = spectral

    def forward(self, Z: torch.Tensor):
        """
        Z: (B, T_in, N, C_in)
        """
        B, T_in, N, C_in = Z.shape
        assert N == self.n_nodes and C_in == self.in_channels, "Shape mismatch."

        # pick K_in/K_out if not provided
        k_in = self.k_in or min(T_in,  min(64, T_in))  # cap to 64 by default
        k_out = self.k_out or min(self.t_out, min(64, self.t_out))

        self._maybe_build_spectral(k_in, k_out)

        # Baseline (last observed frame) for residualization in time domain
        baseline = Z[:, -1, :, :].unsqueeze(1)  # (B, 1, N, C_in)

        # Spatial processing per frame (time domain): (B,T,N,C) -> (B*T,C,N) -> hier -> back
        Z_bt = Z.permute(0, 1, 3, 2).contiguous().view(B * T_in, C_in, N)
        Z_bt = self.hier_spatial(Z_bt)
        Z = Z_bt.view(B, T_in, C_in, N).permute(0, 1, 3, 2).contiguous()  # (B,T_in,N,C_in)

        # Residualize and move to frequency domain (length T_in)
        Z_res = Z - baseline  # broadcast over T_in
        Z_dct = _dct_ortho(Z_res, dim=1)  # (B, T_in, N, C_in)
        Xcin = Z_dct[:, :k_in, :, :]      # (B, K_in, N, C_in)

        # Predict low-frequency coeffs for the T_out basis
        C_hat_low = self.spectral(Xcin)   # (B, K_out, N, C_out)

        # Assemble full coefficient vector for T_out: zero high frequencies
        C_full = torch.zeros(B, self.t_out, N, self.d_out, device=Z.device, dtype=Z.dtype)
        C_full[:, :k_out, :, :] = C_hat_low

        # Go back to time domain with IDCT of length T_out, add baseline (trim channels if needed)
        Vhat = _idct_ortho(C_full, dim=1)  # (B, T_out, N, d_out)
        # Add baseline over channels (use first d_out channels)
        assert self.d_out <= self.in_channels, "d_out must be ≤ in_channels for baseline addition."
        Vhat = Vhat + baseline[:, :, :, :self.d_out]  # broadcast over T_out

        aux = {"coeff_pred": C_hat_low}
        return Vhat, aux


__all__ = ["HierarchicalSpatialTemporalMambaDCT"]
