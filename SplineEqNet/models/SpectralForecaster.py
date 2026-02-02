# models/spectral.py
#
# Explicit spectral parameterization forecaster for hand trajectories.
# Input : history (B, T_in, N, D)
# Output: future  (B, T_out, N, D)
#
# Changes vs previous version (as requested):
#  1) Input/output bases use the SAME time normalization denominator (T_total-1).
#  2) Adds a DC (constant) basis term.
#  3) Learns a coefficient-space dynamics: coeff_out = Dyn(coeff_in).
#
# Config keys are unchanged (spectral_*), forward signature unchanged.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class SpectralForecasterConfig:
    # Required to match other models
    input_length: int
    pred_length: int
    num_nodes: int
    feature_dim: int = 0  # if 0, inferred as num_nodes * coord_dim
    coord_dim: int = 3

    # Spectral basis
    num_modes: int = 32              # K (for sin/cos modes with k>=1)
    include_sin_cos: bool = True     # always True here; kept for clarity
    time_normalize: bool = True      # time t mapped by shared denom (T_total-1)

    # Preprocessing / stability
    subtract_last: bool = True       # predict residual w.r.t. last frame
    subtract_mean: bool = False      # optionally also remove per-feature mean over T_in

    # Coefficient-space dynamics (low capacity)
    use_coeff_mlp: bool = True
    coeff_mlp_hidden: int = 128
    coeff_dropout: float = 0.0
    coeff_residual: bool = True      # coeff_out = coeff_in + mlp(coeff_in)

    # Optional small ridge when computing pseudo-inverse
    pinv_ridge: float = 0.0          # 0 = exact pinv, >0 = (B^T B + λI)^-1 B^T


class SpectralForecaster(nn.Module):
    """
    history: (B, T_in, N, D)
    returns: (B, T_out, N, D)
    """

    def __init__(self, cfg: SpectralForecasterConfig) -> None:
        super().__init__()
        self.cfg = cfg

        T_in = int(cfg.input_length)
        T_out = int(cfg.pred_length)
        if T_in <= 1:
            raise ValueError("input_length must be >= 2.")
        if T_out <= 0:
            raise ValueError("pred_length must be >= 1.")

        self.T_in = T_in
        self.T_out = T_out
        self.T_total = T_in + T_out

        self.N = int(cfg.num_nodes)
        self.D = int(cfg.coord_dim)
        self.F = int(cfg.feature_dim) if int(cfg.feature_dim) > 0 else self.N * self.D
        if self.F != self.N * self.D:
            raise ValueError(
                f"feature_dim must equal num_nodes*coord_dim (got {self.F}, expected {self.N*self.D})."
            )

        # Mode count K must be stable given T_in; keep conservative cap.
        # Basis dimension will be (1 + 2K) due to DC term.
        K_req = int(cfg.num_modes)
        K_max = max(1, (self.T_in - 1) // 2)
        self.K = int(min(K_req, K_max))
        self.basis_dim = 1 + 2 * self.K  # DC + sin/cos

        # Shared time normalization denominator
        self._time_denom = float(self.T_total - 1) if cfg.time_normalize else 1.0
        if self._time_denom <= 0:
            self._time_denom = 1.0

        # Build basis matrices
        # B_in : (T_in, basis_dim)
        # B_out: (T_out, basis_dim)
        t_in = torch.arange(self.T_in, dtype=torch.float32)
        t_out = torch.arange(self.T_in, self.T_total, dtype=torch.float32)

        B_in = self._make_basis(t_in, self.K, denom=self._time_denom)
        B_out = self._make_basis(t_out, self.K, denom=self._time_denom)

        # Projector from time series -> coefficients: coeff_in = x_time @ P_in
        P_in = self._make_projection_matrix(B_in, ridge=float(cfg.pinv_ridge))  # (T_in, basis_dim)

        self.register_buffer("B_out", B_out, persistent=True)  # (T_out, basis_dim)
        self.register_buffer("P_in", P_in, persistent=True)    # (T_in, basis_dim)

        # Coefficient dynamics network: maps coeff_in -> coeff_out in coefficient space
        if cfg.use_coeff_mlp:
            self.coeff_mlp = nn.Sequential(
                nn.LayerNorm(self.basis_dim),
                nn.Linear(self.basis_dim, int(cfg.coeff_mlp_hidden)),
                nn.GELU(),
                nn.Dropout(float(cfg.coeff_dropout)),
                nn.Linear(int(cfg.coeff_mlp_hidden), self.basis_dim),
            )
        else:
            self.coeff_mlp = None

    @staticmethod
    def _make_basis(t: torch.Tensor, K: int, denom: float) -> torch.Tensor:
        """
        t: (T,) raw indices
        returns basis: (T, 1+2K) with DC + sin/cos modes
        """
        t = t.to(dtype=torch.float32) / max(1.0, float(denom))  # shared normalization

        # DC term
        dc = torch.ones((t.shape[0], 1), dtype=torch.float32)

        # sin/cos for k=1..K with omega_k = 2πk
        k = torch.arange(1, K + 1, dtype=torch.float32)
        omega = 2.0 * math.pi * k  # (K,)
        angles = t[:, None] * omega[None, :]  # (T, K)

        sin = torch.sin(angles)
        cos = torch.cos(angles)

        B = torch.cat([dc, sin, cos], dim=1)  # (T, 1+2K)

        # Column normalization for numeric stability (keeps DC as well-defined)
        B = B / (B.norm(dim=0, keepdim=True) + 1e-8)
        return B

    @staticmethod
    def _make_projection_matrix(B_in: torch.Tensor, ridge: float) -> torch.Tensor:
        """
        Given B_in (T_in, C), returns P (T_in, C) such that:
          coeff = x_time @ P   where x_time is (B,F,T_in), coeff is (B,F,C).
        """
        T, C = B_in.shape
        if ridge <= 0.0:
            # pinv(B_in): (C, T) -> transpose to (T, C)
            return torch.linalg.pinv(B_in).transpose(0, 1).contiguous()

        BtB = B_in.transpose(0, 1) @ B_in  # (C, C)
        BtB = BtB + ridge * torch.eye(C, dtype=B_in.dtype, device=B_in.device)
        inv = torch.linalg.inv(BtB)
        pinv_ridge = inv @ B_in.transpose(0, 1)  # (C, T)
        return pinv_ridge.transpose(0, 1).contiguous()  # (T, C)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 4:
            raise ValueError("history must be shaped (B, T_in, N, D).")
        B, T_in, N, D = history.shape
        if T_in != self.T_in:
            raise ValueError(f"Expected T_in={self.T_in}, got {T_in}.")
        if N != self.N:
            raise ValueError(f"Expected N={self.N}, got {N}.")
        if D != self.D:
            raise ValueError(f"Expected D={self.D}, got {D}.")

        # Flatten joints/coords into features
        x = history.reshape(B, T_in, self.F)  # (B, T_in, F)

        # Stabilize by predicting residuals
        if self.cfg.subtract_last:
            ref = x[:, -1:, :]  # (B,1,F)
            x0 = x - ref
        else:
            ref = None
            x0 = x

        if self.cfg.subtract_mean:
            mu = x0.mean(dim=1, keepdim=True)  # (B,1,F)
            x0 = x0 - mu
        else:
            mu = None

        # (B, T_in, F) -> (B, F, T_in)
        x_time = x0.transpose(1, 2).contiguous()  # (B, F, T_in)

        # Coefficients on input window in the shared basis coordinates
        # coeff_in: (B, F, C)
        coeff_in = x_time @ self.P_in  # (B, F, basis_dim)

        # Coefficient-space dynamics: coeff_out = Dyn(coeff_in)
        if self.coeff_mlp is not None:
            delta = self.coeff_mlp(coeff_in)
            coeff_out = coeff_in + delta if self.cfg.coeff_residual else delta
        else:
            coeff_out = coeff_in

        # Reconstruct future residuals: (B,F,C) @ (C,T_out) -> (B,F,T_out)
        y_future = coeff_out @ self.B_out.transpose(0, 1)  # (B, F, T_out)

        # Back to (B, T_out, F)
        y_future = y_future.transpose(1, 2).contiguous()

        # Undo preprocessing
        if mu is not None:
            y_future = y_future + mu.expand(-1, self.T_out, -1)
        if ref is not None:
            y_future = y_future + ref.expand(-1, self.T_out, -1)

        return y_future.reshape(B, self.T_out, self.N, self.D)
