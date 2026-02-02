from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


@dataclass
class HandForecastConfig:
    n_landmarks: int          # N
    coord_dim: int            # C
    t_out: int                # T_out

    # These now must match N*C because there is no in/out projection.
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 2
    dropout: float = 0.1

    decoder_conv_kernel: int = 5


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        inv_rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        return x * inv_rms * self.weight


class MambaBlock(nn.Module):
    """
    Residual temporal block.
    Input/Output: (B, T, D)
    """
    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mamba(self.norm(x))
        return x + self.drop(y)


class TimeResizeDecoder(nn.Module):
    """
    Resizes time from T_in to T_out via interpolation + Conv1d refinement.
    Input:  (B, T_in, D)
    Output: (B, T_out, D)
    """
    def __init__(self, dim: int, t_out: int, kernel_size: int, dropout: float):
        super().__init__()
        self.t_out = int(t_out)
        k = int(kernel_size)
        if k % 2 == 0:
            k += 1
        pad = k // 2

        self.refine = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=k, padding=pad, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim, dim, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, D)
        b, t_in, d = x.shape

        x_t = x.transpose(1, 2)  # (B, D, T_in)
        if t_in != self.t_out:
            x_t = F.interpolate(x_t, size=self.t_out, mode="linear", align_corners=False)
        y_t = self.refine(x_t)   # (B, D, T_out)
        return y_t.transpose(1, 2)  # (B, T_out, D)


class HandMotionForecasterResidual(nn.Module):
    """
    Forecast as residual w.r.t. the last observed pose.

    Input:
        x: (B, T_in, N, C)
    Output:
        y: (B, T_out, N, C)

    No input/output projections:
        D is fixed to N*C (flattened pose dimension).
    """
    def __init__(self, cfg: HandForecastConfig):
        super().__init__()
        self.cfg = cfg
        self.n_landmarks = cfg.n_landmarks
        self.coord_dim = cfg.coord_dim
        self.t_out = cfg.t_out

        self.d_model = cfg.n_landmarks * cfg.coord_dim  # fixed
        self.time_bias = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.drop_in = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    dim=self.d_model,
                    d_state=cfg.d_state,
                    d_conv=cfg.d_conv,
                    expand=cfg.expand,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_layers)
            ]
        )

        self.norm_out = RMSNorm(self.d_model)

        self.decoder = TimeResizeDecoder(
            dim=self.d_model,
            t_out=cfg.t_out,
            kernel_size=cfg.decoder_conv_kernel,
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, N, C)
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B, T_in, N, C), got {tuple(x.shape)}")

        b, t_in, n, c = x.shape
        if n != self.n_landmarks:
            raise ValueError(f"Expected N={self.n_landmarks}, got N={n}")
        if c != self.coord_dim:
            raise ValueError(f"Expected C={self.coord_dim}, got C={c}")

        last_pose = x[:, -1, :, :]  # (B, N, C)

        # Flatten poses per timestep to D = N*C
        h = x.reshape(b, t_in, self.d_model)         # (B, T_in, D)

        # Temporal modeling
        h = self.drop_in(h + self.time_bias)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm_out(h)                         # (B, T_in, D)

        # Produce a delta trajectory in flattened space
        delta_flat = self.decoder(h)                 # (B, T_out, D)
        delta = delta_flat.reshape(b, self.t_out, n, c)  # (B, T_out, N, C)

        # Residual prediction w.r.t. last observed pose
        y = last_pose.unsqueeze(1) + delta           # (B, T_out, N, C)
        return y



