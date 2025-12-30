"""siMLPe-inspired temporal MLP baseline.

This module adapts the original siMLPe architecture to operate directly on
Assembly sequences (B, T, N, 3) without the DCT pre/post processing that the
original codebase relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


class _SpatialNorm(nn.Module):
    """LayerNorm-style normalization over the spatial/channel axis."""

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
    """LayerNorm-style normalization over the temporal axis."""

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
    """Linear projection applied across the feature/channel axis."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> transpose to operate over channels
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.proj(x)
        return x.transpose(1, 2)


class _TemporalFC(nn.Module):
    """Linear projection applied across the temporal axis."""

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nn.Linear acts on the last dimension, which is the temporal axis here.
        return self.proj(x)


class _MLPBlock(nn.Module):
    """Residual block used by siMLPe."""

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
        if use_spatial_fc:
            self.fc = _SpatialFC(dim)
        else:
            self.fc = _TemporalFC(seq_len)

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

        # Follow the initialization from the public siMLPe implementation.
        if hasattr(self.fc, "proj"):
            nn.init.xavier_uniform_(self.fc.proj.weight, gain=1e-8)
            nn.init.constant_(self.fc.proj.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc(x)
        out = self.norm(out)
        return residual + out


class _TransMLP(nn.Module):
    """Stack of `_MLPBlock`s mirroring the siMLPe backbone."""

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
        if num_layers <= 0:
            raise ValueError("num_layers must be positive for TransMLP.")

        if mix_spatial_temporal:
            layer_spatial_flags = [(idx % 2) == 1 for idx in range(num_layers)]
        else:
            layer_spatial_flags = [use_spatial_fc] * num_layers

        layers = [
            _MLPBlock(
                dim,
                seq_len,
                use_norm=use_norm,
                use_spatial_fc=use_spatial_flag,
                norm_axis=norm_axis,
            )
            for use_spatial_flag in layer_spatial_flags
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class SiMLPeConfig:
    """Configuration for the siMLPe competitor."""

    input_length: int
    pred_length: int
    num_nodes: int
    hidden_dim: int
    num_layers: int
    use_norm: bool = True
    use_spatial_fc_only: bool = False
    mix_spatial_temporal: bool = False
    norm_axis: Literal["spatial", "temporal", "all"] = "all"
    add_last_offset: bool = True

    @property
    def total_length(self) -> int:
        return self.input_length + self.pred_length

    @property
    def feature_dim(self) -> int:
        return self.num_nodes * 3


class SiMLPeBackbone(nn.Module):
    """Feature-space implementation of the siMLPe temporal MLP."""

    def __init__(self, cfg: SiMLPeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fc_in = nn.Linear(cfg.feature_dim, cfg.hidden_dim)
        self.mlp = _TransMLP(
            cfg.hidden_dim,
            cfg.total_length,
            use_norm=cfg.use_norm,
            use_spatial_fc=cfg.use_spatial_fc_only,
            num_layers=cfg.num_layers,
            norm_axis=cfg.norm_axis,
            mix_spatial_temporal=cfg.mix_spatial_temporal,
        )
        self.fc_out = nn.Linear(cfg.hidden_dim, cfg.feature_dim)

        nn.init.xavier_uniform_(self.fc_out.weight, gain=1e-8)
        nn.init.constant_(self.fc_out.bias, 0.0)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, T_total, F)
        x = self.fc_in(seq)
        x = x.transpose(1, 2).contiguous()  # (B, hidden_dim, T_total)
        x = self.mlp(x)
        x = x.transpose(1, 2).contiguous()
        return self.fc_out(x)


class SiMLPeForecaster(nn.Module):
    """End-to-end forecaster operating on Assembly hand sequences."""

    def __init__(self, cfg: SiMLPeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = SiMLPeBackbone(cfg)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Predict future 3D joint positions.

        Args:
            history: Tensor with shape (B, T_in, N, 3).

        Returns:
            Tensor with shape (B, T_out, N, 3) containing predicted absolute
            joint coordinates.
        """
        B, T_in, N, C = history.shape
        assert T_in == self.cfg.input_length, "Unexpected input length"
        assert N == self.cfg.num_nodes and C == 3, "Input must be (B, T, N, 3)"

        flat_history = history.reshape(B, T_in, -1)
        seed = flat_history[:, -1:, :].expand(B, self.cfg.pred_length, flat_history.size(-1)).clone()
        seq = torch.cat([flat_history, seed], dim=1)
        full_pred = self.backbone(seq)
        future = full_pred[:, self.cfg.input_length :, :]
        future = future.reshape(B, self.cfg.pred_length, N, C)
        if self.cfg.add_last_offset:
            future = future + history[:, -1:, :, :]
        return future


__all__ = ["SiMLPeConfig", "SiMLPeForecaster"]
