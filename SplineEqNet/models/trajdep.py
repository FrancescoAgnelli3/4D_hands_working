"""Wrapper around the LearnTrajDep GCN for datasets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class GraphConvolution(nn.Module):
    """Graph convolution layer adapted from LearnTrajDep."""

    def __init__(self, in_features: int, out_features: int, *, node_n: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.att = nn.Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(inputs, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCBlock(nn.Module):
    """Two-layer residual GCN block."""

    def __init__(self, features: int, dropout: float, *, node_n: int, bias: bool = True):
        super().__init__()
        self.gc1 = GraphConvolution(features, features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * features)
        self.gc2 = GraphConvolution(features, features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * features)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.Tanh()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.gc1(inputs)
        batch, nodes, feats = out.shape
        out = self.bn1(out.view(batch, -1)).view(batch, nodes, feats)
        out = self.activation(out)
        out = self.drop(out)

        out = self.gc2(out)
        out = self.bn2(out.view(batch, -1)).view(batch, nodes, feats)
        out = self.activation(out)
        out = self.drop(out)
        return out + inputs


class TrajDepGCN(nn.Module):
    """LearnTrajDep residual GCN stack."""

    def __init__(
        self,
        input_feature: int,
        hidden_feature: int,
        *,
        dropout: float,
        num_stage: int,
        node_n: int,
    ) -> None:
        super().__init__()
        self.num_stage = num_stage
        self.gc_in = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn_in = nn.BatchNorm1d(node_n * hidden_feature)
        self.blocks = nn.ModuleList(
            [GCBlock(hidden_feature, dropout, node_n=node_n) for _ in range(num_stage)]
        )
        self.gc_out = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.Tanh()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.gc_in(inputs)
        batch, nodes, feats = out.shape
        out = self.bn_in(out.view(batch, -1)).view(batch, nodes, feats)
        out = self.activation(out)
        out = self.drop(out)
        for block in self.blocks:
            out = block(out)
        out = self.gc_out(out)
        return out + inputs


@dataclass
class TrajDepConfig:
    input_length: int
    pred_length: int
    num_nodes: int
    coord_dim: int = 3
    hidden_dim: int = 256
    num_stages: int = 12
    dropout: float = 0.5
    dct_components: int = 35

    @property
    def total_length(self) -> int:
        return self.input_length + self.pred_length


def _build_dct(total_len: int) -> torch.Tensor:
    """Construct an orthonormal DCT-II matrix."""

    n = torch.arange(total_len, dtype=torch.float32).unsqueeze(1)
    k = torch.arange(total_len, dtype=torch.float32).unsqueeze(0)
    scale = torch.sqrt(torch.tensor(2.0 / total_len, dtype=torch.float32))
    mat = scale * torch.cos(math.pi * (n + 0.5) * k / float(total_len))
    mat[0, :] = torch.sqrt(torch.tensor(1.0 / total_len, dtype=torch.float32))
    return mat


class TrajDepForecaster(nn.Module):
    """Adapter that maps inputs to the TrajDep GCN."""

    def __init__(self, cfg: TrajDepConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.coord_dim = int(cfg.coord_dim)
        self.node_dim = int(cfg.num_nodes * self.coord_dim)
        total_len = int(cfg.total_length)
        if total_len <= 0:
            raise ValueError("total sequence length must be positive")

        dct = _build_dct(total_len)
        idct = dct.t()  # orthonormal DCT-II -> inverse is transpose
        dct_n = int(min(cfg.dct_components, total_len))
        self.register_buffer("_dct", dct[:dct_n, :])
        self.register_buffer("_idct", idct[:, :dct_n])
        self.dct_n = dct_n
        self.total_len = total_len

        stages = max(1, int(cfg.num_stages))
        self.backbone = TrajDepGCN(
            input_feature=dct_n,
            hidden_feature=int(cfg.hidden_dim),
            dropout=float(cfg.dropout),
            num_stage=stages,
            node_n=self.node_dim,
        )

    def forward(self, obs_3d: torch.Tensor) -> torch.Tensor:
        """Predict future coordinates from observed 3D trajectory.

        Args:
            obs_3d: Tensor shaped (B, T_in, N, C) where C matches cfg.coord_dim.

        Returns:
            Tensor shaped (B, T_out, N, C) containing the predicted future trajectory.
        """

        self._validate_inputs(obs_3d)
        return self._run_model(obs_3d, include_observed=False)

    def forward_full(self, obs_3d: torch.Tensor) -> torch.Tensor:
        """Return the reconstructed sequence including the observed frames."""

        self._validate_inputs(obs_3d)
        return self._run_model(obs_3d, include_observed=True)

    def _run_model(self, obs_3d: torch.Tensor, *, include_observed: bool) -> torch.Tensor:
        batch = obs_3d.size(0)
        last_frame = obs_3d[:, -1:, :, :]
        pad = last_frame.expand(-1, self.cfg.pred_length, -1, -1)
        padded = torch.cat([obs_3d, pad], dim=1)
        flat = padded.view(batch, self.total_len, self.node_dim)
        dct_coeffs = torch.einsum("kd, bdt -> bkt", self._dct, flat)
        dct_coeffs = dct_coeffs.permute(0, 2, 1).contiguous()
        pred_coeffs = self.backbone(dct_coeffs)
        recon = torch.einsum("tk, bnk -> bnt", self._idct, pred_coeffs)
        recon = recon.view(batch, self.node_dim, self.total_len)
        recon = recon.view(batch, self.cfg.num_nodes, self.coord_dim, self.total_len)
        recon = recon.permute(0, 3, 1, 2).contiguous()
        if include_observed:
            return recon
        return recon[:, -self.cfg.pred_length :, :, :]

    def _validate_inputs(self, obs_3d: torch.Tensor) -> None:
        if obs_3d.ndim != 4:
            raise ValueError("obs_3d must be a 4D tensor (B, T, N, C)")
        _, t_in, nodes, coord = obs_3d.shape
        if t_in != self.cfg.input_length:
            raise ValueError(f"expected {self.cfg.input_length} input frames, got {t_in}")
        if nodes != self.cfg.num_nodes:
            raise ValueError(f"expected {self.cfg.num_nodes} nodes, got {nodes}")
        if coord != self.coord_dim:
            raise ValueError(f"expected {self.coord_dim} coords, got {coord}")
