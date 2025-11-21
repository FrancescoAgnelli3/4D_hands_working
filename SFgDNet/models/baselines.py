"""Baseline spatio-temporal models implemented with vanilla PyTorch and PyG.

Replaces the previous torch-spatiotemporal dependency while keeping the
high-level architecture used for Assembly MPT baselines. The core model
supports configurable temporal backbones (GRU/LSTM/Mamba) together with
optional spatial message passing via GCN or GAT layers.
"""

from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv


class TimeSpaceModel(nn.Module):
    """Configurable temporal + spatial baseline for motion forecasting."""

    def __init__(
        self,
        input_size: int,
        n_nodes: int,
        horizon: int,
        hidden_size: int = 64,
        output_size: Optional[int] = None,
        t_layers: int = 1,
        gnn_layers: int = 0,
        gnn_type: Optional[str] = None,
        temporal_type: str = "gru",
        gat_heads: int = 4,
    ) -> None:
        super().__init__()

        self.input_size = int(input_size)
        self.output_size = int(output_size or input_size)
        self.n_nodes = int(n_nodes)
        self.horizon = int(horizon)
        self.hidden = int(hidden_size)
        self.temporal_type = temporal_type.lower()
        self.gnn_type = gnn_type.lower() if gnn_type is not None else None

        self.encoder = nn.Linear(self.input_size, self.hidden)
        self.node_embedding = nn.Parameter(torch.zeros(self.n_nodes, self.hidden))
        nn.init.xavier_uniform_(self.node_embedding)

        temporal_layers = max(1, int(t_layers))
        if self.temporal_type == "gru":
            self.temporal = nn.GRU(
                input_size=self.hidden,
                hidden_size=self.hidden,
                num_layers=temporal_layers,
                batch_first=True,
            )
        elif self.temporal_type == "lstm":
            self.temporal = nn.LSTM(
                input_size=self.hidden,
                hidden_size=self.hidden,
                num_layers=temporal_layers,
                batch_first=True,
            )
        elif self.temporal_type == "mamba":
            from mamba_ssm import Mamba

            self.temporal = Mamba(self.hidden)
        else:
            raise ValueError(f"Unknown temporal_type '{temporal_type}'")

        self.spatial_layers = nn.ModuleList()
        self.activation = nn.ReLU()
        if gnn_layers > 0:
            if self.gnn_type not in {"gcn", "gat"}:
                raise ValueError("gnn_type must be 'gcn' or 'gat' when gnn_layers>0")
            for _ in range(int(gnn_layers)):
                if self.gnn_type == "gcn":
                    conv = GCNConv(self.hidden, self.hidden, add_self_loops=True)
                else:
                    conv = GATConv(
                        in_channels=self.hidden,
                        out_channels=self.hidden,
                        heads=gat_heads,
                        concat=False,
                        add_self_loops=True,
                    )
                self.spatial_layers.append(conv)
            self.spatial_skip = nn.Linear(self.hidden, self.hidden)
        else:
            self.spatial_skip = None

        self.decoder = nn.Linear(self.hidden, self.output_size * self.horizon)

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forecast the next `horizon` frames for each node."""

        B, T_in, N, _ = x.shape
        if N != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes} nodes, received {N}")

        h = self.encoder(x)
        h = h + self.node_embedding.view(1, 1, N, self.hidden)
        h = h.view(B * N, T_in, self.hidden)

        if self.temporal_type == "mamba":
            h = self.temporal(h)  # (B*N, T_in, hidden)
            h = h[:, -1, :]
        else:
            out, _ = self.temporal(h)
            h = out[:, -1, :]

        h = h.view(B, N, self.hidden)

        if self.spatial_layers:
            if edge_index is None:
                raise ValueError("edge_index is required when spatial layers are enabled")
            edge_index = edge_index.to(h.device)
            skip = self.spatial_skip(h)
            for conv in self.spatial_layers:
                updated = []
                for b in range(B):
                    out = conv(h[b], edge_index)
                    updated.append(self.activation(out))
                h = torch.stack(updated, dim=0)
            h = h + skip

        out = self.decoder(h)
        out = out.view(B, N, self.horizon, self.output_size)
        return out.permute(0, 2, 1, 3).contiguous()


class ConstantModel(nn.Module):
    """Repeats the last observed frame over the prediction horizon."""

    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = int(horizon)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        last = x[:, -1:, :, :]
        return last.repeat(1, self.horizon, 1, 1) + 0.0 * self.bias
