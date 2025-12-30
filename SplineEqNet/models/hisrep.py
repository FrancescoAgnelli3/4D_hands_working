from dataclasses import dataclass
import warnings
import math
from typing import Optional

import numpy as np
import torch
from torch import nn


# ---- Minimal dependencies replicated locally (no external imports) ----

def get_dct_matrix(N: int):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.att = nn.Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        super().__init__()
        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)
        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        return y + x


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        super().__init__()
        self.num_stage = num_stage
        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)
        self.gcbs = nn.ModuleList([GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n) for _ in range(num_stage)])
        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x, is_out_resi=True):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        for i in range(self.num_stage):
            y = self.gcbs[i](y)
        y = self.gc7(y)
        if is_out_resi:
            y = y + x
        return y


class AttModel(nn.Module):
    def __init__(self, in_features=48, kernel_size=10, d_model=512, num_stage=2, dct_n=10):
        super().__init__()
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dct_n = dct_n
        assert kernel_size == 10
        self.convQ = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, bias=False),
            nn.ReLU(),
        )
        self.convK = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5, bias=False),
            nn.ReLU(),
        )
        self.gcn = GCN(input_feature=(dct_n) * 2, hidden_feature=d_model, p_dropout=0.3, num_stage=num_stage, node_n=in_features)

    def forward(self, src, output_n=25, input_n=50, itera=1):
        dct_n = self.dct_n
        src = src[:, :input_n]
        src_tmp = src.clone()
        bs = src.shape[0]
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        dct_m, idct_m = get_dct_matrix(self.kernel_size + output_n)
        dev = src.device
        dct_m = torch.from_numpy(dct_m).float().to(dev)
        idct_m = torch.from_numpy(idct_m).float().to(dev)

        vn = input_n - self.kernel_size - output_n + 1
        vl = self.kernel_size + output_n
        idx = np.expand_dims(np.arange(vl), axis=0) + np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape([bs * vn, vl, -1])
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape([bs, vn, dct_n, -1]).transpose(2, 3).reshape([bs, vn, -1])

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []

        key_tmp = self.convK(src_key_tmp / 1000.0)
        for i in range(itera):
            query_tmp = self.convQ(src_query_tmp / 1000.0)
            score_tmp = torch.matmul(query_tmp.transpose(1, 2), key_tmp) + 1e-15
            att_tmp = score_tmp / (torch.sum(score_tmp, dim=2, keepdim=True))
            dct_att_tmp = torch.matmul(att_tmp, src_value_tmp)[:, 0].reshape([bs, -1, dct_n])

            input_gcn = src_tmp[:, idx]
            dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)
            dct_in_tmp = torch.cat([dct_in_tmp, dct_att_tmp], dim=-1)
            dct_out_tmp = self.gcn(dct_in_tmp)
            out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0), dct_out_tmp[:, :, :dct_n].transpose(1, 2))
            outputs.append(out_gcn.unsqueeze(2))
            if itera > 1:
                out_tmp = out_gcn.clone()[:, 0 - output_n:]
                src_tmp = torch.cat([src_tmp, out_tmp], dim=1)
                vn = 1 - 2 * self.kernel_size - output_n
                vl = self.kernel_size + output_n
                idx_dct = np.expand_dims(np.arange(vl), axis=0) + np.expand_dims(np.arange(vn, -self.kernel_size - output_n + 1), axis=1)
                src_key_tmp = src_tmp[:, idx_dct[0, :-1]].transpose(1, 2)
                key_new = self.convK(src_key_tmp / 1000.0)
                key_tmp = torch.cat([key_tmp, key_new], dim=2)
                src_dct_tmp = src_tmp[:, idx_dct].clone().reshape([bs * self.kernel_size, vl, -1])
                src_dct_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_dct_tmp).reshape([bs, self.kernel_size, dct_n, -1]).transpose(2, 3).reshape([bs, self.kernel_size, -1])
                src_value_tmp = torch.cat([src_value_tmp, src_dct_tmp], dim=1)
                src_query_tmp = src_tmp[:, -self.kernel_size:].transpose(1, 2)

        outputs = torch.cat(outputs, dim=2)
        return outputs


@dataclass
class HisRepConfig:
    num_nodes: int
    input_length: int
    pred_length: int
    hidden_dim: int = 128
    num_stages: int = 2
    dct_n: int = 10
    kernel_size: int = 10  # AttModel asserts kernel_size == 10


class HisRepForecaster(nn.Module):
    """Thin wrapper around embedded AttModel for pipeline.

    Expects inputs shaped (B, T_in, N, 3) and outputs (B, T_out, N, 3).
    Internally flattens the node/coord dimension to match AttModel's
    expected feature layout.
    """

    def __init__(self, cfg: HisRepConfig):
        super().__init__()
        max_coeffs = int(cfg.kernel_size + cfg.pred_length)
        raw_dct = cfg.dct_n if cfg.dct_n is not None else max_coeffs
        clamped_dct = max(1, min(int(raw_dct), max_coeffs))
        if clamped_dct != raw_dct:
            warnings.warn(
                f"HisRep: clamping dct_n from {raw_dct} to {clamped_dct} to respect kernel_size + pred_length "
                f"({cfg.kernel_size}+{cfg.pred_length})",
                RuntimeWarning,
            )
        cfg.dct_n = clamped_dct

        self.cfg = cfg
        in_features = cfg.num_nodes * 3
        self.backbone = AttModel(
            in_features=in_features,
            kernel_size=cfg.kernel_size,
            d_model=cfg.hidden_dim,
            num_stage=cfg.num_stages,
            dct_n=cfg.dct_n,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T_in, N, 3) -> (B, T_out, N, 3)"""
        B, T_in, N, C = x.shape
        assert C == 3, f"Expected 3 coords, got {C}"
        assert N == self.cfg.num_nodes, f"num_nodes mismatch: {N} != {self.cfg.num_nodes}"

        required_history = self.cfg.pred_length + self.backbone.kernel_size
        if T_in < required_history:
            pad_len = required_history - T_in
            pad = x[:, -1:].expand(B, pad_len, N, C)
            x = torch.cat([x, pad], dim=1)
            T_in = required_history

        flat = x.reshape(B, T_in, N * C)
        y = self.backbone(flat, output_n=self.cfg.pred_length, input_n=T_in, itera=1)
        # y: (B, kernel_size+T_out, 1, N*3); take horizon slice and reshape
        pred_flat = y[:, self.cfg.kernel_size:, 0]
        pred = pred_flat.reshape(B, self.cfg.pred_length, N, C)
        return pred
