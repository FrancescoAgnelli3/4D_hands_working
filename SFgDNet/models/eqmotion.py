import numpy as np
import torch
from dataclasses import dataclass
from torch import nn
import torch.nn.functional as F

__all__ = ["EqMotionConfig", "EqMotionForecaster"]


class Feature_learning_layer(nn.Module):
    """Graph reasoning layer copied from EqMotion with minimal changes."""

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        input_c,
        hidden_c,
        output_c,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.ReLU(),
        recurrent=True,
        coords_weight=1.0,
        attention=False,
        norm_diff=False,
        tanh=False,
        input_reasoning=False,
        category_num=2,
    ):
        super(Feature_learning_layer, self).__init__()
        self.norm_diff = norm_diff

        self.coord_vel = nn.Linear(hidden_c, hidden_c, bias=False)
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.hidden_c = hidden_c
        edge_coords_nf = hidden_c
        self.hidden_nf = hidden_nf

        layer = nn.Linear(hidden_nf, hidden_c, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.clamp = False
        coord_mlp = [
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
        ]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1) * 3)
        self.coord_mlp = nn.Sequential(*coord_mlp)

        self.tao = 0.2
        self.category_num = category_num
        self.input_reasoning = input_reasoning

        if input_reasoning:
            self.edge_mlp = nn.Sequential(
                nn.Linear(input_edge + edge_coords_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
            )
            self.category_mlp = []
            for _ in range(category_num):
                self.category_mlp.append(
                    nn.Sequential(
                        nn.Linear(input_edge + edge_coords_nf, hidden_nf),
                        act_fn,
                        nn.Linear(hidden_nf, hidden_c),
                        act_fn,
                    )
                )
            self.category_mlp = nn.ModuleList(self.category_mlp)

            self.factor_mlp = nn.Sequential(
                nn.Linear(hidden_c, hidden_c),
                act_fn,
                nn.Linear(hidden_c, hidden_c),
                act_fn,
            )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        self.node_att_mlp = nn.Sequential(
            nn.Linear(hidden_c, hidden_c),
            act_fn,
            nn.Linear(hidden_c, 1),
        )

        self.add_non_linear = True
        if self.add_non_linear:
            self.layer_q = nn.Linear(hidden_c, hidden_c, bias=False)
            self.layer_k = nn.Linear(hidden_c, hidden_c, bias=False)

        self.add_inner_agent_attention = True
        if self.add_inner_agent_attention:
            self.mlp_q = nn.Sequential(nn.Linear(hidden_nf, int(hidden_c)), act_fn)

    def edge_model(self, h, coord, edge_attr=None):
        batch_size, agent_num, _ = coord.shape[0], coord.shape[1], coord.shape[2]
        h1 = h[:, :, None, :].repeat(1, 1, agent_num, 1)
        h2 = h[:, None, :, :].repeat(1, agent_num, 1, 1)
        coord_diff = coord[:, :, None, :, :] - coord[:, None, :, :, :]
        coord_dist = torch.norm(coord_diff, dim=-1)
        edge_feat = torch.cat([h1, h2, coord_dist], dim=-1)
        edge_feat = self.edge_mlp(edge_feat)
        return edge_feat, coord_diff

    def aggregate_coord(self, coord, edge_feat, coord_diff):
        factors = self.coord_mlp(edge_feat).unsqueeze(-1)
        neighbor_effect = torch.sum(factors * coord_diff, dim=2)
        coord = coord + neighbor_effect
        return coord

    def aggregate_coord_reasoning(self, coord, edge_feat, coord_diff, category, h):
        batch_size, agent_num, _ = coord.shape[0], coord.shape[1], coord.shape[2]
        h1 = h[:, :, None, :].repeat(1, 1, agent_num, 1)
        h2 = h[:, None, :, :].repeat(1, agent_num, 1, 1)
        coord_dist = torch.norm(coord_diff, dim=-1)
        edge_h = torch.cat([h1, h2, coord_dist], dim=-1)
        coord_channels = coord.shape[2]
        factors = torch.zeros(batch_size, agent_num, agent_num, coord_channels).type_as(coord)
        for i in range(self.category_num):
            factors += category[:, :, :, i : i + 1] * self.category_mlp[i](edge_h)
        factors = self.factor_mlp(factors)

        factors = factors.unsqueeze(-1)
        neighbor_effect = torch.sum(factors * coord_diff, dim=2)
        coord = coord + neighbor_effect
        return coord

    def node_model(self, x, edge_feat):
        batch_size, agent_num = edge_feat.shape[0], edge_feat.shape[1]
        mask = (torch.ones((agent_num, agent_num)) - torch.eye(agent_num)).type_as(edge_feat)
        mask = mask[None, :, :, None].repeat(batch_size, 1, 1, 1)
        aggregated_edge = torch.sum(mask * edge_feat, dim=2)
        out = self.node_mlp(torch.cat([x, aggregated_edge], dim=-1))

        if self.recurrent:
            out = x + out
        return out

    def inner_agent_attention(self, coord, h):
        att = self.mlp_q(h).unsqueeze(-1)
        v = coord - torch.mean(coord, dim=(1, 2), keepdim=True)
        out = att * v
        if True:
            out = out + coord
        return out

    def non_linear(self, coord):
        coord_mean = torch.mean(coord, dim=(1, 2), keepdim=True)
        coord = coord - coord_mean
        q = self.layer_q(coord.transpose(2, 3)).transpose(2, 3)
        k = self.layer_k(coord.transpose(2, 3)).transpose(2, 3)
        product = torch.matmul(q.unsqueeze(-2), k.unsqueeze(-1)).squeeze(-1)
        mask = (product >= 0).float()
        eps = 1e-4
        k_norm_sq = torch.sum(k * k, dim=-1, keepdim=True)
        coord = mask * q + (1 - mask) * (q - (product / (k_norm_sq + eps)) * k)
        coord = coord + coord_mean
        return coord

    def forward(self, h, coord, vel, edge_attr=None, node_attr=None, category=None):
        edge_feat, coord_diff = self.edge_model(h, coord, edge_attr)

        if self.add_inner_agent_attention:
            coord = self.inner_agent_attention(coord, h)

        if self.input_reasoning:
            coord = self.aggregate_coord_reasoning(coord, edge_feat, coord_diff, category, h)
        else:
            coord = self.aggregate_coord(coord, edge_feat, coord_diff)

        coord += self.coord_vel(vel.transpose(2, 3)).transpose(2, 3)

        if self.add_non_linear:
            coord = self.non_linear(coord)

        h = self.node_model(h, edge_feat)

        return h, coord, category


class EqMotion(nn.Module):
    """EqMotion model copied from EqMotion repository."""

    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        in_channel,
        hid_channel,
        out_channel,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        coords_weight=1.0,
        recurrent=False,
        norm_diff=False,
        tanh=False,
    ):
        super(EqMotion, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        self.embedding = nn.Linear(in_node_nf, int(self.hidden_nf / 2))
        self.embedding2 = nn.Linear(in_node_nf, int(self.hidden_nf / 2))

        self.coord_trans = nn.Linear(in_channel, int(hid_channel), bias=False)
        self.vel_trans = nn.Linear(in_channel, int(hid_channel), bias=False)

        self.predict_head = nn.Linear(hid_channel, out_channel, bias=False)
        self.apply_dct = True
        self.validate_reasoning = True
        self.in_channel = in_channel
        self.out_channel = out_channel

        category_num = 2
        self.category_num = category_num
        self.tao = 0.5

        self.given_category = False
        if not self.given_category:
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_nf * 2 + hid_channel * 2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
            )

            self.coord_mlp = nn.Sequential(
                nn.Linear(hid_channel * 2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hid_channel * 2),
                act_fn,
            )

            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + int(1 * hidden_nf), hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn,
            )

            self.category_mlp = nn.Sequential(
                nn.Linear(hidden_nf * 2 + hid_channel * 2, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, category_num),
                act_fn,
            )

        for i in range(0, n_layers):
            self.add_module(
                f"gcl_{i}",
                Feature_learning_layer(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    in_channel,
                    hid_channel,
                    out_channel,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    coords_weight=coords_weight,
                    recurrent=recurrent,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    input_reasoning=True,
                ),
            )

        self.to(self.device)

    def get_dct_matrix(self, n, x):
        dct_m = np.eye(n)
        for k in np.arange(n):
            for i in np.arange(n):
                w = np.sqrt(2 / n)
                if k == 0:
                    w = np.sqrt(1 / n)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / n)
        idct_m = np.linalg.inv(dct_m)
        dct_m = torch.from_numpy(dct_m).type_as(x)
        idct_m = torch.from_numpy(idct_m).type_as(x)
        return dct_m, idct_m

    def transform_edge_attr(self, edge_attr):
        edge_attr = (edge_attr / 2) + 1
        interaction_category = F.one_hot(edge_attr.long(), num_classes=self.category_num)
        return interaction_category

    def calc_category(self, h, coord):
        batch_size, agent_num, channels = coord.shape[0], coord.shape[1], coord.shape[2]
        h1 = h[:, :, None, :].repeat(1, 1, agent_num, 1)
        h2 = h[:, None, :, :].repeat(1, agent_num, 1, 1)
        coord_diff = coord[:, :, None, :, :] - coord[:, None, :, :, :]
        coord_dist = torch.norm(coord_diff, dim=-1)
        coord_dist = self.coord_mlp(coord_dist)
        edge_feat_input = torch.cat([h1, h2, coord_dist], dim=-1)
        edge_feat = self.edge_mlp(edge_feat_input)
        mask = (torch.ones((agent_num, agent_num)) - torch.eye(agent_num)).type_as(edge_feat)
        mask = mask[None, :, :, None].repeat(batch_size, 1, 1, 1)
        node_new = self.node_mlp(torch.cat([h, torch.sum(mask * edge_feat, dim=2)], dim=-1))
        node_new1 = node_new[:, :, None, :].repeat(1, 1, agent_num, 1)
        node_new2 = node_new[:, None, :, :].repeat(1, agent_num, 1, 1)

        edge_feat_input_new = torch.cat([node_new1, node_new2, coord_dist], dim=-1)
        interaction_category = F.softmax(self.category_mlp(edge_feat_input_new) / self.tao, dim=-1)

        return interaction_category

    def forward(self, h, x, vel, edge_attr=None):
        vel_pre = torch.zeros_like(vel)
        vel_pre[:, :, 1:] = vel[:, :, :-1]
        vel_pre[:, :, 0] = vel[:, :, 0]
        eps = 1e-6
        vel_cosangle = torch.sum(vel_pre * vel, dim=-1) / (
            (torch.norm(vel_pre, dim=-1) + eps) * (torch.norm(vel, dim=-1) + eps)
        )

        vel_angle = torch.acos(torch.clamp(vel_cosangle, -1, 1))

        batch_size, agent_num, length = x.shape[0], x.shape[1], x.shape[2]
        if self.apply_dct:
            x_center = torch.mean(x, dim=(1, 2), keepdim=True)
            x = x - x_center
            dct_m, _ = self.get_dct_matrix(self.in_channel, x)
            _, idct_m = self.get_dct_matrix(self.out_channel, x)
            dct_m = dct_m[None, None, :, :].repeat(batch_size, agent_num, 1, 1)
            idct_m = idct_m[None, None, :, :].repeat(batch_size, agent_num, 1, 1)
            x = torch.matmul(dct_m, x)
            vel = torch.matmul(dct_m, vel)

        h = self.embedding(h)
        vel_angle_embedding = self.embedding2(vel_angle)
        h = torch.cat([h, vel_angle_embedding], dim=-1)
        x_mean = torch.mean(torch.mean(x, dim=-2, keepdim=True), dim=-3, keepdim=True)
        x = self.coord_trans((x - x_mean).transpose(2, 3)).transpose(2, 3) + x_mean
        vel = self.vel_trans(vel.transpose(2, 3)).transpose(2, 3)
        x_cat = torch.cat([x, vel], dim=-2)
        cagegory_per_layer = []
        if self.given_category:
            category = self.transform_edge_attr(edge_attr)
        else:
            category = self.calc_category(h, x_cat)

        for i in range(0, self.n_layers):
            h, x, _ = self._modules[f"gcl_{i}"](h, x, vel, edge_attr=edge_attr, category=category)
            cagegory_per_layer.append(category)
        x_mean = torch.mean(torch.mean(x, dim=-2, keepdim=True), dim=-3, keepdim=True)
        x = self.predict_head((x - x_mean).transpose(2, 3)).transpose(2, 3) + x_mean
        if self.apply_dct:
            x = torch.matmul(idct_m, x)
            x = x + x_center
        if self.validate_reasoning:
            return x, cagegory_per_layer
        else:
            return x, None


@dataclass
class EqMotionConfig:
    """Wrapper configuration for integrating EqMotion into MPT codebase."""

    input_length: int
    pred_length: int
    num_nodes: int
    hidden_dim: int
    hid_channel: int


class EqMotionForecaster(nn.Module):
    """Thin wrapper that adapts EqMotion outputs to (B, T_out, N, 3)."""

    def __init__(self, cfg: EqMotionConfig):
        super().__init__()
        self.cfg = cfg
        self.model = EqMotion(
            in_node_nf=cfg.input_length,
            in_edge_nf=2,
            hidden_nf=cfg.hidden_dim,
            in_channel=cfg.input_length,
            hid_channel=cfg.hid_channel,
            out_channel=cfg.pred_length,
            device="cpu",
            n_layers=4,
            recurrent=True,
            norm_diff=False,
            tanh=False,
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() != 4:
            raise ValueError("Expected history of shape (B, T, N, C).")
        if history.size(1) != self.cfg.input_length:
            raise ValueError(f"Expected input length {self.cfg.input_length}, got {history.size(1)}.")

        B, T, N, C = history.shape
        if C != 3:
            raise ValueError(f"EqMotionForecaster expects 3D coordinates, got {C}.")

        pos = history.permute(0, 2, 1, 3).contiguous()
        vel = torch.zeros_like(pos)
        vel[:, :, 1:] = pos[:, :, 1:] - pos[:, :, :-1]
        vel[:, :, 0] = vel[:, :, 1]
        node_feat = vel.norm(dim=-1)

        pred, _ = self.model(node_feat, pos, vel)
        pred = pred.permute(0, 2, 1, 3).contiguous()
        if pred.size(1) > self.cfg.pred_length:
            pred = pred[:, : self.cfg.pred_length]
        return pred
