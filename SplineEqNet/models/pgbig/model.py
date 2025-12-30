from dataclasses import dataclass
from typing import Optional
import math

import torch
from torch import nn
from torch.nn import Parameter


def _build_dct_matrix(size: int) -> torch.Tensor:
    """Create an orthonormal DCT-II matrix."""
    device = torch.device("cpu")
    i = torch.arange(size, dtype=torch.float32, device=device)
    j = torch.arange(size, dtype=torch.float32, device=device).unsqueeze(1)
    mat = torch.cos(math.pi / size * (i + 0.5) * j)
    mat *= math.sqrt(2.0 / size)
    mat[0] /= math.sqrt(2.0)
    return mat


@dataclass
class PGBIGConfig:
    input_n: int
    output_n: int
    in_features: int
    d_model: int = 64
    dct_n: int = 20
    drop_out: float = 0.3
    encoder_layers: int = 1
    decoder_layers: int = 2
    kernel_size: Optional[int] = None


class GraphConvolution(nn.Module):
    def __init__(self, in_c: int, out_c: int, *, node_n: int, seq_len: int, bias: bool = True):
        super().__init__()
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        self.weight_seq = Parameter(torch.FloatTensor(seq_len, seq_len))
        self.weight_c = Parameter(torch.FloatTensor(in_c, out_c))
        if bias:
            self.bias = Parameter(torch.FloatTensor(seq_len))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.att.size(1))
        for param in (self.att, self.weight_seq, self.weight_c):
            param.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, N, L)
        support = torch.matmul(self.att, x.permute(0, 3, 2, 1))
        output_gcn = torch.matmul(support, self.weight_c)
        output_fc = torch.matmul(
            output_gcn.permute(0, 2, 3, 1),
            self.weight_seq
        ).permute(0, 2, 1, 3)
        if self.bias is not None:
            output_fc = output_fc + self.bias
        return output_fc


class GC_Block(nn.Module):
    def __init__(self, channal: int, *, p_dropout: float, node_n: int, seq_len: int, bias: bool = True):
        super().__init__()
        self.gc1 = GraphConvolution(channal, channal, node_n=node_n, seq_len=seq_len, bias=bias)
        self.bn1 = nn.BatchNorm1d(channal * node_n * seq_len)
        self.gc2 = GraphConvolution(channal, channal, node_n=node_n, seq_len=seq_len, bias=bias)
        self.bn2 = nn.BatchNorm1d(channal * node_n * seq_len)
        self.dropout = nn.Dropout(p_dropout)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gc1(x)
        b, c, n, l = y.shape
        y = self.bn1(y.reshape(b, -1)).reshape(b, c, n, l)
        y = self.dropout(self.activation(y))

        y = self.gc2(y)
        b, c, n, l = y.shape
        y = self.bn2(y.reshape(b, -1)).reshape(b, c, n, l)
        y = self.dropout(self.activation(y))
        return y + x


class GCN(nn.Module):
    def __init__(self, in_channal: int, out_channal: int, *, node_n: int, seq_len: int, p_dropout: float, num_stage: int):
        super().__init__()
        self.gc1 = GraphConvolution(in_channal, out_channal, node_n=node_n, seq_len=seq_len)
        self.bn1 = nn.BatchNorm1d(out_channal * node_n * seq_len)
        self.blocks = nn.ModuleList(
            GC_Block(out_channal, p_dropout=p_dropout, node_n=node_n, seq_len=seq_len) for _ in range(num_stage)
        )
        self.gc7 = GraphConvolution(out_channal, in_channal, node_n=node_n, seq_len=seq_len)
        self.dropout = nn.Dropout(p_dropout)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gc1(x)
        b, c, n, l = y.shape
        y = self.bn1(y.reshape(b, -1)).reshape(b, c, n, l)
        y = self.dropout(self.activation(y))
        for block in self.blocks:
            y = block(y)
        y = self.gc7(y)
        return y + x


class GCNEncoder(nn.Module):
    def __init__(self, in_channal: int, out_channal: int, *, node_n: int, seq_len: int, p_dropout: float, num_stage: int):
        super().__init__()
        self.gc1 = GraphConvolution(in_channal, out_channal, node_n=node_n, seq_len=seq_len)
        self.bn1 = nn.BatchNorm1d(out_channal * node_n * seq_len)
        self.blocks = nn.ModuleList(
            GC_Block(out_channal, p_dropout=p_dropout, node_n=node_n, seq_len=seq_len) for _ in range(num_stage)
        )
        self.gc7 = GraphConvolution(out_channal, out_channal, node_n=node_n, seq_len=seq_len)
        self.reshape_conv = nn.Conv2d(in_channels=in_channal, out_channels=out_channal, kernel_size=(1, 1))
        self.dropout = nn.Dropout(p_dropout)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gc1(x)
        b, c, n, l = y.shape
        y = self.bn1(y.reshape(b, -1)).reshape(b, c, n, l)
        y = self.dropout(self.activation(y))
        for block in self.blocks:
            y = block(y)
        y = self.gc7(y)
        return y + self.reshape_conv(x)


class GCNDecoder(nn.Module):
    def __init__(self, in_channal: int, out_channal: int, *, node_n: int, seq_len: int, p_dropout: float, num_stage: int):
        super().__init__()
        self.gc1 = GraphConvolution(in_channal, in_channal, node_n=node_n, seq_len=seq_len)
        self.bn1 = nn.BatchNorm1d(in_channal * node_n * seq_len)
        self.blocks = nn.ModuleList(
            GC_Block(in_channal, p_dropout=p_dropout, node_n=node_n, seq_len=seq_len) for _ in range(num_stage)
        )
        self.gc7 = GraphConvolution(in_channal, out_channal, node_n=node_n, seq_len=seq_len)
        self.reshape_conv = nn.Conv2d(in_channels=in_channal, out_channels=out_channal, kernel_size=(1, 1))
        self.dropout = nn.Dropout(p_dropout)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gc1(x)
        b, c, n, l = y.shape
        y = self.bn1(y.reshape(b, -1)).reshape(b, c, n, l)
        y = self.dropout(self.activation(y))
        for block in self.blocks:
            y = block(y)
        y = self.gc7(y) + self.reshape_conv(x)
        return y


class PGBIGMultiStageModel(nn.Module):
    def __init__(self, cfg: PGBIGConfig):
        super().__init__()
        self.cfg = cfg
        self.input_n = int(cfg.input_n)
        self.output_n = int(cfg.output_n)
        self.kernel_size = int(cfg.kernel_size) if cfg.kernel_size is not None else self.input_n
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        if self.kernel_size != self.input_n:
            raise ValueError("kernel_size must match input_n for this configuration.")
        self.in_features = int(cfg.in_features)
        if self.in_features % 3 != 0:
            raise ValueError("in_features must be divisible by 3 for XYZ coordinates.")
        self.node_n = self.in_features // 3
        total_frames = self.input_n + self.output_n
        if total_frames <= 0:
            raise ValueError("Total number of frames must be positive.")
        self.dct_n = min(int(cfg.dct_n), total_frames)
        self.encoder_layers = int(cfg.encoder_layers)
        self.decoder_layers = int(cfg.decoder_layers)
        self.d_model = int(cfg.d_model)
        self.drop_out = float(cfg.drop_out)

        dct = _build_dct_matrix(total_frames)
        self.register_buffer("dct_m", dct[: self.dct_n])
        self.register_buffer("idct_m", dct.t())

        stage_count = 4
        self.encoders = nn.ModuleList(
            GCNEncoder(
                in_channal=3,
                out_channal=self.d_model,
                node_n=self.node_n,
                seq_len=self.dct_n,
                p_dropout=self.drop_out,
                num_stage=self.encoder_layers,
            )
            for _ in range(stage_count)
        )
        self.decoders = nn.ModuleList(
            GCNDecoder(
                in_channal=self.d_model,
                out_channal=3,
                node_n=self.node_n,
                seq_len=self.dct_n * 2,
                p_dropout=self.drop_out,
                num_stage=self.decoder_layers,
            )
            for _ in range(stage_count)
        )
        index = list(range(self.kernel_size))
        if self.output_n > 0:
            index.extend([self.kernel_size - 1] * self.output_n)
        self.register_buffer("input_index", torch.tensor(index, dtype=torch.long))

    def forward(self, src: torch.Tensor, *, input_n: Optional[int] = None, output_n: Optional[int] = None):
        if src.dim() != 3:
            raise ValueError(f"Expected src with shape (B, T, F); got {tuple(src.shape)}")
        bs, seq_len, feat = src.shape
        if feat != self.in_features:
            raise ValueError(f"Expected feature size {self.in_features}, got {feat}")
        if seq_len < self.kernel_size:
            raise ValueError(f"Sequence length {seq_len} is shorter than kernel_size {self.kernel_size}")
        _ = input_n  # unused; kept for API parity
        _ = output_n

        idx = self.input_index[: self.kernel_size + self.output_n]
        idx = torch.clamp(idx, max=seq_len - 1)
        input_gcn = torch.index_select(src, 1, idx)

        dct_m = self.dct_m.to(device=src.device, dtype=src.dtype)
        idct_m = self.idct_m.to(device=src.device, dtype=src.dtype)

        input_gcn_dct = torch.matmul(dct_m, input_gcn).permute(0, 2, 1)
        input_gcn_dct = input_gcn_dct.reshape(bs, self.node_n, 3, self.dct_n).permute(0, 2, 1, 3)

        stage_inputs = input_gcn_dct
        stage_outputs = []
        for encoder, decoder in zip(self.encoders, self.decoders):
            latent = encoder(stage_inputs)
            latent = torch.cat((latent, latent), dim=3)
            out_dct = decoder(latent)[:, :, :, : self.dct_n]
            stage_outputs.append(out_dct)
            stage_inputs = out_dct

        time_outputs = []
        for dct_out in stage_outputs:
            flat = dct_out.permute(0, 2, 1, 3).reshape(bs, -1, self.dct_n)
            time_seq = torch.matmul(idct_m[:, : self.dct_n], flat.permute(0, 2, 1))
            time_outputs.append(time_seq)

        # Return in reverse order to match the original implementation signature.
        return tuple(time_outputs[::-1])
