import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, use_max_pooling: bool = False):
        super().__init__()
        pool_cls = nn.AdaptiveMaxPool1d if use_max_pooling else nn.AdaptiveAvgPool1d
        self.squeeze = pool_cls(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        y = self.squeeze(x).view(batch_size, seq_len)
        y = self.excitation(y).view(batch_size, seq_len, 1)
        return x * y.expand_as(x)


def mish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(F.softplus(x))


class MlpBlock(nn.Module):
    def __init__(
        self,
        mlp_hidden_dim: int,
        mlp_input_dim: int,
        mlp_bn_dim: int,
        activation: str = "gelu",
        regularization: float = 0.0,
        initialization: str = "none",
    ):
        super().__init__()
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_input_dim = mlp_input_dim
        self.mlp_bn_dim = mlp_bn_dim
        self.fc1 = nn.Linear(self.mlp_input_dim, self.mlp_hidden_dim)
        self.fc2 = nn.Linear(self.mlp_hidden_dim, self.mlp_input_dim)

        if regularization > 0.0:
            self.reg1 = nn.Dropout(regularization)
            self.reg2 = nn.Dropout(regularization)
        elif regularization == -1.0:
            self.reg1 = nn.BatchNorm1d(self.mlp_bn_dim)
            self.reg2 = nn.BatchNorm1d(self.mlp_bn_dim)
        else:
            self.reg1 = None
            self.reg2 = None

        if activation == "gelu":
            self.act1 = nn.GELU()
        elif activation == "mish":
            self.act1 = mish
        else:
            raise ValueError(f"Unknown activation function type: {activation}")

        if initialization != "none":
            raise ValueError(f"Unsupported initialization mode: {initialization}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act1(x)
        if self.reg1 is not None:
            x = self.reg1(x)
        x = self.fc2(x)
        if self.reg2 is not None:
            x = self.reg2(x)
        return x


class MixerBlock(nn.Module):
    def __init__(
        self,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        seq_len: int,
        hidden_dim: int,
        activation: str = "gelu",
        regularization: float = 0.0,
        initialization: str = "none",
        reduction: int = 4,
        use_max_pooling: bool = False,
        use_se: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.mlp_block_token = MlpBlock(
            tokens_mlp_dim,
            self.seq_len,
            self.hidden_dim,
            activation=activation,
            regularization=regularization,
            initialization=initialization,
        )
        self.mlp_block_channel = MlpBlock(
            channels_mlp_dim,
            self.hidden_dim,
            self.seq_len,
            activation=activation,
            regularization=regularization,
            initialization=initialization,
        )
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(self.seq_len, reduction=reduction, use_max_pooling=use_max_pooling)

        self.ln_tokens = nn.LayerNorm(self.hidden_dim)
        self.ln_channels = nn.LayerNorm(self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln_tokens(x)
        y = y.transpose(1, 2)
        y = self.mlp_block_token(y)
        y = y.transpose(1, 2)

        if self.use_se:
            y = self.se(y)
        x = x + y

        y = self.ln_channels(x)
        y = self.mlp_block_channel(y)
        if self.use_se:
            y = self.se(y)

        return x + y


class MixerBlockChannel(nn.Module):
    def __init__(
        self,
        channels_mlp_dim: int,
        seq_len: int,
        hidden_dim: int,
        activation: str = "gelu",
        regularization: float = 0.0,
        initialization: str = "none",
        reduction: int = 4,
        use_max_pooling: bool = False,
        use_se: bool = True,
    ):
        super().__init__()
        self.mlp_block_channel = MlpBlock(
            channels_mlp_dim,
            hidden_dim,
            seq_len,
            activation=activation,
            regularization=regularization,
            initialization=initialization,
        )
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(seq_len, reduction=reduction, use_max_pooling=use_max_pooling)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        if self.use_se:
            y = self.se(y)
        x = x + y
        y = self.ln(x)
        y = self.mlp_block_channel(y)
        if self.use_se:
            y = self.se(y)
        return x + y


class MixerBlockToken(nn.Module):
    def __init__(
        self,
        tokens_mlp_dim: int,
        seq_len: int,
        hidden_dim: int,
        activation: str = "gelu",
        regularization: float = 0.0,
        initialization: str = "none",
        reduction: int = 4,
        use_max_pooling: bool = False,
        use_se: bool = True,
    ):
        super().__init__()
        self.mlp_block_token = MlpBlock(
            tokens_mlp_dim,
            seq_len,
            hidden_dim,
            activation=activation,
            regularization=regularization,
            initialization=initialization,
        )
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(seq_len, reduction=reduction, use_max_pooling=use_max_pooling)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = y.transpose(1, 2)
        y = self.mlp_block_token(y)
        y = y.transpose(1, 2)
        if self.use_se:
            y = self.se(y)
        x = x + y
        return x + y


class MlpMixer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_blocks: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        seq_len: int,
        pred_len: int,
        activation: str = "gelu",
        mlp_block_type: str = "normal",
        regularization: float = 0.0,
        input_size: int = 51,
        initialization: str = "none",
        reduction: int = 4,
        use_max_pooling: bool = False,
        use_se: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.input_size = input_size
        self.pred_len = pred_len
        self.conv = nn.Conv2d(1, self.hidden_dim, (1, self.input_size), stride=1)
        self.channel_only = False
        self.token_only = False

        if mlp_block_type == "channel":
            self.channel_only = True
        elif mlp_block_type == "token":
            self.token_only = True

        if self.channel_only:
            blocks = (
                MixerBlockChannel(
                    self.channels_mlp_dim,
                    self.seq_len,
                    self.hidden_dim,
                    activation=activation,
                    regularization=regularization,
                    initialization=initialization,
                    reduction=reduction,
                    use_max_pooling=use_max_pooling,
                    use_se=use_se,
                )
                for _ in range(num_blocks)
            )
        elif self.token_only:
            blocks = (
                MixerBlockToken(
                    self.tokens_mlp_dim,
                    self.seq_len,
                    self.hidden_dim,
                    activation=activation,
                    regularization=regularization,
                    initialization=initialization,
                    reduction=reduction,
                    use_max_pooling=use_max_pooling,
                    use_se=use_se,
                )
                for _ in range(num_blocks)
            )
        else:
            blocks = (
                MixerBlock(
                    self.tokens_mlp_dim,
                    self.channels_mlp_dim,
                    self.seq_len,
                    self.hidden_dim,
                    activation=activation,
                    regularization=regularization,
                    initialization=initialization,
                    reduction=reduction,
                    use_max_pooling=use_max_pooling,
                    use_se=use_se,
                )
                for _ in range(num_blocks)
            )

        self.mixer_blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, self.num_classes)
        self.conv_out = nn.Conv1d(self.seq_len, self.pred_len, 1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        y = self.conv(x)
        y = y.squeeze(dim=3).transpose(1, 2)
        for block in self.mixer_blocks:
            y = block(y)
        y = self.ln(y)
        out = self.fc_out(self.conv_out(y))
        return out
