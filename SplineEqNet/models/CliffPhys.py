#================================================================================================================================
#==            'CliffPhys: Camera-based Respiratory Measurement using Clifford Neural Networks' (Paper ID #11393)              ==
#================================================================================================================================

"""
Code containing functions and classes for the definition of the CliffPhys family of prediction models.

CLIFFORD:
    This script contains utility functions and classes for the definition of the CLI. 
    It includes implementations of the Clifford product for different algebra signatures (Clifford kernels), 
    the implementation of the Linear and 2D, 3D convolutional Clifford layers.
    It also provides the Negative Pearson loss implementation and the Processor() class to use any CliffPhys model in prediction mode.
    It provides the implementation of each model in the CliffPhys family, 4 models working using depth information, 4 depth-lacking models.

        MODELS:                                                     training version choices: 
                'CliffPhys02_d'         'CliffPhys02'                                 'PT-scamps_XYZ_FT-cohface_XYZ'                   
                'CliffPhys03_d'         'CliffPhys03'
                'CliffPhys30_d'         'CliffPhys30'
                'CliffPhys20_d'         'CliffPhys20'
"""


import os
import numpy as np
import torch
import torch.nn as nn
import os
import numpy as np
import math
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple
from typing import Callable, Optional, Tuple, Union

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Neg_Pearson(nn.Module):
    """
    Custom PyTorch module to compute the negative Pearson correlation coefficient loss.

    Attributes:
        None
    """
    def __init__(self):
        super(Neg_Pearson, self).__init__()
        return

    def forward(self, preds, labels):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        pearson = cos(preds - preds.mean(dim=0, keepdim=True), labels - labels.mean(dim=0, keepdim=True))
        return torch.mean(1 - pearson)

    
def _w_assert(w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]) -> torch.Tensor:
    """Convert Clifford weights to tensor .
    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Clifford weights.
    Raises:
        ValueError: Unknown weight type.
    Returns:
        torch.Tensor: Clifford weights as torch.Tensor.
    """
    if type(w) in (tuple, list):
        w = torch.stack(w)
        return w
    elif isinstance(w, torch.Tensor):
        return w
    elif isinstance(w, nn.Parameter):
        return w
    elif isinstance(w, nn.ParameterList):
        return w
    else:
        raise ValueError("Unknown weight type.")

class CliffordSignature:
    def __init__(self, g: Union[tuple, list, torch.Tensor]):
        super().__init__()
        self.g = self._g_tensor(g)
        self.dim = self.g.numel()
        if self.dim == 1:
            self.n_blades = 2
        elif self.dim == 2:
            self.n_blades = 4
        elif self.dim == 3:
            self.n_blades = 8
        else:
            raise NotImplementedError("Wrong Clifford signature.")

    def _g_tensor(self, g: Union[tuple, list, torch.Tensor]) -> torch.Tensor:
        """Convert Clifford signature to tensor.
        Args:
            g (Union[tuple, list, torch.Tensor]): Clifford signature.
        Raises:
            ValueError: Unknown metric.
        Returns:
            torch.Tensor: Clifford signature as torch.Tensor.
        """
        if type(g) in (tuple, list):
            g = torch.as_tensor(g, dtype=torch.float32)
        elif isinstance(g, torch.Tensor):
            pass
        else:
            raise ValueError("Unknown signature.")
        if not torch.any(abs(g) == 1.0):
            raise ValueError("Clifford signature should have at least one element as 1.")
        return g

def get_2d_clifford_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Clifford kernel for 2d Clifford algebras, g = [-1, -1] corresponds to a quaternion kernel.

    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(4, d~input~, d~output~, ...)`.
        g (torch.Tensor): Signature of Clifford algebra.

    Raises:
        ValueError: Wrong encoding/decoding options provided.

    Returns:
        (Tuple[int, torch.Tensor]): Number of output blades, weight output of shape `(d~output~ * 4, d~input~ * 4, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 2
    w = _w_assert(w)
    assert len(w) == 4

    k0 = torch.cat([w[0], g[0] * w[1], g[1] * w[2], -g[0] * g[1] * w[3]], dim=1)
    k1 = torch.cat([w[1], w[0], -g[1] * w[3], g[1] * w[2]], dim=1)
    k2 = torch.cat([w[2], g[0] * w[3], w[0], -g[0] * w[1]], dim=1)
    k3 = torch.cat([w[3], w[2], -w[1], w[0]], dim=1)
    k = torch.cat([k0, k1, k2, k3], dim=0)
    return 4, k

def get_3d_clifford_kernel(
    w: Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList], g: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """Clifford kernel for 3d Clifford algebras, g = [-1, -1, -1] corresponds to an octonion kernel.
    Args:
        w (Union[tuple, list, torch.Tensor, nn.Parameter, nn.ParameterList]): Weight input of shape `(8, d~input~, d~output~, ...)`.
        g (torch.Tensor): Signature of Clifford algebra.

    Raises:
        ValueError: Wrong encoding/decoding options provided.

    Returns:
        (Tuple[int, torch.Tensor]): Number of output blades, weight output of dimension `(d~output~ * 8, d~input~ * 8, ...)`.
    """
    assert isinstance(g, torch.Tensor)
    assert g.numel() == 3
    w = _w_assert(w)
    assert len(w) == 8

    k0 = torch.cat([w[0], w[1] * g[0], w[2] * g[1], w[3] * g[2], -w[4] * g[0] * g[1], -w[5] * g[0] * g[2], -w[6] * g[1] * g[2], -w[7] * g[0] * g[1] * g[2],], dim=1,)
    k1 = torch.cat([w[1], w[0], -w[4] * g[1], -w[5] * g[2], w[2] * g[1], w[3] * g[2], -w[7] * g[1] * g[2], -w[6] * g[2] * g[1]], dim=1,)
    k2 = torch.cat([w[2], w[4] * g[0], w[0], -w[6] * g[2], -w[1] * g[0], w[7] * g[0] * g[2], w[3] * g[2], w[5] * g[2] * g[0]], dim=1,)
    k3 = torch.cat([w[3], w[5] * g[0], w[6] * g[1], w[0], -w[7] * g[0] * g[1], -w[1] * g[0], -w[2] * g[1], -w[4] * g[0] * g[1]], dim=1,)
    k4 = torch.cat([w[4], w[2], -w[1], g[2] * w[7], w[0], -w[6] * g[2], w[5] * g[2], w[3] * g[2]], dim=1)
    k5 = torch.cat([w[5], w[3], -w[7] * g[1], -w[1], w[6] * g[1], w[0], -w[4] * g[1], -w[2] * g[1]], dim=1)
    k6 = torch.cat([w[6], w[7] * g[0], w[3], -w[2], -w[5] * g[0], w[4] * g[0], w[0], w[1] * g[0]], dim=1)
    k7 = torch.cat([w[7], w[6], -w[5], w[4], w[3], -w[2], w[1], w[0]], dim=1)
    k = torch.cat([k0, k1, k2, k3, k4, k5, k6, k7], dim=0)
    return 8, k

def clifford_convnd(conv_fn: Callable, x: torch.Tensor, output_blades: int, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, **kwargs,
                    ) -> torch.Tensor:
    """Apply a Clifford convolution to a tensor.
    Args:
        conv_fn (Callable): The convolution function to use.
        x (torch.Tensor): Input tensor.
        output_blades (int): The output blades of the Clifford algebra.
        Different from the default n_blades when using encoding and decoding layers.
        weight (torch.Tensor): Weight tensor.
        bias (torch.Tensor, optional): Bias tensor. Defaults to None.
    Returns:
        torch.Tensor: Convolved output tensor.
    """
    # Reshape x such that the convolution function can be applied.
    #print('x shape: '+str(x.shape))
    B, *_ = x.shape
    B_dim, C_dim, *D_dims, I_dim = range(len(x.shape))
    x = x.permute(B_dim, -1, C_dim, *D_dims)
    x = x.reshape(B, -1, *x.shape[3:])
    # Apply convolution function
    output = conv_fn(x, weight, bias=bias, **kwargs)
    #print('\n weights shape: '+str(weight.shape))
    # Reshape back.
    output = output.view(B, output_blades, -1, *output.shape[2:])
    B_dim, I_dim, C_dim, *D_dims = range(len(output.shape))
    output = output.permute(B_dim, C_dim, *D_dims, I_dim)
    return output


class CliffordLinear(nn.Module):
    """Clifford linear layer.
    Args:
        g (Union[List, Tuple]): Clifford signature tensor.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
    """

    def __init__(self, g, in_channels: int, out_channels: int, bias: bool = True,) -> None:
        super().__init__()
        sig = CliffordSignature(g)

        self.register_buffer("g", sig.g)
        self.dim = sig.dim
        self.n_blades = sig.n_blades

        if self.dim == 2:
            self._get_kernel = get_2d_clifford_kernel
        elif self.dim == 3:
            self._get_kernel = get_3d_clifford_kernel
        else:
            raise NotImplementedError(
                f"Clifford linear layers are not implemented for {self.dim} dimensions. Wrong Clifford signature."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(self.n_blades, out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.n_blades, out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization of the Clifford linear weight and bias tensors.
        # The number of blades is taken into account when calculated the bounds of Kaiming uniform.
        nn.init.kaiming_uniform_(
            self.weight.view(self.out_channels, self.in_channels * self.n_blades),
            a=math.sqrt(5),
        )
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.weight.view(self.out_channels, self.in_channels * self.n_blades)
            )
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape x such that the Clifford kernel can be applied.
        B, _, I = x.shape
        if not (I == self.n_blades):
            raise ValueError(f"Input has {I} blades, but Clifford layer expects {self.n_blades}.")
        B_dim, C_dim, I_dim = range(len(x.shape))
        x = x.permute(B_dim, -1, C_dim)
        x = x.reshape(B, -1)
        # Get Clifford kernel, apply it.
        _, weight = self._get_kernel(self.weight, self.g)
        bias = self.bias.view(-1) if self.bias is not None else None
        output = F.linear(x, weight, bias)
        # Reshape back.
        output = output.view(B, I, -1)
        B_dim, I_dim, C_dim = range(len(output.shape))
        output = output.permute(B_dim, C_dim, I_dim)
        return output

class CliffMLP(nn.Module): 

    """CliffordPhys30 model.

    Args:
        model_params (dict): Dictionary containing model parameters.
        dropout_rate1 (float, optional): Dropout rate for the first dropout layer. Defaults to 0.25.
        dropout_rate2 (float, optional): Dropout rate for the second dropout layer. Defaults to 0.5.
        pool_size (Tuple, optional): Size of the pooling window. Defaults to (2, 2).
        device (str, optional): Device to run the model on. Defaults to 'cpu'.
    """

    def __init__(self, algebra, num_layers, dropout_rate1=0.25, dropout_rate2=0.5):

        super(CliffMLP, self).__init__()

        self.in_channels = 100
        self.num_layers = num_layers

        self.g = algebra

        blocks = []

        # Dropout layers
        half = num_layers // 2
        channels = [self.in_channels]
        for _ in range(half):
            channels.append(max(1, channels[-1] // 2))
        for _ in range(num_layers - half):
            channels.append(channels[-1] * 2)
        channels[-1] = self.in_channels

        for l in range(num_layers):
            blocks.append(CliffordLinear(self.g, channels[l], channels[l + 1], bias=False))
    
        self.net = nn.Sequential(*blocks)

        self.readout = nn.Linear(8, 3, bias = True)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        B, K, C = x.size()

        x = x.permute(0, 2, 1).contiguous()
        x = x.reshape(B, -1, 3, K).permute(0, 1, 3, 2).contiguous()

        B, N, K, _ = x.size()

        scalar = torch.zeros(B, N, K, 1).to(x.device)
        multiv = torch.zeros(B, N, K, 4).to(x.device)
        m = torch.cat((scalar, x, multiv), dim=-1)

        Q = m.size(-1)
        m = m.reshape(B*N, K, Q)

        m = self.net(m)

        out = self.readout(m.reshape(B * N * K, Q))
        out = out.reshape(B, N, K, 3)
        return out.permute(0, 2, 1, 3).reshape(B, K, N * 3)
