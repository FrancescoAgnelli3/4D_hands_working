import math
from typing import Optional

import torch
import torch.nn as nn


class DCTTemporalFilter(nn.Module):
    """Optional DCT projector inspired by HisRep's frequency handling.

    Projects the trailing window onto low-order DCT bases, applies learnable
    per-channel frequency gains, then blends the reconstructed signal with the
    original window through a sigmoid gate so the downstream stack can decide
    how much filtering it wants.
    """

    def __init__(
        self,
        in_channels: int,
        window_size: Optional[int] = None,
        keep_coeffs: int = 10,
        learnable: bool = True,
        input_scale: float = 1.0,
    ):
        super().__init__()
        if keep_coeffs <= 0:
            raise ValueError("keep_coeffs must be positive")
        self.in_channels = int(in_channels)
        self.window_size = int(window_size) if window_size is not None else None
        self.keep_coeffs = int(keep_coeffs)
        self.learnable = bool(learnable)
        self.input_scale = float(input_scale)

        gain_init = torch.zeros(self.in_channels, self.keep_coeffs)
        if self.learnable:
            self.gain_bias = nn.Parameter(gain_init)
        else:
            self.register_buffer("gain_bias", gain_init)

        # Per-channel gate initialised to favour the original signal.
        self.mix_logit = nn.Parameter(torch.full((self.in_channels,), -2.0))

    @staticmethod
    def _build_dct_matrix(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        n = torch.arange(length, device=device, dtype=dtype).unsqueeze(0)  # (1, L)
        k = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)  # (L, 1)
        mat = torch.cos((n + 0.5) * k * (math.pi / length))              # (L, L)
        weights = torch.full((length, 1), math.sqrt(2.0 / length), device=device, dtype=dtype)
        weights[0, 0] = math.sqrt(1.0 / length)
        return mat * weights

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        if Z.ndim != 4:
            raise ValueError("Expected input shape (B, T, N, C)")
        B, T, N, C = Z.shape
        if C != self.in_channels:
            raise ValueError(f"Expected last dimension {self.in_channels}, got {C}")

        window_len = self.window_size if self.window_size is not None else T
        window_len = max(1, min(window_len, T))
        coeffs_to_keep = min(self.keep_coeffs, window_len)

        dct_full = self._build_dct_matrix(window_len, Z.device, Z.dtype)
        dct_subset = dct_full[:coeffs_to_keep, :]  # (K, L)

        start = T - window_len
        window = Z[:, start:, :, :]  # (B, L, N, C)
        window = window.permute(0, 2, 3, 1).contiguous()  # (B, N, C, L)
        window_flat = window.view(-1, window_len)  # (B*N*C, L)

        scaled = window_flat * self.input_scale if self.input_scale != 1.0 else window_flat
        coeffs = scaled @ dct_subset.transpose(0, 1)  # (B*N*C, K)
        coeffs = coeffs.view(B, N, C, coeffs_to_keep)

        gain = 1.0 + self.gain_bias[..., :coeffs_to_keep]
        gain_dc = torch.relu(gain[..., :1]) + 1e-6
        if coeffs_to_keep > 1:
            gain = torch.cat([gain_dc, gain[..., 1:]], dim=-1)
        else:
            gain = gain_dc
        coeffs = coeffs * gain.view(1, 1, C, coeffs_to_keep)
        coeffs = coeffs.view(-1, coeffs_to_keep)

        filtered = coeffs @ dct_subset  # (B*N*C, L)
        filtered = filtered.view(B, N, C, window_len)
        filtered = filtered / self.input_scale if self.input_scale != 1.0 else filtered
        filtered = filtered.permute(0, 3, 1, 2).contiguous()  # (B, L, N, C)
        orig_window = window.permute(0, 3, 1, 2).contiguous()

        alpha = torch.sigmoid(self.mix_logit).view(1, 1, 1, C)
        mixed_window = alpha * filtered + (1.0 - alpha) * orig_window

        if start == 0:
            return mixed_window
        return torch.cat([Z[:, :start, :, :], mixed_window], dim=1)


__all__ = ["DCTTemporalFilter"]
