#!/usr/bin/env python3
"""
Two-stage DCT coarse + conditional diffusion refiner (TIME DOMAIN diffusion).

Drop-in replacement for the previous `two_stage_dct_diffusion.py`, keeping the same
public API expected by your training code:

- TwoStageDCTDiffusionConfig
- TwoStageDCTDiffusionForecaster with:
    - .coarse(history) -> (B, T_out, N, 3)
    - .diffusion_loss(history, future_gt) -> (loss, coarse_future)
    - .predict(history, deterministic=True, seed=0) -> refined (B, T_out, N, 3)
    - .forward(history): returns coarse in train mode, refined in eval mode (unchanged behavior)

Stage A (coarse): unchanged: predicts low-frequency DCT content of the total sequence
                 and reconstructs a coarse future in time domain.

Stage B (refiner): diffusion is now trained/sampled ONLY in time domain on the residual:
    r = y_gt - y_coarse
Conditioning: history (time-domain) + coarse future (time-domain).
No frequency-domain targets are used in the diffusion stage.

This file assumes your project already provides:
- models.simlpe_dct.SiMLPeConfig, SiMLPeBackbone, _build_dct_matrix
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from models.simlpe_dct import SiMLPeConfig, SiMLPeBackbone, _build_dct_matrix


# ----------------------------- Small utilities --------------------------------

def _assert_shape(t: torch.Tensor, shape: Tuple[Optional[int], ...], name: str) -> None:
    if t.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dims, got {t.ndim} with shape {tuple(t.shape)}")
    for i, s in enumerate(shape):
        if s is not None and int(t.shape[i]) != int(s):
            raise ValueError(f"{name} dim {i} expected {s}, got {t.shape[i]}")

def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    if timesteps.ndim != 1:
        timesteps = timesteps.view(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, device=timesteps.device, dtype=torch.float32)
        / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(emb.size(0), 1, device=emb.device, dtype=emb.dtype)], dim=1)
    return emb


# ----------------------------- Configs ----------------------------------------

@dataclass
class TwoStageDCTDiffusionConfig:
    input_length: int
    pred_length: int
    num_nodes: int

    # Coarse backbone (siMLPe-like)
    hidden_dim: int
    num_layers: int
    simlpe_use_norm: bool = True
    simlpe_spatial_fc_only: bool = False
    simlpe_mix_spatial_temporal: bool = False
    simlpe_norm_axis: str = "spatial"
    simlpe_add_last_offset: bool = True

    # DCT split (still used by coarse)
    k_low: int = 16

    # Diffusion
    diffusion_steps: int = 100
    ddim_steps: int = 50
    beta_schedule: str = "cosine"  # "cosine" or "linear"

    # Denoiser architecture
    denoiser_dim: int = 256
    denoiser_depth: int = 6
    denoiser_heads: int = 8
    dropout: float = 0.0

    # Training
    freeze_coarse: bool = True
    stopgrad_coarse_condition: bool = True  # stop-grad coarse conditioning into diffusion during training

    @property
    def total_length(self) -> int:
        return int(self.input_length + self.pred_length)

    @property
    def feature_dim(self) -> int:
        return int(self.num_nodes * 3)

    # Kept for backward-compat; no longer used by diffusion stage
    @property
    def k_high(self) -> int:
        return int(max(0, self.total_length - self.k_low))


# ----------------------------- Stage A ----------------------------------------

class CoarseDCTForecaster(nn.Module):
    """Predict low-frequency DCT coefficients and reconstruct a coarse future."""

    def __init__(self, cfg: TwoStageDCTDiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        T = cfg.total_length
        if not (1 <= cfg.k_low <= T):
            raise ValueError(f"k_low must be in [1, {T}], got {cfg.k_low}")

        # DCT matrices for total sequence.
        dct = _build_dct_matrix(T)          # (T, T) orthonormal
        idct = dct.t()                      # (T, T)
        self.register_buffer("_dct_low", dct[: cfg.k_low, :], persistent=False)      # (K_low, T)
        self.register_buffer("_idct_low", idct[:, : cfg.k_low], persistent=False)   # (T, K_low)

        # siMLPe backbone works over sequences of length T and feature_dim channels.
        sim_cfg = SiMLPeConfig(
            input_length=cfg.input_length,
            pred_length=cfg.pred_length,
            num_nodes=cfg.num_nodes,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            use_norm=cfg.simlpe_use_norm,
            use_spatial_fc_only=cfg.simlpe_spatial_fc_only,
            mix_spatial_temporal=cfg.simlpe_mix_spatial_temporal,
            norm_axis=cfg.simlpe_norm_axis,  # type: ignore[arg-type]
            add_last_offset=cfg.simlpe_add_last_offset,
        )
        self.backbone = SiMLPeBackbone(sim_cfg)

    def _pad_total(self, history: torch.Tensor) -> torch.Tensor:
        # history: (B, T_in, N, 3) -> (B, T_in, F), pad to total length by repeating last frame
        _assert_shape(history, (None, self.cfg.input_length, self.cfg.num_nodes, 3), "history")
        B = history.size(0)
        flat = history.reshape(B, self.cfg.input_length, -1)
        last = flat[:, -1:, :].expand(-1, self.cfg.pred_length, -1)
        padded = torch.cat([flat, last], dim=1)  # (B, T_total, F)
        return padded

    def predict_low_coeffs(self, history: torch.Tensor) -> torch.Tensor:
        """Return predicted low-band DCT coefficients for the total sequence. Shape: (B, K_low, F)."""
        padded = self._pad_total(history)  # (B, T, F)
        dct_low = self._dct_low.unsqueeze(0).to(device=padded.device, dtype=padded.dtype)  # (1, K_low, T)
        low_coeffs = torch.matmul(dct_low, padded)  # (B, K_low, F)

        # Build full-length coefficient tensor for the backbone (zeros for high bands).
        B, _, F = low_coeffs.shape
        coeffs_full = torch.zeros(B, self.cfg.total_length, F, device=padded.device, dtype=padded.dtype)
        coeffs_full[:, : self.cfg.k_low, :] = low_coeffs

        pred_full = self.backbone(coeffs_full)  # (B, T, F) interpreted as coefficient sequence
        pred_low = pred_full[:, : self.cfg.k_low, :]  # (B, K_low, F)
        return pred_low

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Return coarse future in time domain. Shape: (B, T_out, N, 3)."""
        pred_low = self.predict_low_coeffs(history)  # (B, K_low, F)

        idct_low = self._idct_low.unsqueeze(0).to(device=pred_low.device, dtype=pred_low.dtype)  # (1, T, K_low)
        seq_low = torch.matmul(idct_low, pred_low)  # (B, T, F)

        future = seq_low[:, -self.cfg.pred_length :, :].reshape(
            history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3
        )

        if self.cfg.simlpe_add_last_offset:
            future = future + history[:, -1:, :, :]
        return future


# ----------------------------- Diffusion scheduler ----------------------------

class DiffusionSchedule:
    def __init__(self, steps: int, schedule: str = "cosine", device: Optional[torch.device] = None) -> None:
        if steps <= 1:
            raise ValueError("diffusion steps must be > 1")
        self.steps = int(steps)
        self.schedule = str(schedule)
        self.device = device

        betas = self._make_betas(self.steps, self.schedule)
        self.betas = betas.to(device=device)  # (T,)
        alphas = 1.0 - self.betas
        self.alphas = alphas
        self.alpha_bars = torch.cumprod(alphas, dim=0)  # (T,)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    @staticmethod
    def _make_betas(T: int, schedule: str) -> torch.Tensor:
        if schedule == "linear":
            beta_start = 1e-4
            beta_end = 2e-2
            return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        if schedule == "cosine":
            s = 0.008
            steps = T
            x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
            alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 1e-8, 0.999).float()
            return betas
        raise ValueError(f"Unknown beta schedule: {schedule}")


# ----------------------------- Denoiser (multi-rate) --------------------------

class _Downsample1D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1, groups=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv(x)
        return x.transpose(1, 2)

class _Upsample1D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, groups=1)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, D, L)
        x = torch.nn.functional.interpolate(x, size=target_len, mode="linear", align_corners=False)
        x = self.conv(x)
        return x.transpose(1, 2)

class _FiLM(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, dim)
        self.to_shift = nn.Linear(cond_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale = self.to_scale(cond).unsqueeze(1)
        shift = self.to_shift(cond).unsqueeze(1)
        return x * (1.0 + scale) + shift


class MRTimeTransformerDenoiser(nn.Module):
    """
    Denoiser for TIME-DOMAIN residual diffusion.

    Query tokens: noisy residual sequence over future horizon: (B, T_out, F).
    Conditioning tokens: concat(history, coarse_future): (B, T_in + T_out, F).
    """

    def __init__(
        self,
        *,
        in_feat: int,
        t_in: int,
        t_out: int,
        d_model: int,
        depth: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_feat = int(in_feat)
        self.t_in = int(t_in)
        self.t_out = int(t_out)
        self.d_model = int(d_model)

        # Projections
        self.proj_q = nn.Linear(in_feat, d_model)
        self.proj_hist = nn.Linear(in_feat, d_model)
        self.proj_coarse = nn.Linear(in_feat, d_model)

        # Learned positional embeddings
        self.pos_q = nn.Parameter(torch.zeros(1, t_out, d_model))
        self.pos_hist = nn.Parameter(torch.zeros(1, t_in, d_model))
        self.pos_coarse = nn.Parameter(torch.zeros(1, t_out, d_model))

        # Timestep embedding + FiLM
        self.t_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.film_blocks = nn.ModuleList([_FiLM(d_model, d_model) for _ in range(depth)])

        # Conditioning encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cond_encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, depth // 2))

        # Decoder-like blocks (self-attn + cross-attn)
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])

        # Multi-rate path over the future length axis
        self.down = _Downsample1D(d_model)
        self.up = _Upsample1D(d_model)

        # Output projection to predicted noise in residual space
        self.out = nn.Linear(d_model, in_feat)

        nn.init.normal_(self.pos_q, std=0.02)
        nn.init.normal_(self.pos_hist, std=0.02)
        nn.init.normal_(self.pos_coarse, std=0.02)

    def forward(
        self,
        r_noisy: torch.Tensor,      # (B, T_out, F)
        timesteps: torch.Tensor,    # (B,)
        history: torch.Tensor,      # (B, T_in, F)
        coarse_future: torch.Tensor # (B, T_out, F)
    ) -> torch.Tensor:
        B = r_noisy.size(0)
        _assert_shape(r_noisy, (B, self.t_out, self.in_feat), "r_noisy")
        _assert_shape(history, (B, self.t_in, self.in_feat), "history_flat")
        _assert_shape(coarse_future, (B, self.t_out, self.in_feat), "coarse_future_flat")

        # Conditioning tokens
        h_hist = self.proj_hist(history) + self.pos_hist
        h_coarse = self.proj_coarse(coarse_future) + self.pos_coarse
        cond_tokens = torch.cat([h_hist, h_coarse], dim=1)  # (B, T_in + T_out, D)
        cond_ctx = self.cond_encoder(cond_tokens)

        # Query tokens
        h = self.proj_q(r_noisy) + self.pos_q  # (B, T_out, D)

        # Time embedding
        te = _timestep_embedding(timesteps, self.d_model).to(dtype=h.dtype, device=h.device)
        te = self.t_embed(te)  # (B, D)

        # Multi-rate branch
        h_down = self.down(h)

        for i, block in enumerate(self.blocks):
            h = self.film_blocks[i](h, te)
            h_down = self.film_blocks[i](h_down, te)

            h = block(tgt=h, memory=cond_ctx)
            h_down = block(tgt=h_down, memory=cond_ctx)

            h_up = self.up(h_down, target_len=h.size(1))
            h = h + 0.5 * h_up

        eps_hat = self.out(h)  # (B, T_out, F)
        return eps_hat


# ----------------------------- Diffusion module (time-domain residual) ---------

class TimeResidualConditionalDiffusion(nn.Module):
    def __init__(self, cfg: TwoStageDCTDiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.schedule = DiffusionSchedule(cfg.diffusion_steps, cfg.beta_schedule)

        self.denoiser = MRTimeTransformerDenoiser(
            in_feat=cfg.feature_dim,
            t_in=cfg.input_length,
            t_out=cfg.pred_length,
            d_model=cfg.denoiser_dim,
            depth=cfg.denoiser_depth,
            n_heads=cfg.denoiser_heads,
            dropout=cfg.dropout,
        )

    def _schedule_tensors(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.schedule.alpha_bars.to(device=device, dtype=dtype),
            self.schedule.sqrt_alpha_bars.to(device=device, dtype=dtype),
            self.schedule.sqrt_one_minus_alpha_bars.to(device=device, dtype=dtype),
        )

    def training_loss(
        self,
        *,
        residual_gt: torch.Tensor,      # (B, T_out, F)
        history_flat: torch.Tensor,     # (B, T_in, F)
        coarse_future_flat: torch.Tensor# (B, T_out, F)
    ) -> torch.Tensor:
        B = residual_gt.size(0)
        _assert_shape(residual_gt, (B, self.cfg.pred_length, self.cfg.feature_dim), "residual_gt")

        device = residual_gt.device
        alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars = self._schedule_tensors(device, residual_gt.dtype)
        t = torch.randint(0, self.cfg.diffusion_steps, (B,), device=device, dtype=torch.long)
        eps = torch.randn_like(residual_gt)

        sqrt_ab = sqrt_alpha_bars[t].view(B, 1, 1)
        sqrt_omb = sqrt_one_minus_alpha_bars[t].view(B, 1, 1)

        r_t = sqrt_ab * residual_gt + sqrt_omb * eps

        eps_hat = self.denoiser(r_t, t, history_flat, coarse_future_flat)
        return torch.mean((eps - eps_hat) ** 2)

    @torch.no_grad()
    def sample_ddim(
        self,
        *,
        history_flat: torch.Tensor,       # (B, T_in, F)
        coarse_future_flat: torch.Tensor, # (B, T_out, F)
        seed: int = 0,
        steps: Optional[int] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM sampling for residual sequence (eta=0 -> deterministic)."""
        if steps is None:
            steps = self.cfg.ddim_steps
        steps = int(steps)
        if steps <= 0:
            raise ValueError("DDIM steps must be positive")

        B = history_flat.size(0)
        device = history_flat.device
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

        x = torch.randn(
            (B, self.cfg.pred_length, self.cfg.feature_dim),
            device=device,
            dtype=history_flat.dtype,
            generator=gen,
        )

        T = self.cfg.diffusion_steps
        t_seq = torch.linspace(T - 1, 0, steps, device=device).long()

        alpha_bars, _, _ = self._schedule_tensors(device, history_flat.dtype)

        for idx in range(len(t_seq)):
            t = int(t_seq[idx].item())
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            ab_t = alpha_bars[t]
            sqrt_ab_t = torch.sqrt(ab_t)
            sqrt_omb_t = torch.sqrt(1.0 - ab_t)

            eps_hat = self.denoiser(x, t_batch, history_flat, coarse_future_flat)

            x0 = (x - sqrt_omb_t * eps_hat) / (sqrt_ab_t + 1e-12)

            if idx == len(t_seq) - 1:
                ab_prev = torch.tensor(1.0, device=device, dtype=history_flat.dtype)
            else:
                t_prev = int(t_seq[idx + 1].item())
                ab_prev = alpha_bars[t_prev]

            sqrt_ab_prev = torch.sqrt(ab_prev)
            sqrt_omb_prev = torch.sqrt(1.0 - ab_prev)

            if eta != 0.0:
                sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))
                noise = torch.randn_like(x, generator=gen)
                x = (
                    sqrt_ab_prev * x0
                    + torch.sqrt(torch.clamp(sqrt_omb_prev**2 - sigma**2, min=0.0)) * eps_hat
                    + sigma * noise
                )
            else:
                x = sqrt_ab_prev * x0 + sqrt_omb_prev * eps_hat

        return x  # predicted residual sequence (approx x0 at end)


# ----------------------------- Wrapper model ----------------------------------

class TwoStageDCTDiffusionForecaster(nn.Module):
    def __init__(self, cfg: TwoStageDCTDiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.coarse = CoarseDCTForecaster(cfg)
        self.diffusion = TimeResidualConditionalDiffusion(cfg)

        if cfg.freeze_coarse:
            for p in self.coarse.parameters():
                p.requires_grad = False

    def _flat_history(self, history: torch.Tensor) -> torch.Tensor:
        _assert_shape(history, (None, self.cfg.input_length, self.cfg.num_nodes, 3), "history")
        return history.reshape(history.size(0), self.cfg.input_length, -1)

    def _flat_future(self, future: torch.Tensor) -> torch.Tensor:
        _assert_shape(future, (None, self.cfg.pred_length, self.cfg.num_nodes, 3), "future")
        return future.reshape(future.size(0), self.cfg.pred_length, -1)

    def diffusion_loss(self, history: torch.Tensor, future_gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute diffusion training loss (on time-domain residuals) and also return the coarse prediction for logging.
        Signature unchanged for your training code.
        """
        history_flat = self._flat_history(history)  # (B, T_in, F)

        # Coarse future in time domain
        coarse_future = self.coarse(history)  # (B, T_out, N, 3)
        coarse_future_flat = coarse_future.reshape(history.size(0), self.cfg.pred_length, -1)  # (B, T_out, F)

        # Optionally stop-grad conditioning into diffusion
        if self.cfg.stopgrad_coarse_condition:
            coarse_cond = coarse_future_flat.detach()
        else:
            coarse_cond = coarse_future_flat

        # Residual target in time domain: r = y_gt - y_coarse
        future_gt_flat = self._flat_future(future_gt)  # (B, T_out, F)
        residual_gt = future_gt_flat - coarse_future_flat.detach()  # always detach for target stability

        loss = self.diffusion.training_loss(
            residual_gt=residual_gt,
            history_flat=history_flat,
            coarse_future_flat=coarse_cond,
        )
        return loss, coarse_future

    @torch.no_grad()
    def predict(self, history: torch.Tensor, deterministic: bool = True, seed: int = 0) -> torch.Tensor:
        """Full two-stage prediction (single reconstruction). Signature unchanged."""
        history_flat = self._flat_history(history)  # (B, T_in, F)

        # Coarse prediction (time domain)
        coarse_future = self.coarse(history)  # (B, T_out, N, 3)
        coarse_future_flat = coarse_future.reshape(history.size(0), self.cfg.pred_length, -1)  # (B, T_out, F)
        coarse_cond = coarse_future_flat.detach() if self.cfg.stopgrad_coarse_condition else coarse_future_flat

        # Sample residual in time domain
        residual_pred = self.diffusion.sample_ddim(
            history_flat=history_flat,
            coarse_future_flat=coarse_cond,
            seed=seed,
            steps=self.cfg.ddim_steps,
            eta=0.0 if deterministic else 1.0,
        )

        # Refined future = coarse + residual
        future_flat = coarse_future_flat + residual_pred
        future = future_flat.reshape(history.size(0), self.cfg.pred_length, self.cfg.num_nodes, 3)

        # IMPORTANT: do NOT add last offset here; coarse already applied it (if enabled).
        return future

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # For compatibility with existing callers: during train, return coarse; during eval, return refined.
        if self.training:
            return self.coarse(history)
        return self.predict(history, deterministic=True, seed=0)
