import hashlib
import math
import os
import random
import time
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DatasetCfg, TrainCfg
from train_utils import train


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int % (2**32))
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)


def _prepare_wandb_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    """Filter numeric metrics for logging to Weights & Biases."""
    payload: Dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, bool):
            payload[key] = float(value)
            continue
        if isinstance(value, (int, float)):
            val = float(value)
            if math.isfinite(val):
                payload[key] = val
    return payload


def run_experiment(
    ds: DatasetCfg,
    train_cfg: TrainCfg,
    *,
    save_root: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader],
    log_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_prefix: Optional[str] = None,
    save_best_model: bool = False,
    load_model_path: Optional[str] = None,
    twostage_eval_phase: Optional[str] = None,
    twostage_eval_best_of_k: Optional[int] = None,
    twostage_eval_collect_all: bool = False,
) -> Optional[Dict[str, float]]:
    """Run one training+evaluation experiment and return metrics."""
    set_global_seed(ds.seed)
    tag = (
        f"dataset={ds.dataset}__model={train_cfg.model}__hs={train_cfg.hidden_size}__gl={train_cfg.gru_layers}__tl={train_cfg.t_layers}__gls={train_cfg.gnn_layers}__"
        f"lr={train_cfg.lr}__gc={train_cfg.gradient_clip}__bs={ds.batch_size}__"
        f"in={ds.input_n}__out={ds.output_n}__st={ds.stride}__ti={ds.time_interp}__sp={int(train_cfg.use_space)}__"
        f"bw={train_cfg.bone_loss_weight}__velw={train_cfg.velocity_loss_weight}__seed={ds.seed}"
    )
    if train_cfg.dct_keep_coeffs is not None:
        tag += f"__dctk={int(train_cfg.dct_keep_coeffs)}"
    if train_cfg.hisrep_hand_dct_n is not None:
        tag += f"__hh_dct={int(train_cfg.hisrep_hand_dct_n)}"
    if train_cfg.hisrep_dct_n is not None:
        tag += f"__h_dct={int(train_cfg.hisrep_dct_n)}"
    if train_cfg.simlpe_norm_axis is not None:
        tag += f"__simnorm={train_cfg.simlpe_norm_axis}"
    if train_cfg.simlpe_use_norm is not None:
        tag += f"__simusenorm={int(train_cfg.simlpe_use_norm)}"
    if train_cfg.simlpe_use_spatial_fc_only is not None:
        tag += f"__simspfc={int(train_cfg.simlpe_use_spatial_fc_only)}"
    if train_cfg.simlpe_mix_spatial_temporal is not None:
        tag += f"__simmix={int(train_cfg.simlpe_mix_spatial_temporal)}"
    if train_cfg.simlpe_add_last_offset is not None:
        tag += f"__simoffset={int(train_cfg.simlpe_add_last_offset)}"
    if train_cfg.simlpe_mask_temperature is not None:
        tag += f"__simtemp={float(train_cfg.simlpe_mask_temperature):g}"
    if train_cfg.simlpe_slow_layers is not None:
        tag += f"__simslow={int(train_cfg.simlpe_slow_layers)}"
    if train_cfg.simlpe_fast_layers is not None:
        tag += f"__simfast={int(train_cfg.simlpe_fast_layers)}"
    if train_cfg.simlpe_mid_gate_max is not None:
        tag += f"__simgate={float(train_cfg.simlpe_mid_gate_max):g}"
    if train_cfg.simlpe_wave_num_scales is not None:
        tag += f"__simwaveJ={int(train_cfg.simlpe_wave_num_scales)}"
    if train_cfg.simlpe_wave_wavelet is not None:
        tag += f"__simwaveW={train_cfg.simlpe_wave_wavelet}"
    if train_cfg.simlpe_wave_wavelet_mode is not None:
        tag += f"__simwaveMode={train_cfg.simlpe_wave_wavelet_mode}"
    if train_cfg.simlpe_wave_padding_mode is not None:
        tag += f"__simwavePad={train_cfg.simlpe_wave_padding_mode}"
    if train_cfg.simlpe_wave_padding_predictor is not None:
        tag += f"__simwavePred={train_cfg.simlpe_wave_padding_predictor}"
    if train_cfg.simlpe_wave_padding_full_horizon is not None:
        tag += f"__simwaveFull={int(train_cfg.simlpe_wave_padding_full_horizon)}"
    if train_cfg.simlpe_wave_norm_axis is not None:
        tag += f"__simwaveNorm={train_cfg.simlpe_wave_norm_axis}"
    if train_cfg.simlpe_wave_use_norm is not None:
        tag += f"__simwaveUseNorm={int(train_cfg.simlpe_wave_use_norm)}"
    if train_cfg.simlpe_wave_add_last_offset is not None:
        tag += f"__simwaveOffset={int(train_cfg.simlpe_wave_add_last_offset)}"
    if train_cfg.twostage_diffusion_epochs is not None:
        tag += f"__twepochs={int(train_cfg.twostage_diffusion_epochs)}"
    if train_cfg.twostage_k_low_coarse is not None:
        tag += f"__twkc={int(train_cfg.twostage_k_low_coarse)}"
    if train_cfg.twostage_k_low_diffusion is not None:
        tag += f"__twkd={int(train_cfg.twostage_k_low_diffusion)}"
    if train_cfg.twostage_diffusion_steps is not None:
        tag += f"__twsteps={int(train_cfg.twostage_diffusion_steps)}"
    if train_cfg.twostage_ddim_steps is not None:
        tag += f"__twddim={int(train_cfg.twostage_ddim_steps)}"
    if train_cfg.twostage_beta_schedule is not None:
        tag += f"__twbeta={train_cfg.twostage_beta_schedule}"
    if train_cfg.twostage_denoiser_dim is not None:
        tag += f"__twdim={int(train_cfg.twostage_denoiser_dim)}"
    if train_cfg.twostage_denoiser_depth is not None:
        tag += f"__twdepth={int(train_cfg.twostage_denoiser_depth)}"
    if train_cfg.twostage_denoiser_heads is not None:
        tag += f"__twheads={int(train_cfg.twostage_denoiser_heads)}"
    if train_cfg.twostage_dropout is not None:
        tag += f"__twdrop={float(train_cfg.twostage_dropout):g}"
    if train_cfg.twostage_freeze_coarse is not None:
        tag += f"__twfreeze={int(train_cfg.twostage_freeze_coarse)}"
    if train_cfg.twostage_stopgrad_coarse_condition is not None:
        tag += f"__twstopgrad={int(train_cfg.twostage_stopgrad_coarse_condition)}"
    if train_cfg.twostage_diffusion_conditioning is not None:
        tag += f"__twcond={train_cfg.twostage_diffusion_conditioning}"
    if train_cfg.twostage_residual_ema_decay is not None:
        tag += f"__twema={float(train_cfg.twostage_residual_ema_decay):g}"
    if train_cfg.twostage_residual_rms_eps is not None:
        tag += f"__twrms={float(train_cfg.twostage_residual_rms_eps):g}"
    if train_cfg.twostage_residual_ema_warmup_steps is not None:
        tag += f"__twemaWarm={int(train_cfg.twostage_residual_ema_warmup_steps)}"
    if train_cfg.twostage_x0_clip is not None:
        tag += f"__twclip={float(train_cfg.twostage_x0_clip):g}"
    if train_cfg.twostage_snr_weighting is not None:
        tag += f"__twsnr={train_cfg.twostage_snr_weighting}"
    if train_cfg.twostage_min_snr_gamma is not None:
        tag += f"__twgamma={float(train_cfg.twostage_min_snr_gamma):g}"
    if train_cfg.twostage_t_min_frac is not None:
        tag += f"__twtmin={float(train_cfg.twostage_t_min_frac):g}"
    if train_cfg.twostage_mpjpe_weight is not None:
        tag += f"__twmpjpe={float(train_cfg.twostage_mpjpe_weight):g}"
    if train_cfg.spectral_num_modes is not None:
        tag += f"__specK={int(train_cfg.spectral_num_modes)}"
    if train_cfg.spectral_time_normalize is not None:
        tag += f"__specTNorm={int(train_cfg.spectral_time_normalize)}"
    if train_cfg.spectral_subtract_last is not None:
        tag += f"__specSubLast={int(train_cfg.spectral_subtract_last)}"
    if train_cfg.spectral_subtract_mean is not None:
        tag += f"__specSubMean={int(train_cfg.spectral_subtract_mean)}"
    if train_cfg.spectral_use_coeff_mlp is not None:
        tag += f"__specMLP={int(train_cfg.spectral_use_coeff_mlp)}"
    if train_cfg.spectral_coeff_mlp_hidden is not None:
        tag += f"__specMLPH={int(train_cfg.spectral_coeff_mlp_hidden)}"
    if train_cfg.spectral_coeff_dropout is not None:
        tag += f"__specMLPDrop={float(train_cfg.spectral_coeff_dropout):g}"
    if train_cfg.spectral_coeff_residual is not None:
        tag += f"__specMLPRes={int(train_cfg.spectral_coeff_residual)}"
    if train_cfg.spectral_pinv_ridge is not None:
        tag += f"__specRidge={float(train_cfg.spectral_pinv_ridge):g}"

    mixer_tokens_dim = int(train_cfg.mixer_tokens_dim) if train_cfg.mixer_tokens_dim is not None else int(train_cfg.hidden_size)
    mixer_channels_dim = int(train_cfg.mixer_channels_dim) if train_cfg.mixer_channels_dim is not None else int(train_cfg.hidden_size)
    config = {
        "batch_size": int(ds.batch_size),
        "input_n": int(ds.input_n),
        "output_n": int(ds.output_n),
        "learning_rate": float(train_cfg.lr),
        "train_epoches": int(train_cfg.epochs),
        "hidden_size": int(train_cfg.hidden_size),
        "gru_layers": int(train_cfg.gru_layers),
        "gradient_clip": float(train_cfg.gradient_clip),
        "node_num": int(ds.node_count),
        "use_space": bool(train_cfg.use_space),
        "velocity_loss_weight": float(train_cfg.velocity_loss_weight),
        "save_eval_examples": bool(train_cfg.save_eval_examples),
        "log_gcn_stats": False,
        "t_layers": int(train_cfg.t_layers),
        "gnn_layers": int(train_cfg.gnn_layers),
        "trajdep_dropout": float(train_cfg.trajdep_dropout),
        "trajdep_dct": int(train_cfg.trajdep_dct),
        "pgbig_dct": int(train_cfg.pgbig_dct),
        "pgbig_dropout": float(train_cfg.pgbig_dropout),
        "pgbig_encoder_layers": int(train_cfg.pgbig_encoder_layers),
        "pgbig_decoder_layers": int(train_cfg.pgbig_decoder_layers),
        "use_lr_scheduler": bool(train_cfg.use_lr_scheduler),
        "lr_patience": int(train_cfg.lr_patience),
        "lr_factor": float(train_cfg.lr_factor),
        "lr_min": float(train_cfg.lr_min),
        "lr_threshold": float(train_cfg.lr_threshold),
        "lr_cooldown": int(train_cfg.lr_cooldown),
        "mixer_blocks": int(train_cfg.mixer_blocks),
        "mixer_tokens_dim": mixer_tokens_dim,
        "mixer_channels_dim": mixer_channels_dim,
        "mixer_regularization": float(train_cfg.mixer_regularization),
        "mixer_use_se": bool(train_cfg.mixer_use_se),
        "mixer_use_delta": bool(train_cfg.mixer_use_delta),
        "mixer_input_scale": float(train_cfg.mixer_input_scale),
        "mixer_activation": str(train_cfg.mixer_activation),
        "mixer_block_type": str(train_cfg.mixer_block_type),
        "mixer_reduction": int(train_cfg.mixer_reduction),
        "mixer_use_max_pooling": bool(train_cfg.mixer_use_max_pooling),
        "log_wandb": bool(log_wandb),
        "dataset": ds.dataset,
        "edge_index": tuple(ds.edge_index),
        "adjacency": tuple(tuple(int(val) for val in row) for row in ds.adjacency) if ds.adjacency else tuple(),
    }
    if train_cfg.augment_node_dropout is not None:
        config["augment_node_dropout"] = float(train_cfg.augment_node_dropout)
    if train_cfg.augment_channel_dropout is not None:
        config["augment_channel_dropout"] = float(train_cfg.augment_channel_dropout)
    if train_cfg.augment_time_dropout is not None:
        config["augment_time_dropout"] = float(train_cfg.augment_time_dropout)
    if save_best_model:
        examples_models_dir = os.path.join(os.path.dirname(__file__), "examples", "models")
        os.makedirs(examples_models_dir, exist_ok=True)
        action_segment = ds.action_filter or "all"
        action_segment = action_segment.replace(os.sep, "_").replace(" ", "_")
        model_file = f"{train_cfg.model}_{ds.dataset}_{action_segment}.pt"
        config["best_model_path"] = os.path.join(examples_models_dir, model_file)
        config["best_model_tag"] = tag
    if load_model_path:
        config["load_model_path"] = str(load_model_path)
    if twostage_eval_phase:
        config["twostage_eval_phase"] = str(twostage_eval_phase)
    if twostage_eval_best_of_k is not None:
        config["twostage_eval_best_of_k"] = max(1, int(twostage_eval_best_of_k))
    if twostage_eval_collect_all:
        config["twostage_eval_collect_all"] = True
    if train_cfg.save_coarse_model:
        examples_models_dir = os.path.join(os.path.dirname(__file__), "examples", "models")
        os.makedirs(examples_models_dir, exist_ok=True)
        action_segment = ds.action_filter or "all"
        action_segment = action_segment.replace(os.sep, "_").replace(" ", "_")
        tag_hash = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:12]
        model_file = f"{train_cfg.model}_coarse_{ds.dataset}_{action_segment}_{tag_hash}.pt"
        config["coarse_model_path"] = os.path.join(examples_models_dir, model_file)
        config["coarse_model_tag"] = tag
        config["save_coarse_model"] = True
    if train_cfg.dct_keep_coeffs is not None:
        config["dct_keep_coeffs"] = int(train_cfg.dct_keep_coeffs)
    if train_cfg.hisrep_hand_dct_n is not None:
        config["hisrep_hand_dct_n"] = int(train_cfg.hisrep_hand_dct_n)
    if train_cfg.hisrep_dct_n is not None:
        config["hisrep_dct_n"] = int(train_cfg.hisrep_dct_n)
    if train_cfg.simlpe_norm_axis is not None:
        config["simlpe_norm_axis"] = str(train_cfg.simlpe_norm_axis)
    if train_cfg.simlpe_use_norm is not None:
        config["simlpe_use_norm"] = bool(train_cfg.simlpe_use_norm)
    if train_cfg.simlpe_use_spatial_fc_only is not None:
        config["simlpe_spatial_fc_only"] = bool(train_cfg.simlpe_use_spatial_fc_only)
    if train_cfg.simlpe_mix_spatial_temporal is not None:
        config["simlpe_mix_spatial_temporal"] = bool(train_cfg.simlpe_mix_spatial_temporal)
    if train_cfg.simlpe_add_last_offset is not None:
        config["simlpe_add_last_offset"] = bool(train_cfg.simlpe_add_last_offset)
    if train_cfg.simlpe_mask_temperature is not None:
        config["simlpe_mask_temperature"] = float(train_cfg.simlpe_mask_temperature)
    if train_cfg.simlpe_slow_layers is not None:
        config["simlpe_slow_layers"] = int(train_cfg.simlpe_slow_layers)
    if train_cfg.simlpe_fast_layers is not None:
        config["simlpe_fast_layers"] = int(train_cfg.simlpe_fast_layers)
    if train_cfg.simlpe_mid_gate_max is not None:
        config["simlpe_mid_gate_max"] = float(train_cfg.simlpe_mid_gate_max)
    if train_cfg.simlpe_wave_num_scales is not None:
        config["simlpe_wave_num_scales"] = int(train_cfg.simlpe_wave_num_scales)
    if train_cfg.simlpe_wave_wavelet is not None:
        config["simlpe_wave_wavelet"] = str(train_cfg.simlpe_wave_wavelet)
    if train_cfg.simlpe_wave_wavelet_mode is not None:
        config["simlpe_wave_wavelet_mode"] = str(train_cfg.simlpe_wave_wavelet_mode)
    if train_cfg.simlpe_wave_padding_mode is not None:
        config["simlpe_wave_padding_mode"] = str(train_cfg.simlpe_wave_padding_mode)
    if train_cfg.simlpe_wave_padding_predictor is not None:
        config["simlpe_wave_padding_predictor"] = str(train_cfg.simlpe_wave_padding_predictor)
    if train_cfg.simlpe_wave_padding_full_horizon is not None:
        config["simlpe_wave_padding_full_horizon"] = bool(train_cfg.simlpe_wave_padding_full_horizon)
    if train_cfg.simlpe_wave_norm_axis is not None:
        config["simlpe_wave_norm_axis"] = str(train_cfg.simlpe_wave_norm_axis)
    if train_cfg.simlpe_wave_use_norm is not None:
        config["simlpe_wave_use_norm"] = bool(train_cfg.simlpe_wave_use_norm)
    if train_cfg.simlpe_wave_add_last_offset is not None:
        config["simlpe_wave_add_last_offset"] = bool(train_cfg.simlpe_wave_add_last_offset)
    if train_cfg.twostage_diffusion_epochs is not None:
        config["twostage_diffusion_epochs"] = int(train_cfg.twostage_diffusion_epochs)
    if train_cfg.twostage_k_low_coarse is not None:
        config["twostage_k_low_coarse"] = int(train_cfg.twostage_k_low_coarse)
    if train_cfg.twostage_k_low_diffusion is not None:
        config["twostage_k_low_diffusion"] = int(train_cfg.twostage_k_low_diffusion)
    if train_cfg.twostage_diffusion_steps is not None:
        config["twostage_diffusion_steps"] = int(train_cfg.twostage_diffusion_steps)
    if train_cfg.twostage_ddim_steps is not None:
        config["twostage_ddim_steps"] = int(train_cfg.twostage_ddim_steps)
    if train_cfg.twostage_beta_schedule is not None:
        config["twostage_beta_schedule"] = str(train_cfg.twostage_beta_schedule)
    if train_cfg.twostage_denoiser_dim is not None:
        config["twostage_denoiser_dim"] = int(train_cfg.twostage_denoiser_dim)
    if train_cfg.twostage_denoiser_depth is not None:
        config["twostage_denoiser_depth"] = int(train_cfg.twostage_denoiser_depth)
    if train_cfg.twostage_denoiser_heads is not None:
        config["twostage_denoiser_heads"] = int(train_cfg.twostage_denoiser_heads)
    if train_cfg.twostage_dropout is not None:
        config["twostage_dropout"] = float(train_cfg.twostage_dropout)
    if train_cfg.twostage_freeze_coarse is not None:
        config["twostage_freeze_coarse"] = bool(train_cfg.twostage_freeze_coarse)
    if train_cfg.twostage_stopgrad_coarse_condition is not None:
        config["twostage_stopgrad_coarse_condition"] = bool(train_cfg.twostage_stopgrad_coarse_condition)
    if train_cfg.twostage_diffusion_conditioning is not None:
        config["twostage_diffusion_conditioning"] = str(train_cfg.twostage_diffusion_conditioning)
    if train_cfg.twostage_residual_ema_decay is not None:
        config["twostage_residual_ema_decay"] = float(train_cfg.twostage_residual_ema_decay)
    if train_cfg.twostage_residual_rms_eps is not None:
        config["twostage_residual_rms_eps"] = float(train_cfg.twostage_residual_rms_eps)
    if train_cfg.twostage_residual_ema_warmup_steps is not None:
        config["twostage_residual_ema_warmup_steps"] = int(train_cfg.twostage_residual_ema_warmup_steps)
    if train_cfg.twostage_x0_clip is not None:
        config["twostage_x0_clip"] = float(train_cfg.twostage_x0_clip)
    if train_cfg.twostage_snr_weighting is not None:
        config["twostage_snr_weighting"] = str(train_cfg.twostage_snr_weighting)
    if train_cfg.twostage_min_snr_gamma is not None:
        config["twostage_min_snr_gamma"] = float(train_cfg.twostage_min_snr_gamma)
    if train_cfg.twostage_t_min_frac is not None:
        config["twostage_t_min_frac"] = float(train_cfg.twostage_t_min_frac)
    if train_cfg.twostage_mpjpe_weight is not None:
        config["twostage_mpjpe_weight"] = float(train_cfg.twostage_mpjpe_weight)
    if train_cfg.spectral_num_modes is not None:
        config["spectral_num_modes"] = int(train_cfg.spectral_num_modes)
    if train_cfg.spectral_time_normalize is not None:
        config["spectral_time_normalize"] = bool(train_cfg.spectral_time_normalize)
    if train_cfg.spectral_subtract_last is not None:
        config["spectral_subtract_last"] = bool(train_cfg.spectral_subtract_last)
    if train_cfg.spectral_subtract_mean is not None:
        config["spectral_subtract_mean"] = bool(train_cfg.spectral_subtract_mean)
    if train_cfg.spectral_use_coeff_mlp is not None:
        config["spectral_use_coeff_mlp"] = bool(train_cfg.spectral_use_coeff_mlp)
    if train_cfg.spectral_coeff_mlp_hidden is not None:
        config["spectral_coeff_mlp_hidden"] = int(train_cfg.spectral_coeff_mlp_hidden)
    if train_cfg.spectral_coeff_dropout is not None:
        config["spectral_coeff_dropout"] = float(train_cfg.spectral_coeff_dropout)
    if train_cfg.spectral_coeff_residual is not None:
        config["spectral_coeff_residual"] = bool(train_cfg.spectral_coeff_residual)
    if train_cfg.spectral_pinv_ridge is not None:
        config["spectral_pinv_ridge"] = float(train_cfg.spectral_pinv_ridge)
    if train_cfg.physmamba_dropout is not None:
        config["physmamba_dropout"] = float(train_cfg.physmamba_dropout)
    if train_cfg.physmamba_d_state is not None:
        config["physmamba_d_state"] = int(train_cfg.physmamba_d_state)
    if train_cfg.physmamba_d_conv is not None:
        config["physmamba_d_conv"] = int(train_cfg.physmamba_d_conv)
    if train_cfg.physmamba_expand is not None:
        config["physmamba_expand"] = int(train_cfg.physmamba_expand)
    if train_cfg.physmamba_decoder_kernel is not None:
        config["physmamba_decoder_kernel"] = int(train_cfg.physmamba_decoder_kernel)
    if train_cfg.save_SFgDNet_mask:
        mask_dir = os.path.join(save_root, "SFgDNet_masks")
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, f"{tag}_mask.pt")
        config["SFgDNet_mask_path"] = mask_path
    if train_cfg.save_SFgDNet_gates:
        examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
        os.makedirs(examples_dir, exist_ok=True)
        gate_path = os.path.join(examples_dir, f"gate_slow_and_fast.pt")
        config["SFgDNet_gate_path"] = gate_path
    if train_cfg.save_SplineEqNet_mask:
        mask_dir = os.path.join(save_root, "SplineEqNet_masks")
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, f"{tag}_mask.pt")
        config["SplineEqNet_mask_path"] = mask_path
    if train_cfg.save_SplineEqNet_gates:
        examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
        os.makedirs(examples_dir, exist_ok=True)
        gate_path = os.path.join(examples_dir, f"gate_slow_and_fast_try.pt")
        config["SplineEqNet_gate_path"] = gate_path
    if train_cfg.SFgDNet_fixed_mask is not None:
        config["SFgDNet_fixed_mask"] = float(train_cfg.SFgDNet_fixed_mask)
    if train_cfg.SFgDNet_freeze_gate:
        config["SFgDNet_freeze_gate"] = True
    if train_cfg.SFgDNet_disable_slow_branch:
        config["SFgDNet_disable_slow_branch"] = True
    if train_cfg.SFgDNet_disable_fast_branch:
        config["SFgDNet_disable_fast_branch"] = True
    if train_cfg.SFgDNet_fixed_mid_gate is not None:
        config["SFgDNet_fixed_mid_gate"] = float(train_cfg.SFgDNet_fixed_mid_gate)
    if train_cfg.SFgDNet_skip_dct:
        config["SFgDNet_skip_dct"] = True
    if train_cfg.SFgDNet_padding_mode is not None:
        config["SFgDNet_padding_mode"] = str(train_cfg.SFgDNet_padding_mode)
    if train_cfg.SFgDNet_force_mlp_padding is not None:
        config["SFgDNet_force_mlp_padding"] = bool(train_cfg.SFgDNet_force_mlp_padding)
    if train_cfg.SFgDNet_padding_predictor is not None:
        config["SFgDNet_padding_predictor"] = str(train_cfg.SFgDNet_padding_predictor)
    if train_cfg.SFgDNet_padding_full_horizon is not None:
        config["SFgDNet_padding_full_horizon"] = bool(train_cfg.SFgDNet_padding_full_horizon)
    if train_cfg.SFgDNet_learn_lambda is not None:
        config["SFgDNet_learn_lambda"] = bool(train_cfg.SFgDNet_learn_lambda)
    if train_cfg.SFgDNet_learn_centers is not None:
        config["SFgDNet_learn_centers"] = bool(train_cfg.SFgDNet_learn_centers)
    if train_cfg.SFgDNet_fixed_lambda is not None:
        config["SFgDNet_fixed_lambda"] = float(train_cfg.SFgDNet_fixed_lambda)
    if train_cfg.SFgDNet_fixed_centers is not None:
        config["SFgDNet_fixed_centers"] = tuple(float(v) for v in train_cfg.SFgDNet_fixed_centers)
    if train_cfg.SplineEqNet_fixed_mask is not None:
        config["SplineEqNet_fixed_mask"] = float(train_cfg.SplineEqNet_fixed_mask)
    if train_cfg.SplineEqNet_freeze_gate:
        config["SplineEqNet_freeze_gate"] = True
    if train_cfg.SplineEqNet_disable_slow_branch:
        config["SplineEqNet_disable_slow_branch"] = True
    if train_cfg.SplineEqNet_disable_fast_branch:
        config["SplineEqNet_disable_fast_branch"] = True
    if train_cfg.SplineEqNet_fixed_mid_gate is not None:
        config["SplineEqNet_fixed_mid_gate"] = float(train_cfg.SplineEqNet_fixed_mid_gate)
    if train_cfg.SplineEqNet_skip_dct:
        config["SplineEqNet_skip_dct"] = True
    if train_cfg.SplineEqNet_padding_mode is not None:
        config["SplineEqNet_padding_mode"] = str(train_cfg.SplineEqNet_padding_mode)
    if train_cfg.SplineEqNet_force_mlp_padding is not None:
        config["SplineEqNet_force_mlp_padding"] = bool(train_cfg.SplineEqNet_force_mlp_padding)
    if train_cfg.SplineEqNet_padding_predictor is not None:
        config["SplineEqNet_padding_predictor"] = str(train_cfg.SplineEqNet_padding_predictor)
    if train_cfg.SplineEqNet_padding_full_horizon is not None:
        config["SplineEqNet_padding_full_horizon"] = bool(train_cfg.SplineEqNet_padding_full_horizon)
    if train_cfg.SplineEqNet_learn_lambda is not None:
        config["SplineEqNet_learn_lambda"] = bool(train_cfg.SplineEqNet_learn_lambda)
    if train_cfg.SplineEqNet_learn_centers is not None:
        config["SplineEqNet_learn_centers"] = bool(train_cfg.SplineEqNet_learn_centers)
    if train_cfg.SplineEqNet_fixed_lambda is not None:
        config["SplineEqNet_fixed_lambda"] = float(train_cfg.SplineEqNet_fixed_lambda)
    if train_cfg.SplineEqNet_fixed_centers is not None:
        config["SplineEqNet_fixed_centers"] = tuple(float(v) for v in train_cfg.SplineEqNet_fixed_centers)
    if train_cfg.SplineEqNet_num_gates is not None:
        config["SplineEqNet_num_gates"] = int(train_cfg.SplineEqNet_num_gates)
    if train_cfg.SplineEqNet_gate_basis_degree is not None:
        config["SplineEqNet_gate_basis_degree"] = int(train_cfg.SplineEqNet_gate_basis_degree)
    if train_cfg.SplineEqNet_gate_basis_size is not None:
        config["SplineEqNet_gate_basis_size"] = int(train_cfg.SplineEqNet_gate_basis_size)
    if train_cfg.SplineEqNet_num_layers is not None:
        config["SplineEqNet_num_layers"] = int(train_cfg.SplineEqNet_num_layers)
    if train_cfg.SplineEqNet_lambda_l1 is not None:
        config["SplineEqNet_lambda_l1"] = float(train_cfg.SplineEqNet_lambda_l1)
    if train_cfg.SplineEqNet_lambda_div is not None:
        config["SplineEqNet_lambda_div"] = float(train_cfg.SplineEqNet_lambda_div)
    if train_cfg.SplineEqNet_negpearson_weight is not None:
        config["SplineEqNet_negpearson_weight"] = float(train_cfg.SplineEqNet_negpearson_weight)
    if train_cfg.SplineEqNet_routing is not None:
        config["SplineEqNet_routing"] = str(train_cfg.SplineEqNet_routing)
    wandb_run = None
    wandb_logging_active = False
    if log_wandb:
        try:
            import wandb  # type: ignore
        except ImportError:
            print("Weights & Biases logging requested but wandb is not installed; continuing without logging.")
            config["log_wandb"] = False
        else:
            wandb_config = {
                "tag": tag,
                "seed": ds.seed,
                "train": asdict(train_cfg),
                "dataset": asdict(ds),
            }
            wandb_kwargs: Dict[str, object] = {
                "config": wandb_config,
                "reinit": True,
            }
            if wandb_project:
                wandb_kwargs["project"] = wandb_project
            if wandb_entity:
                wandb_kwargs["entity"] = wandb_entity
            run_name = tag
            if wandb_run_prefix:
                run_name = f"{wandb_run_prefix}-{run_name}"
            wandb_kwargs["name"] = run_name
            try:
                wandb_run = wandb.init(**wandb_kwargs)
            except Exception as exc:
                print(f"Failed to initialise Weights & Biases run ({exc}); disabling logging for this experiment.")
                wandb_run = None
                config["log_wandb"] = False
            else:
                wandb_logging_active = True
                config["log_wandb"] = True

    print(f"\n=== Training: {tag} ===")
    metrics: Optional[Dict[str, float]] = None
    try:
        metrics = train(
            config,
            epochs=train_cfg.epochs,
            lr=train_cfg.lr,
            bone_loss_weight=train_cfg.bone_loss_weight,
            model=train_cfg.model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            log_wandb=wandb_logging_active,
            wandb_run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            if metrics:
                final_metrics = {f"final/{k}": v for k, v in _prepare_wandb_metrics(metrics).items()}
                if final_metrics:
                    try:
                        wandb_run.log(final_metrics)
                    except Exception as exc:
                        print(f"Failed to log final metrics to Weights & Biases ({exc}).")
            wandb_run.finish()

    if metrics is None:
        print("Training returned no metrics; skipping this experiment and continuing.")
        return None

    return metrics
