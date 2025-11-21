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
from train_and_test import train


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
    test_loader: DataLoader,
    log_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_prefix: Optional[str] = None,
    save_best_model: bool = False,
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
    if train_cfg.save_sfgdnet_try_mask:
        mask_dir = os.path.join(save_root, "sfgdnet_try_masks")
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, f"{tag}_mask.pt")
        config["sfgdnet_try_mask_path"] = mask_path
    if train_cfg.save_sfgdnet_try_gates:
        examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
        os.makedirs(examples_dir, exist_ok=True)
        gate_path = os.path.join(examples_dir, f"gate_slow_and_fast_try.pt")
        config["sfgdnet_try_gate_path"] = gate_path
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
    if train_cfg.sfgdnet_try_fixed_mask is not None:
        config["sfgdnet_try_fixed_mask"] = float(train_cfg.sfgdnet_try_fixed_mask)
    if train_cfg.sfgdnet_try_freeze_gate:
        config["sfgdnet_try_freeze_gate"] = True
    if train_cfg.sfgdnet_try_disable_slow_branch:
        config["sfgdnet_try_disable_slow_branch"] = True
    if train_cfg.sfgdnet_try_disable_fast_branch:
        config["sfgdnet_try_disable_fast_branch"] = True
    if train_cfg.sfgdnet_try_fixed_mid_gate is not None:
        config["sfgdnet_try_fixed_mid_gate"] = float(train_cfg.sfgdnet_try_fixed_mid_gate)
    if train_cfg.sfgdnet_try_skip_dct:
        config["sfgdnet_try_skip_dct"] = True
    if train_cfg.sfgdnet_try_padding_mode is not None:
        config["sfgdnet_try_padding_mode"] = str(train_cfg.sfgdnet_try_padding_mode)
    if train_cfg.sfgdnet_try_force_mlp_padding is not None:
        config["sfgdnet_try_force_mlp_padding"] = bool(train_cfg.sfgdnet_try_force_mlp_padding)
    if train_cfg.sfgdnet_try_padding_predictor is not None:
        config["sfgdnet_try_padding_predictor"] = str(train_cfg.sfgdnet_try_padding_predictor)
    if train_cfg.sfgdnet_try_padding_full_horizon is not None:
        config["sfgdnet_try_padding_full_horizon"] = bool(train_cfg.sfgdnet_try_padding_full_horizon)
    if train_cfg.sfgdnet_try_learn_lambda is not None:
        config["sfgdnet_try_learn_lambda"] = bool(train_cfg.sfgdnet_try_learn_lambda)
    if train_cfg.sfgdnet_try_learn_centers is not None:
        config["sfgdnet_try_learn_centers"] = bool(train_cfg.sfgdnet_try_learn_centers)
    if train_cfg.sfgdnet_try_fixed_lambda is not None:
        config["sfgdnet_try_fixed_lambda"] = float(train_cfg.sfgdnet_try_fixed_lambda)
    if train_cfg.sfgdnet_try_fixed_centers is not None:
        config["sfgdnet_try_fixed_centers"] = tuple(float(v) for v in train_cfg.sfgdnet_try_fixed_centers)
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
