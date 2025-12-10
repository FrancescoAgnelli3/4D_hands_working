import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


DEFAULT_MODELS = "last,hisrep,eqmotion,simlpe_dct,SFgDNet,sfgdnet_try,mpt,estag,hstgcnn,trajdep,pgbig,motionmixer"
DATASET_CHOICES: Tuple[str, ...] = ("assembly", "h2o", "bighands", "fpha")

MODEL_DEFAULT_CONFIGS: Dict[str, Dict[str, object]] = {
    "eqmotion": {
        "model": "eqmotion",
        "input_n": 70,
        "output_n": 30,
        "stride": 5,
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_size": 16,
        "gradient_clip": 10.0,
        "bone_loss_weight": 0.0,
        "epochs": 50,
    },
    "estag": {
        "model": "estag",
        "input_n": 70,
        "output_n": 30,
        "stride": 5,
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_size": 16,
        "t_layers": 2,
        "gradient_clip": 10.0,
        "bone_loss_weight": 0.0,
        "epochs": 5,
    },
    "hisrep": {
        "model": "hisrep",
        "input_n": 70,
        "output_n": 30,
        "stride": 5,
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_size": 16,
        "gru_layers": 8,
        "gradient_clip": 10.0,
        "bone_loss_weight": 0.0,
        "hisrep_dct_n": 20,
        "epochs": 100,
    },
    "hstgcnn": {
        "model": "hstgcnn",
        "input_n": 70,
        "output_n": 30,
        "stride": 5,
        "batch_size": 512,
        "lr": 1e-3,
        "gradient_clip": 10.0,
        "bone_loss_weight": 0.0,
        "epochs": 100,
    },
    "last": {
        "model": "last",
        "input_n": 70,
        "output_n": 30,
        "stride": 5,
        "batch_size": 512,
        "bone_loss_weight": 0.0,
        "epochs": 1,
    },
    "motionmixer": {
        "model": "motionmixer",
        "input_n": 70,
        "output_n": 30,
        "stride": 5,
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_size": 64,
        "gradient_clip": 10.0,
        "bone_loss_weight": 0.0,
        "epochs": 100,
    },
    "mpt": {
        "model": "mpt",
        "input_n": 70,
        "output_n": 30,
        "stride": 5,
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_size": 128,
        "gru_layers": 4,
        "gradient_clip": 10.0,
        "use_space": 0,
        "bone_loss_weight": 0.0,
        "epochs": 100,
    },
    "pgbig": {
        "model": "pgbig",
        "input_n": 70,
        "output_n": 30,
        "stride": 5,
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_size": 6,
        "gradient_clip": 10.0,
        "bone_loss_weight": 0.0,
        "pgbig_dct": 100,
        "pgbig_dropout": 0.3,
        "pgbig_encoder_layers": 1,
        "pgbig_decoder_layers": 2,
        "epochs": 50,
    },
    "simlpe_dct": {
        "model": "simlpe_dct",
        "input_n": 70,
        "output_n": 30,
        "stride": 5,
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_size": 128,
        "gru_layers": 8,
        "gradient_clip": 10.0,
        "bone_loss_weight": 0.0,
        "dct_keep_coeffs": None,
        "simlpe_norm_axis": "all",
        "simlpe_use_norm": None,
        "simlpe_use_spatial_fc_only": None,
        "simlpe_mix_spatial_temporal": None,
        "simlpe_add_last_offset": None,
        "epochs": 50,
    },
    "sfgdnet": {
        "model": "SFgDNet",
        "input_n": 70,
        "output_n": 30,
        "stride": 5,
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_size": 64,
        "gradient_clip": 10.0,
        "bone_loss_weight": 0.0,
        "dct_keep_coeffs": None,
        "simlpe_norm_axis": "all",
        "simlpe_use_norm": None,
        "simlpe_use_spatial_fc_only": None,
        "simlpe_mix_spatial_temporal": None,
        "simlpe_add_last_offset": None,
        "simlpe_mask_temperature": 0.1,
        "simlpe_slow_layers": 6,
        "simlpe_fast_layers": 6,
        "simlpe_mid_gate_max": 0.5,
        "epochs": 100,
        "num_layers": 32,
    },
    "sfgdnet_try": {
        "model": "sfgdnet_try",
        "input_n": 70,
        "output_n": 30,
        "stride": 5,
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_size": 32,
        "gradient_clip": 10.0,
        "bone_loss_weight": 0.0,
        "dct_keep_coeffs": None,
        "simlpe_norm_axis": "all",
        "simlpe_use_norm": None,
        "simlpe_use_spatial_fc_only": None,
        "simlpe_mix_spatial_temporal": None,
        "simlpe_add_last_offset": None,
        "simlpe_mask_temperature": 0.1,
        "simlpe_mid_gate_max": 0.5,
        "epochs": 100,
        "num_layers": 4,
        "sfgdnet_try_lambda_l1": 1e-2
    },
    "trajdep": {
        "model": "trajdep",
        "input_n": 70,
        "output_n": 30,
        "stride": 5,
        "batch_size": 512,
        "lr": 1e-3,
        "hidden_size": 32,
        "gnn_layers": 1,
        "gradient_clip": 10.0,
        "bone_loss_weight": 0.0,
        "trajdep_dropout": 0.5,
        "trajdep_dct": 35,
        "epochs": 100,
    },
}


@dataclass
class DatasetCfg:
    data_dir: str
    action_filter: str = "pick_up_screwd"
    input_n: int = 90
    output_n: int = 10
    stride: int = 1
    time_interp: Optional[int] = None
    window_norm: Optional[int] = None
    batch_size: int = 128
    seed: int = 0
    wrist_indices: Tuple[int, ...] = ()
    dataset: str = "assembly"
    node_count: int = 42
    edge_index: Tuple[Tuple[int, int], ...] = tuple()
    adjacency: Tuple[Tuple[int, ...], ...] = tuple()
    subset_files: Optional[Tuple[str, ...]] = None


@dataclass
class TrainCfg:
    model: str = "mpt"
    epochs: int = 10
    lr: float = 1e-3
    hidden_size: int = 64
    gru_layers: int = 2
    gradient_clip: float = 10.0
    bone_loss_weight: float = 0.0
    velocity_loss_weight: float = 0.0
    save_eval_examples: bool = False
    use_space: bool = True
    # Per-model additional knobs
    t_layers: int = 2        # used by estag variants
    gnn_layers: int = 2      # used by graph-based baselines
    # LR scheduler
    use_lr_scheduler: bool = True
    lr_patience: int = 5
    lr_factor: float = 0.5
    lr_min: float = 1e-6
    lr_threshold: float = 1e-4
    lr_cooldown: int = 0
    trajdep_dropout: float = 0.5
    trajdep_dct: int = 35
    pgbig_dct: int = 60
    pgbig_dropout: float = 0.3
    pgbig_encoder_layers: int = 1
    pgbig_decoder_layers: int = 2
    mixer_blocks: int = 4
    mixer_tokens_dim: Optional[int] = None
    mixer_channels_dim: Optional[int] = None
    mixer_regularization: float = 0.0
    mixer_use_se: bool = True
    mixer_use_delta: bool = True
    mixer_input_scale: float = 1.0
    mixer_activation: str = "gelu"
    mixer_block_type: str = "normal"
    mixer_reduction: int = 4
    mixer_use_max_pooling: bool = False
    augment_node_dropout: Optional[float] = None
    augment_channel_dropout: Optional[float] = None
    augment_time_dropout: Optional[float] = None
    dct_keep_coeffs: Optional[int] = 100
    hisrep_hand_dct_n: Optional[int] = None
    hisrep_dct_n: Optional[int] = None
    simlpe_norm_axis: Optional[str] = None
    simlpe_use_norm: Optional[bool] = None
    simlpe_use_spatial_fc_only: Optional[bool] = None
    simlpe_mix_spatial_temporal: Optional[bool] = None
    simlpe_add_last_offset: Optional[bool] = None
    simlpe_mask_temperature: Optional[float] = None
    simlpe_slow_layers: Optional[int] = None
    simlpe_fast_layers: Optional[int] = None
    simlpe_mid_gate_max: Optional[float] = None
    save_SFgDNet_mask: bool = False
    save_SFgDNet_gates: bool = False
    SFgDNet_fixed_mask: Optional[float] = None
    SFgDNet_freeze_gate: bool = False
    SFgDNet_disable_slow_branch: bool = False
    SFgDNet_disable_fast_branch: bool = False
    SFgDNet_fixed_mid_gate: Optional[float] = None
    SFgDNet_skip_dct: bool = False
    SFgDNet_padding_mode: Optional[str] = None
    SFgDNet_force_mlp_padding: Optional[bool] = None
    SFgDNet_padding_predictor: Optional[str] = None
    SFgDNet_padding_full_horizon: Optional[bool] = None
    SFgDNet_learn_lambda: Optional[bool] = None
    SFgDNet_learn_centers: Optional[bool] = None
    SFgDNet_fixed_lambda: Optional[float] = None
    SFgDNet_fixed_centers: Optional[Tuple[float, float]] = None
    save_sfgdnet_try_mask: bool = False
    save_sfgdnet_try_gates: bool = False
    sfgdnet_try_fixed_mask: Optional[float] = None
    sfgdnet_try_freeze_gate: bool = False
    sfgdnet_try_disable_slow_branch: bool = False
    sfgdnet_try_disable_fast_branch: bool = False
    sfgdnet_try_fixed_mid_gate: Optional[float] = None
    sfgdnet_try_skip_dct: bool = False
    sfgdnet_try_padding_mode: Optional[str] = None
    sfgdnet_try_force_mlp_padding: Optional[bool] = None
    sfgdnet_try_padding_predictor: Optional[str] = None
    sfgdnet_try_padding_full_horizon: Optional[bool] = None
    sfgdnet_try_learn_lambda: Optional[bool] = None
    sfgdnet_try_learn_centers: Optional[bool] = None
    sfgdnet_try_fixed_lambda: Optional[float] = None
    sfgdnet_try_fixed_centers: Optional[Tuple[float, float]] = None
    sfgdnet_try_num_gates: Optional[int] = None
    sfgdnet_try_gate_basis_degree: Optional[int] = None
    sfgdnet_try_gate_basis_size: Optional[int] = None
    sfgdnet_try_num_layers: Optional[int] = None
    sfgdnet_try_lambda_l1: Optional[float] = None
    sfgdnet_try_lambda_div: Optional[float] = 1e-2
    sfgdnet_try_routing: Optional[str] = None
    simlpe_reg_freq_kl: Optional[float] = None
    simlpe_reg_tv: Optional[float] = None
    simlpe_reg_entropy: Optional[float] = None


def parse_list(arg: str, cast):
    return [cast(x) for x in arg.split(',') if x != '']


def build_argument_parser(base_dir: str, default_data_dir: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run ablation studies for training.")
    ap.add_argument("--dataset", type=str, default="assembly", choices=DATASET_CHOICES, help="Dataset to use for experiments.")
    ap.add_argument("--data-dir", type=str, default=default_data_dir)
    ap.add_argument("--action-filter", type=str, default="")
    ap.add_argument("--input-n", type=int, default=70)
    ap.add_argument("--output-n", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--time-interp", type=int, default=None)
    ap.add_argument("--window-norm", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--wrist-indices", type=str, default="", help="Comma-separated left and right wrist indices (zero-based). Uses dataset defaults when omitted.")

    # Ablation grids
    ap.add_argument("--models", type=str, default=DEFAULT_MODELS, help="Comma list of models to run")
    ap.add_argument("--hidden-sizes", type=str, default="8,16,32,64,128")
    ap.add_argument("--gru-layers", type=str, default="4,6,8")
    ap.add_argument("--lrs", type=str, default="0.001")
    ap.add_argument("--use-space", type=str, default="1", help="For mpt only: 1 or 0 (comma list)")
    ap.add_argument("--bone-loss-weights", type=str, default="0.0,0.01")
    ap.add_argument("--t-layers", type=str, default="1,2,4", help="Temporal/layer count")
    ap.add_argument("--gnn-layers", type=str, default="1,2,3", help="GNN layers")
    ap.add_argument("--gradient-clips", type=str, default="10.0", help="Gradient clip norms to try")
    ap.add_argument("--hisrep-hand-dct-ns", type=str, default="20,30", help="Comma-separated list of DCT components for hisrep_hand. maximum value: 10+output_n")
    ap.add_argument("--hisrep-dct-ns", type=str, default="20,30", help="Comma-separated list of DCT components for hisrep.")
    ap.add_argument("--simlpe-norm-axes", type=str, default="all", help="Comma-separated norm_axis values for siMLPe variants (choices: spatial,temporal,all).")
    ap.add_argument("--simlpe-use-norms", type=str, default="", help="Comma-separated 1/0 flags toggling normalization inside siMLPe variants (leave empty to use defaults).")
    ap.add_argument("--simlpe-spatial-fc-only", type=str, default="", help="Comma-separated 1/0 flags to enable spatial-only projections in siMLPe variants.")
    ap.add_argument("--simlpe-mix-spatial-temporal", type=str, default="", help="Comma-separated 1/0 flags to alternate temporal/spatial projections within siMLPe blocks.")
    ap.add_argument("--simlpe-add-last-offsets", type=str, default="", help="Comma-separated 1/0 flags controlling the additive last-frame offset in siMLPe variants.")
    ap.add_argument("--simlpe-dct-components", type=str, default="", help="Comma-separated DCT component counts retained in siMLPe DCT-based variants.")
    ap.add_argument("--simlpe-mask-temperatures", type=str, default="", help="Comma-separated mask temperatures for siMLPe dual-path gating (leave empty for defaults).")
    ap.add_argument("--simlpe-slow-layers", type=str, default="4,6,8", help="Comma-separated slow-branch layer counts for siMLPe dual-path.")
    ap.add_argument("--simlpe-fast-layers", type=str, default="4,6,8", help="Comma-separated fast-branch layer counts for siMLPe dual-path.")
    ap.add_argument("--simlpe-mid-gate-maxes", type=str, default="0.5", help="Comma-separated maximum gate values for the siMLPe dual MID branch.")
    ap.add_argument(
        "--sfgdnet-try-num-layers",
        type=str,
        default="48, 64",
        help="Comma-separated expert depths for sfgdnet_try mixtures (num_layers override).",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=str, default=None, help="Optional seed list to override --seed")

    # Study mode
    ap.add_argument("--preset", type=str, choices=["basic", "extended", "full"], default="basic")
    ap.add_argument("--max-runs", type=int, default=None, help="Optional hard cap on total experiments")

    # Outputs
    ap.add_argument("--save-root", type=str, default=os.path.join(base_dir, "out", "ablation"))
    ap.add_argument("--csv", type=str, default=None, help="Results CSV path; defaults to save-root/results.csv")
    ap.add_argument("--log-wandb", action="store_true", help="Enable per-epoch logging to Weights & Biases.")
    ap.add_argument("--wandb-project", type=str, default=None, help="Optional W&B project name; enabling this also turns on logging.")
    ap.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity (team) name.")
    ap.add_argument("--wandb-run-prefix", type=str, default=None, help="Optional prefix prepended to W&B run names.")
    ap.add_argument(
        "--save-eval-examples",
        action="store_true",
        help="When set, persist evaluation examples (merged_pred/merged_tgt tensors) during validation/testing.",
    )

    return ap
