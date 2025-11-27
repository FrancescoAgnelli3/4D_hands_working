#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import random
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

from config import DATASET_CHOICES, DatasetCfg, TrainCfg
from data import build_datasets, get_dataset_metadata, make_loaders
from runner import run_experiment


BEST_MODELS_PATH = os.path.join("out", "best_runs", "assembly/best_models.json")
BEST_RESULTS_PATH = os.path.join("out", "best_runs", "results_cross_action.csv")
DEFAULT_DATA_DIR = ""
DEFAULT_SAVE_ROOT = os.path.join("out", "best_runs", "checkpoints")
WINDOWS = (51, 70, 90)
HORIZONS = (49, 30, 10)
ALLOW_EQUAL_WINDOW_HORIZON = True

FIELDNAMES = [
    "timestamp",
    "model",
    "dataset",
    "seed",
    "input_n",
    "output_n",
    "stride",
    "batch_size",
    "hidden_size",
    "gru_layers",
    "t_layers",
    "gnn_layers",
    "lr",
    "gradient_clip",
    "use_space",
    "bone_loss_weight",
    "velocity_loss_weight",
    "dct_keep_coeffs",
    "hisrep_hand_dct_n",
    "hisrep_dct_n",
    "simlpe_norm_axis",
    "simlpe_use_norm",
    "simlpe_use_spatial_fc_only",
    "simlpe_mix_spatial_temporal",
    "simlpe_add_last_offset",
    "simlpe_mask_temperature",
    "simlpe_slow_layers",
    "simlpe_fast_layers",
    "simlpe_mid_gate_max",
    "epochs",
    "params",
    "train_mpjpe_best",
    "train_mpjpe_norm_best",
    "validation_mpjpe_best",
    "validation_mpjpe_norm_best",
    "test_mpjpe_best",
    "test_mpjpe_norm_best",
    "train_actions",
    "withheld_action",
]

DEFAULT_EPOCHS = 100
MODEL_EPOCH_OVERRIDES = {
    "pgbig": 50,
    "eqmotion": 50,
}


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _coerce_optional(value: object, cast):
    if value in (None, "", "None"):
        return None
    if _is_nan(value):
        return None
    try:
        if cast is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(int(value))
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes"}:
                    return True
                if lowered in {"0", "false", "no"}:
                    return False
        return cast(value)
    except (TypeError, ValueError):
        return None


def _coerce_required(value: object, cast, *, default):
    coerced = _coerce_optional(value, cast)
    return coerced if coerced is not None else default


def _dataset_file_path(base_path: str, dataset: str) -> str:
    base_dir, base_file = os.path.split(base_path)
    dataset_dir = os.path.join(base_dir, dataset) if base_dir else dataset
    return os.path.join(dataset_dir, base_file)


def _load_best_config(best_json: str, model_name: str) -> Dict[str, object]:
    if not os.path.exists(best_json):
        raise FileNotFoundError(f"Best-model JSON not found: {best_json}")
    with open(best_json, "r") as f:
        data = json.load(f)
    if model_name not in data:
        raise KeyError(f"Model '{model_name}' not present in {best_json}")
    return data[model_name]


def _build_dataset_cfg(
    best_cfg: Dict[str, object],
    *,
    data_dir: str,
    action_filter: str,
    dataset_name: str,
    wrist_indices: Tuple[int, ...],
    node_count: int,
    edge_index: Tuple[Tuple[int, int], ...],
    adjacency: Tuple[Tuple[int, ...], ...],
    window_override: Optional[int] = None,
    horizon_override: Optional[int] = None,
    subset_files: Optional[Tuple[str, ...]] = None,
    seed_override: Optional[int] = None,
) -> DatasetCfg:
    batch_size = _coerce_required(best_cfg.get("batch_size"), int, default=128)
    input_n = window_override if window_override is not None else _coerce_required(best_cfg.get("input_n"), int, default=90)
    output_n = (
        horizon_override if horizon_override is not None else _coerce_required(best_cfg.get("output_n"), int, default=10)
    )
    stride = _coerce_required(best_cfg.get("stride"), int, default=1)
    seed = seed_override if seed_override is not None else _coerce_required(best_cfg.get("seed"), int, default=0)
    return DatasetCfg(
        data_dir=data_dir,
        action_filter=action_filter,
        input_n=input_n,
        output_n=output_n,
        stride=stride,
        time_interp=None,
        window_norm=None,
        batch_size=batch_size,
        seed=seed,
        wrist_indices=wrist_indices,
        dataset=dataset_name,
        node_count=node_count,
        edge_index=edge_index,
        adjacency=adjacency,
        subset_files=subset_files,
    )


def _default_epochs_for_model(model_name: Optional[str]) -> int:
    if model_name is None:
        return DEFAULT_EPOCHS
    return MODEL_EPOCH_OVERRIDES.get(model_name.lower(), DEFAULT_EPOCHS)


def _build_train_cfg(best_cfg: Dict[str, object], *, model_name: Optional[str] = None) -> TrainCfg:
    use_space_raw = best_cfg.get("use_space")
    use_space = bool(int(use_space_raw)) if use_space_raw not in (None, "", "None") else True

    cfg = TrainCfg(
        model=str(best_cfg.get("model")),
        epochs=_coerce_required(best_cfg.get("epochs"), int, default=_default_epochs_for_model(model_name)),
        lr=_coerce_required(best_cfg.get("lr"), float, default=1e-3),
        hidden_size=_coerce_required(best_cfg.get("hidden_size"), int, default=64),
        gru_layers=_coerce_required(best_cfg.get("gru_layers"), int, default=2),
        gradient_clip=_coerce_required(best_cfg.get("gradient_clip"), float, default=10.0),
        bone_loss_weight=_coerce_required(best_cfg.get("bone_loss_weight"), float, default=0.0),
        use_space=use_space,
        t_layers=_coerce_required(best_cfg.get("t_layers"), int, default=2),
        gnn_layers=_coerce_required(best_cfg.get("gnn_layers"), int, default=2),
    )

    cfg.dct_keep_coeffs = _coerce_optional(best_cfg.get("dct_keep_coeffs"), int)
    cfg.hisrep_hand_dct_n = _coerce_optional(best_cfg.get("hisrep_hand_dct_n"), int)
    cfg.hisrep_dct_n = _coerce_optional(best_cfg.get("hisrep_dct_n"), int)
    cfg.simlpe_norm_axis = _coerce_optional(best_cfg.get("simlpe_norm_axis"), str)
    cfg.simlpe_use_norm = _coerce_optional(best_cfg.get("simlpe_use_norm"), bool)
    cfg.simlpe_use_spatial_fc_only = _coerce_optional(best_cfg.get("simlpe_use_spatial_fc_only"), bool)
    cfg.simlpe_mix_spatial_temporal = _coerce_optional(best_cfg.get("simlpe_mix_spatial_temporal"), bool)
    cfg.simlpe_add_last_offset = _coerce_optional(best_cfg.get("simlpe_add_last_offset"), bool)
    cfg.simlpe_mask_temperature = _coerce_optional(best_cfg.get("simlpe_mask_temperature"), float)
    cfg.simlpe_slow_layers = _coerce_optional(best_cfg.get("simlpe_slow_layers"), int)
    cfg.simlpe_fast_layers = _coerce_optional(best_cfg.get("simlpe_fast_layers"), int)
    cfg.simlpe_mid_gate_max = _coerce_optional(best_cfg.get("simlpe_mid_gate_max"), float)

    return cfg


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_results_row(results_csv: str, row: Dict[str, object]) -> None:
    write_header = not os.path.exists(results_csv)
    if not write_header:
        with open(results_csv, "r", newline="") as existing:
            existing_header = existing.readline().strip()
        if not existing_header:
            write_header = True

    with open(results_csv, "a", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for key in FIELDNAMES:
            if row.get(key) is None:
                row[key] = ""
        writer.writerow(row)


def _append_window_horizon(path: str, window: int, horizon: int) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_w{window}_h{horizon}{ext}"


def _sanitize_action_name(action: str) -> str:
    stripped = action.strip()
    if not stripped:
        return ""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", stripped.lower())
    return normalized.strip("_")


def _build_action_suffix(train_actions: Sequence[str], test_action: str) -> str:
    tokens: List[str] = []
    train_tags = [_sanitize_action_name(act) for act in train_actions if _sanitize_action_name(act)]
    if train_tags:
        tokens.append("train-" + "-".join(train_tags))
    test_tag = _sanitize_action_name(test_action)
    if test_tag:
        tokens.append(f"withheld-{test_tag}")
    return "_" + "_".join(tokens) if tokens else ""


def _collect_action_files(data_dir: str, actions: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    pat = os.path.join(data_dir, "*.npy")
    all_files = sorted(glob.glob(pat))
    if not all_files:
        raise SystemExit(f"No dataset files found in '{data_dir}'.")
    mapping: Dict[str, Tuple[str, ...]] = {}
    for action in actions:
        matched = tuple(f for f in all_files if action in os.path.basename(f))
        mapping[action] = matched
    return mapping


def _sample_files(files: Sequence[str], n: int, rng: random.Random) -> Tuple[str, ...]:
    if len(files) <= n:
        return tuple(files)
    return tuple(rng.sample(files, n))


def _stable_seed(base_seed: int, dataset: str, window: int, horizon: int) -> int:
    dataset_sum = sum(ord(ch) for ch in dataset)
    return (base_seed + dataset_sum + window * 10007 + horizon * 1009) % (2**32)


def _prepare_result_row(
    ds_cfg: DatasetCfg,
    train_cfg: TrainCfg,
    metrics: Dict[str, float],
    train_actions: Sequence[str],
    test_action: str,
) -> Dict[str, object]:
    row = {
        "timestamp": int(time.time()),
        "model": train_cfg.model,
        "dataset": ds_cfg.dataset,
        "seed": ds_cfg.seed,
        "input_n": ds_cfg.input_n,
        "output_n": ds_cfg.output_n,
        "stride": ds_cfg.stride,
        "batch_size": ds_cfg.batch_size,
        "hidden_size": train_cfg.hidden_size,
        "gru_layers": train_cfg.gru_layers,
        "t_layers": train_cfg.t_layers,
        "gnn_layers": train_cfg.gnn_layers,
        "lr": train_cfg.lr,
        "gradient_clip": train_cfg.gradient_clip,
        "use_space": int(train_cfg.use_space),
        "bone_loss_weight": train_cfg.bone_loss_weight,
        "velocity_loss_weight": train_cfg.velocity_loss_weight,
        "dct_keep_coeffs": train_cfg.dct_keep_coeffs,
        "hisrep_hand_dct_n": train_cfg.hisrep_hand_dct_n,
        "hisrep_dct_n": train_cfg.hisrep_dct_n,
        "simlpe_norm_axis": train_cfg.simlpe_norm_axis,
        "simlpe_use_norm": train_cfg.simlpe_use_norm,
        "simlpe_use_spatial_fc_only": train_cfg.simlpe_use_spatial_fc_only,
        "simlpe_mix_spatial_temporal": train_cfg.simlpe_mix_spatial_temporal,
        "simlpe_add_last_offset": train_cfg.simlpe_add_last_offset,
        "simlpe_mask_temperature": train_cfg.simlpe_mask_temperature,
        "simlpe_slow_layers": train_cfg.simlpe_slow_layers,
        "simlpe_fast_layers": train_cfg.simlpe_fast_layers,
        "simlpe_mid_gate_max": train_cfg.simlpe_mid_gate_max,
        "epochs": train_cfg.epochs,
        "params": metrics.get("params", 0),
        "train_mpjpe_best": metrics.get("train_mpjpe_best"),
        "train_mpjpe_norm_best": metrics.get("train_mpjpe_norm_best"),
        "validation_mpjpe_best": metrics.get("validation_mpjpe_best"),
        "validation_mpjpe_norm_best": metrics.get("validation_mpjpe_norm_best"),
        "test_mpjpe_best": metrics.get("test_mpjpe_best"),
        "test_mpjpe_norm_best": metrics.get("test_mpjpe_norm_best"),
        "train_actions": ",".join(train_actions),
        "withheld_action": test_action,
    }
    return row


def _with_suffix(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def main():
    ap = argparse.ArgumentParser(description="Train models on a subset of actions and test on a held-out action.")
    ap.add_argument(
        "--model",
        nargs="+",
        default="last,hisrep,simlpe_dct,SFgDNet,sfgdnet_try,mpt,estag,hstgcnn,trajdep,pgbig,motionmixer,eqmotion",
        help="One or more models (space- or comma-separated) whose best configuration should run.",
    )
    ap.add_argument("--dataset", default="assembly", choices=DATASET_CHOICES, help="Dataset to use for experiments.")
    ap.add_argument("--best-json", default=BEST_MODELS_PATH, help="Path to JSON file with best configurations.")
    ap.add_argument("--results-csv", default=BEST_RESULTS_PATH, help="CSV file where evaluation results will be logged.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Root directory containing dataset files.")
    ap.add_argument("--actions", type=str, default="attempt,rotate,remove,unscrew,pick", help="Comma-separated action substring filters that define available actions.")
    ap.add_argument(
        "--train-actions",
        type=str,
        default=None,
        help="Comma-separated actions to train on; defaults to --actions minus the test action.",
    )
    ap.add_argument(
        "--test-action",
        type=str,
        default=None,
        help="Action to reserve for testing; defaults to the last action from --actions.",
    )
    ap.add_argument("--samples-per-action", type=int, default=500, help="Maximum files to sample per action.")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed for sampling.")
    ap.add_argument("--save-root", default=DEFAULT_SAVE_ROOT, help="Directory to store model artifacts.")
    ap.add_argument("--log-wandb", action="store_true", help="Enable logging to Weights & Biases.")
    ap.add_argument("--wandb-project", type=str, default=None, help="W&B project name.")
    ap.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team.")
    ap.add_argument("--wandb-run-prefix", type=str, default=None, help="Run name prefix for W&B.")
    ap.add_argument("--save-SFgDNet-mask", action="store_true", help="Persist SFgDNet mask values per epoch.")
    ap.add_argument("--save-SFgDNet-gates", action="store_true", help="Save SFgDNet gate traces per epoch.")
    ap.add_argument("--save-sfgdnet-try-mask", action="store_true", help="Persist sfgdnet_try mask values per epoch.")
    ap.add_argument("--save-sfgdnet-try-gates", action="store_true", help="Save sfgdnet_try gate traces per epoch.")
    ap.add_argument("--save-eval-examples", action="store_true", help="Persist evaluation examples during validation/testing.")
    ap.add_argument("--window", type=int, default=None, help="Optional override for the input window length (input_n).")
    ap.add_argument("--horizon", type=int, default=None, help="Optional override for the prediction horizon length (output_n).")
    args = ap.parse_args()

    dataset_name = args.dataset.lower()
    metadata = get_dataset_metadata(dataset_name)

    data_dir = args.data_dir or metadata.get("default_dir", "")
    if not data_dir:
        raise SystemExit("A data directory must be provided either via --data-dir or dataset metadata.")

    raw_actions = [a.strip() for a in args.actions.split(",")]
    actions = tuple(a for a in raw_actions if a)
    if len(actions) < 2:
        raise SystemExit("At least two actions must be provided to define train and test splits.")

    provided_test_action = args.test_action.strip() if args.test_action else ""
    test_action = provided_test_action if provided_test_action else actions[-1]
    if test_action not in actions:
        raise SystemExit(f"Test action '{test_action}' must be one of the --actions values.")

    def _parse_action_list(raw: str) -> Tuple[str, ...]:
        items = [a.strip() for a in raw.split(",")] if raw else []
        return tuple(a for a in items if a)

    train_actions = tuple(dict.fromkeys(_parse_action_list(args.train_actions))) if args.train_actions else tuple(
        a for a in actions if a != test_action
    )
    if not train_actions:
        raise SystemExit("At least one training action must be specified.")
    invalid_train_actions = [act for act in train_actions if act not in actions]
    if invalid_train_actions:
        raise SystemExit(f"Train actions {invalid_train_actions} must be among --actions values.")
    if test_action in train_actions:
        raise SystemExit("The test action cannot also be listed among the train actions.")

    default_wrist_indices: Tuple[int, ...] = tuple(metadata.get("default_wrist_indices", ()))
    if not default_wrist_indices:
        raise SystemExit("Dataset metadata must define default wrist indices.")
    wrist_indices = tuple(int(idx) for idx in default_wrist_indices)

    best_json_path = _dataset_file_path(args.best_json, dataset_name)
    results_csv_base = _dataset_file_path(args.results_csv, dataset_name)
    if not os.path.exists(best_json_path) and os.path.exists(args.best_json):
        best_json_path = args.best_json

    node_count = int(metadata.get("node_count", 42))
    edge_index = tuple(metadata.get("edge_index", ()))
    adjacency = tuple(metadata.get("adjacency", ()))

    raw_models = args.model if isinstance(args.model, list) else [str(args.model)]
    model_names = []
    for token in raw_models:
        model_names.extend([m.strip() for m in str(token).split(",") if m.strip()])

    if not model_names:
        raise SystemExit("No valid model names provided to --model")

    save_root = os.path.join(args.save_root, dataset_name)
    _ensure_dir(save_root)

    any_failed = False

    if args.window is not None and args.horizon is not None:
        window_horizon_pairs = [(args.window, args.horizon)]
    else:
        window_horizon_pairs = [
            (
                args.window if args.window is not None else w,
                args.horizon if args.horizon is not None else h,
            )
            for w, h in zip(WINDOWS, HORIZONS)
        ]

    for window, horizon in window_horizon_pairs:
        current_window = window
        current_horizon = horizon

        if not ALLOW_EQUAL_WINDOW_HORIZON and current_horizon >= current_window:
            print(f"Skipping horizon {current_horizon} >= window {current_window}", file=os.sys.stderr)
            continue

        selected_actions = tuple(train_actions) + (test_action,)
        action_file_map = _collect_action_files(data_dir, selected_actions)
        rng_seed = _stable_seed(args.seed, dataset_name, current_window, current_horizon)
        rng = random.Random(rng_seed)
        sampled_files: Dict[str, Tuple[str, ...]] = {}
        for action in selected_actions:
            files = action_file_map.get(action, ())
            if not files:
                raise SystemExit(f"No files found for action '{action}' in '{data_dir}'.")
            target = min(args.samples_per_action, len(files))
            if target == 0:
                raise SystemExit(f"Unable to sample files for action '{action}'.")
            sampled_files[action] = _sample_files(files, target, rng)
            if target < args.samples_per_action:
                print(f"Only {target} samples available for action '{action}' (requested {args.samples_per_action}).")

        train_subset = tuple(f for act in train_actions for f in sampled_files[act])
        test_subset = sampled_files[test_action]
        if not train_subset:
            raise SystemExit("Training subset is empty after sampling the requested actions.")
        if not test_subset:
            raise SystemExit(f"Test subset is empty for the specified test action '{test_action}'.")

        suffix = _build_action_suffix(train_actions, test_action)
        results_csv_with_suffix = _with_suffix(results_csv_base, suffix)
        results_csv_path = _append_window_horizon(results_csv_with_suffix, current_window, current_horizon)
        _ensure_dir(os.path.dirname(results_csv_path) or ".")

        for model_name in model_names:
            try:
                best_cfg = _load_best_config(best_json_path, model_name)

                train_ds_cfg = _build_dataset_cfg(
                    best_cfg,
                    data_dir=data_dir,
                    action_filter="",
                    dataset_name=dataset_name,
                    wrist_indices=wrist_indices,
                    node_count=node_count,
                    edge_index=edge_index,
                    adjacency=adjacency,
                    window_override=current_window,
                    horizon_override=current_horizon,
                    subset_files=train_subset,
                    seed_override=args.seed,
                )

                test_ds_cfg = _build_dataset_cfg(
                    best_cfg,
                    data_dir=data_dir,
                    action_filter="",
                    dataset_name=dataset_name,
                    wrist_indices=wrist_indices,
                    node_count=node_count,
                    edge_index=edge_index,
                    adjacency=adjacency,
                    window_override=current_window,
                    horizon_override=current_horizon,
                    subset_files=test_subset,
                    seed_override=args.seed,
                )

                train_dataset, val_dataset, train_test_dataset = build_datasets(train_ds_cfg)
                train_loader, val_loader, _ = make_loaders(
                    train_dataset, val_dataset, train_test_dataset, train_ds_cfg.batch_size, train_ds_cfg.seed
                )

                cross_train_dataset, cross_val_dataset, cross_test_dataset = build_datasets(test_ds_cfg)
                _, _, cross_test_loader = make_loaders(
                    cross_train_dataset,
                    cross_val_dataset,
                    cross_test_dataset,
                    test_ds_cfg.batch_size,
                    test_ds_cfg.seed,
                )

                train_cfg = _build_train_cfg(best_cfg, model_name=model_name)
                train_cfg.model = model_name
                train_cfg.save_SFgDNet_mask = bool(args.save_SFgDNet_mask)
                train_cfg.save_SFgDNet_gates = bool(args.save_SFgDNet_gates)
                train_cfg.save_sfgdnet_try_mask = bool(args.save_sfgdnet_try_mask)
                train_cfg.save_sfgdnet_try_gates = bool(args.save_sfgdnet_try_gates)
                train_cfg.save_eval_examples = bool(args.save_eval_examples)

                metrics = run_experiment(
                    ds=train_ds_cfg,
                    train_cfg=train_cfg,
                    save_root=save_root,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=cross_test_loader,
                    log_wandb=args.log_wandb or bool(args.wandb_project),
                    wandb_project=args.wandb_project,
                    wandb_entity=args.wandb_entity,
                    wandb_run_prefix=args.wandb_run_prefix,
                )
                if metrics is None:
                    raise RuntimeError("Training failed to produce metrics.")

                row = _prepare_result_row(train_ds_cfg, train_cfg, metrics, train_actions, test_action)
                _write_results_row(results_csv_path, row)
                print(
                    f"Recorded cross-action results for {model_name}: test_action='{test_action}', window={current_window}, horizon={current_horizon}",
                    row,
                )

            except Exception as e:
                any_failed = True
                print(f"Error while training model '{model_name}' with test action '{test_action}': {e}")

    if any_failed:
        raise SystemExit("One or more models failed during training. Check logs above.")


if __name__ == "__main__":
    main()
