import argparse
import json
import math
import os
import time
from dataclasses import asdict
from typing import Dict, Optional, Tuple

from config import DATASET_CHOICES, DatasetCfg, TrainCfg
from data import build_datasets, get_dataset_metadata, make_loaders
from runner import run_experiment


BEST_MODELS_PATH = os.path.join("out", "best_runs", "assembly/best_models.json")
BEST_RESULTS_PATH = os.path.join("out", "best_runs", "results.csv")
DEFAULT_DATA_DIR = ""
DEFAULT_SAVE_ROOT = os.path.join("out", "best_runs", "checkpoints")

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
]


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
    )


def _build_train_cfg(best_cfg: Dict[str, object]) -> TrainCfg:
    use_space_raw = best_cfg.get("use_space")
    use_space = bool(int(use_space_raw)) if use_space_raw not in (None, "", "None") else True

    cfg = TrainCfg(
        model=str(best_cfg.get("model")),
        epochs=_coerce_required(best_cfg.get("epochs"), int, default=50),
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


def _prepare_result_row(ds_cfg: DatasetCfg, train_cfg: TrainCfg, metrics: Dict[str, float]) -> Dict[str, object]:
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
    }
    return row


def main():
    ap = argparse.ArgumentParser(description="Train, validate, and test the best configuration for one or more models.")
    ap.add_argument(
        "--model",
        nargs="+",
        default="last,hisrep,simlpe_dct,SFgDNet,sfgdnet_try,mpt,estag,hstgcnn,trajdep,pgbig,motionmixer,eqmotion",
        help="One or more model names (space- or comma-separated) whose best configuration should be trained.",
    )
    ap.add_argument("--dataset", default="assembly", choices=DATASET_CHOICES, help="Dataset to use for training the best model.")
    ap.add_argument("--best-json", default=BEST_MODELS_PATH, help="Path to JSON file with best configurations.")
    ap.add_argument("--results-csv", default=BEST_RESULTS_PATH, help="CSV file where evaluation results will be logged.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Root directory containing dataset files.")
    ap.add_argument("--action-filter", default="", help="Action substring filter for dataset files.")
    ap.add_argument("--save-root", default=DEFAULT_SAVE_ROOT, help="Directory to store model artifacts.")
    ap.add_argument("--seed", type=int, default=None, help="Override random seed for dataset/build pipeline.")
    ap.add_argument("--log-wandb", action="store_true", help="Enable logging to Weights & Biases.")
    ap.add_argument("--wandb-project", type=str, default=None, help="W&B project name.")
    ap.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team.")
    ap.add_argument("--wandb-run-prefix", type=str, default=None, help="Optional run name prefix for W&B.")
    ap.add_argument("--save-SFgDNet-mask", action="store_true", help="When set, persist the SFgDNet mask values (m) for every training epoch.")
    ap.add_argument("--save-SFgDNet-gates", action="store_true", help="When set, enable gate_logging_enabled so slow/fast gate traces are saved into the examples folder.")
    ap.add_argument("--save-sfgdnet-try-mask", action="store_true", help="When set, persist the sfgdnet_try mask values (m) for every training epoch.")
    ap.add_argument("--save-sfgdnet-try-gates", action="store_true", help="When set, enable gate logging for the sfgdnet_try variant.")
    ap.add_argument("--save-eval-examples", action="store_true", help="When set, persist evaluation examples (merged_pred/merged_tgt tensors) during validation/testing.")
    ap.add_argument("--save-best-model", action="store_true", help="When set, persist the best model checkpoint into the examples directory.")
    ap.add_argument("--window", type=int,default=None, help="Optional override for the input window length (input_n).")
    ap.add_argument("--horizon",type=int,default=None,help="Optional override for the prediction horizon length (output_n).")
    args = ap.parse_args()

    dataset_name = args.dataset.lower()
    metadata = get_dataset_metadata(dataset_name)

    data_dir = args.data_dir or metadata.get("default_dir", "")
    action_filter = args.action_filter or metadata.get("default_action_filter", "")
    default_wrist_indices: Tuple[int, ...] = tuple(metadata.get("default_wrist_indices", ()))
    if not default_wrist_indices:
        raise SystemExit("Dataset metadata must define default wrist indices.")
    wrist_indices = tuple(int(idx) for idx in default_wrist_indices)

    best_json_path = _dataset_file_path(args.best_json, dataset_name)
    results_csv_base = _dataset_file_path(args.results_csv, dataset_name)
    if args.action_filter:
        results_csv_base = results_csv_base.replace(".csv", f"_{args.action_filter}.csv")
    save_root = os.path.join(args.save_root, dataset_name)

    if not os.path.exists(best_json_path) and os.path.exists(args.best_json):
        best_json_path = args.best_json

    # Parse model names: support space- and comma-separated inputs
    raw_models = args.model if isinstance(args.model, list) else [str(args.model)]
    model_names = []
    for token in raw_models:
        model_names.extend([m.strip() for m in str(token).split(",") if m.strip()])

    if not model_names:
        raise SystemExit("No valid model names provided to --model")

    _ensure_dir(save_root)

    results_csv_path: Optional[str] = None

    any_failed = False
    for model_name in model_names:
        try:
            best_cfg = _load_best_config(best_json_path, model_name)

            ds_cfg = _build_dataset_cfg(
                best_cfg,
                data_dir=data_dir,
                action_filter=action_filter,
                dataset_name=dataset_name,
                wrist_indices=wrist_indices,
                node_count=int(metadata.get("node_count", 42)),
                edge_index=tuple(metadata.get("edge_index", ())),
                adjacency=tuple(metadata.get("adjacency", ())),
                window_override=args.window,
                horizon_override=args.horizon,
                seed_override=args.seed,
            )

            train_cfg = _build_train_cfg(best_cfg)
            # Ensure the model field reflects the current target model
            train_cfg.model = model_name
            train_cfg.save_SFgDNet_mask = bool(args.save_SFgDNet_mask)
            train_cfg.save_SFgDNet_gates = bool(args.save_SFgDNet_gates)
            train_cfg.save_sfgdnet_try_mask = bool(args.save_sfgdnet_try_mask)
            train_cfg.save_sfgdnet_try_gates = bool(args.save_sfgdnet_try_gates)
            train_cfg.save_eval_examples = bool(args.save_eval_examples)

            train_dataset, val_dataset, test_dataset = build_datasets(ds_cfg)
            train_loader, val_loader, test_loader = make_loaders(
                train_dataset, val_dataset, test_dataset, ds_cfg.batch_size, ds_cfg.seed
            )

            if results_csv_path is None:
                results_csv_path = _append_window_horizon(results_csv_base, ds_cfg.input_n, ds_cfg.output_n)
                _ensure_dir(os.path.dirname(results_csv_path) or ".")
            else:
                expected_path = _append_window_horizon(results_csv_base, ds_cfg.input_n, ds_cfg.output_n)
                if expected_path != results_csv_path:
                    raise ValueError(
                        "Inconsistent dataset window/horizon detected across models; "
                        "cannot log to a shared results file with differing configurations."
                    )

            metrics = run_experiment(
                ds=ds_cfg,
                train_cfg=train_cfg,
                save_root=save_root,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                log_wandb=args.log_wandb or bool(args.wandb_project),
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_run_prefix=args.wandb_run_prefix,
                save_best_model=args.save_best_model,
            )
            if metrics is None:
                raise RuntimeError("Training failed to produce metrics.")

            row = _prepare_result_row(ds_cfg, train_cfg, metrics)
            _write_results_row(results_csv_path, row)
            print(f"Recorded results for {model_name}:", row)

        except Exception as e:
            any_failed = True
            print(f"Error while training model '{model_name}': {e}")

    if any_failed:
        raise SystemExit("One or more models failed during training. Check logs above.")


if __name__ == "__main__":
    main()
