import argparse
import csv
import os
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from config import DATASET_CHOICES, DatasetCfg, TrainCfg
from data import build_datasets, get_dataset_metadata, make_loaders
from runner import run_experiment
from train_best_models import (
    BEST_MODELS_PATH,
    DEFAULT_SAVE_ROOT,
    FIELDNAMES as BASE_FIELDNAMES,
    _append_window_horizon,
    _build_dataset_cfg,
    _build_train_cfg,
    _dataset_file_path,
    _load_best_config,
    _prepare_result_row,
)


def _parse_comma_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    tokens = [token.strip() for token in raw.split(",")]
    values = [token for token in tokens if token]
    return values or None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _apply_overrides(
    train_cfg: TrainCfg,
    overrides: Dict[str, object],
    ds_cfg: DatasetCfg,
) -> None:
    for key, value in overrides.items():
        resolved = value(train_cfg, ds_cfg) if callable(value) else value
        setattr(train_cfg, key, resolved)


def _make_scenario(
    name: str,
    description: str,
    groups,
    overrides: Dict[str, object],
) -> Dict[str, object]:
    """Helper to keep scenario definitions concise."""

    if isinstance(groups, str):
        group_tuple = (groups,)
    else:
        group_tuple = tuple(groups)

    return {
        "name": name,
        "description": description,
        "groups": group_tuple,
        "train_overrides": overrides,
    }


def _build_reference_scenario() -> Dict[str, object]:
    """Baseline configuration shared across ablation groups."""

    return _make_scenario(
        name="reference_learned_softmax",
        description="Default dual-path model: DCT features, learned gates, and learned padding.",
        groups=("reference",),
        overrides={
            "SFgDNet_padding_mode": "mlp",
            "SFgDNet_learn_lambda": True,
            "SFgDNet_learn_centers": True,
            "SFgDNet_skip_dct": False,
        },
    )


def _build_component_ablation_scenarios() -> List[Dict[str, object]]:
    """Remove each core component to showcase additive contributions."""

    return [
        _make_scenario(
            name="component_no_dct",
            description="Bypass the DCT projection entirely (raw temporal MLP).",
            groups=("component", "dct"),
            overrides={
                "SFgDNet_skip_dct": True,
            },
        ),
        _make_scenario(
            name="component_no_gate_single_branch",
            description="Remove the learned gate by keeping only the slow branch (mask fixed to 1).",
            groups=("component", "gate"),
            overrides={
                "SFgDNet_disable_fast_branch": True,
                "SFgDNet_disable_slow_branch": False,
                "SFgDNet_freeze_gate": True,
                "SFgDNet_fixed_mask": 1.0,
            },
        ),
        _make_scenario(
            name="component_fixed_gate_centers",
            description="Freeze λ and gate centers at their defaults to remove gate learning.",
            groups=("component", "gate"),
            overrides={
                "SFgDNet_freeze_gate": True,
                "SFgDNet_learn_lambda": False,
                "SFgDNet_learn_centers": False,
                "SFgDNet_fixed_lambda": 1.0,
                "SFgDNet_fixed_centers": (0.25, 0.75),
            },
        ),
        _make_scenario(
            name="component_frozen_gates",
            description="Keep the slow/fast gates fixed by freezing every gate parameter after initialization.",
            groups=("component", "gate"),
            overrides={
                "SFgDNet_freeze_gate": True,
                "SFgDNet_learn_lambda": False,
                "SFgDNet_learn_centers": False,
            },
        ),
        _make_scenario(
            name="component_no_padder_repeat_last",
            description="Remove the padder and simply repeat the final observed frame.",
            groups=("component", "padder"),
            overrides={
                "SFgDNet_padding_mode": "repeat_last",
                "SFgDNet_force_mlp_padding": False,
            },
        ),
        _make_scenario(
            name="component_padder_rnn_single_step",
            description="Use an RNN-based padder to predict the single frame before duplication.",
            groups=("component", "padder"),
            overrides={
                "SFgDNet_padding_mode": "mlp",
                "SFgDNet_force_mlp_padding": True,
                "SFgDNet_padding_predictor": "rnn",
            },
        ),
        _make_scenario(
            name="component_padder_mlp_full_horizon",
            description="Have the padder MLP predict the entire horizon instead of repeating one step.",
            groups=("component", "padder"),
            overrides={
                "SFgDNet_padding_mode": "mlp",
                "SFgDNet_force_mlp_padding": True,
                "SFgDNet_padding_full_horizon": True,
            },
        ),
        _make_scenario(
            name="component_transmlp_mixed_layers",
            description="Enforce alternating temporal/spatial layers inside the TransMLP backbone.",
            groups=("component", "backbone"),
            overrides={
                "simlpe_mix_spatial_temporal": True,
            },
        ),
    ]


def _make_dct_override(divisor: int):
    def _override(train_cfg: TrainCfg, ds_cfg: DatasetCfg) -> int:
        total = int(ds_cfg.input_n + ds_cfg.output_n)
        if divisor <= 1:
            return max(1, total)
        keep = max(1, int(round(total / divisor)))
        return keep

    return _override


def _build_frequency_sweep_scenarios() -> List[Dict[str, object]]:
    """Vary the number of retained DCT coefficients to show efficiency."""

    configs = [
        ("freq_coeff_T_over_8", 8, "Keep K=T/8 coefficients (rounded) to test heavy truncation."),
        ("freq_coeff_T_over_4", 4, "Keep K=T/4 coefficients to test moderate truncation."),
        ("freq_coeff_T_over_2", 2, "Keep K=T/2 coefficients for light truncation."),
        ("freq_coeff_T_over_1", 1, "Keep all K=T coefficients to recover the full DCT."),
    ]
    scenarios: List[Dict[str, object]] = []
    for name, divisor, desc in configs:
        scenarios.append(
            _make_scenario(
                name=name,
                description=desc,
                groups=("frequency", "dct"),
                overrides={
                    "dct_keep_coeffs": _make_dct_override(divisor),
                },
            )
        )
    return scenarios


def _build_loss_ablation_scenarios(train_cfg: TrainCfg) -> List[Dict[str, object]]:
    """Remove each auxiliary loss to expose its effect."""

    base_bone = float(train_cfg.bone_loss_weight)
    base_vel = float(train_cfg.velocity_loss_weight)
    return [
        _make_scenario(
            name="loss_no_bone",
            description=f"Drop L_bone (baseline weight {base_bone:.4f}) to show skeleton prior impact.",
            groups=("loss", "bone"),
            overrides={
                "bone_loss_weight": 0.0,
            },
        ),
        _make_scenario(
            name="loss_no_velocity",
            description=f"Drop L_vel (baseline weight {base_vel:.4f}) to show smoothness contribution.",
            groups=("loss", "velocity"),
            overrides={
                "velocity_loss_weight": 0.0,
            },
        ),
    ]


def _append_custom_row(
    path: str,
    fieldnames: List[str],
    row: Dict[str, object],
) -> None:
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        sanitized = {}
        for key in fieldnames:
            value = row.get(key, "")
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, bool):
                sanitized[key] = int(value)
            else:
                sanitized[key] = value
        writer.writerow(sanitized)


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds // 60
    rem = seconds % 60
    return f"{int(minutes)}m{rem:04.1f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Architecture ablations for the SiMLPe Dual model.")
    parser.add_argument("--dataset", type=str, default="assembly", choices=DATASET_CHOICES)
    parser.add_argument("--best-json", type=str, default=BEST_MODELS_PATH, help="JSON file with best-model configurations.")
    parser.add_argument("--data-dir", type=str, default="", help="Override dataset root directory.")
    parser.add_argument("--action-filter", type=str, default="pick", help="Optional action substring filter.")
    parser.add_argument("--window", type=int, default=None, help="Override input window length (input_n).")
    parser.add_argument("--horizon", type=int, default=None, help="Override prediction horizon length (output_n).")
    parser.add_argument("--seed", type=int, default=None, help="Single seed to run.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list of seeds.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override for all runs.")
    parser.add_argument("--scenarios", type=str, default=None, help="Comma list of ablation scenario names to run.")
    parser.add_argument("--results-root", type=str, default=os.path.join("out", "ablation"), help="Root directory for ablation CSVs.")
    parser.add_argument("--results-name", type=str, default="results.csv", help="Filename for the results CSV (before window/horizon suffix).")
    parser.add_argument("--save-root", type=str, default=os.path.join(DEFAULT_SAVE_ROOT, "architecture"), help="Root for model checkpoints.")
    parser.add_argument("--log-wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team.")
    parser.add_argument("--wandb-run-prefix", type=str, default=None, help="Optional W&B run-name prefix.")
    parser.add_argument("--save-SFgDNet-mask", action="store_true", help="Persist SFgDNet masks for each epoch.")
    parser.add_argument("--save-eval-examples",action="store_true",help="When set, persist evaluation examples (merged_pred/merged_tgt tensors) during validation/testing.")
    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    metadata = get_dataset_metadata(dataset_name)

    default_wrist_indices: Tuple[int, ...] = tuple(metadata.get("default_wrist_indices", ()))
    if not default_wrist_indices:
        raise SystemExit(f"Dataset '{dataset_name}' must define default wrist indices.")
    wrist_indices = tuple(int(idx) for idx in default_wrist_indices)

    data_dir = args.data_dir or metadata.get("default_dir", "")
    action_filter = args.action_filter or metadata.get("default_action_filter", "")
    node_count = int(metadata.get("node_count", 42))
    edge_index = tuple(metadata.get("edge_index", ()))
    adjacency = tuple(metadata.get("adjacency", ()))

    best_json_path = _dataset_file_path(args.best_json, dataset_name)
    if not os.path.exists(best_json_path) and os.path.exists(args.best_json):
        best_json_path = args.best_json
    best_cfg = _load_best_config(best_json_path, "SFgDNet")

    ds_cfg_template = _build_dataset_cfg(
        best_cfg,
        data_dir=data_dir,
        action_filter=action_filter,
        dataset_name=dataset_name,
        wrist_indices=wrist_indices,
        node_count=node_count,
        edge_index=edge_index,
        adjacency=adjacency,
        window_override=args.window,
        horizon_override=args.horizon,
    )

    train_cfg_template = _build_train_cfg(best_cfg)
    train_cfg_template.model = "SFgDNet"
    train_cfg_template.save_SFgDNet_mask = bool(args.save_SFgDNet_mask)
    train_cfg_template.save_eval_examples = bool(args.save_eval_examples)
    if args.epochs is not None:
        train_cfg_template.epochs = int(args.epochs)

    # Ensure default values are explicit so they appear in the result CSV.
    if train_cfg_template.simlpe_mask_temperature is None:
        train_cfg_template.simlpe_mask_temperature = 1.0
    if train_cfg_template.simlpe_slow_layers is None:
        train_cfg_template.simlpe_slow_layers = 6
    if train_cfg_template.simlpe_fast_layers is None:
        train_cfg_template.simlpe_fast_layers = 6
    if train_cfg_template.simlpe_mid_gate_max is None:
        train_cfg_template.simlpe_mid_gate_max = 0.9

    seeds_cli = _parse_comma_list(args.seeds)
    if seeds_cli:
        seed_values = [int(value) for value in seeds_cli]
    elif args.seed is not None:
        seed_values = [int(args.seed)]
    else:
        seed_values = [ds_cfg_template.seed]

    scenarios: List[Dict[str, object]] = [
        _build_reference_scenario(),
        *_build_component_ablation_scenarios(),
        *_build_frequency_sweep_scenarios(),
        *_build_loss_ablation_scenarios(train_cfg_template),
    ]
    scenario_filter = _parse_comma_list(args.scenarios)
    if scenario_filter is not None:
        scenario_filter_set = {name.lower() for name in scenario_filter}
        scenarios = [sc for sc in scenarios if sc["name"].lower() in scenario_filter_set]
        if not scenarios:
            raise SystemExit(f"No scenarios match the provided filter: {scenario_filter}.")

    results_dir = os.path.join(args.results_root, dataset_name, "simlpe_architecture")
    _ensure_dir(results_dir)
    results_csv = os.path.join(results_dir, args.results_name)
    results_csv = _append_window_horizon(results_csv, ds_cfg_template.input_n, ds_cfg_template.output_n)
    if action_filter:
        results_csv = results_csv.replace(".csv", f"_{action_filter}.csv")

    save_root_base = os.path.join(args.save_root, dataset_name, "simlpe_architecture")
    _ensure_dir(save_root_base)

    fieldnames = list(BASE_FIELDNAMES)
    extra_fields = [
        "ablation",
        "ablation_desc",
        "ablation_groups",
        "SFgDNet_padding_mode",
        "SFgDNet_force_mlp_padding",
        "SFgDNet_padding_predictor",
        "SFgDNet_padding_full_horizon",
        "simlpe_mix_spatial_temporal",
        "SFgDNet_skip_dct",
        "SFgDNet_learn_lambda",
        "SFgDNet_learn_centers",
        "SFgDNet_fixed_lambda",
        "SFgDNet_fixed_centers",
        "SFgDNet_fixed_mask",
        "SFgDNet_disable_slow_branch",
        "SFgDNet_disable_fast_branch",
    ]
    for extra in extra_fields:
        if extra not in fieldnames:
            fieldnames.append(extra)

    loader_cache: Dict[int, Tuple[DatasetCfg, Tuple[object, object, object]]] = {}

    def get_loaders_for_seed(seed_value: int) -> Tuple[DatasetCfg, object, object, object]:
        if seed_value not in loader_cache:
            cfg_seed = replace(ds_cfg_template, seed=seed_value)
            train_dataset, val_dataset, test_dataset = build_datasets(cfg_seed)
            train_loader, val_loader, test_loader = make_loaders(
                train_dataset, val_dataset, test_dataset, cfg_seed.batch_size, seed_value
            )
            loader_cache[seed_value] = (cfg_seed, (train_loader, val_loader, test_loader))
        cfg_seed, loaders = loader_cache[seed_value]
        return cfg_seed, *loaders

    total_runs = len(seed_values) * len(scenarios)
    print(
        f"Preparing {total_runs} experiments "
        f"(seeds={seed_values}, scenarios={[sc['name'] for sc in scenarios]})"
    )
    start_time = time.time()
    completed_runs = 0
    failed_runs: List[str] = []

    for seed in seed_values:
        ds_cfg_seed, train_loader, val_loader, test_loader = get_loaders_for_seed(seed)
        for scenario in scenarios:
            run_label = f"seed={seed} scenario={scenario['name']}"
            print(f"\n[Run] {run_label}")

            scenario_train_cfg = replace(train_cfg_template)
            _apply_overrides(scenario_train_cfg, scenario["train_overrides"], ds_cfg_seed)

            current_save_root = os.path.join(
                save_root_base,
                scenario["name"],
                f"seed_{seed}",
            )
            _ensure_dir(current_save_root)

            try:
                metrics = run_experiment(
                    ds=ds_cfg_seed,
                    train_cfg=scenario_train_cfg,
                    save_root=current_save_root,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    log_wandb=args.log_wandb or bool(args.wandb_project),
                    wandb_project=args.wandb_project,
                    wandb_entity=args.wandb_entity,
                    wandb_run_prefix=args.wandb_run_prefix,
                )
            except Exception as exc:
                failed_runs.append(f"{run_label}: {exc}")
                print(f"[Error] {run_label}: {exc}")
                continue

            if metrics is None:
                failed_runs.append(f"{run_label}: metrics returned None")
                print(f"[Warning] {run_label} returned no metrics.")
                continue

            row = _prepare_result_row(ds_cfg_seed, scenario_train_cfg, metrics)
            row["ablation"] = scenario["name"]
            row["ablation_desc"] = scenario["description"]
            groups = scenario.get("groups", ())
            if isinstance(groups, (list, tuple)):
                row["ablation_groups"] = ",".join(str(g) for g in groups if g)
            elif isinstance(groups, str):
                row["ablation_groups"] = groups

            # Ensure the CSV captures the effective hyper-parameters used by the scenario.
            if row.get("simlpe_mask_temperature") in (None, "") and scenario_train_cfg.simlpe_mask_temperature is not None:
                row["simlpe_mask_temperature"] = scenario_train_cfg.simlpe_mask_temperature
            if row.get("simlpe_slow_layers") in (None, "") and scenario_train_cfg.simlpe_slow_layers is not None:
                row["simlpe_slow_layers"] = scenario_train_cfg.simlpe_slow_layers
            if row.get("simlpe_fast_layers") in (None, "") and scenario_train_cfg.simlpe_fast_layers is not None:
                row["simlpe_fast_layers"] = scenario_train_cfg.simlpe_fast_layers
            if row.get("simlpe_mid_gate_max") in (None, "") and scenario_train_cfg.simlpe_mid_gate_max is not None:
                row["simlpe_mid_gate_max"] = scenario_train_cfg.simlpe_mid_gate_max
            supplemental_keys = [
                "SFgDNet_padding_mode",
                "SFgDNet_force_mlp_padding",
                "SFgDNet_padding_predictor",
                "SFgDNet_padding_full_horizon",
                "simlpe_mix_spatial_temporal",
                "SFgDNet_skip_dct",
                "SFgDNet_learn_lambda",
                "SFgDNet_learn_centers",
                "SFgDNet_fixed_lambda",
                "SFgDNet_fixed_centers",
                "SFgDNet_fixed_mask",
                "SFgDNet_disable_slow_branch",
                "SFgDNet_disable_fast_branch",
            ]
            for key in supplemental_keys:
                value = getattr(scenario_train_cfg, key, None)
                if value is None:
                    continue
                if isinstance(value, (tuple, list)):
                    serialized = "|".join(str(item) for item in value)
                else:
                    serialized = value
                if row.get(key) in (None, ""):
                    row[key] = serialized

            _append_custom_row(results_csv, fieldnames, row)
            completed_runs += 1

    duration = time.time() - start_time
    print(f"\nCompleted {completed_runs}/{total_runs} runs in {_format_seconds(duration)}.")
    if failed_runs:
        print("The following runs failed:")
        for entry in failed_runs:
            print(f"  - {entry}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
