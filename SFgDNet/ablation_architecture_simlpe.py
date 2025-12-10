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


def _total_sequence_length(ds_cfg: DatasetCfg) -> int:
    return int(ds_cfg.input_n + ds_cfg.output_n)


def _constant_override(value: int):
    def _override(train_cfg: TrainCfg, ds_cfg: DatasetCfg) -> int:
        return int(value)

    return _override


def _full_sequence_override(train_cfg: TrainCfg, ds_cfg: DatasetCfg) -> int:
    return _total_sequence_length(ds_cfg)


def _fractional_sequence_override(divisor: float):
    def _override(train_cfg: TrainCfg, ds_cfg: DatasetCfg) -> int:
        total = _total_sequence_length(ds_cfg)
        if divisor <= 0:
            return total
        keep = max(1, int(round(total / divisor)))
        return keep

    return _override


def _build_sfgdnet_try_scenarios(default_dct_len: int, default_layers: int) -> List[Dict[str, object]]:
    """Return the requested Block A–H scenarios for sfgdnet_try."""

    scenarios: List[Dict[str, object]] = []
    default_layers = max(1, int(default_layers))
    layers_minus_two = max(1, default_layers - 2)
    layers_plus_two = max(1, default_layers + 2)

    # Block A — DCT necessity
    scenarios.extend(
        [
            _make_scenario(
                name="A1_skip_dct_single_expert",
                description="Disables the DCT entirely and uses a single expert, testing whether the baseline time-domain model is competitive.",
                groups=("A", "dct"),
                overrides={
                    "sfgdnet_try_skip_dct": True,
                    "sfgdnet_try_num_gates": 1,
                    "dct_keep_coeffs": _full_sequence_override,
                },
            ),
            _make_scenario(
                name="A2_dct_single_expert",
                description="Enables DCT but still uses a single expert, testing whether the DCT transform alone adds value.",
                groups=("A", "dct"),
                overrides={
                    "sfgdnet_try_skip_dct": False,
                    "sfgdnet_try_num_gates": 1,
                    "dct_keep_coeffs": _constant_override(default_dct_len),
                },
            ),
        ]
    )

    # Block B — DCT length sensitivity
    scenarios.extend(
        [
            _make_scenario(
                name="B1_dct_full_spectrum",
                description="No truncation; uses the full spectrum.",
                groups=("B", "dct_length"),
                overrides={
                    "dct_keep_coeffs": _full_sequence_override,
                },
            ),
            _make_scenario(
                name="B2_dct_half_spectrum",
                description="Keeps half of the frequency components; tests mid-level truncation.",
                groups=("B", "dct_length"),
                overrides={
                    "dct_keep_coeffs": _fractional_sequence_override(2),
                },
            ),
            _make_scenario(
                name="B4_dct_30",
                description="Tests aggressive truncation to measure robustness under limited bandwidth.",
                groups=("B", "dct_length"),
                overrides={
                    "dct_keep_coeffs": _constant_override(30),
                },
            ),
        ]
    )

    # Block C — Number of experts
    scenarios.extend(
        [
            _make_scenario(
                name="C1_one_expert",
                description="Removes the MoE; baseline single-expert model.",
                groups=("C", "experts"),
                overrides={"sfgdnet_try_num_gates": 1},
            ),
            _make_scenario(
                name="C2_two_experts",
                description="Minimal mixture; tests if two experts already help.",
                groups=("C", "experts"),
                overrides={"sfgdnet_try_num_gates": 2},
            ),
            _make_scenario(
                name="C4_six_experts",
                description="Tests whether additional experts saturate or degrade performance through overspecialization.",
                groups=("C", "experts"),
                overrides={"sfgdnet_try_num_gates": 6},
            ),
        ]
    )

    # Block D — Padding mechanism
    scenarios.extend(
        [
            _make_scenario(
                name="D1_repeat_last_padding",
                description="No learned padding; baseline.",
                groups=("D", "padding"),
                overrides={
                    "sfgdnet_try_padding_mode": "repeat_last",
                    "sfgdnet_try_force_mlp_padding": False,
                },
            ),
            _make_scenario(
                name="D3_mlp_full_horizon",
                description="Predicts the full forecast horizon at once; tests whether horizon-wide padding is beneficial.",
                groups=("D", "padding"),
                overrides={
                    "sfgdnet_try_padding_mode": "mlp",
                    "sfgdnet_try_force_mlp_padding": True,
                    "sfgdnet_try_padding_predictor": "mlp",
                    "sfgdnet_try_padding_full_horizon": True,
                },
            ),
            _make_scenario(
                name="D4_rnn_single_frame",
                description="Tests whether using an RNN to predict the padding frame helps.",
                groups=("D", "padding"),
                overrides={
                    "sfgdnet_try_padding_mode": "mlp",
                    "sfgdnet_try_force_mlp_padding": True,
                    "sfgdnet_try_padding_predictor": "rnn",
                    "sfgdnet_try_padding_full_horizon": False,
                },
            ),
        ]
    )

    # Block E — Spline basis structure
    scenarios.extend(
        [
            _make_scenario(
                name="E1_degree0_minimal_basis",
                description="Piecewise-constant gating; minimal capacity.",
                groups=("E", "spline"),
                overrides={
                    "sfgdnet_try_gate_basis_degree": 0,
                    "sfgdnet_try_gate_basis_size": 1,
                },
            ),
            _make_scenario(
                name="E2_degree0_wide_basis",
                description="Step-wise gating with higher capacity.",
                groups=("E", "spline"),
                overrides={
                    "sfgdnet_try_gate_basis_degree": 0,
                    "sfgdnet_try_gate_basis_size": 16,
                },
            ),
            _make_scenario(
                name="E3_degree1_linear_basis",
                description="Linear spline gating; tests usefulness of mild smoothness.",
                groups=("E", "spline"),
                overrides={
                    "sfgdnet_try_gate_basis_degree": 1,
                    "sfgdnet_try_gate_basis_size": 16,
                },
            ),
            _make_scenario(
                name="E4_degree3_smooth_basis",
                description="Full spline gates with strong smoothness (BASELINE). Tests whether continuity is required.",
                groups=("E", "spline"),
                overrides={
                    "sfgdnet_try_gate_basis_degree": 3,
                    "sfgdnet_try_gate_basis_size": 32,
                },
            ),
        ]
    )

    # Block F — Gate regularization
    scenarios.extend(
        [
            _make_scenario(
                name="F1_no_gate_regularization",
                description="No regularization; baseline gate behavior.",
                groups=("F", "regularization"),
                overrides={
                    "sfgdnet_try_lambda_l1": 0.0,
                    "sfgdnet_try_lambda_div": 0.0,
                },
            ),
            _make_scenario(
                name="F2_l1_only",
                description="Encourages experts to use sparse frequency regions.",
                groups=("F", "regularization"),
                overrides={
                    "sfgdnet_try_lambda_l1": 1e-3,
                    "sfgdnet_try_lambda_div": 0.0,
                },
            ),
            _make_scenario(
                name="F3_diversity_only",
                description="Encourages diversity across experts.",
                groups=("F", "regularization"),
                overrides={
                    "sfgdnet_try_lambda_l1": 0.0,
                    "sfgdnet_try_lambda_div": 1e-3,
                },
            ),
        ]
    )

    # Block G — Routing mechanism
    scenarios.extend(
        [
            _make_scenario(
                name="G2_top1_routing",
                description="Hard top-k routing with k=1; each frequency goes to the most active expert.",
                groups=("G", "routing"),
                overrides={"sfgdnet_try_routing": "top1"},
            ),
            _make_scenario(
                name="G3_hard_partition_routing",
                description="Explicitly partitions the frequency axis into hard bands.",
                groups=("G", "routing"),
                overrides={"sfgdnet_try_routing": "hard_partition"},
            ),
        ]
    )

    # Block H — Backbone behaviour
    scenarios.extend(
        [
            _make_scenario(
                name="H2_mixed_projections",
                description="Alternates spatial/temporal projections; tests coupling.",
                groups=("H", "backbone"),
                overrides={
                    "simlpe_mix_spatial_temporal": True,
                    "simlpe_norm_axis": "spatial",
                    "sfgdnet_try_num_layers": default_layers,
                },
            ),
        ]
    )

    return scenarios


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
    parser.add_argument(
        "--model",
        type=str,
        default="sfgdnet_try",
        help="Model key to pull from the best-model JSON (default: sfgdnet_try).",
    )
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
    best_cfg = _load_best_config(best_json_path, args.model)

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

    train_cfg_template = _build_train_cfg(best_cfg, model_name=args.model)
    train_cfg_template.model = args.model
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

    default_total_len = ds_cfg_template.input_n + ds_cfg_template.output_n
    if train_cfg_template.dct_keep_coeffs is None:
        baseline_dct_len = default_total_len
    else:
        baseline_dct_len = int(train_cfg_template.dct_keep_coeffs)
    candidate_layers = [
        getattr(train_cfg_template, "sfgdnet_try_num_layers", None),
        train_cfg_template.simlpe_fast_layers,
        train_cfg_template.simlpe_slow_layers,
    ]
    baseline_layers = next((int(val) for val in candidate_layers if isinstance(val, int)), 6)
    baseline_layers = max(1, baseline_layers)

    seeds_cli = _parse_comma_list(args.seeds)
    if seeds_cli:
        seed_values = [int(value) for value in seeds_cli]
    elif args.seed is not None:
        seed_values = [int(args.seed)]
    else:
        seed_values = [ds_cfg_template.seed]

    scenarios: List[Dict[str, object]] = _build_sfgdnet_try_scenarios(baseline_dct_len, baseline_layers)
    scenario_filter = _parse_comma_list(args.scenarios)
    if scenario_filter is not None:
        scenario_filter_set = {name.lower() for name in scenario_filter}
        scenarios = [sc for sc in scenarios if sc["name"].lower() in scenario_filter_set]
        if not scenarios:
            raise SystemExit(f"No scenarios match the provided filter: {scenario_filter}.")

    results_dir = os.path.join(args.results_root, dataset_name, "sfgdnet_try_architecture")
    _ensure_dir(results_dir)
    results_csv = os.path.join(results_dir, args.results_name)
    results_csv = _append_window_horizon(results_csv, ds_cfg_template.input_n, ds_cfg_template.output_n)
    if action_filter:
        results_csv = results_csv.replace(".csv", f"_{action_filter}.csv")

    save_root_base = os.path.join(args.save_root, dataset_name, "sfgdnet_try_architecture")
    _ensure_dir(save_root_base)

    fieldnames = list(BASE_FIELDNAMES)
    extra_fields = [
        "ablation",
        "ablation_desc",
        "ablation_groups",
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
                "dct_keep_coeffs",
                "sfgdnet_try_skip_dct",
                "sfgdnet_try_num_gates",
                "sfgdnet_try_padding_mode",
                "sfgdnet_try_force_mlp_padding",
                "sfgdnet_try_padding_predictor",
                "sfgdnet_try_padding_full_horizon",
                "sfgdnet_try_gate_basis_degree",
                "sfgdnet_try_gate_basis_size",
                "sfgdnet_try_lambda_l1",
                "sfgdnet_try_lambda_div",
                "sfgdnet_try_routing",
                "sfgdnet_try_num_layers",
                "simlpe_mix_spatial_temporal",
                "simlpe_norm_axis",
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
