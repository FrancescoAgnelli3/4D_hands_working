import csv
import itertools
import math
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Set, Tuple

from torch.utils.data import DataLoader

from config import DatasetCfg, TrainCfg, build_argument_parser, parse_list
from data import build_datasets, get_dataset_metadata, make_loaders
from runner import run_experiment


CONFIG_SIGNATURE_FIELDS = [
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
    "SplineEqNet_num_layers",
    "epochs",
]


def _normalize_config_value(value: Optional[object]) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _config_signature(ds: DatasetCfg, cfg: TrainCfg, seed: int) -> Tuple[str, ...]:
    values = {
        "model": cfg.model,
        "dataset": ds.dataset,
        "seed": seed,
        "input_n": ds.input_n,
        "output_n": ds.output_n,
        "stride": ds.stride,
        "batch_size": ds.batch_size,
        "hidden_size": cfg.hidden_size,
        "gru_layers": cfg.gru_layers,
        "t_layers": cfg.t_layers,
        "gnn_layers": cfg.gnn_layers,
        "lr": cfg.lr,
        "gradient_clip": cfg.gradient_clip,
        "use_space": cfg.use_space,
        "bone_loss_weight": cfg.bone_loss_weight,
        "dct_keep_coeffs": cfg.dct_keep_coeffs,
        "hisrep_hand_dct_n": cfg.hisrep_hand_dct_n,
        "hisrep_dct_n": cfg.hisrep_dct_n,
        "simlpe_norm_axis": cfg.simlpe_norm_axis,
        "simlpe_use_norm": cfg.simlpe_use_norm,
        "simlpe_use_spatial_fc_only": cfg.simlpe_use_spatial_fc_only,
        "simlpe_mix_spatial_temporal": cfg.simlpe_mix_spatial_temporal,
        "simlpe_add_last_offset": cfg.simlpe_add_last_offset,
        "simlpe_mask_temperature": cfg.simlpe_mask_temperature,
        "simlpe_slow_layers": cfg.simlpe_slow_layers,
        "simlpe_fast_layers": cfg.simlpe_fast_layers,
        "simlpe_mid_gate_max": cfg.simlpe_mid_gate_max,
        "SplineEqNet_num_layers": cfg.SplineEqNet_num_layers,
        "epochs": cfg.epochs,
    }
    return tuple(_normalize_config_value(values[field]) for field in CONFIG_SIGNATURE_FIELDS)


def _load_existing_configs(csv_path: str) -> Set[Tuple[str, ...]]:
    configs: Set[Tuple[str, ...]] = set()
    if not os.path.exists(csv_path):
        return configs
    with open(csv_path, "r", newline="") as existing_file:
        reader = csv.DictReader(existing_file)
        if not reader.fieldnames:
            return configs
        for row in reader:
            configs.add(tuple((row.get(field, "") or "") for field in CONFIG_SIGNATURE_FIELDS))
    return configs


def _append_window_horizon(path: str, window: int, horizon: int) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_w{window}_h{horizon}{ext}"


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = ""

    parser = build_argument_parser(base_dir, default_data_dir)
    args = parser.parse_args()

    dataset_name = args.dataset.lower()
    metadata = get_dataset_metadata(dataset_name)

    default_wrist_indices: Tuple[int, ...] = tuple(metadata.get("default_wrist_indices", ()))
    if args.wrist_indices:
        wrist_indices_list = parse_list(args.wrist_indices, int)
        if default_wrist_indices and len(wrist_indices_list) != len(default_wrist_indices):
            raise SystemExit(
                f"Expected --wrist-indices to provide exactly {len(default_wrist_indices)} integers for dataset '{dataset_name}'."
            )
        wrist_indices = tuple(int(x) for x in wrist_indices_list)
    else:
        wrist_indices = default_wrist_indices

    if not wrist_indices:
        raise SystemExit("Dataset metadata must provide wrist indices, or specify them via --wrist-indices.")

    data_dir = args.data_dir or metadata.get("default_dir", "")
    action_filter = args.action_filter or metadata.get("default_action_filter", "")
    node_count = metadata.get("node_count", 42)
    edge_index = tuple(metadata.get("edge_index", ()))
    adjacency = tuple(metadata.get("adjacency", ()))

    ds = DatasetCfg(
        data_dir=data_dir,
        action_filter=action_filter,
        input_n=args.input_n,
        output_n=args.output_n,
        stride=args.stride,
        time_interp=args.time_interp,
        window_norm=args.window_norm,
        batch_size=args.batch_size,
        seed=args.seed,
        wrist_indices=wrist_indices,
        dataset=dataset_name,
        node_count=node_count,
        edge_index=edge_index,
        adjacency=adjacency,
    )

    seeds = parse_list(args.seeds, int) if args.seeds else [ds.seed]

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    hidden_sizes = parse_list(args.hidden_sizes, int)
    gru_layers = parse_list(args.gru_layers, int)
    lrs = parse_list(args.lrs, float)
    use_space_list = [bool(int(x)) for x in args.use_space.split(',') if x != '']
    bone_ws = parse_list(args.bone_loss_weights, float)
    t_layers_list = parse_list(args.t_layers, int)
    gnn_layers_list = parse_list(args.gnn_layers, int)
    SplineEqNet_num_layers = parse_list(args.SplineEqNet_num_layers, int) if args.SplineEqNet_num_layers else []
    if not SplineEqNet_num_layers:
        SplineEqNet_num_layers = [1, 2, 4, 8]
    simlpe_norm_axes = parse_list(args.simlpe_norm_axes, str) if args.simlpe_norm_axes else ["all"]
    simlpe_use_norms = parse_list(args.simlpe_use_norms, int) if args.simlpe_use_norms else []
    if simlpe_use_norms:
        simlpe_use_norms = [bool(x) for x in simlpe_use_norms]
    else:
        simlpe_use_norms = [None]
    simlpe_spatial_fc_only_raw = parse_list(args.simlpe_spatial_fc_only, int) if args.simlpe_spatial_fc_only else []
    simlpe_spatial_fc_only_flags = [bool(x) for x in simlpe_spatial_fc_only_raw] if simlpe_spatial_fc_only_raw else [None]
    simlpe_mix_spatial_temporal_raw = parse_list(args.simlpe_mix_spatial_temporal, int) if args.simlpe_mix_spatial_temporal else []
    simlpe_mix_spatial_temporal_flags = [bool(x) for x in simlpe_mix_spatial_temporal_raw] if simlpe_mix_spatial_temporal_raw else [None]
    simlpe_add_last_offsets_raw = parse_list(args.simlpe_add_last_offsets, int) if args.simlpe_add_last_offsets else []
    simlpe_add_last_offsets = [bool(x) for x in simlpe_add_last_offsets_raw] if simlpe_add_last_offsets_raw else [None]
    simlpe_dct_components = parse_list(args.simlpe_dct_components, int) if args.simlpe_dct_components else []
    if not simlpe_dct_components:
        simlpe_dct_components = [None]
    simlpe_mask_temperatures = parse_list(args.simlpe_mask_temperatures, float) if args.simlpe_mask_temperatures else []
    if not simlpe_mask_temperatures:
        simlpe_mask_temperatures = [None]
    simlpe_slow_layers = parse_list(args.simlpe_slow_layers, int) if args.simlpe_slow_layers else []
    if not simlpe_slow_layers:
        simlpe_slow_layers = [None]
    simlpe_fast_layers = parse_list(args.simlpe_fast_layers, int) if args.simlpe_fast_layers else []
    if not simlpe_fast_layers:
        simlpe_fast_layers = [None]
    simlpe_mid_gate_maxes = parse_list(args.simlpe_mid_gate_maxes, float) if args.simlpe_mid_gate_maxes else []
    if not simlpe_mid_gate_maxes:
        simlpe_mid_gate_maxes = [None]
    grad_clips = parse_list(args.gradient_clips, float)
    hisrep_hand_dct_ns = parse_list(args.hisrep_hand_dct_ns, int)
    if not hisrep_hand_dct_ns:
        hisrep_hand_dct_ns = [10]
    hisrep_dct_ns = parse_list(args.hisrep_dct_ns, int) if args.hisrep_dct_ns else []
    if not hisrep_dct_ns:
        hisrep_dct_ns = [10]

    SplineEqNet_num_grid = list(SplineEqNet_num_layers)

    if args.preset == "basic":
        models_grid = models
        hs_grid = hidden_sizes
        gl_grid = [gru_layers[0]] if gru_layers else [2]
        lr_grid = [lrs[0]] if lrs else [1e-3]
        space_grid = [use_space_list[0]] if use_space_list else [True]
        bone_grid = [bone_ws[0]] if bone_ws else [0.0]
        tl_grid = [t_layers_list[0]] if t_layers_list else [2]
        gnn_grid = [gnn_layers_list[0]] if gnn_layers_list else [2]
        gc_grid = [grad_clips[0]] if grad_clips else [10.0]
        SplineEqNet_num_grid = SplineEqNet_num_layers
    elif args.preset == "extended":
        models_grid = models
        hs_grid = hidden_sizes
        gl_grid = [gru_layers[0], gru_layers[-1]] if len(gru_layers) > 1 else gru_layers
        lr_grid = lrs
        space_grid = [True, False]
        bone_grid = [0.0, 0.01]
        tl_grid = [t_layers_list[0], t_layers_list[-1]] if len(t_layers_list) > 1 else t_layers_list
        gnn_grid = [gnn_layers_list[0], gnn_layers_list[-1]] if len(gnn_layers_list) > 1 else gnn_layers_list
        gc_grid = grad_clips
    else:
        models_grid = models
        hs_grid = hidden_sizes
        gl_grid = gru_layers
        lr_grid = lrs
        space_grid = use_space_list
        bone_grid = bone_ws
        tl_grid = t_layers_list
        gnn_grid = gnn_layers_list
        gc_grid = grad_clips

    save_root = os.path.join(args.save_root, dataset_name)
    os.makedirs(save_root, exist_ok=True)
    if args.csv:
        csv_dir, csv_file = os.path.split(args.csv)
        results_dir = os.path.join(csv_dir, dataset_name) if csv_dir else dataset_name
        os.makedirs(results_dir, exist_ok=True)
        results_csv = os.path.join(results_dir, csv_file)
    else:
        results_csv = os.path.join(save_root, "results.csv")

    if ds.action_filter:
        results_csv = results_csv.replace(".csv", f"_{args.action_filter}.csv")

    results_csv = _append_window_horizon(results_csv, ds.input_n, ds.output_n)
    wandb_logging_requested = bool(args.log_wandb or args.wandb_project)

    dataset_cache: Dict[Tuple, Tuple[object, object, object, int]] = {}
    save_eval_examples_flag = bool(args.save_eval_examples)

    def _apply_save_flag(cfg: TrainCfg) -> TrainCfg:
        cfg.save_eval_examples = save_eval_examples_flag
        return cfg

    def get_loaders_for(seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        key = (
            seed,
            ds.input_n,
            ds.output_n,
            ds.stride,
            ds.time_interp,
            ds.window_norm,
            ds.batch_size,
            ds.action_filter,
            ds.wrist_indices,
            ds.dataset,
            ds.node_count,
            ds.edge_index,
            ds.adjacency,
        )
        if key not in dataset_cache:
            dsc = DatasetCfg(**{**asdict(ds), "seed": seed})
            train_dataset, val_dataset, test_dataset = build_datasets(dsc)
            dataset_cache[key] = (train_dataset, val_dataset, test_dataset, dsc.batch_size)
        train_dataset, val_dataset, test_dataset, batch_size = dataset_cache[key]
        return make_loaders(train_dataset, val_dataset, test_dataset, batch_size, seed)

    exps: List[Tuple[int, TrainCfg]] = []
    for seed in seeds:
        for model in models_grid:
            fixed5_models = {"estag"}
            if model == "last":
                epochs_for_model = 1
            elif model in fixed5_models:
                epochs_for_model = 5
            else:
                epochs_for_model = args.epochs
            if model == "last":
                exps.append((seed, _apply_save_flag(TrainCfg(model=model, epochs=epochs_for_model))))
            elif model == "mpt":
                for hs, gl, lr, sp, bw, gc in itertools.product(hs_grid, gl_grid, lr_grid, space_grid, bone_grid, gc_grid):
                    exps.append((seed, _apply_save_flag(TrainCfg(model=model, epochs=epochs_for_model, lr=lr, hidden_size=hs, gru_layers=gl, use_space=sp, bone_loss_weight=bw, gradient_clip=gc))))
            elif model == "hisrep_hand":
                for hs, gl, lr, gc, dct_val in itertools.product(
                    hs_grid,
                    gl_grid,
                    lr_grid,
                    gc_grid,
                    hisrep_hand_dct_ns,
                ):
                    exps.append((
                        seed,
                        _apply_save_flag(
                            TrainCfg(
                                model=model,
                                epochs=epochs_for_model,
                                lr=lr,
                                hidden_size=hs,
                                gru_layers=gl,
                                gradient_clip=gc,
                                hisrep_hand_dct_n=dct_val,
                            )
                        ),
                    ))
            elif model == "hisrep":
                for hs, gl, lr, gc, dct_val in itertools.product(hs_grid, gl_grid, lr_grid, gc_grid, hisrep_dct_ns):
                    exps.append((
                        seed,
                        _apply_save_flag(
                            TrainCfg(
                                model=model,
                                epochs=epochs_for_model,
                                lr=lr,
                                hidden_size=hs,
                                gru_layers=gl,
                                gradient_clip=gc,
                                hisrep_dct_n=dct_val,
                            )
                        ),
                    ))
            elif model in ("simlpe", "simlpe_dct"):
                for hs, gl, lr, gc, norm_axis, use_norm, spatial_fc, mix_flag, add_offset, dct_val in itertools.product(
                    hs_grid,
                    gl_grid,
                    lr_grid,
                    gc_grid,
                    simlpe_norm_axes,
                    simlpe_use_norms,
                    simlpe_spatial_fc_only_flags,
                    simlpe_mix_spatial_temporal_flags,
                    simlpe_add_last_offsets,
                    simlpe_dct_components,
                ):
                    exps.append((
                        seed,
                        _apply_save_flag(
                            TrainCfg(
                                model=model,
                                epochs=epochs_for_model,
                                lr=lr,
                                hidden_size=hs,
                                gru_layers=gl,
                                gradient_clip=gc,
                                simlpe_norm_axis=norm_axis,
                                simlpe_use_norm=use_norm,
                                simlpe_use_spatial_fc_only=spatial_fc,
                                simlpe_mix_spatial_temporal=mix_flag,
                                simlpe_add_last_offset=add_offset,
                                dct_keep_coeffs=dct_val if model != "simlpe" else None,
                            )
                        ),
                    ))
            elif model == "SFgDNet":
                for hs, lr, gc, norm_axis, use_norm, spatial_fc, mix_flag, add_offset, dct_val, temp, slow_lay, fast_lay, gate_max in itertools.product(
                    hs_grid,
                    lr_grid,
                    gc_grid,
                    simlpe_norm_axes,
                    simlpe_use_norms,
                    simlpe_spatial_fc_only_flags,
                    simlpe_mix_spatial_temporal_flags,
                    simlpe_add_last_offsets,
                    simlpe_dct_components,
                    simlpe_mask_temperatures,
                    simlpe_slow_layers,
                    simlpe_fast_layers,
                    simlpe_mid_gate_maxes,
                ):
                    exps.append((
                        seed,
                        _apply_save_flag(
                            TrainCfg(
                                model=model,
                                epochs=epochs_for_model,
                                lr=lr,
                                hidden_size=hs,
                                gradient_clip=gc,
                                simlpe_norm_axis=norm_axis,
                                simlpe_use_norm=use_norm,
                                simlpe_use_spatial_fc_only=spatial_fc,
                                simlpe_mix_spatial_temporal=mix_flag,
                                simlpe_add_last_offset=add_offset,
                                dct_keep_coeffs=dct_val,
                                simlpe_mask_temperature=temp,
                                simlpe_slow_layers=slow_lay,
                                simlpe_fast_layers=fast_lay,
                                simlpe_mid_gate_max=gate_max,
                            )
                        ),
                    ))
            elif model == "SplineEqNet":
                if args.preset == "basic":
                    norm_axis = simlpe_norm_axes[0] if simlpe_norm_axes else None
                    use_norm = simlpe_use_norms[0] if simlpe_use_norms else None
                    spatial_fc = simlpe_spatial_fc_only_flags[0] if simlpe_spatial_fc_only_flags else None
                    mix_flag = simlpe_mix_spatial_temporal_flags[0] if simlpe_mix_spatial_temporal_flags else None
                    add_offset = simlpe_add_last_offsets[0] if simlpe_add_last_offsets else None
                    dct_val = simlpe_dct_components[0] if simlpe_dct_components else None
                    temp = simlpe_mask_temperatures[0] if simlpe_mask_temperatures else None
                    slow_lay = simlpe_slow_layers[0] if simlpe_slow_layers else None
                    fast_lay = simlpe_fast_layers[0] if simlpe_fast_layers else None
                    gate_max = simlpe_mid_gate_maxes[0] if simlpe_mid_gate_maxes else None
                    for hs, lr, gc, moe_layers in itertools.product(hs_grid, lr_grid, gc_grid, SplineEqNet_num_grid):
                        exps.append((
                            seed,
                            _apply_save_flag(
                                TrainCfg(
                                    model=model,
                                    epochs=epochs_for_model,
                                    lr=lr,
                                    hidden_size=hs,
                                    gradient_clip=gc,
                                    simlpe_norm_axis=norm_axis,
                                    simlpe_use_norm=use_norm,
                                    simlpe_use_spatial_fc_only=spatial_fc,
                                    simlpe_mix_spatial_temporal=mix_flag,
                                    simlpe_add_last_offset=add_offset,
                                    dct_keep_coeffs=dct_val,
                                    simlpe_mask_temperature=temp,
                                    simlpe_slow_layers=slow_lay,
                                    simlpe_fast_layers=fast_lay,
                                    simlpe_mid_gate_max=gate_max,
                                    SplineEqNet_num_layers=moe_layers,
                                )
                            ),
                        ))
                else:
                    for hs, lr, gc, norm_axis, use_norm, spatial_fc, mix_flag, add_offset, dct_val, temp, slow_lay, fast_lay, gate_max, moe_layers in itertools.product(
                        hs_grid,
                        lr_grid,
                        gc_grid,
                        simlpe_norm_axes,
                        simlpe_use_norms,
                        simlpe_spatial_fc_only_flags,
                        simlpe_mix_spatial_temporal_flags,
                        simlpe_add_last_offsets,
                        simlpe_dct_components,
                        simlpe_mask_temperatures,
                        simlpe_slow_layers,
                        simlpe_fast_layers,
                        simlpe_mid_gate_maxes,
                        SplineEqNet_num_grid,
                    ):
                        exps.append((
                            seed,
                            _apply_save_flag(
                                TrainCfg(
                                    model=model,
                                    epochs=epochs_for_model,
                                    lr=lr,
                                    hidden_size=hs,
                                    gradient_clip=gc,
                                    simlpe_norm_axis=norm_axis,
                                    simlpe_use_norm=use_norm,
                                    simlpe_use_spatial_fc_only=spatial_fc,
                                    simlpe_mix_spatial_temporal=mix_flag,
                                    simlpe_add_last_offset=add_offset,
                                    dct_keep_coeffs=dct_val,
                                    simlpe_mask_temperature=temp,
                                    simlpe_slow_layers=slow_lay,
                                    simlpe_fast_layers=fast_lay,
                                    simlpe_mid_gate_max=gate_max,
                                    SplineEqNet_num_layers=moe_layers,
                                )
                            ),
                        ))
            elif model in ("hisrep_rigidwrist", "eqmotion"):
                for hs, gl, lr, gc in itertools.product(hs_grid, gl_grid, lr_grid, gc_grid):
                    exps.append((
                        seed,
                        _apply_save_flag(
                            TrainCfg(
                                model=model,
                                epochs=epochs_for_model,
                                lr=lr,
                                hidden_size=hs,
                                gru_layers=gl,
                                gradient_clip=gc,
                            )
                        ),
                    ))
            elif model == "trajdep":
                for hs, lr, gls, gc in itertools.product(hs_grid, lr_grid, gnn_grid, gc_grid):
                    exps.append((
                        seed,
                        _apply_save_flag(
                            TrainCfg(
                                model=model,
                                epochs=epochs_for_model,
                                lr=lr,
                                hidden_size=hs,
                                gnn_layers=gls,
                                gradient_clip=gc,
                            )
                        ),
                    ))
            elif model == "pgbig":
                for hs, lr, gc in itertools.product(hs_grid, lr_grid, gc_grid):
                    exps.append((
                        seed,
                        _apply_save_flag(
                            TrainCfg(
                                model=model,
                                epochs=epochs_for_model,
                                lr=lr,
                                hidden_size=hs,
                                gradient_clip=gc,
                            )
                        ),
                    ))
            elif model == "motionmixer":
                for hs, blocks, lr, gc in itertools.product(hs_grid, gl_grid, lr_grid, gc_grid):
                    exps.append((
                        seed,
                        _apply_save_flag(
                            TrainCfg(
                                model=model,
                                epochs=epochs_for_model,
                                lr=lr,
                                hidden_size=hs,
                                gru_layers=blocks,
                                mixer_blocks=blocks,
                                gradient_clip=gc,
                            )
                        ),
                    ))
            elif model == "mamba":
                for hs, lr, gls, gc in itertools.product(hs_grid, lr_grid, gnn_grid, gc_grid):
                    exps.append((
                        seed,
                        _apply_save_flag(
                            TrainCfg(
                                model=model,
                                epochs=epochs_for_model,
                                lr=lr,
                                hidden_size=hs,
                                gnn_layers=gls,
                                gradient_clip=gc,
                            )
                        ),
                    ))
            elif model in ("estag", "hstgcnn", "gwnet"):
                for hs, lr, gc in itertools.product(hs_grid, lr_grid, gc_grid):
                    exps.append((
                        seed,
                        _apply_save_flag(
                            TrainCfg(
                                model=model,
                                epochs=epochs_for_model,
                                lr=lr,
                                hidden_size=hs,
                                gradient_clip=gc,
                            )
                        ),
                    ))

    if args.max_runs is not None:
        exps = exps[: args.max_runs]

    print(f"Planned experiments: {len(exps)}")

    fieldnames = [
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
        "SplineEqNet_num_layers",
        "epochs",
        "params",
        "train_mpjpe_best",
        "train_mpjpe_norm_best",
        "validation_mpjpe_best",
        "validation_mpjpe_norm_best",
        "test_mpjpe_best",
        "test_mpjpe_norm_best",
    ]
    write_header = not os.path.exists(results_csv)
    if not write_header:
        existing_rows: List[Dict[str, str]] = []
        header_needs_rewrite = False
        with open(results_csv, "r", newline="") as existing_file:
            reader = csv.DictReader(existing_file)
            existing_header = reader.fieldnames or []
            if not existing_header:
                write_header = True
            elif existing_header != fieldnames:
                header_needs_rewrite = True
                existing_rows = list(reader)
        if header_needs_rewrite:
            with open(results_csv, "w", newline="") as rewrite_file:
                writer = csv.DictWriter(rewrite_file, fieldnames=fieldnames)
                writer.writeheader()
                for row in existing_rows:
                    row_copy = dict(row)
                    writer.writerow({fname: row_copy.get(fname, "") for fname in fieldnames})
            write_header = False
        elif not header_needs_rewrite:
            write_header = False

    existing_configs = _load_existing_configs(results_csv)

    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, (seed, tr) in enumerate(exps, 1):
            config_signature = _config_signature(ds, tr, seed)
            if config_signature in existing_configs:
                print(f"\n--- [{i}/{len(exps)}] seed={seed} model={tr.model} ---")
                print("Configuration already present in results CSV; skipping experiment.")
                continue
            print(f"\n--- [{i}/{len(exps)}] seed={seed} model={tr.model} ---")
            train_loader, val_loader, test_loader = get_loaders_for(seed)
            metrics = run_experiment(
                ds=DatasetCfg(**{**asdict(ds), "seed": seed}),
                train_cfg=tr,
                save_root=save_root,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                log_wandb=wandb_logging_requested,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_run_prefix=args.wandb_run_prefix,
            )
            if metrics is None:
                print("No metrics returned for this experiment; moving on.")
                continue
            test_best_mpjpe = metrics.get("test_mpjpe_best", float("nan"))
            if test_best_mpjpe is None or math.isnan(float(test_best_mpjpe)):
                print("Encountered NaN test loss; skipping to next experiment.")
                continue
            test_best_mpjpe_norm = metrics.get("test_mpjpe_norm_best", float("nan"))
            train_best_mpjpe = metrics.get("train_mpjpe_best", float("nan"))
            train_best_mpjpe_norm = metrics.get("train_mpjpe_norm_best", float("nan"))
            val_best_mpjpe = metrics.get("validation_mpjpe_best", float("nan"))
            val_best_mpjpe_norm = metrics.get("validation_mpjpe_norm_best", float("nan"))

            row = {
                "timestamp": int(time.time()),
                "model": tr.model,
                "dataset": ds.dataset,
                "seed": seed,
                "input_n": ds.input_n,
                "output_n": ds.output_n,
                "stride": ds.stride,
                "batch_size": ds.batch_size,
                "hidden_size": tr.hidden_size,
                "gru_layers": tr.gru_layers,
                "t_layers": tr.t_layers,
                "gnn_layers": tr.gnn_layers,
                "lr": tr.lr,
                "gradient_clip": tr.gradient_clip,
                "use_space": int(tr.use_space),
                "bone_loss_weight": tr.bone_loss_weight,
                "dct_keep_coeffs": tr.dct_keep_coeffs,
                "hisrep_hand_dct_n": tr.hisrep_hand_dct_n,
                "hisrep_dct_n": tr.hisrep_dct_n,
                "simlpe_norm_axis": tr.simlpe_norm_axis,
                "simlpe_use_norm": int(tr.simlpe_use_norm) if tr.simlpe_use_norm is not None else None,
                "simlpe_use_spatial_fc_only": int(tr.simlpe_use_spatial_fc_only) if tr.simlpe_use_spatial_fc_only is not None else None,
                "simlpe_mix_spatial_temporal": int(tr.simlpe_mix_spatial_temporal) if tr.simlpe_mix_spatial_temporal is not None else None,
                "simlpe_add_last_offset": int(tr.simlpe_add_last_offset) if tr.simlpe_add_last_offset is not None else None,
                "simlpe_mask_temperature": tr.simlpe_mask_temperature,
                "simlpe_slow_layers": tr.simlpe_slow_layers,
                "simlpe_fast_layers": tr.simlpe_fast_layers,
                "simlpe_mid_gate_max": tr.simlpe_mid_gate_max,
                "SplineEqNet_num_layers": tr.SplineEqNet_num_layers,
                "epochs": tr.epochs,
                "params": metrics.get("params", 0),
                "train_mpjpe_best": train_best_mpjpe,
                "train_mpjpe_norm_best": train_best_mpjpe_norm,
                "validation_mpjpe_best": val_best_mpjpe,
                "validation_mpjpe_norm_best": val_best_mpjpe_norm,
                "test_mpjpe_best": test_best_mpjpe,
                "test_mpjpe_norm_best": test_best_mpjpe_norm,
            }
            for key in (
                "hisrep_hand_dct_n",
                "hisrep_dct_n",
                "simlpe_norm_axis",
                "simlpe_use_norm",
                "simlpe_use_spatial_fc_only",
                "simlpe_add_last_offset",
                "simlpe_mask_temperature",
                "simlpe_slow_layers",
                "simlpe_fast_layers",
                "simlpe_mid_gate_max",
                "SplineEqNet_num_layers",
            ):
                if row[key] is None:
                    row[key] = ""
            for key in (
                "train_mpjpe_best",
                "train_mpjpe_norm_best",
                "validation_mpjpe_best",
                "validation_mpjpe_norm_best",
                "test_mpjpe_best",
                "test_mpjpe_norm_best",
            ):
                value = row.get(key)
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    row[key] = ""
            writer.writerow(row)
            f.flush()
            existing_configs.add(config_signature)
            print(f"Recorded: {row}")


if __name__ == "__main__":
    main()
