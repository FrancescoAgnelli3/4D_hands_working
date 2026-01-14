import copy
import math
import os
import time
from typing import Optional, Dict, List, Tuple, Any

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

__all__ = ["train"]

# Local deps from MPT
from models.mpt import model_4GRU
from models.simlpe import SiMLPeConfig, SiMLPeForecaster
from models.simlpe_dct import SiMLPeDCTConfig, SiMLPeDCTForecaster
from models.simlpe_wave import SiMLPeWaveletConfig, SiMLPeWaveletForecaster
from models.SFgDNet import SFgDNetConfig, SFgDNetForecaster
from models.SplineEqNet import SplineEqNetConfig, SplineEqNetForecaster
from models import hisrep as hisrep_module
from models import hisrep_hand as hisrep_handmodule
from models import hisrep_rigidwrist as hisrep_rigidwrist_module
from models.eqmotion import EqMotionConfig, EqMotionForecaster
from models.trajdep import TrajDepConfig, TrajDepForecaster
from models.pgbig import PGBIGConfig, PGBIGMultiStageModel
from models.motionmixer import MlpMixer
import utils as U


# Hand skeleton helpers with graceful fallback if the new hisrep_hand module
# no longer exposes the legacy constants/utilities.
_HAND_BONE_LINK_1 = getattr(hisrep_handmodule, "bone_link_1", hisrep_rigidwrist_module.bone_link_1)
_HAND_BONE_LINK_2 = getattr(hisrep_handmodule, "bone_link_2", hisrep_rigidwrist_module.bone_link_2)
_HAND_BUILD_TREE = getattr(hisrep_handmodule, "build_tree", hisrep_rigidwrist_module.build_tree)
_HAND_BUILD_UNDIRECTED = getattr(hisrep_handmodule, "_build_undirected", hisrep_rigidwrist_module._build_undirected)

def _print_model_parameters(model_or_modules, title: str = "Model") -> int:
    """Calculate and print the number of trainable parameters.

    Accepts a single `nn.Module` or an iterable of modules. Returns the total
    count of trainable parameters.
    """
    def count_params(m) -> int:
        return sum(p.numel() for p in m.parameters() if getattr(p, "requires_grad", False))

    if isinstance(model_or_modules, (list, tuple)):
        total = sum(count_params(m) for m in model_or_modules)
    else:
        total = count_params(model_or_modules)

    # Human-friendly formatting
    if total >= 1_000_000:
        pretty = f"{total/1_000_000:.2f}M"
    elif total >= 1_000:
        pretty = f"{total/1_000:.2f}K"
    else:
        pretty = str(total)

    print(f"{title} parameters: {total} ({pretty})")
    return total

def _build_mpt_generators(
    input_size: int,
    hidden_size: int,
    output_n: int,
    node_num: int,
    batch_size: int,
    device: torch.device,
    *,
    adjacency: torch.Tensor,
    edge_index: Tuple[Tuple[int, int], ...],
    space: bool = True,
    gru_layers: int = 4,
):
    """Factory for the four MPT generators (x/y/z/v)."""
    graph_kwargs = {"adjacency": adjacency, "edge_index": edge_index}
    gen_x = model_4GRU.Generator(input_size, hidden_size, output_n, node_num, batch_size, space, num_layers=gru_layers, **graph_kwargs).to(device)
    gen_y = model_4GRU.Generator(input_size, hidden_size, output_n, node_num, batch_size, space, num_layers=gru_layers, **graph_kwargs).to(device)
    gen_z = model_4GRU.Generator(input_size, hidden_size, output_n, node_num, batch_size, space, num_layers=gru_layers, **graph_kwargs).to(device)
    gen_v = model_4GRU.Generator(input_size, hidden_size, output_n, node_num, batch_size, space, num_layers=gru_layers, **graph_kwargs).to(device)
    return gen_x, gen_y, gen_z, gen_v

def train(
    config: dict,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    bone_loss_weight: float = 0.0,
    model: str = "mpt",
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    log_wandb: bool = False,
    wandb_run: Optional[Any] = None,
):
    device = U.device()

    raw_model_name = str(model or "").strip()
    model_key = raw_model_name.lower()
    if not model_key:
        model_key = "mpt"
    if model_key == "sfgdnet":
        model = "SFgDNet"
    elif model_key in {"splineeqnet"}:
        model = "SplineEqNet"
    else:
        model = model_key

    log_wandb = bool(log_wandb or config.get("log_wandb", False))
    wandb_handle = wandb_run
    if log_wandb:
        try:
            import wandb  # type: ignore
        except ImportError:
            print("Weights & Biases requested for logging but package is unavailable; disabling logging.")
            log_wandb = False
        else:
            if wandb_handle is None:
                wandb_handle = wandb.run
            if wandb_handle is None:
                print("Weights & Biases logging requested without an active run; call wandb.init() before train(). Disabling logging.")
                log_wandb = False

    def _maybe_add_metric(bucket: Dict[str, float], key: str, value: Optional[float]) -> None:
        if value is None:
            return
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(scalar):
            bucket[key] = scalar

    save_eval_examples = bool(config.get("save_eval_examples", False))
    params_count = 0
    node_num = int(config.get("node_num", 42))
    input_n = int(config.get("input_n", 90))
    output_n = int(config.get("output_n", 42))
    hidden_size = int(config.get("hidden_size", 128))
    t_layers = int(config.get("t_layers", 2))
    gnn_layers = int(config.get("gnn_layers", 2))
    batch_size = int(config.get("batch_size", 16))
    space = bool(config.get("use_space", True))
    gru_layers = int(config.get("gru_layers", 4))
    train_epochs = int(epochs if epochs is not None else config.get("train_epoches", 50))
    if model == "eqmotion" or model == "pgbig":
        epochs = 50
    if model == "estag":
        epochs =5
    learning_rate = float(lr if lr is not None else config.get("learning_rate", 1e-3))
    dct_keep_coeffs_cfg = config.get("dct_keep_coeffs")
    dct_keep_coeffs = int(dct_keep_coeffs_cfg) if dct_keep_coeffs_cfg is not None else None
    dual_prefix = "SFgDNet" if model == "SFgDNet" else "SplineEqNet" if model == "SplineEqNet" else None
    mask_save_path = config.get(f"{dual_prefix}_mask_path") if dual_prefix else None
    mask_logging_enabled = bool(mask_save_path) and dual_prefix is not None
    mask_epoch_history: List[torch.Tensor] = []
    gate_save_path = config.get(f"{dual_prefix}_gate_path") if dual_prefix else None
    gate_logging_enabled = bool(gate_save_path) and dual_prefix is not None
    gate_param_save_path = config.get(f"{dual_prefix}_gate_param_path") if dual_prefix else None
    gate_param_logging_enabled = bool(gate_param_save_path) and dual_prefix is not None
    gate_epoch_history: List[torch.Tensor] = []
    gate_param_snapshot: Optional[Dict[str, torch.Tensor]] = None
    velocity_loss_weight = float(config.get("velocity_loss_weight", 0.0))
    negpearson_weight = float(config.get("SplineEqNet_negpearson_weight", 0.0))

    adjacency_cfg = config.get("adjacency", ())
    adjacency_tensor = None
    if adjacency_cfg:
        adjacency_tensor = torch.tensor(adjacency_cfg, dtype=torch.float32)
    else:
        adjacency_tensor = torch.eye(node_num, dtype=torch.float32)
    if adjacency_tensor.ndim != 2:
        raise ValueError("Adjacency matrix must be a 2D square array.")
    if adjacency_tensor.shape[0] != node_num or adjacency_tensor.shape[1] != node_num:
        adj_resized = torch.zeros((node_num, node_num), dtype=torch.float32)
        rows = min(node_num, adjacency_tensor.shape[0])
        cols = min(node_num, adjacency_tensor.shape[1])
        adj_resized[:rows, :cols] = adjacency_tensor[:rows, :cols]
        adjacency_tensor = adj_resized
    adjacency_tensor = adjacency_tensor.clone().detach()
    adjacency_tensor = 0.5 * (adjacency_tensor + adjacency_tensor.transpose(0, 1))
    diag_indices = torch.arange(node_num)
    adjacency_tensor[diag_indices, diag_indices] = 1.0

    edge_index_cfg = config.get("edge_index", ())
    edge_pairs: List[Tuple[int, int]] = []
    for pair in edge_index_cfg:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            i, j = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            continue
        if 0 <= i < node_num and 0 <= j < node_num:
            edge_pairs.append((i, j))
    if not edge_pairs:
        nz = adjacency_tensor.nonzero(as_tuple=False)
        edge_pairs = [(int(i), int(j)) for i, j in nz]

    wrists_hint_cfg = config.get("wrists_hint")
    wrists_hint: Optional[List[int]] = None
    if wrists_hint_cfg is not None:
        if isinstance(wrists_hint_cfg, str):
            raw_tokens = wrists_hint_cfg.replace(";", ",").split(",")
            parsed: List[int] = []
            for token in raw_tokens:
                token = token.strip()
                if not token:
                    continue
                try:
                    parsed.append(int(float(token)))
                except ValueError:
                    continue
            filtered = [w for w in parsed if 0 <= w < node_num]
            wrists_hint = filtered if filtered else None
        elif isinstance(wrists_hint_cfg, (list, tuple)):
            parsed: List[int] = []
            for token in wrists_hint_cfg:
                try:
                    parsed.append(int(float(token)))
                except (ValueError, TypeError):
                    continue
            filtered = [w for w in parsed if 0 <= w < node_num]
            wrists_hint = filtered if filtered else None

    wrist_links = _HAND_BONE_LINK_1 + _HAND_BONE_LINK_2
    try:
        wrist_graph = _HAND_BUILD_UNDIRECTED(node_num, wrist_links)
    except Exception:
        wrist_graph = {i: [] for i in range(node_num)}
    try:
        _, _, wrists_auto, _ = _HAND_BUILD_TREE(node_num, wrist_links, wrists_hint)
    except Exception:
        wrists_auto = []
    wrists_candidates: List[int] = []
    if wrists_hint:
        wrists_candidates.extend(wrists_hint)
    wrists_candidates.extend(int(w) for w in wrists_auto if 0 <= int(w) < node_num)

    seen_wrists = set()
    ordered_wrists: List[int] = []
    for wrist_id in wrists_candidates:
        if wrist_id in seen_wrists:
            continue
        seen_wrists.add(wrist_id)
        ordered_wrists.append(wrist_id)

    fallback_wrists = [idx for idx in (5, 26) if idx < node_num]
    if not ordered_wrists:
        ordered_wrists = fallback_wrists or [0]

    node2wrist_map: List[int] = [-1] * node_num
    for root in ordered_wrists:
        stack = [root]
        while stack:
            node = stack.pop()
            if node2wrist_map[node] != -1:
                continue
            node2wrist_map[node] = root
            for neigh in wrist_graph.get(node, []):
                if 0 <= neigh < node_num and node2wrist_map[neigh] == -1:
                    stack.append(neigh)
    for idx, owner in enumerate(node2wrist_map):
        if owner == -1:
            node2wrist_map[idx] = ordered_wrists[0]
    node2wrist_tensor = torch.tensor(node2wrist_map, dtype=torch.long, device=device)

    def _delta_to_absolute(pred: torch.Tensor, last_frame: torch.Tensor) -> torch.Tensor:
        pred = pred.clone()
        pred[:, 0, :] = pred[:, 0, :] + last_frame
        return torch.cumsum(pred, dim=1)

    def _velocity_magnitude(future_seq: torch.Tensor, last_observed: torch.Tensor) -> torch.Tensor:
        """Compute per-timestep velocity magnitudes for future sequence.

        Args:
            future_seq: Tensor shaped (B, T_out, N, 3) with predicted positions.
            last_observed: Tensor shaped (B, N, 3) representing final observed pose.

        Returns:
            Tensor shaped (B, T_out, N) containing velocity magnitudes.
        """
        if future_seq.ndim != 4 or last_observed.ndim != 3:
            raise ValueError("Invalid shapes for velocity magnitude computation.")
        prev = torch.cat([last_observed.unsqueeze(1), future_seq[:, :-1]], dim=1)
        diffs = future_seq - prev
        return torch.norm(diffs, dim=-1)

    loader = train_loader

    if edge_pairs:
        edge_index_tensor_device = torch.tensor(edge_pairs, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index_tensor_device = torch.zeros((2, 0), dtype=torch.long, device=device)
    graph_edges_no_self = [(i, j) for (i, j) in edge_pairs if i != j]
    if graph_edges_no_self:
        edges_t = torch.tensor(graph_edges_no_self, dtype=torch.long, device=device)
    else:
        edges_t = torch.zeros((0, 2), dtype=torch.long, device=device)

    # Align model batch_size with the actual train loader to avoid GRU hidden-state mismatches
    if hasattr(loader, "batch_size") and loader.batch_size is not None:
        try:
            batch_size = int(loader.batch_size)
        except Exception:
            pass

    # Build models and optimizer
    simlpe = None
    sfgdnet_model = None
    SplineEqNet_model = None
    params: List[torch.nn.Parameter] = []
    eqmotion_model = None
    trajdep = None
    pgbig_model = None
    motion_mixer = None
    factorized_loss = None
    def _parse_fixed_centers_cfg(raw_value: Any) -> Optional[Tuple[float, float]]:
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            tokens = [tok.strip() for tok in raw_value.replace("|", ",").split(",") if tok.strip()]
            if len(tokens) >= 2:
                return float(tokens[0]), float(tokens[1])
        elif isinstance(raw_value, (list, tuple)):
            if len(raw_value) >= 2:
                return float(raw_value[0]), float(raw_value[1])
        return None

    def _build_dual_branch_model(
        prefix: str,
        config_cls,
        forecaster_cls,
        label: str,
    ) -> torch.nn.Module:
        nonlocal params, params_count, optimizer
        slow_layers = config.get("simlpe_slow_layers")
        fast_layers = config.get("simlpe_fast_layers")
        force_mlp_padding_cfg = config.get(f"{prefix}_force_mlp_padding")
        force_mlp_padding_flag = True if force_mlp_padding_cfg is None else bool(force_mlp_padding_cfg)
        padding_mode_cfg = config.get(f"{prefix}_padding_mode")
        padding_predictor_cfg = config.get(f"{prefix}_padding_predictor")
        padding_full_horizon_cfg = config.get(f"{prefix}_padding_full_horizon")
        learn_lambda_cfg = config.get(f"{prefix}_learn_lambda")
        learn_centers_cfg = config.get(f"{prefix}_learn_centers")
        learn_lambda_flag = True if learn_lambda_cfg is None else bool(learn_lambda_cfg)
        learn_centers_flag = True if learn_centers_cfg is None else bool(learn_centers_cfg)
        fixed_lambda_cfg = config.get(f"{prefix}_fixed_lambda")
        fixed_lambda_value = None if fixed_lambda_cfg is None else float(fixed_lambda_cfg)
        fixed_centers_cfg = config.get(f"{prefix}_fixed_centers")
        fixed_centers_value = _parse_fixed_centers_cfg(fixed_centers_cfg)
        gate_basis_degree_cfg = config.get(f"{prefix}_gate_basis_degree")
        gate_basis_size_cfg = config.get(f"{prefix}_gate_basis_size")
        num_gates_cfg = config.get(f"{prefix}_num_gates")
        num_layers_cfg = config.get(f"{prefix}_num_layers")
        routing_cfg = config.get(f"{prefix}_routing")
        sim_cfg = config_cls(
            input_length=input_n,
            pred_length=output_n,
            num_nodes=node_num,
            hidden_dim=hidden_size,
            use_norm=bool(config.get("simlpe_use_norm", True)),
            use_spatial_fc_only=bool(config.get("simlpe_spatial_fc_only", False)),
            mix_spatial_temporal=bool(config.get("simlpe_mix_spatial_temporal", False)),
            norm_axis=str(config.get("simlpe_norm_axis", "spatial")),
            add_last_offset=bool(config.get("simlpe_add_last_offset", True)),
            dct_components=dct_keep_coeffs,
            mask_temperature=float(config.get("simlpe_mask_temperature", 1)),
            slow_num_layers=None if slow_layers is None else int(slow_layers),
            fast_num_layers=None if fast_layers is None else int(fast_layers),
            mid_gate_max=float(config.get("simlpe_mid_gate_max", 0.9)),
            force_mlp_padding=force_mlp_padding_flag,
            padding_mode=str(padding_mode_cfg) if padding_mode_cfg is not None else None,
            padding_predictor=str(padding_predictor_cfg) if padding_predictor_cfg is not None else "mlp",
            padding_full_horizon=bool(padding_full_horizon_cfg) if padding_full_horizon_cfg is not None else False,
            learn_lambda=learn_lambda_flag,
            learn_centers=learn_centers_flag,
            fixed_lambda=fixed_lambda_value,
            fixed_centers=fixed_centers_value,
        )
        if gate_basis_degree_cfg is not None:
            setattr(sim_cfg, "gate_basis_degree", int(gate_basis_degree_cfg))
        if gate_basis_size_cfg is not None:
            setattr(sim_cfg, "gate_basis_size", int(gate_basis_size_cfg))
        if num_gates_cfg is not None:
            setattr(sim_cfg, "num_gates", int(num_gates_cfg))
        if num_layers_cfg is not None:
            setattr(sim_cfg, "num_layers", int(num_layers_cfg))
        if routing_cfg is not None:
            setattr(sim_cfg, "routing", str(routing_cfg))
        model_instance = forecaster_cls(sim_cfg).to(device)
        if bool(config.get(f"{prefix}_skip_dct", False)):
            model_instance.set_skip_dct(True)
        fixed_mask_cfg = config.get(f"{prefix}_fixed_mask")
        if fixed_mask_cfg is not None:
            model_instance.set_fixed_mask(float(fixed_mask_cfg))
        fixed_mid_gate_cfg = config.get(f"{prefix}_fixed_mid_gate")
        if fixed_mid_gate_cfg is not None:
            model_instance.set_fixed_mid_gate(float(fixed_mid_gate_cfg))
        if fixed_lambda_value is not None:
            model_instance.set_fixed_lambda(fixed_lambda_value, freeze=not learn_lambda_flag)
        if fixed_centers_value is not None:
            model_instance.set_fixed_centers(fixed_centers_value[0], fixed_centers_value[1], freeze=not learn_centers_flag)
        disable_slow = bool(config.get(f"{prefix}_disable_slow_branch", False))
        disable_fast = bool(config.get(f"{prefix}_disable_fast_branch", False))
        if disable_slow or disable_fast:
            model_instance.set_branch_usage(not disable_slow, not disable_fast)
        if bool(config.get(f"{prefix}_freeze_gate", False)):
            model_instance.freeze_frequency_gate(True)
        params = list(model_instance.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(model_instance, title=label)
        return model_instance

    if model == "mpt":
        gen_x, gen_y, gen_z, gen_v = _build_mpt_generators(
            input_n - 1,
            hidden_size,
            output_n,
            node_num,
            batch_size,
            device,
            adjacency=adjacency_tensor,
            edge_index=tuple(edge_pairs),
            space=space,
            gru_layers=gru_layers,
        )
        params = list(gen_x.parameters()) + list(gen_y.parameters()) + list(gen_z.parameters()) + list(gen_v.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        # Print parameter counts for each generator and total
        params_count = _print_model_parameters([gen_x, gen_y, gen_z, gen_v], title="MPT Generators (x,y,z,v)")
    elif model in ("hisrep", "hisrep_hand", "hisrep_rigidwrist"):
        # HisRep variants: AttModel-based forecasters operating on wrist/body sequences
        if model == "hisrep_rigidwrist":
            dct_keep = config.get("dct_keep_coeffs")
            dct_n = int(dct_keep) if dct_keep is not None else 10
            wrist_cfg = hisrep_rigidwrist_module.WristOnlyConfig(
                num_nodes=node_num,
                input_length=input_n,
                pred_length=output_n,
                hidden_dim=hidden_size,
                num_stages=gru_layers,
                dct_n=dct_n,
                wrists_hint=config.get("wrists_hint"),
            )
            hisrep = hisrep_rigidwrist_module.WristRigidForecaster(wrist_cfg).to(device)
            tag = "WristRigidForecaster"
        else:
            module = hisrep_module if model == "hisrep" else hisrep_handmodule
            hisrep_kwargs = {
                "num_nodes": node_num,
                "input_length": input_n,
                "pred_length": output_n,
                "hidden_dim": hidden_size,
                "num_stages": gru_layers,
            }
            wrists_hint = config.get("wrists_hint")
            if wrists_hint is not None:
                hisrep_kwargs["wrists_hint"] = wrists_hint
            if model == "hisrep_hand":
                dct_override = config.get("hisrep_handdct_n")
                dropout_cfg = config.get("hisrep_handdropout")
                if dropout_cfg is not None:
                    hisrep_kwargs["dropout"] = float(dropout_cfg)
                token_hidden_cfg = config.get("hisrep_handtoken_hidden")
                if token_hidden_cfg is not None:
                    hisrep_kwargs["token_hidden"] = int(token_hidden_cfg)
                channel_hidden_cfg = config.get("hisrep_handchannel_hidden")
                if channel_hidden_cfg is not None:
                    hisrep_kwargs["channel_hidden"] = int(channel_hidden_cfg)
                gcn_hidden_cfg = config.get("hisrep_handgcn_hidden")
                if gcn_hidden_cfg is not None:
                    hisrep_kwargs["gcn_hidden"] = int(gcn_hidden_cfg)
                input_scale_cfg = config.get("hisrep_handinput_scale")
                if input_scale_cfg is not None:
                    hisrep_kwargs["input_scale"] = float(input_scale_cfg)
            else:
                dct_override = config.get("hisrep_dct_n")
            if dct_override is not None:
                hisrep_kwargs["dct_n"] = int(dct_override)
            his_cfg = module.HisRepConfig(**hisrep_kwargs)
            hisrep = module.HisRepForecaster(his_cfg).to(device)
            tag = "HisRepForecaster" if model == "hisrep" else "HisRepHandForecaster"
        params = list(hisrep.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(hisrep, title=tag)
        if model == "hisrep_hand":
            factorized_loss = hisrep_handmodule.FactorizedLoss().to(device)
            hand_mask = torch.zeros(node_num, device=device)
            hand_mask[hisrep.distal_idx] = 1.0
            hisrep_handdir_mask = hand_mask
    elif model == "trajdep":
        traj_cfg = TrajDepConfig(
            input_length=input_n,
            pred_length=output_n,
            num_nodes=node_num,
            hidden_dim=hidden_size,
            num_stages=max(1, gnn_layers),
            dropout=float(config.get("trajdep_dropout", 0.5)),
            dct_components=int(config.get("trajdep_dct", min(input_n + output_n, 35))),
        )
        trajdep = TrajDepForecaster(traj_cfg).to(device)
        params = list(trajdep.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(trajdep, title="TrajDepGCN")
    elif model == "pgbig":
        cfg = PGBIGConfig(
            input_n=input_n,
            output_n=output_n,
            in_features=node_num * 3,
            d_model=hidden_size,
            dct_n=min(int(config.get("pgbig_dct", input_n + output_n)), input_n + output_n),
            drop_out=float(config.get("pgbig_dropout", 0.3)),
            encoder_layers=max(1, int(config.get("pgbig_encoder_layers", 1))),
            decoder_layers=max(1, int(config.get("pgbig_decoder_layers", 2))),
        )
        pgbig_model = PGBIGMultiStageModel(cfg).to(device)
        params = list(pgbig_model.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(pgbig_model, title="PGBIGMultiStageModel")
    elif model == "motionmixer":
        mixer_blocks = int(config.get("mixer_blocks", 4))
        mixer_tokens_dim = int(config.get("mixer_tokens_dim", hidden_size))
        mixer_channels_dim = int(config.get("mixer_channels_dim", hidden_size))
        mixer_regularization = float(config.get("mixer_regularization", 0.0))
        mixer_use_se = bool(config.get("mixer_use_se", True))
        mixer_activation = str(config.get("mixer_activation", "gelu"))
        mixer_block_type = str(config.get("mixer_block_type", "normal"))
        mixer_reduction = int(config.get("mixer_reduction", 4))
        mixer_use_max_pool = bool(config.get("mixer_use_max_pooling", False))
        motion_mixer = MlpMixer(
            num_classes=node_num * 3,
            num_blocks=mixer_blocks,
            hidden_dim=hidden_size,
            tokens_mlp_dim=mixer_tokens_dim,
            channels_mlp_dim=mixer_channels_dim,
            seq_len=input_n,
            pred_len=output_n,
            activation=mixer_activation,
            mlp_block_type=mixer_block_type,
            regularization=mixer_regularization,
            input_size=node_num * 3,
            reduction=mixer_reduction,
            use_max_pooling=mixer_use_max_pool,
            use_se=mixer_use_se,
        ).to(device)
        params = list(motion_mixer.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(motion_mixer, title="MotionMixer")
    elif model == "eqmotion":
        eq_cfg = EqMotionConfig(
            input_length=input_n,
            pred_length=output_n,
            num_nodes=node_num,
            hidden_dim=hidden_size,
            hid_channel=hidden_size,
        )
        eqmotion_model = EqMotionForecaster(eq_cfg).to(device)
        params = list(eqmotion_model.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(eqmotion_model, title="EqMotionForecaster")
    elif model == "estag":
        # Lazy import to keep deps optional
        from models.model import ESTAG  # wrapper points to Assembly implementation
        estag = ESTAG(
            num_past=input_n,
            num_future=output_n,
            in_node_nf=1,
            hidden_nf=hidden_size,
            fft=True,
            eat=True,
            device=str(device),
            n_layers=t_layers,
            n_nodes=node_num,
            nodes_att_dim=0,
            coords_weight=1.0,
            tempo=True,
            filter=True,
        )
        params = list(estag.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(estag, title="ESTAG")
    elif model == "simlpe":
        sim_cfg = SiMLPeConfig(
            input_length=input_n,
            pred_length=output_n,
            num_nodes=node_num,
            hidden_dim=hidden_size,
            num_layers=gru_layers,
            use_norm=bool(config.get("simlpe_use_norm", True)),
            use_spatial_fc_only=bool(config.get("simlpe_spatial_fc_only", False)),
            mix_spatial_temporal=bool(config.get("simlpe_mix_spatial_temporal", False)),
            norm_axis=str(config.get("simlpe_norm_axis", "spatial")),
            add_last_offset=bool(config.get("simlpe_add_last_offset", True)),
        )
        simlpe = SiMLPeForecaster(sim_cfg).to(device)
        params = list(simlpe.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(simlpe, title="SiMLPeForecaster")
    elif model == "simlpe_dct":
        sim_cfg = SiMLPeDCTConfig(
            input_length=input_n,
            pred_length=output_n,
            num_nodes=node_num,
            hidden_dim=hidden_size,
            num_layers=gru_layers,
            use_norm=bool(config.get("simlpe_use_norm", True)),
            use_spatial_fc_only=bool(config.get("simlpe_spatial_fc_only", False)),
            mix_spatial_temporal=bool(config.get("simlpe_mix_spatial_temporal", False)),
            norm_axis=str(config.get("simlpe_norm_axis", "spatial")),
            add_last_offset=bool(config.get("simlpe_add_last_offset", True)),
            dct_components=dct_keep_coeffs,
        )
        simlpe_dct = SiMLPeDCTForecaster(sim_cfg).to(device)
        params = list(simlpe_dct.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(simlpe_dct, title="SiMLPeDCTForecaster")
    elif model == "simlpe_wave":
        sim_cfg = SiMLPeWaveletConfig(
            input_length=input_n,
            pred_length=output_n,
            num_nodes=node_num,
            hidden_dim=hidden_size,
            num_layers=gru_layers,
            use_norm=bool(config.get("simlpe_wave_use_norm", True)),
            norm_axis=str(config.get("simlpe_wave_norm_axis", "all")),
            add_last_offset_time_domain=bool(config.get("simlpe_wave_add_last_offset", False)),
            wavelet=str(config.get("simlpe_wave_wavelet", "db4")),
            swt_level=int(config.get("simlpe_wave_swt_level", 5)),
            include_approx=bool(config.get("simlpe_wave_include_approx", True)),
        )
        simlpe_wave = SiMLPeWaveletForecaster(sim_cfg).to(device)
        params = list(simlpe_wave.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(simlpe_wave, title="SiMLPeWaveletForecaster")
    elif model == "SFgDNet":
        sfgdnet_model = _build_dual_branch_model(
            prefix="SFgDNet",
            config_cls=SFgDNetConfig,
            forecaster_cls=SFgDNetForecaster,
            label="SFgDNetForecaster",
        )
    elif model == "SplineEqNet":
        SplineEqNet_model = _build_dual_branch_model(
            prefix="SplineEqNet",
            config_cls=SplineEqNetConfig,
            forecaster_cls=SplineEqNetForecaster,
            label="SplineEqNetForecaster",
        )
    elif model == "mamba":
        from models.baselines import TimeSpaceModel
        gtype = "gcn"
        temporal_type = "mamba"
        tl = t_layers
        gl = gnn_layers
        bas = TimeSpaceModel(
            input_size=3,
            n_nodes=node_num,
            horizon=output_n,
            hidden_size=hidden_size,
            t_layers=tl,
            gnn_layers=gl,
            gnn_type=gtype,
            temporal_type=temporal_type,
        ).to(device)
        params = list(bas.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(bas, title=f"TimeSpaceModel[{model}]")
    elif model == "cde":
        from models.cde import MultiNeuralCDE
        cde = MultiNeuralCDE(input_channels=3, hidden_channels=hidden_size, output_channels=3, horizon=output_n, fast=False).to(device)
        params = list(cde.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(cde, title="MultiNeuralCDE")
    elif model == "hstgcnn":
        from models.HSTGCN import hstgcnn
        hst = hstgcnn(input_feat=3, output_feat=3, seq_len=input_n, pred_seq_len=output_n).to(device)
        params = list(hst.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(hst, title="HSTGCN")
    elif model == "gwnet":
        from models.wavenet import gwnet
        gwn = gwnet(device, node_num, in_dim=3, out_dim=3, pred_seq_len=output_n).to(device)
        params = list(gwn.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        params_count = _print_model_parameters(gwn, title="GWNet")
    else:
        gen_x = gen_y = gen_z = gen_v = None
        params = []
        optimizer = None

    model_modules: Dict[str, torch.nn.Module] = {}
    if model == "mpt" and all(m is not None for m in (gen_x, gen_y, gen_z, gen_v)):
        model_modules = {
            "gen_x": gen_x,
            "gen_y": gen_y,
            "gen_z": gen_z,
            "gen_v": gen_v,
        }
    elif model in ("hisrep", "hisrep_hand", "hisrep_rigidwrist") and 'hisrep' in locals():
        model_modules = {"hisrep": hisrep}
    elif model == "trajdep" and 'trajdep' in locals():
        model_modules = {"trajdep": trajdep}
    elif model == "pgbig" and 'pgbig_model' in locals():
        model_modules = {"pgbig_model": pgbig_model}
    elif model == "motionmixer" and 'motion_mixer' in locals():
        model_modules = {"motion_mixer": motion_mixer}
    elif model == "eqmotion" and 'eqmotion_model' in locals():
        model_modules = {"eqmotion_model": eqmotion_model}
    elif model == "simlpe" and 'simlpe' in locals():
        model_modules = {"simlpe": simlpe}
    elif model == "simlpe_dct" and 'simlpe_dct' in locals():
        model_modules = {"simlpe_dct": simlpe_dct}
    elif model == "simlpe_wave" and 'simlpe_wave' in locals():
        model_modules = {"simlpe_wave": simlpe_wave}
    elif model == "SFgDNet" and sfgdnet_model is not None:
        model_modules = {"SFgDNet": sfgdnet_model}
    elif model == "SplineEqNet" and SplineEqNet_model is not None:
        model_modules = {"SplineEqNet": SplineEqNet_model}
    elif model == "mamba" and 'bas' in locals():
        model_modules = {"bas": bas}
    elif model == "cde" and 'cde' in locals():
        model_modules = {"cde": cde}
    elif model == "hstgcnn" and 'hst' in locals():
        model_modules = {"hst": hst}
    elif model == "gwnet" and 'gwn' in locals():
        model_modules = {"gwn": gwn}
    elif model == "estag" and 'estag' in locals():
        model_modules = {"estag": estag}

    def _set_mode(training: bool) -> None:
        for module in model_modules.values():
            if training:
                module.train()
            else:
                module.eval()

    def _capture_state() -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        if not model_modules:
            return None
        return {name: copy.deepcopy(module.state_dict()) for name, module in model_modules.items()}

    def _load_state(state: Optional[Dict[str, Dict[str, torch.Tensor]]]) -> None:
        if not state:
            return
        for name, module in model_modules.items():
            if name in state:
                module.load_state_dict(state[name])

    def _get_active_dual_model() -> Optional[torch.nn.Module]:
        if model == "SFgDNet":
            return sfgdnet_model
        if model == "SplineEqNet":
            return SplineEqNet_model
        return None

    def _evaluate_loader(
        eval_loader: Optional[DataLoader],
        *,
        collect_examples: bool = False,
        restore_train: bool = True,
        save_examples: bool = False,
    ) -> Optional[Dict[str, float]]:
        if eval_loader is None:
            return None
        total_loss = 0.0
        total_loss_norm = 0.0
        total_score = 0.0
        total_velocity_mae = 0.0
        total_samples = 0
        collect_for_saving = bool(collect_examples and save_examples)
        collected_preds: List[torch.Tensor] = []
        collected_tgts: List[torch.Tensor] = []

        _set_mode(False)
        try:
            with torch.no_grad():
                for batch in eval_loader:
                    if isinstance(batch, (list, tuple)):
                        if len(batch) < 2:
                            raise ValueError("Loader batch must provide input and target tensors.")
                        inp, out = batch[0], batch[1]
                        norm_factor = batch[2] if len(batch) > 2 else None
                    else:
                        raise TypeError("Expected batch from loader to be a tuple/list.")

                    inp = inp.to(device).float()
                    out = out.to(device).float()
                    if norm_factor is not None:
                        norm_factor_t = norm_factor.to(device=device).float().view(-1)
                    else:
                        norm_factor_t = torch.ones(inp.size(0), device=device)

                    in_angle = inp[:, 1:, :, :3]
                    in_vel = inp[:, 1:, :, 3].permute(0, 2, 1)
                    tgt_3d = out[:, :, :, 4:]
                    tgt_vel = out[:, :, :, 3]
                    in_3d = inp[:, :, :, 4:]
                    in_ang_x = in_angle[:, :, :, 0].permute(0, 2, 1).contiguous()
                    in_ang_y = in_angle[:, :, :, 1].permute(0, 2, 1).contiguous()
                    in_ang_z = in_angle[:, :, :, 2].permute(0, 2, 1).contiguous()
                    B, T_out, N = tgt_3d.shape[0], tgt_3d.shape[1], tgt_3d.shape[2]

                    if model == "mpt":
                        out_v, _ = gen_v(in_vel, hidden_size)
                        out_x, _ = gen_x(in_ang_x, hidden_size)
                        out_y, _ = gen_y(in_ang_y, hidden_size)
                        out_z, _ = gen_z(in_ang_z, hidden_size)
                        pred_v = out_v.view(B, N, output_n).permute(0, 2, 1)
                        ang_x = out_x.view(B, N, output_n).permute(0, 2, 1)
                        ang_y = out_y.view(B, N, output_n).permute(0, 2, 1)
                        ang_z = out_z.view(B, N, output_n).permute(0, 2, 1)
                        pred_angles = torch.stack([ang_x, ang_y, ang_z], dim=-1)
                        init_pose = in_3d[:, -1, :, :]
                        recons = U.reconstruct_sequence(pred_v, pred_angles, init_pose, node_num)
                    elif model in ("hisrep", "hisrep_hand", "hisrep_rigidwrist"):
                        recons = hisrep(in_3d)
                    elif model == "eqmotion":
                        recons = eqmotion_model(in_3d)
                    elif model == "estag":
                        x_hist = in_3d.permute(1, 0, 2, 3).reshape(input_n, B * N, 3)
                        h = torch.ones(B * N, 1, device=device)
                        x_hat = estag(h, x_hist, edge_index_tensor_device)
                        recons = x_hat.view(T_out, B, N, 3).permute(1, 0, 2, 3)
                    elif model == "simlpe":
                        recons = simlpe(in_3d)
                    elif model == "simlpe_dct":
                        recons = simlpe_dct(in_3d)
                    elif model == "simlpe_wave":
                        recons = simlpe_wave(in_3d)
                    elif model in ("SFgDNet", "SplineEqNet"):
                        dual_model = _get_active_dual_model()
                        if dual_model is None:
                            raise RuntimeError(f"{model} model is not initialized.")
                        recons = dual_model(in_3d)
                    elif model == "pgbig":
                        seq_flat = in_3d.reshape(B, input_n, -1)
                        last_pose = seq_flat[:, -1:, :]
                        padded = torch.cat([seq_flat, last_pose.repeat(1, output_n, 1)], dim=1)
                        pred_full = pgbig_model(padded)[0]
                        recons = pred_full[:, -output_n:, :].reshape(B, T_out, N, 3)
                    elif model == "motionmixer":
                        seq_flat = in_3d.reshape(B, input_n, pose_dim)
                        tgt_flat = tgt_3d.reshape(B, T_out, pose_dim)
                        if mixer_input_scale != 1.0:
                            seq_scaled = seq_flat / mixer_input_scale
                            tgt_scaled = tgt_flat / mixer_input_scale
                        else:
                            seq_scaled = seq_flat
                            tgt_scaled = tgt_flat
                        if mixer_use_delta:
                            sequences_all = torch.cat([seq_scaled, tgt_scaled], dim=1)
                            diffs = sequences_all[:, 1:, :] - sequences_all[:, :-1, :]
                            diffs = torch.cat([diffs[:, 0:1, :], diffs], dim=1)
                            mixer_in = diffs[:, :input_n, :]
                            pred_scaled = motion_mixer(mixer_in)
                            pred_scaled = _delta_to_absolute(pred_scaled, seq_scaled[:, -1, :])
                        else:
                            mixer_in = seq_scaled
                            pred_scaled = motion_mixer(mixer_in)
                        pred_flat = pred_scaled * mixer_input_scale if mixer_input_scale != 1.0 else pred_scaled
                        recons = pred_flat.view(B, T_out, N, 3)
                    elif model == "trajdep":
                        recons = trajdep(in_3d)
                    elif model == "mamba":
                        recons = bas(in_3d, edge_index_tensor_device)
                    elif model == "cde":
                        recons = cde(in_3d, edge_index_tensor_device)
                    elif model == "hstgcnn":
                        recons = hst(in_3d, edge_index_tensor_device)
                    elif model == "gwnet":
                        recons = gwn(in_3d, edge_index_tensor_device)
                    else:
                        init_pose = in_3d[:, -1, :, :]
                        recons = init_pose.unsqueeze(1).expand(B, T_out, N, 3)

                    mpjpe_terms = torch.norm(recons - tgt_3d, dim=-1)
                    per_sample_mpjpe = mpjpe_terms.mean(dim=(1, 2))
                    per_sample_mpjpe_norm = per_sample_mpjpe * norm_factor_t

                    total_loss += float(per_sample_mpjpe.sum().item())
                    total_loss_norm += float(per_sample_mpjpe_norm.sum().item())
                    last_pose_eval = in_3d[:, -1, :, :]
                    pred_vel_mag_eval = _velocity_magnitude(recons, last_pose_eval)
                    vel_mae_eval = (pred_vel_mag_eval - tgt_vel).abs().mean(dim=(1, 2))
                    total_velocity_mae += float(vel_mae_eval.sum().item())
                    score_val = U.weighted_joint_loss(joint_weights, recons, tgt_3d, metric='mae')
                    total_score += float(score_val.item())
                    total_samples += inp.size(0)

                    if collect_for_saving:
                        collected_preds.append(recons[0].detach().cpu())
                        collected_tgts.append(tgt_3d[0].detach().cpu())
        finally:
            if restore_train:
                _set_mode(True)

        if total_samples == 0:
            return {
                "mpjpe": float("nan"),
                "mpjpe_norm": float("nan"),
                "weighted_mae": float("nan"),
                "samples": 0,
            }

        avg_loss = total_loss / total_samples
        avg_loss_norm = total_loss_norm / total_samples
        avg_score = total_score / max(1, total_samples)
        avg_velocity_mae = total_velocity_mae / max(1, total_samples)

        if collect_for_saving and collected_preds and collected_tgts:
            examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
            os.makedirs(examples_dir, exist_ok=True)
            merged_pred = torch.stack(collected_preds, dim=0)
            merged_tgt = torch.stack(collected_tgts, dim=0)
            torch.save(merged_pred, os.path.join(examples_dir, f"{model}_pred.pt"))
            torch.save(merged_tgt, os.path.join(examples_dir, f"{model}_target.pt"))

        return {
            "mpjpe": float(avg_loss),
            "mpjpe_norm": float(avg_loss_norm),
            "weighted_mae": float(avg_score),
            "velocity_mae": float(avg_velocity_mae),
            "samples": float(total_samples),
        }

    # Optional LR scheduler that reduces LR on plateau of the monitored loss
    scheduler = None
    if optimizer is not None and bool(config.get("use_lr_scheduler", True)):
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(config.get("lr_factor", 0.5)),
            patience=int(config.get("lr_patience", 5)),
            threshold=float(config.get("lr_threshold", 1e-4)),
            cooldown=int(config.get("lr_cooldown", 0)),
            min_lr=float(config.get("lr_min", 0.0))
        )

    # Compute joint weights for scoring (not used in backprop)
    joint_weights = U.compute_node_motion_weights(loader, device=device, normalize='sum')

    n_train_windows = len(getattr(loader, "dataset", []))
    print(f"Training on {n_train_windows} windows")
    print(f"Config: input_n={input_n}, output_n={output_n}, hidden_size={hidden_size}, batch_size={batch_size}, model={model}")

    pose_dim = node_num * 3
    mixer_use_delta = bool(config.get("mixer_use_delta", True))
    mixer_input_scale = float(config.get("mixer_input_scale", 1.0))

    best_test_mpjpe = None
    best_test_mpjpe_norm = None
    best_test_epoch = None
    best_test_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    best_test_velocity_mae = None

    best_train_mpjpe = None
    best_train_mpjpe_norm = None
    best_train_epoch = None
    best_train_velocity_mae = None

    # Optional: log initial SemSkeConv parameter norms to assess magnitude
    if bool(config.get("log_gcn_stats", False)) and model == "mpt":
        layers = U.find_semskeconvs(gen_x, gen_y, gen_z, gen_v)
        U.print_semskeconv_stats(U.semskeconv_stats(layers), prefix="[GCN][epoch 000]")

    for epoch in range(1, train_epochs + 1):
        _set_mode(True)
        running = 0.0
        score_running = 0.0
        mpjpe_running = 0.0
        mpjpe_norm_running = 0.0
        vel_running = 0.0
        n_seen = 0
        t0 = time.time()
        epoch_masks: List[torch.Tensor] = [] if mask_logging_enabled else []
        epoch_gate_slow: List[torch.Tensor] = [] if gate_logging_enabled else []
        epoch_gate_fast: List[torch.Tensor] = [] if gate_logging_enabled else []

        for it, batch in enumerate(loader):
            if isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    raise ValueError("Loader batch must provide input and target tensors.")
                inp, out = batch[0], batch[1]
                norm_factor = batch[2] if len(batch) > 2 else None
            else:
                raise TypeError("Expected batch from loader to be a tuple/list.")
            # Shapes: inp (B, T_in, N, 7), out (B, T_out, N, 7)
            inp = inp.to(device).float()
            out = out.to(device).float()
            if norm_factor is not None:
                norm_factor_t = norm_factor.to(device=device).float().view(-1)
            else:
                norm_factor_t = torch.ones(inp.size(0), device=device)

            # Prepare inputs/targets as in prediction_model.py
            # Drop first frame for velocity/angle inputs
            in_angle = inp[:, 1:, :, :3]
            in_vel = inp[:, 1:, :, 3].permute(0, 2, 1)  # (B, N, T_in-1)

            tgt_angle = out[:, :, :, :3]
            tgt_vel = out[:, :, :, 3]
            in_3d = inp[:, :, :, 4:]
            tgt_3d = out[:, :, :, 4:]

            # Build per-axis inputs
            in_ang_x = in_angle[:, :, :, 0].permute(0, 2, 1).contiguous()  # (B, N, T_in-1)
            in_ang_y = in_angle[:, :, :, 1].permute(0, 2, 1).contiguous()
            in_ang_z = in_angle[:, :, :, 2].permute(0, 2, 1).contiguous()

            B, T_out, N = tgt_3d.shape[0], tgt_3d.shape[1], tgt_3d.shape[2]
            aux = None

            if model == "mpt":
                optimizer.zero_grad()

                # One-shot horizon prediction: decode full sequences of v and angles
                out_v, _ = gen_v(in_vel, hidden_size)
                out_x, _ = gen_x(in_ang_x, hidden_size)
                out_y, _ = gen_y(in_ang_y, hidden_size)
                out_z, _ = gen_z(in_ang_z, hidden_size)

                # Reshape to (B, T_out, N) and (B, T_out, N, 3)
                pred_v = out_v.view(B, N, output_n).permute(0, 2, 1)
                ang_x = out_x.view(B, N, output_n).permute(0, 2, 1)
                ang_y = out_y.view(B, N, output_n).permute(0, 2, 1)
                ang_z = out_z.view(B, N, output_n).permute(0, 2, 1)
                pred_angles = torch.stack([ang_x, ang_y, ang_z], dim=-1)

                # Reconstruct 3D trajectory from last observed pose
                init_pose = in_3d[:, -1, :, :]
                recons = U.reconstruct_sequence(pred_v, pred_angles, init_pose, node_num)
            elif model == "estag":
                optimizer.zero_grad()
                x_hist = in_3d.permute(1, 0, 2, 3).reshape(input_n, B * N, 3)
                h = torch.ones(B * N, 1, device=device)
                x_hat = estag(h, x_hist, edge_index_tensor_device)
                recons = x_hat.view(T_out, B, N, 3).permute(1, 0, 2, 3)
            elif model == "simlpe":
                optimizer.zero_grad()
                recons = simlpe(in_3d)
            elif model == "simlpe_dct":
                optimizer.zero_grad()
                recons = simlpe_dct(in_3d)
            elif model == "simlpe_wave":
                optimizer.zero_grad()
                recons = simlpe_wave(in_3d)
            elif model in ("SFgDNet", "SplineEqNet"):
                optimizer.zero_grad()
                dual_model = _get_active_dual_model()
                if dual_model is None:
                    raise RuntimeError(f"{model} model is not initialized.")
                recons = dual_model(in_3d)
            elif model == "pgbig":
                optimizer.zero_grad()
                seq_flat = in_3d.reshape(B, input_n, -1)
                last_pose = seq_flat[:, -1:, :]
                padded = torch.cat([seq_flat, last_pose.repeat(1, output_n, 1)], dim=1)
                stage_outputs = pgbig_model(padded)
                pred_full = stage_outputs[0]
                pred_future = pred_full[:, -output_n:, :].reshape(B, T_out, N, 3)
                recons = pred_future
            elif model == "motionmixer":
                optimizer.zero_grad()
                seq_flat = in_3d.reshape(B, input_n, pose_dim)
                tgt_flat = tgt_3d.reshape(B, T_out, pose_dim)
                if mixer_input_scale != 1.0:
                    seq_scaled = seq_flat / mixer_input_scale
                    tgt_scaled = tgt_flat / mixer_input_scale
                else:
                    seq_scaled = seq_flat
                    tgt_scaled = tgt_flat
                if mixer_use_delta:
                    sequences_all = torch.cat([seq_scaled, tgt_scaled], dim=1)
                    diffs = sequences_all[:, 1:, :] - sequences_all[:, :-1, :]
                    diffs = torch.cat([diffs[:, 0:1, :], diffs], dim=1)
                    mixer_in = diffs[:, :input_n, :]
                    pred_scaled = motion_mixer(mixer_in)
                    pred_scaled = _delta_to_absolute(pred_scaled, seq_scaled[:, -1, :])
                else:
                    mixer_in = seq_scaled
                    pred_scaled = motion_mixer(mixer_in)
                pred_flat = pred_scaled * mixer_input_scale if mixer_input_scale != 1.0 else pred_scaled
                recons = pred_flat.view(B, T_out, N, 3)
            elif model == "trajdep":
                optimizer.zero_grad()
                recons = trajdep(in_3d)
            elif model in ("hisrep", "hisrep_hand", "hisrep_rigidwrist"):
                optimizer.zero_grad()
                aux = None
                if model == "hisrep_hand":
                    recons, aux = hisrep(in_3d, return_aux=True)
                else:
                    recons = hisrep(in_3d)
            elif model == "eqmotion":
                optimizer.zero_grad()
                recons = eqmotion_model(in_3d)
            elif model == "mamba":
                optimizer.zero_grad()
                recons = bas(in_3d, edge_index_tensor_device)
            elif model == "cde":
                optimizer.zero_grad()
                recons = cde(in_3d, edge_index_tensor_device)
            elif model == "hstgcnn":
                optimizer.zero_grad()
                recons = hst(in_3d, edge_index_tensor_device)
            elif model == "gwnet":
                optimizer.zero_grad()
                recons = gwn(in_3d, edge_index_tensor_device)
            elif model == "last":
                init_pose = in_3d[:, -1, :, :]
                recons = init_pose.unsqueeze(1).expand(B, T_out, N, 3)
            else:
                init_pose = in_3d[:, -1, :, :]
                recons = init_pose.unsqueeze(1).expand(B, T_out, N, 3)

            if gate_logging_enabled:
                dual_model = _get_active_dual_model()
                gate_slow_tensor = getattr(dual_model, "last_gate_slow", None) if dual_model is not None else None
                gate_fast_tensor = getattr(dual_model, "last_gate_fast", None) if dual_model is not None else None
                if gate_slow_tensor is not None and gate_fast_tensor is not None:
                    epoch_gate_slow.append(gate_slow_tensor.view(-1).detach().cpu())
                    epoch_gate_fast.append(gate_fast_tensor.view(-1).detach().cpu())
            if gate_param_logging_enabled:
                params_dict = getattr(dual_model, "last_gate_params", None) if dual_model is not None else None
                if isinstance(params_dict, dict):
                    gate_param_snapshot = {
                        key: value.detach().clone() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
                        for key, value in params_dict.items()
                    }

            if mask_logging_enabled:
                mask_tensor = getattr(dual_model, "last_mask", None) if dual_model is not None else None
                if mask_tensor is not None:
                    epoch_masks.append(mask_tensor.view(-1).detach().cpu())

            # Loss: MPJPE + bone length regularization
            # Compute mean over batch and horizon
            mpjpe_terms = torch.norm(recons - tgt_3d, dim=-1)  # (B, T_out, N)
            per_sample_mpjpe = mpjpe_terms.mean(dim=(1, 2))
            per_sample_mpjpe_norm = per_sample_mpjpe * norm_factor_t
            mpjpe_loss = per_sample_mpjpe.mean()
            mpjpe_running += float(per_sample_mpjpe.sum().item())
            mpjpe_norm_running += float(per_sample_mpjpe_norm.sum().item())


            if model == "hisrep_hand":
                gt_dirs, _ = hisrep_handmodule.compute_dirs_lens(tgt_3d, hisrep.parents)
                loss = factorized_loss(
                    pred_xyz=recons,
                    gt_xyz=tgt_3d,
                    pred_dirs=aux["pred_dirs_full"],
                    gt_dirs=gt_dirs,
                    dir_mask=hisrep_handdir_mask,
                    pred_wrists=aux["pred_wrists"],
                    delta_lengths=aux.get("delta_lengths"))
            else:
                # Optional bone-length loss on hand skeleton
                if bone_loss_weight > 0:
                    bl_loss = U.bone_length_loss_edges(recons, tgt_3d, edges_t)
                    loss = mpjpe_loss + float(bone_loss_weight) * bl_loss
                else:
                    loss = mpjpe_loss
            
            if model == "SplineEqNet":
                reg_model = _get_active_dual_model()
                lambda_l1 = config.get("SplineEqNet_lambda_l1")
                lambda_div = config.get("SplineEqNet_lambda_div")
                if negpearson_weight > 0:
                    neg_pearson = U.neg_pearson_loss(recons, tgt_3d)
                    loss = loss + negpearson_weight * neg_pearson
                if reg_model is not None:
                    if lambda_l1 not in (None, 0, 0.0):
                        loss = loss + reg_model.gate_usage_l1(float(lambda_l1))
                    if lambda_div not in (None, 0, 0.0):
                        loss = loss + reg_model.gate_diversity_loss(float(lambda_div))
            
            last_observed_pose = in_3d[:, -1, :, :]
            pred_velocity_mag = _velocity_magnitude(recons, last_observed_pose)
            tgt_velocity_mag = tgt_vel
            vel_mae_per_sample = (pred_velocity_mag - tgt_velocity_mag).abs().mean(dim=(1, 2))
            vel_loss = vel_mae_per_sample.mean()
            if velocity_loss_weight > 0:
                loss = loss + velocity_loss_weight * vel_loss
            vel_running += float(vel_mae_per_sample.sum().item())

            # Abort immediately if loss becomes NaN/Inf during any batch
            if not torch.isfinite(loss):
                print("[NaN detected] Loss is NaN/Inf; aborting training and returning NaN metrics.")
                return {
                    "test_mpjpe": float('nan'),
                    "test_weighted_mae": float('nan'),
                }

            if model in ("mpt", "estag", "simlpe", "simlpe_dct", "simlpe_wave", "SFgDNet", "SplineEqNet", "trajdep", "hisrep", "hisrep_hand", "hisrep_rigidwrist", "eqmotion", "pgbig", "motionmixer", "mamba", "cde", "hstgcnn", "gwnet"):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(config.get("gradient_clip", 5.0)))
                optimizer.step()

            running += float(loss.item()) * inp.size(0)
            # Scoring loss (weighted joint MAE) as in test_prediction.py (not used for backprop)
            with torch.no_grad():
                score_val = U.weighted_joint_loss(joint_weights, recons, tgt_3d, metric='mae')
                score_running += float(score_val.item())
            n_seen += inp.size(0)

        if mask_logging_enabled and epoch_masks:
            epoch_stack = torch.stack(epoch_masks, dim=0)
            mask_epoch_history.append(epoch_stack.mean(dim=0))
        if gate_logging_enabled and epoch_gate_slow and epoch_gate_fast:
            slow_mean = torch.stack(epoch_gate_slow, dim=0)
            fast_mean = torch.stack(epoch_gate_fast, dim=0)
            gate_epoch_history.append(torch.stack([slow_mean, fast_mean], dim=0))

        epoch_loss = running / max(1, n_seen)
        dt = time.time() - t0
        score_avg = score_running / max(1, n_seen)
        mpjpe_avg = mpjpe_running / max(1, n_seen)
        mpjpe_norm_avg = mpjpe_norm_running / max(1, n_seen)
        vel_avg = vel_running / max(1, n_seen)
        vel_clause = f"| vel_mae={vel_avg:.6f} "
        print(
            f"[Epoch {epoch:03d}] loss={epoch_loss:.6f} | mpjpe={mpjpe_avg:.6f} | "
            f"mpjpe_norm={mpjpe_norm_avg:.6f}{vel_clause}"
            f"| score(mae-wj)={score_avg:.6f} | time={dt:.1f}s | lr={optimizer.param_groups[0]['lr'] if optimizer is not None else 'n/a'}"
        )

        if best_train_mpjpe is None or mpjpe_avg < best_train_mpjpe:
            best_train_mpjpe = mpjpe_avg
            best_train_mpjpe_norm = mpjpe_norm_avg
            best_train_epoch = epoch
        if best_train_velocity_mae is None or vel_avg < best_train_velocity_mae:
            best_train_velocity_mae = vel_avg

        if bool(config.get("log_gcn_stats", False)) and model == "mpt":
            layers = U.find_semskeconvs(gen_x, gen_y, gen_z, gen_v)
            U.print_semskeconv_stats(U.semskeconv_stats(layers), prefix=f"[GCN][epoch {epoch:03d}]")

    # -----------------
    # Test/Evaluation
    # -----------------
        avg_loss = None
        avg_loss_norm = None
        avg_score = None
        avg_velocity_mae = None
        total_samples = 0
        test_metrics = _evaluate_loader(test_loader, collect_examples=False, save_examples=save_eval_examples) if test_loader is not None else None
        if test_metrics is not None:
            avg_loss = float(test_metrics["mpjpe"])
            avg_loss_norm = float(test_metrics["mpjpe_norm"])
            avg_score = float(test_metrics["weighted_mae"])
            avg_velocity_mae = float(test_metrics.get("velocity_mae", float("nan")))
            total_samples = int(test_metrics["samples"])
            vel_print = "nan"
            if math.isfinite(avg_velocity_mae):
                vel_print = f"{avg_velocity_mae:.6f}"
            print(
                f"[Test] loss={avg_loss:.6f} | score(mae-wj)={avg_score:.6f} | loss(norm)={avg_loss_norm:.6f} | "
                f"vel_mae={vel_print} | samples={total_samples}"
            )

            if best_test_mpjpe is None or avg_loss < best_test_mpjpe:
                best_test_mpjpe = avg_loss
                best_test_mpjpe_norm = avg_loss_norm
                best_test_epoch = epoch
                best_test_state = _capture_state()
                print(f"[Info] New best test MPJPE {best_test_mpjpe:.6f} at epoch {epoch}")

        if log_wandb and wandb_handle is not None:
            wandb_metrics: Dict[str, float] = {}
            _maybe_add_metric(wandb_metrics, "epoch", float(epoch))
            _maybe_add_metric(wandb_metrics, "train/loss", epoch_loss)
            _maybe_add_metric(wandb_metrics, "train/mpjpe", mpjpe_avg)
            _maybe_add_metric(wandb_metrics, "train/mpjpe_norm", mpjpe_norm_avg)
            _maybe_add_metric(wandb_metrics, "train/weighted_mae", score_avg)
            _maybe_add_metric(wandb_metrics, "train/velocity_mae", vel_avg)
            _maybe_add_metric(wandb_metrics, "train/time_s", dt)
            lr_value = optimizer.param_groups[0]['lr'] if optimizer is not None else None
            _maybe_add_metric(wandb_metrics, "train/lr", lr_value)
            if avg_loss is not None:
                _maybe_add_metric(wandb_metrics, "test/loss", avg_loss)
                _maybe_add_metric(wandb_metrics, "test/loss_norm", avg_loss_norm)
                _maybe_add_metric(wandb_metrics, "test/weighted_mae", avg_score)
                if avg_velocity_mae is not None:
                    _maybe_add_metric(wandb_metrics, "test/velocity_mae", avg_velocity_mae)
                _maybe_add_metric(wandb_metrics, "test/samples", float(total_samples))
            if wandb_metrics:
                wandb_handle.log(wandb_metrics, step=epoch)

        # Step LR scheduler based on validation loss if enabled
        if avg_loss is not None:
            if scheduler is not None:
                # Prefer validation loss when available
                scheduler.step(float(avg_loss))
        else:
            # Step LR scheduler based on training loss if no validation loader
            if scheduler is not None:
                scheduler.step(float(epoch_loss))

    if mask_logging_enabled and mask_epoch_history:
        mask_matrix = torch.stack(mask_epoch_history, dim=0)
        torch.save(mask_matrix, mask_save_path)
    if gate_logging_enabled and gate_epoch_history:
        gate_matrix = torch.stack(gate_epoch_history, dim=0)
        torch.save(gate_matrix, gate_save_path)
    if gate_param_logging_enabled and gate_param_snapshot is not None:
        torch.save(gate_param_snapshot, gate_param_save_path)

    val_metrics_best = None
    if val_loader is not None:
        state_snapshot = _capture_state()
        if best_test_state is not None:
            _load_state(best_test_state)
        val_metrics_best = _evaluate_loader(val_loader, collect_examples=True, restore_train=False, save_examples=save_eval_examples)
        if state_snapshot is not None:
            _load_state(state_snapshot)

    if val_metrics_best is not None:
        val_mpjpe_best = float(val_metrics_best["mpjpe"])
        val_mpjpe_norm_best = float(val_metrics_best["mpjpe_norm"])
        val_weighted_mae_best = float(val_metrics_best["weighted_mae"])
        val_velocity_mae_best = float(val_metrics_best.get("velocity_mae", float("nan")))
        val_samples = int(val_metrics_best["samples"])
        ref_epoch = best_test_epoch if best_test_epoch is not None else "final"
        print(
            f"[Validation @ epoch {ref_epoch}] mpjpe={val_mpjpe_best:.6f} | "
            f"mpjpe_norm={val_mpjpe_norm_best:.6f} | "
            f"score(mae-wj)={val_weighted_mae_best:.6f} | vel_mae={val_velocity_mae_best:.6f} | samples={val_samples}"
        )
    else:
        val_mpjpe_best = None
        val_mpjpe_norm_best = None
        val_weighted_mae_best = None
        val_velocity_mae_best = None
        val_samples = 0

    best_model_path = config.get("best_model_path")
    if best_model_path:
        if best_test_state is not None:
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(
                {
                    "best_model_state": best_test_state,
                    "metadata": {
                        "tag": config.get("best_model_tag"),
                        "model": model,
                        "dataset": config.get("dataset"),
                    },
                },
                best_model_path,
            )
            print(f"Saved best model checkpoint to {best_model_path}")
        else:
            print(f"Best model checkpoint was requested but no best state was captured for {best_model_path}")

    return {
        "train_mpjpe_best": float(best_train_mpjpe) if best_train_mpjpe is not None else None,
        "train_mpjpe_norm_best": float(best_train_mpjpe_norm) if best_train_mpjpe_norm is not None else None,
        "train_best_epoch": int(best_train_epoch) if best_train_epoch is not None else None,
        "train_mpjpe_best": float(best_train_mpjpe) if best_train_mpjpe is not None else None,
        "train_mpjpe_norm_best": float(best_train_mpjpe_norm) if best_train_mpjpe_norm is not None else None,
        "validation_mpjpe_best": val_mpjpe_best if val_loader is not None else None,
        "validation_mpjpe_norm_best": val_mpjpe_norm_best if val_loader is not None else None,
        "test_mpjpe_best": best_test_mpjpe if test_loader is not None else None,
        "test_mpjpe_norm_best": best_test_mpjpe_norm if test_loader is not None else None,
        "params": params_count,
    }
