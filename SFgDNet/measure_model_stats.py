#!/usr/bin/env python3
"""Measure parameter counts, FLOPs, and batch latency for all models."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import platform
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile
from config import DATASET_CHOICES, DEFAULT_MODELS
from data import get_dataset_metadata
from models import hisrep as hisrep_module
from models.baselines import ConstantModel
from models.eqmotion import EqMotionConfig, EqMotionForecaster
from models.HSTGCN import hstgcnn
from models.motionmixer import MlpMixer
from models.mpt import model_4GRU
from models.pgbig.model import PGBIGConfig, PGBIGMultiStageModel
from models.simlpe_dct import SiMLPeDCTConfig, SiMLPeDCTForecaster
from models.SFgDNet import SFgDNetConfig, SFgDNetForecaster
from models.sfgdnet_try import SFgDNetTryConfig, SFgDNetTryForecaster
from models.trajdep import TrajDepConfig, TrajDepForecaster
from models.model import ESTAG
from utils import reconstruct_sequence
BEST_MODELS_PATH = os.path.join("out", "best_runs", "assembly", "best_models.json")
def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "nan"}:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False
def _resolve_value(value: object, cast, default):
    if _is_missing(value):
        return default
    try:
        if cast is bool:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes"}
            if isinstance(value, (int, float)):
                return bool(int(value))
            return bool(value)
        return cast(value)
    except Exception:
        return default
def load_best_configs(path: str, dataset: str) -> Dict[str, Dict[str, object]]:
    try:
        raw = Path(path).read_text()
    except FileNotFoundError:
        return {}
    data = json.loads(raw)
    normalized = {}
    target = dataset.lower()
    for model_name, cfg in data.items():
        if not isinstance(cfg, dict):
            continue
        ds = cfg.get("dataset")
        if ds is None or str(ds).lower() != target:
            continue
        normalized[model_name] = cfg
    return normalized
def _describe_cpu() -> str:
    name = platform.processor()
    if not name:
        name = platform.machine()
    if not name:
        name = platform.platform()
    return name or "unknown CPU"
def _describe_gpu() -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "cuda-device"
def _profile_module(
    module: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    device: torch.device,
    settings: MeasurementSettings,
) -> Dict[str, object]:
    module.eval()
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(device)
    warmups = 3
    repeats = 10
    with torch.inference_mode():
        for _ in range(warmups):
            module(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeats):
            module(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - start
    latency = duration / repeats if repeats > 0 else float("inf")
    throughput = (settings.input_n + settings.output_n) / latency if latency > 0 else float("inf")
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    with torch.no_grad():
        with profile(activities=activities, with_flops=True) as prof:
            with torch.inference_mode():
                module(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_flops = sum(int(evt.flops) for evt in prof.key_averages())
    return {
        "latency_ms": latency * 1000.0,
        "throughput_fps": throughput,
        "memory_bytes": peak_memory,
        "flops": total_flops,
    }
def measure_model_on_device(
    spec: ModelSpec,
    settings: MeasurementSettings,
    best_cfg: Optional[Dict[str, object]],
    device: torch.device,
) -> Tuple[Dict[str, object], MeasurementSettings]:
    effective_settings = settings.with_best(best_cfg)
    module = spec.builder(effective_settings).to(device)
    bundle = create_input_bundle(effective_settings, device)
    inputs = spec.build_inputs(bundle, effective_settings)
    metrics = _profile_module(module, inputs, device, effective_settings)
    return metrics, effective_settings
@dataclass
class MeasurementSettings:
    dataset: str
    batch_size: int
    input_n: int
    output_n: int
    hidden_size: int
    gru_layers: int
    t_layers: int
    gnn_layers: int
    dct_keep: int
    use_space: bool
    node_count: int
    adjacency: Tuple[Tuple[int, ...], ...]
    edge_index: Tuple[Tuple[int, int], ...]
    mixer_blocks: int
    mixer_tokens_dim: Optional[int]
    mixer_channels_dim: Optional[int]
    mixer_regularization: float
    mixer_use_se: bool
    mixer_activation: str
    mixer_block_type: str
    mixer_reduction: int
    mixer_use_max_pooling: bool
    mixer_use_delta: bool
    mixer_input_scale: float
    simlpe_use_norm: bool
    simlpe_spatial_fc_only: bool
    simlpe_mix_spatial_temporal: bool
    simlpe_norm_axis: str
    simlpe_add_last_offset: bool
    SFgDNet_force_mlp_padding: bool
    SFgDNet_mask_temperature: float
    SFgDNet_mid_gate_max: float
    SFgDNet_slow_layers: Optional[int]
    SFgDNet_fast_layers: Optional[int]
    SFgDNet_fixed_lambda: Optional[float]
    SFgDNet_fixed_centers: Optional[Tuple[float, float]]
    sfgdnet_try_force_mlp_padding: bool
    sfgdnet_try_mask_temperature: float
    sfgdnet_try_mid_gate_max: float
    sfgdnet_try_slow_layers: Optional[int]
    sfgdnet_try_fast_layers: Optional[int]
    sfgdnet_try_fixed_lambda: Optional[float]
    sfgdnet_try_fixed_centers: Optional[Tuple[float, float]]
    def with_best(self, best_cfg: Optional[Dict[str, object]]) -> "MeasurementSettings":
        if not best_cfg:
            return self
        overrides: Dict[str, object] = {}
        overrides["batch_size"] = _resolve_value(best_cfg.get("batch_size"), int, self.batch_size)
        overrides["input_n"] = _resolve_value(best_cfg.get("input_n"), int, self.input_n)
        overrides["output_n"] = _resolve_value(best_cfg.get("output_n"), int, self.output_n)
        overrides["hidden_size"] = _resolve_value(best_cfg.get("hidden_size"), int, self.hidden_size)
        overrides["gru_layers"] = _resolve_value(best_cfg.get("gru_layers"), int, self.gru_layers)
        overrides["t_layers"] = _resolve_value(best_cfg.get("t_layers"), int, self.t_layers)
        overrides["gnn_layers"] = _resolve_value(best_cfg.get("gnn_layers"), int, self.gnn_layers)
        overrides["dct_keep"] = _resolve_value(best_cfg.get("dct_keep_coeffs"), int, self.dct_keep)
        overrides["use_space"] = _resolve_value(best_cfg.get("use_space"), bool, self.use_space)
        overrides["simlpe_use_norm"] = _resolve_value(best_cfg.get("simlpe_use_norm"), bool, self.simlpe_use_norm)
        overrides["simlpe_spatial_fc_only"] = _resolve_value(
            best_cfg.get("simlpe_use_spatial_fc_only"), bool, self.simlpe_spatial_fc_only
        )
        overrides["simlpe_mix_spatial_temporal"] = _resolve_value(
            best_cfg.get("simlpe_mix_spatial_temporal"), bool, self.simlpe_mix_spatial_temporal
        )
        overrides["simlpe_norm_axis"] = _resolve_value(
            best_cfg.get("simlpe_norm_axis"), str, self.simlpe_norm_axis
        )
        overrides["simlpe_add_last_offset"] = _resolve_value(
            best_cfg.get("simlpe_add_last_offset"), bool, self.simlpe_add_last_offset
        )
        overrides["SFgDNet_mask_temperature"] = _resolve_value(
            best_cfg.get("simlpe_mask_temperature"), float, self.SFgDNet_mask_temperature
        )
        overrides["SFgDNet_slow_layers"] = _resolve_value(
            best_cfg.get("simlpe_slow_layers"), int, self.SFgDNet_slow_layers
        )
        overrides["SFgDNet_fast_layers"] = _resolve_value(
            best_cfg.get("simlpe_fast_layers"), int, self.SFgDNet_fast_layers
        )
        overrides["SFgDNet_mid_gate_max"] = _resolve_value(
            best_cfg.get("simlpe_mid_gate_max"), float, self.SFgDNet_mid_gate_max
        )
        overrides["sfgdnet_try_mask_temperature"] = _resolve_value(
            best_cfg.get("simlpe_mask_temperature"), float, self.sfgdnet_try_mask_temperature
        )
        overrides["sfgdnet_try_slow_layers"] = _resolve_value(
            best_cfg.get("simlpe_slow_layers"), int, self.sfgdnet_try_slow_layers
        )
        overrides["sfgdnet_try_fast_layers"] = _resolve_value(
            best_cfg.get("simlpe_fast_layers"), int, self.sfgdnet_try_fast_layers
        )
        overrides["sfgdnet_try_mid_gate_max"] = _resolve_value(
            best_cfg.get("simlpe_mid_gate_max"), float, self.sfgdnet_try_mid_gate_max
        )
        return replace(self, **overrides)
@dataclass
class ModelInputBundle:
    history: torch.Tensor
    future: torch.Tensor
    adjacency: torch.Tensor
    edge_index: torch.Tensor
    node_count: int
    pose_dim: int
@dataclass
class ModelSpec:
    name: str
    builder: Callable[[MeasurementSettings], nn.Module]
    build_inputs: Callable[[ModelInputBundle, MeasurementSettings], Tuple[torch.Tensor, ...]]
def _build_mpt_generators(
    input_size: int,
    hidden_size: int,
    output_n: int,
    node_num: int,
    batch_size: int,
    device: torch.device,
    *,
    adjacency: torch.Tensor,
    edge_index: Sequence[Tuple[int, int]],
    space: bool = True,
    gru_layers: int = 4,
) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    graph_kwargs = {"adjacency": adjacency, "edge_index": edge_index}
    gen_x = model_4GRU.Generator(input_size, hidden_size, output_n, node_num, batch_size, space, num_layers=gru_layers, **graph_kwargs).to(device)
    gen_y = model_4GRU.Generator(input_size, hidden_size, output_n, node_num, batch_size, space, num_layers=gru_layers, **graph_kwargs).to(device)
    gen_z = model_4GRU.Generator(input_size, hidden_size, output_n, node_num, batch_size, space, num_layers=gru_layers, **graph_kwargs).to(device)
    gen_v = model_4GRU.Generator(input_size, hidden_size, output_n, node_num, batch_size, space, num_layers=gru_layers, **graph_kwargs).to(device)
    return gen_x, gen_y, gen_z, gen_v
class MPTWrapper(nn.Module):
    def __init__(
        self,
        gen_x: nn.Module,
        gen_y: nn.Module,
        gen_z: nn.Module,
        gen_v: nn.Module,
        settings: MeasurementSettings,
    ) -> None:
        super().__init__()
        self.gen_x = gen_x
        self.gen_y = gen_y
        self.gen_z = gen_z
        self.gen_v = gen_v
        self.node_num = settings.node_count
        self.output_n = settings.output_n
        self.hidden_size = settings.hidden_size
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 4 or history.shape[2] != self.node_num or history.shape[3] != 3:
            raise ValueError("History must be (B, T, N, 3)")
        B, T, N, _ = history.shape
        vel = history[:, 1:, :, :] - history[:, :-1, :, :]
        dist = torch.norm(vel, dim=-1, keepdim=True)
        angle = vel / (dist + 1e-8)
        in_vel = dist.squeeze(-1).permute(0, 2, 1).contiguous()
        in_ang_x = angle[..., 0].permute(0, 2, 1).contiguous()
        in_ang_y = angle[..., 1].permute(0, 2, 1).contiguous()
        in_ang_z = angle[..., 2].permute(0, 2, 1).contiguous()
        def _run(gen: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
            out, _ = gen(tensor, self.hidden_size)
            return out.squeeze(0).view(B, self.node_num, self.output_n)
        pred_v = _run(self.gen_v, in_vel).permute(0, 2, 1)
        ang_x = _run(self.gen_x, in_ang_x).permute(0, 2, 1)
        ang_y = _run(self.gen_y, in_ang_y).permute(0, 2, 1)
        ang_z = _run(self.gen_z, in_ang_z).permute(0, 2, 1)
        pred_angles = torch.stack([ang_x, ang_y, ang_z], dim=-1)
        init_pose = history[:, -1, :, :]
        return reconstruct_sequence(pred_v, pred_angles, init_pose, self.node_num)
class MotionMixerWrapper(nn.Module):
    def __init__(self, motion_mixer: nn.Module, settings: MeasurementSettings) -> None:
        super().__init__()
        self.motion_mixer = motion_mixer
        self.input_n = settings.input_n
        self.output_n = settings.output_n
        self.pose_dim = settings.node_count * 3
        self.mixer_input_scale = settings.mixer_input_scale
        self.mixer_use_delta = settings.mixer_use_delta
    def forward(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        B = history.shape[0]
        seq_flat = history.reshape(B, self.input_n, self.pose_dim)
        tgt_flat = future.reshape(B, self.output_n, self.pose_dim)
        if self.mixer_input_scale != 1.0:
            seq_scaled = seq_flat / self.mixer_input_scale
            tgt_scaled = tgt_flat / self.mixer_input_scale
        else:
            seq_scaled = seq_flat
            tgt_scaled = tgt_flat
        if self.mixer_use_delta:
            sequences_all = torch.cat([seq_scaled, tgt_scaled], dim=1)
            diffs = sequences_all[:, 1:, :] - sequences_all[:, :-1, :]
            diffs = torch.cat([diffs[:, 0:1, :], diffs], dim=1)
            mixer_in = diffs[:, : self.input_n, :]
            pred_scaled = self.motion_mixer(mixer_in)
            pred_scaled = delta_to_absolute(pred_scaled, seq_scaled[:, -1, :])
        else:
            mixer_in = seq_scaled
            pred_scaled = self.motion_mixer(mixer_in)
        pred_flat = pred_scaled * self.mixer_input_scale if self.mixer_input_scale != 1.0 else pred_scaled
        return pred_flat.view(B, self.output_n, -1, 3)
class PGBIGWrapper(nn.Module):
    def __init__(self, pgbig: nn.Module, settings: MeasurementSettings) -> None:
        super().__init__()
        self.pgbig = pgbig
        self.input_n = settings.input_n
        self.output_n = settings.output_n
        self.pose_dim = settings.node_count * 3
        self.node_count = settings.node_count
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        B = history.shape[0]
        seq_flat = history.reshape(B, self.input_n, self.pose_dim)
        last_pose = seq_flat[:, -1:, :]
        padded = torch.cat([seq_flat, last_pose.repeat(1, self.output_n, 1)], dim=1)
        pred_full = self.pgbig(padded)[0]
        tail = pred_full[:, -self.output_n :, :]
        return tail.view(B, self.output_n, self.node_count, 3)
class EstagWrapper(nn.Module):
    def __init__(self, estag: nn.Module, settings: MeasurementSettings) -> None:
        super().__init__()
        self.estag = estag
        self.input_n = settings.input_n
        self.output_n = settings.output_n
        self.node_count = settings.node_count
        self.edge_index = torch.tensor(settings.edge_index, dtype=torch.long).t().contiguous()
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        B = history.shape[0]
        current_device = history.device
        self.estag.device = str(current_device)
        x_hist = history.permute(1, 0, 2, 3).reshape(self.input_n, B * self.node_count, 3)
        h = torch.ones(B * self.node_count, 1, device=history.device)
        edge_index = self.edge_index.to(history.device)
        x_hat = self.estag(h, x_hist, edge_index)
        return x_hat.view(self.output_n, B, self.node_count, 3).permute(1, 0, 2, 3)
def delta_to_absolute(pred: torch.Tensor, last_frame: torch.Tensor) -> torch.Tensor:
    pred = pred.clone()
    pred[:, 0, :] = pred[:, 0, :] + last_frame
    return torch.cumsum(pred, dim=1)
def create_input_bundle(settings: MeasurementSettings, device: torch.device) -> ModelInputBundle:
    history = torch.randn(settings.batch_size, settings.input_n, settings.node_count, 3, device=device)
    future = torch.randn(settings.batch_size, settings.output_n, settings.node_count, 3, device=device)
    adjacency = torch.tensor(settings.adjacency, dtype=torch.float32, device=device)
    edge_index = torch.tensor(settings.edge_index, dtype=torch.long, device=device).t().contiguous()
    return ModelInputBundle(
        history=history,
        future=future,
        adjacency=adjacency,
        edge_index=edge_index,
        node_count=settings.node_count,
        pose_dim=settings.node_count * 3,
    )
def build_hisrep(settings: MeasurementSettings) -> nn.Module:
    kwargs = {
        "num_nodes": settings.node_count,
        "input_length": settings.input_n,
        "pred_length": settings.output_n,
        "hidden_dim": settings.hidden_size,
        "num_stages": settings.gru_layers,
        "dct_n": settings.dct_keep,
    }
    cfg = hisrep_module.HisRepConfig(**kwargs)
    return hisrep_module.HisRepForecaster(cfg)
def build_specs() -> Dict[str, ModelSpec]:
    return {
        "mpt": ModelSpec(
            name="mpt",
            builder=lambda settings: MPTWrapper(
                *_build_mpt_generators(
                    settings.input_n - 1,
                    settings.hidden_size,
                    settings.output_n,
                    settings.node_count,
                    settings.batch_size,
                    torch.device("cpu"),
                    adjacency=torch.tensor(settings.adjacency, dtype=torch.float32),
                    edge_index=settings.edge_index,
                    space=settings.use_space,
                    gru_layers=settings.gru_layers,
                ),
                settings,
            ),
            build_inputs=lambda bundle, _: (bundle.history,),
        ),
        "hisrep": ModelSpec(
            name="hisrep",
            builder=lambda settings: build_hisrep(settings),
            build_inputs=lambda bundle, _: (bundle.history,),
        ),
        "trajdep": ModelSpec(
            name="trajdep",
            builder=lambda settings: TrajDepForecaster(
                TrajDepConfig(
                    input_length=settings.input_n,
                    pred_length=settings.output_n,
                    num_nodes=settings.node_count,
                    hidden_dim=settings.hidden_size,
                    num_stages=max(1, settings.gnn_layers),
                    dropout=0.5,
                    dct_components=min(settings.input_n + settings.output_n, 35),
                )
            ),
            build_inputs=lambda bundle, _: (bundle.history,),
        ),
        "pgbig": ModelSpec(
            name="pgbig",
            builder=lambda settings: PGBIGWrapper(
                PGBIGMultiStageModel(
                    PGBIGConfig(
                        input_n=settings.input_n,
                        output_n=settings.output_n,
                        in_features=settings.node_count * 3,
                        d_model=settings.hidden_size,
                        dct_n=min(settings.dct_keep, settings.input_n + settings.output_n),
                        drop_out=0.3,
                        encoder_layers=1,
                        decoder_layers=2,
                    )
                ),
                settings,
            ),
            build_inputs=lambda bundle, _: (bundle.history,),
        ),
        "motionmixer": ModelSpec(
            name="motionmixer",
            builder=lambda settings: MotionMixerWrapper(
                MlpMixer(
                    num_classes=settings.node_count * 3,
                    num_blocks=settings.mixer_blocks,
                    hidden_dim=settings.hidden_size,
                    tokens_mlp_dim=settings.mixer_tokens_dim or settings.hidden_size,
                    channels_mlp_dim=settings.mixer_channels_dim or settings.hidden_size,
                    seq_len=settings.input_n,
                    pred_len=settings.output_n,
                    activation=settings.mixer_activation,
                    mlp_block_type=settings.mixer_block_type,
                    regularization=settings.mixer_regularization,
                    input_size=settings.node_count * 3,
                    reduction=settings.mixer_reduction,
                    use_max_pooling=settings.mixer_use_max_pooling,
                    use_se=settings.mixer_use_se,
                ),
                settings,
            ),
            build_inputs=lambda bundle, settings: (bundle.history, bundle.future),
        ),
        "eqmotion": ModelSpec(
            name="eqmotion",
            builder=lambda settings: EqMotionForecaster(
                EqMotionConfig(
                    input_length=settings.input_n,
                    pred_length=settings.output_n,
                    num_nodes=settings.node_count,
                    hidden_dim=settings.hidden_size,
                    hid_channel=settings.hidden_size,
                )
            ),
            build_inputs=lambda bundle, _: (bundle.history,),
        ),
        "estag": ModelSpec(
            name="estag",
            builder=lambda settings: EstagWrapper(
                ESTAG(
                    num_past=settings.input_n,
                    num_future=settings.output_n,
                    in_node_nf=1,
                    hidden_nf=settings.hidden_size,
                    fft=True,
                    eat=True,
                    device="cpu",
                    n_layers=settings.t_layers,
                    n_nodes=settings.node_count,
                    nodes_att_dim=0,
                    coords_weight=1.0,
                    tempo=True,
                    filter=True,
                ),
                settings,
            ),
            build_inputs=lambda bundle, _: (bundle.history,),
        ),
        "simlpe_dct": ModelSpec(
            name="simlpe_dct",
            builder=lambda settings: SiMLPeDCTForecaster(
                SiMLPeDCTConfig(
                    input_length=settings.input_n,
                    pred_length=settings.output_n,
                    num_nodes=settings.node_count,
                    hidden_dim=settings.hidden_size,
                    num_layers=settings.gru_layers,
                    use_norm=settings.simlpe_use_norm,
                    use_spatial_fc_only=settings.simlpe_spatial_fc_only,
                    mix_spatial_temporal=settings.simlpe_mix_spatial_temporal,
                    norm_axis=settings.simlpe_norm_axis,
                    add_last_offset=settings.simlpe_add_last_offset,
                    dct_components=settings.dct_keep,
                )
            ),
            build_inputs=lambda bundle, _: (bundle.history,),
        ),
        "SFgDNet": ModelSpec(
            name="SFgDNet",
            builder=lambda settings: SFgDNetForecaster(
                SFgDNetConfig(
                    input_length=settings.input_n,
                    pred_length=settings.output_n,
                    num_nodes=settings.node_count,
                    hidden_dim=settings.hidden_size,
                    use_norm=settings.simlpe_use_norm,
                    use_spatial_fc_only=settings.simlpe_spatial_fc_only,
                    mix_spatial_temporal=settings.simlpe_mix_spatial_temporal,
                    norm_axis=settings.simlpe_norm_axis,
                    add_last_offset=settings.simlpe_add_last_offset,
                    dct_components=settings.dct_keep,
                    mask_temperature=settings.SFgDNet_mask_temperature,
                    slow_num_layers=settings.SFgDNet_slow_layers,
                    fast_num_layers=settings.SFgDNet_fast_layers,
                    mid_gate_max=settings.SFgDNet_mid_gate_max,
                    force_mlp_padding=settings.SFgDNet_force_mlp_padding,
                    learn_lambda=True,
                    learn_centers=True,
                    fixed_lambda=settings.SFgDNet_fixed_lambda,
                    fixed_centers=settings.SFgDNet_fixed_centers,
                )
            ),
            build_inputs=lambda bundle, _: (bundle.history,),
        ),
        "sfgdnet_try": ModelSpec(
            name="sfgdnet_try",
            builder=lambda settings: SFgDNetTryForecaster(
                SFgDNetTryConfig(
                    input_length=settings.input_n,
                    pred_length=settings.output_n,
                    num_nodes=settings.node_count,
                    hidden_dim=settings.hidden_size,
                    use_norm=settings.simlpe_use_norm,
                    use_spatial_fc_only=settings.simlpe_spatial_fc_only,
                    mix_spatial_temporal=settings.simlpe_mix_spatial_temporal,
                    norm_axis=settings.simlpe_norm_axis,
                    add_last_offset=settings.simlpe_add_last_offset,
                    dct_components=settings.dct_keep,
                    mask_temperature=settings.sfgdnet_try_mask_temperature,
                    slow_num_layers=settings.sfgdnet_try_slow_layers,
                    fast_num_layers=settings.sfgdnet_try_fast_layers,
                    mid_gate_max=settings.sfgdnet_try_mid_gate_max,
                    force_mlp_padding=settings.sfgdnet_try_force_mlp_padding,
                    learn_lambda=True,
                    learn_centers=True,
                    fixed_lambda=settings.sfgdnet_try_fixed_lambda,
                    fixed_centers=settings.sfgdnet_try_fixed_centers,
                )
            ),
            build_inputs=lambda bundle, _: (bundle.history,),
        ),
        "hstgcnn": ModelSpec(
            name="hstgcnn",
            builder=lambda settings: hstgcnn(
                input_feat=3,
                output_feat=3,
                seq_len=settings.input_n,
                pred_seq_len=settings.output_n,
            ),
            build_inputs=lambda bundle, _: (bundle.history, bundle.edge_index),
        ),
        "last": ModelSpec(
            name="last",
            builder=lambda settings: ConstantModel(settings.output_n),
            build_inputs=lambda bundle, _: (bundle.history, bundle.edge_index),
        ),
    }
def parse_fixed_centers(value: Optional[str]) -> Optional[Tuple[float, float]]:
    if not value:
        return None
    tokens = [tok.strip() for tok in value.replace("|", ",").split(",") if tok.strip()]
    if len(tokens) < 2:
        return None
    return (float(tokens[0]), float(tokens[1]))
def main() -> None:
    parser = argparse.ArgumentParser(description="Measure parameters, FLOPs, and per-batch time for all models.")
    parser.add_argument("--dataset", type=str, default="assembly", choices=DATASET_CHOICES)
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS, help="Comma-separated list of models to profile.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-n", type=int, default=70)
    parser.add_argument("--output-n", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--gru-layers", type=int, default=4)
    parser.add_argument("--t-layers", type=int, default=2)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--dct-keep", type=int, default=None)
    parser.add_argument("--use-space", action="store_true", default=True)
    parser.add_argument("--no-space", dest="use_space", action="store_false")
    parser.add_argument("--mixer-blocks", type=int, default=4)
    parser.add_argument("--mixer-tokens-dim", type=int, default=None)
    parser.add_argument("--mixer-channels-dim", type=int, default=None)
    parser.add_argument("--mixer-regularization", type=float, default=0.0)
    parser.add_argument("--mixer-use-se", dest="mixer_use_se", action="store_true", default=True)
    parser.add_argument("--no-mixer-use-se", dest="mixer_use_se", action="store_false")
    parser.add_argument("--mixer-activation", type=str, default="gelu")
    parser.add_argument("--mixer-block-type", type=str, default="normal")
    parser.add_argument("--mixer-reduction", type=int, default=4)
    parser.add_argument("--mixer-use-max-pooling", action="store_true", default=False)
    parser.add_argument("--mixer-use-delta", action="store_true", dest="mixer_use_delta", default=True)
    parser.add_argument("--no-mixer-use-delta", action="store_false", dest="mixer_use_delta")
    parser.add_argument("--mixer-input-scale", type=float, default=1.0)
    parser.add_argument("--SFgDNet-force-mlp-padding", action="store_true", default=True)
    parser.add_argument("--SFgDNet-mask-temperature", type=float, default=1.0)
    parser.add_argument("--SFgDNet-mid-gate-max", type=float, default=0.9)
    parser.add_argument("--SFgDNet-slow-layers", type=int, default=None)
    parser.add_argument("--SFgDNet-fast-layers", type=int, default=None)
    parser.add_argument("--SFgDNet-fixed-lambda", type=float, default=None)
    parser.add_argument("--SFgDNet-fixed-centers", type=str, default=None)
    parser.add_argument("--sfgdnet-try-force-mlp-padding", action="store_true", default=True)
    parser.add_argument("--sfgdnet-try-mask-temperature", type=float, default=1.0)
    parser.add_argument("--sfgdnet-try-mid-gate-max", type=float, default=0.9)
    parser.add_argument("--sfgdnet-try-slow-layers", type=int, default=None)
    parser.add_argument("--sfgdnet-try-fast-layers", type=int, default=None)
    parser.add_argument("--sfgdnet-try-fixed-lambda", type=float, default=None)
    parser.add_argument("--sfgdnet-try-fixed-centers", type=str, default=None)
    parser.add_argument("--csv", type=str, default="model_stats.csv")
    parser.add_argument("--best-models", type=str, default=BEST_MODELS_PATH)
    parser.add_argument("--device", type=str, choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    metadata = get_dataset_metadata(args.dataset)
    adjacency = metadata["adjacency"]
    edge_index = metadata["edge_index"]
    node_count = metadata["node_count"]
    dct_keep = args.dct_keep or (args.input_n + args.output_n)
    fixed_centers = parse_fixed_centers(args.SFgDNet_fixed_centers)
    sfgdnet_try_fixed_centers = parse_fixed_centers(args.sfgdnet_try_fixed_centers)
    settings = MeasurementSettings(
        dataset=args.dataset,
        batch_size=args.batch_size,
        input_n=args.input_n,
        output_n=args.output_n,
        hidden_size=args.hidden_size,
        gru_layers=args.gru_layers,
        t_layers=args.t_layers,
        gnn_layers=args.gnn_layers,
        dct_keep=dct_keep,
        use_space=args.use_space,
        node_count=node_count,
        adjacency=adjacency,
        edge_index=edge_index,
        mixer_blocks=args.mixer_blocks,
        mixer_tokens_dim=args.mixer_tokens_dim,
        mixer_channels_dim=args.mixer_channels_dim,
        mixer_regularization=args.mixer_regularization,
        mixer_use_se=args.mixer_use_se,
        mixer_activation=args.mixer_activation,
        mixer_block_type=args.mixer_block_type,
        mixer_reduction=args.mixer_reduction,
        mixer_use_max_pooling=args.mixer_use_max_pooling,
        mixer_use_delta=args.mixer_use_delta,
        mixer_input_scale=args.mixer_input_scale,
        simlpe_use_norm=True,
        simlpe_spatial_fc_only=False,
        simlpe_mix_spatial_temporal=False,
        simlpe_norm_axis="spatial",
        simlpe_add_last_offset=True,
        SFgDNet_force_mlp_padding=args.SFgDNet_force_mlp_padding,
        SFgDNet_mask_temperature=args.SFgDNet_mask_temperature,
        SFgDNet_mid_gate_max=args.SFgDNet_mid_gate_max,
        SFgDNet_slow_layers=args.SFgDNet_slow_layers,
        SFgDNet_fast_layers=args.SFgDNet_fast_layers,
        SFgDNet_fixed_lambda=args.SFgDNet_fixed_lambda,
        SFgDNet_fixed_centers=fixed_centers,
        sfgdnet_try_force_mlp_padding=args.sfgdnet_try_force_mlp_padding,
        sfgdnet_try_mask_temperature=args.sfgdnet_try_mask_temperature,
        sfgdnet_try_mid_gate_max=args.sfgdnet_try_mid_gate_max,
        sfgdnet_try_slow_layers=args.sfgdnet_try_slow_layers,
        sfgdnet_try_fast_layers=args.sfgdnet_try_fast_layers,
        sfgdnet_try_fixed_lambda=args.sfgdnet_try_fixed_lambda,
        sfgdnet_try_fixed_centers=sfgdnet_try_fixed_centers,
    )
    specs = build_specs()
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    selected = [name for name in model_names if name in specs]
    if not selected:
        raise SystemExit("No known models selected.")
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
    best_configs = load_best_configs(args.best_models, args.dataset)
    cpu_hw = _describe_cpu()
    gpu_hw = _describe_gpu()
    devices = []
    if args.device in {"cpu", "both"}:
        devices.append(torch.device("cpu"))
    if args.device in {"gpu", "both"}:
        if gpu_hw is not None:
            devices.append(torch.device("cuda"))
        else:
            print("GPU unavailable; skipping GPU metrics.")
    if not devices:
        raise SystemExit("No devices selected for profiling.")
    results = []
    for model_name in selected:
        spec = specs[model_name]
        best_cfg = best_configs.get(model_name)
        for device in devices:
            metrics, eff_settings = measure_model_on_device(
                spec, settings, best_cfg, device
            )
            hw_desc = gpu_hw if device.type == "cuda" else cpu_hw
            results.append(
                {
                    "model": model_name,
                    "device": device.type,
                    "hardware": hw_desc,
                    "batch_size": eff_settings.batch_size,
                    "input_n": eff_settings.input_n,
                    "output_n": eff_settings.output_n,
                    "latency_ms": metrics["latency_ms"],
                    "throughput_fps": metrics["throughput_fps"],
                    "memory_bytes": metrics["memory_bytes"],
                    "flops": metrics["flops"],
                }
            )
            print(
                f"{model_name} [{device.type}] ({hw_desc}) "
                f"latency={metrics['latency_ms']:.2f}ms throughput={metrics['throughput_fps']:.2f}fps "
                f"memory={metrics['memory_bytes']} flops={metrics['flops']:,}"
            )
    with open(args.csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "model",
                "device",
                "hardware",
                "batch_size",
                "input_n",
                "output_n",
                "latency_ms",
                "throughput_fps",
                "memory_bytes",
                "flops",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "model": row["model"],
                    "device": row["device"],
                    "hardware": row["hardware"],
                    "batch_size": row["batch_size"],
                    "input_n": row["input_n"],
                    "output_n": row["output_n"],
                    "latency_ms": f"{row['latency_ms']:.4f}",
                    "throughput_fps": f"{row['throughput_fps']:.4f}",
                    "memory_bytes": row["memory_bytes"],
                    "flops": row["flops"],
                }
            )
if __name__ == "__main__":
    main()
