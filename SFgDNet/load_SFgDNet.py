from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader

from config import DatasetCfg, MODEL_DEFAULT_CONFIGS
from data import build_datasets, get_dataset_metadata
from models.sfgdnet_try import SFgDNetTryConfig, SFgDNetTryForecaster # <- MoE forecaster


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "examples" / "models"
MODEL_FILE_TEMPLATE = "{model}_{dataset}_{action}.pt"
DEFAULT_MODEL_NAME = "sfgdnet_try"


@dataclass(frozen=True)
class ModelComponents:
    model_key: str
    config_cls: type
    forecaster_cls: type


SUPPORTED_MODELS: dict[str, ModelComponents] = {
    "sfgdnet_try": ModelComponents(
        model_key="sfgdnet_try",
        config_cls=SFgDNetTryConfig,
        forecaster_cls=SFgDNetTryForecaster,
    ),
}


def _model_defaults(model_key: str) -> dict[str, object]:
    defaults = MODEL_DEFAULT_CONFIGS.get(model_key)
    if defaults is None:
        raise ValueError(
            f"No MODEL_DEFAULT_CONFIGS entry found for '{model_key}'. "
            "Add the configuration to config.py to enable inspection."
        )
    return defaults


def parse_tag(tag: str | None) -> dict[str, str | None]:
    if not tag:
        return {}
    parsed: dict[str, str | None] = {}
    for part in tag.split("__"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        value = value.strip()
        if not value or value.lower() in {"none", "null"}:
            parsed[key] = None
        else:
            parsed[key] = value
    return parsed


def coerce_int(value: Any, default: int) -> int:
    if isinstance(value, int):
        return value
    if value is None:
        return default
    val = str(value).strip()
    return default if not val or val.lower() in {"none", "null"} else int(val)


def coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    val = str(value).strip()
    if not val or val.lower() in {"none", "null"}:
        return None
    return int(val)


def coerce_float(value: Any, default: float) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if value is None:
        return default
    val = str(value).strip()
    return default if not val or val.lower() in {"none", "null"} else float(val)


def find_available_actions(model_dir: Path, model: str, dataset: str) -> Iterable[str]:
    prefix = f"{model}_{dataset}_"
    for candidate in sorted(model_dir.glob(f"{prefix}*.pt")):
        stem = candidate.stem
        if stem.startswith(prefix):
            yield stem[len(prefix):]


def resolve_checkpoint_path(model_dir: Path, model: str, dataset: str, action: str) -> Path:
    path = model_dir / MODEL_FILE_TEMPLATE.format(model=model, dataset=dataset, action=action)
    if path.exists():
        return path
    available = ", ".join(find_available_actions(model_dir, model, dataset))
    raise FileNotFoundError(
        f"Checkpoint for action '{action}' not found in {model_dir}. "
        f"Available actions: {available or '<none>'}"
    )


def _resolve_data_dir(dataset: str, override: Path | None) -> Path:
    if override is not None:
        data_dir = override
    else:
        metadata = get_dataset_metadata(dataset)
        default_dir = metadata.get("default_dir")
        if not default_dir:
            raise ValueError(f"No default data directory configured for dataset '{dataset}'.")
        data_dir = Path(default_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory '{data_dir}' does not exist.")
    return data_dir


def _parse_preprocessing_params(parsed_tag: dict[str, str | None]) -> tuple[int, int | None, int]:
    stride = coerce_int(parsed_tag.get("st"), 1)
    time_interp = coerce_optional_int(parsed_tag.get("ti"))
    seed = coerce_int(parsed_tag.get("seed"), 0)
    return stride, time_interp, seed


def _build_dataset_cfg(
    dataset: str,
    action: str,
    *,
    input_length: int,
    pred_length: int,
    num_nodes: int,
    stride: int,
    time_interp: int | None,
    data_seed: int,
    data_dir: Path | None,
) -> DatasetCfg:
    metadata = get_dataset_metadata(dataset)
    resolved_dir = _resolve_data_dir(dataset, data_dir)
    wrist_indices = tuple(metadata.get("default_wrist_indices", ()))
    edge_index = tuple(metadata.get("edge_index", ()))
    adjacency = tuple(metadata.get("adjacency", ()))
    return DatasetCfg(
        data_dir=str(resolved_dir),
        action_filter=action,
        input_n=input_length,
        output_n=pred_length,
        stride=stride,
        time_interp=time_interp,
        window_norm=input_length,
        batch_size=1,
        seed=data_seed,
        wrist_indices=wrist_indices,
        dataset=dataset,
        node_count=num_nodes,
        edge_index=edge_index,
        adjacency=adjacency,
    )


def _build_history_loader(
    ds_cfg: DatasetCfg,
    *,
    loader_batch_size: int,
    data_seed: int,
) -> DataLoader:
    train_dataset, _, _ = build_datasets(ds_cfg)
    generator = torch.Generator()
    generator.manual_seed(int(data_seed))
    batch_size = max(1, loader_batch_size)
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )


def load_normalized_history_samples(
    dataset: str,
    action: str,
    *,
    input_length: int,
    pred_length: int,
    num_nodes: int,
    num_samples: int,
    stride: int,
    time_interp: int | None,
    data_seed: int,
    data_dir: Path | None = None,
    loader_batch_size: int = 1,
) -> torch.Tensor:
    ds_cfg = _build_dataset_cfg(
        dataset,
        action,
        input_length=input_length,
        pred_length=pred_length,
        num_nodes=num_nodes,
        stride=stride,
        time_interp=time_interp,
        data_seed=data_seed,
        data_dir=data_dir,
    )
    loader = _build_history_loader(
        ds_cfg,
        loader_batch_size=min(max(1, loader_batch_size), max(1, num_samples)),
        data_seed=data_seed,
    )
    histories: list[torch.Tensor] = []
    collected = 0
    for batch in loader:
        inputs = batch[0].to(dtype=torch.float32)
        histories.append(inputs[:, :, :, 4:])
        collected += inputs.size(0)
        if collected >= num_samples:
            break

    if not histories:
        raise RuntimeError(
            f"Unable to collect normalized histories for action '{action}'."
        )

    stacked = torch.cat(histories, dim=0)
    if stacked.size(0) < num_samples:
        print(
            f"  Only found {stacked.size(0)} samples for action '{action}', "
            f"less than requested {num_samples}."
        )
    return stacked[:num_samples]


def resolve_model_components(model_name: str | None) -> ModelComponents:
    normalized = (model_name or DEFAULT_MODEL_NAME).lower()
    components = SUPPORTED_MODELS.get(normalized)
    if components is None:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")
    return components


def _get_first_tag(parsed: dict[str, str | None], *keys: str) -> str | None:
    for key in keys:
        value = parsed.get(key)
        if value not in (None, "", "none", "null", "None", "Null"):
            return value
    return None


def build_config(dataset: str, parsed_tag: dict[str, str | None], components: ModelComponents) -> Any:
    metadata = get_dataset_metadata(dataset)
    defaults = _model_defaults(components.model_key)

    input_length = coerce_int(parsed_tag.get("in"), int(defaults.get("input_n", 70)))
    pred_length = coerce_int(parsed_tag.get("out"), int(defaults.get("output_n", 30)))
    hidden_dim = coerce_int(parsed_tag.get("hs"), int(defaults.get("hidden_size", 64)))
    norm_axis = parsed_tag.get("simnorm") or defaults.get("simlpe_norm_axis", "spatial")

    dct_override = coerce_optional_int(parsed_tag.get("dctk"))
    gate_override = parsed_tag.get("simgate")
    mask_source = _get_first_tag(parsed_tag, "simmask", "simlpe_mask_temperature")

    layers_override = coerce_optional_int(parsed_tag.get("simlayers"))
    num_layers = layers_override if layers_override is not None else defaults.get("num_layers")
    if num_layers is None:
        num_layers = 6
    cfg_kwargs = {
        "input_length": input_length,
        "pred_length": pred_length,
        "num_nodes": int(metadata["node_count"]),
        "hidden_dim": hidden_dim,
        "num_layers": int(num_layers),
        "norm_axis": norm_axis,
        "dct_components": dct_override if dct_override is not None else defaults.get("dct_keep_coeffs"),
        "mid_gate_max": coerce_float(gate_override, float(defaults.get("simlpe_mid_gate_max", 0.9))),
        "mask_temperature": coerce_float(mask_source, float(defaults.get("simlpe_mask_temperature", 1.0))),
    }
    return components.config_cls(**cfg_kwargs)


def list_actions(model_dir: Path, model: str, dataset: str, explicit: Sequence[str] | None) -> list[str]:
    available = list(find_available_actions(model_dir, model, dataset))
    if explicit is None:
        if not available:
            raise ValueError(f"No checkpoints found in {model_dir} for dataset '{dataset}'.")
        return available
    missing = [act for act in explicit if act not in available]
    if missing:
        raise ValueError(
            f"Actions {missing} were requested but not found for dataset '{dataset}' in {model_dir}."
        )
    return list(explicit)


@dataclass
class ActionGateSummary:
    action: str
    gates: torch.Tensor  # shape: [K, G] or [B, K, G]


def inspect_checkpoint(
    model: str,
    model_dir: Path,
    dataset: str,
    action: str,
    mode: str,
    components: ModelComponents,
    num_samples: int | None,
    *,
    data_dir: Path | None = None,
) -> ActionGateSummary:
    if mode != "gate":
        raise ValueError(f"Mode '{mode}' not supported for MoE plot. Use 'gate'.")

    checkpoint_path = resolve_checkpoint_path(model_dir, model, dataset, action)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    parsed_tag = parse_tag(metadata.get("tag"))
    cfg = build_config(dataset, parsed_tag, components)

    forecaster = components.forecaster_cls(cfg)
    state_dict = checkpoint.get("best_model_state", checkpoint)
    # Optional wrapper handling
    if isinstance(state_dict, dict) and "sfgdnet_try" in state_dict:
        state_dict = state_dict["sfgdnet_try"]
    forecaster.load_state_dict(state_dict)
    forecaster.eval()

    stride, time_interp, data_seed = _parse_preprocessing_params(parsed_tag)
    train_batch_size = max(1, coerce_int(parsed_tag.get("bs"), 5))
    effective_samples = num_samples if num_samples is not None else train_batch_size
    if effective_samples < 1:
        effective_samples = 1
    if num_samples is None:
        print(f"  Using training batch size ({effective_samples}) samples for gate inspection.")
    else:
        print(f"  Using {effective_samples} requested samples (training batch size={train_batch_size}).")

    loader_batch_size = max(1, min(effective_samples, train_batch_size))
    history_batch = load_normalized_history_samples(
        dataset,
        action,
        input_length=cfg.input_length,
        pred_length=cfg.pred_length,
        num_nodes=cfg.num_nodes,
        num_samples=effective_samples,
        stride=max(1, stride),
        time_interp=time_interp,
        data_seed=data_seed,
        data_dir=data_dir,
        loader_batch_size=loader_batch_size,
    )

    with torch.no_grad():
        _ = forecaster(history_batch)

    gates = forecaster.last_gates
    if gates is None:
        raise RuntimeError(f"Gates were not populated for action '{action}'.")

    # Accept both [1, K, G] and [B, K, G]
    gates = gates.cpu()
    return ActionGateSummary(
        action=action,
        gates=gates,
    )


def plot_gates(summaries: Sequence[ActionGateSummary], output: Path, mode: str) -> None:
    if mode != "gate":
        raise ValueError(f"Mode '{mode}' not supported for MoE plot. Use 'gate'.")

    if not summaries:
        raise ValueError("No summaries provided for plotting.")

    # Infer number of experts from first summary
    first_gates = summaries[0].gates
    if first_gates.ndim == 2:
        # [K, G]
        num_experts = first_gates.shape[1]
    elif first_gates.ndim == 3:
        # [B, K, G]
        num_experts = first_gates.shape[2]
    else:
        raise ValueError(f"Unexpected gate tensor shape: {first_gates.shape}")

    with plt.rc_context({"font.size": plt.rcParams["font.size"] * 3}):
        fig, axes = plt.subplots(num_experts, 1, figsize=(24, 6 + 4 * num_experts), sharex=True)
        if num_experts == 1:
            axes = [axes]

        linewidth = 3.5

        for expert_idx in range(num_experts):
            ax = axes[expert_idx]
            ax.set_title(f"Expert {expert_idx} gate", fontsize="medium")

            for summary in summaries:
                gates = summary.gates

                # Normalize shape to [B, K, G]
                if gates.ndim == 2:
                    # [K, G] -> [1, K, G]
                    gates_bkg = gates.unsqueeze(0)
                else:
                    gates_bkg = gates

                batch_size, k_len, g = gates_bkg.shape
                assert g == num_experts, "Inconsistent number of experts across summaries."

                xs = range(k_len)
                for b in range(batch_size):
                    label = summary.action if b == 0 else None
                    ax.plot(xs, gates_bkg[b, :, expert_idx].numpy(), label=label, linewidth=linewidth)

            ax.set_ylabel("Gate value", fontsize="medium")
            ax.tick_params(axis="both", labelsize="medium")
            ax.grid(True, alpha=0.25)

        axes[-1].set_xlabel("DCT component index", fontsize="medium")
        axes[-1].legend(loc="lower right", fontsize="medium")

        fig.tight_layout()
        fig.savefig(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MoE sfgdnet checkpoints across actions.")
    parser.add_argument(
        "--dataset",
        "-d",
        default="assembly",
        help="Dataset prefix used when naming checkpoints (default: assembly).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Model prefix used when naming checkpoints (default: sfgdnet_try).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory with model checkpoints (default: {DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "-a",
        "--actions",
        nargs="+",
        help="Specific actions to inspect (default: all available).",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("last_.png"),
        help="Where to write the gate visualization (default: last_.png).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="gate",
        help="Mode for gate extraction (only 'gate' supported for MoE).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of history samples per action; defaults to the training batch size stored in the checkpoint tag.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional dataset directory override when loading real Assembly samples.",
    )
    args = parser.parse_args()

    model_name = (args.model or DEFAULT_MODEL_NAME).lower()
    components = resolve_model_components(model_name)
    requested = None if not args.actions else [act.strip() for act in args.actions if act.strip()]
    actions = list_actions(args.model_dir, model_name, args.dataset, requested)

    summaries: list[ActionGateSummary] = []
    for action in actions:
        print(f"Loading action {action}...")
        summary = inspect_checkpoint(
            model_name,
            args.model_dir,
            args.dataset,
            action,
            args.mode,
            components,
            num_samples=args.num_samples,
            data_dir=args.data_dir,
        )
        summaries.append(summary)
        gates = summary.gates
        if gates.ndim == 2:
            # [K, G]
            bsz = 1
            dct_len = gates.shape[0]
        else:
            # [B, K, G]
            bsz = gates.shape[0]
            dct_len = gates.shape[1]
        print(
            f"Action {action}: batch_size={bsz}, "
            f"dct_len={dct_len}, num_experts={gates.shape[-1]}"
        )

    plot_output = str(args.plot_output).split(".")[0] + args.mode
    plot_gates(summaries, plot_output, args.mode)
    print(f"Saved gate visualization to {plot_output}")


if __name__ == "__main__":
    main()
