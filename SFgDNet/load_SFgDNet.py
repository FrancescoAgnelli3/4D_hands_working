from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import torch

from data import collect_sequences_from_files_wrist_centered, get_dataset_metadata
from models.sfgdnet_try import SFgDNetTryConfig, SFgDNetTryForecaster # <- MoE forecaster


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "examples" / "models"
MODEL_FILE_TEMPLATE = "{model}_{dataset}_{action}.pt"


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


@dataclass(frozen=True)
class ModelComponents:
    config_cls: type
    forecaster_cls: type


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


def load_real_history_samples(
    dataset: str,
    action: str,
    *,
    input_length: int,
    num_nodes: int,
    num_samples: int,
    data_dir: Path | None = None,
) -> torch.Tensor:
    metadata = get_dataset_metadata(dataset)
    resolved_dir = _resolve_data_dir(dataset, data_dir)
    pattern = f"*{action}*.npy"
    files = sorted(resolved_dir.glob(pattern))[:num_samples]
    if not files:
        raise FileNotFoundError(
            f"No files found in '{resolved_dir}' matching action '{action}'. "
            "Pass --data-dir to point at a directory with Assembly .npy files."
        )

    sequences = collect_sequences_from_files_wrist_centered(
        [str(path) for path in files],
        node_count=num_nodes,
        hand_groups=metadata["hand_groups"],
        dataset=dataset,
        time_interp=None,
        window_norm=input_length,
    )

    histories: list[torch.Tensor] = []
    for seq, _ in sequences:
        coords = torch.from_numpy(seq[:, :, 4:]).to(dtype=torch.float32)
        total_frames = coords.shape[0]
        if total_frames < input_length:
            continue
        max_start = total_frames - input_length + 1
        for start in range(max_start):
            histories.append(coords[start:start + input_length])
            if len(histories) >= num_samples:
                break
        if len(histories) >= num_samples:
            break

    if not histories:
        raise RuntimeError(
            f"Unable to extract any {input_length}-frame history windows for action '{action}'."
        )

    if len(histories) < num_samples:
        print(
            f"Only found {len(histories)} history windows for action '{action}', "
            f"less than the requested batch size {num_samples}."
        )

    batch = torch.stack(histories[:num_samples], dim=0)
    return batch


def resolve_model_components(model_name: str) -> ModelComponents:
    # Only the MoE sfgdnet model
    return ModelComponents(SFgDNetTryConfig, SFgDNetTryForecaster)


def build_config(dataset: str, parsed_tag: dict[str, str | None], config_cls: type) -> Any:
    metadata = get_dataset_metadata(dataset)
    return config_cls(
        input_length=coerce_int(parsed_tag.get("in"), 70),
        pred_length=coerce_int(parsed_tag.get("out"), 30),
        num_nodes=int(metadata["node_count"]),
        hidden_dim=coerce_int(parsed_tag.get("hs"), 64),
        norm_axis=parsed_tag.get("simnorm") or "spatial",
        dct_components=coerce_optional_int(parsed_tag.get("dctk")),
        slow_num_layers=coerce_optional_int(parsed_tag.get("simslow")),  # ignored by MoE if not used
        fast_num_layers=coerce_optional_int(parsed_tag.get("simfast")),  # ignored by MoE if not used
        mid_gate_max=coerce_float(parsed_tag.get("simgate"), 0.9),
        mask_temperature=coerce_float(
            parsed_tag.get("simmask") or parsed_tag.get("simlpe_mask_temperature"),
            1.0,
        ),
    )


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
    num_samples: int,
    *,
    data_dir: Path | None = None,
) -> ActionGateSummary:
    if mode != "gate":
        raise ValueError(f"Mode '{mode}' not supported for MoE plot. Use 'gate'.")

    checkpoint_path = resolve_checkpoint_path(model_dir, model, dataset, action)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    parsed_tag = parse_tag(metadata.get("tag"))
    cfg = build_config(dataset, parsed_tag, components.config_cls)

    forecaster = components.forecaster_cls(cfg)
    state_dict = checkpoint.get("best_model_state", checkpoint)
    # Optional wrapper handling
    if isinstance(state_dict, dict) and "sfgdnet_try" in state_dict:
        state_dict = state_dict["sfgdnet_try"]
    forecaster.load_state_dict(state_dict)
    forecaster.eval()

    history_batch = load_real_history_samples(
        dataset,
        action,
        input_length=cfg.input_length,
        num_nodes=cfg.num_nodes,
        num_samples=num_samples,
        data_dir=data_dir,
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
        default="sfgdnet_try",
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
        default=5,
        help="Number of samples in the batch to use for gate inspection (default: 5).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional dataset directory override when loading real Assembly samples.",
    )
    args = parser.parse_args()

    components = resolve_model_components(args.model)
    requested = None if not args.actions else [act.strip() for act in args.actions if act.strip()]
    actions = list_actions(args.model_dir, args.model, args.dataset, requested)

    summaries: list[ActionGateSummary] = []
    for action in actions:
        print(f"Loading action {action}...")
        summary = inspect_checkpoint(
            args.model,
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
