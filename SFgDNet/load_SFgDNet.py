
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import torch
import os

from data import get_dataset_metadata
from models.SFgDNet import SFgDNetConfig, SFgDNetForecaster
from models.sfgdnet_try import SFgDNetTryConfig, SFgDNetTryForecaster


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
            yield stem[len(prefix) :]


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


def resolve_model_components(model_name: str) -> ModelComponents:
    normalized = str(model_name).lower()
    if normalized == "sfgdnet_try":
        return ModelComponents(SFgDNetTryConfig, SFgDNetTryForecaster)
    return ModelComponents(SFgDNetConfig, SFgDNetForecaster)


def build_config(dataset: str, parsed_tag: dict[str, str | None], config_cls: type) -> Any:
    metadata = get_dataset_metadata(dataset)
    return config_cls(
        input_length=coerce_int(parsed_tag.get("in"), 70),
        pred_length=coerce_int(parsed_tag.get("out"), 30),
        num_nodes=int(metadata["node_count"]),
        hidden_dim=coerce_int(parsed_tag.get("hs"), 64),
        norm_axis=parsed_tag.get("simnorm") or "spatial",
        dct_components=coerce_optional_int(parsed_tag.get("dctk")),
        slow_num_layers=coerce_optional_int(parsed_tag.get("simslow")),
        fast_num_layers=coerce_optional_int(parsed_tag.get("simfast")),
        mid_gate_max=coerce_float(parsed_tag.get("simgate"), 0.9),
        mask_temperature=coerce_float(parsed_tag.get("simmask") or parsed_tag.get("simlpe_mask_temperature"), 1.0),
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
    gate_slow: torch.Tensor
    gate_fast: torch.Tensor


def inspect_checkpoint(
    model: str,
    model_dir: Path,
    dataset: str,
    action: str,
    mode: str,
    components: ModelComponents,
) -> ActionGateSummary:
    checkpoint_path = resolve_checkpoint_path(model_dir, model, dataset, action)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = checkpoint.get("metadata", {})
    parsed_tag = parse_tag(metadata.get("tag"))
    cfg = build_config(dataset, parsed_tag, components.config_cls)

    forecaster = components.forecaster_cls(cfg)
    state_dict = checkpoint.get("best_model_state", checkpoint)
    if isinstance(state_dict, dict) and "SFgDNet" in state_dict:
        state_dict = state_dict["SFgDNet"]
    if isinstance(state_dict, dict) and "sfgdnet_try" in state_dict:
        state_dict = state_dict["sfgdnet_try"]
    forecaster.load_state_dict(state_dict)
    forecaster.eval()

    dummy_history = torch.zeros(1, cfg.input_length, cfg.num_nodes, 3, dtype=torch.float32)
    with torch.no_grad():
        _ = forecaster(dummy_history)

    if mode == "gate":
        gate_slow = forecaster.last_gate_slow
        gate_fast = forecaster.last_gate_fast
    if mode == "mask":
        gate_slow = forecaster.last_mask
        gate_fast = 1 - forecaster.last_mask
    if gate_slow is None or gate_fast is None:
        raise RuntimeError(f"Gates were not populated for action '{action}'.")

    gate_slow = gate_slow.squeeze(0).squeeze(-1).cpu()
    gate_fast = gate_fast.squeeze(0).squeeze(-1).cpu()

    return ActionGateSummary(
        action=action,
        gate_slow=gate_slow,
        gate_fast=gate_fast,
    )


def plot_gates(summaries: Sequence[ActionGateSummary], output: Path, mode: str) -> None:
    with plt.rc_context({"font.size": plt.rcParams["font.size"] * 3}):
        fig, axes = plt.subplots(2, 1, figsize=(24, 16), sharex=True)
        if mode == "gate":
            axes[0].set_title("Slow gate", fontsize="medium")
            axes[1].set_title("Fast gate", fontsize="medium")
        if mode == "mask":
            axes[0].set_title("Slow mask", fontsize="medium")
            axes[1].set_title("Fast mask", fontsize="medium")

        for summary in summaries:
            n_slow = summary.gate_slow.numel()
            n_fast = summary.gate_fast.numel()

            xs_slow = range(n_slow)
            xs_fast = range(n_fast)

            linewidth = 3.5

            slow_line = axes[0].plot(xs_slow, summary.gate_slow.numpy(), label=summary.action, linewidth=linewidth)[0]
            fast_line = axes[1].plot(xs_fast, summary.gate_fast.numpy(), label=summary.action, linewidth=linewidth)[0]

        for ax in axes:
            if mode == "gate":
                ax.set_ylabel("Gate value", fontsize="medium")
            if mode == "mask":
                ax.set_ylabel("Mask value", fontsize="medium")
            ax.tick_params(axis="both", labelsize="medium")
            ax.grid(True, alpha=0.25)

        axes[-1].set_xlabel("DCT component index", fontsize="medium")
        axes[1].legend(loc="lower right", fontsize="medium")

        fig.tight_layout()
        fig.savefig(output)

def plot_mean_sigmoid(summaries: Sequence[ActionGateSummary], output: Path) -> None:
    if not summaries:
        raise ValueError("Need at least one action summary to compute mean sigmoid.")

    def _calculate_stats(attr_gate: str, attr_center: str):
        gates = [getattr(summary, attr_gate) for summary in summaries]
        min_len = min(g.numel() for g in gates)
        if min_len == 0:
            raise ValueError(f"Gate '{attr_gate}' has zero length for at least one action.")

        trimmed = torch.stack([g[:min_len] for g in gates])


        return trimmed.mean(0), min_len

    mean_slow, slow_len, nu_s = _calculate_stats("gate_slow", "center_slow")
    mean_fast, fast_len, nu_f = _calculate_stats("gate_fast", "center_fast")

    max_len = max(slow_len, fast_len)

    with plt.rc_context({"font.size": plt.rcParams["font.size"] * 2.5}):
        fig, ax = plt.subplots(figsize=(18, 6))

        xs_slow = range(slow_len)
        xs_fast = range(fast_len)

        slow_line = ax.plot(xs_slow, mean_slow.numpy(), label="Slow gate", linewidth=3.0)[0]
        fast_line = ax.plot(xs_fast, mean_fast.numpy(), label="Fast gate", linewidth=3.0)[0]

        ax.set_title("Sigmoid gates", fontsize="medium")
        ax.set_xlabel(r"$\nu$", fontsize="medium")
        ax.set_ylabel("Gate value", fontsize="medium")
        ax.tick_params(axis="both", labelsize="medium")
        ax.grid(True, alpha=0.25)

        color_slow = slow_line.get_color()
        color_fast = fast_line.get_color()

        ax.axvline(nu_s, color=color_slow, linestyle=":", linewidth=2.5, label=r"$\nu_s$")
        ax.axvline(nu_f, color=color_fast, linestyle=":", linewidth=2.5, label=r"$\nu_f$")

        base_ticks = [0, 25, 50, 75, 100]
        extra_ticks = sorted({int(nu_s), int(nu_f)})
        ax.set_xticks(base_ticks + extra_ticks)
        ax.set_xticklabels(
            [f"{t}" for t in base_ticks] +
            [r"$\nu_s$" if t == int(nu_s) else r"$\nu_f$" for t in extra_ticks]
        )

        if max_len > 1:
            ax.set_xlim(0, max_len - 1)

        ax.legend(loc="lower right", bbox_to_anchor=(0.75, 0.0), fontsize="medium")

        fig.tight_layout()
        fig.savefig(output)

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SFgDNet checkpoints across actions.")
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
        help=f"Directory with SFgDNet checkpoints (default: {DEFAULT_MODEL_DIR}).",
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
        "--mean-plot-output",
        type=Path,
        default=Path("mean_sigmoid.png"),
        help="Where to write the mean sigmoid visualization (default: mean_sigmoid.png).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="gate",
        help="Mode for gate extraction (default: gates, opt: mask).",
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
        )
        summaries.append(summary)
        print(
            f"Action {action}: dct_len={summary.gate_slow.numel()}"
        )

    plot_output = str(args.plot_output).split(".")[0] + args.mode
    mean_plot_output = str(args.mean_plot_output).split(".")[0] + args.mode


    plot_gates(summaries, plot_output, args.mode)
    print(f"Saved gate visualization to {plot_output}")
    # plot_mean_sigmoid(summaries, mean_plot_output)
    # print(f"Saved mean sigmoid visualization to {mean_plot_output}")


if __name__ == "__main__":
    main()
