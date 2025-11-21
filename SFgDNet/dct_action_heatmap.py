#!/usr/bin/env python3
"""Generate assembly-action heatmaps of simple dual DCT coefficients per node."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable, Sequence
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch

from config import DATASET_CHOICES
from data import (
    collect_sequences_from_files_wrist_centered,
    get_dataset_metadata,
)
from datasets import AssemblyDataset


def _build_dct_matrix(total_len: int) -> torch.Tensor:
    if total_len <= 0:
        raise ValueError("Sequence length must be positive to build DCT.")
    n = torch.arange(total_len).unsqueeze(0)
    k = torch.arange(total_len).unsqueeze(1)
    scale = math.sqrt(2.0 / total_len)
    mat = scale * torch.cos(torch.pi * (n + 0.5) * k / float(total_len))
    mat[0, :] = math.sqrt(1.0 / total_len)
    return mat


def _select_files(data_dir: Path, action_filter: str, max_files: int | None) -> list[Path]:
    pattern = data_dir.glob("*.npy")
    candidates = [p for p in sorted(pattern) if action_filter in p.name]
    if max_files is not None:
        candidates = candidates[:max_files]
    return candidates


def _prepare_dataset(
    sequences: Sequence[tuple[np.ndarray, float]],
    input_n: int,
    output_n: int,
    stride: int,
) -> AssemblyDataset:
    return AssemblyDataset(sequences, input_n=input_n, output_n=output_n, stride=stride)


def _compute_action_heatmap(
    dataset: AssemblyDataset,
    node_count: int,
    dct_matrix: torch.Tensor,
    output_n: int,
    max_windows: int | None,
) -> tuple[np.ndarray, int] | None:
    total_len = dct_matrix.size(1)
    num_freq = dct_matrix.size(0)
    acc = torch.zeros(node_count-2, num_freq, dtype=torch.float64)
    seen = 0

    for idx in range(len(dataset)):
        if max_windows is not None and seen >= max_windows:
            break
        sample = dataset[idx]
        inp = sample[0]
        if len(sample) > 2:
            _ = sample[2]
        if not isinstance(inp, torch.Tensor):
            continue
        inp = inp[..., 4:].to(dtype=torch.float32)
        if inp.shape[0] == 0:
            continue
        last_frame = inp[-1:].reshape(1, -1)
        padded = torch.cat([inp.reshape(inp.shape[0], -1), last_frame.expand(output_n, -1)], dim=0)
        if padded.size(0) < total_len:
            pad_len = total_len - padded.size(0)
            padded = torch.cat([padded, padded[-1:].expand(pad_len, -1)], dim=0)
        elif padded.size(0) > total_len:
            padded = padded[-total_len:]
        coeffs = dct_matrix.to(dtype=padded.dtype) @ padded
        coeffs = coeffs.view(num_freq, node_count, 3)
        mask = (coeffs.abs() != 0)          # True for non-zero rows
        coeffs_nz = coeffs[mask].reshape(num_freq, -1, 3)                       # (f, r - m, c)
        mag = torch.log(coeffs_nz.abs().mean(dim=-1))
        acc += mag.transpose(0, 1).to(dtype=acc.dtype)
        seen += 1

    if seen == 0:
        return None
    return (acc / seen).cpu().numpy(), seen


def _plot_heatmaps(
    heatmaps: list[tuple[str, np.ndarray | None, int]],
    output_path: Path,
    node_count: int,
) -> None:
    actions = [entry[0] for entry in heatmaps]
    valid_maps = [entry[1] for entry in heatmaps if entry[1] is not None]
    if not valid_maps:
        raise SystemExit("No heatmaps generated; please check the dataset inputs.")

    vmin = float(min(np.nanmin(m) for m in valid_maps))
    vmax = float(max(np.nanmax(m) for m in valid_maps))
    cols = min(3, len(actions))
    rows = (len(actions) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 3 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    # plt.set_loglevel("warning")

    for idx, (action, hm, count) in enumerate(heatmaps):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        title = f"{action} ({count} windows)" if hm is not None else f"{action} (no data)"
        ax.set_title(title)
        ax.set_xlabel("DCT component")
        ax.set_ylabel("Node index")
        ax.set_ylim(-0.5, node_count - 0.5)
        if hm is None:
            ax.text(0.5, 0.5, "no samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            continue
        freq_vals = np.linspace(1e-4, 1.0, hm.shape[1], dtype=np.float64)
        extent = (freq_vals[0], freq_vals[-1], -0.5, node_count - 0.5)
        im = ax.imshow(
            hm,
            extent=extent,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        # ax.set_xscale("log")
        tick_vals = np.geomspace(freq_vals[0], freq_vals[-1], num=5)
        ax.set_xticks(tick_vals)
        ax.set_xticklabels([f"{t:.3f}" for t in tick_vals])
        ax.set_yticks(range(0, node_count, max(1, node_count // 6)))
    for idx in range(len(actions), rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical")
    cbar.set_label("Mean |DCT coefficient|")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_mean(
    heatmaps: List[Tuple[str, np.ndarray | None, int]],
    output_path: Path,
    node_count: int,  # kept for API compatibility, not used
) -> None:
    """
    Plot a single figure with one line per action.
    Each line is the mean over node indices of the corresponding heatmap.
    """
    # Filter out entries with no heatmap
    valid_entries = [(action, hm, count) for action, hm, count in heatmaps if hm is not None]
    if not valid_entries:
        raise SystemExit("No heatmaps generated; please check the dataset inputs.")

    # Compute mean over node axis for each action (independent of node_count)
    mean_curves: dict[tuple[str, int], np.ndarray] = {}
    for action, hm, count in valid_entries:
        # mean over node axis -> 1D over DCT components
        mean_curve = np.nanmean(hm, axis=0)  # shape: (num_freqs,)
        mean_curves[(action, count)] = mean_curve

    # Frequency axis (based on first heatmap)
    first_hm = valid_entries[0][1]
    num_freqs = first_hm.shape[1]
    freq_vals = np.linspace(1e-4, 1.0, num_freqs, dtype=np.float64)

    # (Optional) sanity check: all heatmaps have same number of frequency bins
    for action, hm, _ in valid_entries:
        if hm.shape[1] != num_freqs:
            raise ValueError(
                f"Heatmap for action '{action}' has {hm.shape[1]} frequency bins, "
                f"expected {num_freqs}."
            )

    # Global y-limits based on all mean curves
    all_vals = np.concatenate(list(mean_curves.values()))
    ymin = float(np.nanmin(all_vals))
    ymax = float(np.nanmax(all_vals))

    print(ymin, ymax)

    # Create single plot
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)

    for (action, count), curve in mean_curves.items():
        label = f"{action} ({count} windows)"
        ax.plot(freq_vals, curve, label=label)

    ax.set_xlabel("DCT component (frequency)")
    ax.set_ylabel("Mean |DCT coefficient| over nodes")
    ax.set_title("Mean |DCT coefficient| over nodes for each action")
    ax.set_ylim(ymin, ymax)

    # Use geometric spaced ticks on x-axis
    tick_vals = np.geomspace(freq_vals[0], freq_vals[-1], num=5)
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f"{t:.3f}" for t in tick_vals])

    ax.legend()
    ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    
def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot mean DCT frequencies per node for assembly actions.")
    parser.add_argument("--data-dir", type=str, default=None, help="Root path to the assembly .npy files.")
    parser.add_argument(
        "--actions",
        type=str,
        default="pick,put,remove,unscrew,rotate,attempt",
        help="Comma-separated action substrings to include.",
    )
    parser.add_argument("--input-n", type=int, default=90)
    parser.add_argument("--output-n", type=int, default=10)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--dct-components", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=2000, help="Max windows processed per action.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files examined per action.")
    parser.add_argument("--time-interp", type=int, default=None, help="Optional temporal interpolation length.")
    parser.add_argument("--window-norm", type=int, default=None, help="Window size used for scale normalization.")
    parser.add_argument("--output", type=str, default="out/assembly_dct_heatmaps.png")
    parser.add_argument("--dataset", type=str, default="assembly", choices=DATASET_CHOICES)
    args = parser.parse_args(argv)

    dataset_name = args.dataset.lower()
    metadata = get_dataset_metadata(dataset_name)
    data_dir = Path(args.data_dir or metadata.get("default_dir", ""))
    if not data_dir.exists():
        raise SystemExit(f"Data directory {data_dir} does not exist.")

    actions = [act.strip() for act in args.actions.split(",") if act.strip()]
    total_len = args.input_n + args.output_n
    dct_matrix = _build_dct_matrix(total_len)
    keep = args.dct_components or total_len
    keep = max(1, min(int(keep), total_len))
    dct_keep = dct_matrix[:keep]

    hand_groups = tuple(metadata.get("hand_groups", ()))
    time_interp = args.time_interp
    window_norm = args.window_norm or args.input_n

    results: list[tuple[str, np.ndarray | None, int]] = []
    for action in actions:
        files = _select_files(data_dir, action, args.max_files)
        if not files:
            print(f"[warn] No files for action '{action}' in {data_dir}")
            results.append((action, None, 0))
            continue
        sequences = collect_sequences_from_files_wrist_centered(
            [str(p) for p in files],
            node_count=int(metadata.get("node_count", 42)),
            hand_groups=hand_groups,
            dataset=dataset_name,
            time_interp=time_interp,
            window_norm=window_norm,
        )
        if not sequences:
            print(f"[warn] No sequences were loaded for action '{action}'.")
            results.append((action, None, 0))
            continue
        dataset = _prepare_dataset(sequences, args.input_n, args.output_n, args.stride)
        heatmap_result = _compute_action_heatmap(
            dataset,
            node_count=int(metadata.get("node_count", 42)),
            dct_matrix=dct_keep,
            output_n=args.output_n,
            max_windows=args.max_windows,
        )
        if heatmap_result is None:
            results.append((action, None, 0))
            continue
        heatmap, count = heatmap_result
        results.append((action, heatmap, count))
        print(f"[info] Processed {count} windows for action '{action}'.")

    # _plot_heatmaps(results, Path(args.output), int(metadata.get("node_count", 42)))
    _plot_mean(results, Path(args.output), int(metadata.get("node_count", 42)))
    print(f"Saved heatmap grid to {args.output}")


if __name__ == "__main__":
    main()
