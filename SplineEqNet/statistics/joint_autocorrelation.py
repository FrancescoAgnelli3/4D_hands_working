#!/usr/bin/env python3
"""
Compute and plot per-joint autocorrelation for a dataset.

By default, loads all files for the dataset (default: assembly), with no action
filter, then computes autocorrelation per joint over all frames and saves a plot.

Example:
  python joint_autocorrelation.py --dataset assembly --max_lag 80
  python joint_autocorrelation.py --dataset assembly --action_filter pick --output out/autocorr_pick.png
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tqdm import tqdm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import get_dataset_metadata


def _skel_defaults_for_middle(dataset: str) -> Tuple[int, int]:
    return (0, 9)


def _rotation_from_a_to_b(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.eye(3, dtype=np.float64)
    a = a / na
    b = b / nb

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))

    if s < eps:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(axis, a))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - float(np.dot(axis, a)) * a
        axis = axis / np.maximum(np.linalg.norm(axis), eps)
        K = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)

    axis = v / s
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + K * s + (K @ K) * (1.0 - c)


def _normalize_hand(
    hand: np.ndarray,
    *,
    dataset: str,
    wrist_l: int,
    mode: str,
    eps: float = 1e-8,
) -> np.ndarray:
    out = hand.astype(np.float64, copy=True)
    if mode == "none":
        return out

    if mode == "mean_var":
        mean = out.mean(axis=0, keepdims=True)
        std = out.std(axis=0, keepdims=True)
        std = np.where(std > eps, std, 1.0)
        return (out - mean) / std

    _, middle_l = _skel_defaults_for_middle(dataset)
    if mode == "frame0":
        origin = out[0, wrist_l, :3].copy()
        out -= origin[None, None, :]
        if 0 <= middle_l < out.shape[1]:
            u = out[0, middle_l, :3] - out[0, wrist_l, :3]
            scale = float(np.linalg.norm(u))
            if np.isfinite(scale) and scale > eps:
                out /= scale
        return out

    if mode == "frame0_rot":
        origin = out[0, wrist_l, :3].copy()
        out -= origin[None, None, :]
        if 0 <= middle_l < out.shape[1]:
            u = out[0, middle_l, :3] - out[0, wrist_l, :3]
            scale = float(np.linalg.norm(u))
            if np.isfinite(scale) and scale > eps:
                out /= scale
                u_scaled = out[0, middle_l, :3] - out[0, wrist_l, :3]
                R = _rotation_from_a_to_b(u_scaled, np.array([1.0, 0.0, 0.0], dtype=np.float64), eps=eps)
                out = out @ R
        return out

    if mode == "all_frames":
        origin = out[:, wrist_l, :3].copy()
        out -= origin[:, None, :]
        if 0 <= middle_l < out.shape[1]:
            u = out[:, middle_l, :3] - out[:, wrist_l, :3]
            scales = np.linalg.norm(u, axis=-1)
            valid = np.isfinite(scales) & (scales > eps)
            out[valid] /= scales[valid, None, None]
        return out

    if mode == "all_frames_rot":
        origin = out[:, wrist_l, :3].copy()
        out -= origin[:, None, :]
        if 0 <= middle_l < out.shape[1]:
            u = out[:, middle_l, :3] - out[:, wrist_l, :3]
            scales = np.linalg.norm(u, axis=-1)
            valid = np.isfinite(scales) & (scales > eps)
            out[valid] /= scales[valid, None, None]
            for t in np.where(valid)[0]:
                u_scaled = out[t, middle_l, :3] - out[t, wrist_l, :3]
                R = _rotation_from_a_to_b(
                    u_scaled, np.array([1.0, 0.0, 0.0], dtype=np.float64), eps=eps
                )
                out[t] = out[t] @ R
        return out

    raise ValueError(f"Unknown normalization mode: {mode}")


def _select_most_active_hand(
    coords: np.ndarray, hand_groups: Tuple[dict, ...]
) -> Optional[Tuple[np.ndarray, int]]:
    T, N, C = coords.shape
    if C != 3 or T < 2:
        return None

    candidates: List[Tuple[np.ndarray, float, int]] = []
    for group in hand_groups:
        nodes = [idx for idx in group["nodes"] if idx < N]
        if not nodes:
            continue
        g2l = {g: li for li, g in enumerate(nodes)}
        wrist_g = int(group["wrist_index"])
        if wrist_g not in g2l:
            continue
        wrist_l = int(g2l[wrist_g])
        hand = coords[:, nodes, :3].astype(np.float64, copy=False)
        hand_centered = hand - hand[:, wrist_l : wrist_l + 1, :]
        diff = hand_centered[1:] - hand_centered[:-1]
        motion_energy = float(np.mean(np.linalg.norm(diff, axis=-1)))
        candidates.append((hand, motion_energy, wrist_l))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_hand, _, best_wrist_l = candidates[0]
    return best_hand, best_wrist_l


def _resolve_data_dir(dataset: str, override: Optional[str]) -> Path:
    if override:
        data_dir = Path(override)
    else:
        metadata = get_dataset_metadata(dataset)
        default_dir = metadata.get("default_dir")
        if not default_dir:
            raise ValueError(f"No default data directory configured for dataset '{dataset}'.")
        data_dir = Path(default_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory '{data_dir}' does not exist.")
    return data_dir


def _list_dataset_files(data_dir: Path, action_filter: str) -> List[str]:
    files = sorted(data_dir.glob("*.npy"))
    if action_filter:
        files = [path for path in files if action_filter in path.name]
    return [str(path) for path in files]


def _load_sequences(
    *,
    dataset: str,
    data_dir: Optional[str],
    action_filter: Optional[str],
    time_interp: Optional[int],
    max_files: Optional[int],
    seed: int,
    normalization: str,
) -> Tuple[List[np.ndarray], int]:
    metadata = get_dataset_metadata(dataset)
    hand_groups: Tuple[dict, ...] = metadata["hand_groups"]
    resolved_dir = _resolve_data_dir(dataset, data_dir)
    action = "" if action_filter in (None, "", "none") else str(action_filter)
    files = _list_dataset_files(resolved_dir, action)
    if not files:
        raise RuntimeError(f"No .npy files found for dataset '{dataset}' in {resolved_dir}.")

    if max_files is not None and max_files > 0 and len(files) > max_files:
        rng = np.random.default_rng(seed)
        rng.shuffle(files)
        files = files[:max_files]

    sequences: List[np.ndarray] = []
    for f in tqdm(files, desc="Loading sequences"):
        try:
            d = np.load(f)
            dataset_lower = dataset.lower()
            if dataset_lower in {"assembly"}:
                d = d.transpose(0, 1, 3, 2).reshape((d.shape[0], d.shape[1], d.shape[2] * d.shape[3]))
                d = np.transpose(d, (1, 2, 0))
            elif dataset_lower == "h2o":
                d = d.transpose(1, 3, 2, 0).reshape(d.shape[1], d.shape[2] * d.shape[3], d.shape[0])
            # fpha assumed already in (T, N, 3) format
            if d.ndim != 3 or d.shape[-1] != 3:
                continue
            if time_interp is not None and time_interp > 0 and time_interp != d.shape[0]:
                t_orig = np.linspace(0, 1, d.shape[0])
                t_new = np.linspace(0, 1, time_interp)
                out = np.zeros((time_interp, d.shape[1], 3), dtype=d.dtype)
                for i in range(d.shape[1]):
                    for k in range(3):
                        out[:, i, k] = np.interp(t_new, t_orig, d[:, i, k])
                d = out
            coords = d  # (T, N, 3), raw
            if coords.ndim != 3 or coords.shape[-1] != 3 or coords.shape[0] < 2:
                continue
            selected = _select_most_active_hand(coords, hand_groups)
            if selected is None:
                continue
            coords, wrist_l = selected
            coords = _normalize_hand(
                coords,
                dataset=dataset,
                wrist_l=wrist_l,
                mode=normalization,
            )
            sequences.append(coords.astype(np.float64, copy=False))
        except Exception:
            continue

    if not sequences:
        raise RuntimeError("No usable sequences after preprocessing.")

    node_count = int(sequences[0].shape[1])
    if any(seq.shape[1] != node_count for seq in sequences):
        raise RuntimeError("Inconsistent joint counts across selected hand sequences.")
    return sequences, node_count


def _compute_autocorr(
    sequences: List[np.ndarray],
    node_count: int,
    max_lag: int,
    eps: float = 1e-12,
) -> np.ndarray:
    total_frames = sum(int(seq.shape[0]) for seq in sequences)
    if total_frames <= 1:
        raise RuntimeError("Not enough frames to compute autocorrelation.")

    # variance: mean over frames of (||x_t - mu||^2 / 3), per joint
    var_sum = np.zeros(node_count, dtype=np.float64)
    for seq in sequences:
        mean = seq.mean(axis=0)                      # (N, 3)
        centered = seq - mean[None, :, :]           # (T, N, 3)
        dots = np.einsum("tnc,tnc->tn", centered, centered) / 3.0  # (T, N)
        var_sum += dots.sum(axis=0)
    var = var_sum / float(total_frames)             # (N,)

    # covariance for each lag via np.correlate, per joint and per coordinate,
    # then average over coordinates and normalize by the number of pairs (T - lag).
    cov_sum = np.zeros((node_count, max_lag + 1), dtype=np.float64)
    count = np.zeros((max_lag + 1), dtype=np.float64)

    for seq in sequences:
        T = int(seq.shape[0])
        mean = seq.mean(axis=0)
        centered = seq - mean[None, :, :]           # (T, N, 3)

        max_lag_seq = min(max_lag, T - 1)
        lags = np.arange(max_lag_seq + 1, dtype=np.int64)

        for j in range(node_count):
            # accumulate sum_t (x_{t} * x_{t+lag}) for lag=0..max_lag_seq
            s = np.zeros(max_lag_seq + 1, dtype=np.float64)
            for c in range(3):
                x = centered[:, j, c]
                r_full = np.correlate(x, x, mode="full")   # length 2T-1
                r = r_full[T - 1 : T - 1 + (max_lag_seq + 1)]
                s += r
            cov_sum[j, : max_lag_seq + 1] += (s / 3.0)

        count[: max_lag_seq + 1] += (T - lags)

    cov = cov_sum / np.maximum(count[None, :], 1.0)
    acf = cov # / (var[:, None] + eps)
    return acf


def _plot_autocorr(acf: np.ndarray, output_path: Path, title: str) -> None:
    node_count, n_lags = acf.shape
    lags = np.arange(n_lags)

    if node_count >= 36:
        ncols = 6
    elif node_count >= 25:
        ncols = 5
    else:
        ncols = 4
    nrows = int(np.ceil(node_count / ncols))

    fig_w = max(10.0, ncols * 2.2)
    fig_h = max(6.0, nrows * 1.8)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for j in range(node_count):
        ax = axes[j]
        ax.plot(lags, acf[j], color="tab:blue", linewidth=1.0)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_title(f"Joint {j}", fontsize=8)
        if j % ncols == 0:
            ax.set_ylabel("ACF", fontsize=8)
        if j // ncols == nrows - 1:
            ax.set_xlabel("Lag", fontsize=8)
        ax.tick_params(labelsize=7)

    for k in range(node_count, axes.size):
        axes[k].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="assembly")
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--action_filter", type=str, default="pick_up_screwd")
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_lag", type=int, default=200)
    ap.add_argument(
        "--normalization",
        type=str,
        default="all_frames",
        choices=("none", "mean_var", "frame0", "frame0_rot", "all_frames", "all_frames_rot"),
        help="none | mean_var | frame0 | frame0_rot | all_frames | all_frames_rot",
    )
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    sequences, node_count = _load_sequences(
        dataset=args.dataset,
        data_dir=args.data_dir,
        action_filter=args.action_filter,
        time_interp=None,
        max_files=args.max_files,
        seed=args.seed,
        normalization=args.normalization,
    )

    acf = _compute_autocorr(sequences, node_count=node_count, max_lag=int(args.max_lag))

    action_suffix = ""
    if args.action_filter not in (None, "", "none"):
        action_suffix = f"_{args.action_filter}"
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).resolve().parent / "out" / f"autocorr_{args.dataset}{action_suffix}.png"

    title = f"Autocorrelation per joint ({args.dataset}{action_suffix})"
    _plot_autocorr(acf, output_path=output_path, title=title)

    print(f"Saved autocorrelation plot to {output_path}")


if __name__ == "__main__":
    main()
