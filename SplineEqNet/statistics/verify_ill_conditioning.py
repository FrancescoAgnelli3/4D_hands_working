#!/usr/bin/env python3
"""
Data-only ill-conditioning checks using per-coordinate RMS distances.

Implements:

(1) Quantile-defined neighborhood:
    Let dX_nn(i) be the (RMS) distance from query i to its nearest neighbor in X.
    Define tau_x = quantile(dX_nn, alpha).
    Then measure coverage and ambiguity using tau_x (with kNN approximation).

(2) Ratio-based conditioning proxy:
    For each query i, define R_i = dY_nn(i) / dX_nn(i),
    where dY_nn(i) is the (RMS) future distance to the nearest X-neighbor.
    Report quantiles/statistics of R.

Assumptions:
  - Windows are built with a large stride (e.g. 100) so overlap artifacts are reduced.
  - X: (N, Tin, Dflat), Y: (N, Tout, Dflat), already normalized in your pipeline.

Example:
  python verify_conditioning_dataonly.py --device cuda --stride 100 --alpha 0.2 --tau_y 0.5 --k 10 --max_queries 2000
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import calculate_features, get_dataset_metadata


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
    coords: np.ndarray,
    hand_groups: Tuple[dict, ...],
    window: Optional[int] = None,
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

        T_use = hand_centered.shape[0]
        T_win = T_use if window is None or window <= 0 else min(T_use, int(window))
        if T_win < 2:
            continue
        diff = hand_centered[1:T_win] - hand_centered[: T_win - 1]
        motion_energy = float(np.mean(np.linalg.norm(diff, axis=-1)))
        candidates.append((hand, motion_energy, wrist_l))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_hand, _, best_wrist_l = candidates[0]
    return best_hand, best_wrist_l



def flatten_seq(a: torch.Tensor) -> torch.Tensor:
    # (N, T, D) -> (N, T*D)
    return a.reshape(a.shape[0], -1)


def _stats(t: torch.Tensor) -> Dict[str, float]:
    t = t.detach().float().cpu()
    return {
        "mean": float(t.mean().item()),
        "median": float(t.median().item()),
        "p10": float(torch.quantile(t, 0.10).item()),
        "p25": float(torch.quantile(t, 0.25).item()),
        "p75": float(torch.quantile(t, 0.75).item()),
        "p90": float(torch.quantile(t, 0.90).item()),
        "p95": float(torch.quantile(t, 0.95).item()),
        "p99": float(torch.quantile(t, 0.99).item()),
        "max": float(t.max().item()),
    }


@torch.no_grad()
def conditioning_checks_rms(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    alpha: float,
    k: int = 10,
    max_queries: int = 2000,
    seed: int = 0,
    device: str = "cpu",
    top_m: int = 20,
    eps: float = 1e-12,
) -> Dict:
    """
    Runs both checks (1) quantile neighborhood and (2) ratio proxy.

    Distances:
      dx_rms = ||x_i - x_j|| / sqrt(dX)
      dy_rms = ||y_i - y_j|| / sqrt(dY)

    kNN approximation:
      - we compute k nearest neighbors in X for each query
      - define "NN" as the first neighbor in that kNN list (k must be >= 1)
      - define "within tau_x" among those k neighbors only
    """
    assert 0.0 < alpha < 1.0
    assert X.ndim == 3 and Y.ndim == 3
    assert X.shape[0] == Y.shape[0]
    N = X.shape[0]

    rng = np.random.default_rng(seed)
    qN = min(max_queries, N)
    q_idx = torch.as_tensor(rng.choice(N, size=qN, replace=False), device=device, dtype=torch.long)

    Xf = flatten_seq(X.to(device))  # (N, dX)
    Yf = flatten_seq(Y.to(device))  # (N, dY)
    dX = Xf.shape[1]
    dY = Yf.shape[1]
    sx = float(dX) ** 0.5
    sy = float(dY) ** 0.5

    # X distances (RMS) from queries to all points
    dx = torch.cdist(Xf[q_idx], Xf) / sx  # (qN, N)
    dx[torch.arange(qN, device=device), q_idx] = float("inf")

    # kNN in X-space
    knn_dx, knn_j = torch.topk(dx, k=k, dim=1, largest=False)  # (qN,k)

    # Define NN (closest in X among kNN)
    nn_dx = knn_dx[:, 0]          # (qN,)
    nn_j = knn_j[:, 0]            # (qN,)

    # Corresponding future distances to that NN (RMS)
    nn_dy = torch.norm(Yf[q_idx] - Yf[nn_j], dim=1) / sy  # (qN,)

    # (1) Quantile-defined tau_x from empirical NN distribution
    tau_x = float(torch.quantile(nn_dx.detach().float().cpu(), alpha).item())
    tau_y = float(torch.quantile(nn_dy.detach().float().cpu(), 1-alpha).item())

    # Coverage among kNN: does query have any neighbor within tau_x?
    min_k_dx = knn_dx.min(dim=1).values  # (qN,)
    covered_mask = (min_k_dx <= tau_x)
    covered_count = int(covered_mask.sum().item())
    coverage_fraction = covered_count / max(1, qN)

    # Ambiguity test using tau_x (quantile-derived) and given tau_y
    # For each query, check any neighbor in kNN with dx<=tau_x and dy>=tau_y
    examples: List[Dict] = []
    ambiguous_mask_per_query = torch.zeros(qN, device=device, dtype=torch.bool)

    for qi in range(qN):
        i = int(q_idx[qi].item())
        js = knn_j[qi]      # (k,)
        dxi = knn_dx[qi]    # (k,)

        dy = torch.norm(Yf[i] - Yf[js], dim=1) / sy  # (k,)

        close_mask = (dxi <= tau_x)
        amb_mask = close_mask & (dy >= tau_y)

        if amb_mask.any():
            ambiguous_mask_per_query[qi] = True
            sel = torch.nonzero(amb_mask, as_tuple=False).squeeze(1)
            for t in sel.tolist():
                examples.append(
                    {
                        "i": i,
                        "j": int(js[t].item()),
                        "past_rms": float(dxi[t].item()),
                        "future_rms": float(dy[t].item()),
                    }
                )

    ambiguous_count = int(ambiguous_mask_per_query.sum().item())
    ambiguous_fraction = ambiguous_count / max(1, qN)
    ambiguous_given_covered = ambiguous_count / max(1, covered_count)

    examples.sort(key=lambda d: d["future_rms"], reverse=True)

    # (2) Ratio proxy R_i = nn_dy / nn_dx
    ratio = nn_dy / (nn_dx + eps)

    return {
        "N": int(N),
        "queries": int(qN),
        "k": int(k),
        "alpha": float(alpha),
        "tau_x_from_nn_quantile": float(tau_x),
        "tau_y": float(tau_y),
        "dX_flat": int(dX),
        "dY_flat": int(dY),

        # # NN distributions
        # "nn_past_rms_stats": _stats(nn_dx),
        # "nn_future_rms_stats": _stats(nn_dy),

        # # (2) Ratio proxy stats
        # "ratio_stats_R = nn_future_rms / nn_past_rms": _stats(ratio),

        # (1) Quantile-neighborhood ambiguity test
        "covered_count": int(covered_count),
        "coverage_fraction": float(coverage_fraction),
        "ambiguous_count": int(ambiguous_count),
        "ambiguous_fraction": float(ambiguous_fraction),
        "ambiguous_given_covered": float(ambiguous_given_covered),
        "num_pairs_found": int(len(examples)),
    }


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


def load_windows_from_dataset(
    *,
    dataset: str,
    data_dir: Optional[str],
    action_filter: Optional[str],
    input_n: int,
    output_n: int,
    stride: int,
    window_norm: Optional[int],
    normalization: str,
    seed: int,
    max_files: Optional[int],
    max_windows: Optional[int],
) -> Dict[str, torch.Tensor]:
    metadata = get_dataset_metadata(dataset)
    resolved_dir = _resolve_data_dir(dataset, data_dir)
    action = action_filter if action_filter is not None else metadata.get("default_action_filter", "")
    files = _list_dataset_files(resolved_dir, action)
    if not files:
        raise RuntimeError(f"No .npy files found for dataset '{dataset}' in {resolved_dir}.")

    rng = np.random.default_rng(seed)
    if max_files is not None and max_files > 0 and len(files) > max_files:
        rng.shuffle(files)
        files = files[:max_files]

    hand_groups: Tuple[dict, ...] = metadata["hand_groups"]
    window_norm = input_n if window_norm is None else int(window_norm)

    sequences: List[Tuple[np.ndarray, float]] = []
    for f in files:
        try:
            d = np.load(f)
            dataset_lower = dataset.lower()
            if dataset_lower in {"assembly"}:
                d = d.transpose(0, 1, 3, 2).reshape((d.shape[0], d.shape[1], d.shape[2] * d.shape[3]))
                d = np.transpose(d, (1, 2, 0))
            elif dataset_lower == "h2o":
                d = d.transpose(1, 3, 2, 0).reshape(d.shape[1], d.shape[2] * d.shape[3], d.shape[0])

            if d.ndim != 3 or d.shape[-1] != 3 or d.shape[0] < 2:
                continue


            selected = _select_most_active_hand(d, hand_groups, window=window_norm)
            if selected is None:
                continue
            coords, wrist_l = selected
            coords = _normalize_hand(coords, dataset=dataset, wrist_l=wrist_l, mode=normalization)
            # feats = calculate_features(coords)
            # if feats.shape[0] >= 4:
            sequences.append((coords, 1.0))
        except Exception:
            continue

    if not sequences:
        raise RuntimeError("No usable sequences after preprocessing.")

    # Build windows locally to avoid dataset class side effects.
    windows: List[Tuple[torch.Tensor, torch.Tensor]] = []
    in_n = int(input_n)
    out_n = int(output_n)
    stride = max(1, int(stride))
    for seq_arr, _ in sequences:
        T = int(seq_arr.shape[0])
        max_start = T - (in_n + out_n)
        if max_start <= 0:
            continue
        seq_t = torch.from_numpy(seq_arr)
        for st in range(0, max_start, stride):
            a = st
            b = st + in_n
            c = b + out_n
            x = seq_t[a:b, :, :3]  # only positions, ignore velocities
            y = seq_t[b:c, :, :3]
            windows.append((x, y))

    if not windows:
        raise RuntimeError("No windows found for the requested input/output lengths.")

    indices = np.arange(len(windows))
    if max_windows is not None and max_windows > 0 and len(indices) > max_windows:
        indices = rng.choice(indices, size=max_windows, replace=False)

    X_list: List[torch.Tensor] = []
    Y_list: List[torch.Tensor] = []
    for idx in indices.tolist():
        x, y = windows[idx]
        X_list.append(x.reshape(x.shape[0], -1).float())  # (Tin, Dflat)
        Y_list.append(y.reshape(y.shape[0], -1).float())  # (Tout, Dflat)

    return {
        "X": torch.stack(X_list, dim=0),
        "Y": torch.stack(Y_list, dim=0),
        "num_files": torch.tensor(len(files)),
        "num_sequences": torch.tensor(len(sequences)),
        "num_windows": torch.tensor(len(indices)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="assembly")
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--action_filter", type=str, default="pick_up_screwd")

    ap.add_argument("--input_n", type=int, default=90)
    ap.add_argument("--output_n", type=int, default=10)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--time_interp", type=int, default=None)
    ap.add_argument("--window_norm", type=int, default=None)
    ap.add_argument(
        "--normalization",
        type=str,
        default="all_frames",
        choices=("none", "mean_var", "frame0", "frame0_rot", "all_frames", "all_frames_rot"),
        help="none | mean_var | frame0 | frame0_rot | all_frames | all_frames_rot",
    )

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--max_windows", type=int, default=None)

    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max_queries", type=int, default=2000)

    # Point (1): alpha-quantile for tau_x
    ap.add_argument("--alpha", type=float, default=0.3, help="Quantile of NN past distances used as tau_x (0,1)")

    ap.add_argument("--top_m", type=int, default=0)
    args = ap.parse_args()

    loaded = load_windows_from_dataset(
        dataset=args.dataset,
        data_dir=args.data_dir,
        action_filter=args.action_filter,
        input_n=args.input_n,
        output_n=args.output_n,
        stride=args.stride,
        window_norm=args.window_norm,
        normalization=args.normalization,
        seed=args.seed,
        max_files=args.max_files,
        max_windows=args.max_windows,
    )

    out = conditioning_checks_rms(
        loaded["X"],
        loaded["Y"],
        alpha=args.alpha,
        k=args.k,
        max_queries=args.max_queries,
        seed=args.seed,
        device=args.device,
        top_m=args.top_m,
    )

    # attach dataset info
    out["dataset"] = args.dataset
    out["input_n"] = int(args.input_n)
    out["output_n"] = int(args.output_n)
    out["stride"] = int(args.stride)
    out["normalization"] = str(args.normalization)
    out["num_files"] = int(loaded["num_files"].item())
    out["num_sequences"] = int(loaded["num_sequences"].item())
    out["num_windows"] = int(loaded["num_windows"].item())

    import json
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
