#!/usr/bin/env python3
"""
Second-order stationarity checks for windowed hand sequences.

Key points:
- Fixes ragged stacking bug by summarizing autocovariance per lag without stacking variable-length tensors.
- Can run on X, Y, or concatenated XY (whole vector timeline).
- Optional landmark/coordinate weighting to avoid dominance of specific landmarks.

Example:
  python second_order_stationary.py --dataset assembly --input_n 90 --output_n 10 --stride 100 \
    --mode XY --landmark_weight equal --max_lag 20 --n_perm 200 --device cuda
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from data import collect_sequences_from_files_wrist_centered, get_dataset_metadata


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
    time_interp: Optional[int],
    window_norm: Optional[int],
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

    hand_groups = metadata["hand_groups"]
    node_count = int(metadata["node_count"])
    window_norm = input_n if window_norm is None else int(window_norm)

    sequences = collect_sequences_from_files_wrist_centered(
        files,
        node_count=node_count,
        hand_groups=hand_groups,
        dataset=dataset,
        time_interp=time_interp,
        window_norm=window_norm,
    )
    if not sequences:
        raise RuntimeError("No usable sequences after preprocessing.")

    dataset_cls = metadata["dataset_cls"]
    windowed = dataset_cls(sequences, input_n=input_n, output_n=output_n, stride=stride)
    if len(windowed) == 0:
        raise RuntimeError("No windows found for the requested input/output lengths.")

    indices = np.arange(len(windowed))
    if max_windows is not None and max_windows > max_windows:
        # dead branch, kept minimal
        pass
    if max_windows is not None and max_windows > 0 and len(indices) > max_windows:
        indices = rng.choice(indices, size=max_windows, replace=False)

    X_list: List[torch.Tensor] = []
    Y_list: List[torch.Tensor] = []
    for idx in indices.tolist():
        x, y, _ = windowed[idx]
        X_list.append(x.reshape(x.shape[0], -1).float())  # (Tin, D)
        Y_list.append(y.reshape(y.shape[0], -1).float())  # (Tout, D)

    return {
        "X": torch.stack(X_list, dim=0),  # (N, Tin, D)
        "Y": torch.stack(Y_list, dim=0),  # (N, Tout, D)
        "num_files": torch.tensor(len(files)),
        "num_sequences": torch.tensor(len(sequences)),
        "num_windows": torch.tensor(len(indices)),
        "node_count": torch.tensor(node_count),
    }


def build_Z(X: torch.Tensor, Y: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "X":
        return X
    if mode == "Y":
        return Y
    if mode == "XY":
        # concatenate along time: (N, Tin+Tout, D)
        return torch.cat([X, Y], dim=1)
    raise ValueError("mode must be one of {X,Y,XY}")


@torch.no_grad()
def apply_weighting(
    Z: torch.Tensor,
    *,
    node_count: int,
    landmark_weight: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Z: (N, T, Dflat) where Dflat = node_count * coord_dim (typically 3)
    landmark_weight:
      - none: unchanged
      - equal: each landmark block has unit RMS over dataset/time (balances landmarks)
      - per_coord: each coordinate standardized over dataset/time (balances all coords)
    """
    if landmark_weight == "none":
        return Z

    N, T, D = Z.shape
    if D % node_count != 0:
        raise ValueError(f"Dflat={D} not divisible by node_count={node_count}.")
    coord_dim = D // node_count

    if landmark_weight == "equal":
        z = Z.view(N, T, node_count, coord_dim)
        # RMS per landmark over (N,T,coord_dim)
        rms = torch.sqrt((z ** 2).mean(dim=(0, 1, 3)) + eps)  # (node_count,)
        w = (1.0 / rms).view(1, 1, node_count, 1)
        z = z * w
        return z.view(N, T, D)

    if landmark_weight == "per_coord":
        # per-coordinate z-score over (N,T)
        mean = Z.mean(dim=(0, 1), keepdim=True)
        std = Z.std(dim=(0, 1), unbiased=False, keepdim=True)
        return (Z - mean) / (std + eps)

    raise ValueError("landmark_weight must be one of {none,equal,per_coord}")


@torch.no_grad()
def timewise_mean_var(Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Z: (N,T,D)
    Returns:
      mean_norm_t: (T,) RMS norm of mu_t
      var_t:       (T,) average per-dim variance
      mu_t:        (T,D)
    """
    mu_t = Z.mean(dim=0)  # (T,D)
    mean_norm_t = torch.sqrt((mu_t ** 2).mean(dim=1) + 1e-12)  # (T,)
    var_t = Z.var(dim=0, unbiased=False).mean(dim=1)  # (T,)
    return mean_norm_t, var_t, mu_t


@torch.no_grad()
def stationarity_score_1d(v: torch.Tensor) -> Dict[str, float]:
    v = v.detach().float()
    m = v.mean()
    s = v.std(unbiased=False)
    ptp = v.max() - v.min()
    return {
        "T": int(v.numel()),
        "mean": float(m.item()),
        "std": float(s.item()),
        "min": float(v.min().item()),
        "max": float(v.max().item()),
        "ptp": float(ptp.item()),
        "rel_ptp": float((ptp / (m.abs() + 1e-12)).item()),
        "rel_std": float((s / (m.abs() + 1e-12)).item()),
    }


@torch.no_grad()
def autocov_lag_summaries(Z: torch.Tensor, max_lag: int) -> Dict[str, List[float]]:
    """
    Returns lag-wise summaries without stacking ragged tensors.

    gamma_lag(t) = E[ <z_t - mu_t, z_{t+lag} - mu_{t+lag}> ] / D, length T-lag.
    For each lag we report:
      mean_over_t, std_over_t, min_over_t, max_over_t
    """
    N, T, D = Z.shape
    _, _, mu_t = timewise_mean_var(Z)
    Zc = Z - mu_t.unsqueeze(0)

    L = min(max_lag, T - 1)
    means, stds, mins, maxs = [], [], [], []
    for lag in range(1, L + 1):
        a = Zc[:, : T - lag, :]   # (N, T-lag, D)
        b = Zc[:, lag:, :]        # (N, T-lag, D)
        g_t = (a * b).sum(dim=2).mean(dim=0) / float(D)  # (T-lag,)

        means.append(float(g_t.mean().item()))
        stds.append(float(g_t.std(unbiased=False).item()))
        mins.append(float(g_t.min().item()))
        maxs.append(float(g_t.max().item()))

    return {
        "L": int(L),
        "mean_over_t": means,
        "std_over_t": stds,
        "min_over_t": mins,
        "max_over_t": maxs,
    }


@torch.no_grad()
def permutation_pvalues(Z: torch.Tensor, max_lag: int, n_perm: int, seed: int) -> Dict[str, float]:
    """
    Observed invariance statistics:
      S_mean = std_t(mean_norm_t)
      S_var  = std_t(var_t)
      S_acov = mean_lag std_t(gamma_lag(t))

    Permute time indices (same permutation for all samples/dims) and compute p-values.
    """
    rng = np.random.default_rng(seed)
    N, T, D = Z.shape

    mean_norm_t, var_t, _ = timewise_mean_var(Z)
    S_mean_obs = mean_norm_t.std(unbiased=False)
    S_var_obs = var_t.std(unbiased=False)

    ac = autocov_lag_summaries(Z, max_lag=max_lag)
    if ac["L"] == 0:
        S_acov_obs = torch.tensor(0.0, device=Z.device)
    else:
        S_acov_obs = torch.tensor(np.mean(ac["std_over_t"]), device=Z.device)

    S_mean_perm = []
    S_var_perm = []
    S_acov_perm = []

    for _ in range(int(n_perm)):
        perm = torch.as_tensor(rng.permutation(T), device=Z.device, dtype=torch.long)
        Zp = Z[:, perm, :]

        mn_t, vv_t, _ = timewise_mean_var(Zp)
        S_mean_perm.append(mn_t.std(unbiased=False))
        S_var_perm.append(vv_t.std(unbiased=False))

        acp = autocov_lag_summaries(Zp, max_lag=max_lag)
        if acp["L"] == 0:
            S_acov_perm.append(torch.tensor(0.0, device=Z.device))
        else:
            S_acov_perm.append(torch.tensor(np.mean(acp["std_over_t"]), device=Z.device))

    S_mean_perm = torch.stack(S_mean_perm) if S_mean_perm else torch.empty(0, device=Z.device)
    S_var_perm = torch.stack(S_var_perm) if S_var_perm else torch.empty(0, device=Z.device)
    S_acov_perm = torch.stack(S_acov_perm) if S_acov_perm else torch.empty(0, device=Z.device)

    def pval(perm_vals: torch.Tensor, obs: torch.Tensor) -> float:
        if perm_vals.numel() == 0:
            return 1.0
        return float(((perm_vals >= obs).float().mean().item()))

    return {
        "S_mean_obs_std_over_t": float(S_mean_obs.item()),
        "S_var_obs_std_over_t": float(S_var_obs.item()),
        "S_acov_obs_meanlag_std_over_t": float(S_acov_obs.item()),
        "p_mean": pval(S_mean_perm, S_mean_obs),
        "p_var": pval(S_var_perm, S_var_obs),
        "p_acov": pval(S_acov_perm, S_acov_obs),
        "n_perm": int(n_perm),
    }


@torch.no_grad()
def report(Z: torch.Tensor, name: str, max_lag: int, n_perm: int, seed: int) -> Dict:
    mean_norm_t, var_t, _ = timewise_mean_var(Z)
    ac = autocov_lag_summaries(Z, max_lag=max_lag)
    p = permutation_pvalues(Z, max_lag=max_lag, n_perm=n_perm, seed=seed)

    ac_summary = {"L": ac["L"]}
    if ac["L"] > 0:
        stds = np.array(ac["std_over_t"], dtype=np.float64)
        means = np.array(ac["mean_over_t"], dtype=np.float64)
        ac_summary.update(
            {
                "lag_mean_mean": float(means.mean()),
                "lag_mean_std": float(means.std()),
                "lag_std_mean": float(stds.mean()),
                "lag_std_max": float(stds.max()),
            }
        )

    return {
        "name": name,
        "N": int(Z.shape[0]),
        "T": int(Z.shape[1]),
        "D": int(Z.shape[2]),
        "mean_norm_t_stats": stationarity_score_1d(mean_norm_t),
        "var_t_stats": stationarity_score_1d(var_t),
        "autocov_summary": ac_summary,
        "permutation_test": p,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="assembly")
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--action_filter", type=str, default=None)

    ap.add_argument("--input_n", type=int, default=90)
    ap.add_argument("--output_n", type=int, default=10)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--time_interp", type=int, default=None)
    ap.add_argument("--window_norm", type=int, default=None)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_files", type=int, default=None)
    ap.add_argument("--max_windows", type=int, default=None)

    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--mode", type=str, default="XY", choices=["X", "Y", "XY"])
    ap.add_argument("--landmark_weight", type=str, default="none", choices=["none", "equal", "per_coord"])

    ap.add_argument("--max_lag", type=int, default=20)
    ap.add_argument("--n_perm", type=int, default=200)

    args = ap.parse_args()

    loaded = load_windows_from_dataset(
        dataset=args.dataset,
        data_dir=args.data_dir,
        action_filter=args.action_filter,
        input_n=args.input_n,
        output_n=args.output_n,
        stride=args.stride,
        time_interp=args.time_interp,
        window_norm=args.window_norm,
        seed=args.seed,
        max_files=args.max_files,
        max_windows=args.max_windows,
    )

    X = loaded["X"].to(args.device)
    Y = loaded["Y"].to(args.device)
    node_count = int(loaded["node_count"].item())

    Z = build_Z(X, Y, mode=args.mode)  # (N, T, D)
    Z = apply_weighting(Z, node_count=node_count, landmark_weight=args.landmark_weight)

    out = {
        "dataset": args.dataset,
        "mode": args.mode,
        "landmark_weight": args.landmark_weight,
        "input_n": int(args.input_n),
        "output_n": int(args.output_n),
        "stride": int(args.stride),
        "num_files": int(loaded["num_files"].item()),
        "num_sequences": int(loaded["num_sequences"].item()),
        "num_windows": int(loaded["num_windows"].item()),
        "max_lag": int(args.max_lag),
        "n_perm": int(args.n_perm),
        "report": report(Z, name=f"Z_{args.mode}", max_lag=args.max_lag, n_perm=args.n_perm, seed=args.seed),
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
