#!/usr/bin/env python3
"""
Oracle Nearest-Neighbor evaluation for multimodal futures.

Goal
----
Compute the "oracle NN" baseline described earlier:
for each test window x*, pick the closest training window x_i (in past-space),
use its ground-truth future y_i as the prediction, and evaluate MPJPE against y*.

This script follows the same MPJPE definition used in train_utils.py:
  mpjpe_terms = torch.norm(pred - tgt_3d, dim=-1)     # (B, T_out, N)
  per_sample_mpjpe = mpjpe_terms.mean(dim=(1, 2))     # (B,)
  per_sample_mpjpe_norm = per_sample_mpjpe * norm_factor_t

Assumptions (match your pipeline)
---------------------------------
- DataLoader batches are tuples/lists with:
    inp, out, norm_factor(optional)
- Feature layout is the same as in your code:
    coords are at [:, :, :, 4:] (3D), and scalar "dist" at [:, :, :, 3].
- build_loaders(ds_cfg) is available (you provided it), and uses your dataset classes.

Usage
-----
Option A (recommended): import and call `evaluate_oracle_nn(ds_cfg)` from your code.

Option B: run as a script and pass dataset parameters as CLI args.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple
import torch

# Allow running as a script from this subpackage.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Your project imports
from config import DatasetCfg
from data import build_loaders, get_dataset_metadata


@torch.no_grad()
def _extract_past_and_future_from_batch(
    batch,
    device: torch.device,
    use_full_features_for_distance: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      past_flat: (B, D) float32
      tgt_3d:     (B, T_out, N, 3) float32
      norm_factor_t: (B,) float32
    """
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise TypeError("Expected batch to be (inp, out, [norm_factor]).")

    inp, out = batch[0], batch[1]
    norm_factor = batch[2] if len(batch) > 2 else None

    inp = inp.to(device).float()
    out = out.to(device).float()

    if norm_factor is not None:
        norm_factor_t = norm_factor.to(device=device).float().view(-1)
    else:
        norm_factor_t = torch.ones(inp.size(0), device=device)

    # Coordinates (exactly like train_utils.py)
    in_3d = inp[:, :, :, 4:]     # (B, T_in, N, 3)
    tgt_3d = out[:, :, :, 4:]    # (B, T_out, N, 3)

    if use_full_features_for_distance:
        # flatten all features of the input window, not just coords
        past = inp
    else:
        # default: use only coords for NN search
        past = in_3d

    past_flat = past.reshape(past.shape[0], -1).contiguous()  # (B, D)

    return past_flat, tgt_3d.contiguous(), norm_factor_t


@torch.no_grad()
def _mpjpe_per_sample(pred_3d: torch.Tensor, tgt_3d: torch.Tensor) -> torch.Tensor:
    """
    pred_3d, tgt_3d: (B, T_out, N, 3)
    Returns per-sample MPJPE: (B,)
    """
    mpjpe_terms = torch.norm(pred_3d - tgt_3d, dim=-1)      # (B, T_out, N)
    return mpjpe_terms.mean(dim=(1, 2))                     # (B,)


@torch.no_grad()
def _nearest_neighbor_indices(
    query: torch.Tensor,          # (Bq, D) on device
    database: torch.Tensor,       # (Bd, D) on device
    chunk_db: int = 16384,
) -> torch.Tensor:
    """
    Brute-force exact 1-NN indices with chunking over the database.
    Returns indices into database: (Bq,)

    Distance: squared L2.
    """
    if query.ndim != 2 or database.ndim != 2:
        raise ValueError("query and database must be 2D tensors (B, D).")
    if query.size(1) != database.size(1):
        raise ValueError("query and database must have the same feature dimension D.")

    Bq, D = query.shape
    Bd = database.shape[0]

    # Keep best
    best_dist = torch.full((Bq,), float("inf"), device=query.device)
    best_idx = torch.zeros((Bq,), dtype=torch.long, device=query.device)

    # Precompute query norms
    q_norm = (query * query).sum(dim=1)  # (Bq,)

    # Chunked: dist^2(q, x) = ||q||^2 + ||x||^2 - 2 q·x
    for start in range(0, Bd, chunk_db):
        end = min(Bd, start + chunk_db)
        x = database[start:end]  # (C, D)
        x_norm = (x * x).sum(dim=1)  # (C,)

        # (Bq, C)
        dots = query @ x.t()
        d2 = q_norm[:, None] + x_norm[None, :] - 2.0 * dots

        # best within chunk
        chunk_best_dist, chunk_best_pos = torch.min(d2, dim=1)  # (Bq,), (Bq,)
        improved = chunk_best_dist < best_dist
        if improved.any():
            best_dist[improved] = chunk_best_dist[improved]
            best_idx[improved] = chunk_best_pos[improved] + start

    return best_idx


@torch.no_grad()
def evaluate_oracle_nn(
    ds_cfg: DatasetCfg,
    *,
    device: Optional[str] = None,
    use_full_features_for_distance: bool = False,
    db_chunk: int = 16384,
    max_train_windows: Optional[int] = None,
    max_test_windows: Optional[int] = None,
) -> dict:
    """
    Builds train/val/test loaders from ds_cfg, constructs a train window database,
    and evaluates oracle 1-NN MPJPE on the test set.

    Returns a dict with:
      mpjpe, mpjpe_norm, n_test
    """
    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    # Build loaders with your pipeline (exactly as in data.py)
    # Note: build_loaders uses ds_cfg.batch_size and ds_cfg.eval_batch_mult.
    train_loader, _, test_loader = build_loaders(ds_cfg)

    # --- build training database: past_flat_db and future_db (for retrieval) ---
    past_db_list = []
    fut_db_list = []

    n_seen = 0
    for batch in train_loader:
        past_flat, tgt_3d, _ = _extract_past_and_future_from_batch(
            batch, device_t, use_full_features_for_distance
        )
        past_db_list.append(past_flat)
        fut_db_list.append(tgt_3d)

        n_seen += past_flat.size(0)
        if max_train_windows is not None and n_seen >= int(max_train_windows):
            break

    past_db = torch.cat(past_db_list, dim=0)
    fut_db = torch.cat(fut_db_list, dim=0)

    if max_train_windows is not None and past_db.size(0) > int(max_train_windows):
        past_db = past_db[: int(max_train_windows)]
        fut_db = fut_db[: int(max_train_windows)]

    # --- evaluate on test: for each query, retrieve NN future and compute MPJPE ---
    total_mpjpe = 0.0
    total_mpjpe_norm = 0.0
    total_count = 0

    n_seen_test = 0
    for batch in test_loader:
        past_q, tgt_q, norm_factor_t = _extract_past_and_future_from_batch(
            batch, device_t, use_full_features_for_distance
        )

        idx = _nearest_neighbor_indices(past_q, past_db, chunk_db=int(db_chunk))
        pred_q = fut_db.index_select(0, idx)  # (B, T_out, N, 3)

        per_sample = _mpjpe_per_sample(pred_q, tgt_q)  # (B,)
        per_sample_norm = per_sample * norm_factor_t

        total_mpjpe += float(per_sample.sum().item())
        total_mpjpe_norm += float(per_sample_norm.sum().item())
        total_count += int(per_sample.numel())

        n_seen_test += int(per_sample.numel())
        if max_test_windows is not None and n_seen_test >= int(max_test_windows):
            break

    if total_count == 0:
        raise RuntimeError("No test windows evaluated (empty test loader?).")

    return {
        "mpjpe": total_mpjpe / total_count,
        "mpjpe_norm": total_mpjpe_norm / total_count,
        "n_test": total_count,
        "n_train_db": int(past_db.size(0)),
        "distance_space": "full_features" if use_full_features_for_distance else "coords_only",
        "device": str(device_t),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="assembly", choices=["assembly", "h2o", "bighands", "fpha"])
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--action_filter", type=str, default="pick_up_screwd")
    p.add_argument("--input_n", type=int, default=90)
    p.add_argument("--output_n", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time_interp", type=int, default=None)
    p.add_argument("--window_norm", type=int, default=None)
    p.add_argument("--eval_batch_mult", type=int, default=1)

    p.add_argument("--use_full_features_for_distance", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--db_chunk", type=int, default=16384)

    p.add_argument("--max_train_windows", type=int, default=None)
    p.add_argument("--max_test_windows", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    metadata = get_dataset_metadata(args.dataset)
    data_dir = args.data_dir if args.data_dir is not None else metadata["default_dir"]
    action_filter = args.action_filter if args.action_filter is not None else metadata["default_action_filter"]

    # Construct DatasetCfg. If your DatasetCfg has different field names,
    # update this section accordingly.
    ds_cfg = DatasetCfg(
        dataset=args.dataset,
        data_dir=str(data_dir),
        action_filter=str(action_filter),
        input_n=int(args.input_n),
        output_n=int(args.output_n),
        batch_size=int(args.batch_size),
        stride=int(args.stride),
        seed=int(args.seed),
        time_interp=(None if args.time_interp is None or int(args.time_interp) <= 0 else int(args.time_interp)),
        window_norm=(None if args.window_norm is None or int(args.window_norm) <= 0 else int(args.window_norm)),
        eval_batch_mult=int(args.eval_batch_mult),
    )

    res = evaluate_oracle_nn(
        ds_cfg,
        device=args.device,
        use_full_features_for_distance=bool(args.use_full_features_for_distance),
        db_chunk=int(args.db_chunk),
        max_train_windows=args.max_train_windows,
        max_test_windows=args.max_test_windows,
    )

    print("Oracle 1-NN evaluation")
    print(f"  distance_space: {res['distance_space']}")
    print(f"  n_train_db:     {res['n_train_db']}")
    print(f"  n_test:         {res['n_test']}")
    print(f"  MPJPE:          {res['mpjpe']:.6f}")
    print(f"  MPJPE_norm:     {res['mpjpe_norm']:.6f}")
    print(f"  device:         {res['device']}")


if __name__ == "__main__":
    main()
