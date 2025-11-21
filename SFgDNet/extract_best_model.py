import argparse
import glob
import json
import os
import re
from typing import List, Optional, Tuple

import pandas as pd
from config import DATASET_CHOICES


DEFAULT_RESULTS = os.path.join("out", "ablation", "results_pick_w70_h30.csv")
DEFAULT_BEST_JSON = os.path.join("out", "best_runs", "best_models.json")


def _dataset_file_path(base_path: str, dataset: str) -> str:
    base_dir, base_file = os.path.split(base_path)
    dataset_dir = os.path.join(base_dir, dataset) if base_dir else dataset
    return os.path.join(dataset_dir, base_file)


def _append_window_horizon(path: str, window: int, horizon: int) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_w{window}_h{horizon}{ext}"


def _find_suffixed_variant(base_path: str) -> Optional[str]:
    root, ext = os.path.splitext(base_path)
    pattern = f"{root}_w*_h*{ext}"
    matches = sorted(glob.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    return None


def _parse_window_horizon_from_path(path: str) -> Tuple[Optional[int], Optional[int]]:
    match = re.search(r"_w(\\d+)_h(\\d+)(?=\\.[^./\\\\]+$)", path)
    if not match:
        return None, None
    window, horizon = match.groups()
    return int(window), int(horizon)


def _infer_uniform_value(df: pd.DataFrame, column: str) -> Optional[int]:
    if column not in df.columns:
        return None
    values = df[column].dropna().unique()
    if len(values) != 1:
        return None
    try:
        return int(values[0])
    except (TypeError, ValueError):
        return None


def extract_best_models(csv_path: str, json_base_path: str):
    # Load CSV
    df = pd.read_csv(csv_path)

    window = _infer_uniform_value(df, "input_n")
    horizon = _infer_uniform_value(df, "output_n")
    path_window, path_horizon = _parse_window_horizon_from_path(csv_path)
    if window is None:
        window = path_window
    if horizon is None:
        horizon = path_horizon
    if window is None or horizon is None:
        raise SystemExit(
            "Unable to determine window/horizon from results; ensure CSV contains "
            "input_n/output_n columns or the filename is suffixed with _w*_h*."
        )

    json_path = _append_window_horizon(json_base_path, window, horizon)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

    # Find best configuration per model (lowest test_mpjpe)
    metric_candidates: List[str] = [
        "validation_mpjpe_norm_best",
    ]
    metric_col = next((c for c in metric_candidates if c in df.columns), None)
    if metric_col is None:
        raise SystemExit(
            f"None of expected metric columns present in CSV: {metric_candidates}. Found columns: {list(df.columns)}"
        )
    best = df.loc[df.groupby("model")[metric_col].idxmin()]

    # Convert to dict (one dict per model)
    best_dict = best.set_index("model").to_dict(orient="index")

    # Save to JSON
    with open(json_path, "w") as f:
        json.dump(best_dict, f, indent=2)

    return best_dict, json_path

def main():
    ap = argparse.ArgumentParser(description="Extract best configurations per model from results CSV.")
    ap.add_argument("--dataset", default="assembly", choices=DATASET_CHOICES, help="Dataset name to select paths.")
    ap.add_argument("--csv", default=DEFAULT_RESULTS, help="Base results CSV path. Resolved under dataset directory.")
    ap.add_argument("--json", default=DEFAULT_BEST_JSON, help="Base output JSON path. Resolved under dataset directory.")
    args = ap.parse_args()

    dataset = args.dataset.lower()
    results_csv = _dataset_file_path(args.csv, dataset)
    best_json = _dataset_file_path(args.json, dataset)

    if args.action_filter:
        # Add suffix before extension for action-specific files
        def with_suffix(path: str, suffix: str) -> str:
            root, ext = os.path.splitext(path)
            return f"{root}_{suffix}{ext}"

        results_csv = with_suffix(results_csv, args.action_filter)
        best_json = with_suffix(best_json, args.action_filter)

    if not os.path.exists(results_csv):
        candidate = _find_suffixed_variant(results_csv)
        if candidate is None:
            raise FileNotFoundError(f"Results CSV not found: {results_csv}")
        results_csv = candidate

    best, best_json_path = extract_best_models(results_csv, best_json)
    print("Best configurations saved to:", best_json_path)


if __name__ == "__main__":
    main()
