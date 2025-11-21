#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Directory containing the CSVs to aggregate
BASE_DIR = Path("")
# Output CSV path
OUTPUT_CSV = BASE_DIR / "aggregate_simlpe_architecture.csv"

# Columns to keep (order matters)
FINAL_COLUMNS = [
    "validation_mpjpe_norm_best",
    "action",  # added: parsed from filename like results_w70_h30_pick.csv
    "ablation",
    "ablation_desc",
    "ablation_groups",
    "SFgDNet_padding_mode",
    "SFgDNet_padding_predictor",
    "SFgDNet_padding_full_horizon",
    "SFgDNet_force_mlp_padding",
    "SFgDNet_skip_dct",
    "SFgDNet_learn_lambda",
    "SFgDNet_learn_centers",
    "SFgDNet_fixed_lambda",
    "SFgDNet_fixed_centers",
    "SFgDNet_fixed_mask",
    "SFgDNet_disable_slow_branch",
    "SFgDNet_disable_fast_branch",
    "sfgdnet_try_padding_mode",
    "sfgdnet_try_padding_predictor",
    "sfgdnet_try_padding_full_horizon",
    "sfgdnet_try_force_mlp_padding",
    "sfgdnet_try_skip_dct",
    "sfgdnet_try_learn_lambda",
    "sfgdnet_try_learn_centers",
    "sfgdnet_try_fixed_lambda",
    "sfgdnet_try_fixed_centers",
    "sfgdnet_try_fixed_mask",
    "sfgdnet_try_disable_slow_branch",
    "sfgdnet_try_disable_fast_branch",
]

def parse_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    x = x.strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None

def parse_action_from_filename(csv_path: Path) -> str:
    """
    Extract the action token from a filename like:
      results_w70_h30_pick.csv  -> 'pick'
      results_pick.csv          -> 'pick'
    Returns '' if not found.
    """
    stem = csv_path.stem  # e.g., 'results_w70_h30_pick'
    if not stem.startswith("results_"):
        return ""
    tokens = stem.split("_")[1:]  # drop 'results'
    if not tokens:
        return ""
    candidate = tokens[-1].strip().lower()
    # Accept alphabetic actions like 'pick', 'place', 'push'. Otherwise empty.
    return candidate if candidate.isalpha() else ""

def collect_all_rows(base_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for csv_path in sorted(base_dir.rglob("*.csv")):
        if csv_path.resolve() == OUTPUT_CSV.resolve():
            continue
        if not (csv_path.name.startswith("results_") and csv_path.suffix == ".csv"):
            continue

        try:
            with csv_path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                if not reader.fieldnames:
                    continue
                action = parse_action_from_filename(csv_path)
                for row in reader:
                    metric = row.get("validation_mpjpe_norm_best") or row.get("val_mpjpe_norm_best")
                    val_metric = parse_float(metric)
                    if val_metric is None:
                        continue

                    out_row = {c: "" for c in FINAL_COLUMNS}
                    out_row["validation_mpjpe_norm_best"] = f"{val_metric:.6f}"
                    out_row["action"] = action

                    # Copy through known columns if present
                    for c in FINAL_COLUMNS:
                        if c in ("validation_mpjpe_norm_best", "action"):
                            continue
                        v = row.get(c)
                        out_row[c] = v.strip() if isinstance(v, str) else (v if v is not None else "")
                    rows.append(out_row)
        except Exception as e:
            import sys
            print(f"[warn] Skipping {csv_path}: {e}", file=sys.stderr)
            continue
    return rows

def write_csv(rows: Iterable[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FINAL_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in FINAL_COLUMNS})

def main() -> None:
    rows = collect_all_rows(BASE_DIR)
    if not rows:
        print("No rows found with a valid 'validation_mpjpe_norm_best'.")
        return
    write_csv(rows, OUTPUT_CSV)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
