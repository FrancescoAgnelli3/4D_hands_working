#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


KEEP_COLUMNS = [
    "model",
    "input_n",
    "output_n",
    "validation_mpjpe_norm_best",
    "action",
    "dataset",
]
TIME_SPLIT_RE = re.compile(r"w\d+_h\d+", re.IGNORECASE)


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


def extract_action_and_time(csv_path: Path) -> Tuple[str, str]:
    stem = csv_path.stem
    if not stem.startswith("results_"):
        return "", ""
    suffix = stem[len("results_") :]
    match = TIME_SPLIT_RE.search(suffix)
    time_split = match.group(0).lower() if match else ""
    action_part = suffix[: match.start()].rstrip("_") if match else suffix
    return action_part.lower(), time_split


def available_dataset_names(base_dir: Path) -> List[str]:
    names: List[str] = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("results_*.csv")):
            names.append(child.name)
    return names


def collect_csv_paths(
    base_dir: Path,
    dataset_names: Sequence[str],
    time_splits: Optional[Set[str]],
    action_filter: Optional[Set[str]],
    filename_contains: Optional[str],
) -> List[Tuple[Path, str]]:
    paths: List[Tuple[Path, str]] = []
    for dataset in dataset_names:
        dataset_dir = base_dir / dataset
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise ValueError(f"dataset directory not found: {dataset_dir}")

        for csv_path in sorted(dataset_dir.glob("results_*.csv")):
            action, time_split = extract_action_and_time(csv_path)
            if action_filter is not None and action not in action_filter:
                continue
            if time_splits is not None and time_split not in time_splits:
                continue
            if filename_contains and filename_contains not in csv_path.name:
                continue
            paths.append((csv_path, dataset))
    return paths


def read_rows(csv_path: Path, action_value: str, dataset: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return out

        metric_key = (
            "validation_mpjpe_norm_best"
            if "validation_mpjpe_norm_best" in reader.fieldnames
            else ("val_mpjpe_norm_best" if "val_mpjpe_norm_best" in reader.fieldnames else None)
        )
        if metric_key is None:
            return out

        for row in reader:
            metric = row.get(metric_key)
            val = parse_float(metric)
            if val is None:
                continue

            out.append(
                {
                    "model": (row.get("model") or "").strip(),
                    "input_n": (row.get("input_n") or "").strip(),
                    "output_n": (row.get("output_n") or "").strip(),
                    "validation_mpjpe_norm_best": f"{val:.6f}",
                    "action": action_value,
                    "dataset": dataset,
                }
            )
    return out


def write_csv(rows: Iterable[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=KEEP_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in KEEP_COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize selected best-run CSVs into a single aggregate.csv with the five key columns."
    )
    parser.add_argument(
        "--best-runs-dir",
        type=Path,
        default=Path(""),
        help="Directory that contains dataset subfolders produced by best-run extraction.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset subfolders under best-runs to include. Defaults to all folders containing results_*.csv.",
    )
    parser.add_argument(
        "--time-splits",
        nargs="+",
        help="Optional time splits (e.g. w70_h30) to restrict which CSVs are read."
        "Pass values exactly as they appear in filenames.",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        help="Optional action keys (lowercase) to keep when parsing filenames.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Location to write aggregate.csv. Defaults to best-runs-dir/aggregate.csv.",
    )
    parser.add_argument(
        "--filename-contains",
        help="Optional substring filter; only CSVs whose filename contains this string are aggregated.",
    )
    args = parser.parse_args()

    base_dir = args.best_runs_dir

    if not base_dir.exists() or not base_dir.is_dir():
        print(f"error: best runs directory not found: {base_dir}", file=sys.stderr)
        sys.exit(2)

    dataset_candidates = available_dataset_names(base_dir)
    if not dataset_candidates:
        print(f"error: no dataset folders with results_*.csv found in {base_dir}", file=sys.stderr)
        sys.exit(0)

    dataset_names: List[str]
    if args.datasets:
        invalid = [d for d in args.datasets if d not in dataset_candidates]
        if invalid:
            print(
                f"error: requested dataset folders not found or have no results_*.csv: {', '.join(invalid)}",
                file=sys.stderr,
            )
            sys.exit(2)
        dataset_names = args.datasets
    else:
        dataset_names = dataset_candidates

    time_splits: Optional[Set[str]] = {
        split.lower() for split in args.time_splits
    } if args.time_splits else None
    actions: Optional[Set[str]] = {
        act.lower() for act in args.actions
    } if args.actions else None
    contains = args.filename_contains

    try:
        csv_paths = collect_csv_paths(
            base_dir, dataset_names, time_splits, actions, contains
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    if not csv_paths:
        print("No matching CSV files found.")
        sys.exit(0)

    all_rows: List[Dict[str, str]] = []
    for csv_path, dataset in csv_paths:
        action, _ = extract_action_and_time(csv_path)
        action_label = action
        try:
            rows = read_rows(csv_path, action_value=action_label, dataset=dataset)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"[warn] Skipping {csv_path}: {exc}", file=sys.stderr)
            continue

    if not all_rows:
        print("No rows with a valid 'validation_mpjpe_norm_best' found.")
        sys.exit(0)

    output_path = args.output or (base_dir / "aggregate.csv")
    write_csv(all_rows, output_path)
    print(f"Wrote {len(all_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
