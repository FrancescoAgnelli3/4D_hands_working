import argparse
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd

COLS_TO_DROP: List[str] = [
    "gru_layers",
    "t_layers",
    "gnn_layers",
    "hisrep_hand_dct_n",
    "hisrep_dct_n",
    "simlpe_norm_axis",
    "simlpe_use_norm",
    "simlpe_use_spatial_fc_only",
    "simlpe_mix_spatial_temporal",
    "simlpe_add_last_offset",
    "simlpe_mask_temperature",
    "simlpe_slow_layers",
    "simlpe_fast_layers",
    "simlpe_mid_gate_max",
]

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "out"
DEFAULT_SUFFIX = ""


def _find_csv_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory '{root}' does not exist.")
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def _clean_file(csv_path: Path, suffix: str) -> Path:
    df = pd.read_csv(csv_path)
    drop_candidates = [column for column in COLS_TO_DROP if column in df.columns]
    if drop_candidates:
        df = df.drop(columns=drop_candidates)

    output_path = csv_path.with_name(f"{csv_path.stem}{suffix}{csv_path.suffix}") if suffix else csv_path
    df.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drop unused columns from CSV files recursively.")
    parser.add_argument(
        "--root",
        "-r",
        default=DEFAULT_ROOT,
        help=f"Directory to scan recursively for CSV files (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--suffix",
        "-s",
        default=DEFAULT_SUFFIX,
        help="Suffix appended to cleaned CSV filenames. Default overwrites original files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser()
    cleaned_files = []

    for csv_file in _find_csv_files(root):
        output_path = _clean_file(csv_file, args.suffix)
        cleaned_files.append(output_path)
        print(f"Cleaned '{csv_file}' -> '{output_path}'")

    if not cleaned_files:
        print(f"No CSV files found under '{root}'.")


if __name__ == "__main__":
    main()
