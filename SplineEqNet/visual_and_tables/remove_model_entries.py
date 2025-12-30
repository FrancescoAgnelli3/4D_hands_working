from pathlib import Path
import argparse
import csv
import sys

BASE_OUT_DIR = Path(__file__).resolve().parents[1] / "out"


def collect_csv_targets(target_path: Path) -> list[Path]:
    if target_path.is_file():
        if target_path.suffix.lower() != ".csv":
            raise ValueError(f"File {target_path} is not a CSV file")
        return [target_path]
    if target_path.is_dir():
        return sorted(p for p in target_path.rglob("*.csv") if p.is_file())
    raise FileNotFoundError(f"Path {target_path} does not exist")


def filter_file(path: Path, model_name: str) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        if not fieldnames or "model" not in fieldnames:
            return 0, 0
        rows = list(reader)
    kept_rows = [row for row in rows if row.get("model") != model_name]
    removed = len(rows) - len(kept_rows)
    if removed:
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)
    return removed, len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove rows for a given model from CSV files inside the out directory"
    )
    parser.add_argument("model", help="Model name to remove (case sensitive)")
    parser.add_argument(
        "target",
        nargs="?",
        default=BASE_OUT_DIR,
        help="Path to out directory, a subdirectory, or a specific CSV file (default: SplineEqNet/out)",
    )
    args = parser.parse_args()
    model_name = args.model
    target_path = Path(args.target).expanduser().resolve()
    base_out = BASE_OUT_DIR.resolve()
    try:
        target_path.relative_to(base_out)
    except ValueError:
        if target_path != base_out:
            parser.error(f"Target path {target_path} is outside of {base_out}")
    try:
        csv_targets = collect_csv_targets(target_path)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    total_removed = 0
    processed_files = 0
    for csv_file in csv_targets:
        removed, total_rows = filter_file(csv_file, model_name)
        if total_rows:
            processed_files += 1
        total_removed += removed
    if not csv_targets:
        print("No CSV files found to process.")
        return 1
    print(
        f"Processed {processed_files} files. Removed {total_removed} rows for model '{model_name}'."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
