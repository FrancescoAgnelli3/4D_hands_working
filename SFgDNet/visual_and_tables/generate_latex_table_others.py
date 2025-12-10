#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import pandas as pd

# --- Configuration ---
# Canonical model name mapping (merge variants into paper row names)
MODEL_CANON = {
    "last": "Last",
    "sfgdnet": "SFgDNet",
    "sfgdnet_try": "ourmodel",
    "simlpe": "siMLPe",
    "simlpe_dct": "siMLPe",
    "hisrep": "HisRep",
    "mpt": "MPT",
    "estag": "Estag",
    "hstgcnn": "HSTGCNN",
    "trajdep": "TrajDep",
    "pgbig": "PGBig",
    "motionmixer": "MotionMixer",
    "eqmotion": "EqMotion",
}

# Order of rows in the table (as in the provided LaTeX)
ROW_ORDER = [
    "Last",
    "SFgDNet",
    "siMLPe",
    "ourmodel",
    "HisRep",
    "MPT",
    "Estag",
    "HSTGCNN",
    "TrajDep",
    "PGBig",
    "MotionMixer",
    "EqMotion",
]

# Datasets and splits for the table (LaTeX column order)
DATASET_ORDER = ["H2O", "FPHA", "BigHand2.2M"]

SPLIT_LABELS = ["50/50", "60/40", "70/30", "80/20", "90/10"]

# Mapping with ±1 tolerance around the intended split buckets
SPLIT_BUCKETS = [
    (50, 50, "50/50"),
    (60, 40, "60/40"),
    (70, 30, "70/30"),
    (80, 20, "80/20"),
    (90, 10, "90/10"),
]

# Canonical dataset names from CSV to LaTeX labels
DATASET_CANON = {
    "h2o": "H2O",
    "h20": "H2O",
    "h_2_o": "H2O",
    "fpha": "FPHA",
    "first-person hand action": "FPHA",
    "first person hand action": "FPHA",
    "bighands": "BigHand2.2M",
    "bighand2.2m": "BigHand2.2M",
    "bighand2": "BigHand2.2M",
    "bighand": "BigHand2.2M",
    "bighand2_2m": "BigHand2.2M",
}


def normalize_split_label(input_n: int, output_n: int, n: int = 1) -> str:
    for a, b, label in SPLIT_BUCKETS:
        if abs(int(input_n) - a) <= n and abs(int(output_n) - b) <= n:
            return label
    return f"{int(input_n)}/{int(output_n)}"


def canonical_model(model_name: str) -> str | None:
    key = model_name.strip().lower()
    return MODEL_CANON.get(key)


def canonical_dataset(dataset_name: str) -> str | None:
    key = dataset_name.strip().lower()
    return DATASET_CANON.get(key)


def build_store_from_csv(
    df: pd.DataFrame,
    metric_col: str = "validation_mpjpe_norm_best",
    require_cols: tuple[str, ...] = ("model", "dataset", "input_n", "output_n"),
) -> dict:
    """
    Build a dict: results[(model, dataset_label, split_label)] = best_mpjpe
    Merges multiple variants into the same canonical model by taking the min MPJPE.
    Expects at least columns in `require_cols` and `metric_col`.
    """
    missing = [c for c in (*require_cols, metric_col) if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    store: dict[tuple[str, str, str], float] = {}

    for _, row in df.iterrows():
        # canonical model
        model_raw = str(row["model"])
        model = canonical_model(model_raw)
        if not model:
            continue

        # canonical dataset
        dataset_raw = str(row["dataset"])
        dataset_label = canonical_dataset(dataset_raw)
        if dataset_label not in DATASET_ORDER:
            continue

        # split label
        split_label = normalize_split_label(int(row["input_n"]), int(row["output_n"]))
        if split_label not in SPLIT_LABELS:
            continue

        # metric
        try:
            mpjpe = float(row[metric_col])
        except Exception:
            continue

        key = (model, dataset_label, split_label)
        if key not in store or mpjpe < store[key]:
            store[key] = mpjpe

    return store


def render_three_dataset_table(store: dict, digits: int = 3, scale_h2o: float = 1000.0) -> str:
    r"""
    Render the LaTeX table with highlighting:
    - Best (lowest) value per column in \textbf{bold}
    - Second-best in \underline{}
    Lower is better.
    H2O values are scaled by `scale_h2o` for display only.
    """
    # Prepare column keys in the exact order of the LaTeX table
    col_keys: list[tuple[str, str]] = []
    for dataset in DATASET_ORDER:
        for split in SPLIT_LABELS:
            col_keys.append((dataset, split))

    # Precompute display values (already scaled for H2O) for all cells
    # table_vals[(model, dataset, split)] = float or None
    table_vals: dict[tuple[str, str, str], float] = {}
    for model in ROW_ORDER:
        for dataset in DATASET_ORDER:
            for split in SPLIT_LABELS:
                raw_val = store.get((model, dataset, split))
                if raw_val is None:
                    continue
                disp_val = raw_val * scale_h2o if dataset == "H2O" else raw_val
                table_vals[(model, dataset, split)] = disp_val

    # Row position map to break ties according to ROW_ORDER
    row_pos = {m: i for i, m in enumerate(ROW_ORDER)}

    # Determine best and second-best per column (lower is better)
    best_per_col: dict[tuple[str, str], str] = {}
    second_per_col: dict[tuple[str, str], str] = {}

    for dataset, split in col_keys:
        values_for_col = []
        for model in ROW_ORDER:
            key = (model, dataset, split)
            if key in table_vals:
                values_for_col.append((model, table_vals[key]))

        if not values_for_col:
            continue

        # Sort by value, then by row order index to have deterministic ties
        values_for_col.sort(key=lambda x: (x[1], row_pos[x[0]]))

        best_per_col[(dataset, split)] = values_for_col[0][0]
        if len(values_for_col) > 1:
            second_per_col[(dataset, split)] = values_for_col[1][0]

    # Build LaTeX body lines with highlighting
    lines = []
    fmt = f"{{:.{digits}f}}"
    for model in ROW_ORDER:
        cells = []
        for dataset, split in col_keys:
            key = (model, dataset, split)
            val = table_vals.get(key)
            if val is None:
                cells.append("")
                continue

            base_str = fmt.format(val)
            if best_per_col.get((dataset, split)) == model:
                cell_str = r"\textbf{" + base_str + "}"
            elif second_per_col.get((dataset, split)) == model:
                cell_str = r"\underline{" + base_str + "}"
            else:
                cell_str = base_str
            cells.append(cell_str)

        line = f"{model:<12} & " + " & ".join(cells) + r" \\"
        lines.append(line)

    header = r"""
\begin{table*}[!ht]
\centering
% \scriptsize
\caption{MPJPE (mm) on H2O, FPHA, and BigHand2.2M with added 60/40 and 80/20 splits. The \textbf{best} and \underline{second-best} results per column are highlighted. Lower is better.}
\label{tab:results_three_datasets_competitors}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccccccccccccc}
\hiderowcolors
\toprule
\textbf{Dataset} & \multicolumn{5}{c}{H2O} & \multicolumn{5}{c}{FPHA} & \multicolumn{5}{c}{BigHand2.2M} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-11}\cmidrule(lr){12-16}
\textbf{Split} & 50/50 & 60/40 & 70/30 & 80/20 & 90/10 & 50/50 & 60/40 & 70/30 & 80/20 & 90/10 & 50/50 & 60/40 & 70/30 & 80/20 & 90/10 \\
\midrule
\showrowcolors
""".strip("\n")

    footer = r"""
\bottomrule
\end{tabular}
}
\end{table*}
""".strip("\n")

    body = "\n".join(lines)
    return "\n".join([header, body, footer])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fill 3-dataset LaTeX MPJPE table (H2O, FPHA, BigHand2.2M) from a single CSV, "
            "highlighting best and second-best results per column (lower is better)."
        )
    )
    parser.add_argument(
        "--csv", 
        default="/home/agnelli/projects/4D_hands_working/SFgDNet/out/best_runs/aggregate.csv", 
        help="Path to input CSV (single file with all results)."
    )
    parser.add_argument(
        "--metric-col",
        default="validation_mpjpe_norm_best",
        help="Column name for MPJPE metric (default: validation_mpjpe_norm_best).",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Number of decimal digits for values (default: 3).",
    )
    parser.add_argument(
        "--scale-h2o",
        type=float,
        default=1000.0,
        help="Display scale factor applied to H2O values only (default: 1000.0).",
    )
    parser.add_argument(
        "--out",
        default="results_three_datasets_competitors.txt",
        help="Output path for LaTeX table, or '-' for stdout (default: '-').",
    )

    args = parser.parse_args()

    # Load CSV (single file containing all datasets)
    df = pd.read_csv(args.csv)

    # Build store (min over variants per canonical model/dataset/split)
    store = build_store_from_csv(df, metric_col=args.metric_col)

    # Render LaTeX
    latex = render_three_dataset_table(store, digits=args.digits, scale_h2o=args.scale_h2o)

    # Output
    if args.out == "-" or args.out.strip() == "":
        sys.stdout.write(latex + ("\n" if not latex.endswith("\n") else ""))
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(latex)
        print(f"Wrote table to {out_path}")

if __name__ == "__main__":
    main()
