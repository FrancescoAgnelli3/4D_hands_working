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
    "ourmodel",
    "siMLPe",
    "HisRep",
    "MPT",
    "Estag",
    "HSTGCNN",
    "TrajDep",
    "PGBig",
    "MotionMixer",
    "EqMotion",
]

# Column groups for the two subtables
TABLE1_ACTIONS = ["put", "rotate", "unscrew"]
TABLE2_ACTIONS = ["attempt", "remove", "pick"]

# Target split labels in the LaTeX table
SPLIT_LABELS = ["50/50", "70/30", "90/10"]

# Mapping with ±1 tolerance around the intended split buckets
SPLIT_BUCKETS = [
    (50, 50, "50/50"),
    (70, 30, "70/30"),
    (90, 10, "90/10"),
]


def normalize_split_label(input_n: int, output_n: int, n: int = 1) -> str:
    for a, b, label in SPLIT_BUCKETS:
        if abs(int(input_n) - a) <= n and abs(int(output_n) - b) <= n:
            return label
    # Fallback to explicit fraction if not matching desired buckets
    return f"{int(input_n)}/{int(output_n)}"


def extract_action_suffix(action_field: str) -> str:
    """
    Example: 'cross_action_train-..._withheld-rotate' -> 'rotate'
             'rotate' -> 'rotate'
    """
    return action_field.split("-")[-1].strip().lower()


def canonical_model(model_name: str) -> str | None:
    key = model_name.strip().lower()
    return MODEL_CANON.get(key)


def best_mpjpe_by_key(
    df: pd.DataFrame,
    metric_col: str = "validation_mpjpe_norm_best",
    dataset_filter: str | None = "assembly",
    action_col: str = "action",
    input_n_col: str = "input_n",
    output_n_col: str = "output_n",
    config_col: str | None = "configuration",
) -> dict:
    """
    Build a dict: results[(model, action, split_label)] = best_mpjpe
    Merges multiple variants into the same canonical model by taking the min MPJPE.
    Works from a single CSV that contains everything.
    """
    # Optional dataset filter if a 'dataset' column exists
    if dataset_filter is not None and "dataset" in df.columns:
        df = df[df["dataset"].astype(str).str.strip().str.lower() == dataset_filter.lower()].copy()

    required = [action_col, input_n_col, output_n_col, "model", metric_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    store: dict[tuple[str, str, str], float] = {}
    has_config = bool(config_col) and config_col in df.columns
    per_config_best: dict[tuple[str, str, str, str], float] = {}

    for _, row in df.iterrows():
        # canonical model
        model_raw = str(row["model"])
        model = canonical_model(model_raw)
        if not model:
            continue

        # action suffix
        action = extract_action_suffix(str(row[action_col]))

        # split label with tolerance
        split_label = normalize_split_label(int(row[input_n_col]), int(row[output_n_col]))

        # only collect values for the three standard splits targeted in the LaTeX
        if split_label not in SPLIT_LABELS:
            continue

        # metric
        try:
            mpjpe = float(row[metric_col])
        except Exception:
            continue

        if has_config:
            raw_cfg = row.get(config_col)
            if isinstance(raw_cfg, str):
                config_val = raw_cfg.strip() or None
            elif pd.notna(raw_cfg):
                config_val = str(raw_cfg)
            else:
                config_val = None
            cfg_key = (model, action, split_label, config_val or "__NO_CFG__")
            prev_cfg_best = per_config_best.get(cfg_key)
            if prev_cfg_best is not None and mpjpe >= prev_cfg_best:
                continue
            per_config_best[cfg_key] = mpjpe

        key = (model, action, split_label)
        if key not in store or mpjpe < store[key]:
            store[key] = mpjpe
    return store


def render_table_block(actions: list[str], store: dict, digits: int = 3) -> str:
    r"""
    Render one 3-action LaTeX table block:
    - 1 label column + 9 numeric columns (3 actions × 3 splits).
    - Best per column in \\textbf{}, second-best in \\underline{} (lower is better).
    """
    # Header: action names and cmidrule lines
    action_header = " & ".join([f"\\multicolumn{{3}}{{c}}{{{a}}}" for a in actions])
    cmid = " ".join([f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(actions))])

    # Column specification: 1 label + 9 numeric columns
    tabular_cols = "l" + "c" * 9  # -> lccccccccc

    # Precompute per-column rankings (per action, split)
    # Gather values for each column across models
    row_pos = {m: i for i, m in enumerate(ROW_ORDER)}  # deterministic tie-break
    best: dict[tuple[str, str], str] = {}
    second: dict[tuple[str, str], str] = {}
    for action in actions:
        for split in SPLIT_LABELS:
            vals = []
            for model in ROW_ORDER:
                v = store.get((model, action, split))
                if v is not None:
                    vals.append((model, v))
            if not vals:
                continue
            vals.sort(key=lambda x: (x[1], row_pos[x[0]]))  # lower is better
            best[(action, split)] = vals[0][0]
            if len(vals) > 1:
                second[(action, split)] = vals[1][0]

    # Build rows with highlighting
    fmt = "{:." + str(digits) + "f}"
    lines = []
    for model in ROW_ORDER:
        cells = []
        for action in actions:
            for split in SPLIT_LABELS:
                val = store.get((model, action, split))
                if val is None:
                    cells.append("")
                    continue
                s = fmt.format(val)
                if best.get((action, split)) == model:
                    s = r"\textbf{" + s + "}"
                elif second.get((action, split)) == model:
                    s = r"\underline{" + s + "}"
                cells.append(s)
        line = f"{model:<12} & " + " & ".join(cells) + r" \\"
        lines.append(line)

    block = rf"""
\resizebox{{\textwidth}}{{!}}{{
\begin{{tabular}}{{{tabular_cols}}}
\hiderowcolors
\toprule
\textbf{{Action}} & {action_header}  \\
{cmid}
\textbf{{Split}} & 50/50 & 70/30 & 90/10 & 50/50 & 70/30 & 90/10 & 50/50 & 70/30 & 90/10 \\
\midrule
\showrowcolors
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
}}
""".strip("\n")
    return block


def render_full_table(
    store: dict,
    digits: int = 3,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    block1 = render_table_block(TABLE1_ACTIONS, store, digits=digits)
    block2 = render_table_block(TABLE2_ACTIONS, store, digits=digits)

    caption = (
        caption
        or "Comparison on the AssemblyHands dataset (MPJPE, mm). The \\textbf{best} and \\underline{second-best} results per column are highlighted. Lower is better."
    )
    label = label or "tab:assembly_results_competitors"

    full = rf"""
\begin{{table*}}[!ht]
\centering
\scriptsize
\caption{{{caption}}}
\label{{{label}}}

% ---- FIRST SUBTABLE ----
{block1}

\vspace{{0.1em}}

% ---- SECOND SUBTABLE ----
{block2}
\end{{table*}}
""".strip("\n")
    return full


def main():
    parser = argparse.ArgumentParser(
        description="Fill AssemblyHands LaTeX MPJPE tables from a single CSV; highlight best and second-best per column."
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
        "--dataset",
        default="assembly",
        help="Dataset filter value in 'dataset' column (default: 'assembly'). Set '' to disable filtering.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Decimal digits in table cells (default: 3).",
    )
    parser.add_argument(
        "--cross-action",
        action="store_true",
        help="Use the cross-action caption/label variant for the AssemblyHands results table.",
    )
    parser.add_argument(
        "--out",
        default="results_assembly_competitors.txt",
        help="Output path for LaTeX table, or '-' for stdout (default: '-').",
    )
    # Optional column name overrides
    parser.add_argument("--action-col", default="action", help="Column name for action (default: action).")
    parser.add_argument("--input-n-col", default="input_n", help="Column name for input horizon (default: input_n).")
    parser.add_argument("--output-n-col", default="output_n", help="Column name for output horizon (default: output_n).")
    parser.add_argument(
        "--config-col",
        default="configuration",
        help=(
            "Optional configuration identifier column. If multiple configurations exist for the same cell, "
            "the configuration with the lowest MPJPE is used (default: configuration)."
        ),
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    dataset_filter = args.dataset if args.dataset.strip() else None

    store = best_mpjpe_by_key(
        df,
        metric_col=args.metric_col,
        dataset_filter=dataset_filter,
        action_col=args.action_col,
        input_n_col=args.input_n_col,
        output_n_col=args.output_n_col,
        config_col=(args.config_col.strip() or None),
    )
    caption = None
    label = None
    if args.cross_action:
        caption = (
            "Cross--action results on the AssemblyHands dataset under a leave--one--action--out protocol. Lower is better. "
            "The best and second--best values per column are highlighted in bold and underlined, respectively."
        )
        label = "tab:cross_assembly_results_competitors"

    latex = render_full_table(store, digits=args.digits, caption=caption, label=label)

    if args.out == "-" or args.out.strip() == "":
        sys.stdout.write(latex + ("\n" if not latex.endswith("\n") else ""))
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(latex, encoding="utf-8")
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
