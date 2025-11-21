#!/usr/bin/env python3
import argparse
import pandas as pd
import re
from collections import defaultdict

# --- Configuration ---
# Canonical model name mapping (merge variants into paper row names)
MODEL_CANON = {
    "last": "Last",
    "SFgDNet": "SFgDNet",
    "sfgdnet_try": "SFgDNet-Try",
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

# Parameter counts to print in the last column
PARAMS = {
    "Last": 0.00,
    "SFgDNet": 0.31,
    "SFgDNet-Try": 0.31,
    "siMLPe": 0.32,
    "HisRep": 0.39,
    "MPT": 3.23,
    "Estag": 0.02,
    "HSTGCNN": 0.03,
    "TrajDep": 0.19,
    "PGBig": 2.05,
    "MotionMixer": 0.10,
    "EqMotion": 0.12,
}

# Order of rows in the table
ROW_ORDER = [
    "Last",
    "SFgDNet",
    "SFgDNet-Try",
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

# Column groups for the two tables
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


def normalize_split_label(input_n: int, output_n: int, n = 1) -> str:
    for a, b, label in SPLIT_BUCKETS:
        if abs(int(input_n) - a) <= n and abs(int(output_n) - b) <= n:
            return label
    # Fallback to explicit fraction if not matching desired buckets
    return f"{int(input_n)}/{int(output_n)}"


def extract_action_suffix(action_field: str) -> str:
    # Use the last token after '_' as specified
    # Example: "cross_action_train-..._withheld-rotate" -> "rotate"
    return action_field.split("-")[-1].strip().lower()


def canonical_model(model_name: str) -> str | None:
    key = model_name.strip().lower()
    return MODEL_CANON.get(key)


def best_mpjpe_by_key(df: pd.DataFrame,
                      dataset_filter: str | None = "assembly") -> dict:
    """
    Build a dict: results[(model, action, split_label)] = best_mpjpe
    Merges multiple variants into the same canonical model by taking the min MPJPE.
    """
    if dataset_filter is not None and "dataset" in df.columns:
        df = df[df["dataset"].str.strip().str.lower() == dataset_filter.lower()].copy()

    store = {}
    for _, row in df.iterrows():
        model_raw = str(row["model"])
        model = canonical_model(model_raw)
        if not model:
            continue

        # action suffix
        action = extract_action_suffix(str(row["action"]))

        # split label with tolerance
        split_label = normalize_split_label(int(row["input_n"]), int(row["output_n"]))
        
        # only collect values for the three standard splits targeted in the LaTeX
        if split_label not in SPLIT_LABELS:
            # ignore off-bucket splits so the LaTeX stays aligned
            continue

        mpjpe = float(row["validation_mpjpe_norm_best"])

        key = (model, action, split_label)
        if key not in store or mpjpe < store[key]:
            store[key] = mpjpe
    return store


def render_table_block(actions: list[str],
                       store: dict) -> str:
    """
    Render one 3-action LaTeX table block matching the user’s template.
    """
    # Header: action names and cmidrule lines
    action_header = " & ".join([f"\\multicolumn{{3}}{{c}}{{{a}}}" for a in actions])
    cmid = " ".join([f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(actions))])

    # Build rows
    lines = []
    for model in ROW_ORDER:
        cells = []
        for action in actions:
            for split in SPLIT_LABELS:
                val = store.get((model, action, split))
                cells.append(f"{val:.3f}" if val is not None else "")
        # params
        params = PARAMS.get(model, 0.0)
        line = f"{model:<12} & " + " & ".join(cells) + f" & {params:.3f} \\\\"
        lines.append(line)

    block = rf"""
\resizebox{{\textwidth}}{{!}}{{
\begin{{tabular}}{{lcccccccccccc}}
\hiderowcolors
\toprule
\textbf{{Action}} & {action_header} & {{\textbf{{\#Params (M)}}}} \\
{cmid}
\textbf{{Split}} & 50/50 & 70/30 & 90/10 & 50/50 & 70/30 & 90/10 & 50/50 & 70/30 & 90/10 &  \\
\midrule
\showrowcolors
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
}}
""".strip("\n")
    return block


def render_full_table(store: dict) -> str:
    block1 = render_table_block(TABLE1_ACTIONS, store)
    block2 = render_table_block(TABLE2_ACTIONS, store)

    full = rf"""
\begin{{table*}}[!ht]
\centering
\scriptsize

{block1}

\vspace{{1em}}

{block2}

\caption{{Comparison on the AssemblyHands dataset.}}
\label{{tab:assembly_results_competitors}}
\end{{table*}}
""".strip("\n")
    return full


def main():
    parser = argparse.ArgumentParser(description="Fill LaTeX MPJPE table from CSV.")
    parser.add_argument("--csv", required=True, help="Path to input CSV.")
    parser.add_argument("--out", default="assembly_results_competitors.txt",
                        help="Path to output .txt containing LaTeX.")
    parser.add_argument("--dataset", default="assembly",
                        help="Dataset filter, default 'assembly'. Set '' to disable.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    dataset_filter = args.dataset if args.dataset.strip() else None

    store = best_mpjpe_by_key(df, dataset_filter=dataset_filter)
    latex = render_full_table(store)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
