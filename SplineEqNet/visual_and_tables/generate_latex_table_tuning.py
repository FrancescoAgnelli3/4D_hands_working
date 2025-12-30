#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import pandas as pd


def build_grid(df, metric_col):
    """
    Build grids per action:
        store[action][(layer, hidden_size)] = best MPJPE
    Also returns lists of sorted layers and hidden_sizes.
    """

    required = ["hidden_size", "SplineEqNet_num_layers", "action", metric_col]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing required column: {r}")

    df["hidden_size"] = pd.to_numeric(df["hidden_size"], errors="coerce")
    df["SplineEqNet_num_layers"] = pd.to_numeric(
        df["SplineEqNet_num_layers"], errors="coerce"
    )
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=["hidden_size", "SplineEqNet_num_layers", metric_col])

    actions = sorted(df["action"].astype(str).str.lower().unique())
    layers = sorted(df["SplineEqNet_num_layers"].unique())
    hidden_sizes = sorted(df["hidden_size"].unique())

    store = {act: {} for act in actions}

    for act in actions:
        df_a = df[df["action"].astype(str).str.lower() == act]
        for hs in hidden_sizes:
            for nl in layers:
                sub = df_a[(df_a["hidden_size"] == hs) &
                           (df_a["SplineEqNet_num_layers"] == nl)]
                if sub.empty:
                    continue
                store[act][(nl, hs)] = sub[metric_col].min()

    return actions, layers, hidden_sizes, store


def latex_single_action_table(action, layers, hidden_sizes, store_act, digits=3):
    """
    Produces one stand-alone LaTeX table for a single action.
    Rows = layers
    Columns = hidden_size
    """

    fmt = "{:." + str(digits) + "f}"

    # Find the global best and second-best across the entire table
    all_vals = []   # list of (nl, hs, value)
    for (nl, hs), v in store_act.items():
        all_vals.append((nl, hs, float(v)))

    # Sort by metric (ascending)
    all_vals.sort(key=lambda x: x[2])

    # Global best and second-best
    best_nl, best_hs, _ = all_vals[0]

    if len(all_vals) > 1:
        second_nl, second_hs, _ = all_vals[1]
    else:
        second_nl = second_hs = None



    # Column specification
    tabular_cols = "l" + "c" * len(hidden_sizes)

    # Header row
    header = " & " + " & ".join([str(int(hs)) for hs in hidden_sizes]) + r" \\"

    # Body rows
    lines = []
    for nl in layers:
        cells = []
        for hs in hidden_sizes:
            v = store_act.get((nl, hs))
            if v is None:
                cells.append("")
                continue

            s = fmt.format(float(v))

            if nl == best_nl and hs == best_hs:
                s = r"\textbf{" + s + "}"
            elif nl == second_nl and hs == second_hs:
                s = r"\underline{" + s + "}"



            cells.append(s)

        lines.append(f"{int(nl)} & " + " & ".join(cells) + r" \\")

    table = rf"""
% =============================
% Action: {action}
% =============================
\begin{{table}}[!ht]
\centering
\scriptsize
\caption{{Parameter tuning results for action \texttt{{{action}}} (MPJPE, mm). Lower is better.}}
\label{{tab:param_tuning_{action}}}

\begin{{tabular}}{{{tabular_cols}}}
\toprule
\textbf{{Layers}} {header}
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}

\end{{table}}
""".strip("\n")

    return table


def main():
    parser = argparse.ArgumentParser(
        description="Create separate parameter tuning tables (layers × hidden_size) for each action."
    )
    parser.add_argument("--csv", required=True, help="Aggregated CSV file.")
    parser.add_argument("--model", default="", help="Optional model filter.")
    parser.add_argument(
        "--metric-col",
        default="validation_mpjpe_norm_best",
        help="Metric column (default: validation_mpjpe_norm_best)."
    )
    parser.add_argument("--digits", type=int, default=3, help="Decimal digits.")
    parser.add_argument("--out", default="results_tuning.txt", help="Output file or '-' for stdout.")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if args.model.strip():
        df = df[df["model"].astype(str).str.lower() == args.model.lower()]

    actions, layers, hidden_sizes, store = build_grid(df, metric_col=args.metric_col)

    blocks = []
    for act in actions:
        block = latex_single_action_table(act, layers, hidden_sizes, store[act], digits=args.digits)
        blocks.append(block)

    final_output = "\n\n".join(blocks)

    if args.out == "-" or args.out.strip() == "":
        sys.stdout.write(final_output + ("\n" if not final_output.endswith("\n") else ""))
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(final_output, encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
