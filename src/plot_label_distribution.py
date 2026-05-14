"""
plot_label_distribution.py
Shows positive rate per head and top-10 most changed labels.
Run:
    python plot_label_distribution.py --pairs artifacts/train_pairs_with_delta.csv
Saves: images/fig_label_dist.pdf
"""

import ast, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="artifacts/train_pairs_with_delta.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.pairs, low_memory=False)
    for c in ["delta_new", "delta_resolved", "delta_stable_present", "label_cols"]:
        df[c] = df[c].apply(parse_list)

    label_cols = df.iloc[0]["label_cols"]
    K = len(label_cols)

    new = np.array(df["delta_new"].tolist(),            dtype=np.int32)
    res = np.array(df["delta_resolved"].tolist(),       dtype=np.int32)
    stp = np.array(df["delta_stable_present"].tolist(), dtype=np.int32)

    # Positive counts per label per head
    cnt_new = new.sum(axis=0)
    cnt_res = res.sum(axis=0)
    cnt_stp = stp.sum(axis=0)

    # Pretty label names (strip y_ prefix)
    pretty = [c.replace("y_", "").replace("_", " ") for c in label_cols]

    # Sort by total positive count
    total = cnt_new + cnt_res + cnt_stp
    order = np.argsort(-total)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(K)
    w = 0.28

    ax.bar(x[order] - w, cnt_new[order], w, label="NEW",             color="#4C72B0", zorder=3)
    ax.bar(x[order],     cnt_res[order], w, label="RESOLVED",         color="#55A868", zorder=3)
    ax.bar(x[order] + w, cnt_stp[order], w, label="STABLE\_PRESENT", color="#C44E52", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([pretty[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Positive count (train split)", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig("images/fig_label_dist.pdf", dpi=300, bbox_inches="tight")
    fig.savefig("images/fig_label_dist.png", dpi=200, bbox_inches="tight")
    print("Saved: images/fig_label_dist.pdf")

    # Print summary stats for writing
    N = len(df)
    total_cells = N * K
    for name, arr in [("NEW", new), ("RESOLVED", res), ("STABLE_PRESENT", stp)]:
        rate = arr.sum() / (total_cells + 1e-9)
        print(f"{name}: total_positives={arr.sum()} | positive_rate={rate:.4f} ({rate*100:.2f}%)")

if __name__ == "__main__":
    main()