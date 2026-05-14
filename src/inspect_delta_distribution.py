import ast
import numpy as np
import pandas as pd
import argparse

def parse_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="e.g. artifacts/train_pairs_with_delta.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.pairs, low_memory=False)
    for c in ["delta_new","delta_resolved","delta_stable_present","label_cols"]:
        df[c] = df[c].apply(parse_list)

    label_cols = df.iloc[0]["label_cols"]
    K = len(label_cols)

    new = np.array(df["delta_new"].tolist(), dtype=np.int32)
    res = np.array(df["delta_resolved"].tolist(), dtype=np.int32)
    stp = np.array(df["delta_stable_present"].tolist(), dtype=np.int32)

    print("Pairs:", len(df))
    print("Labels K:", K)
    print("dt_days min/median/max:", df["dt_days"].min(), df["dt_days"].median(), df["dt_days"].max())

    def summarize(name, arr):
        total_pos = arr.sum()
        pos_rate = total_pos / (arr.shape[0]*arr.shape[1] + 1e-9)
        print(f"\n{name}:")
        print("  total positives:", int(total_pos))
        print("  positive rate:", float(pos_rate))
        per_label = arr.sum(axis=0)
        top = np.argsort(-per_label)[:10]
        print("  top changing labels:")
        for j in top:
            if per_label[j] == 0:
                break
            print(f"    {label_cols[j]} : {int(per_label[j])}")

    summarize("delta_new", new)
    summarize("delta_resolved", res)
    summarize("delta_stable_present", stp)

if __name__ == "__main__":
    main()