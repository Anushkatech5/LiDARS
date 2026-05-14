import ast
import pandas as pd
from pathlib import Path

def parse_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else x

def main():
    for split in ["train", "val", "test"]:
        inp = Path(f"artifacts/{split}_pairs.csv")
        out = Path(f"artifacts/{split}_pairs_with_delta.csv")
        df = pd.read_csv(inp, low_memory=False)

        df["prev_y"] = df["prev_y"].apply(parse_list)
        df["curr_y"] = df["curr_y"].apply(parse_list)
        df["label_cols"] = df["label_cols"].apply(parse_list)
        df["prev_image_paths"] = df["prev_image_paths"].apply(parse_list)
        df["curr_image_paths"] = df["curr_image_paths"].apply(parse_list)

        delta_new = []
        delta_res = []
        delta_stp = []

        for py, cy in zip(df["prev_y"], df["curr_y"]):
            new = [1 if (p == 0 and c == 1) else 0 for p, c in zip(py, cy)]
            res = [1 if (p == 1 and c == 0) else 0 for p, c in zip(py, cy)]
            stp = [1 if (p == 1 and c == 1) else 0 for p, c in zip(py, cy)]
            delta_new.append(new)
            delta_res.append(res)
            delta_stp.append(stp)

        df["delta_new"] = delta_new
        df["delta_resolved"] = delta_res
        df["delta_stable_present"] = delta_stp

        # keep only what training needs
        keep_cols = ["patient_id", "dt_days", "prev_image_paths", "curr_image_paths",
                     "delta_new", "delta_resolved", "delta_stable_present", "label_cols"]
        df[keep_cols].to_csv(out, index=False)
        print("Saved:", out, "pairs=", len(df))

if __name__ == "__main__":
    main()
