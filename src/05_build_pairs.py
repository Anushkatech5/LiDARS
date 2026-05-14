import ast
import pandas as pd
from pathlib import Path

def load(studies_csv: Path):
    df = pd.read_csv(studies_csv, low_memory=False)
    df["study_date"] = pd.to_datetime(df["study_date"], errors="coerce")
    df["image_paths"] = df["image_paths"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    return df

def build_pairs(df: pd.DataFrame):
    pairs = []
    for pid, g in df.groupby("patient_id"):
        g = g.sort_values(["study_date", "study_id"])
        rows = g.to_dict("records")
        for i in range(1, len(rows)):
            prev = rows[i-1]
            curr = rows[i]
            dt_days = (curr["study_date"] - prev["study_date"]).days if pd.notna(curr["study_date"]) and pd.notna(prev["study_date"]) else 0
            pairs.append({
                "patient_id": pid,
                "prev_study_id": prev["study_id"],
                "curr_study_id": curr["study_id"],
                "dt_days": float(max(dt_days, 0)),
                "prev_image_paths": prev["image_paths"],
                "curr_image_paths": curr["image_paths"],
                # y columns are already in the study table
                "prev_y": [prev[c] for c in df.columns if c.startswith("y_")],
                "curr_y": [curr[c] for c in df.columns if c.startswith("y_")],
                "label_cols": [c for c in df.columns if c.startswith("y_")]
            })
    return pd.DataFrame(pairs)

def main():
    for split in ["train", "val", "test"]:
        inp = Path(f"artifacts/studies_{split}.csv")
        out = Path(f"artifacts/{split}_pairs.csv")
        df = load(inp)
        pairs = build_pairs(df)
        pairs.to_csv(out, index=False)
        print("Saved:", out, "pairs=", len(pairs))

if __name__ == "__main__":
    main()