import json
import re
import pandas as pd
from collections import Counter
from pathlib import Path

INP = Path("artifacts/studies.csv")
OUT = Path("artifacts/studies_topk.csv")
LABEL_JSON = Path("artifacts/label_cols.json")

TOP_K = 20
# EXCLUDE = {
#     "unchanged", "normal",
#     "nsg tube",
#     "central venous catheter via jugular vein",
# } # extend if needed

EXCLUDE = {
    "unchanged", "normal",
    "nsg tube",
    "central venous catheter via jugular vein",
    "scoliosis", "kyphosis", "vertebral degenerative changes",
    "aortic elongation",
    "chronic changes",
}

def clean_name(lbl: str) -> str:
    lbl = lbl.strip().lower()
    lbl = re.sub(r"[^a-z0-9]+", "_", lbl).strip("_")
    return "y_" + lbl

def main():
    df = pd.read_csv(INP, low_memory=False)

    # labels_list was saved as string; convert back
    import ast
    df["labels_list"] = df["labels_list"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    # count label frequency over studies
    c = Counter()
    for labs in df["labels_list"]:
        for l in labs:
            l2 = str(l).strip().lower()
            if l2 in EXCLUDE:
                continue
            c[l2] += 1

    top = [lbl for (lbl, _) in c.most_common(TOP_K)]
    label_cols = [clean_name(x) for x in top]

    # make multi-hot columns
    top_set = set(top)
    for orig, col in zip(top, label_cols):
        df[col] = df["labels_list"].apply(lambda labs: 1 if orig in [str(x).strip().lower() for x in labs] else 0)

    df.to_csv(OUT, index=False)

    with LABEL_JSON.open("w", encoding="utf-8") as f:
        json.dump({"top_labels": top, "label_cols": label_cols}, f, indent=2)

    print("Saved:", OUT)
    print("Saved:", LABEL_JSON)
    print("Top labels:", top)

if __name__ == "__main__":
    main()