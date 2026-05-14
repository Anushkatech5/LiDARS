import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset_pairs import PairDeltaDataset
from model_delta_lnn import DeltaLNN

TEST = Path("artifacts/test_pairs_with_delta.csv")
CKPT = Path("artifacts/delta_runs/best.pt")
OUT  = Path("artifacts/delta_reports_test.csv")
THR_JSON = Path("artifacts/best_thresholds.json")

# === OPTION 0 SETTINGS ===
# - strict eval: fallback OFF
TOPN_FALLBACK_EVAL = 0
# - human readable: fallback ON
TOPN_FALLBACK_HUMAN = 3
MAX_ITEMS = 5

def parse_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else x

def pretty(lbl: str) -> str:
    return lbl.replace("y_", "").replace("_", " ")

def strip_score(x: str) -> str:
    # "y_label(0.82)" -> "y_label"
    if "(" in x:
        return x.split("(", 1)[0]
    return x

def decode(probs, label_cols, thresh, topn=TOPN_FALLBACK, max_items=MAX_ITEMS):
    probs = np.asarray(probs, dtype=np.float32)

    idx = np.where(probs >= thresh)[0].tolist()
    idx = sorted(idx, key=lambda i: float(probs[i]), reverse=True)[:max_items]

    if len(idx) > 0:
        return [label_cols[i] for i in idx]

    # fallback: show topn with score for transparency
    top = np.argsort(-probs)[:topn]
    return [f"{label_cols[i]}({float(probs[i]):.2f})" for i in top]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if not THR_JSON.exists():
        raise FileNotFoundError("Missing artifacts/best_thresholds.json. Run 08_calibrate_threshold.py first.")

    thr_cfg = json.loads(THR_JSON.read_text(encoding="utf-8"))
    THRESH_NEW = float(thr_cfg["per_head"]["NEW"])
    THRESH_RES = float(thr_cfg["per_head"]["RESOLVED"])
    THRESH_STP = float(thr_cfg["per_head"]["STABLE_PRESENT"])

    print(f"Thresholds (VAL-calibrated): NEW={THRESH_NEW:.2f} RES={THRESH_RES:.2f} STABLE={THRESH_STP:.2f}")

    ds = PairDeltaDataset(str(TEST), img_size=224)

    model = DeltaLNN(num_labels=ds.K, freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(str(CKPT), map_location=device))
    model.eval()

    df = pd.read_csv(TEST, low_memory=False)
    df["label_cols"] = df["label_cols"].apply(parse_list)
    label_cols = df.iloc[0]["label_cols"]

    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            prev_imgs, curr_imgs, dt, _ = ds[i]

            logits = model(
                prev_imgs.unsqueeze(0).to(device),
                curr_imgs.unsqueeze(0).to(device),
                dt.unsqueeze(0).to(device),
            )[0]  # (3,K)

            probs = torch.sigmoid(logits).cpu().numpy()

            new_raw = decode(probs[0], label_cols, thresh=THRESH_NEW)
            res_raw = decode(probs[1], label_cols, thresh=THRESH_RES)
            stp_raw = decode(probs[2], label_cols, thresh=THRESH_STP)

            # EVAL columns must keep y_* IDs (strip score only)
            new_eval = [strip_score(x) for x in new_raw]
            res_eval = [strip_score(x) for x in res_raw]
            stp_eval = [strip_score(x) for x in stp_raw]

            rows.append({
                "pair_index": i,
                "dt_days": float(df.iloc[i]["dt_days"]),

                "new": "; ".join(new_eval),
                "resolved": "; ".join(res_eval),
                "stable_present": "; ".join(stp_eval),

                # human columns only for display in paper
                "new_human": "; ".join(pretty(strip_score(x)) for x in new_raw),
                "resolved_human": "; ".join(pretty(strip_score(x)) for x in res_raw),
                "stable_present_human": "; ".join(pretty(strip_score(x)) for x in stp_raw),

                "new_count": len(new_eval),
                "resolved_count": len(res_eval),
                "stable_present_count": len(stp_eval),
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT, index=False)
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
