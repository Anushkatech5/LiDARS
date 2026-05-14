"""
evaluate_comparison.py  —  Side-by-side evaluation: DeltaLNN vs DeltaMLP
Save this to: D:\ImagingInformatics\src\evaluate_comparison.py

Run AFTER both report CSVs exist:
    python D:\ImagingInformatics\src\evaluate_comparison.py

Reads:
  D:\ImagingInformatics\artifacts\delta_reports_test.csv       (DeltaLNN)
  D:\ImagingInformatics\artifacts\baseline_reports_test.csv    (DeltaMLP)
  D:\ImagingInformatics\artifacts\test_pairs_with_delta.csv    (ground truth)

Saves:
  D:\ImagingInformatics\artifacts\comparison_results.json
  (and prints a table you can paste straight into paper Table 3)
"""

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
SRC_DIR  = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

GT            = str(BASE_DIR / "artifacts" / "test_pairs_with_delta.csv")
PRED_LNN      = str(BASE_DIR / "artifacts" / "delta_reports_test.csv")
PRED_BASELINE = str(BASE_DIR / "artifacts" / "baseline_reports_test.csv")
OUT_JSON      = str(BASE_DIR / "artifacts" / "comparison_results.json")


# ── Helper functions ──────────────────────────────────────────────────────────

def parse_list(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def parse_pred(cell):
    """Convert a semicolon-separated prediction string to a Python set."""
    if pd.isna(cell) or str(cell).strip() == "":
        return set()
    return set(x.strip() for x in str(cell).split(";") if x.strip())


def vec_to_set(vec, label_cols):
    """Convert a binary vector like [0,1,1,0] to {'y_cardiomegaly', 'y_pneumonia'}."""
    return set(label_cols[i] for i, v in enumerate(vec) if int(v) == 1)


def micro_prf(pred_sets, true_sets):
    """
    Micro-averaged Precision, Recall, F1 across all samples.
    Micro-averaging sums raw TP/FP/FN counts before dividing, which gives
    more weight to common labels — appropriate for our imbalanced setting.
    """
    tp = fp = fn = 0
    for ps, ts in zip(pred_sets, true_sets):
        tp += len(ps & ts)   # correct predictions
        fp += len(ps - ts)   # predicted but not in ground truth
        fn += len(ts - ps)   # in ground truth but not predicted
    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return p, r, f1, tp, fp, fn


def evaluate_model(pred_path, gt, label_cols):
    """Load one prediction CSV and compute P/R/F1 for all three heads."""
    pred = pd.read_csv(pred_path)

    pred_new = pred["new"].apply(parse_pred).tolist()
    pred_res = pred["resolved"].apply(parse_pred).tolist()
    pred_stp = pred["stable_present"].apply(parse_pred).tolist()

    true_new = gt["delta_new"].apply(lambda v: vec_to_set(v, label_cols)).tolist()
    true_res = gt["delta_resolved"].apply(lambda v: vec_to_set(v, label_cols)).tolist()
    true_stp = gt["delta_stable_present"].apply(lambda v: vec_to_set(v, label_cols)).tolist()

    results = {}
    for name, ps, ts in [("NEW",            pred_new, true_new),
                          ("RESOLVED",       pred_res, true_res),
                          ("STABLE_PRESENT", pred_stp, true_stp)]:
        p, r, f1, tp, fp, fn = micro_prf(ps, ts)
        results[name] = {"P": p, "R": r, "F1": f1,
                         "TP": tp, "FP": fp, "FN": fn}

    # Macro F1 = simple average of the three per-head F1 scores
    results["MACRO_F1"] = float(np.mean([results[h]["F1"] for h in
                                         ["NEW", "RESOLVED", "STABLE_PRESENT"]]))
    return results


def print_comparison_table(lnn, mlp):
    """Print a formatted table for copy-pasting into the paper."""
    heads = ["NEW", "RESOLVED", "STABLE_PRESENT"]
    W = 72

    print("\n" + "=" * W)
    print(f"{'Head':<22} {'Model':<18} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("=" * W)

    for head in heads:
        l = lnn[head]
        m = mlp[head]
        print(f"{head:<22} {'DeltaLNN (ours)':<18} "
              f"{l['P']:>8.3f} {l['R']:>8.3f} {l['F1']:>8.3f}")
        print(f"{'':<22} {'DeltaMLP (base)':<18} "
              f"{m['P']:>8.3f} {m['R']:>8.3f} {m['F1']:>8.3f}")
        print("-" * W)

    print(f"\n{'Macro F1':<22} {'DeltaLNN (ours)':<18} {lnn['MACRO_F1']:>26.3f}")
    print(f"{'Macro F1':<22} {'DeltaMLP (base)':<18} {mlp['MACRO_F1']:>26.3f}")
    print("=" * W)

    print("\n-- F1 improvement of DeltaLNN over DeltaMLP (positive = LNN wins) --")
    for head in heads:
        delta = lnn[head]["F1"] - mlp[head]["F1"]
        sign  = "+" if delta >= 0 else ""
        print(f"  {head:<22} {sign}{delta:.3f}")
    macro_d = lnn["MACRO_F1"] - mlp["MACRO_F1"]
    print(f"  {'Macro':<22} {'+' if macro_d >= 0 else ''}{macro_d:.3f}")


def main():
    print("Ground truth :", GT)
    print("DeltaLNN pred:", PRED_LNN)
    print("Baseline pred:", PRED_BASELINE)

    # Load and parse ground truth
    gt = pd.read_csv(GT, low_memory=False)
    for c in ["delta_new", "delta_resolved", "delta_stable_present", "label_cols"]:
        gt[c] = gt[c].apply(parse_list)
    label_cols = gt.iloc[0]["label_cols"]

    print("\nEvaluating DeltaLNN ...")
    lnn_res = evaluate_model(PRED_LNN,      gt, label_cols)

    print("Evaluating DeltaMLP baseline ...")
    mlp_res = evaluate_model(PRED_BASELINE, gt, label_cols)

    print_comparison_table(lnn_res, mlp_res)

    # Save numeric results for later use in the paper
    out = {"DeltaLNN": lnn_res, "DeltaMLP_baseline": mlp_res}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nNumeric results saved to: {OUT_JSON}")


if __name__ == "__main__":
    main()