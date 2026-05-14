"""
calibrate_ablation.py
Compares fixed threshold (0.5) vs per-head calibrated threshold on TEST split.
Run AFTER train_delta.py and 08_calibrate_threshold.py:
    python calibrate_ablation.py
Prints a table you paste into your paper.
Also saves: images/fig_calibration.pdf
"""

import ast, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_pairs import PairDeltaDataset
from model_delta_lnn import DeltaLNN

TEST_CSV = "artifacts/test_pairs_with_delta.csv"
CKPT     = "artifacts/delta_runs/best.pt"
THR_JSON = "artifacts/best_thresholds.json"

HEAD_NAMES = ["NEW", "RESOLVED", "STABLE_PRESENT"]

def prf(y_true, y_hat):
    tp = int((y_hat * y_true).sum())
    fp = int((y_hat * (1 - y_true)).sum())
    fn = int(((1 - y_hat) * y_true).sum())
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f = 2 * p * r / (p + r + 1e-9)
    return p, r, f

def collect_probs(model, ds, device):
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    all_probs, all_true = [], []
    model.eval()
    with torch.no_grad():
        for prev_imgs, curr_imgs, dt, y in dl:
            logits = model(prev_imgs.to(device), curr_imgs.to(device), dt.to(device))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_true.append(y.numpy())
    return np.concatenate(all_probs, 0), np.concatenate(all_true, 0)   # (N,3,K)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ds    = PairDeltaDataset(TEST_CSV, img_size=224)
    model = DeltaLNN(num_labels=ds.K, freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))

    probs, ytrue = collect_probs(model, ds, device)   # (N,3,K)

    thr_cfg = json.loads(open(THR_JSON).read())
    calibrated_thrs = [
        float(thr_cfg["per_head"]["NEW"]),
        float(thr_cfg["per_head"]["RESOLVED"]),
        float(thr_cfg["per_head"]["STABLE_PRESENT"]),
    ]
    fixed_thr = 0.5

    rows = []
    for h, name in enumerate(HEAD_NAMES):
        # Fixed threshold
        y_hat_f = (probs[:, h, :] >= fixed_thr).astype(np.int32)
        pf, rf, ff = prf(ytrue[:, h, :], y_hat_f)

        # Calibrated threshold
        y_hat_c = (probs[:, h, :] >= calibrated_thrs[h]).astype(np.int32)
        pc, rc, fc = prf(ytrue[:, h, :], y_hat_c)

        rows.append({
            "head": name,
            "thr_fixed": fixed_thr,   "P_fixed": pf, "R_fixed": rf, "F1_fixed": ff,
            "thr_calib": calibrated_thrs[h], "P_calib": pc, "R_calib": rc, "F1_calib": fc,
            "delta_F1": fc - ff,
        })
        print(f"{name:20s}  fixed(thr=0.50): P={pf:.3f} R={rf:.3f} F1={ff:.3f} "
              f"| calib(thr={calibrated_thrs[h]:.2f}): P={pc:.3f} R={rc:.3f} F1={fc:.3f} "
              f"| ΔF1={fc-ff:+.3f}")

    # ---- Grouped bar chart ----
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.2), sharey=False)
    metrics = [("Precision", "P_fixed", "P_calib"),
               ("Recall",    "R_fixed", "R_calib"),
               ("F1-score",  "F1_fixed","F1_calib")]

    colors = {"Fixed (0.50)": "#8da0cb", "Calibrated": "#fc8d62"}

    for ax, (metric, col_f, col_c) in zip(axes, metrics):
        vals_fixed = [rows[i][col_f] for i in range(3)]
        vals_calib = [rows[i][col_c] for i in range(3)]
        x = np.arange(3)
        w = 0.34
        ax.bar(x - w/2, vals_fixed, w, label="Fixed (0.50)",
               color=colors["Fixed (0.50)"], zorder=3)
        ax.bar(x + w/2, vals_calib, w, label="Calibrated",
               color=colors["Calibrated"], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(["NEW", "RES", "STP"], fontsize=8)
        ax.set_title(metric, fontsize=9)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("Score", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.04))

    fig.tight_layout()
    fig.savefig("images/fig_calibration.pdf", dpi=300, bbox_inches="tight")
    fig.savefig("images/fig_calibration.png", dpi=200, bbox_inches="tight")
    print("\nSaved: images/fig_calibration.pdf and images/fig_calibration.png")

if __name__ == "__main__":
    main()