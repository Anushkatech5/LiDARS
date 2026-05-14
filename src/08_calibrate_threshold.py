import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_pairs import PairDeltaDataset
from model_delta_lnn import DeltaLNN

VAL = "artifacts/val_pairs_with_delta.csv"
CKPT = "artifacts/delta_runs/best.pt"

def prf(tp, fp, fn):
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f = 2 * p * r / (p + r + 1e-9)
    return p, r, f

def score_threshold(y_true, y_prob, thr):
    y_hat = (y_prob >= thr).astype(np.int32)
    tp = int((y_hat * y_true).sum())
    fp = int((y_hat * (1 - y_true)).sum())
    fn = int(((1 - y_hat) * y_true).sum())
    return prf(tp, fp, fn), (tp, fp, fn)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PairDeltaDataset(VAL, img_size=224)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    model = DeltaLNN(num_labels=ds.K, freeze_backbone=True).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    all_probs = []
    all_true = []

    with torch.no_grad():
        for prev_imgs, curr_imgs, dt, y in dl:
            prev_imgs = prev_imgs.to(device)
            curr_imgs = curr_imgs.to(device)
            dt = dt.to(device)

            logits = model(prev_imgs, curr_imgs, dt)          # (B,3,K)
            probs = torch.sigmoid(logits).cpu().numpy()       # (B,3,K)

            all_probs.append(probs)
            all_true.append(y.numpy())

    probs = np.concatenate(all_probs, axis=0)   # (N,3,K)
    ytrue = np.concatenate(all_true, axis=0)    # (N,3,K)

    thresholds = np.arange(0.10, 0.91, 0.05)

    head_names = ["NEW", "RESOLVED", "STABLE_PRESENT"]

    print("=== Per-head best thresholds (maximize F1 on VAL) ===")
    best_thr = {}
    for h, name in enumerate(head_names):
        best = (-1, None, None)  # f1, thr, (p,r)
        for t in thresholds:
            (p,r,f1), _ = score_threshold(ytrue[:,h,:], probs[:,h,:], t)
            if f1 > best[0]:
                best = (f1, t, (p,r))
        best_thr[name] = best[1]
        print(f"{name}: best_thr={best[1]:.2f} | F1={best[0]:.3f} | P={best[2][0]:.3f} R={best[2][1]:.3f}")

    print("\n=== Single global threshold (all heads together) ===")
    ytrue_flat = ytrue.reshape(ytrue.shape[0], -1)
    probs_flat = probs.reshape(probs.shape[0], -1)

    best = (-1, None, None)
    for t in thresholds:
        (p,r,f1), _ = score_threshold(ytrue_flat, probs_flat, t)
        if f1 > best[0]:
            best = (f1, t, (p,r))
    print(f"GLOBAL: best_thr={best[1]:.2f} | F1={best[0]:.3f} | P={best[2][0]:.3f} R={best[2][1]:.3f}")

    # save to json for reuse
    import json
    out = {
        "per_head": best_thr,
        "global": float(best[1]),
        "threshold_grid": thresholds.tolist()
    }
    with open("artifacts/best_thresholds.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: artifacts/best_thresholds.json")

if __name__ == "__main__":
    main()