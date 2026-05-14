import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from model_delta_lnn import DeltaLNN

# -----------------------
# CONFIG (YOUR PATHS)
# -----------------------
PRIOR_FRONTAL   = "data/selected_by_zip/YOUR_PRIOR_FRONTAL.png"
CURRENT_FRONTAL = "data/selected_by_zip/YOUR_CURRENT_FRONTAL.png"

PRIOR_LATERAL   = "data/selected_by_zip/YOUR_PRIOR_LATERAL.png"
CURRENT_LATERAL = "data/selected_by_zip/YOUR_CURRENT_LATERAL.png"

DT_DAYS = 41  # for naming only (model encoding uses dt inside DeltaLNN, not needed here)

CHECKPOINT = "artifacts/delta_runs/best.pt"
OUT_DIR    = "artifacts/stageB_embeddings_manual"
IMG_SIZE   = 224
PAIR_TAG   = "manual_581"
# -----------------------

os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Image loading (2 views)
# -----------------------
def load_img_rgb(path, size=224):
    """
    Loads image -> RGB float tensor (3,H,W) in [0,1], with simple contrast normalization.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing image file: {path}")

    im = Image.open(path).convert("L")
    im = im.resize((size, size))
    arr = np.array(im).astype(np.float32)

    # simple robust contrast normalize for nicer input
    lo, hi = np.percentile(arr, (2, 98))
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)

    # convert grayscale -> 3 channel
    arr3 = np.stack([arr, arr, arr], axis=0)  # (3,H,W)
    return torch.from_numpy(arr3)


def make_2view_tensor(frontal_path, lateral_path, size=224):
    """
    Returns tensor: (1,2,3,H,W)
    """
    frontal = load_img_rgb(frontal_path, size=size)
    lateral = load_img_rgb(lateral_path, size=size)
    two = torch.stack([frontal, lateral], dim=0)  # (2,3,H,W)
    return two.unsqueeze(0)  # (1,2,3,H,W)


# -----------------------
# Encoder forward: per-view + mean pool
# -----------------------
@torch.no_grad()
def encode_with_views(model_encoder, imgs_2view):
    """
    imgs_2view: (B, V=2, 3, H, W)
    returns:
      e_views: (B, 2, D)
      e_pool : (B, D)
    """
    B, V, C, H, W = imgs_2view.shape
    x = imgs_2view.view(B * V, C, H, W)

    feat = model_encoder.backbone(x)

    # --- FIX: handle (B*V, 512, 1, 1) or any 4D feature map ---
    if feat.dim() == 4:
        # GAP -> (B*V, 512, 1, 1) then flatten -> (B*V, 512)
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        feat = feat.flatten(1)
    elif feat.dim() == 3:
        # just in case (B*V, 512, 1)
        feat = feat.flatten(1)

    e = model_encoder.proj(feat)      # (B*V, D)
    e_views = e.view(B, V, -1)        # (B, 2, D)
    e_pool = e_views.mean(dim=1)      # (B, D)
    return e_views, e_pool

# -----------------------
# Pretty visualizations
# -----------------------
def save_vec_heatmap(vec, out_png):
    """
    D=256 -> 16x16 heatmap, else 1xD strip.
    """
    v = vec.astype(np.float32)
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    v = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)

    D = v.shape[0]
    s = int(np.sqrt(D))
    if s * s == D:
        img = v.reshape(s, s)
        figsize = (1.5, 1.5)
    else:
        img = v.reshape(1, D)
        figsize = (3.0, 0.4)

    plt.figure(figsize=figsize, dpi=300)
    plt.imshow(img, cmap="gray", aspect="auto", interpolation="nearest")
    plt.axis("off")
    plt.savefig(out_png, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_hist_overlay(a, b, out_png, label_a="e⁻", label_b="e⁺", bins=40):
    """
    Nice for paper: distribution shift of embedding values.
    """
    plt.figure(figsize=(4.2, 2.6), dpi=250)
    plt.hist(a, bins=bins, alpha=0.55, density=True, label=label_a)
    plt.hist(b, bins=bins, alpha=0.55, density=True, label=label_b)
    plt.xlabel("Embedding value")
    plt.ylabel("Density")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, transparent=True, bbox_inches="tight")
    plt.close()


def save_absdiff_strip(diff, out_png):
    """
    |e⁺-e⁻| as 1xD heat strip (looks clean + interpretable).
    """
    d = np.abs(diff).astype(np.float32)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)

    plt.figure(figsize=(4.0, 0.55), dpi=300)
    plt.imshow(d.reshape(1, -1), cmap="gray", aspect="auto", interpolation="nearest")
    plt.axis("off")
    plt.savefig(out_png, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_pca_scatter(points, names, out_png):
    """
    points: list of np arrays shape (D,)
    Creates a clean 2D PCA scatter (no sklearn needed).
    """
    X = np.stack(points, axis=0).astype(np.float32)  # (N,D)
    X = X - X.mean(axis=0, keepdims=True)

    # SVD PCA
    # X = U S V^T -> take first 2 PCs from V
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt[:2].T  # (D,2)
    Z = X @ W     # (N,2)

    plt.figure(figsize=(3.6, 3.0), dpi=250)
    plt.scatter(Z[:, 0], Z[:, 1], s=55)
    for i, n in enumerate(names):
        plt.text(Z[i, 0], Z[i, 1], f" {n}", fontsize=9)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_png, transparent=True, bbox_inches="tight")
    plt.close()


# -----------------------
# Main
# -----------------------
def main():
    # Build model (num_labels can be any positive int for loading encoder weights)
    # Use same setting as training if you can; but encoder loads fine regardless.
    model = DeltaLNN(num_labels=20, freeze_backbone=True).to(device)
    model.eval()

    if os.path.exists(CHECKPOINT):
        # safer in new torch versions:
        try:
            sd = torch.load(CHECKPOINT, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(sd, strict=False)
        print("[OK] Loaded checkpoint:", CHECKPOINT)
    else:
        print("[WARN] Checkpoint not found. Using ImageNet backbone + random projection weights.")

    # Make tensors (1,2,3,H,W)
    prev_imgs = make_2view_tensor(PRIOR_FRONTAL, PRIOR_LATERAL, size=IMG_SIZE).to(device)
    curr_imgs = make_2view_tensor(CURRENT_FRONTAL, CURRENT_LATERAL, size=IMG_SIZE).to(device)

    with torch.no_grad():
        prev_views, e_minus = encode_with_views(model.encoder, prev_imgs)
        curr_views, e_plus  = encode_with_views(model.encoder, curr_imgs)

    e_minus_np = e_minus.squeeze(0).cpu().numpy()
    e_plus_np  = e_plus.squeeze(0).cpu().numpy()
    e_diff_np  = (e_plus - e_minus).squeeze(0).cpu().numpy()

    D = e_minus_np.shape[0]
    print(f"[OK] D = {D}")

    # Save raw vectors
    np.save(os.path.join(OUT_DIR, f"{PAIR_TAG}_e_minus.npy"), e_minus_np)
    np.save(os.path.join(OUT_DIR, f"{PAIR_TAG}_e_plus.npy"),  e_plus_np)
    np.save(os.path.join(OUT_DIR, f"{PAIR_TAG}_e_diff.npy"),  e_diff_np)

    # Paper-friendly visuals
    save_vec_heatmap(e_minus_np, os.path.join(OUT_DIR, f"{PAIR_TAG}_e_minus_heat.png"))
    save_vec_heatmap(e_plus_np,  os.path.join(OUT_DIR, f"{PAIR_TAG}_e_plus_heat.png"))
    save_absdiff_strip(e_diff_np, os.path.join(OUT_DIR, f"{PAIR_TAG}_absdiff_strip.png"))
    save_hist_overlay(e_minus_np, e_plus_np, os.path.join(OUT_DIR, f"{PAIR_TAG}_hist_overlay.png"))

    # PCA scatter: show per-view + pooled points
    # points: prior frontal, prior lateral, pooled prior, current frontal, current lateral, pooled current
    pv = prev_views.squeeze(0).cpu().numpy()  # (2,D)
    cv = curr_views.squeeze(0).cpu().numpy()  # (2,D)
    points = [pv[0], pv[1], e_minus_np, cv[0], cv[1], e_plus_np]
    names  = ["p-frontal", "p-lateral", "p-mean", "c-frontal", "c-lateral", "c-mean"]
    save_pca_scatter(points, names, os.path.join(OUT_DIR, f"{PAIR_TAG}_pca_points.png"))

    print("[DONE] Saved everything to:", OUT_DIR)


if __name__ == "__main__":
    main()
