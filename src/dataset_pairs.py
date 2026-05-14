import ast
import math
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


# -------------------------
# Path normalization (Windows -> Linux safe)
# -------------------------
def norm_path(p: str) -> str:
    """
    Make file paths portable across Windows/Linux.
    Your CSV may contain paths like: data\\selected_by_zip\\20_selectedimages\\xxx.png
    On Linux this must become:      data/selected_by_zip/20_selectedimages/xxx.png
    """
    if p is None:
        return None

    p = str(p).strip()

    # 1) Convert backslashes to forward slashes (critical for Linux)
    p = p.replace("\\", "/")

    # 2) Normalize redundant separators and ".."
    # Note: os.path.normpath on Linux will keep "/" separators
    p = os.path.normpath(p)

    return p


def load_img(p: str):
    p = norm_path(p)
    return Image.open(p).convert("RGB")


# class PairDeltaDataset(Dataset):
class PairDeltaDataset:
    def __init__(self, pairs_csv, img_size=224):
        self.df = pd.read_csv(pairs_csv, low_memory=False)

        # convert list-like columns (stored as strings in CSV)
        for c in [
            "prev_image_paths",
            "curr_image_paths",
            "delta_new",
            "delta_resolved",
            "delta_stable_present",
            "label_cols",
        ]:
            self.df[c] = self.df[c].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        self.label_cols = self.df.iloc[0]["label_cols"]
        self.K = len(self.label_cols)

        # IMPORTANT: ImageNet normalization for pretrained ResNet18
        self.tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def _load_two(self, paths, idx=None, which=""):
        """
        Loads up to 2 images. If only 1 is available, duplicates it.
        Normalizes paths so both Windows and Linux work.
        """
        imgs = []
        for p in paths[:2]:
            p2 = norm_path(p)

            # Helpful error if file missing
            if not os.path.exists(p2):
                raise FileNotFoundError(
                    f"[Missing image] idx={idx} ({which})\n"
                    f"CSV path: {p}\n"
                    f"Normalized: {p2}\n"
                    f"Current working dir: {os.getcwd()}\n"
                )

            im = load_img(p2)
            imgs.append(self.tf(im))

        if len(imgs) == 1:
            imgs.append(imgs[0].clone())

        return torch.stack(imgs, dim=0)  # (2,3,H,W)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        prev_imgs = self._load_two(r["prev_image_paths"], idx=idx, which="prev")
        curr_imgs = self._load_two(r["curr_image_paths"], idx=idx, which="curr")

        y_new = torch.tensor(r["delta_new"], dtype=torch.float32)
        y_res = torch.tensor(r["delta_resolved"], dtype=torch.float32)
        y_stp = torch.tensor(r["delta_stable_present"], dtype=torch.float32)
        y = torch.stack([y_new, y_res, y_stp], dim=0)  # (3,K)

        # dt scaling: clip + log1p
        dt = float(r["dt_days"])
        dt = max(0.0, min(dt, 365.0))
        dt = math.log1p(dt)
        dt = torch.tensor(dt, dtype=torch.float32)

        return prev_imgs, curr_imgs, dt, y