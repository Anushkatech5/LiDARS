import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_pairs import PairDeltaDataset
from model_delta_lnn import DeltaLNN

TRAIN = "artifacts/train_pairs_with_delta.csv"
VAL   = "artifacts/val_pairs_with_delta.csv"
OUT_DIR = "artifacts/delta_runs"
os.makedirs(OUT_DIR, exist_ok=True)

def compute_pos_weight(ds, device):
    # y: (B,3,K) -> flatten (B, 3*K)
    pos = None
    total = 0
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    for _, _, _, y in loader:
        y = y.view(y.size(0), -1)
        if pos is None:
            pos = y.sum(dim=0)
        else:
            pos += y.sum(dim=0)
        total += y.size(0)

    neg = total - pos
    pw = (neg / (pos + 1e-6)).clamp(min=1.0, max=30.0).to(device)
    return pw

def run_epoch(model, loader, loss_fn, opt=None, device="cpu"):
    train = opt is not None
    model.train() if train else model.eval()

    total_loss = 0.0
    n = 0

    for prev_imgs, curr_imgs, dt, y in tqdm(loader):
        prev_imgs = prev_imgs.to(device)
        curr_imgs = curr_imgs.to(device)
        dt = dt.to(device)
        y = y.to(device)

        logits = model(prev_imgs, curr_imgs, dt)           # (B,3,K)
        B = logits.size(0)
        loss = loss_fn(logits.view(B, -1), y.view(B, -1))

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_loss += loss.item() * B
        n += B

    return total_loss / max(1, n)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = PairDeltaDataset(TRAIN, img_size=224)
    val_ds   = PairDeltaDataset(VAL, img_size=224)

    model = DeltaLNN(num_labels=train_ds.K, freeze_backbone=True).to(device)

    # pos_weight to prevent all-zero collapse
    pos_weight = compute_pos_weight(train_ds, device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    best = 1e9
    for epoch in range(1, 31):
        tr = run_epoch(model, train_loader, loss_fn, opt=opt, device=device)
        va = run_epoch(model, val_loader, loss_fn, opt=None, device=device)
        print(f"epoch={epoch} train_loss={tr:.4f} val_loss={va:.4f}")

        if va < best:
            best = va
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))
            print("Saved best checkpoint.")

if __name__ == "__main__":
    main()
