"""
M5：Loss 改良
- class weight = 1 / sqrt(pixel_count_per_class)，從 train mask 實際統計
- CE（動態 class weight）+ Dice Loss
"""
import os
import sys
import time
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex

# 讓 m5/ 能 import 上層的 dataset.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import SteelBallDataset, NUM_CLASSES, get_train_transform, get_val_transform

# ── 路徑 ──────────────────────────────────────────────────────────
SMP_DIR  = r"D:\coca\smp"
M5_DIR   = r"D:\coca\smp\m5"
CKPT_DIR = os.path.join(M5_DIR, "checkpoints")
LOG_PATH = os.path.join(M5_DIR, "train_log.json")

EPOCHS       = 100
BATCH_SIZE   = 8
LR           = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 20
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(train_txt):
    """統計 train mask 各 class 像素數，回傳 1/sqrt(count) 正規化權重"""
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.float64)

    with open(train_txt, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"統計 {len(lines)} 張 train mask 的 pixel 分布...")
    for line in lines:
        _, mask_path = line.split("\t")
        mask = np.array(Image.open(mask_path))
        for c in range(NUM_CLASSES):
            pixel_counts[c] += (mask == c).sum()

    print("\nPixel 統計：")
    for c in range(NUM_CLASSES):
        print(f"  Class {c:>2}: {pixel_counts[c]:>12,.0f} px")

    # 1 / sqrt(count)，count=0 的類別給 0
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    for c in range(NUM_CLASSES):
        if pixel_counts[c] > 0:
            weights[c] = 1.0 / np.sqrt(pixel_counts[c])

    # 正規化：最小 weight = 1（讓數值可讀）
    weights = weights / weights.min()

    print("\nClass weights（1/sqrt pixel，正規化）：")
    for c in range(NUM_CLASSES):
        print(f"  Class {c:>2}: {weights[c]:.4f}")

    return torch.tensor(weights, dtype=torch.float32)


def build_model():
    return smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )


def main():
    print(f"Device: {DEVICE}")

    train_txt = os.path.join(SMP_DIR, "train.txt")
    class_weights = compute_class_weights(train_txt)

    train_ds = SteelBallDataset(train_txt, get_train_transform())
    val_ds   = SteelBallDataset(os.path.join(SMP_DIR, "val.txt"), get_val_transform())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"\nTrain: {len(train_ds)}  Val: {len(val_ds)}")

    model     = build_model().to(DEVICE)
    ce_loss   = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=list(range(NUM_CLASSES)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    miou_metric = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average="macro").to(DEVICE)

    best_miou  = 0.0
    no_improve = 0
    log = []

    print()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = ce_loss(preds, masks) + dice_loss(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        # ── Val ──
        model.eval()
        val_loss = 0.0
        miou_metric.reset()
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                loss  = ce_loss(preds, masks) + dice_loss(preds, masks)
                val_loss += loss.item() * imgs.size(0)
                miou_metric.update(preds.argmax(dim=1), masks)
        val_loss /= len(val_ds)
        val_miou = miou_metric.compute().item()

        scheduler.step()
        elapsed = time.time() - t0

        print(f"[{epoch:>3}/{EPOCHS}] "
              f"Loss={train_loss:.4f}  Val Loss={val_loss:.4f}  Val mIoU={val_miou:.4f}  "
              f"({elapsed:.1f}s)")

        log.append({"epoch": epoch, "train_loss": train_loss,
                    "val_loss": val_loss, "val_miou": val_miou})
        with open(LOG_PATH, "w") as f:
            json.dump(log, f, indent=2)

        if val_miou > best_miou:
            best_miou  = val_miou
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best.pth"))
            print(f"  ✓ 新最佳 mIoU={best_miou:.4f}，已儲存")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"早停（{PATIENCE} epochs 無進步）")
                break

    print(f"\n訓練完成，最佳 Val mIoU = {best_miou:.4f}")
    print(f"Checkpoint：{CKPT_DIR}/best.pth")


if __name__ == "__main__":
    main()
