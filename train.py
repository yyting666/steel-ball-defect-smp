"""
M4：smp U-Net + EfficientNet-B0 baseline
Loss: CrossEntropy（class weight）+ Dice
"""
import os
import time
import json
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex

from dataset import SteelBallDataset, NUM_CLASSES, get_train_transform, get_val_transform

# ── 設定 ──────────────────────────────────────────────────────────
SMP_DIR    = r"D:\coca\smp"
CKPT_DIR   = os.path.join(SMP_DIR, "checkpoints")
LOG_PATH   = os.path.join(SMP_DIR, "train_log.json")
os.makedirs(CKPT_DIR, exist_ok=True)

EPOCHS     = 100
BATCH_SIZE = 8
LR         = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE   = 20
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class weight：瑕疵類給高權重，背景低
CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
CLASS_WEIGHTS[0] = 0.1   # 背景
for i in range(1, NUM_CLASSES):
    CLASS_WEIGHTS[i] = 1.0


def build_model():
    return smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )


def main():
    print(f"Device: {DEVICE}")

    # 資料集
    train_ds = SteelBallDataset(os.path.join(SMP_DIR, "train.txt"), get_train_transform())
    val_ds   = SteelBallDataset(os.path.join(SMP_DIR, "val.txt"),   get_val_transform())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # 模型 + loss + optimizer
    model = build_model().to(DEVICE)
    ce_loss   = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))
    dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=list(range(NUM_CLASSES)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    miou_metric = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average="macro").to(DEVICE)

    best_miou   = 0.0
    no_improve  = 0
    log = []

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

        # 儲存最佳
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


if __name__ == "__main__":
    main()
