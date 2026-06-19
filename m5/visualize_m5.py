"""
M5 視覺化：從 test set 抽 20 張，顯示 原圖 / GT Mask / 預測 Mask / Overlay
結果存於 m5/figures/visualize/
"""
import os
import sys
import random
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.rcParams["font.family"] = ["Microsoft JhengHei", "DejaVu Sans"]
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import SteelBallDataset, NUM_CLASSES, get_val_transform

# ── 路徑 ──────────────────────────────────────────────────────────
SMP_DIR   = r"D:\coca\smp"
M5_DIR    = r"D:\coca\smp\m5"
CKPT_PATH = os.path.join(M5_DIR, "checkpoints", "best.pth")
OUT_DIR   = os.path.join(M5_DIR, "figures", "visualize")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 20
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    "背景", "100小黑傷", "101灰傷刻痕", "102麻點",
    "103大黑傷", "104研磨傷", "105肯傷", "106刮傷",
    "107生鏽", "108霧面", "109亮傷暗", "110小白點線", "111亮傷亮"
]

PALETTE = np.array([
    (0,   0,   0),
    (255, 0,   0),
    (0,   255, 0),
    (255, 255, 0),
    (0,   0,   255),
    (255, 128, 0),
    (128, 0,   255),
    (0,   255, 255),
    (255, 0,   255),
    (128, 255, 0),
    (255, 200, 200),
    (0,   128, 128),
    (200, 150, 255),
], dtype=np.uint8)


def mask_to_color(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        rgb[mask == i] = color
    return rgb


def denormalize(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(STD) + np.array(MEAN)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def build_model():
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()


def main():
    random.seed(42)

    dataset = SteelBallDataset(os.path.join(SMP_DIR, "test.txt"), get_val_transform())
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:N_SAMPLES]

    model = build_model()
    print(f"模型載入：{CKPT_PATH}")
    print(f"共輸出 {N_SAMPLES} 張視覺化圖到 {OUT_DIR}")

    for rank, idx in enumerate(indices):
        img_tensor, gt_mask = dataset[idx]
        img_np = denormalize(img_tensor)
        gt_np  = gt_mask.numpy()

        with torch.no_grad():
            pred_logits = model(img_tensor.unsqueeze(0).to(DEVICE))
            pred_mask   = pred_logits.argmax(dim=1).squeeze(0).cpu().numpy()

        gt_color   = mask_to_color(gt_np)
        pred_color = mask_to_color(pred_mask)
        overlay    = (img_np * 0.5 + pred_color * 0.5).astype(np.uint8)

        ious = []
        for c in range(1, NUM_CLASSES):
            inter = ((gt_np == c) & (pred_mask == c)).sum()
            union = ((gt_np == c) | (pred_mask == c)).sum()
            if union > 0:
                ious.append((c, inter / union))

        iou_str = "  ".join([f"{CLASS_NAMES[c]}={v:.2f}" for c, v in ious]) if ious else "背景only"

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_np);     axes[0].set_title("原圖");         axes[0].axis("off")
        axes[1].imshow(gt_color);   axes[1].set_title("GT Mask");      axes[1].axis("off")
        axes[2].imshow(pred_color); axes[2].set_title("預測 Mask");    axes[2].axis("off")
        axes[3].imshow(overlay);    axes[3].set_title("預測 Overlay"); axes[3].axis("off")

        legend_classes = sorted(set(gt_np.flatten()) | set(pred_mask.flatten()))
        patches = [mpatches.Patch(color=tuple(PALETTE[c]/255), label=CLASS_NAMES[c])
                   for c in legend_classes]
        fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 7),
                   fontsize=7, framealpha=0.8)

        fig.suptitle(f"[M5] [{rank:02d}] IoU → {iou_str}", fontsize=9)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        out = os.path.join(OUT_DIR, f"{rank:02d}.png")
        plt.savefig(out, dpi=100)
        plt.close()
        print(f"  [{rank+1:>2}/{N_SAMPLES}] 儲存：{out}")

    print("\n視覺化完成！")


if __name__ == "__main__":
    main()
