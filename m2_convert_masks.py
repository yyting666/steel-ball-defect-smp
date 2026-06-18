"""
M2：LabelMe JSON → mask PNG 轉換
每張 .jpg 產生同名 _mask.png，像素值 = class id（0=背景, 1-12=瑕疵）
支援 polygon, circle, rectangle, line
"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.rcParams["font.family"] = ["Microsoft JhengHei", "DejaVu Sans"]
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = r"D:\coca\steel ball dataset"
MASK_DIR  = r"D:\coca\smp\masks"
VIS_DIR   = r"D:\coca\smp\figures\m2_verify"
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(VIS_DIR,  exist_ok=True)

LABEL_MAP = {
    "100": 1, "101": 2, "102": 3,  "103": 4,
    "104": 5, "105": 6, "106": 7,  "107": 8,
    "108": 9, "109": 10,"110": 11, "111": 12,
    "good": 0,
}

CLASS_NAMES = [
    "背景", "100小黑傷", "101灰傷刻痕", "102麻點",
    "103大黑傷", "104研磨傷", "105肯傷", "106刮傷",
    "107生鏽", "108霧面", "109亮傷暗", "110小白點線", "111亮傷亮"
]

PALETTE = [
    (0,   0,   0),    # 0  背景       黑
    (255, 0,   0),    # 1  100小黑傷  紅
    (0,   255, 0),    # 2  101灰傷刻痕 綠
    (255, 255, 0),    # 3  102麻點    黃
    (0,   0,   255),  # 4  103大黑傷  藍
    (255, 128, 0),    # 5  104研磨傷  橙
    (128, 0,   255),  # 6  105肯傷    紫
    (0,   255, 255),  # 7  106刮傷    青
    (255, 0,   255),  # 8  107生鏽    粉
    (128, 255, 0),    # 9  108霧面    黃綠
    (255, 200, 200),  # 10 109亮傷暗  淡粉
    (0,   128, 128),  # 11 110小白點線 墨綠
    (200, 150, 255),  # 12 111亮傷亮  淡紫
]


def json_to_mask(json_path, img_w, img_h):
    """LabelMe JSON → (H, W) uint8 mask，像素值為 class id"""
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    for shape in ann.get("shapes", []):
        label      = shape["label"]
        cls        = LABEL_MAP.get(label, -1)
        shape_type = shape.get("shape_type", "polygon")
        if cls <= 0:   # 背景或 good 或未知 → 跳過
            continue

        pts = shape["points"]
        poly_img = Image.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(poly_img)

        if shape_type == "circle":
            cx, cy = float(pts[0][0]), float(pts[0][1])
            px, py = float(pts[1][0]), float(pts[1][1])
            r = ((px-cx)**2 + (py-cy)**2) ** 0.5
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=1)

        elif shape_type == "rectangle":
            x1, y1 = float(pts[0][0]), float(pts[0][1])
            x2, y2 = float(pts[1][0]), float(pts[1][1])
            draw.rectangle([x1, y1, x2, y2], fill=1)

        elif shape_type in ("polygon", "linestrip"):
            flat = [(float(x), float(y)) for x, y in pts]
            if len(flat) >= 3:
                draw.polygon(flat, fill=1)

        elif shape_type == "line" and len(pts) == 2:
            x1, y1 = float(pts[0][0]), float(pts[0][1])
            x2, y2 = float(pts[1][0]), float(pts[1][1])
            draw.line([x1, y1, x2, y2], fill=1, width=5)

        region = np.array(poly_img, dtype=bool)
        mask[region] = cls

    return mask


def mask_to_color(mask):
    """(H,W) → (H,W,3) RGB 彩色 mask"""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(PALETTE):
        rgb[mask == cls_id] = color
    return rgb


def main():
    all_pairs = []   # (img_path, json_path, mask_save_path)

    for folder in sorted(os.listdir(DATA_ROOT)):
        folder_path = os.path.join(DATA_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith(".jpg"):
                continue
            img_path  = os.path.join(folder_path, fname)
            json_path = img_path.replace(".jpg", ".json")
            stem      = os.path.splitext(fname)[0]
            mask_name = f"{folder}_{stem}_mask.png"
            mask_path = os.path.join(MASK_DIR, mask_name)
            all_pairs.append((img_path, json_path, mask_path, folder))

    print(f"共 {len(all_pairs)} 張圖片，開始轉換...")

    converted = 0
    for img_path, json_path, mask_path, folder in all_pairs:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        if os.path.exists(json_path):
            mask = json_to_mask(json_path, w, h)
        else:
            # good 圖片：全 0 mask
            mask = np.zeros((h, w), dtype=np.uint8)

        Image.fromarray(mask).save(mask_path)
        converted += 1
        if converted % 100 == 0:
            print(f"  {converted}/{len(all_pairs)} 完成...")

    print(f"全部轉換完成：{converted} 張 mask 儲存至 {MASK_DIR}")

    # ── 驗證：每個類別各抽一張 ──────────────────────────────────
    print("\n產生驗證圖（每類別各一張）...")
    import random
    random.seed(42)

    # 按 folder 分組，每個 folder 抽一張
    from collections import defaultdict
    folder_samples = defaultdict(list)
    for item in all_pairs:
        folder_samples[item[3]].append(item)

    verify_samples = []
    for folder_name in sorted(folder_samples.keys()):
        verify_samples.append(random.choice(folder_samples[folder_name]))

    for i, (img_path, json_path, mask_path, folder) in enumerate(verify_samples):
        img  = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        color_mask = mask_to_color(mask)
        overlay    = (img * 0.6 + color_mask * 0.4).astype(np.uint8)

        present = sorted(set(mask.flatten().tolist()) - {0})
        labels  = [CLASS_NAMES[c] for c in present] if present else ["背景（good）"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img);         axes[0].set_title("原圖");        axes[0].axis("off")
        axes[1].imshow(color_mask);  axes[1].set_title("Mask");        axes[1].axis("off")
        axes[2].imshow(overlay);     axes[2].set_title("Overlay");     axes[2].axis("off")

        fig.suptitle(f"[{folder}]  類別：{', '.join(labels)}", fontsize=10)
        plt.tight_layout()
        out = os.path.join(VIS_DIR, f"{i:02d}_{folder}.png")
        plt.savefig(out, dpi=100)
        plt.close()

    print(f"驗證圖儲存至：{VIS_DIR}")
    print("\nM2 完成！")


if __name__ == "__main__":
    main()
