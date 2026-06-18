"""
M1：資料盤點 + 批次標籤
從檔名 Roi1_MMDDhhmmss_* 抽取日期，畫「日期 × 類別」分布表
"""
import os
import re
from collections import defaultdict
import matplotlib
matplotlib.rcParams["font.family"] = ["Microsoft JhengHei", "DejaVu Sans"]
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = r"D:\coca\steel ball dataset"
OUT_DIR   = r"D:\coca\smp\figures"
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_MAP = {
    "100小黑傷": "100小黑傷",
    "101灰傷、刻痕": "101灰傷刻痕",
    "102麻點": "102麻點",
    "103大黑傷": "103大黑傷",
    "104研磨傷": "104研磨傷",
    "105肯傷": "105肯傷",
    "106刮傷": "106刮傷",
    "107生鏽": "107生鏽",
    "108霧面": "108霧面",
    "109亮傷-暗": "109亮傷暗",
    "110小白點、線": "110小白點線",
    "111亮傷-亮": "111亮傷亮",
    "good": "good",
}


def extract_date(filename):
    """從 Roi1_MMDDhhmmss_* 抽取 MMDD"""
    m = re.search(r'Roi1_(\d{4})', filename)
    return m.group(1) if m else "unknown"


def main():
    # 收集所有樣本
    records = []   # (class_name, date, filepath)
    class_counts = defaultdict(int)
    date_class_counts = defaultdict(lambda: defaultdict(int))

    for folder in sorted(os.listdir(DATA_ROOT)):
        folder_path = os.path.join(DATA_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue
        cls_name = LABEL_MAP.get(folder, folder)
        for fname in os.listdir(folder_path):
            if not fname.endswith(".jpg"):
                continue
            date = extract_date(fname)
            records.append((cls_name, date, os.path.join(folder_path, fname)))
            class_counts[cls_name] += 1
            date_class_counts[date][cls_name] += 1

    # ── 1. 各類別張數 ────────────────────────────────────────
    print(f"\n總圖片數：{len(records)}")
    print(f"\n{'類別':<20} {'張數':>6}")
    print("-" * 28)
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"{cls:<20} {cnt:>6}")

    # ── 2. 各日期批次有哪些類別 ──────────────────────────────
    dates = sorted(date_class_counts.keys())
    classes = list(LABEL_MAP.values())

    print(f"\n共 {len(dates)} 個日期批次：{dates}")

    # ── 3. 畫「日期 × 類別」分布熱圖 ────────────────────────
    matrix = np.zeros((len(classes), len(dates)), dtype=int)
    for j, date in enumerate(dates):
        for i, cls in enumerate(classes):
            matrix[i, j] = date_class_counts[date].get(cls, 0)

    fig, ax = plt.subplots(figsize=(max(10, len(dates)*0.8), 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_title("日期批次 × 類別 分布（張數）", fontsize=12)
    ax.set_xlabel("日期 (MMDD)")
    ax.set_ylabel("類別")
    plt.colorbar(im, ax=ax)

    # 在格子裡標數字
    for i in range(len(classes)):
        for j in range(len(dates)):
            v = matrix[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7,
                        color="black" if v < matrix.max()*0.6 else "white")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "m1_date_class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n儲存熱圖：{path}")

    # ── 4. 儲存批次清單 ──────────────────────────────────────
    out_txt = os.path.join(OUT_DIR, "m1_batch_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"總圖片數：{len(records)}\n\n")
        f.write("各類別張數：\n")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {cls:<20} {cnt}\n")
        f.write(f"\n各日期批次：\n")
        for date in dates:
            f.write(f"\n  [{date}]\n")
            for cls, cnt in sorted(date_class_counts[date].items()):
                f.write(f"    {cls:<20} {cnt}\n")
    print(f"儲存摘要：{out_txt}")


if __name__ == "__main__":
    main()
