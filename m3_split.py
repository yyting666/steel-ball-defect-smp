"""
M3：Stratified Random Split（60/20/20，seed=42）
按類別分層抽樣，確保每個 set 都有所有類別
輸出：train.txt / val.txt / test.txt（每行：img_path mask_path）
"""
import os
import random
from collections import defaultdict

DATA_ROOT = r"D:\coca\steel ball dataset"
MASK_DIR  = r"D:\coca\smp\masks"
OUT_DIR   = r"D:\coca\smp"

LABEL_MAP_FOLDER = {
    "100小黑傷":    1,
    "101灰傷、刻痕": 2,
    "102麻點":      3,
    "103大黑傷":    4,
    "104研磨傷":    5,
    "105肯傷":      6,
    "106刮傷":      7,
    "107生鏽":      8,
    "108霧面":      9,
    "109亮傷-暗":   10,
    "110小白點、線": 11,
    "111亮傷-亮":   12,
    "good":         0,
}

random.seed(42)


def main():
    # 收集所有 (img_path, mask_path, class_id)
    class_samples = defaultdict(list)

    for folder in sorted(os.listdir(DATA_ROOT)):
        folder_path = os.path.join(DATA_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue
        cls_id = LABEL_MAP_FOLDER.get(folder, -1)
        if cls_id < 0:
            continue

        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith(".jpg"):
                continue
            img_path  = os.path.join(folder_path, fname)
            stem      = os.path.splitext(fname)[0]
            mask_name = f"{folder}_{stem}_mask.png"
            mask_path = os.path.join(MASK_DIR, mask_name)

            if not os.path.exists(mask_path):
                print(f"  [警告] mask 不存在：{mask_path}")
                continue

            class_samples[cls_id].append((img_path, mask_path))

    train_list, val_list, test_list = [], [], []

    print("\n各類別分層切分：")
    print(f"{'cls':>4}  {'total':>6}  {'train':>6}  {'val':>5}  {'test':>5}")
    print("-" * 35)

    for cls_id in sorted(class_samples.keys()):
        samples = class_samples[cls_id]
        random.shuffle(samples)
        n = len(samples)
        n_train = int(n * 0.6)
        n_val   = int(n * 0.2)
        # test 取剩餘（避免因 int() 丟失樣本）
        n_test  = n - n_train - n_val

        train_list.extend(samples[:n_train])
        val_list.extend(samples[n_train:n_train+n_val])
        test_list.extend(samples[n_train+n_val:])

        print(f"  {cls_id:>2}  {n:>6}  {n_train:>6}  {n_val:>5}  {n_test:>5}")

    print("-" * 35)
    print(f"{'合計':>4}  {len(train_list)+len(val_list)+len(test_list):>6}  "
          f"{len(train_list):>6}  {len(val_list):>5}  {len(test_list):>5}")

    # 寫入 txt 檔
    for name, lst in [("train", train_list), ("val", val_list), ("test", test_list)]:
        path = os.path.join(OUT_DIR, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for img_p, mask_p in lst:
                f.write(f"{img_p}\t{mask_p}\n")
        print(f"\n儲存：{path}  ({len(lst)} 筆)")

    print("\nM3 完成！")


if __name__ == "__main__":
    main()
