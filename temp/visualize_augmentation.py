"""
可视化不同 brightness / contrast 范围对同一帧图像的效果。
运行: python temp/visualize_augmentation.py
输出: temp/aug_visualization.png
"""

import sys
sys.path.insert(0, "/mnt/data/xidong_data/tac_infra")

from PIL import Image
import torchvision.transforms.v2 as v2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 载入样本帧 ──────────────────────────────────────────────────────────────
img_path = "/mnt/data/xidong_data/tac_infra/temp/sample_frame.jpg"
img_pil = Image.open(img_path).convert("RGB")
# 缩小一点加速处理
img_pil = img_pil.resize((480, 270))

def pil_to_tensor(img):
    return v2.functional.to_image(img)          # uint8 [3, H, W]

def tensor_to_np(t):
    return t.permute(1, 2, 0).numpy()

orig_t = pil_to_tensor(img_pil)

# ── 要测试的范围 ─────────────────────────────────────────────────────────────
# 每行: (label, brightness_range, contrast_range)
configs = [
    ("原图 (无增强)",        None,           None),
    ("当前默认\nb=(0.8,1.2) c=(0.8,1.2)",  (0.8, 1.2), (0.8, 1.2)),
    ("稍强\nb=(0.5,1.5) c=(0.5,1.5)",      (0.5, 1.5), (0.5, 1.5)),
    ("激进\nb=(0.3,1.8) c=(0.4,1.8)",      (0.3, 1.8), (0.4, 1.8)),
    ("极端\nb=(0.1,2.0) c=(0.2,2.0)",      (0.1, 2.0), (0.2, 2.0)),
]

# 每种配置采样 N 次，展示变化范围
N_SAMPLES = 4
torch.manual_seed(42)

# ── 逐配置生成样本 ────────────────────────────────────────────────────────────
rows = len(configs)
cols = N_SAMPLES + 1   # +1 列用来放原图 / label

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.4))
fig.patch.set_facecolor("#1e1e1e")

for r, (label, b_range, c_range) in enumerate(configs):
    # 第 0 列放"原图"或 label 背景色块
    ax0 = axes[r][0]
    ax0.imshow(tensor_to_np(orig_t))
    ax0.set_title(label, fontsize=8, color="white", pad=4)
    ax0.axis("off")

    for c in range(1, cols):
        ax = axes[r][c]
        if b_range is None and c_range is None:
            # 原图行：其余列留空或显示一样的原图
            ax.imshow(tensor_to_np(orig_t))
            ax.set_title(f"原图 #{c}", fontsize=7, color="#aaaaaa", pad=3)
        else:
            # 对 brightness / contrast 分别独立 sample，模拟 RandomSubsetApply 随机选到两者的情况
            b_factor = torch.empty(1).uniform_(*b_range).item()
            c_factor = torch.empty(1).uniform_(*c_range).item()
            t = orig_t.clone()
            t = v2.functional.adjust_brightness(t, b_factor)
            t = v2.functional.adjust_contrast(t, c_factor)
            ax.imshow(tensor_to_np(t))
            ax.set_title(f"b={b_factor:.2f}  c={c_factor:.2f}", fontsize=7, color="#cccccc", pad=3)
        ax.axis("off")
        ax.set_facecolor("#1e1e1e")

plt.suptitle("Brightness & Contrast Augmentation 效果对比", color="white", fontsize=12, y=1.01)
plt.tight_layout(pad=0.5)

out_path = "/mnt/data/xidong_data/tac_infra/temp/aug_visualization.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
