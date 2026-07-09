"""生成 tiny 训练用几何图像数据集。

原论文使用 DataComp-1B 等大规模图文数据；本脚本在无网络、无真实数据时
生成彩色矩形/椭圆图像，保证训练、重建和可视化流程可以一键跑通。

输出目录结构:
    data/demo/train/*.jpg
    data/demo/val/*.jpg
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def draw_sample(path: Path, image_size: int, seed: int) -> None:
    """绘制一张彩色几何图。

    输入:
        path: 输出图片路径。
        image_size: 图片边长，默认 64。
        seed: 样本级随机种子，保证可复现。

    输出:
        保存 RGB jpg，shape 为 `[image_size, image_size, 3]`。
    """

    rng = random.Random(seed)
    bg = tuple(rng.randint(20, 235) for _ in range(3))
    img = Image.new("RGB", (image_size, image_size), bg)
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(4, 9)):
        color = tuple(rng.randint(0, 255) for _ in range(3))
        x0 = rng.randint(0, image_size - 8)
        y0 = rng.randint(0, image_size - 8)
        x1 = rng.randint(x0 + 4, image_size)
        y1 = rng.randint(y0 + 4, image_size)
        if rng.random() < 0.5:
            draw.rectangle((x0, y0, x1, y1), fill=color)
        else:
            draw.ellipse((x0, y0, x1, y1), fill=color)
    img.save(path)


def main() -> None:
    """生成 train/val 两个 split，供 ImageFolderDataset 读取。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/demo")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--val-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.output)
    for split, count in (("train", args.train_count), ("val", args.val_count)):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            draw_sample(split_dir / f"{split}_{i:04d}.jpg", args.image_size, args.seed + i + (10000 if split == "val" else 0))
    print(f"Demo data written to {root}")


if __name__ == "__main__":
    main()
