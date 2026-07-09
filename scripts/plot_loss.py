"""从训练 CSV 绘制 loss 曲线。

训练脚本每个 step 向 `outputs/curves/loss.csv` 追加 total/recon/vq loss。
本脚本读取 CSV 并输出 `outputs/curves/loss.png`，供 README 和 PPT 直接引用。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    """读取 loss.csv 并保存 PNG 曲线图。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="outputs/curves/loss.csv")
    parser.add_argument("--output", default="outputs/curves/loss.png")
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.csv, "r", encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"No rows found in {args.csv}")
    steps = [int(r["step"]) for r in rows]
    for key in ("total_loss", "recon_loss", "vq_loss"):
        plt.plot(steps, [float(r[key]) for r in rows], label=key)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
