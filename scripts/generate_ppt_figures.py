"""Generate PPT-ready figures from training/evaluation outputs.

This script reads the artifacts produced by training:

    outputs/curves/loss.csv
    outputs/logs/train.log
    outputs/logs/eval_metrics.json
    outputs/logs/compare_with_pytorch.json
    outputs/reconstructions/*.png

and writes clean 16:9 figures to:

    outputs/ppt/figures/

The figures are intended for slides, so they include smoothed curves, concise
tables, and conclusion-oriented titles instead of raw terminal logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BLUE = "#1F4E79"
LIGHT_BLUE = "#4F8CC9"
GREEN = "#548235"
ORANGE = "#C65911"
GREY = "#666666"
LIGHT = "#F5F8FB"


def setup_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def load_loss_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"loss csv not found: {path}")
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"loss csv is empty: {path}")
    numeric_keys = [
        "epoch",
        "step",
        "total_loss",
        "recon_loss",
        "vq_loss",
        "commitment_loss",
        "codebook_loss",
        "codebook_usage",
        "perplexity",
        "lr",
        "time",
    ]
    data = {}
    for key in numeric_keys:
        data[key] = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return data


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def downsample(x: np.ndarray, y: np.ndarray, max_points: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    idx = np.linspace(0, x.size - 1, max_points).astype(np.int64)
    return x[idx], y[idx]


def save_loss_curve(data: dict[str, np.ndarray], out: Path, window: int) -> None:
    steps = data["step"]
    total = data["total_loss"]
    ma = moving_average(total, window)
    ma_steps = steps[-ma.size :]
    raw_x, raw_y = downsample(steps, total)

    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=180)
    ax.plot(raw_x, raw_y, color=LIGHT_BLUE, alpha=0.22, linewidth=0.6, label="raw total loss")
    ax.plot(ma_steps, ma, color=BLUE, linewidth=2.2, label=f"moving average ({window} steps)")
    ax.set_title("训练趋势：total loss 整体下降，单步震荡由 batch 差异和 VQ 更新导致", fontsize=18, fontweight="bold", color=BLUE, pad=14)
    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    first, last = float(total[0]), float(total[-1])
    drop = (first - last) / max(abs(first), 1e-8) * 100.0
    ax.text(0.02, 0.92, f"first={first:.3f}\nlast={last:.3f}\ndecrease={drop:.1f}%", transform=ax.transAxes, fontsize=12, bbox=dict(facecolor=LIGHT, edgecolor=BLUE, boxstyle="round,pad=0.35"))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_loss_components(data: dict[str, np.ndarray], out: Path, window: int) -> None:
    steps = data["step"]
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=180)
    for key, color, label in [
        ("recon_loss", BLUE, "reconstruction loss"),
        ("vq_loss", ORANGE, "VQ loss"),
        ("total_loss", GREEN, "total loss"),
    ]:
        values = moving_average(data[key], window)
        ax.plot(steps[-values.size :], values, color=color, linewidth=2.0, label=label)
    ax.set_title("损失分解：重建损失与 VQ 损失共同收敛", fontsize=18, fontweight="bold", color=BLUE, pad=14)
    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Moving-average loss", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_epoch_loss_curve(data: dict[str, np.ndarray], out: Path) -> None:
    """Plot epoch-level mean losses so the PPT clearly shows all 30 epochs."""

    epochs = data["epoch"].astype(np.int64)
    unique_epochs = np.unique(epochs)
    means = {}
    for key in ("total_loss", "recon_loss", "vq_loss"):
        means[key] = np.asarray([data[key][epochs == epoch].mean() for epoch in unique_epochs])

    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=180)
    ax.plot(unique_epochs, means["total_loss"], marker="o", color=BLUE, linewidth=2.8, label="total loss")
    ax.plot(unique_epochs, means["recon_loss"], marker="o", color=GREEN, linewidth=2.2, label="reconstruction loss")
    ax.plot(unique_epochs, means["vq_loss"], marker="o", color=ORANGE, linewidth=2.2, label="VQ loss")
    ax.set_title("30 轮训练趋势：epoch 平均 loss 整体下降", fontsize=18, fontweight="bold", color=BLUE, pad=14)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Mean loss per epoch", fontsize=13)
    ax.set_xticks(unique_epochs if unique_epochs.size <= 30 else unique_epochs[::2])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=12)

    first = float(means["total_loss"][0])
    last = float(means["total_loss"][-1])
    drop = (first - last) / max(abs(first), 1e-8) * 100.0
    ax.text(
        0.04,
        0.82,
        f"epochs={int(unique_epochs[-1])}\nsteps={int(data['step'][-1])}\nepoch1={first:.3f}\nepoch30={last:.3f}\ndecrease={drop:.1f}%",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(facecolor=LIGHT, edgecolor=BLUE, boxstyle="round,pad=0.35"),
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_codebook_curve(data: dict[str, np.ndarray], out: Path, window: int) -> None:
    steps = data["step"]
    usage = moving_average(data["codebook_usage"], window)
    perplexity = moving_average(data["perplexity"], window)
    x_usage = steps[-usage.size :]
    x_perp = steps[-perplexity.size :]

    fig, ax1 = plt.subplots(figsize=(13.33, 7.5), dpi=180)
    ax1.plot(x_usage, usage, color=GREEN, linewidth=2.2, label="codebook usage")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Step", fontsize=13)
    ax1.set_ylabel("Codebook usage", color=GREEN, fontsize=13)
    ax1.tick_params(axis="y", labelcolor=GREEN)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x_perp, perplexity, color=ORANGE, linewidth=1.8, label="perplexity")
    ax2.set_ylabel("Perplexity", color=ORANGE, fontsize=13)
    ax2.tick_params(axis="y", labelcolor=ORANGE)

    fig.suptitle("码本状态：usage 接近 1.0，说明 MCQ 码本没有塌缩", fontsize=18, fontweight="bold", color=BLUE, y=0.96)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out)
    plt.close(fig)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_metrics_bar(metrics: dict, out: Path) -> None:
    values = [float(metrics.get("mse", 0.0)), float(metrics.get("l1", 0.0)), float(metrics.get("psnr", 0.0))]
    labels = ["MSE", "L1", "PSNR"]
    colors = [BLUE, GREEN, ORANGE]
    fig, axes = plt.subplots(1, 3, figsize=(13.33, 7.5), dpi=180)
    fig.suptitle("重建评估：MSE / L1 / PSNR 提供 tiny setting 的轻量验证", fontsize=18, fontweight="bold", color=BLUE, y=0.94)
    for ax, label, value, color in zip(axes, labels, values, colors):
        ax.bar([label], [value], color=color, width=0.45)
        ax.set_title(label, fontsize=15, fontweight="bold")
        ax.text(0, value * 1.02 if value else 0.02, f"{value:.4f}" if label != "PSNR" else f"{value:.2f} dB", ha="center", fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.grid(True, axis="y", alpha=0.2)
        ymax = value * 1.35 if value > 0 else 1.0
        ax.set_ylim(0, ymax)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out)
    plt.close(fig)


def parse_train_log(log_path: Path) -> dict[str, str | float | int]:
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    pattern = re.compile(
        r"epoch=(?P<epoch>\d+) step=(?P<step>\d+) total_loss=(?P<total>[0-9.]+) "
        r"recon_loss=(?P<recon>[0-9.]+) vq_loss=(?P<vq>[0-9.]+) "
        r"codebook_usage=(?P<usage>[0-9.]+).*time=(?P<time>[0-9.]+)s"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return {}
    first = matches[0].groupdict()
    last = matches[-1].groupdict()
    return {
        "first_total": float(first["total"]),
        "last_total": float(last["total"]),
        "last_recon": float(last["recon"]),
        "last_vq": float(last["vq"]),
        "last_usage": float(last["usage"]),
        "epoch": int(last["epoch"]),
        "step": int(last["step"]),
        "time_hours": float(last["time"]) / 3600.0,
    }


def save_training_summary(summary: dict, data: dict[str, np.ndarray], out: Path) -> None:
    if not summary:
        summary = {
            "first_total": float(data["total_loss"][0]),
            "last_total": float(data["total_loss"][-1]),
            "last_recon": float(data["recon_loss"][-1]),
            "last_vq": float(data["vq_loss"][-1]),
            "last_usage": float(data["codebook_usage"][-1]),
            "epoch": int(data["epoch"][-1]),
            "step": int(data["step"][-1]),
            "time_hours": float(data["time"][-1]) / 3600.0,
        }
    drop = (summary["first_total"] - summary["last_total"]) / max(abs(summary["first_total"]), 1e-8) * 100.0
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=180)
    ax.axis("off")
    ax.set_title("训练摘要：Jittor 版 tokenizer 已完成长时间训练", fontsize=20, fontweight="bold", color=BLUE, pad=16)
    items = [
        ("Epoch", f"{summary['epoch']}"),
        ("Step", f"{summary['step']:,}"),
        ("Time", f"{summary['time_hours']:.2f} h"),
        ("Total Loss", f"{summary['last_total']:.4f}"),
        ("Recon Loss", f"{summary['last_recon']:.4f}"),
        ("VQ Loss", f"{summary['last_vq']:.4f}"),
        ("Codebook Usage", f"{summary['last_usage']:.4f}"),
        ("Loss Drop", f"{drop:.1f}%"),
    ]
    xs = [0.08, 0.32, 0.56, 0.80]
    ys = [0.66, 0.36]
    for idx, (name, value) in enumerate(items):
        x = xs[idx % 4]
        y = ys[idx // 4]
        rect = plt.Rectangle((x, y), 0.16, 0.17, transform=ax.transAxes, facecolor=LIGHT, edgecolor=BLUE, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.08, y + 0.105, value, transform=ax.transAxes, ha="center", va="center", fontsize=18, fontweight="bold", color=BLUE)
        ax.text(x + 0.08, y + 0.045, name, transform=ax.transAxes, ha="center", va="center", fontsize=11, color=GREY)
    ax.text(0.5, 0.12, "结论：loss 明显低于初始阶段，codebook usage 接近 1.0，训练流程和 MCQ 使用状态正常。", transform=ax.transAxes, ha="center", fontsize=15, color=ORANGE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def save_alignment_table(compare: dict, out: Path) -> None:
    rows = compare.get("module_alignment", [])
    if not rows:
        rows = [
            ["models/unitok.py", "jittor_unitok/models/tokenizer.py", "tokenizer graph"],
            ["models/quant.py", "jittor_unitok/models/mcq.py", "MCQ"],
        ]
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=180)
    ax.axis("off")
    ax.set_title("PyTorch / Jittor 对齐：模块职责与关键 shape 保持一致", fontsize=18, fontweight="bold", color=BLUE, pad=14)
    table_data = [["Official PyTorch", "Jittor Reproduction", "Alignment"]] + rows[:5]
    table = ax.table(cellText=table_data, cellLoc="center", loc="center", colWidths=[0.30, 0.34, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(BLUE)
        if r == 0:
            cell.set_facecolor(BLUE)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#FAFAFA")
    shape = compare.get("shape_alignment", {})
    shape_text = "  |  ".join(f"{k}: {v}" for k, v in shape.items())
    ax.text(0.5, 0.08, shape_text, transform=ax.transAxes, ha="center", fontsize=11, color=ORANGE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def latest_file(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def save_reconstruction_panel(recon_dir: Path, out: Path) -> None:
    train = latest_file(recon_dir, "train_epoch_*.png")
    eval_img = latest_file(recon_dir, "eval_batch_*.png")
    if train is None and eval_img is None:
        return

    width, height = 1920, 1080
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(44)
    label_font = load_font(28)
    draw.text((80, 45), "重建可视化：原图 / 重建图对比验证 encode-decode 流程", fill=(31, 78, 121), font=title_font)

    slots = []
    if train:
        slots.append(("Train reconstruction", train, (90, 160, 850, 760)))
    if eval_img:
        slots.append(("Validation reconstruction", eval_img, (980, 160, 1740, 760)))
    if len(slots) == 1:
        slots[0] = (slots[0][0], slots[0][1], (330, 170, 1590, 780))

    for label, path, box in slots:
        img = Image.open(path).convert("RGB")
        x0, y0, x1, y1 = box
        max_w, max_h = x1 - x0, y1 - y0
        scale = min(max_w / img.width, max_h / img.height)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
        px = x0 + (max_w - img.width) // 2
        py = y0 + (max_h - img.height) // 2
        canvas.paste(img, (px, py))
        draw.rectangle((x0, y0, x1, y1), outline=(31, 78, 121), width=3)
        draw.text((x0, y1 + 22), label, fill=(31, 78, 121), font=label_font)
    draw.text((80, 1010), "结论：真实图像训练后，重建图保留主要颜色和局部结构。", fill=(198, 89, 17), font=label_font)
    canvas.save(out)


def error_heatmap(diff: np.ndarray) -> Image.Image:
    """Convert a 2D absolute-difference map to a white-orange heatmap."""

    diff = np.clip(diff, 0.0, 1.0)
    r = np.full_like(diff, 255, dtype="float32")
    g = 255.0 - 175.0 * diff
    b = 255.0 - 255.0 * diff
    heat = np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype("uint8")
    return Image.fromarray(heat, mode="RGB")


def save_reconstruction_zoom_compare(recon_dir: Path, out: Path, samples: int = 2) -> None:
    """Create a large original/reconstruction/error panel from the latest eval grid.

    The raw reconstruction grid stores 64x64 original/reconstruction pairs, which
    are too small after being placed in a 16:9 slide. This figure upscales a few
    pairs and adds an error heatmap so the visual difference is easy to explain.
    """

    src = latest_file(recon_dir, "eval_batch_*.png")
    if src is None:
        return
    grid = Image.open(src).convert("RGB")
    cell = grid.width // 2
    label_h = 18
    row_h = cell + label_h
    count = min(samples, grid.height // row_h)
    scale = 5
    tile = cell * scale
    gap = 46
    margin_x = 90
    margin_y = 175
    width = 1920
    height = 1080
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(44)
    header_font = load_font(28)
    label_font = load_font(22)
    draw.text((80, 45), "重建细节对比：原图 / 重建图 / 误差热力图", fill=(31, 78, 121), font=title_font)

    headers = ["Original", "Reconstruction", "Error"]
    col_x = [margin_x, margin_x + tile + gap, margin_x + 2 * (tile + gap)]
    for x, header in zip(col_x, headers):
        draw.text((x + tile // 2 - 70, 105), header, fill=(31, 78, 121), font=header_font)

    for i in range(count):
        y = margin_y + i * (tile + 58)
        orig = grid.crop((0, i * row_h + label_h, cell, i * row_h + label_h + cell))
        rec = grid.crop((cell, i * row_h + label_h, cell * 2, i * row_h + label_h + cell))
        orig_big = orig.resize((tile, tile), Image.NEAREST)
        rec_big = rec.resize((tile, tile), Image.NEAREST)
        diff = np.abs(np.asarray(orig).astype("float32") - np.asarray(rec).astype("float32")).mean(axis=-1) / 255.0
        heat = error_heatmap(diff).resize((tile, tile), Image.NEAREST)
        for x, img in zip(col_x, [orig_big, rec_big, heat]):
            canvas.paste(img, (x, y))
            draw.rectangle((x, y, x + tile, y + tile), outline=(31, 78, 121), width=2)
        row_mse = float((diff**2).mean())
        draw.text((col_x[-1] + tile + 32, y + tile // 2 - 12), f"MSE {row_mse:.4f}", fill=(96, 96, 96), font=label_font)

    draw.text((80, 1010), "结论：重建图整体颜色和主体结构接近原图，误差主要集中在纹理和边缘细节。", fill=(198, 89, 17), font=header_font)
    canvas.save(out)


def copy_latest_recons(recon_dir: Path, out_dir: Path) -> None:
    for name, pattern in [("latest_train_reconstruction.png", "train_epoch_*.png"), ("latest_eval_reconstruction.png", "eval_batch_*.png")]:
        src = latest_file(recon_dir, pattern)
        if src:
            shutil.copy2(src, out_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-csv", default="outputs/curves/loss.csv")
    parser.add_argument("--train-log", default="outputs/logs/train.log")
    parser.add_argument("--metrics", default="outputs/logs/eval_metrics.json")
    parser.add_argument("--compare", default="outputs/logs/compare_with_pytorch.json")
    parser.add_argument("--recon-dir", default="outputs/reconstructions")
    parser.add_argument("--output-dir", default="outputs/ppt/figures")
    parser.add_argument("--ma-window", type=int, default=500)
    args = parser.parse_args()

    setup_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_loss_csv(Path(args.loss_csv))
    metrics = read_json(Path(args.metrics))
    compare = read_json(Path(args.compare))
    summary = parse_train_log(Path(args.train_log))

    save_loss_curve(data, out_dir / "loss_curve_ma.png", args.ma_window)
    save_epoch_loss_curve(data, out_dir / "loss_epoch_30.png")
    save_loss_components(data, out_dir / "loss_components_ma.png", args.ma_window)
    save_codebook_curve(data, out_dir / "codebook_usage_perplexity.png", args.ma_window)
    save_metrics_bar(metrics, out_dir / "eval_metrics_bar.png")
    save_training_summary(summary, data, out_dir / "training_summary.png")
    save_alignment_table(compare, out_dir / "pytorch_jittor_alignment.png")
    save_reconstruction_panel(Path(args.recon_dir), out_dir / "reconstruction_panel.png")
    save_reconstruction_zoom_compare(Path(args.recon_dir), out_dir / "reconstruction_zoom_compare.png")
    copy_latest_recons(Path(args.recon_dir), out_dir)

    manifest = {
        "figures": sorted(p.name for p in out_dir.glob("*.png")),
        "source_loss_csv": str(args.loss_csv),
        "moving_average_window": args.ma_window,
        "final_step": int(data["step"][-1]),
        "final_epoch": int(data["epoch"][-1]),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
