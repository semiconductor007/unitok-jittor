"""评估 tiny UniTok tokenizer 的图像重建质量。

评估脚本加载训练 checkpoint，对 `data_root/val` 图像执行 reconstruct，
保存重建对比图并计算 MSE/L1/PSNR。论文中常用 rFID 等生成指标；本 tiny
复现用这些轻量指标验证 forward/decode 链路是否正确。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("nvcc_path", "")
import jittor as jt

from jittor_unitok.data import ImageFolderDataset, iter_image_batches
from jittor_unitok.models import UniTokTokenizer
from jittor_unitok.utils.image_utils import save_reconstruction_grid
from jittor_unitok.utils.metrics import reconstruction_metrics


def load_tokenizer_state(model: UniTokTokenizer, state: dict) -> None:
    """加载 tokenizer 权重，并单独处理 MCQ 的 ParameterList 码本。

    Jittor 的 `ParameterList` 在不同版本中 state_dict 加载行为不完全一致。
    这里先加载普通参数，再逐个 assign `quantizer.codebooks.i`，避免评估时
    codebook 丢失。
    """

    filtered = {k: v for k, v in state.items() if not k.startswith("quantizer.codebooks.")}
    model.load_state_dict(filtered)
    for i in range(model.quantizer.num_codebooks):
        key = f"quantizer.codebooks.{i}"
        if key in state:
            model.quantizer.codebooks[i].assign(state[key])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/unitok_last.pkl")
    parser.add_argument("--data-root", default="data/demo")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None, help="Limit evaluation batches for quick PPT visualization.")
    parser.add_argument("--save-interval", type=int, default=1, help="Save one reconstruction grid every N batches.")
    parser.add_argument("--progress-interval", type=int, default=20, help="Print progress every N batches.")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def evaluate() -> None:
    """运行验证集重建评估。

    输入:
        checkpoint 中包含模型 state_dict 和训练 config。
        val images shape 为 `[B, 3, H, W]`。

    输出:
        `outputs/reconstructions/eval_batch_*.png`：原图/重建网格。
        `outputs/logs/eval_metrics.json`：MSE、L1、PSNR 平均值。
    """

    args = parse_args()
    if args.no_cuda:
        jt.flags.use_cuda = 0
    ckpt = jt.load(args.checkpoint)
    cfg = ckpt.get("config", {})
    model = UniTokTokenizer(cfg.get("image_size", 64), cfg.get("hidden_dim", 128), cfg.get("latent_dim", 64), cfg.get("num_codebooks", 4), cfg.get("codebook_size", 64))
    load_tokenizer_state(model, ckpt["model"])
    model.eval()
    val_set = ImageFolderDataset(Path(args.data_root) / "val", cfg.get("image_size", 64), train=False)
    out_dir = Path(args.output_dir)
    recon_dir = out_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    metric_sum = {"mse": 0.0, "l1": 0.0, "psnr": 0.0}
    batches = 0
    total_batches = (len(val_set) + args.batch_size - 1) // args.batch_size
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)
    print(f"evaluating {total_batches} batches from {len(val_set)} validation images")
    for images_np, names in iter_image_batches(val_set, args.batch_size, shuffle=False):
        if args.max_batches is not None and batches >= args.max_batches:
            break
        images = jt.array(images_np)
        recon = model.reconstruct(images)
        metrics = reconstruction_metrics(recon, images)
        for key in metric_sum:
            metric_sum[key] += metrics[key]
        batches += 1
        if args.save_interval > 0 and (batches == 1 or batches % args.save_interval == 0):
            save_reconstruction_grid(images, recon, recon_dir / f"eval_batch_{batches}.png", names)
        if args.progress_interval > 0 and (batches == 1 or batches % args.progress_interval == 0 or batches == total_batches):
            print(f"eval batch {batches}/{total_batches}: mse={metrics['mse']:.6f} l1={metrics['l1']:.6f} psnr={metrics['psnr']:.2f}")
    averaged = {k: v / max(batches, 1) for k, v in metric_sum.items()}
    metric_path = out_dir / "logs" / "eval_metrics.json"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps(averaged, indent=2), encoding="utf-8")
    print(json.dumps(averaged, indent=2))


if __name__ == "__main__":
    evaluate()
