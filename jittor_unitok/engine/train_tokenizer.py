"""训练 tiny Jittor UniTok tokenizer。

训练流程:
    1. 从 YAML 和命令行合并配置。
    2. 读取 `data_root/train` 下的 demo 图像。
    3. 执行 tokenizer forward，优化 `L_R + lambda_vq * L_VQ`。
    4. 每个 step 追加 `outputs/curves/loss.csv`，用于画 loss 曲线。
    5. 每隔 log_interval 写 `outputs/logs/train.log`。
    6. 每个 epoch 保存 checkpoint 和 reconstruction grid。

本脚本默认支持 CPU 跑通小规模复现实验，不依赖 DataComp-1B 或大模型权重。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import yaml

os.environ.setdefault("nvcc_path", "")
import jittor as jt
from jittor import optim

from jittor_unitok.data import ImageFolderDataset, iter_image_batches
from jittor_unitok.models import UniTokTokenizer
from jittor_unitok.utils.image_utils import save_reconstruction_grid
from jittor_unitok.utils.logger import append_csv, setup_logger
from jittor_unitok.utils.seed import set_seed


CSV_FIELDS = ["epoch", "step", "total_loss", "recon_loss", "vq_loss", "commitment_loss", "codebook_loss", "codebook_usage", "perplexity", "lr", "time"]


def scalar(x) -> float:
    """把标量 Jittor Var 转成 Python float，便于写 log/CSV。"""

    return float(x.numpy().reshape(-1)[0])


def load_config(path: str | None) -> Dict[str, Any]:
    """读取 YAML 配置；没有传 config 时返回空字典。"""

    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default="data/demo")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-codebooks", type=int, default=None)
    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--lambda-vq", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Path to a saved unitok_epoch_*.pkl checkpoint.")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def merged_config(args: argparse.Namespace) -> Dict[str, Any]:
    """合并默认配置、YAML 配置和命令行参数。

    优先级:
        命令行参数 > YAML 文件 > 代码默认值。
    """

    cfg = {"data_root": "data/demo", "output_dir": "outputs", "epochs": 2, "batch_size": 4, "lr": 1e-3, "image_size": 64, "num_codebooks": 4, "codebook_size": 64, "latent_dim": 64, "hidden_dim": 128, "lambda_vq": 1.0, "log_interval": 1, "seed": 0}
    cfg.update(load_config(args.config))
    for key, value in {
        "data_root": args.data_root,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "image_size": args.image_size,
        "num_codebooks": args.num_codebooks,
        "codebook_size": args.codebook_size,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "lambda_vq": args.lambda_vq,
        "log_interval": args.log_interval,
        "seed": args.seed,
    }.items():
        if value is not None:
            cfg[key] = value
    return cfg


def load_tokenizer_state(model: UniTokTokenizer, state: dict) -> None:
    """加载模型权重，并兼容 MCQ ParameterList 码本。

    resume 训练时需要恢复 encoder/decoder/attention/quantizer 权重。Jittor 的
    ParameterList 在不同版本下直接 load_state_dict 可能漏掉 codebooks，所以
    这里复用评估脚本的稳妥做法，普通参数先加载，码本再逐个 assign。
    """

    filtered = {k: v for k, v in state.items() if not k.startswith("quantizer.codebooks.")}
    model.load_state_dict(filtered)
    for i in range(model.quantizer.num_codebooks):
        key = f"quantizer.codebooks.{i}"
        if key in state:
            model.quantizer.codebooks[i].assign(state[key])


def train() -> None:
    """执行完整训练并保存作业展示所需产物。

    关键保存逻辑:
        `outputs/logs/train.log` 保存人类可读训练摘要。
        `outputs/curves/loss.csv` 保存结构化 loss，供 `scripts/plot_loss.py` 画图。
        `outputs/checkpoints/unitok_epoch_*.pkl` 保存每轮权重。
        `outputs/checkpoints/unitok_last.pkl` 保存最后权重，评估脚本默认读取它。
        `outputs/reconstructions/train_epoch_*.png` 保存原图/重建对比图。

    主要张量 shape:
        images: `[B, 3, H, W]`。
        out["recon"]: `[B, 3, H, W]`。
        out["indices"]: `[B, num_codebooks, N]`。
    """

    args = parse_args()
    if args.no_cuda:
        jt.flags.use_cuda = 0
    cfg = merged_config(args)
    set_seed(int(cfg["seed"]))
    output_dir = Path(cfg["output_dir"])
    log_dir = output_dir / "logs"
    curve_dir = output_dir / "curves"
    recon_dir = output_dir / "reconstructions"
    ckpt_dir = output_dir / "checkpoints"
    for p in (log_dir, curve_dir, recon_dir, ckpt_dir):
        p.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir / "train.log")
    logger.info("config=%s", json.dumps(cfg, ensure_ascii=False))
    train_set = ImageFolderDataset(Path(cfg["data_root"]) / "train", cfg["image_size"], train=True)
    model = UniTokTokenizer(cfg["image_size"], cfg["hidden_dim"], cfg["latent_dim"], cfg["num_codebooks"], cfg["codebook_size"])
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    global_step = 0
    start_epoch = 1
    if args.resume:
        ckpt = jt.load(args.resume)
        load_tokenizer_state(model, ckpt["model"])
        global_step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        logger.info("resumed checkpoint: %s epoch=%d step=%d", args.resume, start_epoch - 1, global_step)
        if start_epoch > int(cfg["epochs"]):
            logger.info("resume checkpoint already reached target epochs=%s", cfg["epochs"])
            return
    start_time = time.time()
    for epoch in range(start_epoch, int(cfg["epochs"]) + 1):
        last_batch = None
        for images_np, names in iter_image_batches(train_set, cfg["batch_size"], shuffle=True, drop_last=False):
            images = jt.array(images_np)
            out = model(images)
            total_loss = out["recon_loss"] + cfg["lambda_vq"] * out["vq_loss"]
            optimizer.step(total_loss)
            global_step += 1
            last_batch = (images, out["recon"], names)
            row = {
                "epoch": epoch,
                "step": global_step,
                "total_loss": scalar(total_loss),
                "recon_loss": scalar(out["recon_loss"]),
                "vq_loss": scalar(out["vq_loss"]),
                "commitment_loss": scalar(out["commitment_loss"]),
                "codebook_loss": scalar(out["codebook_loss"]),
                "codebook_usage": scalar(out["codebook_usage"]),
                "perplexity": scalar(out["perplexity"]),
                "lr": cfg["lr"],
                "time": round(time.time() - start_time, 3),
            }
            append_csv(curve_dir / "loss.csv", row, CSV_FIELDS)
            if global_step % int(cfg["log_interval"]) == 0:
                logger.info("epoch=%d step=%d total_loss=%.6f recon_loss=%.6f vq_loss=%.6f codebook_usage=%.4f lr=%.6g time=%.2fs", epoch, global_step, row["total_loss"], row["recon_loss"], row["vq_loss"], row["codebook_usage"], row["lr"], row["time"])
        ckpt_path = ckpt_dir / f"unitok_epoch_{epoch}.pkl"
        jt.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "step": global_step}, str(ckpt_path))
        logger.info("saved checkpoint: %s", ckpt_path)
        if last_batch is not None:
            images, recon, names = last_batch
            vis_path = recon_dir / f"train_epoch_{epoch}.png"
            save_reconstruction_grid(images, recon, vis_path, names)
            logger.info("saved reconstruction: %s", vis_path)
    jt.save({"model": model.state_dict(), "config": cfg, "epoch": cfg["epochs"], "step": global_step}, str(ckpt_dir / "unitok_last.pkl"))
    logger.info("training finished")


if __name__ == "__main__":
    train()
