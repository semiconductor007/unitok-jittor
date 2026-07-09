"""Tiny UniTok 复现中的损失函数。

论文完整目标包含重建、VQ、感知、对抗和图文对比监督：

    L_recon = L_R + lambda_VQ L_VQ + lambda_P L_P + lambda_G L_G
    L = L_recon + lambda_contra L_contra

本项目的默认训练为了在 CPU/普通 GPU 上稳定跑通，只启用：

    L = L_R + lambda_VQ L_VQ

其中 L_R 是像素 L1/MSE 重建损失，L_VQ 是 commitment loss 与轻量 codebook
loss 的和。LPIPS、GAN 和大规模 CLIP text tower 不作为默认训练依赖。
"""

from __future__ import annotations

import jittor as jt
from jittor import nn


def reconstruction_loss(pred: jt.Var, target: jt.Var, loss_type: str = "l1") -> jt.Var:
    """像素重建损失 L_R。

    输入:
        pred: `[B, 3, H, W]`，decoder 输出。
        target: `[B, 3, H, W]`，原图。

    输出:
        标量 loss。默认 L1 更稳定；也支持 MSE 方便指标对齐。
    """

    # L_R: 默认使用 L1，训练时对小模型更稳定，也更不容易被少数异常像素主导。
    if loss_type == "l1":
        return jt.abs(pred - target).mean()
    # MSE 分支主要用于和 eval_reconstruction.py 中的 MSE/PSNR 指标对齐。
    if loss_type == "mse":
        return ((pred - target) ** 2).mean()
    raise ValueError(f"Unsupported reconstruction loss: {loss_type}")


def vq_loss(commitment_loss: jt.Var, codebook_loss: jt.Var) -> jt.Var:
    """VQ 损失。

    输入:
        commitment_loss: 标量，约束 encoder latent 靠近选中码字。
        codebook_loss: 标量，更新码本向量。

    输出:
        标量 `L_VQ`。完整论文还会配合感知/对抗损失；tiny 训练只使用这里的
        VQ 项与像素重建项。
    """

    # L_VQ = commitment loss + codebook loss；lambda_VQ 在训练循环中统一加权。
    return commitment_loss + codebook_loss


def contrastive_loss(image_features: jt.Var, text_features: jt.Var, temperature: float = 0.07) -> jt.Var:
    """简化 CLIP 风格双向对比损失。

    输入:
        image_features: `[B, D]`，tokenizer pooled visual feature。
        text_features: `[B, D]`，真实或 dummy text embedding。

    输出:
        标量 `L_contra`。

    说明:
        UniTok 论文使用大规模图文对进行对比学习；本函数保留公式接口和 demo
        能力，但默认训练没有真实 caption，因此不强制启用。
    """

    # CLIP 风格对比学习先做 L2 normalize，使相似度退化为余弦相似度。
    image_norm = jt.sqrt((image_features * image_features).sum(dim=-1, keepdims=True) + 1e-6)
    text_norm = jt.sqrt((text_features * text_features).sum(dim=-1, keepdims=True) + 1e-6)
    image_features = image_features / image_norm
    text_features = text_features / text_norm

    # logits[i, j] 表示第 i 张图与第 j 条文本的匹配分数；对角线是正样本。
    logits = image_features @ text_features.transpose() / temperature
    labels = jt.arange(logits.shape[0])
    loss_i2t = nn.cross_entropy_loss(logits, labels)
    loss_t2i = nn.cross_entropy_loss(logits.transpose(), labels)
    return 0.5 * (loss_i2t + loss_t2i)


class PatchDiscriminator(nn.Module):
    """可选 PatchGAN 风格轻量判别器。

    输入 shape: `[B, 3, H, W]`。
    输出 shape: `[B, 1, H/4, W/4]` 左右的 patch logits。

    说明:
        论文中的 lambda_G L_G 对重建质量很重要，但需要更复杂的训练调度和
        更多算力。本类只作为后续扩展占位，默认 tiny setting 不启用。
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm(base_channels * 2),
            nn.LeakyReLU(0.2),
            nn.Conv(base_channels * 2, 1, 3, stride=1, padding=1),
        )

    def execute(self, x: jt.Var) -> jt.Var:
        """输入图像 `[B,3,H,W]`，输出 patch 判别 logits。"""

        return self.net(x)
