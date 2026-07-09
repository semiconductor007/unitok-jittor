"""Tiny UniTok 图像解码器。

解码器接收量化并 expansion 后的 token 序列，将 `[B, N, C]` 重新折回
二维 feature map，再用转置卷积上采样回 RGB 图像。本实现服务于 tiny
复现实验；原论文大规模生成质量依赖更强的 decoder、感知损失和对抗训练。
"""

from __future__ import annotations

import jittor as jt
from jittor import nn


class UpBlock(nn.Module):
    """转置卷积上采样块。

    输入 shape: `[B, in_ch, H, W]`。
    输出 shape: `[B, out_ch, 2H, 2W]`。
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm(out_ch)

    def execute(self, x: jt.Var) -> jt.Var:
        """执行 2 倍上采样，输入 `[B, C, H, W]`，输出 `[B, C_out, 2H, 2W]`。"""

        return nn.relu(self.bn(self.deconv(x)))


class TinyDecoder(nn.Module):
    """把视觉 token 解码回 RGB 图像。

    输入:
        tokens: `[B, N, hidden_dim]`，其中 `N=(image_size/8)^2`。

    输出:
        recon: `[B, 3, image_size, image_size]`，通过 `tanh` 限制到 `[-1, 1]`。
    """

    def __init__(self, image_size: int = 64, out_channels: int = 3, hidden_dim: int = 128) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8.")
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.grid_size = image_size // 8
        self.net = nn.Sequential(
            nn.Conv(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            UpBlock(hidden_dim, hidden_dim // 2),
            UpBlock(hidden_dim // 2, hidden_dim // 4),
            nn.ConvTranspose(hidden_dim // 4, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def execute(self, tokens: jt.Var) -> jt.Var:
        """前向解码。

        输入 shape: `[B, N, hidden_dim]`。
        输出 shape: `[B, 3, image_size, image_size]`。
        """

        bsz, num_tokens, channels = tokens.shape
        expected = self.grid_size * self.grid_size
        if num_tokens != expected:
            raise ValueError(f"Expected {expected} tokens, got {num_tokens}.")
        feat = tokens.transpose((0, 2, 1)).reshape((bsz, channels, self.grid_size, self.grid_size))
        return jt.tanh(self.net(feat))
