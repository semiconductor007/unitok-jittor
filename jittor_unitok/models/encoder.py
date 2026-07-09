"""Tiny UniTok 视觉编码器。

完整 UniTok 使用 ViTamin-L/16 级别视觉 backbone。本复现为了能在普通
CPU/单卡环境完成训练和展示，使用三层 stride=2 CNN 把图像压成 token grid，
但保留 `[B, 3, H, W] -> [B, N, C]` 的 tokenizer 接口，方便后续替换大模型。
"""

from __future__ import annotations

import jittor as jt
from jittor import nn


class ConvBlock(nn.Module):
    """轻量卷积块：Conv -> BatchNorm -> ReLU。

    输入 shape: `[B, in_ch, H, W]`。
    输出 shape: `[B, out_ch, H/stride, W/stride]`。
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm(out_ch)

    def execute(self, x: jt.Var) -> jt.Var:
        """执行卷积下采样，输入 `[B, C, H, W]`，输出 `[B, C_out, H', W']`。"""

        return nn.relu(self.bn(self.conv(x)))


class TinyEncoder(nn.Module):
    """轻量 CNN encoder，输出视觉 token 序列。

    输入:
        image: `[B, 3, image_size, image_size]`，像素已归一化到 `[-1, 1]`。

    输出:
        tokens: `[B, N, hidden_dim]`，其中 `N=(image_size/8)^2`。

    说明:
        原论文使用 ViTamin-L/16 提供强视觉特征；本项目使用 tiny CNN 只复现
        tokenizer 训练链路和 shape 关系，不声称复现论文大规模指标。
    """

    def __init__(self, image_size: int = 64, in_channels: int = 3, hidden_dim: int = 128) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8.")
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.grid_size = image_size // 8
        self.net = nn.Sequential(
            ConvBlock(in_channels, hidden_dim // 4, stride=2),
            ConvBlock(hidden_dim // 4, hidden_dim // 2, stride=2),
            ConvBlock(hidden_dim // 2, hidden_dim, stride=2),
            nn.Conv(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
        )

    def execute(self, x: jt.Var) -> jt.Var:
        """前向编码。

        输入 shape: `[B, 3, H, W]`。
        输出 shape: `[B, (H/8)*(W/8), hidden_dim]`。
        """

        feat = self.net(x)
        bsz, channels, height, width = feat.shape
        return feat.reshape((bsz, channels, height * width)).transpose((0, 2, 1))
