"""Attention Projection: 用注意力完成通道压缩与恢复。

UniTok 论文 Appendix A Figure 6 用 Attention Projection 替代简单线性层，
在保持 token 数 N 不变的前提下改变通道维度。本 tiny 复现实现两个方向：
compression: `[B, N, C] -> [B, N, c]`
expansion: `[B, N, c] -> [B, N, C]`
其中 B 是 batch，N 是图像 patch/token 数，C 是 encoder hidden dim，c 是
量化前的低维 latent dim。
"""

from __future__ import annotations

import math

import jittor as jt
from jittor import nn


def gelu(x: jt.Var) -> jt.Var:
    """用 Jittor 基础算子实现 GELU，避免额外依赖。"""

    return 0.5 * x * (1.0 + jt.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


class FeedForward(nn.Module):
    """注意力后的两层 MLP。

    输入/输出 shape 都是 `[B, N, dim]`，只在每个 token 的通道维上做变换。
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def execute(self, x: jt.Var) -> jt.Var:
        """输入 `[B, N, dim]`，输出 `[B, N, dim]`。"""

        return self.fc2(gelu(self.fc1(x)))


class ChannelAttentionProjection(nn.Module):
    """可改变通道数的 scaled dot-product attention。

    输入:
        x: `[B, N, C_in]`。

    输出:
        y: `[B, N, C_out]`。

    计算逻辑:
        1. 对输入做 LayerNorm。
        2. q/k/v 都从 C_in 投影到 C_out，并拆成 num_heads 个头。
        3. 在 token 维度 N 上计算 attention，让每个视觉 token 汇聚其它 token。
        4. 用 residual Linear(C_in -> C_out) 对齐维度，再接 MLP。

    这对应论文中的 Attention Projection：不是直接逐 token 线性压缩，而是让
    token 间先交互，再完成通道 compression 或 expansion。
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads.")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(in_dim)
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.proj = nn.Linear(out_dim, out_dim)
        self.residual = nn.Linear(in_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = FeedForward(out_dim, int(out_dim * mlp_ratio))

    def execute(self, x: jt.Var) -> jt.Var:
        """前向传播。

        输入:
            x: `[B, N, C_in]`。
        输出:
            y: `[B, N, C_out]`。
        """

        # q/k/v: [B, heads, N, head_dim]，attention 在 N 维上计算。
        bsz, num_tokens, _ = x.shape
        y = self.norm1(x)
        q = self.q(y).reshape((bsz, num_tokens, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        k = self.k(y).reshape((bsz, num_tokens, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        v = self.v(y).reshape((bsz, num_tokens, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        scores = (q @ k.transpose((0, 1, 3, 2))) * self.scale
        attn = nn.softmax(scores, dim=-1)
        y = attn @ v
        y = y.transpose((0, 2, 1, 3)).reshape((bsz, num_tokens, self.out_dim))
        y = self.proj(y) + self.residual(x)
        y = y + self.mlp(self.norm2(y))
        return y


class ChannelCompressionBlock(ChannelAttentionProjection):
    """通道压缩块。

    论文含义:
        compression 将 encoder token 从高维 C 压到低维 c，降低量化空间维度，
        缓解统一 tokenizer 中的 quantization bottleneck。

    输入 shape: `[B, N, in_dim]`。
    输出 shape: `[B, N, compressed_dim]`。
    """

    def __init__(self, in_dim: int, compressed_dim: int, num_heads: int = 4) -> None:
        super().__init__(in_dim=in_dim, out_dim=compressed_dim, num_heads=num_heads)


class ChannelExpansionBlock(ChannelAttentionProjection):
    """通道恢复块。

    论文含义:
        expansion 将量化后的低维 token 从 c 恢复到 decoder 需要的 C 维，
        让离散 token 既能压缩表示，又能保留足够重建信息。

    输入 shape: `[B, N, compressed_dim]`。
    输出 shape: `[B, N, out_dim]`。
    """

    def __init__(self, compressed_dim: int, out_dim: int, num_heads: int = 4) -> None:
        super().__init__(in_dim=compressed_dim, out_dim=out_dim, num_heads=num_heads)
