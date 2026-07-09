"""Jittor 版 Multi-codebook Quantization.

本文件对应 UniTok 中的 MCQ 量化模块：连续视觉 token 先按通道切成
多个子向量，每个子向量分别在独立码本中做最近邻查找，最后再拼回完整
量化特征。这里保留论文核心公式和接口，同时加入一个轻量的可微
codebook loss，保证 Windows/CPU Jittor demo 中码本也能稳定更新。
"""

from __future__ import annotations

from typing import Dict, Tuple

import jittor as jt
import numpy as np
from jittor import nn


def _l2_normalize(x: jt.Var, dim: int = -1, eps: float = 1e-6) -> jt.Var:
    """沿指定维度做 L2 归一化，用于可选的余弦近邻量化。"""

    return x / jt.sqrt((x * x).sum(dim=dim, keepdims=True) + eps)


class MultiCodebookQuantizer(nn.Module):
    """多码本向量量化器。

    输入输出 shape:
        features: `[B, N, C]`，B 为 batch，N 为视觉 token 数，C 为 latent_dim。
        indices: `[B, num_codebooks, N]`，每个 token 有 num_codebooks 个离散码。
        quantized: `[B, N, C]`，与输入 latent feature 维度一致。

    论文公式对应:
        f = concat(f_1, ..., f_M)
        z_i = argmin_{z in Z_i} ||f_i - z||_2
        hat_f = concat(z_1, ..., z_M)

    也就是把一个 C 维 token 切成 M 个 C/M 维 chunk，每个 chunk 使用独立
    子码本 Z_i 量化。比如 latent_dim=64、num_codebooks=8 时，一个 token
    被切成 8 个 8 维子向量，最终得到 8 个 code index。
    """

    def __init__(
        self,
        latent_dim: int = 64,
        num_codebooks: int = 4,
        codebook_size: int = 128,
        beta: float = 0.25,
        normalize: bool = True,
        soft_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if latent_dim % num_codebooks != 0:
            raise ValueError("latent_dim must be divisible by num_codebooks.")
        self.latent_dim = latent_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.chunk_dim = latent_dim // num_codebooks
        self.beta = beta
        self.normalize = normalize
        self.soft_temperature = soft_temperature
        self.codebooks = nn.ParameterList(
            [jt.randn((codebook_size, self.chunk_dim), dtype="float32") * 0.02 for _ in range(num_codebooks)]
        )

    def _nearest_code(self, chunk: jt.Var, codebook: jt.Var) -> Tuple[jt.Var, jt.Var]:
        """用单个码本量化一个通道 chunk。

        输入:
            chunk: `[B, N, D]`，来自第 i 个通道切片 f_i。
            codebook: `[K, D]`，第 i 个子码本 Z_i，K 为码字数量。

        输出:
            quantized: `[B, N, D]`，最近邻码字 z_i。
            indices: `[B, N]`，每个 token 在该子码本中的离散 id。
        """

        bsz, num_tokens, dim = chunk.shape
        flat = chunk.reshape((-1, dim))
        if self.normalize:
            flat_for_search = _l2_normalize(flat, dim=-1)
            book_for_search = _l2_normalize(codebook, dim=-1)
        else:
            flat_for_search = flat
            book_for_search = codebook
        # 最近邻搜索对应论文中的 argmin，离散选择不可导；这里把搜索路径截断，
        # 后续用 straight-through estimator 和辅助 codebook loss 传递梯度。
        flat_for_search = flat_for_search.stop_grad()
        book_for_search = book_for_search.stop_grad()
        x2 = (flat_for_search * flat_for_search).sum(dim=1, keepdims=True)
        z2 = (book_for_search * book_for_search).sum(dim=1).reshape((1, -1))
        dist = x2 + z2 - 2.0 * (flat_for_search @ book_for_search.transpose())
        indices, _ = jt.argmin(dist, dim=1)
        indices_np = indices.numpy().astype("int32")
        # 常量 one-hot 表示离散选择；one_hot @ codebook 得到查表结果。
        one_hot = jt.array(np.eye(self.codebook_size, dtype="float32")[indices_np])
        indices = jt.array(indices_np).int32()
        quantized = one_hot @ codebook
        return quantized.reshape((bsz, num_tokens, dim)), indices.reshape((bsz, num_tokens))

    def _soft_codebook_loss(self, chunk: jt.Var, codebook: jt.Var) -> jt.Var:
        """轻量可微码本损失。

        论文中的 VQ 训练通常包含 commitment/codebook 两部分；hard argmin 本身
        不可导。本项目仍用 hard indices 作为最终离散 token，但额外用 softmax
        over distance 构造 soft_q，只服务于码本向量更新，不改变离散输出。

        输入 shape: chunk `[B, N, D]`，codebook `[K, D]`。
        输出 shape: 标量 loss。
        """

        _, _, dim = chunk.shape
        flat = chunk.reshape((-1, dim)).stop_grad()
        x2 = (flat * flat).sum(dim=1, keepdims=True)
        z2 = (codebook * codebook).sum(dim=1).reshape((1, -1))
        dist = x2 + z2 - 2.0 * (flat @ codebook.transpose())
        weights = nn.softmax(-dist / self.soft_temperature, dim=1)
        soft_q = weights @ codebook
        return ((soft_q - flat) ** 2).mean()

    def f_to_idx(self, features: jt.Var) -> jt.Var:
        """把连续 latent feature 编码成离散 code indices。

        输入:
            features: `[B, N, C]`。
        输出:
            indices: `[B, num_codebooks, N]`，每个子码本各给出一个 token id。
        """

        chunks = []
        for i in range(self.num_codebooks):
            start = i * self.chunk_dim
            end = (i + 1) * self.chunk_dim
            _, indices = self._nearest_code(features[:, :, start:end], self.codebooks[i])
            chunks.append(indices)
        return jt.stack(chunks, dim=1)

    def idx_to_f(self, indices: jt.Var) -> jt.Var:
        """把离散 code indices 查表还原为量化 feature。

        输入:
            indices: `[B, num_codebooks, N]`。
        输出:
            features: `[B, N, C]`，即 concat(z_1, ..., z_M)。
        """

        chunks = []
        for i in range(self.num_codebooks):
            idx = indices[:, i, :].reshape((-1,))
            idx_np = idx.numpy().astype("int32")
            one_hot = jt.array(np.eye(self.codebook_size, dtype="float32")[idx_np])
            chunk = one_hot @ self.codebooks[i]
            chunks.append(chunk.reshape((indices.shape[0], indices.shape[2], self.chunk_dim)))
        return jt.concat(chunks, dim=-1)

    def execute(self, features: jt.Var):
        """执行 MCQ 前向量化。

        输入:
            features: `[B, N, C]`，Attention Projection 压缩后的 latent token。

        输出:
            quantized_st: `[B, N, C]`，用于解码器训练的 straight-through 量化特征。
            indices: `[B, num_codebooks, N]`，离散视觉 token。
            commitment_loss: 标量，约束 encoder 输出靠近选中码字。
            codebook_loss: 标量，轻量实现中用于更新码本。
            usage: dict，包含 codebook 使用率和 perplexity。

        论文对应关系:
            逐个子码本完成 f_i -> z_i，再 concat 得到 hat_f。ST 写法
            features + stop_grad(quantized - features) 保证前向使用离散码字，
            反向仍把梯度传给 encoder/attention projection。
        """

        quantized_chunks = []
        index_chunks = []
        commitment_loss = jt.zeros((1,), dtype="float32")
        codebook_loss = jt.zeros((1,), dtype="float32")
        for i in range(self.num_codebooks):
            start = i * self.chunk_dim
            end = (i + 1) * self.chunk_dim
            chunk = features[:, :, start:end]
            q, idx = self._nearest_code(chunk, self.codebooks[i])
            quantized_chunks.append(q)
            index_chunks.append(idx)
            commitment_loss = commitment_loss + ((q.stop_grad() - chunk) ** 2).mean() * self.beta
            codebook_loss = codebook_loss + self._soft_codebook_loss(chunk, self.codebooks[i])
        quantized = jt.concat(quantized_chunks, dim=-1)
        indices = jt.stack(index_chunks, dim=1)
        commitment_loss = commitment_loss / self.num_codebooks
        codebook_loss = codebook_loss / self.num_codebooks
        quantized_st = features + (quantized - features).stop_grad()
        usage = self._usage_stats(indices)
        return quantized_st, indices, commitment_loss, codebook_loss, usage

    def _usage_stats(self, indices: jt.Var) -> Dict[str, jt.Var]:
        flat = indices.reshape((-1,))
        one_hot = nn.one_hot(flat, self.codebook_size).float32()
        probs = one_hot.mean(dim=0)
        used = (probs > 0).float32().mean()
        entropy = -(probs * jt.log(probs + 1e-7)).sum()
        return {"usage": used, "perplexity": jt.exp(entropy)}
