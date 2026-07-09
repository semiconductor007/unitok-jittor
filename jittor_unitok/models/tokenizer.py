"""端到端 Tiny UniTok tokenizer。

模型顺序保持与 UniTok 核心思想一致：
image -> encoder -> attention compression -> MCQ -> attention expansion -> decoder。
同时暴露 encode/decode/reconstruct，便于展示离散视觉 token 如何接入后续
统一多模态模型。
"""

from __future__ import annotations

from typing import Dict

import jittor as jt
from jittor import nn

from .attention_projection import ChannelCompressionBlock, ChannelExpansionBlock
from .decoder import TinyDecoder
from .encoder import TinyEncoder
from .losses import reconstruction_loss, vq_loss
from .mcq import MultiCodebookQuantizer


class UniTokTokenizer(nn.Module):
    """Jittor tiny tokenizer 主模型。

    组件对应关系:
        encoder: 图像 `[B,3,H,W]` -> token `[B,N,hidden_dim]`。
        quant_proj: compression `[B,N,hidden_dim]` -> `[B,N,latent_dim]`。
        quantizer: MCQ 产生离散 indices `[B,num_codebooks,N]`。
        post_quant_proj: expansion `[B,N,latent_dim]` -> `[B,N,hidden_dim]`。
        decoder: token `[B,N,hidden_dim]` -> 重建图像 `[B,3,H,W]`。

    本类复现 tokenizer 结构和训练接口；理解侧的大规模 CLIP/MLLM 接入在本项目
    中以 image_features 和文档说明形式保留，不进行 DataComp-1B 级训练。
    """

    def __init__(
        self,
        image_size: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_codebooks: int = 4,
        codebook_size: int = 128,
        beta: float = 0.25,
        num_heads: int = 4,
        recon_loss_type: str = "l1",
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.recon_loss_type = recon_loss_type
        self.encoder = TinyEncoder(image_size=image_size, hidden_dim=hidden_dim)
        self.quant_proj = ChannelCompressionBlock(hidden_dim, latent_dim, num_heads=num_heads)
        self.quantizer = MultiCodebookQuantizer(latent_dim, num_codebooks, codebook_size, beta)
        self.post_quant_proj = ChannelExpansionBlock(latent_dim, hidden_dim, num_heads=num_heads)
        self.decoder = TinyDecoder(image_size=image_size, hidden_dim=hidden_dim)
        self.feature_norm = nn.LayerNorm(hidden_dim)
        self.visual_proj = nn.Linear(hidden_dim, latent_dim)

    def execute(self, images: jt.Var) -> Dict[str, jt.Var]:
        """完整前向传播。

        输入:
            images: `[B, 3, image_size, image_size]`，归一化到 `[-1, 1]`。

        输出 dict:
            recon: `[B, 3, image_size, image_size]`，重建图像。
            tokens: `[B, N, hidden_dim]`，encoder 连续 token。
            latent: `[B, N, latent_dim]`，compression 后用于量化的 feature。
            quantized: `[B, N, latent_dim]`，ST estimator 后的量化 feature。
            indices: `[B, num_codebooks, N]`，离散视觉 token。
            recon_loss/vq_loss: 标量训练损失。
            image_features: `[B, latent_dim]`，用于 tiny contrastive demo 的 pooled 特征。
        """

        # 1. 图像编码为连续视觉 token，对应论文中的 vision encoder。
        tokens = self.encoder(images)
        # 2. Attention Projection compression: hidden_dim -> latent_dim。
        latent = self.quant_proj(tokens)
        # 3. MCQ 多码本量化，得到量化 feature 和离散 code indices。
        quantized, indices, commitment, codebook, stats = self.quantizer(latent)
        # 4. Attention Projection expansion 后送入 decoder 重建图像。
        decoded_tokens = self.post_quant_proj(quantized)
        recon = self.decoder(decoded_tokens)
        rec_loss = reconstruction_loss(recon, images, self.recon_loss_type)
        quant_loss = vq_loss(commitment, codebook)
        pooled = self.feature_norm(decoded_tokens).mean(dim=1)
        image_features = self.visual_proj(pooled)
        return {
            "recon": recon,
            "tokens": tokens,
            "latent": latent,
            "quantized": quantized,
            "indices": indices,
            "commitment_loss": commitment,
            "codebook_loss": codebook,
            "vq_loss": quant_loss,
            "recon_loss": rec_loss,
            "codebook_usage": stats["usage"],
            "perplexity": stats["perplexity"],
            "image_features": image_features,
        }

    def encode(self, images: jt.Var) -> jt.Var:
        """只执行编码和 MCQ 查表，返回离散 token。

        输入:
            images: `[B, 3, image_size, image_size]`。
        输出:
            indices: `[B, num_codebooks, N]`。
        """

        tokens = self.encoder(images)
        latent = self.quant_proj(tokens)
        return self.quantizer.f_to_idx(latent)

    def decode(self, indices: jt.Var) -> jt.Var:
        """从离散 code indices 解码图像。

        输入:
            indices: `[B, num_codebooks, N]`。
        输出:
            recon: `[B, 3, image_size, image_size]`。
        """

        quantized = self.quantizer.idx_to_f(indices)
        decoded_tokens = self.post_quant_proj(quantized)
        return self.decoder(decoded_tokens)

    def reconstruct(self, images: jt.Var) -> jt.Var:
        """图像重建便捷接口。

        输入 shape: `[B, 3, image_size, image_size]`。
        输出 shape: `[B, 3, image_size, image_size]`。
        """

        return self.execute(images)["recon"]
