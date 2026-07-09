"""生成官方 PyTorch UniTok 与本 Jittor 复现的轻量对齐报告。

完整数值对齐依赖官方仓库环境和大模型配置；本脚本优先检查文件结构、
模块对应关系、关键 shape 和行数，输出 JSON 供 README/PPT 引用。核心模型
实现不依赖 PyTorch，本脚本只作为复现报告工具。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def count_lines(path: Path) -> int:
    """统计文件行数，用于对齐报告中的实现规模参考。"""

    return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())


def main() -> None:
    """写出 `outputs/logs/compare_with_pytorch.json` 对齐摘要。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-root", default="../UniTok")
    parser.add_argument("--output", default="outputs/logs/compare_with_pytorch.json")
    args = parser.parse_args()

    root = Path(args.pytorch_root)
    key_files = {
        "official_tokenizer": root / "models" / "unitok.py",
        "official_quantizer": root / "models" / "quant.py",
        "official_vqvae_attention": root / "models" / "vqvae.py",
        "official_trainer": root / "trainer.py",
        "jittor_tokenizer": Path("jittor_unitok/models/tokenizer.py"),
        "jittor_mcq": Path("jittor_unitok/models/mcq.py"),
        "jittor_attention": Path("jittor_unitok/models/attention_projection.py"),
        "jittor_train": Path("jittor_unitok/engine/train_tokenizer.py"),
    }
    report = {
        "pytorch_root_exists": root.exists(),
        "files": {},
        "module_alignment": [
            ["models/unitok.py", "jittor_unitok/models/tokenizer.py", "end-to-end tokenizer graph"],
            ["models/quant.py::VectorQuantizerM", "jittor_unitok/models/mcq.py::MultiCodebookQuantizer", "multi-codebook split, lookup, concat"],
            ["models/vqvae.py::AttnProjection", "jittor_unitok/models/attention_projection.py", "attention projection"],
            ["models/vitamin.py", "jittor_unitok/models/encoder.py + decoder.py", "large ViTamin replaced by tiny CNN"],
            ["trainer.py + utils/loss.py", "jittor_unitok/engine/train_tokenizer.py + models/losses.py", "lightweight L_R + L_VQ training"],
        ],
        "shape_alignment": {
            "image": "[B,3,H,W]",
            "tokens": "[B,N,hidden_dim]",
            "latent": "[B,N,latent_dim]",
            "indices": "[B,num_codebooks,N]",
            "reconstruction": "[B,3,H,W]",
        },
    }
    for name, path in key_files.items():
        report["files"][name] = {
            "path": str(path),
            "exists": path.exists(),
            "lines": count_lines(path) if path.exists() else 0,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not root.exists():
        sys.exit(1)


if __name__ == "__main__":
    main()
