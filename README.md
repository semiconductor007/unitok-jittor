# Jittor UniTok：统一视觉分词器的轻量复现

本项目使用 [Jittor](https://github.com/Jittor/jittor) 复现论文 **UniTok: A Unified Tokenizer for Visual Generation and Understanding** 的核心视觉分词流程。它实现了多码本量化（MCQ）、注意力投影、图像编码与解码，并提供从演示数据生成、训练、评估到结果可视化的完整脚本。

> 本仓库是面向教学、课程作业和算法验证的轻量工程复现，并非 UniTok-Large 的完整训练实现。编码器与解码器采用小型 CNN，默认数据为程序生成的几何图像，不复现 DataComp-1B、大型 ViTamin/OpenCLIP/Llama 组件及论文规模的训练结果。

## 项目特点

- 基于 Jittor 实现端到端视觉分词器
- 支持 `forward`、`encode`、`decode` 和 `reconstruct`
- 实现多码本量化器 `MultiCodebookQuantizer`
- 使用注意力完成潜变量通道的压缩与扩展
- 提供演示数据生成、训练、断点续训、评估和绘图脚本
- 自动保存检查点、训练日志、损失曲线和重建对比图
- 包含核心模块与单步训练测试

## 方法简介

UniTok 的目标是学习一套同时服务于视觉生成和视觉理解的离散图像 token。本项目保留其核心思路：编码器先将图像映射到连续潜空间，注意力投影调整潜变量表示，多码本量化器将潜变量离散化，最后由解码器重建图像。

多码本量化会将潜变量通道划分为多个子空间，每个子空间使用独立码本。当前轻量训练目标为：

```text
L = L_recon + λ_vq × L_vq
L_vq = L_commitment + L_codebook
```

原论文还使用感知损失、对抗损失和图文对比损失；这些部分不在默认轻量流程中启用。

## 环境要求

推荐使用：

- Python 3.10
- Jittor 1.3 或更高版本
- NumPy、Pillow、PyYAML、Matplotlib、pytest
- python-pptx（仅生成答辩材料时需要）

使用 Conda 安装：

```bash
conda env create -f environment.yml
conda activate jittor-unitok
```

也可以使用 pip：

```bash
pip install -r requirements.txt
```

Windows 用户建议使用 `environment.yml` 中指定的 Python 3.10。项目脚本默认设置 `nvcc_path=""`，以便在没有可用 CUDA 环境时使用 CPU，避免 Jittor 自动下载 CUDA 工具包。

## 快速开始

运行随机张量示例，检查模型的前向传播、编码、解码和重建流程：

```bash
python examples/quick_start.py
```

示例输出包含重建图像、离散索引和解码结果的张量形状，以及当前损失值。

## 完整复现流程

### 1. 准备数据

生成一个包含彩色几何图形的小型演示数据集：

```bash
python scripts/prepare_demo_data.py --output data/demo --image-size 64 --train-count 32 --val-count 8
```

数据目录应符合以下结构：

```text
data/demo/
├── train/
│   └── *.jpg
└── val/
    └── *.jpg
```

也可以用自己的图片替换演示图片，只需保持 `train` 和 `val` 两个子目录不变。

### 2. 训练模型

PowerShell：

```powershell
.\scripts\train_tiny.ps1
```

或直接执行：

```bash
python -m jittor_unitok.engine.train_tokenizer --config configs/unitok_tiny.yaml --no-cuda
```

若只想快速验证流程，可改用单轮配置：

```bash
python -m jittor_unitok.engine.train_tokenizer --config configs/unitok_demo.yaml --no-cuda
```

从已有检查点继续训练：

```bash
python -m jittor_unitok.engine.train_tokenizer --config configs/unitok_tiny.yaml --resume outputs/checkpoints/unitok_epoch_1.pkl --no-cuda
```

常用参数包括 `--data-root`、`--output-dir`、`--epochs`、`--batch-size`、`--lr`、`--image-size`、`--num-codebooks`、`--codebook-size`、`--latent-dim` 和 `--hidden-dim`。命令行参数优先级高于 YAML 配置。

训练产物默认保存在 `outputs/`：

```text
outputs/
├── checkpoints/       # 每轮及最终模型权重
├── curves/loss.csv    # 逐步训练指标
├── logs/train.log     # 训练日志
└── reconstructions/   # 每轮重建对比图
```

### 3. 评估重建质量

PowerShell：

```powershell
.\scripts\eval_recon.ps1
```

或直接执行：

```bash
python -m jittor_unitok.engine.eval_reconstruction --checkpoint outputs/checkpoints/unitok_last.pkl --data-root data/demo --output-dir outputs --no-cuda
```

评估脚本会计算 MSE、L1 和 PSNR，并生成：

- `outputs/logs/eval_metrics.json`：平均评估指标
- `outputs/reconstructions/eval_batch_*.png`：原图与重建图对比

这些轻量指标主要用于验证重建链路。论文使用的 rFID 及下游生成、理解指标不属于本项目默认评估范围。

### 4. 绘制损失曲线

```bash
python scripts/plot_loss.py --csv outputs/curves/loss.csv --output outputs/curves/loss.png
```

## 配置说明

仓库提供两套配置：

| 配置文件 | 用途 | 默认训练轮数 | 隐藏维度 | 潜变量维度 |
| --- | --- | ---: | ---: | ---: |
| `configs/unitok_demo.yaml` | 快速连通流程 | 1 | 64 | 32 |
| `configs/unitok_tiny.yaml` | 小型训练实验 | 2 | 128 | 64 |

主要配置项：

| 参数 | 含义 |
| --- | --- |
| `data_root` | 包含 `train/` 和 `val/` 的数据根目录 |
| `output_dir` | 日志、权重和图片输出目录 |
| `image_size` | 输入图像边长 |
| `num_codebooks` | 独立码本数量 |
| `codebook_size` | 每个码本的 token 数量 |
| `latent_dim` | 潜变量通道维度 |
| `lambda_vq` | VQ 损失权重 |

## 测试

运行全部测试：

```bash
pytest -q
```

测试覆盖多码本量化、注意力投影、分词器编解码、图像重建，以及一次包含参数更新的训练步骤。

## 项目结构

```text
jittor-unitok/
├── configs/                    # 训练配置
├── examples/                   # 快速使用示例
├── jittor_unitok/
│   ├── data/                   # 图像数据集与批处理工具
│   ├── engine/                 # 训练与评估入口
│   ├── models/                 # 编码器、解码器、MCQ、注意力投影
│   └── utils/                  # 日志、指标、图像和随机种子工具
├── scripts/                    # 数据、训练、评估与可视化脚本
├── tests/                      # 自动化测试
├── environment.yml
└── requirements.txt
```

## 与官方 PyTorch 实现的对应关系

| 官方实现 | 本项目 | 说明 |
| --- | --- | --- |
| `models/unitok.py` | `jittor_unitok/models/tokenizer.py` | 分词器整体计算图 |
| `models/quant.py` | `jittor_unitok/models/mcq.py` | 多码本量化算法 |
| `models/vqvae.py` 中的 `AttnProjection` | `jittor_unitok/models/attention_projection.py` | 注意力投影 |
| `models/vitamin.py` | `encoder.py`、`decoder.py` | 轻量 CNN 替代实现 |
| `trainer.py` | `jittor_unitok/engine/train_tokenizer.py` | 轻量训练流程 |

如需生成本地代码结构对比报告：

```bash
python scripts/compare_with_pytorch.py --pytorch-root ../UniTok --output outputs/logs/compare_with_pytorch.json
```

## 可选：生成答辩材料

仓库还包含损失图、重建展示图和 PowerPoint 材料的生成脚本：

```bash
python scripts/generate_ppt_figures.py --ma-window 500
python scripts/build_ppt.py
```

生成文件位于 `outputs/ppt/`。`build_ppt.py` 依赖项目上级目录中的 `TemplateMC/TemplateMC-PPT.pptx` 模板；若只使用模型训练与评估功能，可以忽略本节。

## 常见问题

**这是对论文指标的完整复现吗？**

不是。本项目聚焦核心算法结构和 Jittor 工程实现，不能与官方大规模训练结果直接比较。

**可以训练自己的图片吗？**

可以。将图片分别放入数据根目录的 `train/` 和 `val/`，再通过配置文件或 `--data-root` 指定该目录即可。

**为什么默认没有 LPIPS、GAN、FID 或 rFID？**

这些方法需要额外的预训练模型、参考数据和计算资源。默认流程优先保证 VQ 图像重建实验简单、稳定且可运行。

## 论文与致谢

- 论文：[UniTok: A Unified Tokenizer for Visual Generation and Understanding](https://arxiv.org/abs/2502.20321)
- 官方 PyTorch 仓库：[FoundationVision/UniTok](https://github.com/FoundationVision/UniTok)

本项目参考了 FoundationVision 发布的 UniTok 论文与官方实现。

```bibtex
@article{unitok,
  title={UniTok: A Unified Tokenizer for Visual Generation and Understanding},
  author={Ma, Chuofan and Jiang, Yi and Wu, Junfeng and Yang, Jihan and Yu, Xin and Yuan, Zehuan and Peng, Bingyue and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2502.20321},
  year={2025}
}
```
