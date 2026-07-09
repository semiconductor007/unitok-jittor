# jittor-unitok

Tiny Jittor reproduction of **UniTok: A Unified Tokenizer for Visual Generation and Understanding**.

This project is **not** a full retraining of UniTok-Large. It focuses on the tokenizer core: VQVAE-style reconstruction, multi-codebook quantization, attention projection and discrete visual tokens. The official paper-scale setup uses DataComp-1B, large ViTamin/OpenCLIP/Llama components and large GPU training; this repo uses a small Jittor model and generated demo images so the code can run for coursework checks.

## Paper Information

- Title: UniTok: A Unified Tokenizer for Visual Generation and Understanding
- Venue: NeurIPS 2025 spotlight, according to the official README
- arXiv: https://arxiv.org/abs/2502.20321
- Official PyTorch repository: https://github.com/FoundationVision/UniTok
- Local reference path used here: `D:\newproject\UniTok`
- This Jittor repository placeholder: `https://github.com/<your-name>/jittor-unitok`

## Reproduction Target

Implemented:

- Jittor `MultiCodebookQuantizer`
- Jittor attention projection blocks
- Tiny encoder/decoder tokenizer backbone
- End-to-end `forward`, `encode`, `decode`, `reconstruct`
- Demo data generation
- Training/evaluation scripts
- Loss CSV, train log, reconstruction grids
- PyTorch/Jittor alignment notes
- pytest tests for core modules

Lightweight or omitted:

- ViTamin-L is replaced by a tiny CNN.
- Full CLIP text tower is replaced by a simplified contrastive-loss helper.
- LPIPS/GAN/FID/rFID are documented but not default.
- DataComp-1B and unified MLLM training are not reproduced.

## Method Overview

UniTok is a unified visual tokenizer. It learns discrete visual tokens that can reconstruct images and can also support visual understanding.

- Unified visual tokenizer: one token representation serves generation and understanding.
- VQVAE reconstruction supervision: image tokens are decoded back to pixels.
- CLIP contrastive supervision: official training aligns visual and text features.
- Multi-codebook Quantization: latent channels are split into multiple chunks and each chunk uses an independent codebook.
- Attention Projection: channel compression/expansion is done with attention instead of a plain linear projection.

Tiny formula used here:

```text
L = L_R + lambda_VQ * L_VQ
L_VQ = commitment_loss + codebook_loss
```

The full paper objective also includes perceptual, adversarial and contrastive terms:

```text
L_recon = L_R + lambda_VQ L_VQ + lambda_P L_P + lambda_G L_G
L = L_recon + lambda_contra L_contra
```

## Environment

Validated local environment:

- Conda env: `unitok_jittor`
- Python: 3.10.20
- Jittor: 1.3.8.5
- Execution: CPU, with `nvcc_path=""` to avoid Windows CUDA toolkit auto-download.

Recommended:

- Python 3.10 or 3.11. The current machine used Python 3.13.9, and Jittor 1.3.11 failed while compiling its core C++ runtime on Windows/MSVC.
- Jittor 1.3+
- NumPy, Pillow, PyYAML, matplotlib, pytest

Conda install:

```bash
cd D:\newproject\jittor-unitok
conda env create -f environment.yml
conda activate jittor-unitok
```

Pip install:

```bash
cd D:\newproject\jittor-unitok
pip install -r requirements.txt
```

Note for Windows/Jittor: on this machine, importing Jittor first tried downloading `cuda11.2_cudnn8_win.zip`. The repo now sets `nvcc_path=""` before importing Jittor so CPU demo startup skips that download. With Python 3.13.9, Jittor then failed during local C++ compilation; use the `environment.yml` Python 3.10 environment for actual training/testing.

## Data Preparation

Generate a tiny sanity dataset:

```bash
python scripts/prepare_demo_data.py --output data/demo --image-size 64 --train-count 32 --val-count 8
```

Expected format:

```text
data/demo/
  train/*.jpg
  val/*.jpg
```

You can also replace these images with your own local folder images in the same layout.

## Training

PowerShell:

```powershell
.\scripts\train_tiny.ps1
```

Direct command:

```bash
python -m jittor_unitok.engine.train_tokenizer --config configs/unitok_tiny.yaml --no-cuda
```

Important arguments:

- `--config`
- `--data-root`
- `--output-dir`
- `--epochs`
- `--batch-size`
- `--lr`
- `--image-size`
- `--num-codebooks`
- `--codebook-size`
- `--latent-dim`

Training writes:

- `outputs/logs/train.log`
- `outputs/curves/loss.csv`
- `outputs/checkpoints/unitok_epoch_*.pkl`
- `outputs/checkpoints/unitok_last.pkl`
- `outputs/reconstructions/train_epoch_*.png`

## Evaluation

PowerShell:

```powershell
.\scripts\eval_recon.ps1
```

Direct command:

```bash
python -m jittor_unitok.engine.eval_reconstruction --checkpoint outputs/checkpoints/unitok_last.pkl --data-root data/demo --output-dir outputs --no-cuda
```

Evaluation writes:

- `outputs/logs/eval_metrics.json`
- `outputs/reconstructions/eval_batch_*.png`

Metrics are MSE, L1 and PSNR. The official paper reports rFID and understanding/generation metrics; this tiny reproduction uses lightweight reconstruction metrics.

## Loss Curve

```bash
python scripts/plot_loss.py --csv outputs/curves/loss.csv --output outputs/curves/loss.png
```

For the final PPT, generate PPT-ready figures from the full training CSV:

```bash
python scripts/generate_ppt_figures.py --ma-window 500
```

The PPT loss page uses `outputs/ppt/figures/loss_epoch_30.png`, where the x-axis is epoch 1-30.

## PPT and Speech Materials

The final defense materials are saved under `outputs/ppt/`:

- PPT: `outputs/ppt/UniTok_Jittor_Reproduction.pptx`
- PDF: `outputs/ppt/UniTok_Jittor_Reproduction.pdf`
- Per-slide speech script: `outputs/ppt/UniTok_Jittor_Speech.md`

Suggested final submission PDF name:

```text
姓名-播种期.pdf
```

The deck is generated from the provided template:

- Template: `../TemplateMC/TemplateMC-PPT.pptx`

Generate the PPT and speech script:

```bash
python scripts/build_ppt.py
```

On Windows with Microsoft PowerPoint installed, export the generated PPTX to PDF:

```powershell
$pptx = (Resolve-Path 'outputs\ppt\UniTok_Jittor_Reproduction.pptx').Path
$pdf = (Join-Path (Resolve-Path 'outputs\ppt').Path 'UniTok_Jittor_Reproduction.pdf')
$powerpoint = New-Object -ComObject PowerPoint.Application
$presentation = $powerpoint.Presentations.Open($pptx, $true, $false, $false)
$presentation.SaveAs($pdf, 32)
$presentation.Close()
$powerpoint.Quit()
```

PPT result pages use these generated artifacts:

- Training log: `outputs/logs/train.log`
- PPT 30-epoch loss curve: `outputs/ppt/figures/loss_epoch_30.png`
- Step-level smoothed loss curve: `outputs/ppt/figures/loss_curve_ma.png`
- Reconstruction images: `outputs/reconstructions/train_epoch_30.png`, `outputs/reconstructions/eval_batch_1.png`
- PPT clear reconstruction panel: `outputs/ppt/figures/reconstruction_showcase_clear.png`
- PyTorch/Jittor alignment report: `outputs/logs/compare_with_pytorch.json`

GitHub repository placeholder to update before submission:

```text
https://github.com/<your-name>/jittor-unitok
```

## One-Command Reproduction Flow

Run the full tiny experiment and regenerate PPT assets in order:

```bash
python scripts/prepare_demo_data.py --output data/demo --image-size 64 --train-count 32 --val-count 8
python -m jittor_unitok.engine.train_tokenizer --config configs/unitok_demo.yaml --epochs 1 --batch-size 2 --no-cuda
python -m jittor_unitok.engine.eval_reconstruction --checkpoint outputs/checkpoints/unitok_last.pkl --data-root data/demo --output-dir outputs --batch-size 4 --no-cuda
python scripts/plot_loss.py --csv outputs/curves/loss.csv --output outputs/curves/loss_curve.png
python scripts/generate_ppt_figures.py --ma-window 500
python scripts/compare_with_pytorch.py --pytorch-root ../UniTok --output outputs/logs/compare_with_pytorch.json
python scripts/build_ppt.py
```

If the official PyTorch repository is not present at `../UniTok`, the comparison script still writes the Jittor-side alignment structure, then exits with a non-zero status to indicate the missing reference path.

## Quick Start

```bash
python examples/quick_start.py
```

It runs forward, encode, decode and reconstruct on random tensors.

## PyTorch/Jittor Alignment

```bash
python scripts/compare_with_pytorch.py --pytorch-root ../UniTok
```

Alignment summary:

| Official PyTorch | Jittor file | Alignment |
| --- | --- | --- |
| `models/unitok.py` | `jittor_unitok/models/tokenizer.py` | tokenizer graph |
| `models/quant.py` | `jittor_unitok/models/mcq.py` | MCQ algorithm |
| `models/vqvae.py::AttnProjection` | `jittor_unitok/models/attention_projection.py` | attention projection |
| `models/vitamin.py` | `encoder.py`, `decoder.py` | tiny replacement |
| `trainer.py` | `engine/train_tokenizer.py` | lightweight training |

See:

- `docs/pytorch_code_analysis.md`
- `docs/reproduction_notes.md`
- `outputs/logs/compare_with_pytorch.json` after running comparison

## Tests

```bash
pytest -q
```

Tests cover:

- MCQ shapes, scalar losses and backward step
- Attention compression/expansion shapes
- Tokenizer forward/reconstruct/encode/decode
- One random train step and parameter update

## File Structure

```text
jittor-unitok/
  configs/
  docs/
  jittor_unitok/
    models/
    data/
    utils/
    engine/
  scripts/
  tests/
  outputs/
  examples/
```

## Visualization Results

After training and evaluation, inspect:

- `outputs/reconstructions/train_epoch_1.png`
- `outputs/reconstructions/eval_batch_1.png`
- `outputs/curves/loss.png`
- `outputs/curves/loss_curve.png`
- `outputs/ppt/figures/loss_epoch_30.png`

If those files are absent, run data preparation, training, evaluation and `plot_loss.py` in that order.

Current run status in this workspace:

- `data/demo/train` and `data/demo/val` were generated successfully.
- `outputs/logs/compare_with_pytorch.json` was generated successfully.
- `python -m compileall -q .` passed.
- `conda run -n unitok_jittor python examples\quick_start.py` passed.
- `conda run -n unitok_jittor python -m pytest -q` returned `5 passed`.
- CIFAR-10 training completed for 30 epochs / 187500 steps.
- Evaluation completed with MSE `0.0085`, L1 `0.0675`, PSNR `26.76`.
- Generated artifacts:
  - `outputs/logs/train.log`
  - `outputs/curves/loss.csv`
  - `outputs/curves/loss.png`
  - `outputs/curves/loss_curve.png`
  - `outputs/ppt/figures/loss_epoch_30.png`
  - `outputs/checkpoints/unitok_last.pkl`
  - `outputs/reconstructions/train_epoch_30.png`
  - `outputs/reconstructions/eval_batch_1.png`
  - `outputs/ppt/UniTok_Jittor_Reproduction.pptx`
  - `outputs/ppt/UniTok_Jittor_Reproduction.pdf`
  - `outputs/ppt/UniTok_Jittor_Speech.md`

## Performance Log

The real training log is saved to `outputs/logs/train.log`. A typical row contains:

```text
epoch step total_loss recon_loss vq_loss codebook_usage lr time
```

The CSV version is `outputs/curves/loss.csv`.

## Resource-Limited Statement

The official UniTok uses DataComp-1B, ViTamin/OpenCLIP, Llama-2-7B style downstream systems, rFID evaluation and large distributed training. This repo uses a tiny setting to verify algorithmic structure only:

- small generated image dataset
- small CNN encoder/decoder
- few epochs and few batches
- lightweight reconstruction metrics

## FAQ

**Is this a faithful reproduction of official metrics?**

No. It is a structural and engineering reproduction for Jittor.

**Can I train on my own images?**

Yes. Put images under `data/demo/train` and `data/demo/val`, or pass another `--data-root` with the same split layout.

**Why no LPIPS/GAN/FID by default?**

Those require additional pretrained models, reference batches and more resources. The code includes a lightweight discriminator helper, but the default demo focuses on stable VQ reconstruction.

## Acknowledgement and Citation

This work references the official UniTok PyTorch implementation by FoundationVision.

```bibtex
@article{unitok,
  title={UniTok: A Unified Tokenizer for Visual Generation and Understanding},
  author={Ma, Chuofan and Jiang, Yi and Wu, Junfeng and Yang, Jihan and Yu, Xin and Yuan, Zehuan and Peng, Bingyue and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2502.20321},
  year={2025}
}
```
