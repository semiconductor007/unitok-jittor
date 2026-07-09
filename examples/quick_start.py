"""One-file forward/encode/decode/reconstruct demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("nvcc_path", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jittor as jt

from jittor_unitok.models import UniTokTokenizer


def main() -> None:
    jt.flags.use_cuda = 0
    model = UniTokTokenizer(image_size=64, hidden_dim=64, latent_dim=32, num_codebooks=4, codebook_size=32)
    x = jt.rand((2, 3, 64, 64)) * 2.0 - 1.0
    out = model(x)
    indices = model.encode(x)
    decoded = model.decode(indices)
    recon = model.reconstruct(x)
    print("forward recon:", out["recon"].shape)
    print("indices:", indices.shape)
    print("decoded:", decoded.shape)
    print("reconstruct:", recon.shape)
    print("loss:", float((out["recon_loss"] + out["vq_loss"]).numpy().reshape(-1)[0]))


if __name__ == "__main__":
    main()
