import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("nvcc_path", "")

import numpy as np
import pytest

try:
    import jittor as jt
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Jittor runtime is unavailable in this environment: {exc}", allow_module_level=True)
from jittor import optim

from jittor_unitok.models import UniTokTokenizer


def codebook_array(model):
    return np.stack([book.numpy() for book in model.quantizer.codebooks], axis=0)


def test_one_train_step_updates_parameter():
    jt.flags.use_cuda = 0
    model = UniTokTokenizer(image_size=64, hidden_dim=64, latent_dim=32, num_codebooks=4, codebook_size=16)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    x = jt.rand((2, 3, 64, 64)) * 2.0 - 1.0
    before = codebook_array(model).copy()
    out = model(x)
    loss = out["recon_loss"] + out["vq_loss"]
    optimizer.step(loss)
    after = codebook_array(model)
    assert tuple(loss.shape) in [(), (1,)]
    assert not np.allclose(before, after)
