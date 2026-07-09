import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("nvcc_path", "")

import pytest

try:
    import jittor as jt
except Exception as exc:  # pragma: no cover - depends on local Jittor compiler.
    pytest.skip(f"Jittor runtime is unavailable in this environment: {exc}", allow_module_level=True)
from jittor import optim

from jittor_unitok.models import MultiCodebookQuantizer


def test_mcq_shapes_and_backward():
    jt.flags.use_cuda = 0
    quantizer = MultiCodebookQuantizer(latent_dim=32, num_codebooks=4, codebook_size=16)
    x = jt.randn((2, 8, 32))
    q, indices, commitment, codebook, stats = quantizer(x)
    assert q.shape == x.shape
    assert indices.shape == (2, 4, 8)
    assert tuple(commitment.shape) in [(), (1,)]
    assert tuple(codebook.shape) in [(), (1,)]
    assert "usage" in stats
    optimizer = optim.Adam(quantizer.parameters(), lr=1e-3)
    optimizer.step(q.mean() + commitment + codebook)
