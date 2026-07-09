import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("nvcc_path", "")

import pytest

try:
    import jittor as jt
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Jittor runtime is unavailable in this environment: {exc}", allow_module_level=True)

from jittor_unitok.models import ChannelCompressionBlock, ChannelExpansionBlock


def test_attention_projection_shapes():
    jt.flags.use_cuda = 0
    x = jt.randn((2, 16, 64))
    comp = ChannelCompressionBlock(64, 32, num_heads=4)
    exp = ChannelExpansionBlock(32, 64, num_heads=4)
    z = comp(x)
    y = exp(z)
    assert z.shape == (2, 16, 32)
    assert y.shape == (2, 16, 64)
