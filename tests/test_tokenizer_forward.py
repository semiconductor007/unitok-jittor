import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("nvcc_path", "")

import pytest

try:
    import jittor as jt
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Jittor runtime is unavailable in this environment: {exc}", allow_module_level=True)

from jittor_unitok.models import UniTokTokenizer


def test_tokenizer_forward_encode_decode():
    jt.flags.use_cuda = 0
    model = UniTokTokenizer(image_size=64, hidden_dim=64, latent_dim=32, num_codebooks=4, codebook_size=16)
    x = jt.rand((2, 3, 64, 64)) * 2.0 - 1.0
    out = model(x)
    assert out["recon"].shape == x.shape
    assert out["indices"].shape == (2, 4, 64)
    recon = model.reconstruct(x)
    indices = model.encode(x)
    decoded = model.decode(indices)
    assert recon.shape == x.shape
    assert decoded.shape == x.shape
