"""Tiny Jittor reproduction of UniTok tokenizer components.

On Windows, Jittor may auto-download a large CUDA toolkit during import. This
tiny reproduction defaults to CPU-friendly startup by setting `nvcc_path=""`
unless the user already configured another value.
"""

import os

os.environ.setdefault("nvcc_path", "")

__version__ = "0.1.0"
