"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 0) -> None:
    """Set Python, NumPy and Jittor seeds."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import jittor as jt

        jt.set_global_seed(seed)
    except Exception:
        pass

