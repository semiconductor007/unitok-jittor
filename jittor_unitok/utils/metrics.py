"""Reconstruction metrics."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def reconstruction_metrics(pred, target) -> Dict[str, float]:
    """Compute MSE, L1 and PSNR on arrays/tensors in `[-1, 1]`."""

    if hasattr(pred, "numpy"):
        pred = pred.numpy()
    if hasattr(target, "numpy"):
        target = target.numpy()
    pred = np.asarray(pred, dtype="float32")
    target = np.asarray(target, dtype="float32")
    mse = float(np.mean((pred - target) ** 2))
    l1 = float(np.mean(np.abs(pred - target)))
    psnr = 20.0 * math.log10(2.0) - 10.0 * math.log10(max(mse, 1e-12))
    return {"mse": mse, "l1": l1, "psnr": psnr}

