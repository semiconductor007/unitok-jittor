"""Image conversion and visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


def tensor_to_uint8(x) -> np.ndarray:
    """Convert tensor/array in `[-1, 1]` to NHWC uint8."""

    if hasattr(x, "numpy"):
        x = x.numpy()
    arr = np.asarray(x)
    if arr.ndim == 3:
        arr = arr[None]
    arr = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
    arr = (arr * 255.0).round().astype("uint8")
    return arr.transpose(0, 2, 3, 1)


def save_reconstruction_grid(original, recon, path: str | Path, names: Iterable[str] | None = None) -> None:
    """Save side-by-side original/reconstruction pairs."""

    orig = tensor_to_uint8(original)
    rec = tensor_to_uint8(recon)
    count = min(len(orig), len(rec), 8)
    h, w = orig.shape[1], orig.shape[2]
    label_h = 18
    canvas = Image.new("RGB", (w * 2, count * (h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    names = list(names or [f"sample_{i}" for i in range(count)])
    for i in range(count):
        y = i * (h + label_h)
        canvas.paste(Image.fromarray(orig[i]), (0, y + label_h))
        canvas.paste(Image.fromarray(rec[i]), (w, y + label_h))
        draw.text((4, y + 2), f"{names[i]} | original", fill=(0, 0, 0))
        draw.text((w + 4, y + 2), "reconstruction", fill=(0, 0, 0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)

