"""Download CIFAR-10 and convert it to the image-folder layout used here.

The training code only needs:

    data/cifar10/train/*.png
    data/cifar10/val/*.png

CIFAR-10 provides 50,000 training images and 10,000 test images. The original
images are 32x32 RGB; the project dataset loader will resize them to the
configured image_size, normally 64.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MD5 = "c58f30108f718f92721af3b95e74349a"


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() and md5sum(archive) == MD5:
        print(f"Archive already exists: {archive}")
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, archive)
    got = md5sum(archive)
    if got != MD5:
        raise RuntimeError(f"MD5 mismatch for {archive}: expected {MD5}, got {got}")


def extract(archive: Path, raw_dir: Path) -> Path:
    target = raw_dir / "cifar-10-batches-py"
    if target.exists():
        print(f"Extracted directory already exists: {target}")
        return target
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(raw_dir)
    return target


def load_batch(path: Path) -> tuple[np.ndarray, list[int], list[str]]:
    with path.open("rb") as f:
        obj = pickle.load(f, encoding="latin1")
    data = obj["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = obj["labels"]
    names = obj.get("filenames", [f"{i:05d}.png" for i in range(len(labels))])
    return data, labels, names


def write_split(batch_paths: list[Path], out_dir: Path, limit: int | None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for batch_path in batch_paths:
        data, labels, names = load_batch(batch_path)
        for img_arr, label, name in zip(data, labels, names):
            if limit is not None and written >= limit:
                return written
            stem = Path(name).stem
            path = out_dir / f"{written:05d}_class{label}_{stem}.png"
            Image.fromarray(img_arr, mode="RGB").save(path)
            written += 1
    return written


def parse_limit(value: int) -> int | None:
    return None if value < 0 else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/cifar10")
    parser.add_argument("--cache-dir", default="data/_downloads")
    parser.add_argument("--source-dir", default=None, help="Use an existing cifar-10-batches-py directory and skip download.")
    parser.add_argument("--train-count", type=int, default=-1, help="Use -1 for all 50,000 train images.")
    parser.add_argument("--val-count", type=int, default=-1, help="Use -1 for all 10,000 test images.")
    parser.add_argument("--force", action="store_true", help="Clear existing converted images before writing.")
    args = parser.parse_args()

    output = Path(args.output)
    if args.source_dir:
        cifar_dir = Path(args.source_dir)
        if not cifar_dir.exists():
            raise FileNotFoundError(f"source dir does not exist: {cifar_dir}")
        print(f"Using existing CIFAR-10 directory: {cifar_dir}")
    else:
        cache_dir = Path(args.cache_dir)
        archive = cache_dir / "cifar-10-python.tar.gz"
        raw_dir = cache_dir / "raw"
        download(URL, archive)
        cifar_dir = extract(archive, raw_dir)

    train_dir = output / "train"
    val_dir = output / "val"
    if args.force:
        for split_dir in (train_dir, val_dir):
            if split_dir.exists():
                for path in split_dir.glob("*.png"):
                    path.unlink()

    train_batches = [cifar_dir / f"data_batch_{i}" for i in range(1, 6)]
    val_batches = [cifar_dir / "test_batch"]
    train_written = write_split(train_batches, train_dir, parse_limit(args.train_count))
    val_written = write_split(val_batches, val_dir, parse_limit(args.val_count))
    print(f"CIFAR-10 written to {output}")
    print(f"train images: {train_written}")
    print(f"val images: {val_written}")


if __name__ == "__main__":
    main()
