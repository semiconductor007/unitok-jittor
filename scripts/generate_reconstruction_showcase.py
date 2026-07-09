"""Generate a clearer reconstruction showcase for PPT.

CIFAR-10 images are only 32x32, so random samples can look blurry after PPT
upscaling. This script selects validation images with stronger edges/colors,
runs the trained tokenizer, and creates a large original/reconstruction/error
comparison panel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("nvcc_path", "")
import jittor as jt

from jittor_unitok.engine.eval_reconstruction import load_tokenizer_state
from jittor_unitok.models import UniTokTokenizer


BLUE = (31, 78, 121)
ORANGE = (198, 89, 17)
GREY = (96, 96, 96)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/arial.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def center_crop_resize(path: Path, image_size: int) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        return img.resize((image_size, image_size), Image.BICUBIC)


def image_score(img: Image.Image) -> float:
    arr = np.asarray(img).astype("float32") / 255.0
    grey = np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES)).astype("float32") / 255.0
    color_std = arr.std(axis=(0, 1)).mean()
    contrast = np.asarray(img.convert("L")).astype("float32").std() / 255.0
    edge = grey.mean()
    return float(edge * 2.0 + color_std + contrast)


def select_images(val_dir: Path, image_size: int, count: int) -> list[Path]:
    paths = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    scored = []
    for path in paths:
        try:
            img = center_crop_resize(path, image_size)
            scored.append((image_score(img), path))
        except Exception:
            continue
    scored.sort(reverse=True, key=lambda x: x[0])
    # Spread out classes when CIFAR filenames contain class ids.
    selected: list[Path] = []
    used_classes: set[str] = set()
    for _, path in scored:
        cls = "unknown"
        for part in path.stem.split("_"):
            if part.startswith("class"):
                cls = part
                break
        if cls in used_classes and len(selected) < count - 1:
            continue
        selected.append(path)
        used_classes.add(cls)
        if len(selected) >= count:
            return selected
    return [path for _, path in scored[:count]]


def to_tensor(images: list[Image.Image]) -> jt.Var:
    arrays = []
    for img in images:
        arr = np.asarray(img).astype("float32") / 255.0
        arr = arr.transpose(2, 0, 1) * 2.0 - 1.0
        arrays.append(arr)
    return jt.array(np.stack(arrays, axis=0))


def tensor_to_images(x) -> list[Image.Image]:
    arr = x.numpy()
    arr = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
    arr = (arr * 255.0).round().astype("uint8").transpose(0, 2, 3, 1)
    return [Image.fromarray(item, "RGB") for item in arr]


def heatmap(orig: Image.Image, rec: Image.Image) -> tuple[Image.Image, float]:
    o = np.asarray(orig).astype("float32") / 255.0
    r = np.asarray(rec).astype("float32") / 255.0
    diff = np.abs(o - r).mean(axis=-1)
    mse = float(((o - r) ** 2).mean())
    red = np.full_like(diff, 255.0)
    green = 255.0 - 175.0 * np.clip(diff * 3.0, 0.0, 1.0)
    blue = 255.0 - 255.0 * np.clip(diff * 3.0, 0.0, 1.0)
    img = np.stack([red, green, blue], axis=-1).clip(0, 255).astype("uint8")
    return Image.fromarray(img, "RGB"), mse


def draw_panel(originals: list[Image.Image], recons: list[Image.Image], names: list[str], out: Path) -> None:
    width, height = 1920, 1080
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(44)
    header_font = load_font(28)
    small_font = load_font(22)
    draw.text((80, 48), "清晰样本重建对比：输入原图 / 重建图 / 误差", fill=BLUE, font=title_font)
    draw.text((80, 106), "注：CIFAR-10 原始分辨率仅 32×32，这里展示的是模型实际输入的 64×64 放大图。", fill=GREY, font=small_font)

    scale = 5
    tile = originals[0].width * scale
    gap = 48
    start_x = 88
    start_y = 190
    col_x = [start_x, start_x + tile + gap, start_x + 2 * (tile + gap)]
    for x, label in zip(col_x, ["Original input", "Reconstruction", "Error map"]):
        draw.text((x + 42, 145), label, fill=BLUE, font=header_font)

    for i, (orig, rec, name) in enumerate(zip(originals, recons, names)):
        y = start_y + i * (tile + 72)
        err, mse = heatmap(orig, rec)
        for x, img in zip(col_x, [orig, rec, err]):
            big = img.resize((tile, tile), Image.NEAREST)
            canvas.paste(big, (x, y))
            draw.rectangle((x, y, x + tile, y + tile), outline=BLUE, width=3)
        draw.text((col_x[-1] + tile + 32, y + 110), f"MSE {mse:.4f}", fill=GREY, font=small_font)
        draw.text((col_x[-1] + tile + 32, y + 150), name[:30], fill=GREY, font=small_font)

    draw.text((80, 1010), "结论：主体颜色和轮廓可重建，高频纹理与边缘细节更平滑。", fill=ORANGE, font=header_font)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/unitok_last.pkl")
    parser.add_argument("--data-root", default="data/cifar10")
    parser.add_argument("--output", default="outputs/ppt/figures/reconstruction_showcase_clear.png")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    if args.no_cuda:
        jt.flags.use_cuda = 0

    ckpt = jt.load(args.checkpoint)
    cfg = ckpt.get("config", {})
    image_size = int(cfg.get("image_size", 64))
    model = UniTokTokenizer(
        image_size,
        cfg.get("hidden_dim", 128),
        cfg.get("latent_dim", 64),
        cfg.get("num_codebooks", 4),
        cfg.get("codebook_size", 64),
    )
    load_tokenizer_state(model, ckpt["model"])
    model.eval()

    paths = select_images(Path(args.data_root) / "val", image_size, args.count)
    originals = [center_crop_resize(path, image_size) for path in paths]
    images = to_tensor(originals)
    recons = tensor_to_images(model.reconstruct(images))
    draw_panel(originals, recons, [path.name for path in paths], Path(args.output))
    print(f"saved {args.output}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
