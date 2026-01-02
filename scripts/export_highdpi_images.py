"""
Export high-DPI copies of image artifacts while preserving file formats.

Produces:
 - outputs/paper_images/ (mirrored directory structure)

Design notes:
 - SVG files are copied without modification.
 - PNG and JPG files are resaved with DPI metadata set to 600.
 - No format conversion is performed during export.
"""

from pathlib import Path
from PIL import Image
import shutil
import logging

SRC_ROOT = Path("reports")
OUT_ROOT = Path("outputs") / "paper_images"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DPI = 600

LOG = logging.getLogger("highdpi_export")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def handle_svg(src: Path, dst: Path):
    """SVG is vector → just copy unchanged."""
    ensure_parent(dst)
    shutil.copy2(src, dst)
    LOG.info("Copied SVG → %s", dst)


def handle_png(src: Path, dst: Path):
    """PNG → resave with higher DPI metadata (same exact pixels)."""
    ensure_parent(dst)
    img = Image.open(src)
    try:
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        img.save(dst, "PNG", dpi=(DPI, DPI))
    finally:
        img.close()
    LOG.info("Saved high-DPI PNG → %s", dst)


def handle_jpg(src: Path, dst: Path):
    """JPG → resave with higher DPI metadata (same exact pixels)."""
    ensure_parent(dst)
    img = Image.open(src)
    try:
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        img.save(dst, "JPEG", dpi=(DPI, DPI), quality=95)
    finally:
        img.close()
    LOG.info("Saved high-DPI JPG → %s", dst)


def process_file(src: Path):
    rel = src.relative_to(SRC_ROOT)
    dst = OUT_ROOT / rel
    suffix = src.suffix.lower()

    if suffix == ".svg":
        handle_svg(src, dst)
    elif suffix == ".png":
        handle_png(src, dst)
    elif suffix in (".jpg", ".jpeg"):
        handle_jpg(src, dst)
    else:
        LOG.debug("Skipping unsupported file: %s", src)
        return False

    return True


def main():
    count = 0
    for p in SRC_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".svg", ".png", ".jpg", ".jpeg"):
            try:
                if process_file(p):
                    count += 1
            except Exception as e:
                LOG.error("Failed for %s: %s", p, e)

    LOG.info("Done. High-DPI copies created: %d", count)
    print("Outputs saved to:", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()
