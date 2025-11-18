#!/usr/bin/env python3
# scripts/reexport_figs_for_paper.py
"""
Re-export existing diagnostic figures into paper-ready PDFs and high-dpi PNGs.

Usage:
    python scripts/reexport_figs_for_paper.py

Notes:
 - Optional (recommended) dependencies:
     pip install img2pdf cairosvg
 - If cairosvg is missing SVG->PDF conversion will be skipped (and noted).
"""
from pathlib import Path
from PIL import Image
import sys
import time

ROOT = Path(".")
SRC = ROOT / "reports" / "figs" / "diagnostics"
OUT = ROOT / "reports" / "paper_ready"   # place outputs outside SRC to avoid reprocessing
OUT.mkdir(parents=True, exist_ok=True)

# paper size (w,h) inches when using matplotlib fallback (rare)
PAPER_SIZE = (7.0, 4.5)
DPI_PNG = 600

# optional libs
try:
    import img2pdf
except Exception:
    img2pdf = None

try:
    import cairosvg
except Exception:
    cairosvg = None


def png_to_pdf_img2pdf(in_path: Path, out_path: Path, dpi=DPI_PNG):
    """Preferred: embed PNG into PDF using img2pdf (lossless embed)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(in_path, "rb") as f_in:
        img_bytes = f_in.read()
    try:
        pdf_bytes = img2pdf.convert(img_bytes, dpi=(dpi, dpi))
        with open(out_path, "wb") as f_out:
            f_out.write(pdf_bytes)
        return out_path
    except Exception as e:
        raise


def png_to_pdf_pillow(in_path: Path, out_path: Path, dpi=DPI_PNG):
    """Fallback using Pillow. Convert to RGB first."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(in_path)
    try:
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        img.save(out_path, "PDF", resolution=dpi)
    finally:
        try:
            img.close()
        except Exception:
            pass
    return out_path


def svg_to_pdf_cairosvg(in_path: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2pdf(url=str(in_path), write_to=str(out_path))
    return out_path


def svg_to_png_cairosvg(in_path: Path, out_path: Path, dpi=DPI_PNG):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(url=str(in_path), write_to=str(out_path), dpi=dpi)
    return out_path


def process_file(p: Path):
    rel = p.relative_to(SRC)
    out_pdf = OUT / rel.with_suffix(".pdf")
    out_png = OUT / rel.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    suffix = p.suffix.lower()
    if suffix == ".png":
        # Preferred route: img2pdf if available
        if img2pdf:
            try:
                png_to_pdf_img2pdf(p, out_pdf)
            except Exception as e:
                print("img2pdf failed, falling back to Pillow for:", p, "error:", e, file=sys.stderr)
                png_to_pdf_pillow(p, out_pdf)
        else:
            # fallback pillow
            png_to_pdf_pillow(p, out_pdf)

        # also create a high-dpi PNG copy (resave via Pillow)
        img = Image.open(p)
        try:
            img = img.convert("RGB")
            img.save(out_png, "PNG", dpi=(DPI_PNG, DPI_PNG))
        finally:
            try:
                img.close()
            except Exception:
                pass
        return True

    if suffix == ".svg":
        if cairosvg:
            # svg -> pdf
            try:
                svg_to_pdf_cairosvg(p, out_pdf)
            except Exception as e:
                print("cairosvg svg->pdf failed for", p, ":", e, file=sys.stderr)
            # svg -> png (optional)
            try:
                svg_to_png_cairosvg(p, out_png)
            except Exception:
                # png optional
                pass
            return True
        else:
            print("cairosvg not installed — skipping SVG -> PDF/PNG for", p, file=sys.stderr)
            return False

    if suffix in (".jpg", ".jpeg"):
        img = Image.open(p)
        try:
            img = img.convert("RGB")
            # high-dpi png
            img.save(out_png, "PNG", dpi=(DPI_PNG, DPI_PNG))
            # pdf
            if img2pdf:
                with open(p, "rb") as f:
                    pdf_bytes = img2pdf.convert(f.read(), dpi=(DPI_PNG, DPI_PNG))
                with open(out_pdf, "wb") as f:
                    f.write(pdf_bytes)
            else:
                img.save(out_pdf, "PDF", resolution=DPI_PNG)
        finally:
            try:
                img.close()
            except Exception:
                pass
        return True

    return False


def export_all():
    if not SRC.exists():
        print(f"Source not found: {SRC}")
        return

    count = 0
    for p in SRC.rglob("*"):
        if p.is_dir():
            continue
        # skip items in OUT if any
        if OUT in p.parents:
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in (".png", ".svg", ".jpg", ".jpeg"):
            continue

        try:
            ok = process_file(p)
            if ok:
                print("Processed:", p, "->", OUT / p.relative_to(SRC))
                count += 1
            else:
                print("Skipped (no conversion):", p)
        except Exception as e:
            print("Failed to process", p, ":", e, file=sys.stderr)

    print(f"Done. Processed {count} files. Outputs in: {OUT.resolve()}")


if __name__ == "__main__":
    export_all()
