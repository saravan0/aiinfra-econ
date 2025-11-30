#!/usr/bin/env python3
"""
scripts/generate_onepager_compositor.py

Polished compositor:
 - Core plate: 3x stacked slots (top / middle / bottom) with thin outlines + titles.
 - Support plate: 2x3 grid with thin outlines, left column = (qq, en_path, av_trade), right column = (av_infl, comparative, partial_gov).
 - PNG + SVG outputs -> reports/onepager/files/
 - metadata.json + manifest.json -> reports/onepager/
 - Does not change the ONEPAGER_MD_TEXT content (writes markdown to reports/onepager/files/onepager.md)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

# Optional rasterizers
try:
    import cairosvg
except Exception:
    cairosvg = None

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

# -------------------------
# Constants / canvas sizes
# -------------------------
DPI = 300
A4_LANDSCAPE = (int(11.69 * DPI), int(8.27 * DPI))
CANVAS_CORE = A4_LANDSCAPE
CANVAS_SUPPORT = A4_LANDSCAPE

OUT_FILES_DIR = Path("reports/onepager/files")
OUT_FILES_DIR.mkdir(parents=True, exist_ok=True)
OUT_META_DIR = OUT_FILES_DIR.parent  # reports/onepager


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def log(msg: str):
    print(msg, file=sys.stderr)


# -------------------------
# ONEPAGER MD
# -------------------------
ONEPAGER_MD_TEXT = """\
# **Reproducible AI Infrastructure for High-Dimensional Modeling**
### *A Hybrid Machine Learning–Econometrics System for Macroeconomic Forecasting and Diagnostics*

---

## **1. Purpose & Framing**

This one-pager summarizes a fully reproducible AI–econometrics system designed for high-dimensional macroeconomic modeling.
The pipeline integrates elastic-net regularization, fixed-effects econometrics, SHAP attribution, nonlinearity diagnostics, and temporal forecasting into a unified research-grade workflow.

The objective is robust causal-adjacent interpretation of how governance, trade exposure, and inflation shape short-run GDP growth—validated across multiple estimators and through out-of-sample forecasting.

---

## **2. Data & Preprocessing **

- Harmonized global panel (2000–2023).
- Standardization of predictors to enable FE interpretation (coef × SD_x / SD_y).
- Deterministic imputation + consistency enforcement (structural zeros, monotonicity checks).
- ElasticNet artifacts (model + scaler) fully version-controlled.
- All intermediate outputs captured by the baseline snapshot engine.

This ensures bit-for-bit reproducibility of all results.

---

## **3. Core Empirical Findings**

### **Trade exposure → growth **
Across FE (Driscoll–Kraay corrected), OLS, and ElasticNet, trade remains a positive and consistent driver of short-run GDP growth.
SHAP values confirm its high global importance, and LOWESS curves show an increasing and smooth nonlinearity without sign reversals.

### **Governance → temporary negative effect **
Higher governance quality correlates with lower contemporaneous growth, a result stable across all estimators and SHAP.
This is interpreted as a short-run reform cost: high-governance regimes often implement regulatory tightening, fiscal adjustments, or structural reforms that depress short-term growth but improve long-run resilience.

### **Inflation → moderate negative effect**
Inflation’s sign aligns with macroeconomic intuition; magnitude is smaller and more specification-sensitive but directionally stable across FE, OLS, ElasticNet, and SHAP.

---

## **4. Nonlinearity Diagnostics **

The LOWESS/GAM-style nonlinear plates reveal:

- **Trade exposure:** steadily increasing marginal returns; no evidence of thresholds.
- **Governance:** notable curvature — negative at low/mid governance, flattening at high governance (turning-point detected via derivative).
- **Inflation:** mild convexity but stable sign.

These shape analyses confirm that effects are smooth, monotonic, and interpretable, not driven by local instabilities.

---

## **5. SHAP Attribution **

Mean absolute SHAP contributions rank:

1. Governance quality
2. Trade exposure
3. Inflation

This ordering matches the FE and OLS standardized magnitudes, providing cross-method validation.
SHAP also confirms the direction of effects and the absence of strong interactions.

---

## **6. Temporal Forecasting & Stability **

A full expanding-window validation (2000→2023) quantifies temporal stability, not just in-sample fit.

### **Key insights:**
- RMSE stable (~3–4.5) in normal years;
- Expected spikes occur in 2009 (GFC) and 2020 (COVID-19 shock);
- Diebold–Mariano tests confirm statistically significant improvement over a persistence benchmark except during global crises;
- No evidence of model drift or structural breaks outside shock years.

This demonstrates a high-stability forecasting backbone.

---

## **7. Consolidated Baseline Snapshot **

The final baseline snapshot bundles:

- FE coefficients (Driscoll–Kraay)
- OLS coefficients
- ElasticNet coefficients at selected α
- SHAP mean |importance|
- Nonlinearity metrics
- Rolling RMSE (h1, h3)
- Provenance hashes and paths

All variables show sign consistency across methods, a strong indicator of robustness.

---

## **8. Visual Summary **

The generated **onepager_core.png / onepager_support.png** contain:

- Core plate: three 1×3 images stacked → 3×3 presentation (LOWESS gov, SHAP trade, Rolling RMSE).
- Support plate: 2×3 grid containing added-variable individuals, comparative effects, EN path, partials, QQ.

All rendered with consistent layout and 300 DPI export.

---

## **9. Interpretation & Policy Relevance**

- **Trade openness** remains a highly robust and policy-relevant predictor of short-run growth.
- **Governance** shows a reform-cycle effect: short-run negative, long-run stabilizing.
- **Inflation** acts as a standard cyclical drag with moderate effect size.

Forecasting results demonstrate that the system is stable and shock-aware, not overfit.

---

## **10. Reproducibility & Metadata**

Re-run the entire analysis via:

reports/generate_baseline_snapshot/ (JSON + CSV + provenance)

reports/onepager/files/ (PNG + SVG + MD + metadata + manifest)

The system implements full reproducibility, metadata tracking, and artifact integrity checks.
"""


# -------------------------
# Image utilities
# -------------------------
def load_raster(path: Path) -> Image.Image:
    if path is None:
        raise RuntimeError("No path provided to load_raster()")
    path = Path(path)
    if path.suffix.lower() == ".json":
        for ext in (".png", ".svg", ".pdf", ".jpg", ".jpeg"):
            cand = path.with_suffix(ext)
            if cand.exists():
                path = cand
                break
        else:
            raise RuntimeError(f"JSON provided but no same-stem image found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        return Image.open(path).convert("RGBA")

    if suffix == ".svg":
        if cairosvg:
            png_bytes = cairosvg.svg2png(url=str(path), dpi=DPI)
            return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        png_try = path.with_suffix(".png")
        if png_try.exists():
            return Image.open(png_try).convert("RGBA")
        raise RuntimeError(
            f"SVG rasterization requires cairosvg or a PNG sibling: {path}"
        )

    if suffix == ".pdf":
        if convert_from_path:
            pages = convert_from_path(str(path), dpi=DPI, first_page=1, last_page=1)
            if not pages:
                raise RuntimeError(f"pdf2image could not convert {path}")
            return pages[0].convert("RGBA")
        png_try = path.with_suffix(".png")
        if png_try.exists():
            return Image.open(png_try).convert("RGBA")
        raise RuntimeError(
            f"PDF rasterization requires pdf2image/poppler or a PNG sibling: {path}"
        )

    raise RuntimeError(f"Unsupported image type: {path}")


def smart_trim(im: Image.Image, bg_thresh=250) -> Image.Image:
    arr = np.asarray(im.convert("RGB"))
    mask = np.any(arr < bg_thresh, axis=2)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return im
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return im.crop((x0, y0, x1, y1))


def resize_to_fit(im: Image.Image, max_width=None, max_height=None) -> Image.Image:
    w, h = im.size
    scale = 1.0
    if max_width:
        scale = min(scale, max_width / w)
    if max_height:
        scale = min(scale, max_height / h)
    if scale >= 1:
        return im
    return im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def embed_png_in_svg(png_path: Path, svg_path: Path, width: int, height: int):
    b64 = base64.b64encode(png_path.read_bytes()).decode()
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<image width="{width}" height="{height}" href="data:image/png;base64,{b64}"/>
</svg>"""
    svg_path.write_text(svg, encoding="utf8")


def paste_center(
    canvas: Image.Image, im: Image.Image, box: Tuple[int, int, int, int], pad: int = 8
):
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    # reduce available area slightly for inner padding
    aw = max(1, w - pad * 2)
    ah = max(1, h - pad * 2)
    im2 = resize_to_fit(im, max_width=aw, max_height=ah)
    iw, ih = im2.size
    px = x0 + (w - iw) // 2
    py = y0 + (h - ih) // 2
    canvas.paste(im2, (px, py), im2)


# -------------------------
# Drawing helpers (outline, title)
# -------------------------
def draw_box_with_title(
    canvas: Image.Image,
    box: Tuple[int, int, int, int],
    title: str = None,
    outline=(160, 160, 160),
    fill=None,
    font=None,
):
    d = ImageDraw.Draw(canvas)
    if fill:
        d.rectangle(box, fill=fill)
    d.rectangle(box, outline=outline, width=1)
    if title:
        tx = box[0] + 8
        ty = box[1] + 6
        d.text((tx, ty), title, fill=(30, 30, 30), font=font)


# -------------------------
# CORE plate (vertical stack)
# -------------------------
def compose_core_plate(inputs: Dict, out_png: Path, out_svg: Path):
    W, H = CANVAS_CORE
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    # fetch configured paths (support both nested dicts or flat)
    lowess_gov = None
    shap_trade = None
    rmse = None
    if isinstance(inputs.get("lowess"), dict):
        lowess_gov = inputs["lowess"].get("gov_index_zmean")
    else:
        lowess_gov = inputs.get("lowess_gov") or inputs.get("lowess")
    if isinstance(inputs.get("shap"), dict):
        shap_trade = inputs["shap"].get("trade_exposure")
    else:
        shap_trade = inputs.get("shap_trade") or inputs.get("shap")
    if isinstance(inputs.get("rolling_validation"), dict):
        rmse = (
            inputs["rolling_validation"].get("rmse_combined_png")
            or inputs["rolling_validation"].get("rmse_by_year_combined")
            or inputs["rolling_validation"].get("rmse_by_year_combined.png")
        )
    else:
        rmse = inputs.get("rmse")

    def try_load(p, label):
        if not p:
            log(f"[WARN] core plate: missing path for {label} -> {p}")
            return None
        try:
            im = smart_trim(load_raster(Path(p)))
            log(f"[INFO] core plate: loaded {p}")
            return im
        except Exception as e:
            log(f"[WARN] core plate: failed to load {p}: {e}")
            return None

    im_top = try_load(lowess_gov, "LOWESS (gov_index_zmean)")
    im_mid = try_load(shap_trade, "SHAP (trade_exposure)")
    im_bot = try_load(rmse, "RMSE by year (h=1,h=3)")

    imgs = [im_top, im_mid, im_bot]
    titles = [
        "LOWESS (gov_index_zmean)",
        "SHAP (trade_exposure)",
        "RMSE by year (h=1,h=3)",
    ]

    font = ImageFont.load_default()
    top_margin = int(H * 0.02)
    side_margin = int(W * 0.03)
    slot_h = (H - 3 * top_margin) // 3
    for i in range(3):
        y0 = top_margin + i * slot_h
        y1 = y0 + slot_h - 6
        box = (side_margin, y0, W - side_margin, y1)
        # light fill for contrast
        draw_box_with_title(
            canvas,
            box,
            title=titles[i],
            outline=(120, 120, 120),
            fill=(250, 250, 250),
            font=font,
        )
        if imgs[i] is not None:
            # inner box leave small gap for title area
            inner_box = (box[0] + 8, box[1] + 24, box[2] - 8, box[3] - 8)
            paste_center(canvas, imgs[i], inner_box)
        else:
            d = ImageDraw.Draw(canvas)
            d.text(
                (box[0] + 12, box[1] + 28),
                f"Missing: {titles[i]}",
                fill=(120, 120, 120),
                font=font,
            )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_png, dpi=(DPI, DPI))
    log(f"[OK] core plate → {out_png}")
    # SVG wrapper
    try:
        embed_png_in_svg(out_png, out_svg, W, H)
        log(f"[OK] core SVG → {out_svg}")
    except Exception as e:
        log(f"[WARN] core plate: failed to write SVG wrapper: {e}")


# -------------------------
# SUPPORT plate (2x3)
# -------------------------
def compose_support_plate(inputs: Dict, out_png: Path, out_svg: Path):
    W, H = CANVAS_SUPPORT
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    margin_x = int(W * 0.03)
    margin_y = int(H * 0.04)
    col_w = (W - 3 * margin_x) // 2
    # compute slot height so 3 rows fit with small vertical gaps
    slot_h = (H - 4 * margin_y) // 3

    # left slots
    left_slots = []
    for i in range(3):
        x0 = margin_x
        y0 = margin_y + i * slot_h
        left_slots.append((x0, y0, x0 + col_w, y0 + slot_h - 6))
    # right slots
    right_slots = []
    for i in range(3):
        x0 = margin_x * 2 + col_w
        y0 = margin_y + i * slot_h
        right_slots.append((x0, y0, x0 + col_w, y0 + slot_h - 6))

    # Resolve configured paths
    qq = (
        inputs.get("qq_studentized")
        or (inputs.get("partials") or {}).get("qq_studentized")
        or inputs.get("qq")
    )
    enp = (
        inputs.get("en_path") or inputs.get("en_path_png") or inputs.get("en_path_pdf")
    )
    av_trade = (
        (inputs.get("added_variable_individuals") or {}).get("av_trade_exposure")
        or inputs.get("av_trade_exposure")
        or inputs.get("added_trade")
    )
    av_infl = (inputs.get("added_variable_individuals") or {}).get(
        "av_inflation_consumer_prices_pct"
    ) or inputs.get("av_inflation_consumer_prices_pct")
    comp = (
        inputs.get("comparative_effects")
        or inputs.get("plot_comparative_model_effects")
        or inputs.get("comparative_effects_png")
    )
    partial_gov = (
        (inputs.get("partials") or {}).get("partial_resid_gov")
        or inputs.get("partial_resid_gov_index_zmean")
        or inputs.get("partial_resid_gov_index_zmean.png")
    )

    def try_load(p, label):
        if not p:
            log(f"[WARN] support plate: missing path for {label} -> {p}")
            return None
        try:
            im = smart_trim(load_raster(Path(p)))
            log(f"[INFO] support plate: loaded {p}")
            return im
        except Exception as e:
            log(f"[WARN] support plate: failed to load {p}: {e}")
            return None

    im_qq = try_load(qq, "QQ (studentized)")
    im_en = try_load(enp, "ElasticNet path")
    im_av_trade = try_load(av_trade, "AV: trade_exposure")
    im_av_infl = try_load(av_infl, "AV: inflation_consumer_prices_pct")
    im_comp = try_load(comp, "Comparative effects")
    im_partial_gov = try_load(partial_gov, "Partial resid (gov_index_zmean)")

    # left column (top->bottom): qq, en, av_trade
    left_imgs = [im_qq, im_en, im_av_trade]
    left_labels = ["QQ (studentized)", "ElasticNet path", "AV: trade_exposure"]
    for slot, im, lab in zip(left_slots, left_imgs, left_labels):
        # draw box with faint fill
        draw_box_with_title(
            canvas, slot, title=None, outline=(130, 130, 130), fill=(248, 248, 248)
        )
        if im is not None:
            paste_center(
                canvas, im, (slot[0] + 6, slot[1] + 6, slot[2] - 6, slot[3] - 6)
            )
        else:
            d = ImageDraw.Draw(canvas)
            d.text(
                (slot[0] + 12, slot[1] + 12), f"Missing: {lab}", fill=(110, 110, 110)
            )

    # right column (top->bottom): av_infl, comparative, partial_gov
    right_imgs = [im_av_infl, im_comp, im_partial_gov]
    right_labels = [
        "AV: inflation_consumer_prices_pct",
        "Comparative effects",
        "Partial resid (gov_index_zmean)",
    ]
    for slot, im, lab in zip(right_slots, right_imgs, right_labels):
        draw_box_with_title(
            canvas, slot, title=None, outline=(130, 130, 130), fill=(248, 248, 248)
        )
        if im is not None:
            paste_center(
                canvas, im, (slot[0] + 6, slot[1] + 6, slot[2] - 6, slot[3] - 6)
            )
        else:
            d = ImageDraw.Draw(canvas)
            d.text(
                (slot[0] + 12, slot[1] + 12), f"Missing: {lab}", fill=(110, 110, 110)
            )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_png, dpi=(DPI, DPI))
    log(f"[OK] support plate → {out_png}")
    try:
        embed_png_in_svg(out_png, out_svg, W, H)
        log(f"[OK] support SVG → {out_svg}")
    except Exception as e:
        log(f"[WARN] support plate: failed to write SVG wrapper: {e}")


# -------------------------
# Write markdown + metadata/manifest to parent
# -------------------------
def write_markdown(md_path: Path):
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(ONEPAGER_MD_TEXT, encoding="utf8")
    log(f"[OK] onepager.md → {md_path}")


def write_metadata_and_manifest(
    core_png: Path,
    core_svg: Path,
    support_png: Path,
    support_svg: Path,
    md_path: Path,
    inputs: Dict,
):
    meta_path = OUT_META_DIR / "metadata.json"
    manifest_path = OUT_META_DIR / "manifest.json"
    meta = {
        "generated_at_utc": now_iso(),
        "core_png": str(core_png),
        "core_svg": str(core_svg),
        "support_png": str(support_png),
        "support_svg": str(support_svg),
        "markdown": str(md_path),
        "inputs": inputs,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")
    manifest = {"generated_at_utc": now_iso(), "files": meta}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")
    log(f"[OK] metadata + manifest → {OUT_META_DIR}")


# -------------------------
# Runner
# -------------------------
def run(cfg_path: Path):
    cfg = json.loads(cfg_path.read_text(encoding="utf8"))
    inputs = cfg.get("inputs", {})

    core_png = OUT_FILES_DIR / "onepager_core.png"
    core_svg = OUT_FILES_DIR / "onepager_core.svg"
    support_png = OUT_FILES_DIR / "onepager_support.png"
    support_svg = OUT_FILES_DIR / "onepager_support.svg"
    md_path = OUT_FILES_DIR / "onepager.md"

    compose_core_plate(inputs, core_png, core_svg)
    compose_support_plate(inputs, support_png, support_svg)
    write_markdown(md_path)
    write_metadata_and_manifest(
        core_png, core_svg, support_png, support_svg, md_path, inputs
    )
    log("[DONE] core + support plates written.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/onepager_config.json")
    args = ap.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        log(f"Config not found: {cfg_path}")
        raise SystemExit(2)
    run(cfg_path)


if __name__ == "__main__":
    main()
