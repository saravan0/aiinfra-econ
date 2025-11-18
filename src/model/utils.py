# src/model/utils.py
"""Utilities for model training, artifact saving and lightweight plotting.

Designed for reproducibility: deterministic filenames, config snapshot,
and compact plotting helpers used by the modeling pipeline.

Usage examples:
    from src.model.utils import save_model, save_json, plot_scatter_with_fit, now_iso, save_model_with_summary
"""
from __future__ import annotations
from pathlib import Path
import json
import joblib
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Tuple

# --- logging setup (idempotent) ---
LOG = logging.getLogger("src.model.utils")
if not LOG.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(ch)
LOG.setLevel(logging.INFO)


def ensure_parent(path: Path) -> Path:
    """Create parent directory for `path` if missing and return the Path (idempotent)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_model(obj: Any, out_path: Path) -> None:
    """Save `obj` with joblib to out_path. Creates parent dirs if needed."""
    p = ensure_parent(out_path)
    try:
        joblib.dump(obj, str(p))
        LOG.info("Saved model -> %s", p)
    except Exception as e:
        LOG.exception("joblib.dump failed for %s: %s", p, e)
        # fallback: write repr text next to intended file
        try:
            repr_path = Path(str(p) + ".repr.txt")
            ensure_parent(repr_path)
            with open(repr_path, "w", encoding="utf8") as fh:
                fh.write(repr(obj))
            LOG.info("Wrote model repr fallback -> %s", repr_path)
        except Exception as e2:
            LOG.error("Fallback repr write failed for %s: %s", p, e2)
            raise


def save_json(obj: Any, out_path: Path) -> None:
    """Save JSON-serializable `obj` to out_path (pretty-printed)."""
    p = ensure_parent(out_path)
    with open(p, "w", encoding="utf8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    LOG.info("Saved json -> %s", p)


def save_config_snapshot(cfg: Dict[str, Any], out_path: Path) -> None:
    """Save a small reproducibility snapshot with timestamp + config."""
    meta = {"saved_at": now_iso(), "config": cfg}
    save_json(meta, out_path)


def sha256sum(path: Path) -> str:
    """Return SHA256 hex digest of file at path."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(text: str, out_path: Path) -> None:
    """Write text to file (creates parent dir)."""
    p = ensure_parent(out_path)
    with open(p, "w", encoding="utf8") as fh:
        fh.write(text)
    LOG.info("Saved text -> %s", p)


def list_dir(path: Path) -> Dict[str, int]:
    """Return a small map of filename -> size for files in a directory."""
    p = Path(path)
    out: Dict[str, int] = {}
    if not p.exists():
        return out
    for f in sorted(p.iterdir()):
        if f.is_file():
            out[f.name] = f.stat().st_size
    return out


# ----------------------
# Reproducibility helpers
# ----------------------
def now_iso() -> str:
    """Return current UTC timestamp in ISO format with trailing 'Z'."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_sha256_of_file(p: Path) -> Optional[str]:
    """Return sha256 hex for existing file, or None if missing/unreadable."""
    try:
        if not Path(p).exists():
            return None
        return sha256sum(Path(p))
    except Exception as e:
        LOG.debug("sha256 read failed for %s: %s", p, e)
        return None


def _small_text_summary_for_model(obj: Any, max_chars: int = 200) -> str:
    """
    Produce a compact human-friendly summary for common model objects.
    - statsmodels results -> short param list + nobs
    - sklearn pipelines/estimators -> class name + first few params
    - fallback -> type name + truncated repr
    """
    try:
        # statsmodels results-like object
        if hasattr(obj, "params") and hasattr(obj, "nobs"):
            try:
                params = getattr(obj, "params")
                if hasattr(params, "items"):
                    items = list(params.items())[:4]
                    s = ", ".join(f"{k}={float(v):.3g}" for k, v in items)
                else:
                    s = str(params)[:max_chars]
                return f"statsmodel nobs={getattr(obj,'nobs',None)} params=[{s}]"
            except Exception:
                pass

        # sklearn pipeline / estimator
        if hasattr(obj, "get_params"):
            try:
                params = obj.get_params()
                keys = list(params.keys())[:4]
                s = ", ".join(f"{k}={str(params[k])}" for k in keys)
                return f"sklearn {obj.__class__.__name__} params=[{s}]"
            except Exception:
                pass
    except Exception:
        pass

    rep = repr(obj)
    if len(rep) > max_chars:
        rep = rep[: max_chars - 3] + "..."
    return f"{type(obj).__name__}: {rep}"


def save_model_with_summary(obj: Any, out_path: Path, summary_extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Save model artifact + companion summary JSON. Returns dict:
      {"artifact_path": str, "summary_path": str, "sha256": str|None, "size_bytes": int|None, "saved_at": str}

    Behavior:
      - Tries joblib.dump(obj, out_path). If that fails, writes a .repr.txt fallback.
      - Writes out_path + '.summary.json' containing sha, timestamp, short summary, optional extra fields.
    """
    p = ensure_parent(out_path)
    artifact_path = str(p)
    saved_ok = False
    # attempt save
    try:
        joblib.dump(obj, str(p))
        saved_ok = True
    except Exception as e:
        LOG.warning("joblib.dump failed for %s: %s — falling back to repr text", p, e)
        try:
            repr_path = Path(str(p) + ".repr.txt")
            ensure_parent(repr_path)
            with open(repr_path, "w", encoding="utf8") as fh:
                fh.write(repr(obj))
            artifact_path = str(repr_path)
            saved_ok = True
        except Exception as e2:
            LOG.error("Fallback repr write also failed for %s: %s", p, e2)
            saved_ok = False

    sha = _safe_sha256_of_file(Path(artifact_path)) if saved_ok else None
    size = Path(artifact_path).stat().st_size if saved_ok else None
    timestamp = now_iso()
    short_summary = _small_text_summary_for_model(obj)

    summary = {
        "artifact_path": artifact_path,
        "sha256": sha,
        "size_bytes": size,
        "saved_at": timestamp,
        "summary": short_summary,
    }
    if summary_extra:
        summary.update(summary_extra)

    summary_path = Path(str(artifact_path) + ".summary.json")
    try:
        save_json(summary, summary_path)
    except Exception as e:
        LOG.error("Failed to write artifact summary %s: %s", summary_path, e)

    LOG.info("Saved model artifact -> %s (summary -> %s)", artifact_path, summary_path)
    return {"artifact_path": artifact_path, "summary_path": str(summary_path), "sha256": sha, "size_bytes": size, "saved_at": timestamp}


# ----------------------
# Lightweight plotting helper
# ----------------------
def plot_scatter_with_fit(
    df: pd.DataFrame,
    x: str,
    y: str,
    ax=None,
    figsize: Tuple[int, int] = (6, 4),
    fname: Optional[Path] = None,
    scatter_kwargs: Optional[Dict[str, Any]] = None,
    line_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Scatter plot of y vs x with a simple linear-fit line saved to fname (if provided).

    Returns Path to saved file when fname is provided, otherwise returns None.
    """
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Columns not found in df: {x}, {y}")

    scatter_kwargs = scatter_kwargs or {"s": 12, "alpha": 0.45}
    line_kwargs = line_kwargs or {"linestyle": "--", "linewidth": 1.6, "color": "C1"}

    # drop rows with missing x or y
    sub = df[[x, y]].dropna()
    if sub.empty:
        LOG.warning("Empty subset for plotting %s vs %s", x, y)
        return None

    # create axis if not provided
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # scatter
    ax.scatter(sub[x], sub[y], **scatter_kwargs)

    # fit linear (use np.polyfit on finite values)
    xs = sub[x].astype(float).to_numpy()
    ys = sub[y].astype(float).to_numpy()
    # guard against degenerate arrays
    if np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)) and len(xs) >= 2:
        try:
            m, b = np.polyfit(xs, ys, 1)
            xs_line = np.linspace(np.nanmin(xs), np.nanmax(xs), 200)
            ax.plot(xs_line, m * xs_line + b, **line_kwargs)
            ax.annotate(f"slope={m:.3g}", xy=(0.98, 0.02), xycoords="axes fraction", ha="right", fontsize=8, color="gray")
        except Exception as e:
            LOG.debug("Linear fit failed: %s", e)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} ~ {x}")
    if created_fig:
        plt.tight_layout()

    if fname:
        p = ensure_parent(Path(fname))
        try:
            plt.savefig(str(p), dpi=200, bbox_inches="tight")
            LOG.info("Saved plot -> %s", p)
        except Exception as e:
            LOG.error("Failed to save plot -> %s : %s", p, e)
        finally:
            if created_fig:
                plt.close()
        return p
    if created_fig:
        plt.show()
    return None
