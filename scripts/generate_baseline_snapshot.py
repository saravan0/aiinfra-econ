#!/usr/bin/env python3
"""
scripts/generate_baseline_snapshot.py

Create a reproducible, provenance-rich baseline snapshot for Stage-1 results.

Outputs (all under reports/generate_baseline_snapshot/):
  - files/                          (copies of used artifacts)
      fe_<var>.json
      ols_<var>.json
      elasticnet_<var>.json
      standardized/<var>_standardized.json (if present)
  - generate_baseline_snapshot.json  (full snapshot list)
  - model_comparison_table.csv       (wide table for plotting)
  - generate_baseline_snapshot.md    (human-readable summary)
  - meta.json                        (timestamps, hashes, git commit)

Usage:
  python scripts/generate_baseline_snapshot.py
  python scripts/generate_baseline_snapshot.py --rescale-en
  python scripts/generate_baseline_snapshot.py --vars trade_exposure gov_index_zmean
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

LOG = logging.getLogger("generate_baseline_snapshot")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

# Paths / defaults
ROOT = Path(".")
OUT_BASE = ROOT / "reports" / "generate_baseline_snapshot"
OUT_FILES = OUT_BASE / "files"
OUT_STANDARDIZED = Path("outputs") / "standardized"
OUT_VARIABLES = Path("outputs") / "variables"
ARTIFACTS = Path("artifacts")
FEATURES_DEFAULT = Path("data") / "processed" / "features_lean_imputed.csv"

VARS_DEFAULT = ["trade_exposure", "gov_index_zmean", "inflation_consumer_prices_pct"]


def _ensure_dirs():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    OUT_FILES.mkdir(parents=True, exist_ok=True)
    (OUT_FILES / "standardized").mkdir(parents=True, exist_ok=True)


def safe_json_load(p: Path) -> Optional[Any]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf8"))
    except Exception as e:
        LOG.warning("Failed to load JSON %s: %s", p, e)
        return None


def safe_copy_if_exists(src: Path, dst_dir: Path) -> Optional[Path]:
    if not src.exists():
        return None
    dst = dst_dir / src.name
    try:
        shutil.copy2(src, dst)
        return dst
    except Exception as e:
        LOG.warning("Copy failed %s -> %s: %s", src, dst, e)
        return None


def sha256_of_file(p: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf8").strip()
    except Exception:
        return None


def load_per_model_artifacts(var: str) -> Dict[str, Any]:
    """Find and load per-model artifacts either in outputs/ or artifacts/ (best-effort)."""
    candidates = {}
    # prioritized locations
    paths = [
        OUT_VARIABLES / var / f"{var}_summary.json",  # if your extract script wrote this
        OUT_STANDARDIZED / f"{var}_standardized.json",
        Path("outputs") / f"fe_{var}.json",
        Path("outputs") / f"ols_{var}.json",
        Path("outputs") / f"elasticnet_{var}.json",
        ARTIFACTS / f"fe_result.joblib",  # not JSON — will not be loaded but we copy
    ]
    # Standard named artifacts in artifacts/ (joblib/pkl) — copy for provenance
    # For JSONs, attempt to load
    out = {"fe": None, "ols": None, "elasticnet": None, "standardized": None}
    # load from outputs first
    stdp = OUT_STANDARDIZED / f"{var}_standardized.json"
    if stdp.exists():
        out["standardized"] = safe_json_load(stdp)
        safe_copy_if_exists(stdp, OUT_FILES / "standardized")

    for kind in ("fe", "ols", "elasticnet"):
        p1 = Path("outputs") / f"{kind}_{var}.json"
        p2 = Path("outputs") / f"{kind}_{var}.json".replace(".json", ".txt")
        p_art = ARTIFACTS / f"{kind}_result.joblib"
        if p1.exists():
            out[kind] = safe_json_load(p1)
            safe_copy_if_exists(p1, OUT_FILES)
        elif p2.exists():
            out[kind] = safe_json_load(p2)
            safe_copy_if_exists(p2, OUT_FILES)
        elif p_art.exists():
            # copy artifact for provenance; we cannot easily inspect joblib here
            safe_copy_if_exists(p_art, OUT_FILES)
            out[kind] = {"artifact_path": str(p_art)}
        else:
            # try outputs/variables var subfolder produced by the extractor
            alt = OUT_VARIABLES / var / f"{var}_summary.json"
            if alt.exists():
                out[kind] = safe_json_load(alt)
                safe_copy_if_exists(alt, OUT_FILES / "standardized")
    return out


def collect_entry(var: str, en_unscaled_map: Optional[Dict[str, float]] = None, features_path: Optional[Path] = None) -> Dict[str, Any]:
    """Collect all pieces for a single variable and return a snapshot entry."""
    LOG.info("Collecting variable: %s", var)
    row = {"variable": var, "collected_at_utc": datetime.utcnow().isoformat() + "Z", "notes": []}

    artifacts = load_per_model_artifacts(var)
    row.update({
        "standardized_summary": artifacts.get("standardized"),
        "fe_json": artifacts.get("fe"),
        "ols_json": artifacts.get("ols"),
        "elasticnet_json": artifacts.get("elasticnet"),
    })

    # If user requested EN rescale mapping, override the recorded coeff (best-effort)
    if en_unscaled_map and var in en_unscaled_map and row.get("elasticnet_json") is not None:
        try:
            val = float(en_unscaled_map[var])
            row["elasticnet_json"]["coef_unscaled_override"] = val
            row["notes"].append("elasticnet_unscaled_override_applied")
        except Exception:
            row["notes"].append("elasticnet_unscaled_override_failed")

    # attach sd + n from standardized summary if present, else compute from features if available
    if row["standardized_summary"]:
        ss = row["standardized_summary"]
        # support both dict and the older wrapper forms
        if isinstance(ss, dict):
            row["sd_target"] = ss.get("sd_target")
            row["sd_var"] = ss.get("sd_var")
            # if standardized summary contains a per-model 'summary' dict, attach per-model n/coef
            if isinstance(ss.get("summary"), dict):
                row["summary"] = ss["summary"]
    else:
        # attempt compute sd & n from features file if provided / discoverable
        if features_path and features_path.exists():
            try:
                df = pd.read_csv(features_path, low_memory=False)
                if var in df.columns and "gdp_growth_pct" in df.columns:
                    sub = df[[var, "gdp_growth_pct"]].dropna()
                    row["sd_var"] = float(sub[var].std(ddof=0)) if not sub.empty else None
                    row["sd_target"] = float(sub["gdp_growth_pct"].std(ddof=0)) if not sub.empty else None
                    row["n_obs"] = int(len(sub))
            except Exception as e:
                row["notes"].append(f"sd_from_features_failed: {e}")

    return row


def build_model_comparison_table(snapshot: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for entry in snapshot:
        var = entry["variable"]
        std = entry.get("standardized_summary") or {}
        summ = std.get("summary") if isinstance(std, dict) else None
        # gracefully accept different shapes
        fe = summ.get("FE") if summ and isinstance(summ, dict) else (entry.get("fe_json") or {})
        ols = summ.get("OLS") if summ and isinstance(summ, dict) else (entry.get("ols_json") or {})
        en = summ.get("ElasticNet") if summ and isinstance(summ, dict) else (entry.get("elasticnet_json") or {})
        rows.append({
            "variable": var,
            "fe_coef": (fe.get("coef") if isinstance(fe, dict) else None),
            "fe_std_effect": (fe.get("standardized") if isinstance(fe, dict) else None),
            "ols_coef": (ols.get("coef_raw") or ols.get("coef") if isinstance(ols, dict) else None),
            "ols_std_effect": (ols.get("standardized") if isinstance(ols, dict) else None),
            "en_coef": (en.get("coef_raw") or en.get("coef") if isinstance(en, dict) else None),
            "en_std_effect": (en.get("standardized") if isinstance(en, dict) else None),
        })
    return pd.DataFrame(rows)


def write_markdown(snapshot: List[Dict[str, Any]], out_md: Path):
    lines = ["# Baseline snapshot (Stage 1)", f"Generated: {datetime.utcnow().isoformat()}Z", ""]
    for entry in snapshot:
        v = entry["variable"]
        lines.append(f"## {v}")
        if entry.get("sd_target") is not None:
            lines.append(f"- sd_target: {entry.get('sd_target')}")
        if entry.get("sd_var") is not None:
            lines.append(f"- sd_var: {entry.get('sd_var')}")
        lines.append(f"- notes: {entry.get('notes', [])}")
        lines.append("")
        ss = entry.get("standardized_summary") or {}
        if ss and isinstance(ss, dict):
            lines.append("### Standardized summary (json keys)")
            for k in sorted(ss.keys()):
                lines.append(f"- {k}")
            lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vars", nargs="+", default=VARS_DEFAULT)
    parser.add_argument("--rescale-en", action="store_true", help="Apply artifacts/elasticnet_unscaled_coefs.json overrides if present")
    parser.add_argument("--features", default=str(FEATURES_DEFAULT), help="Optional features CSV to compute SDs if missing")
    args = parser.parse_args()

    _ensure_dirs()
    features_path = Path(args.features) if args.features else None

    en_map = None
    if args.rescale_en:
        p = ARTIFACTS / "elasticnet_unscaled_coefs.json"
        en_map = safe_json_load(p) or None
        if en_map:
            LOG.info("Loaded elasticnet unscaled coef map from %s", p)
        else:
            LOG.info("No elasticnet_unscaled_coefs.json found at %s (continuing without)", p)

    snapshot = []
    for v in args.vars:
        entry = collect_entry(v, en_unscaled_map=en_map, features_path=features_path)
        snapshot.append(entry)

    # write main json
    out_json = OUT_BASE / "generate_baseline_snapshot.json"
    out_json.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf8")
    LOG.info("Wrote snapshot JSON -> %s", out_json)

    # build and write model comparison CSV
    df_tab = build_model_comparison_table(snapshot)
    out_csv = OUT_BASE / "model_comparison_table.csv"
    df_tab.to_csv(out_csv, index=False)
    LOG.info("Wrote model comparison table -> %s", out_csv)

    # write markdown summary
    out_md = OUT_BASE / "generate_baseline_snapshot.md"
    write_markdown(snapshot, out_md)
    LOG.info("Wrote markdown summary -> %s", out_md)

    # copy per-variable artifacts into files/ for provenance (best-effort)
    produced = []
    for v in args.vars:
        # standardized json
        s = OUT_STANDARDIZED / f"{v}_standardized.json"
        if s.exists():
            cp = safe_copy_if_exists(s, OUT_FILES / "standardized")
            if cp:
                produced.append(str(cp))
        # per-model jsons in outputs/
        for kind in ("fe", "ols", "elasticnet"):
            p = Path("outputs") / f"{kind}_{v}.json"
            if p.exists():
                cp = safe_copy_if_exists(p, OUT_FILES)
                if cp:
                    produced.append(str(cp))
        # also copy extractor outputs if present
        alt = OUT_VARIABLES / v / f"{v}_summary.json"
        if alt.exists():
            cp = safe_copy_if_exists(alt, OUT_FILES / "standardized")
            if cp:
                produced.append(str(cp))

    # meta file with provenance
    meta = {
        "script": str(Path(__file__).resolve()),
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "vars": args.vars,
        "features_file": str(features_path) if features_path else None,
        "features_sha256": sha256_of_file(features_path) if features_path and features_path.exists() else None,
        "produced_files": produced,
        "git_commit": git_commit_hash(),
    }
    meta_path = OUT_BASE / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf8")
    LOG.info("Wrote meta -> %s", meta_path)

    # summary log
    LOG.info("Baseline snapshot generation complete. See %s", OUT_BASE)
    print("Wrote:")
    print(" -", out_json)
    print(" -", out_csv)
    print(" -", out_md)
    print(" -", meta_path)


if __name__ == "__main__":
    main()
