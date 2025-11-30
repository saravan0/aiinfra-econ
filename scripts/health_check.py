#!/usr/bin/env python3
"""
scripts/health_check.py

Cross-platform health check that mirrors the PowerShell behavior.
Usage:
    python scripts/health_check.py --config config/model.yml
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import shutil
import re
import time
import json

try:
    import pandas as pd
except Exception:
    pd = None

ROOT = Path.cwd()
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

DEFAULT_EXPECTED = [
    "models/ols_model.joblib",
    "models/fe_model.joblib",
    "models/elasticnet_cv.joblib",
    "reports/model_table.csv",
    "reports/model_plots.png",
    "reports/model_artifacts_manifest.json",
    "reports/robustness_vif.csv",
    "reports/robustness_card.md",
]

ERR_PATTERN = re.compile(r"ERROR|Traceback|Exception", flags=re.IGNORECASE)


def run_module(module: str, args: str, out_log: Path, timeout: int | None = None) -> int:
    """
    Run python -u -m <module> <args>, stream output to out_log and stdout.
    Returns the process exit code.
    """
    cmd = [sys.executable, "-u", "-m", module] + (args.split() if args else [])
    out_log.parent.mkdir(parents=True, exist_ok=True)
    with out_log.open("wb") as fh:
        # Start process
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.stdout is None:
            return proc.wait()
        try:
            for raw in proc.stdout:
                fh.write(raw)
                fh.flush()
                try:
                    sys.stdout.buffer.write(raw)
                except Exception:
                    # fallback for environments where buffer is not writable
                    try:
                        sys.stdout.write(raw.decode("utf8", "replace"))
                    except Exception:
                        pass
            ret = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            ret = proc.wait()
        return ret


def scan_logs_for_patterns(paths: list[Path], pattern: re.Pattern) -> list[str]:
    matches = []
    for p in paths:
        if not p.exists():
            continue
        try:
            for i, line in enumerate(p.read_text(encoding="utf8", errors="ignore").splitlines(), start=1):
                if pattern.search(line):
                    matches.append(f"{p}:{i}: {line}")
        except Exception:
            continue
    return matches


def check_expected(expected_list: list[str]) -> list[str]:
    missing = []
    for p in expected_list:
        if not (ROOT / p).exists():
            missing.append(p)
    return missing


def show_vif_top(path: Path, topn: int = 10):
    if pd is None:
        print("pandas not available — cannot show VIF CSV preview.")
        return
    try:
        df = pd.read_csv(path)
        if "vif" in df.columns:
            df["vif"] = pd.to_numeric(df["vif"], errors="coerce")
            top = df.sort_values("vif", ascending=False).head(topn)
            print(top.to_string(index=False))
        else:
            print("VIF column not found; printing head instead:")
            print(df.head(topn).to_string(index=False))
    except Exception as e:
        print("Could not read vif file:", e)


def show_csv_head(path: Path, topn: int = 10):
    if pd is None:
        print(f"pandas not available — print raw head of {path} instead.")
        try:
            print("\n".join(path.read_text(encoding="utf8", errors="ignore").splitlines()[:topn]))
        except Exception as e:
            print("Failed to read file:", e)
        return
    try:
        df = pd.read_csv(path)
        print(df.head(topn).to_string(index=False))
    except Exception as e:
        print("Could not read CSV:", e)


def show_text_head(path: Path, lines: int = 40):
    try:
        s = path.read_text(encoding="utf8", errors="ignore")
        print("\n".join(s.splitlines()[:lines]))
    except Exception as e:
        print(f"Could not read {path}: {e}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/model.yml")
    p.add_argument("--ci", action="store_true", help="Exit nonzero on any issue (useful for CI).")
    args = p.parse_args(argv)

    print("=== START health_check ===\n")

    # 1) Train
    train_log = LOGS_DIR / "train_run.log"
    print("==> Running: python -u -m src.model.train --config", args.config)
    rc = run_module("src.model.train", f"--config {args.config}", train_log)
    print(f"\n[train rc={rc}] logged to {train_log}\n")
    if rc != 0 and args.ci:
        print("Training step failed; aborting (ci mode).")
        return rc

    # 2) Robustness
    robust_log = LOGS_DIR / "robust_run.log"
    print("==> Running: python -u -m src.model.robustness --config", args.config)
    rc2 = run_module("src.model.robustness", f"--config {args.config}", robust_log)
    print(f"\n[robustness rc={rc2}] logged to {robust_log}\n")
    if rc2 != 0 and args.ci:
        print("Robustness step failed; aborting (ci mode).")
        return rc2

    # 3) Artifact checks
    print("Checking expected artifacts...")
    missing = check_expected(DEFAULT_EXPECTED)
    if missing:
        print("\nERROR: missing artifact(s):")
        for m in missing:
            print(" -", m)
        if args.ci:
            return 2
    else:
        print("Artifacts OK.\n")

    # 4) Scan logs for errors
    print("Scanning logs for ERROR/Traceback/Exception")
    matches = scan_logs_for_patterns([train_log, robust_log], ERR_PATTERN)
    if matches:
        print("\nFound log matches:")
        for line in matches:
            print(line)
    else:
        print("No obvious ERROR/Traceback/Exception lines found in logs.\n")

    # 5) VIF top rows
    vif_path = ROOT / "reports" / "robustness_vif.csv"
    print("\nTop VIF (if available):")
    if vif_path.exists():
        show_vif_top(vif_path)
    else:
        print("No VIF file found.")

    # 6) Model table top rows
    mt_path = ROOT / "reports" / "model_table.csv"
    print("\nModel table (top rows):")
    if mt_path.exists():
        show_csv_head(mt_path)
    else:
        print("No model_table.csv")

    # 7) FE summary head
    fe_summary = ROOT / "reports" / "fe_summary.txt"
    print("\nFE summary (first lines):")
    if fe_summary.exists():
        show_text_head(fe_summary)
    else:
        print("fe_summary.txt not found.")

    print("\n=== END health_check ===")
    # return non-zero if any major problems found (only when ci requested)
    if args.ci and (missing or rc != 0 or rc2 != 0 or matches):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
