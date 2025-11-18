# src/pipeline/build_pipeline.py
"""
Pipeline orchestrator (Phase 1)

Runs the main data pipeline stages in order, preferring direct function calls
into polished modules when available and falling back to subprocess execution
for modules that don't expose an importable entrypoint.

Usage:
  python -m src.pipeline.build_pipeline            # run full pipeline
  python -m src.pipeline.build_pipeline --steps harmonize,master,clean
  python -m src.pipeline.build_pipeline --skip safe_transforms

Steps (default order):
  harmonize -> build_master -> clean_master -> verify -> features ->
  lock_schema -> metadata -> safe_transforms -> safe_transforms_vif

Design:
  - Conservative: stops on first error
  - Produces data_manifest.json listing key produced artifacts (sha1 + size)
  - Logs each step with timestamps (good for CI / reviewers)
"""

from __future__ import annotations
import argparse
import importlib
import json
import logging
import subprocess
import sys
import hashlib
import time
from pathlib import Path
from typing import List, Callable, Optional

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
MANIFEST_OUT = ROOT / "data_manifest.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger(__name__)


# Ordered pipeline steps and how we try to execute them:
# - module: import path (module), callable: attribute name to call (if exists)
# - fallback_cmd: command to run via subprocess if callable missing
PIPELINE_STEPS = {
    "harmonize": {
        "module": "src.data.harmonize",
        "callable": "main",
        "fallback_cmd": [sys.executable, "-m", "src.data.harmonize"],
    },
    "build_master": {
        "module": "src.data.build_master_final",
        "callable": "build_master",
        "fallback_cmd": [sys.executable, "-m", "src.data.build_master_final"],
    },
    "clean_master": {
        "module": "src.data.clean_master",
        "callable": "main",
        "fallback_cmd": [sys.executable, "-m", "src.data.clean_master"],
    },
    "verify": {
        "module": "src.data.verify_master_structure",
        "callable": "run_verification",
        "fallback_cmd": [sys.executable, "-m", "src.data.verify_master_structure"],
    },
    "features": {
        "module": "src.data.feature_engineer",
        "callable": "build_features",
        "fallback_cmd": [sys.executable, "-m", "src.data.feature_engineer"],
    },
    "lock_schema": {
        # lock_schema may not define a main callable; we try 'main' then fallback to module run
        "module": "src.data.lock_schema",
        "callable": "main",
        "fallback_cmd": [sys.executable, "-m", "src.data.lock_schema"],
    },
    "metadata": {
        "module": "src.data.generate_metadata",
        "callable": "make_metadata_card",
        "fallback_cmd": [sys.executable, "-m", "src.data.generate_metadata"],
    },
    "safe_transforms": {
        "module": "src.data.safe_transforms_and_vif",
        "callable": "main",
        "fallback_cmd": [sys.executable, "-m", "src.data.safe_transforms_and_vif"],
    },
}


def run_subprocess(cmd: List[str]) -> None:
    LOG.info("Running subprocess: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        LOG.info(proc.stdout.strip())
    if proc.stderr:
        LOG.warning(proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed: {' '.join(cmd)} (exit {proc.returncode})")
    LOG.info("Subprocess finished: %s", " ".join(cmd))


def try_import_and_call(module_name: str, callable_name: Optional[str], *args, **kwargs) -> bool:
    """
    Try to import module and call callable_name. Returns True if executed.
    Raises exceptions on failure of the callable.
    """
    try:
        m = importlib.import_module(module_name)
    except Exception as e:
        LOG.debug("Import failed for %s: %s", module_name, e)
        return False

    if not callable_name:
        return False

    if hasattr(m, callable_name):
        func = getattr(m, callable_name)
        if callable(func):
            LOG.info("Calling %s.%s()", module_name, callable_name)
            func(*args, **kwargs)
            LOG.info("Completed %s.%s()", module_name, callable_name)
            return True
    return False


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(paths: List[Path]) -> dict:
    manifest = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    for p in paths:
        if p.exists():
            manifest["files"][str(p.relative_to(ROOT))] = {"sha1": sha1(p), "size": p.stat().st_size}
    return manifest


def run_step(step_key: str, extra_args: dict) -> None:
    step = PIPELINE_STEPS[step_key]
    module = step["module"]
    callable_name = step.get("callable")
    fallback_cmd = step.get("fallback_cmd")

    # Try import + call
    executed = False
    try:
        executed = try_import_and_call(module, callable_name, **extra_args)
    except Exception as e:
        LOG.exception("Error while executing %s.%s: %s", module, callable_name, e)
        raise

    if executed:
        return

    # Fallback: subprocess-run the module
    if fallback_cmd:
        run_subprocess(fallback_cmd)
        return

    raise RuntimeError(f"No way to execute step {step_key} (module={module}, callable={callable_name})")


def parse_args():
    parser = argparse.ArgumentParser(prog="build_pipeline", description="Run project pipeline stages")
    parser.add_argument("--steps", type=str, default=",".join(PIPELINE_STEPS.keys()),
                        help="Comma-separated list of steps to run (in order).")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated list of steps to skip.")
    parser.add_argument("--keep-manifest", action="store_true",
                        help="Do not overwrite existing data_manifest.json")
    parser.add_argument("--metadata-master", type=str, default=str(INTERIM / "wgi_econ_master.csv"),
                        help="Master file to pass to metadata generator (used when calling make_metadata_card)")
    return parser.parse_args()


def main():
    args = parse_args()
    requested = [s.strip() for s in args.steps.split(",") if s.strip()]
    skips = [s.strip() for s in args.skip.split(",") if s.strip()]
    steps_to_run = [s for s in requested if s not in skips]

    LOG.info("Pipeline start. Steps to run: %s", steps_to_run)

    # Extra args we might pass to some callables
    extra_args = {}
    # For metadata generation, we call make_metadata_card(master_path, mapping_file, out_dir)
    # If make_metadata_card is callable we pass args; else fallback will run module's __main__.
    extra_args_for_metadata = {
        "master_path": Path(args.metadata_master),
        "mapping_file": Path(ROOT / "data" / "raw" / "mappings" / "column_map.csv"),
        "out_dir": INTERIM,
    }

    produced_files: List[Path] = []

    try:
        for step in steps_to_run:
            LOG.info("=== Step: %s ===", step)
            if step == "metadata":
                # special-case: try to call make_metadata_card with args
                step_meta = PIPELINE_STEPS[step]
                executed = try_import_and_call(step_meta["module"], "make_metadata_card",
                                               master_path=extra_args_for_metadata["master_path"],
                                               mapping_file=extra_args_for_metadata["mapping_file"],
                                               out_dir=extra_args_for_metadata["out_dir"])
                if not executed:
                    # fallback to module run
                    run_subprocess(step_meta["fallback_cmd"])
                # Record outputs
                produced_files.append(Path(extra_args_for_metadata["out_dir"]) / "column_provenance_summary.csv")
                produced_files.append(Path(extra_args_for_metadata["mapping_file"].parent / "column_map_with_provenance.csv"))
            else:
                run_step(step, extra_args={})

                # heuristics to add probable outputs to manifest
                if step == "build_master":
                    produced_files.append(INTERIM / "wgi_econ_master_raw.csv")
                    produced_files.append(INTERIM / "wgi_econ_master_missingness.csv")
                if step == "clean_master":
                    produced_files.append(INTERIM / "wgi_econ_master.csv")
                    produced_files.append(INTERIM / "panel_union.csv")
                    produced_files.append(INTERIM / "panel_core.csv")
                if step == "features":
                    produced_files.append(PROCESSED / "features.csv")
                if step == "lock_schema":
                    produced_files.append(PROCESSED / "features_lean.csv")
                if step == "safe_transforms":
                    produced_files.append(INTERIM / "monetary_scale_check.csv")
                    produced_files.append(INTERIM / "top_correlations.csv")
                    produced_files.append(INTERIM / "vif.csv")

        # Build and write manifest
        manifest = build_manifest(produced_files)
        if args.keep_manifest and MANIFEST_OUT.exists():
            LOG.info("Keeping existing manifest (no overwrite): %s", MANIFEST_OUT)
        else:
            MANIFEST_OUT.write_text(json.dumps(manifest, indent=2))
            LOG.info("Wrote manifest -> %s", MANIFEST_OUT)

        LOG.info("Pipeline finished successfully.")
    except Exception as e:
        LOG.exception("Pipeline failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
