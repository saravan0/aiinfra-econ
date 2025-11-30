"""
Simple utilities for dataset provenance:
- compute_md5(path): writes <path>.md5
- update_sources_yaml(canonical_id, checksum): updates data/raw/sources.yaml
"""

from __future__ import annotations
import hashlib
import logging
import datetime
from pathlib import Path
from typing import Optional

import yaml

LOG = logging.getLogger(__name__)

# Resolve project root → works regardless of run location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCES_FILE = PROJECT_ROOT / "data" / "raw" / "sources.yaml"


def compute_md5(file_path: str | Path) -> str:
    """Compute MD5 for a file and write <file>.md5 next to it."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"{file_path} not found")

    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    md5 = h.hexdigest()
    (p.parent / (p.name + ".md5")).write_text(md5, encoding="utf8")
    LOG.debug("Wrote md5 for %s", p)
    return md5


def update_sources_yaml(canonical_id: str, checksum: str) -> bool:
    """
    Update last_fetch + checksum for an entry in sources.yaml.
    Returns True if the entry was updated.
    """
    if not SOURCES_FILE.exists():
        LOG.warning("sources.yaml not found at %s", SOURCES_FILE)
        return False

    with SOURCES_FILE.open("r", encoding="utf8") as f:
        data = yaml.safe_load(f) or {}

    changed = False
    for src in data.get("sources", []):
        if src.get("canonical_id") == canonical_id:
            src["last_fetch"] = datetime.datetime.utcnow().isoformat() + "Z"
            src["checksum"] = checksum
            changed = True
            break

    if changed:
        with SOURCES_FILE.open("w", encoding="utf8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        LOG.info("Updated sources.yaml for %s", canonical_id)
    else:
        LOG.debug("canonical_id %s not found in sources.yaml", canonical_id)

    return changed


def record_artifact(file_path: str | Path, canonical_id: Optional[str] = None) -> Optional[str]:
    """Compute md5 and update sources registry if canonical_id is given."""
    try:
        md5 = compute_md5(file_path)
    except Exception as exc:
        LOG.error("compute_md5 failed for %s: %s", file_path, exc)
        return None

    if canonical_id:
        update_sources_yaml(canonical_id, md5)

    return md5
