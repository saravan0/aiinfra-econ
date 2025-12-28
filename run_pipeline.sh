#!/usr/bin/env bash
set -e

echo "=== AI-Infra Econ Pipeline: START ==="
python -V

# 1. Data prep / validation
if [ -f src/data/safe_transforms_and_vif.py ]; then
  python -m src.data.safe_transforms_and_vif
fi

# 2. Core modeling + diagnostics (CANONICAL BASELINE)
python scripts/generate_baseline_snapshot.py \
  --config config/snapshot_config.json

# 3. One-pager + composites
python scripts/generate_onepager.py

echo "=== AI-Infra Econ Pipeline: DONE ==="
