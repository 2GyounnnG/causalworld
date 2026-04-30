#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON:-python}"

"$PYTHON_BIN" scripts/train_cycle0.py \
  --all \
  --config-dir experiments/configs/cycle0_aspirin_smoke \
  --output experiments/results/cycle0_aspirin_smoke/cycle0_aspirin_smoke_results.json

"$PYTHON_BIN" scripts/analyze_cycle0_smoke.py \
  --results experiments/results/cycle0_aspirin_smoke/cycle0_aspirin_smoke_results.json \
  --report analysis_out/CYCLE0_SMOKE_REPORT.md
