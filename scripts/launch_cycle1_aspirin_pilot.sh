#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON:-python}"

"$PYTHON_BIN" scripts/train_cycle0.py \
  --all \
  --config-dir experiments/configs/cycle1_aspirin_pilot \
  --output experiments/results/cycle1_aspirin_pilot/cycle1_aspirin_pilot_results.json \
  --schema-version cycle1_aspirin_pilot_v1

"$PYTHON_BIN" scripts/analyze_cycle1_aspirin_pilot.py \
  --results experiments/results/cycle1_aspirin_pilot/cycle1_aspirin_pilot_results.json \
  --report analysis_out/CYCLE1_ASPIRIN_PILOT_REPORT.md
