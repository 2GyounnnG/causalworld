#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "$HOME/miniforge3/envs/causalworld/bin/python" ]]; then
  PYTHON_BIN="$HOME/miniforge3/envs/causalworld/bin/python"
else
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" scripts/train_cycle3_ho_networks.py \
  --all \
  --config-dir experiments/configs/cycle3_ho_networks \
  --output experiments/results/cycle3_ho_networks/cycle3_ho_networks_results.json \
  --schema-version cycle3_ho_networks_v1

"$PYTHON_BIN" scripts/analyze_cycle3_ho_networks.py \
  --results experiments/results/cycle3_ho_networks/cycle3_ho_networks_results.json \
  --report analysis_out/CYCLE3_HO_NETWORKS_REPORT.md
