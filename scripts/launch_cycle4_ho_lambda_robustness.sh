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
  --config-dir experiments/configs/cycle4_ho_lambda_robustness \
  --output experiments/results/cycle4_ho_lambda_robustness/cycle4_ho_lambda_robustness_results.json \
  --schema-version cycle4_ho_lambda_robustness_v1

"$PYTHON_BIN" scripts/analyze_cycle4_ho_lambda_robustness.py \
  --results experiments/results/cycle4_ho_lambda_robustness/cycle4_ho_lambda_robustness_results.json \
  --report analysis_out/CYCLE4_HO_LAMBDA_ROBUSTNESS_REPORT.md
