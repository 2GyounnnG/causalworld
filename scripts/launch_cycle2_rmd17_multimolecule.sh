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

"$PYTHON_BIN" scripts/train_cycle0.py \
  --all \
  --config-dir experiments/configs/cycle2_rmd17_multimolecule \
  --output experiments/results/cycle2_rmd17_multimolecule/cycle2_rmd17_multimolecule_results.json \
  --schema-version cycle2_rmd17_multimolecule_v1

"$PYTHON_BIN" scripts/analyze_cycle2_rmd17_multimolecule.py \
  --results experiments/results/cycle2_rmd17_multimolecule/cycle2_rmd17_multimolecule_results.json \
  --report analysis_out/CYCLE2_RMD17_MULTIMOLECULE_REPORT.md
