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

CONFIG_DIR="experiments/configs/cycle8_checkpointed_lattice_latent_alignment"
RESULT_DIR="experiments/results/cycle8_checkpointed_lattice_latent_alignment"
CHECKPOINT_DIR="experiments/checkpoints/cycle8_checkpointed_lattice"
ARTIFACT_DIR="experiments/artifacts/cycle8_checkpointed_lattice"
RESULTS="$RESULT_DIR/cycle8_checkpointed_lattice_latent_alignment_results.json"
REPORT="analysis_out/CYCLE8_CHECKPOINTED_LATENT_ALIGNMENT_REPORT.md"

mkdir -p "$CONFIG_DIR" "$RESULT_DIR" "$CHECKPOINT_DIR" "$ARTIFACT_DIR"

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

config_dir = Path("experiments/configs/cycle8_checkpointed_lattice_latent_alignment")
config_dir.mkdir(parents=True, exist_ok=True)

priors = ["graph", "permuted_graph", "random_graph"]
seeds = [0, 1, 2, 3, 4]

for prior in priors:
    for seed in seeds:
        run_name = f"lattice_gnn_{prior}_lambda0p1_seed{seed}"
        config = {
            "topology": "lattice",
            "num_epochs": 30,
            "batch_size": 32,
            "latent_dim": 16,
            "node_dim": 16,
            "eval_horizons": [1, 2, 4, 8, 16, 32],
            "run_name": run_name,
            "encoder": "gnn_node",
            "prior": prior,
            "prior_weight": 0.1,
            "seed": seed,
        }
        path = config_dir / f"{run_name}.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

"$PYTHON_BIN" scripts/run_cycle8_checkpointed_lattice_latent_alignment.py \
  --all \
  --config-dir "$CONFIG_DIR" \
  --output "$RESULTS" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --artifact-dir "$ARTIFACT_DIR"

"$PYTHON_BIN" scripts/analyze_cycle8_latent_alignment.py \
  --results "$RESULTS" \
  --report "$REPORT"
