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

CONFIG_DIR="experiments/configs/cycle5_ho_lattice_bridge"
RESULT_DIR="experiments/results/cycle5_ho_lattice_bridge"
RESULTS="$RESULT_DIR/cycle5_ho_lattice_bridge_results.json"
REPORT="analysis_out/CYCLE5_HO_LATTICE_BRIDGE_REPORT.md"

mkdir -p "$CONFIG_DIR" "$RESULT_DIR"

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

config_dir = Path("experiments/configs/cycle5_ho_lattice_bridge")
config_dir.mkdir(parents=True, exist_ok=True)

priors = ["graph", "permuted_graph", "random_graph"]
prior_weights = [0.001, 0.005, 0.01, 0.05, 0.1]
seeds = [0, 1, 2, 3, 4]

for prior in priors:
    for prior_weight in prior_weights:
        label = f"{prior_weight:g}".replace(".", "p")
        for seed in seeds:
            run_name = f"lattice_gnn_{prior}_lambda{label}_seed{seed}"
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
                "prior_weight": prior_weight,
                "seed": seed,
            }
            path = config_dir / f"{run_name}.json"
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

"$PYTHON_BIN" scripts/train_cycle3_ho_networks.py \
  --all \
  --config-dir "$CONFIG_DIR" \
  --output "$RESULTS" \
  --schema-version cycle5_ho_lattice_bridge_v1

"$PYTHON_BIN" scripts/analyze_cycle5_ho_lattice_bridge.py \
  --results "$RESULTS" \
  --cycle3-results experiments/results/cycle3_ho_networks/cycle3_ho_networks_results.json \
  --report "$REPORT"
