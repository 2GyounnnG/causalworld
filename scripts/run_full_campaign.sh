#!/usr/bin/env bash
set -euo pipefail

START_EPOCH=$(date +%s)
echo "=== Full multiseed campaign starting at $(date) ==="

echo "--- N-body (largest, ~1.7 hours) ---"
python scripts/run_nbody_robustness_5seed.py --conda-env causalworld

echo "--- METR-LA ---"
python scripts/run_metr_la_3seed.py --conda-env causalworld

echo "--- HO audit ---"
python scripts/run_ho_audit_5seed.py --conda-env causalworld

echo "--- Spring-mass ---"
python scripts/run_spring_mass_5seed.py --conda-env causalworld

echo "--- Graph-wave ---"
python scripts/run_graph_wave_5seed.py --conda-env causalworld

echo "--- Graph-heat ---"
python scripts/run_graph_heat_3seed.py --conda-env causalworld

echo "--- Node-order sanity ---"
python scripts/run_node_order_sanity.py --conda-env causalworld

echo "--- Aggregators ---"
python scripts/aggregate_nbody_robustness_5seed.py
python scripts/aggregate_ho_audit_5seed.py
python scripts/aggregate_spring_mass_5seed.py
python scripts/aggregate_graph_wave_5seed.py
python scripts/aggregate_metr_la_3seed.py
python scripts/aggregate_graph_heat_3seed.py
python scripts/aggregate_node_order_sanity.py
python scripts/aggregate_all_multiseed.py

END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS_LEFT=$((ELAPSED % 60))
echo "=== Campaign complete in ${HOURS}h ${MINUTES}m ${SECONDS_LEFT}s at $(date) ==="
