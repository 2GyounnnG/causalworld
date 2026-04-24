#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
OUT="${2:-analysis_out}"

python scripts/build_manifest.py --root "$ROOT" --out "$OUT"
python scripts/aggregate_results.py --manifest "$OUT/manifest.csv" --out "$OUT"
python scripts/plot_results.py --analysis "$OUT"
python scripts/audit_priors_and_laplacians.py --root "$ROOT" --out "$OUT"

echo "analysis pipeline complete"
echo "manifest: $OUT/manifest.csv"
python - "$OUT" <<'PY'
import csv
import sys
from pathlib import Path

out = Path(sys.argv[1])
manifest = out / "manifest.csv"
failed = out / "failed_or_incomplete_runs.csv"

def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))

def count_incomplete(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as file:
        return sum(1 for row in csv.DictReader(file) if row.get("completed") != "true")

print(f"manifest rows: {count_rows(manifest)}")
print(f"incomplete/failed rows: {count_incomplete(failed)}")
PY
echo "aggregate_by_prior_horizon: $OUT/aggregate_by_prior_horizon.csv"
echo "aggregate_weight_sweep: $OUT/aggregate_weight_sweep.csv"
echo "aggregate_laplacian_ablation: $OUT/aggregate_laplacian_ablation.csv"
echo "aggregate_wolfram: $OUT/aggregate_wolfram.csv"
echo "plots: $OUT/plots/"
echo "audit: $OUT/AUDIT_PRIORS_AND_LAPLACIANS.md"
echo "summary: $OUT/RESULTS_SUMMARY_FOR_PAPER.md"
