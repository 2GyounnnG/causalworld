from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOT = ROOT / "analysis_out" / "preflight_runs"
FIELDS = [
    "path",
    "classification",
    "strict_label",
    "diagnostic_failure_mode",
    "recommended_next_experiment",
    "claim_boundary",
]


def main() -> None:
    print("\t".join(FIELDS))
    for path in sorted(SEARCH_ROOT.glob("**/summary.csv")):
        with path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.get("stage") != "classification":
                    continue
                values = [str(path.relative_to(ROOT))]
                values.extend(str(row.get(field, "")) for field in FIELDS[1:])
                print("\t".join(values))


if __name__ == "__main__":
    main()
