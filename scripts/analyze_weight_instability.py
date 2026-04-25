from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


AGGREGATE_INPUT = Path("analysis_out/aggregate_weight_sweep.csv")
RAW_INPUT = Path("rmd17_aspirin_weight_sweep.json")
REPORT_OUTPUT = Path("analysis_out/WEIGHT_SWEEP_INSTABILITY.md")
RATIO_OUTPUT = Path("analysis_out/weight_sweep_horizon_ratios.csv")
INSTABILITY_RATIO_THRESHOLD = 20.0
INSTABILITY_H16_THRESHOLD = 1.0
EXPECTED_PRIORS = ["none", "euclidean", "spectral"]
EXPECTED_WEIGHTS = ["0.001", "0.01", "0.1", "1.0"]
RATIO_COLUMNS = [
    "level",
    "molecule",
    "encoder",
    "prior",
    "prior_weight",
    "seed",
    "h1",
    "h16",
    "h16_over_h1",
    "unstable",
    "reasons",
]
BEST_COLUMNS = ["prior", "best_weight", "h16_mean", "h16_std", "n"]


def parse_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def format_float(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.10g}"


def instability_reasons(h16: float | None, ratio: float | None) -> list[str]:
    reasons = []
    if ratio is not None and ratio > INSTABILITY_RATIO_THRESHOLD:
        reasons.append(f"H16/H1 > {INSTABILITY_RATIO_THRESHOLD:g}")
    if h16 is not None and h16 > INSTABILITY_H16_THRESHOLD:
        reasons.append(f"H16 > {INSTABILITY_H16_THRESHOLD:g}")
    return reasons


def ratio_row(
    *,
    level: str,
    molecule: str,
    encoder: str,
    prior: str,
    prior_weight: str,
    seed: str = "",
    h1: float | None,
    h16: float | None,
) -> dict[str, str]:
    ratio = h16 / h1 if h1 is not None and h16 is not None and abs(h1) > 1e-12 else None
    reasons = instability_reasons(h16, ratio)
    return {
        "level": level,
        "molecule": molecule,
        "encoder": encoder,
        "prior": prior,
        "prior_weight": prior_weight,
        "seed": seed,
        "h1": format_float(h1),
        "h16": format_float(h16),
        "h16_over_h1": format_float(ratio),
        "unstable": "true" if reasons else "false",
        "reasons": "; ".join(reasons),
    }


def parse_key(key: str) -> dict[str, str]:
    parts = key.split("|")
    meta = {
        "molecule": parts[0] if len(parts) > 0 else "",
        "encoder": parts[1] if len(parts) > 1 else "",
        "prior": parts[2] if len(parts) > 2 else "",
        "prior_weight": "",
        "seed": "",
    }
    for part in parts:
        if part.startswith("w="):
            meta["prior_weight"] = part.split("=", 1)[1]
        elif part.startswith("seed="):
            meta["seed"] = part.split("=", 1)[1]
    return meta


def load_aggregate_rows(path: Path) -> list[dict[str, str]]:
    by_group: dict[tuple[str, str, str, str], dict[int, float]] = defaultdict(dict)
    exemplars: dict[tuple[str, str, str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            horizon = int(float(row.get("horizon", "")))
            mean = parse_float(row.get("mean"))
            if mean is None or horizon not in {1, 16}:
                continue
            key = (
                row.get("molecule", ""),
                row.get("encoder", ""),
                row.get("prior", ""),
                row.get("prior_weight", ""),
            )
            by_group[key][horizon] = mean
            exemplars.setdefault(key, row)

    output = []
    for key, by_horizon in sorted(by_group.items()):
        molecule, encoder, prior, prior_weight = key
        output.append(
            ratio_row(
                level="aggregate",
                molecule=molecule,
                encoder=encoder,
                prior=prior,
                prior_weight=prior_weight,
                h1=by_horizon.get(1),
                h16=by_horizon.get(16),
            )
        )
    return output


def load_aggregate_h16(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("horizon") == "16":
                rows.append(row)
    return rows


def is_weight_sweep_complete(h16_rows: list[dict[str, str]]) -> bool:
    by_group = {
        (row.get("prior", ""), row.get("prior_weight", "")): row
        for row in h16_rows
    }
    for prior in EXPECTED_PRIORS:
        for weight in EXPECTED_WEIGHTS:
            row = by_group.get((prior, weight))
            if row is None or row.get("n") != "3":
                return False
    return True


def best_h16_settings(h16_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for prior in EXPECTED_PRIORS:
        candidates = [
            row
            for row in h16_rows
            if row.get("prior") == prior and parse_float(row.get("mean")) is not None
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: parse_float(row.get("mean")) or float("inf"))
        output.append(
            {
                "prior": prior,
                "best_weight": best.get("prior_weight", ""),
                "h16_mean": format_float(parse_float(best.get("mean"))),
                "h16_std": format_float(parse_float(best.get("std"))),
                "n": best.get("n", ""),
            }
        )
    return output


def best_value(best_rows: list[dict[str, str]], prior: str) -> float | None:
    for row in best_rows:
        if row.get("prior") == prior:
            return parse_float(row.get("h16_mean"))
    return None


def pct_improvement(new: float | None, baseline: float | None) -> float | None:
    if new is None or baseline is None or abs(baseline) < 1e-12:
        return None
    return 100.0 * (baseline - new) / baseline


def load_seed_rows(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", {})
    if not isinstance(results, dict):
        return []

    output = []
    for key, result in sorted(results.items()):
        if not isinstance(result, dict):
            continue
        meta = parse_key(key)
        rollout_errors = result.get("rollout_errors", {})
        if not isinstance(rollout_errors, dict):
            continue
        output.append(
            ratio_row(
                level="seed",
                molecule=meta["molecule"],
                encoder=meta["encoder"],
                prior=meta["prior"],
                prior_weight=meta["prior_weight"],
                seed=meta["seed"],
                h1=parse_float(rollout_errors.get("1", rollout_errors.get(1))),
                h16=parse_float(rollout_errors.get("16", rollout_errors.get(16))),
            )
        )
    return output


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RATIO_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RATIO_COLUMNS})


def write_report(
    path: Path,
    aggregate_rows: list[dict[str, str]],
    seed_rows: list[dict[str, str]],
    h16_rows: list[dict[str, str]],
) -> None:
    unstable_seed_rows = [row for row in seed_rows if row["unstable"] == "true"]
    spectral_unstable = [
        row
        for row in seed_rows
        if row["prior"] == "spectral" and row["prior_weight"] == "0.01" and row["seed"] in {"1", "2"}
    ]
    complete = is_weight_sweep_complete(h16_rows)
    best_rows = best_h16_settings(h16_rows)
    best_none = best_value(best_rows, "none")
    best_euclidean = best_value(best_rows, "euclidean")
    best_spectral = best_value(best_rows, "spectral")
    spectral_vs_euclidean = pct_improvement(best_spectral, best_euclidean)
    spectral_vs_none = pct_improvement(best_spectral, best_none)
    interpretation = []
    if best_spectral is not None and best_euclidean is not None and best_spectral < best_euclidean:
        interpretation.append(f"Best spectral improves over best Euclidean by {format_float(spectral_vs_euclidean)}%.")
    if best_euclidean is not None and best_none is not None and best_euclidean < best_none:
        interpretation.append("Euclidean regularization improves over no-prior baseline when tuned.")
    if any(row["unstable"] == "true" for row in spectral_unstable):
        interpretation.append("Spectral w=0.01 is a long-horizon instability regime, not a uniformly bad spectral result.")
    interpretation.append(f"best_spectral_vs_best_euclidean_pct: {format_float(spectral_vs_euclidean)}")
    interpretation.append(f"best_spectral_vs_none_pct: {format_float(spectral_vs_none)}")
    lines = [
        "# Weight Sweep Instability",
        "",
        "Instability is flagged when `H16/H1 > 20` or `H16 > 1.0`.",
        "",
        "The current weight sweep is complete when aggregate rows contain n=3 for every prior/weight group at H=16.",
        "",
        "Completeness: COMPLETE." if complete else "Completeness: PARTIAL; aggregate ratios should not be treated as final claims.",
        "",
        "## Aggregate H16/H1",
        "",
        markdown_table(aggregate_rows, ["prior", "prior_weight", "h1", "h16", "h16_over_h1", "unstable", "reasons"]),
        "",
        "## Best H=16 Settings",
        "",
        markdown_table(best_rows, BEST_COLUMNS),
        "",
        *interpretation,
        "",
        "## Seed-Level Instability Flags",
        "",
        markdown_table(unstable_seed_rows, ["prior", "prior_weight", "seed", "h1", "h16", "h16_over_h1", "reasons"]),
        "",
        "## Spectral w=0.01 Long-Horizon Instability",
        "",
        "Spectral `w=0.01` seed `1` and seed `2` are long-horizon unstable in the current sweep.",
        "",
        markdown_table(spectral_unstable, ["prior", "prior_weight", "seed", "h1", "h16", "h16_over_h1", "unstable", "reasons"]),
        "",
        "## Notes",
        "",
        "- This script reads existing analysis artifacts only.",
        "- Training code and raw result JSONs are not modified.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", default=str(AGGREGATE_INPUT))
    parser.add_argument("--raw", default=str(RAW_INPUT))
    parser.add_argument("--out-md", default=str(REPORT_OUTPUT))
    parser.add_argument("--out-csv", default=str(RATIO_OUTPUT))
    args = parser.parse_args()

    aggregate_rows = load_aggregate_rows(Path(args.aggregate))
    h16_rows = load_aggregate_h16(Path(args.aggregate))
    seed_rows = load_seed_rows(Path(args.raw))
    all_rows = aggregate_rows + seed_rows
    write_csv(Path(args.out_csv), all_rows)
    write_report(Path(args.out_md), aggregate_rows, seed_rows, h16_rows)
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
