from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


AGGREGATE_INPUT = Path("analysis_out/aggregate_wolfram.csv")
RAW_INPUT = Path("validation_wolfram_flat_10seed_200ep.json")
REPORT_OUTPUT = Path("analysis_out/WOLFRAM_INSTABILITY.md")
RATIO_OUTPUT = Path("analysis_out/wolfram_horizon_ratios.csv")
PRIORS = ["none", "euclidean", "spectral"]
INSTABILITY_RATIO_THRESHOLD = 20.0
INSTABILITY_H16_THRESHOLD = 1.0
RATIO_COLUMNS = [
    "level",
    "encoder",
    "prior",
    "seed",
    "h1",
    "h16",
    "h16_over_h1",
    "unstable",
    "reasons",
]


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


def ratio_row(*, level: str, encoder: str, prior: str, h1: float | None, h16: float | None, seed: str = "") -> dict[str, str]:
    ratio = h16 / h1 if h1 is not None and h16 is not None and abs(h1) > 1e-12 else None
    reasons = instability_reasons(h16, ratio)
    return {
        "level": level,
        "encoder": encoder,
        "prior": prior,
        "seed": seed,
        "h1": format_float(h1),
        "h16": format_float(h16),
        "h16_over_h1": format_float(ratio),
        "unstable": "true" if reasons else "false",
        "reasons": "; ".join(reasons),
    }


def load_aggregate(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    by_prior: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            prior = row.get("prior", "")
            if prior not in PRIORS:
                continue
            try:
                horizon = int(float(row.get("horizon", "")))
            except (TypeError, ValueError):
                continue
            if horizon in {1, 16}:
                by_prior[prior][horizon] = row

    ratio_rows = []
    h16_stats = {}
    for prior in PRIORS:
        h1_row = by_prior.get(prior, {}).get(1, {})
        h16_row = by_prior.get(prior, {}).get(16, {})
        ratio_rows.append(
            ratio_row(
                level="aggregate",
                encoder=h16_row.get("encoder", h1_row.get("encoder", "")),
                prior=prior,
                h1=parse_float(h1_row.get("mean")),
                h16=parse_float(h16_row.get("mean")),
            )
        )
        h16_stats[prior] = {
            "prior": prior,
            "h16_mean": format_float(parse_float(h16_row.get("mean"))),
            "h16_median": format_float(parse_float(h16_row.get("median"))),
            "h16_std": format_float(parse_float(h16_row.get("std"))),
            "n": h16_row.get("n", ""),
        }
    return ratio_rows, h16_stats


def load_seed_rows(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", {})
    completed_seeds = data.get("completed_seeds")
    config = data.get("config", {})
    seeds = completed_seeds if isinstance(completed_seeds, list) else config.get("seeds", [])
    if not isinstance(results, dict):
        return []

    rows = []
    for key, by_horizon in sorted(results.items()):
        if not isinstance(by_horizon, dict):
            continue
        parts = key.split("|")
        encoder = parts[0] if parts else ""
        prior = parts[1] if len(parts) > 1 else ""
        h1_values = by_horizon.get("1", [])
        h16_values = by_horizon.get("16", [])
        if not isinstance(h1_values, list) or not isinstance(h16_values, list):
            continue
        for index, h1_value in enumerate(h1_values):
            if index >= len(h16_values):
                continue
            seed = seeds[index] if isinstance(seeds, list) and index < len(seeds) else index
            rows.append(
                ratio_row(
                    level="seed",
                    encoder=encoder,
                    prior=prior,
                    seed=str(seed),
                    h1=parse_float(h1_value),
                    h16=parse_float(h16_values[index]),
                )
            )
    return rows


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


def write_report(path: Path, aggregate_rows: list[dict[str, str]], seed_rows: list[dict[str, str]], h16_stats: dict[str, dict[str, str]]) -> None:
    unstable_seed_rows = [row for row in seed_rows if row.get("unstable") == "true"]
    spectral_h16_rows = sorted(
        [row for row in seed_rows if row.get("prior") == "spectral"],
        key=lambda row: parse_float(row.get("h16")) or float("-inf"),
        reverse=True,
    )
    best_rows = [
        {
            "setting": "none mean",
            "h16": h16_stats.get("none", {}).get("h16_mean", ""),
            "n": h16_stats.get("none", {}).get("n", ""),
        },
        {
            "setting": "euclidean mean",
            "h16": h16_stats.get("euclidean", {}).get("h16_mean", ""),
            "n": h16_stats.get("euclidean", {}).get("n", ""),
        },
        {
            "setting": "spectral mean",
            "h16": h16_stats.get("spectral", {}).get("h16_mean", ""),
            "n": h16_stats.get("spectral", {}).get("n", ""),
        },
        {
            "setting": "spectral median",
            "h16": h16_stats.get("spectral", {}).get("h16_median", ""),
            "n": h16_stats.get("spectral", {}).get("n", ""),
        },
    ]
    mean_median_rows = [
        {
            "prior": prior,
            "h16_mean": h16_stats.get(prior, {}).get("h16_mean", ""),
            "h16_median": h16_stats.get(prior, {}).get("h16_median", ""),
            "h16_std": h16_stats.get(prior, {}).get("h16_std", ""),
            "n": h16_stats.get(prior, {}).get("n", ""),
        }
        for prior in PRIORS
    ]
    best_mean_prior = min(
        PRIORS,
        key=lambda prior: parse_float(h16_stats.get(prior, {}).get("h16_mean")) or float("inf"),
    )
    lines = [
        "# Wolfram Instability",
        "",
        "Instability is flagged when `H16/H1 > 20` or `H16 > 1.0`.",
        "",
        "## Aggregate H16/H1",
        "",
        markdown_table(aggregate_rows, ["prior", "h1", "h16", "h16_over_h1", "unstable", "reasons"]),
        "",
        "## Seed-Level Instability Flags",
        "",
        markdown_table(unstable_seed_rows, ["prior", "seed", "h1", "h16", "h16_over_h1", "reasons"]),
        "",
        "## Spectral Seed-Level H=16",
        "",
        markdown_table(spectral_h16_rows, ["seed", "h1", "h16", "h16_over_h1", "unstable", "reasons"]),
        "",
        "## Mean Vs Median H=16",
        "",
        markdown_table(mean_median_rows, ["prior", "h16_mean", "h16_median", "h16_std", "n"]),
        "",
        "## Best H=16 Setting",
        "",
        markdown_table(best_rows, ["setting", "h16", "n"]),
        "",
        f"Best H=16 mean setting: `{best_mean_prior}`.",
        "",
        "## Interpretation",
        "",
        "Euclidean has the best H16 mean in the completed Wolfram flat 200ep run.",
        "",
        "Spectral has heavy-tailed long-horizon instability.",
        "",
        "Spectral seed 1, seed 2, and seed 8 explode at H=16.",
        "",
        "Most spectral seeds remain near baseline scale, so the failure is heavy-tailed rather than uniform.",
        "",
        "Wolfram spectral has heavy-tailed long-horizon instability; 7/10 seeds are near baseline scale, but seeds 1, 2, and 8 explode at H=16.",
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

    aggregate_rows, h16_stats = load_aggregate(Path(args.aggregate))
    seed_rows = load_seed_rows(Path(args.raw))
    write_csv(Path(args.out_csv), aggregate_rows + seed_rows)
    write_report(Path(args.out_md), aggregate_rows, seed_rows, h16_stats)
    print(f"wrote {args.out_md}")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
