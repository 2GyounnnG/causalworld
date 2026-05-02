"""Build a master multiseed summary table across preflight regimes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOT = ROOT / "analysis_out" / "preflight_runs"
DEFAULT_TABLE = ROOT / "paper" / "tables" / "master_multiseed_summary.md"
PRIORS = ("none", "graph", "permuted_graph", "temporal_smooth", "random_graph")
LEGACY_SUMMARIES = [
    SEARCH_ROOT / "spring_mass_lattice_ep5_temporal_calibrated" / "summary.csv",
    SEARCH_ROOT / "spring_mass_lattice_ep20" / "summary.csv",
    SEARCH_ROOT / "graph_wave_lattice_ep5_temporal_calibrated" / "summary.csv",
    SEARCH_ROOT / "graph_wave_lattice_ep20" / "summary.csv",
    SEARCH_ROOT / "metr_la_corr_T2000_train160" / "summary.csv",
    SEARCH_ROOT / "graph_heat_lattice" / "summary.csv",
]


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def stats(values: list[float]) -> tuple[float, float, float, float, int]:
    vals = [float(value) for value in values if finite(value)]
    if not vals:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    t_crit = 2.776 if len(vals) >= 5 else 4.303 if len(vals) == 3 else 12.706 if len(vals) == 2 else 0.0
    half = t_crit * std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return mean, std, mean - half, mean + half, len(vals)


def fmt(value: object) -> str:
    return "NA" if not finite(value) else f"{float(value):.4f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def discover_roots() -> list[Path]:
    roots = []
    if SEARCH_ROOT.exists():
        roots.extend(sorted(path for path in SEARCH_ROOT.glob("*_5seed*") if path.is_dir()))
        roots.extend(sorted(path for path in SEARCH_ROOT.glob("*_3seed*") if path.is_dir()))
    return roots


def infer_regime(root: Path) -> str:
    name = root.name
    for suffix in ("_m1", "_m2"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def infer_config(summary_path: Path, root: Path) -> str:
    parent = summary_path.parent
    try:
        rel_parent = parent.relative_to(root)
        return str(rel_parent)
    except ValueError:
        return parent.name


def rows_from_summary(path: Path, *, root: Path, record_status: str) -> list[dict[str, Any]]:
    regime = infer_regime(root)
    config = infer_config(path, root)
    out = []
    for row in read_csv(path):
        if row.get("stage") != "stage1_mini_train" or row.get("status") != "ok":
            continue
        prior = str(row.get("prior", ""))
        if prior not in PRIORS:
            continue
        out.append(
            {
                "regime": regime,
                "config": config,
                "prior": prior,
                "seed": int(row.get("seed", -1)),
                "h32": float(row.get("H32", "nan")),
                "record_status": record_status,
                "source": path,
            }
        )
    return out


def rows_from_ho_result(path: Path, *, root: Path, record_status: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for run in payload.get("runs", {}).values():
        if run.get("status") != "ok" or run.get("failure_flag"):
            continue
        prior = str(run.get("prior") or run.get("config", {}).get("prior", ""))
        if prior not in PRIORS:
            continue
        rows.append(
            {
                "regime": infer_regime(root),
                "config": "ho_lattice_audit",
                "prior": prior,
                "seed": int(run.get("seed", run.get("config", {}).get("seed", -1))),
                "h32": float(run.get("diagnostics", {}).get("rollout_errors", {}).get("32", float("nan"))),
                "record_status": record_status,
                "source": path,
            }
        )
    return rows


def strict_label(values: dict[str, tuple[float, float, float, float, int]]) -> tuple[str, str]:
    none = values.get("none", (float("nan"), 0, 0, 0, 0))
    graph = values.get("graph", (float("nan"), 0, 0, 0, 0))
    perm = values.get("permuted_graph", (float("nan"), 0, 0, 0, 0))
    temporal = values.get("temporal_smooth", (float("nan"), 0, 0, 0, 0))
    if not finite(graph[0]):
        return "inconclusive", "inconclusive"
    if finite(none[0]) and graph[0] >= none[0]:
        overlap = not (graph[2] > none[3] or none[2] > graph[3])
        return "no_graph_gain", "marginal" if overlap else "robust"
    if finite(perm[0]) and graph[0] >= perm[0]:
        overlap = not (graph[2] > perm[3] or perm[2] > graph[3])
        return "generic_smoothing", "marginal" if overlap else "robust"
    if finite(temporal[0]) and graph[0] >= temporal[0]:
        overlap = not (graph[2] > temporal[3] or temporal[2] > graph[3])
        return "temporal_sufficient", "marginal" if overlap else "robust"
    return "candidate_graph_favorable", "robust"


def build_master_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_values: dict[tuple[str, str, str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        key = (str(row["regime"]), str(row["config"]), str(row["prior"]), str(row["record_status"]))
        grouped_values[key][int(row["seed"])] = float(row["h32"])

    config_prior_stats: dict[tuple[str, str, str], tuple[float, float, float, float, int]] = {}
    for (regime, config, prior, status), by_seed in grouped_values.items():
        config_prior_stats[(regime, config, prior)] = stats(list(by_seed.values()))

    labels: dict[tuple[str, str], tuple[str, str]] = {}
    for regime, config, _prior in config_prior_stats:
        values = {
            prior: config_prior_stats.get((regime, config, prior), (float("nan"), float("nan"), float("nan"), float("nan"), 0))
            for prior in PRIORS
        }
        labels[(regime, config)] = strict_label(values)

    out = []
    for key in sorted(grouped_values):
        regime, config, prior, status = key
        mean, std, low, high, n = stats(list(grouped_values[key].values()))
        label, confidence = labels.get((regime, config), ("inconclusive", "inconclusive"))
        if status == "superseded":
            confidence = "superseded"
        out.append(
            {
                "regime": regime,
                "config": config,
                "prior": prior,
                "mean_h32": mean,
                "std_h32": std,
                "ci95_low": low,
                "ci95_high": high,
                "n_seeds": n,
                "protocol_label": label,
                "label_confidence": confidence,
                "record_status": status,
            }
        )
    return out


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Master Multiseed Summary",
        "",
        "| regime | config | prior | mean_h32 | std_h32 | ci95_low | ci95_high | n_seeds | protocol_label | label_confidence | record_status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["regime"]),
                    str(row["config"]),
                    str(row["prior"]),
                    fmt(row["mean_h32"]),
                    fmt(row["std_h32"]),
                    fmt(row["ci95_low"]),
                    fmt(row["ci95_high"]),
                    str(row["n_seeds"]),
                    str(row["protocol_label"]),
                    str(row["label_confidence"]),
                    str(row["record_status"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate all available multiseed preflight outputs.")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = discover_roots()
    rows: list[dict[str, Any]] = []
    discovered_files: list[Path] = []
    for root in roots:
        for summary in sorted(root.glob("**/summary.csv")):
            discovered_files.append(summary)
            rows.extend(rows_from_summary(summary, root=root, record_status="active"))
        ho_result = root / "cycle8_ho_audit_results.json"
        if ho_result.exists():
            discovered_files.append(ho_result)
            rows.extend(rows_from_ho_result(ho_result, root=root, record_status="active"))
    for legacy in LEGACY_SUMMARIES:
        discovered_files.append(legacy)
        legacy_root = legacy.parent
        rows.extend(rows_from_summary(legacy, root=legacy_root, record_status="superseded"))

    master_rows = build_master_rows(rows)
    if args.dry_run:
        print("Dry run only. No files will be written.")
        print("Discovered/legacy files:")
        for path in discovered_files:
            print(f"- {rel(path)} [{'exists' if path.exists() else 'missing'}]")
        print(f"Input stage1 rows: {len(rows)}")
        print(f"Master table rows: {len(master_rows)}")
        return
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(markdown(master_rows), encoding="utf-8")
    print(f"Wrote {rel(args.table)}")


if __name__ == "__main__":
    main()
