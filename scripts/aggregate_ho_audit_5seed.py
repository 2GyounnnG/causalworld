"""Aggregate split-host HO lattice latent audit outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "analysis_out" / "preflight_runs" / "ho_audit_5seed"
DEFAULT_REPORT = ROOT / "analysis_out" / "HO_AUDIT_5SEED_REPORT.md"
DEFAULT_TABLE = ROOT / "paper" / "tables" / "ho_audit_5seed_summary.md"
DEFAULT_SUMMARY_CSV = ROOT / "analysis_out" / "ho_audit_5seed_summary.csv"
PRIORS = ("graph", "permuted_graph", "random_graph")
METRICS = ("H32_rollout", "D_true_Delta_H_norm", "R_low_true_Delta_H_4")
T_CRITICAL_95_N5 = 2.776


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_paths(raw: str) -> list[Path]:
    return [Path(part.strip()) for part in raw.split(",") if part.strip()]


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def stats(values: list[float], *, t_critical: float) -> dict[str, float | int]:
    vals = [float(value) for value in values if finite(value)]
    if not vals:
        return {key: float("nan") for key in ("mean", "std", "ci95_half_width", "ci95_low", "ci95_high", "min", "max")} | {"n": 0}
    mean_value = statistics.mean(vals)
    std_value = statistics.stdev(vals) if len(vals) > 1 else 0.0
    ci_half = t_critical * std_value / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return {
        "mean": mean_value,
        "std": std_value,
        "ci95_half_width": ci_half,
        "ci95_low": mean_value - ci_half,
        "ci95_high": mean_value + ci_half,
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def fmt(value: object, digits: int = 4) -> str:
    return "NA" if not finite(value) else f"{float(value):.{digits}f}"


def fmt_mean_std(stat: dict[str, float | int]) -> str:
    if not finite(stat.get("mean")):
        return "NA"
    return f"{fmt(stat['mean'])} +/- {fmt(stat['std'])}"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def result_path(root: Path) -> Path:
    return root / "cycle8_ho_audit_results.json"


def load_artifact_metrics(path: Path) -> dict[str, float]:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.analyze_cycle8_latent_alignment import artifact_metrics, safe_torch_load

    return artifact_metrics(safe_torch_load(path))


def load_result_rows(roots: list[Path]) -> tuple[list[dict[str, Any]], int, list[Path]]:
    keyed: dict[tuple[int, str], dict[str, Any]] = {}
    duplicates = 0
    paths = [result_path(root) for root in roots]
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for run in payload.get("runs", {}).values():
            config = run.get("config", {})
            prior = str(config.get("prior") or run.get("prior"))
            if prior not in PRIORS:
                continue
            seed = int(config.get("seed", run.get("seed", -1)))
            key = (seed, prior)
            if key in keyed:
                duplicates += 1
            row = {
                "run_name": run.get("run_name"),
                "prior": prior,
                "seed": seed,
                "status": run.get("status"),
                "failure_flag": bool(run.get("failure_flag")),
                "source_result": str(path),
                "H32_rollout": float(run.get("diagnostics", {}).get("rollout_errors", {}).get("32", float("nan"))),
            }
            artifact_path = ROOT / str(run.get("latent_trace_path", ""))
            if run.get("status") == "ok" and artifact_path.exists():
                row.update(load_artifact_metrics(artifact_path))
            keyed[key] = row
    return list(keyed.values()), duplicates, paths


def summarize(rows: list[dict[str, Any]], seeds: list[int], *, t_critical: float) -> list[dict[str, Any]]:
    out = []
    for prior in PRIORS:
        prior_rows = [row for row in rows if row.get("prior") == prior and int(row.get("seed", -1)) in set(seeds)]
        summary = {"prior": prior, "expected_seed_count": len(seeds)}
        for metric in METRICS:
            metric_stats = stats([float(row.get(metric, float("nan"))) for row in prior_rows], t_critical=t_critical)
            for suffix, value in metric_stats.items():
                summary[f"{metric}_{suffix}"] = value
        summary["warning_flag"] = "ok" if all(int(summary[f"{metric}_n"]) == len(seeds) for metric in METRICS) else "warning"
        out.append(summary)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return fieldnames


def table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| prior | warning | H32 rollout | Dtrue_norm(Delta_H) | R_low K=4 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["prior"]),
                    str(row["warning_flag"]),
                    fmt_mean_std({"mean": row["H32_rollout_mean"], "std": row["H32_rollout_std"]}),
                    fmt_mean_std({"mean": row["D_true_Delta_H_norm_mean"], "std": row["D_true_Delta_H_norm_std"]}),
                    fmt_mean_std({"mean": row["R_low_true_Delta_H_4_mean"], "std": row["R_low_true_Delta_H_4_std"]}),
                ]
            )
            + " |"
        )
    return lines


def build_markdown(rows: list[dict[str, Any]], seeds: list[int], title: str) -> str:
    return "\n".join(
        [
            f"# {title}",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            f"Seeds: `{','.join(str(seed) for seed in seeds)}`. Mean +/- sample std; CI95 columns are in the CSV.",
            "",
            *table(rows),
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate HO lattice 5-seed latent audit outputs.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--merge-dirs", type=parse_csv_paths, default=None)
    parser.add_argument("--seeds", type=parse_csv_ints, default=[0, 1, 2, 3, 4])
    parser.add_argument("--t-critical", type=float, default=T_CRITICAL_95_N5)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [path.resolve() for path in args.merge_dirs] if args.merge_dirs else [args.out_root.resolve()]
    rows, duplicates, paths = load_result_rows(roots)
    summary_rows = summarize(rows, args.seeds, t_critical=float(args.t_critical))
    if args.dry_run:
        print("Dry run only. No files will be written.")
        for path in paths:
            print(f"WOULD READ {rel(path)} [{'exists' if path.exists() else 'missing'}]")
        print(f"Unified run row count: {len(rows)}")
        print(f"Duplicate seed/prior rows last-wins: {duplicates}")
        return
    fieldnames = write_csv(args.summary_csv.resolve(), summary_rows)
    args.table.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.table.resolve().write_text(build_markdown(summary_rows, args.seeds, "HO Audit 5-seed Summary"), encoding="utf-8")
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(build_markdown(summary_rows, args.seeds, "HO Audit 5-seed Report"), encoding="utf-8")
    print(f"Wrote {rel(args.summary_csv.resolve())}")
    print(f"Wrote {rel(args.table.resolve())}")
    print(f"Wrote {rel(args.report.resolve())}")
    print("Summary CSV columns:")
    print(",".join(fieldnames))


if __name__ == "__main__":
    main()
