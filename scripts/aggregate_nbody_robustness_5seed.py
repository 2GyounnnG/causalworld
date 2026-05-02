"""Aggregate N-body 5-seed robustness outputs, optionally merging split hosts."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "analysis_out" / "preflight_runs" / "nbody_robustness_5seed"
DEFAULT_REPORT = ROOT / "analysis_out" / "NBODY_ROBUSTNESS_5SEED_REPORT.md"
DEFAULT_TABLE = ROOT / "paper" / "tables" / "nbody_robustness_5seed_summary.md"
DEFAULT_SUMMARY_CSV = ROOT / "analysis_out" / "nbody_robustness_5seed_summary.csv"
PRIORS = ("none", "graph", "permuted_graph", "temporal_smooth")
COMPARATOR_PRIORS = ("none", "permuted_graph", "temporal_smooth")
H32_METRICS = tuple(f"{prior}_h32" for prior in PRIORS)
GAIN_METRICS = (
    "graph_gain_h32_pct",
    "true_vs_permuted_gain_h32_pct",
    "graph_vs_temporal_gain_h32_pct",
)
ALL_METRICS = (*H32_METRICS, *GAIN_METRICS)
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


def to_float(value: object) -> float:
    return float(value) if finite(value) else float("nan")


def pct_gain(base: float, candidate: float) -> float:
    if not finite(base) or not finite(candidate) or float(base) == 0.0:
        return float("nan")
    return 100.0 * (float(base) - float(candidate)) / float(base)


def stats(values: list[float], *, t_critical: float) -> dict[str, float | int]:
    vals = [float(value) for value in values if finite(value)]
    if not vals:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_half_width": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
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


def fmt_value(value: float, digits: int = 4) -> str:
    if not finite(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_mean_std(metric_stats: dict[str, float | int]) -> str:
    mean_value = metric_stats.get("mean", float("nan"))
    std_value = metric_stats.get("std", float("nan"))
    if not finite(mean_value):
        return "NA"
    return f"{fmt_value(float(mean_value))} +/- {fmt_value(float(std_value))}"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def run_dir_for(out_root: Path, distance_k: int, budget: str) -> Path:
    return out_root / f"distance_k_{distance_k:02d}" / budget


def summary_path_for(out_root: Path, distance_k: int, budget: str) -> Path:
    return run_dir_for(out_root, distance_k, budget) / "summary.csv"


def merged_rows_for_config(
    *,
    roots: list[Path],
    distance_k: int,
    budget: str,
) -> tuple[list[dict[str, str]], list[Path], int]:
    keyed: dict[tuple[str, str, str], dict[str, str]] = {}
    extras: list[dict[str, str]] = []
    duplicate_count = 0
    source_paths: list[Path] = []
    for root in roots:
        summary_path = summary_path_for(root, distance_k, budget)
        source_paths.append(summary_path)
        for row in read_rows(summary_path):
            if row.get("stage") == "stage1_mini_train":
                key = (
                    str(row.get("seed", "")).strip(),
                    str(row.get("prior", "")).strip(),
                    str(row.get("stage", "")).strip(),
                )
                if key in keyed:
                    duplicate_count += 1
                tagged = dict(row)
                tagged["_source_summary"] = str(summary_path)
                keyed[key] = tagged
            else:
                tagged = dict(row)
                tagged["_source_summary"] = str(summary_path)
                extras.append(tagged)
    return [*extras, *keyed.values()], source_paths, duplicate_count


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def strict_label(metrics: dict[str, float], *, budget: str) -> str:
    none_h32 = metrics.get("none_h32", float("nan"))
    graph_h32 = metrics.get("graph_h32", float("nan"))
    permuted_h32 = metrics.get("permuted_graph_h32", float("nan"))
    temporal_h32 = metrics.get("temporal_smooth_h32", float("nan"))

    if not all(finite(value) for value in (none_h32, graph_h32, permuted_h32, temporal_h32)):
        return "inconclusive"
    if graph_h32 >= none_h32:
        return "no_graph_gain"
    if graph_h32 >= permuted_h32:
        return "generic_smoothing"
    if graph_h32 >= temporal_h32:
        return "temporal_sufficient"
    if budget.startswith("quick"):
        return "quick_topology_signal"
    return "candidate_graph_favorable"


def ci_overlap(left: dict[str, float | int], right: dict[str, float | int]) -> bool:
    left_low = left.get("ci95_low", float("nan"))
    left_high = left.get("ci95_high", float("nan"))
    right_low = right.get("ci95_low", float("nan"))
    right_high = right.get("ci95_high", float("nan"))
    if not all(finite(value) for value in (left_low, left_high, right_low, right_high)):
        return True
    return not (float(left_high) < float(right_low) or float(right_high) < float(left_low))


def seed_metric_rows(rows: list[dict[str, str]], seeds: list[int], *, budget: str) -> list[dict[str, float | int | str]]:
    by_seed_prior: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        if row.get("stage") != "stage1_mini_train" or row.get("status") != "ok":
            continue
        try:
            seed = int(str(row.get("seed", "")).strip())
        except ValueError:
            continue
        prior = str(row.get("prior", "")).strip()
        if prior in PRIORS:
            by_seed_prior[(seed, prior)] = row

    seed_rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        metrics: dict[str, float | int | str] = {"seed": seed}
        none_h32 = to_float(by_seed_prior.get((seed, "none"), {}).get("H32"))
        graph_h32 = to_float(by_seed_prior.get((seed, "graph"), {}).get("H32"))
        permuted_h32 = to_float(by_seed_prior.get((seed, "permuted_graph"), {}).get("H32"))
        temporal_h32 = to_float(by_seed_prior.get((seed, "temporal_smooth"), {}).get("H32"))
        metrics.update(
            {
                "none_h32": none_h32,
                "graph_h32": graph_h32,
                "permuted_graph_h32": permuted_h32,
                "temporal_smooth_h32": temporal_h32,
                "graph_gain_h32_pct": pct_gain(none_h32, graph_h32),
                "true_vs_permuted_gain_h32_pct": pct_gain(permuted_h32, graph_h32),
                "graph_vs_temporal_gain_h32_pct": pct_gain(temporal_h32, graph_h32),
            }
        )
        metrics["strict_label"] = strict_label(metrics, budget=budget)  # type: ignore[arg-type]
        seed_rows.append(metrics)
    return seed_rows


def failed_stage1_count(rows: list[dict[str, str]], seeds: list[int]) -> int:
    expected_seeds = set(seeds)
    count = 0
    for row in rows:
        if row.get("stage") != "stage1_mini_train" or row.get("status") != "failed":
            continue
        try:
            seed = int(str(row.get("seed", "")).strip())
        except ValueError:
            continue
        if seed in expected_seeds and str(row.get("prior", "")).strip() in PRIORS:
            count += 1
    return count


def strongest_comparator(metric_stats: dict[str, dict[str, float | int]]) -> tuple[str, dict[str, float | int]]:
    candidates = []
    for prior in COMPARATOR_PRIORS:
        metric = f"{prior}_h32"
        mean_value = metric_stats[metric]["mean"]
        if finite(mean_value):
            candidates.append((float(mean_value), prior, metric_stats[metric]))
    if not candidates:
        return "none", stats([], t_critical=T_CRITICAL_95_N5)
    _mean_value, prior, prior_stats = min(candidates, key=lambda item: item[0])
    return prior, prior_stats


def summarize_config(
    *,
    out_root: Path,
    distance_k: int,
    budget_label: str,
    epochs: int,
    seeds: list[int],
    t_critical: float,
    merge_roots: list[Path] | None = None,
) -> dict[str, object]:
    budget = f"{budget_label}_ep{epochs}"
    run_dir = run_dir_for(out_root, distance_k, budget)
    summary_path = summary_path_for(out_root, distance_k, budget)
    duplicate_count = 0
    source_paths = [summary_path]
    if merge_roots:
        rows, source_paths, duplicate_count = merged_rows_for_config(
            roots=merge_roots,
            distance_k=distance_k,
            budget=budget,
        )
        summary_exists = any(path.exists() for path in source_paths)
    else:
        rows = read_rows(summary_path)
        summary_exists = summary_path.exists()
    seed_rows = seed_metric_rows(rows, seeds, budget=budget)
    metric_stats = {
        metric: stats([float(row[metric]) for row in seed_rows if metric in row], t_critical=t_critical)
        for metric in ALL_METRICS
    }
    mean_h32 = {metric: float(metric_stats[metric]["mean"]) for metric in H32_METRICS}
    label = strict_label(mean_h32, budget=budget)
    comparator_name, comparator_stats = strongest_comparator(metric_stats)

    expected_n = len(seeds)
    n_lt_expected = [metric for metric in H32_METRICS if int(metric_stats[metric]["n"]) < expected_n]
    has_missing_mean = any(not finite(metric_stats[metric]["mean"]) for metric in H32_METRICS)
    warning_reasons = []
    if not summary_exists:
        warning_reasons.append("missing_summary")
    if duplicate_count:
        warning_reasons.append(f"duplicate_seed_prior_rows_last_wins:{duplicate_count}")
    if n_lt_expected:
        warning_reasons.append("n_lt_expected:" + ",".join(n_lt_expected))
    failure_count = failed_stage1_count(rows, seeds)
    if failure_count:
        warning_reasons.append(f"failed_stage1_runs:{failure_count}")

    if has_missing_mean:
        label_confidence = "inconclusive"
    elif n_lt_expected:
        label_confidence = "inconclusive"
    elif ci_overlap(metric_stats["graph_h32"], comparator_stats):
        label_confidence = "marginal"
    else:
        label_confidence = "robust"

    counts = Counter(str(row["strict_label"]) for row in seed_rows)
    return {
        "distance_k": distance_k,
        "budget": budget,
        "epochs": epochs,
        "run_dir": run_dir,
        "summary_path": summary_path,
        "source_paths": source_paths,
        "duplicate_count": duplicate_count,
        "summary_exists": summary_exists,
        "complete": summary_exists and not n_lt_expected and not has_missing_mean,
        "warning_flag": "warning" if warning_reasons else "ok",
        "warning_reasons": ";".join(warning_reasons),
        "failed_stage1_count": failure_count,
        "expected_seed_count": expected_n,
        "mean_strict_label": label,
        "label_confidence": label_confidence,
        "strongest_comparator": comparator_name,
        "seed_rows": seed_rows,
        "label_counts": counts,
        "metric_stats": metric_stats,
    }


def label_counts_text(counts: Counter[str], total: int) -> str:
    if not counts:
        return "inconclusive 0/0"
    return ", ".join(f"{label} {count}/{total}" for label, count in counts.most_common())


def csv_row(summary: dict[str, object], seeds: list[int]) -> dict[str, object]:
    metric_stats: dict[str, dict[str, float | int]] = summary["metric_stats"]  # type: ignore[assignment]
    row: dict[str, object] = {
        "distance_k": summary["distance_k"],
        "budget": summary["budget"],
        "epochs": summary["epochs"],
        "complete": summary["complete"],
        "summary_exists": summary["summary_exists"],
        "warning_flag": summary["warning_flag"],
        "warning_reasons": summary["warning_reasons"],
        "failed_stage1_count": summary["failed_stage1_count"],
        "expected_seed_count": summary["expected_seed_count"],
        "mean_strict_label": summary["mean_strict_label"],
        "label_confidence": summary["label_confidence"],
        "strongest_comparator": summary["strongest_comparator"],
        "seed_label_counts": label_counts_text(summary["label_counts"], len(seeds)),  # type: ignore[arg-type]
        "summary_path": summary["summary_path"],
    }
    for metric in ALL_METRICS:
        for suffix in ("mean", "std", "ci95_half_width", "ci95_low", "ci95_high", "min", "max", "n"):
            row[f"{metric}_{suffix}"] = metric_stats[metric][suffix]
    return row


def write_csv(path: Path, summaries: list[dict[str, object]], seeds: list[int]) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [csv_row(summary, seeds) for summary in summaries]
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return fieldnames


def metric_table(summaries: list[dict[str, object]], *, metrics: tuple[str, ...]) -> list[str]:
    header = ["k", "budget", "label", "label_confidence", *metrics]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for summary in summaries:
        metric_stats: dict[str, dict[str, float | int]] = summary["metric_stats"]  # type: ignore[assignment]
        cells = [
            str(summary["distance_k"]),
            str(summary["budget"]),
            str(summary["mean_strict_label"]),
            str(summary["label_confidence"]),
        ]
        for metric in metrics:
            cells.append(fmt_mean_std(metric_stats[metric]))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def paper_lift_table(summaries: list[dict[str, object]]) -> list[str]:
    lines = [
        "| k | budget | label | label_confidence | graph H32 (mean +/- std) | strongest_comparator | comparator H32 (mean +/- std) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        metric_stats: dict[str, dict[str, float | int]] = summary["metric_stats"]  # type: ignore[assignment]
        comparator = str(summary["strongest_comparator"])
        comparator_metric = f"{comparator}_h32"
        comparator_cell = fmt_mean_std(metric_stats[comparator_metric]) if comparator_metric in metric_stats else "NA"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(summary["distance_k"]),
                    str(summary["budget"]),
                    str(summary["mean_strict_label"]),
                    str(summary["label_confidence"]),
                    fmt_mean_std(metric_stats["graph_h32"]),
                    comparator,
                    comparator_cell,
                ]
            )
            + " |"
        )
    return lines


def seed_label_table(summaries: list[dict[str, object]], total: int) -> list[str]:
    lines = [
        "| k | budget | seed-level label counts | warning_flag | warning_reasons |",
        "| --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(summary["distance_k"]),
                    str(summary["budget"]),
                    label_counts_text(summary["label_counts"], total),  # type: ignore[arg-type]
                    str(summary["warning_flag"]),
                    str(summary["warning_reasons"]),
                ]
            )
            + " |"
        )
    return lines


def build_table_md(summaries: list[dict[str, object]], seeds: list[int]) -> str:
    seed_text = ",".join(str(seed) for seed in seeds)
    lines = [
        "# N-body Robustness 5-seed Summary",
        "",
        f"Mean +/- sample std across seeds {seed_text}. 95% CIs use t=2.776 by default.",
        "",
        "## Paper-lift Summary",
        "",
        *paper_lift_table(summaries),
        "",
        "## H32 Rollout Error",
        "",
        *metric_table(summaries, metrics=H32_METRICS),
        "",
        "## H32 Gains (%)",
        "",
        *metric_table(summaries, metrics=GAIN_METRICS),
        "",
        "## Seed Labels and Warnings",
        "",
        *seed_label_table(summaries, len(seeds)),
        "",
    ]
    return "\n".join(lines)


def build_report_md(summaries: list[dict[str, object]], seeds: list[int], out_root: Path, summary_csv: Path, table_path: Path) -> str:
    missing = [summary for summary in summaries if not bool(summary["complete"])]
    lines = [
        "# N-body Robustness 5-seed Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Paper-lift Summary",
        "",
        *paper_lift_table(summaries),
        "",
        "## Scope",
        "",
        f"- Source root: `{rel(out_root)}`",
        f"- Seeds: `{','.join(str(seed) for seed in seeds)}`",
        "- Dataset/settings: `nbody_distance`, `nbody_n=36`, `train_transitions=96`, `eval_transitions=32`, `raw_transitions=64`, strides `5/10/5`, horizons `16 32`, `prior_weight=0.1`, calibrated graph reference, temporal prior included, random graph skipped.",
        "",
        "## Completion",
        "",
    ]
    if missing:
        lines.append(f"- Incomplete or warning configurations: {len(missing)}/{len(summaries)}. See `warning_flag`, `warning_reasons`, and `label_confidence`.")
    else:
        lines.append("- All six configurations have complete seed-level H32 data.")
    lines.extend(
        [
            "",
            "## H32 Rollout Error",
            "",
            *metric_table(summaries, metrics=H32_METRICS),
            "",
            "## H32 Gains (%)",
            "",
            *metric_table(summaries, metrics=GAIN_METRICS),
            "",
            "## Seed-level Strict Label Counts",
            "",
            *seed_label_table(summaries, len(seeds)),
            "",
            "## Files",
            "",
            f"- Summary CSV: `{rel(summary_csv)}`",
            f"- Paper table: `{rel(table_path)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the N-body 5-seed robustness sweep.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--merge-dirs", type=parse_csv_paths, default=None, help="Comma-separated nbody_robustness_5seed* roots to merge; last duplicate seed/prior wins.")
    parser.add_argument("--distance-k", type=parse_csv_ints, default=[4, 8, 12])
    parser.add_argument("--seeds", type=parse_csv_ints, default=[0, 1, 2, 3, 4])
    parser.add_argument("--quick-epochs", type=int, default=5)
    parser.add_argument("--standard-epochs", type=int, default=20)
    parser.add_argument("--t-critical", type=float, default=T_CRITICAL_95_N5)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--dry-run", action="store_true", help="Print merge/read plan and unified row counts without writing.")
    return parser.parse_args()


def build_summaries(args: argparse.Namespace) -> list[dict[str, object]]:
    budgets = [("quick", int(args.quick_epochs)), ("standard", int(args.standard_epochs))]
    merge_roots = [path.resolve() for path in args.merge_dirs] if args.merge_dirs else None
    return [
        summarize_config(
            out_root=args.out_root,
            distance_k=distance_k,
            budget_label=budget_label,
            epochs=epochs,
            seeds=args.seeds,
            t_critical=float(args.t_critical),
            merge_roots=merge_roots,
        )
        for distance_k in args.distance_k
        for budget_label, epochs in budgets
    ]


def main() -> None:
    args = parse_args()
    args.out_root = args.out_root.resolve()
    if args.merge_dirs:
        args.merge_dirs = [path.resolve() for path in args.merge_dirs]
    args.report = args.report.resolve()
    args.table = args.table.resolve()
    args.summary_csv = args.summary_csv.resolve()

    summaries = build_summaries(args)
    if args.dry_run:
        print("Dry run only. No files will be written.")
        if args.merge_dirs:
            print("Merge roots:")
            for root in args.merge_dirs:
                print(f"- {rel(root)}")
        else:
            print(f"Single root: {rel(args.out_root)}")
        total_rows = 0
        total_duplicates = 0
        for summary in summaries:
            metric_stats: dict[str, dict[str, float | int]] = summary["metric_stats"]  # type: ignore[assignment]
            config_rows = int(metric_stats["graph_h32"]["n"]) + int(metric_stats["none_h32"]["n"]) + int(metric_stats["permuted_graph_h32"]["n"]) + int(metric_stats["temporal_smooth_h32"]["n"])
            total_rows += config_rows
            total_duplicates += int(summary["duplicate_count"])
            print(f"Config k={summary['distance_k']} budget={summary['budget']}:")
            for path in summary["source_paths"]:  # type: ignore[index]
                status = "exists" if Path(path).exists() else "missing"
                print(f"  WOULD READ {rel(Path(path))} [{status}]")
            print(f"  unified seed/prior H32 row count: {config_rows}")
            if int(summary["duplicate_count"]):
                print(f"  duplicate rows last-wins: {summary['duplicate_count']}")
        print(f"Unified seed/prior H32 row count across configs: {total_rows}")
        print(f"Duplicate seed/prior rows across configs: {total_duplicates}")
        return

    fieldnames = write_csv(args.summary_csv, summaries, args.seeds)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(build_table_md(summaries, args.seeds), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        build_report_md(summaries, args.seeds, args.out_root, args.summary_csv, args.table),
        encoding="utf-8",
    )
    print(f"Wrote {rel(args.summary_csv)}")
    print(f"Wrote {rel(args.table)}")
    print(f"Wrote {rel(args.report)}")
    print("Summary CSV columns:")
    print(",".join(fieldnames))


if __name__ == "__main__":
    main()
