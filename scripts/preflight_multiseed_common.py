"""Shared launch and aggregation helpers for multiseed preflight regimes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_SCRIPT = ROOT / "scripts" / "graph_prior_preflight_check.py"
EXPECTED_PRIORS = ("none", "graph", "permuted_graph", "temporal_smooth")
H32_METRICS = tuple(f"{prior}_h32" for prior in EXPECTED_PRIORS)
GAIN_METRICS = (
    "graph_gain_h32_pct",
    "true_vs_permuted_gain_h32_pct",
    "graph_vs_temporal_gain_h32_pct",
)
ALL_METRICS = (*H32_METRICS, *GAIN_METRICS)
T_CRITICAL_95_N5 = 2.776
T_CRITICAL_95_N3 = 4.303


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_paths(raw: str) -> list[Path]:
    return [Path(part.strip()) for part in raw.split(",") if part.strip()]


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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


def ci_overlap(left: dict[str, float | int], right: dict[str, float | int]) -> bool:
    values = [left.get("ci95_low"), left.get("ci95_high"), right.get("ci95_low"), right.get("ci95_high")]
    if not all(finite(value) for value in values):
        return True
    return not (float(left["ci95_high"]) < float(right["ci95_low"]) or float(right["ci95_high"]) < float(left["ci95_low"]))


def fmt(value: object, digits: int = 4) -> str:
    return "NA" if not finite(value) else f"{float(value):.{digits}f}"


def fmt_mean_std(metric_stats: dict[str, float | int]) -> str:
    if not finite(metric_stats.get("mean")):
        return "NA"
    return f"{fmt(metric_stats['mean'])} +/- {fmt(metric_stats['std'])}"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def shell_quote(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def interpreter_command(args: argparse.Namespace) -> tuple[list[str], str]:
    if args.conda_env:
        return ["conda", "run", "-n", args.conda_env, "python"], f"conda run -n {args.conda_env}"
    return [args.python_exe], args.python_exe


def display_command(args: argparse.Namespace, command: list[str]) -> str:
    prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(str(args.gpu_id))} " if args.gpu_id is not None else ""
    return prefix + shell_quote(command)


def child_env(args: argparse.Namespace) -> dict[str, str] | None:
    if args.gpu_id is None:
        return None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    return env


def summary_appears_complete(summary_path: Path, seeds: list[int]) -> tuple[bool, str]:
    if not summary_path.exists():
        return False, "missing summary.csv"
    with summary_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    found: set[tuple[int, str]] = set()
    for row in rows:
        if row.get("stage") == "stage1_mini_train" and row.get("status") == "ok":
            try:
                seed = int(str(row.get("seed", "")).strip())
            except ValueError:
                continue
            prior = str(row.get("prior", "")).strip()
            if seed in seeds and prior in EXPECTED_PRIORS and finite(row.get("H32")):
                found.add((seed, prior))
    missing = [
        f"seed{seed}:{prior}"
        for seed in seeds
        for prior in EXPECTED_PRIORS
        if (seed, prior) not in found
    ]
    if missing:
        return False, "missing stage1 H32 rows for " + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
    return True, "complete"


def add_runner_args(parser: argparse.ArgumentParser, *, default_out_root: Path, default_seeds: list[int]) -> None:
    parser.add_argument("--out-root", type=Path, default=default_out_root)
    parser.add_argument("--seeds", type=parse_csv_ints, default=default_seeds, help="Comma-separated seed subset.")
    parser.add_argument("--output-suffix", default="", help="Append suffix to output root directory name, e.g. _m1.")
    parser.add_argument("--gpu-id", default=None, help="Set CUDA_VISIBLE_DEVICES for child preflight commands.")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")


def command_for_config(args: argparse.Namespace, config: dict[str, Any], out_dir: Path) -> list[str]:
    python_cmd, _source = interpreter_command(args)
    command = [
        *python_cmd,
        str(PREFLIGHT_SCRIPT),
        "--dataset",
        str(config["dataset"]),
        "--out-dir",
        str(out_dir),
        "--epochs",
        str(config["epochs"]),
        "--seeds",
        ",".join(str(seed) for seed in args.seeds),
        "--train-transitions",
        str(config["train_transitions"]),
        "--eval-transitions",
        str(config["eval_transitions"]),
        "--raw-transitions",
        str(config["raw_transitions"]),
        "--train-stride",
        str(config["train_stride"]),
        "--eval-stride",
        str(config["eval_stride"]),
        "--raw-stride",
        str(config["raw_stride"]),
        "--horizons",
        "16",
        "32",
        "--prior-weight",
        str(config.get("prior_weight", 0.1)),
        "--batch-size",
        str(config.get("batch_size", 32)),
        "--device",
        str(config.get("device", "auto")),
        "--include-temporal-prior",
        "--calibrate-prior-strength",
        "--calibration-reference-prior",
        "graph",
        "--calibration-target-ratio",
        "1.0",
    ]
    for key, value in config.get("extra_args", []):
        command.extend([str(key), str(value)])
    for flag in config.get("extra_flags", []):
        command.append(str(flag))
    return command


def run_regime_launcher(
    *,
    regime_name: str,
    default_out_root: Path,
    configs: list[dict[str, Any]],
    aggregate_script: Path,
    default_seeds: list[int],
    argv: list[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=f"Run multiseed preflight package for {regime_name}.")
    add_runner_args(parser, default_out_root=default_out_root, default_seeds=default_seeds)
    args = parser.parse_args(argv)
    if args.output_suffix:
        args.out_root = args.out_root.parent / f"{args.out_root.name}{args.output_suffix}"
    args.out_root = args.out_root.resolve()
    _python_cmd, interpreter_source = interpreter_command(args)
    print(f"Interpreter mode: {interpreter_source}")

    manifest_rows = []
    commands: list[list[str]] = []
    for config in configs:
        out_dir = args.out_root / str(config["name"])
        complete, reason = summary_appears_complete(out_dir / "summary.csv", args.seeds)
        should_run = args.force or not complete
        command = command_for_config(args, config, out_dir)
        manifest_rows.append(
            {
                "name": config["name"],
                "out_dir": str(out_dir),
                "complete": complete,
                "completion_reason": reason,
                "will_run": should_run,
                "command": command,
            }
        )
        if should_run:
            commands.append(command)
        else:
            print(f"SKIP complete {out_dir}: {reason}")

    aggregate_command = [
        *interpreter_command(args)[0],
        str(aggregate_script),
        "--out-root",
        str(args.out_root),
        "--seeds",
        ",".join(str(seed) for seed in args.seeds),
    ]
    if args.dry_run:
        for command in commands:
            print(f"DRY RUN command: {display_command(args, command)}")
        if not args.skip_aggregate:
            print(f"DRY RUN summary command: {shell_quote(aggregate_command)}")
        print(f"Dry run only. Pending run command(s): {len(commands)}")
        return

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "regime": regime_name,
                "seeds": args.seeds,
                "runs": manifest_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    failures = []
    env = child_env(args)
    for command in commands:
        print(f"RUN command: {display_command(args, command)}", flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            failures.append({"returncode": exc.returncode, "command": command})
            print(f"RUN FAILED returncode={exc.returncode}: {display_command(args, command)}", flush=True)
            if args.fail_fast:
                raise
    if not args.skip_aggregate:
        print(f"SUMMARY command: {shell_quote(aggregate_command)}", flush=True)
        subprocess.run(aggregate_command, cwd=ROOT, check=True)
    if failures:
        (args.out_root / "command_failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")


def read_summary_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def merged_rows_for_config(roots: list[Path], config_name: str) -> tuple[list[dict[str, str]], list[Path], int]:
    keyed: dict[tuple[str, str, str], dict[str, str]] = {}
    extras: list[dict[str, str]] = []
    duplicates = 0
    paths = [root / config_name / "summary.csv" for root in roots]
    for path in paths:
        for row in read_summary_rows(path):
            tagged = dict(row)
            tagged["_source_summary"] = str(path)
            if row.get("stage") == "stage1_mini_train":
                key = (str(row.get("seed", "")), str(row.get("prior", "")), str(row.get("stage", "")))
                if key in keyed:
                    duplicates += 1
                keyed[key] = tagged
            else:
                extras.append(tagged)
    return [*extras, *keyed.values()], paths, duplicates


def seed_metric_rows(rows: list[dict[str, str]], seeds: list[int], config_name: str) -> list[dict[str, float | int | str]]:
    by_seed_prior: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        if row.get("stage") != "stage1_mini_train" or row.get("status") != "ok":
            continue
        try:
            seed = int(str(row.get("seed", "")).strip())
        except ValueError:
            continue
        prior = str(row.get("prior", "")).strip()
        if prior in EXPECTED_PRIORS:
            by_seed_prior[(seed, prior)] = row
    out = []
    for seed in seeds:
        none = float(by_seed_prior.get((seed, "none"), {}).get("H32", "nan"))
        graph = float(by_seed_prior.get((seed, "graph"), {}).get("H32", "nan"))
        permuted = float(by_seed_prior.get((seed, "permuted_graph"), {}).get("H32", "nan"))
        temporal = float(by_seed_prior.get((seed, "temporal_smooth"), {}).get("H32", "nan"))
        out.append(
            {
                "config": config_name,
                "seed": seed,
                "none_h32": none,
                "graph_h32": graph,
                "permuted_graph_h32": permuted,
                "temporal_smooth_h32": temporal,
                "graph_gain_h32_pct": pct_gain(none, graph),
                "true_vs_permuted_gain_h32_pct": pct_gain(permuted, graph),
                "graph_vs_temporal_gain_h32_pct": pct_gain(temporal, graph),
            }
        )
    return out


def strict_label(metric_stats: dict[str, dict[str, float | int]], config_name: str) -> tuple[str, str, str]:
    none = float(metric_stats["none_h32"]["mean"])
    graph = float(metric_stats["graph_h32"]["mean"])
    permuted = float(metric_stats["permuted_graph_h32"]["mean"])
    temporal = float(metric_stats["temporal_smooth_h32"]["mean"])
    if not all(finite(value) for value in (none, graph, permuted, temporal)):
        return "inconclusive", "inconclusive", "none"
    comparators = {
        "none": metric_stats["none_h32"],
        "permuted_graph": metric_stats["permuted_graph_h32"],
        "temporal_smooth": metric_stats["temporal_smooth_h32"],
    }
    strongest_name, strongest_stats = min(comparators.items(), key=lambda item: float(item[1]["mean"]))
    if graph >= none:
        label = "no_graph_gain"
        comparator = "none"
    elif graph >= permuted:
        label = "generic_smoothing"
        comparator = "permuted_graph"
    elif graph >= temporal:
        label = "temporal_sufficient"
        comparator = "temporal_smooth"
    elif "quick" in config_name:
        label = "quick_topology_signal"
        comparator = strongest_name
    else:
        label = "candidate_graph_favorable"
        comparator = strongest_name
    comparator_stats = metric_stats[f"{comparator}_h32"]
    confidence = "marginal" if ci_overlap(metric_stats["graph_h32"], comparator_stats) else "robust"
    return label, confidence, comparator


def summarize_config(
    *,
    roots: list[Path],
    config_name: str,
    seeds: list[int],
    t_critical: float,
) -> dict[str, Any]:
    rows, source_paths, duplicates = merged_rows_for_config(roots, config_name)
    seed_rows = seed_metric_rows(rows, seeds, config_name)
    metric_stats = {
        metric: stats([float(row[metric]) for row in seed_rows if metric in row], t_critical=t_critical)
        for metric in ALL_METRICS
    }
    label, confidence, comparator = strict_label(metric_stats, config_name)
    warning_reasons = []
    if not any(path.exists() for path in source_paths):
        warning_reasons.append("missing_summary")
    missing_metrics = [metric for metric in H32_METRICS if int(metric_stats[metric]["n"]) < len(seeds)]
    if missing_metrics:
        warning_reasons.append("n_lt_expected:" + ",".join(missing_metrics))
        confidence = "inconclusive"
    if duplicates:
        warning_reasons.append(f"duplicate_seed_prior_rows_last_wins:{duplicates}")
    labels = Counter()
    for row in seed_rows:
        labels[strict_label({metric: stats([float(row[metric])], t_critical=t_critical) for metric in H32_METRICS}, config_name)[0]] += 1
    return {
        "config": config_name,
        "source_paths": source_paths,
        "duplicate_count": duplicates,
        "warning_flag": "warning" if warning_reasons else "ok",
        "warning_reasons": ";".join(warning_reasons),
        "protocol_label": label,
        "label_confidence": confidence,
        "strongest_comparator": comparator,
        "seed_label_counts": ", ".join(f"{key} {value}/{len(seeds)}" for key, value in labels.most_common()),
        "metric_stats": metric_stats,
    }


def csv_row(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "config": summary["config"],
        "warning_flag": summary["warning_flag"],
        "warning_reasons": summary["warning_reasons"],
        "protocol_label": summary["protocol_label"],
        "label_confidence": summary["label_confidence"],
        "strongest_comparator": summary["strongest_comparator"],
        "seed_label_counts": summary["seed_label_counts"],
        "duplicate_count": summary["duplicate_count"],
    }
    for metric in ALL_METRICS:
        for suffix in ("mean", "std", "ci95_half_width", "ci95_low", "ci95_high", "min", "max", "n"):
            row[f"{metric}_{suffix}"] = summary["metric_stats"][metric][suffix]
    return row


def report_table(summaries: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| config | label | confidence | graph H32 | strongest comparator | comparator H32 | warning |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        stats_by_metric = summary["metric_stats"]
        comparator = str(summary["strongest_comparator"])
        comparator_metric = f"{comparator}_h32"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(summary["config"]),
                    str(summary["protocol_label"]),
                    str(summary["label_confidence"]),
                    fmt_mean_std(stats_by_metric["graph_h32"]),
                    comparator,
                    fmt_mean_std(stats_by_metric[comparator_metric]) if comparator_metric in stats_by_metric else "NA",
                    str(summary["warning_flag"]),
                ]
            )
            + " |"
        )
    return lines


def run_regime_aggregator(
    *,
    title: str,
    default_out_root: Path,
    config_names: list[str],
    default_seeds: list[int],
    default_report: Path,
    default_table: Path,
    default_summary_csv: Path,
    seed_count: int,
    argv: list[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=f"Aggregate multiseed preflight package for {title}.")
    parser.add_argument("--out-root", type=Path, default=default_out_root)
    parser.add_argument("--merge-dirs", type=parse_csv_paths, default=None)
    parser.add_argument("--seeds", type=parse_csv_ints, default=default_seeds)
    parser.add_argument("--t-critical", type=float, default=T_CRITICAL_95_N5 if seed_count == 5 else T_CRITICAL_95_N3)
    parser.add_argument("--report", type=Path, default=default_report)
    parser.add_argument("--table", type=Path, default=default_table)
    parser.add_argument("--summary-csv", type=Path, default=default_summary_csv)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    roots = [path.resolve() for path in args.merge_dirs] if args.merge_dirs else [args.out_root.resolve()]
    summaries = [
        summarize_config(roots=roots, config_name=name, seeds=args.seeds, t_critical=float(args.t_critical))
        for name in config_names
    ]
    if args.dry_run:
        print("Dry run only. No files will be written.")
        for summary in summaries:
            print(f"Config {summary['config']}:")
            for path in summary["source_paths"]:
                print(f"  WOULD READ {rel(Path(path))} [{'exists' if Path(path).exists() else 'missing'}]")
            count = sum(int(summary["metric_stats"][metric]["n"]) for metric in H32_METRICS)
            print(f"  unified seed/prior H32 row count: {count}")
        return
    rows = [csv_row(summary) for summary in summaries]
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    markdown = "\n".join(
        [
            f"# {title}",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            *report_table(summaries),
            "",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown, encoding="utf-8")
    args.table.write_text(markdown, encoding="utf-8")
    print(f"Wrote {rel(args.summary_csv)}")
    print(f"Wrote {rel(args.table)}")
    print(f"Wrote {rel(args.report)}")
    print("Summary CSV columns:")
    print(",".join(fieldnames))
