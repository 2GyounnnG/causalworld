from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "experiments" / "results" / "cycle4_ho_lambda_robustness" / "cycle4_ho_lambda_robustness_results.json"
DEFAULT_REPORT = ROOT / "analysis_out" / "CYCLE4_HO_LAMBDA_ROBUSTNESS_REPORT.md"
TOPOLOGIES = ["lattice", "scalefree"]
PRIORS = ["graph", "permuted_graph", "random_graph"]
LAMBDAS = ["0.001", "0.005", "0.01", "0.05"]
SEEDS = [0, 1, 2]
HORIZONS = ["1", "2", "4", "8", "16", "32"]
EXPECTED = {
    (topology, prior, lambda_label, seed)
    for topology in TOPOLOGIES
    for prior in PRIORS
    for lambda_label in LAMBDAS
    for seed in SEEDS
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def lambda_label(value: Any) -> str:
    return f"{float(value):g}"


def mean_std(values: list[float]) -> tuple[float, float, int]:
    array = np.asarray([value for value in values if finite(value)], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan"), 0
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0, int(array.size)


def fmt(value: float, digits: int = 4) -> str:
    if not finite(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def fmt_mean_std(values: list[float], digits: int = 4) -> str:
    mean, std, n = mean_std(values)
    return f"{fmt(mean, digits)} +/- {fmt(std, digits)} (n={n})"


def bootstrap_mean_ci(values: list[float], n_bootstrap: int = 10000) -> tuple[float, float]:
    array = np.asarray([value for value in values if finite(value)], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(1729)
    indices = rng.integers(0, array.size, size=(n_bootstrap, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def group_successful_runs(runs: dict[str, Any]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs.values():
        if run.get("status") != "ok" or run.get("failure_flag"):
            continue
        config = run.get("config", {})
        grouped[
            (
                str(config.get("topology")),
                str(config.get("prior")),
                lambda_label(config.get("prior_weight")),
            )
        ].append(run)
    return grouped


def rollout_values(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    topology: str,
    prior: str,
    lambda_value: str,
    horizon: str,
) -> list[float]:
    return [
        float(run.get("diagnostics", {}).get("rollout_errors", {}).get(horizon, float("nan")))
        for run in grouped.get((topology, prior, lambda_value), [])
    ]


def seed_rollout_map(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    topology: str,
    prior: str,
    lambda_value: str,
    horizon: str,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for run in grouped.get((topology, prior, lambda_value), []):
        config = run.get("config", {})
        value = run.get("diagnostics", {}).get("rollout_errors", {}).get(horizon)
        if finite(value):
            values[int(config.get("seed", -1))] = float(value)
    return values


def paired_deltas(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    topology: str,
    control_prior: str,
    lambda_value: str,
    horizon: str,
) -> list[float]:
    graph_values = seed_rollout_map(grouped, topology, "graph", lambda_value, horizon)
    control_values = seed_rollout_map(grouped, topology, control_prior, lambda_value, horizon)
    deltas = []
    for seed in SEEDS:
        if seed in graph_values and seed in control_values:
            deltas.append(control_values[seed] - graph_values[seed])
    return deltas


def paired_stats(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    topology: str,
    control_prior: str,
    lambda_value: str,
    horizon: str,
) -> dict[str, float | int]:
    deltas = paired_deltas(grouped, topology, control_prior, lambda_value, horizon)
    mean, std, n = mean_std(deltas)
    ci_low, ci_high = bootstrap_mean_ci(deltas)
    return {
        "n": n,
        "mean_delta": mean,
        "std_delta": std,
        "win_count": sum(1 for delta in deltas if delta > 0.0),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def check_expected(runs: dict[str, Any]) -> tuple[list[tuple[str, str, str, int]], list[str], list[str]]:
    observed = set()
    failures = []
    nan_runs = []
    for name, run in runs.items():
        config = run.get("config", {})
        observed.add(
            (
                str(config.get("topology")),
                str(config.get("prior")),
                lambda_label(config.get("prior_weight")),
                int(config.get("seed", -1)),
            )
        )
        if run.get("status") != "ok" or run.get("failure_flag"):
            failures.append(name)
            continue
        rollout = run.get("diagnostics", {}).get("rollout_errors", {})
        for horizon in HORIZONS:
            if not finite(rollout.get(horizon)):
                nan_runs.append(name)
                break
    return sorted(EXPECTED - observed), sorted(failures), sorted(set(nan_runs))


def graph_metadata_ok(runs: dict[str, Any]) -> bool:
    graph_runs = [
        run for run in runs.values()
        if run.get("status") == "ok" and run.get("config", {}).get("prior") in PRIORS
    ]
    return bool(graph_runs) and all(
        run.get("prior_implementation", {}).get("graph_prior_nodewise") is True
        and run.get("prior_implementation", {}).get("graph_prior_form") == "nodewise_trace_HtLH"
        and run.get("prior_implementation", {}).get("uses_latent_projected_laplacian") is False
        for run in graph_runs
    )


def specificity_supported(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    topology: str,
    lambda_value: str,
    horizon: str = "32",
) -> tuple[bool, str]:
    perm = paired_stats(grouped, topology, "permuted_graph", lambda_value, horizon)
    rand = paired_stats(grouped, topology, "random_graph", lambda_value, horizon)
    # With n=3, call graph specificity "supported" only for a clean paired win
    # against both controls at the longest horizon.
    perm_supported = (
        int(perm["n"]) == len(SEEDS)
        and float(perm["mean_delta"]) > 0.0
        and int(perm["win_count"]) == len(SEEDS)
        and float(perm["ci_low"]) > 0.0
    )
    rand_supported = (
        int(rand["n"]) == len(SEEDS)
        and float(rand["mean_delta"]) > 0.0
        and int(rand["win_count"]) == len(SEEDS)
        and float(rand["ci_low"]) > 0.0
    )
    supported = perm_supported and rand_supported
    reason = (
        f"perm delta={fmt(float(perm['mean_delta']))}, wins={int(perm['win_count'])}/{int(perm['n'])}, "
        f"CI=[{fmt(float(perm['ci_low']))}, {fmt(float(perm['ci_high']))}]; "
        f"rand delta={fmt(float(rand['mean_delta']))}, wins={int(rand['win_count'])}/{int(rand['n'])}, "
        f"CI=[{fmt(float(rand['ci_low']))}, {fmt(float(rand['ci_high']))}]"
    )
    return supported, reason


def recommendation(support: dict[tuple[str, str], bool]) -> str:
    lattice_supported = sum(1 for lam in LAMBDAS if support.get(("lattice", lam), False))
    scalefree_supported = sum(1 for lam in LAMBDAS if support.get(("scalefree", lam), False))
    lines = []
    if lattice_supported == len(LAMBDAS):
        lines.append("Lattice specificity is robust across all tested lambda values.")
    elif lattice_supported >= 3:
        lines.append("Lattice specificity is mostly robust, but one lambda should be treated as a sensitivity point.")
    elif lattice_supported > 0:
        lines.append("Lattice specificity is lambda-sensitive rather than robust.")
    else:
        lines.append("Lattice specificity is not supported across the tested lambda values.")

    if scalefree_supported == 0:
        lines.append("Scalefree remains random-control-equivalent across the lambda sweep.")
    else:
        lines.append("Scalefree does not cleanly remain random-control-equivalent; inspect lambda-specific exceptions before making the claim.")

    if lattice_supported >= 3 and scalefree_supported == 0:
        lines.append("Recommendation: write the current controlled-HO results before ISO17; run ISO17 only after the thesis text distinguishes lattice topology-specific structure from generic graph smoothing.")
    else:
        lines.append("Recommendation: do not run ISO17 yet; resolve lambda sensitivity before expanding to ISO17.")
    return "\n".join(lines)


def build_report(results: dict[str, Any], results_path: Path) -> str:
    runs = results.get("runs", {})
    grouped = group_successful_runs(runs)
    missing, failures, nan_runs = check_expected(runs)
    graph_ok = graph_metadata_ok(runs)
    success_count = sum(1 for run in runs.values() if run.get("status") == "ok" and not run.get("failure_flag"))
    result_label = results_path.relative_to(ROOT) if results_path.is_absolute() and results_path.is_relative_to(ROOT) else results_path

    lines = [
        "# Cycle 4 HO Lambda Robustness Report",
        "",
        f"Results file: `{result_label}`",
        f"Schema version: `{results.get('schema_version')}`",
        f"Success/failure count: {success_count} ok / {len(failures)} failed",
        f"Missing expected runs: {len(missing)}",
        f"Runs with non-finite rollout diagnostics: {nan_runs or 'none'}",
        f"Graph prior metadata node-wise: {'YES' if graph_ok else 'NO'}",
        "",
        "## Rollout Error Mean +/- Std",
        "",
        "| topology | lambda | prior | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for topology in TOPOLOGIES:
        for lambda_value in LAMBDAS:
            for prior in PRIORS:
                cells = [
                    fmt_mean_std(rollout_values(grouped, topology, prior, lambda_value, horizon))
                    for horizon in HORIZONS
                ]
                lines.append(f"| {topology} | {lambda_value} | {prior} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Paired Graph-Specificity Deltas",
        "",
        "Deltas are paired by seed as control rollout error minus graph rollout error. Positive values mean graph has lower error than the control.",
        "",
        "| topology | lambda | control | horizon | mean delta +/- std | graph wins | bootstrap 95% CI |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for topology in TOPOLOGIES:
        for lambda_value in LAMBDAS:
            for control_prior in ["permuted_graph", "random_graph"]:
                for horizon in HORIZONS:
                    stats = paired_stats(grouped, topology, control_prior, lambda_value, horizon)
                    lines.append(
                        f"| {topology} | {lambda_value} | {control_prior} | H={horizon} | "
                        f"{fmt(float(stats['mean_delta']))} +/- {fmt(float(stats['std_delta']))} | "
                        f"{int(stats['win_count'])}/{int(stats['n'])} | "
                        f"[{fmt(float(stats['ci_low']))}, {fmt(float(stats['ci_high']))}] |"
                    )

    support: dict[tuple[str, str], bool] = {}
    lines.extend([
        "",
        "## Specificity Summary",
        "",
        "Support requires H=32 paired evidence against both controls: positive mean delta, 3/3 graph wins, and bootstrap CI lower bound above zero.",
        "",
        "| topology | lambda | graph_specificity_supported? | H=32 evidence |",
        "|---|---:|---|---|",
    ])
    for topology in TOPOLOGIES:
        for lambda_value in LAMBDAS:
            supported, reason = specificity_supported(grouped, topology, lambda_value)
            support[(topology, lambda_value)] = supported
            lines.append(f"| {topology} | {lambda_value} | {'YES' if supported else 'NO'} | {reason} |")

    lines.extend([
        "",
        "## Recommendation",
        "",
        recommendation(support),
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Cycle 4 HO lambda robustness sweep.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    results = load_json(args.results)
    report = build_report(results, args.results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
