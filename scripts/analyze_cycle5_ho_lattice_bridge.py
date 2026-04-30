from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "experiments" / "results" / "cycle5_ho_lattice_bridge" / "cycle5_ho_lattice_bridge_results.json"
DEFAULT_CYCLE3_RESULTS = ROOT / "experiments" / "results" / "cycle3_ho_networks" / "cycle3_ho_networks_results.json"
DEFAULT_CYCLE4_RESULTS = ROOT / "experiments" / "results" / "cycle4_ho_lambda_robustness" / "cycle4_ho_lambda_robustness_results.json"
DEFAULT_REPORT = ROOT / "analysis_out" / "CYCLE5_HO_LATTICE_BRIDGE_REPORT.md"
TRAINER = ROOT / "scripts" / "train_cycle3_ho_networks.py"

TOPOLOGY = "lattice"
PRIORS = ["graph", "permuted_graph", "random_graph"]
LAMBDAS = ["0.001", "0.005", "0.01", "0.05", "0.1"]
SEEDS = [0, 1, 2, 3, 4]
HORIZONS = ["1", "2", "4", "8", "16", "32"]
EXPECTED = {
    (TOPOLOGY, prior, lambda_value, seed)
    for prior in PRIORS
    for lambda_value in LAMBDAS
    for seed in SEEDS
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def lambda_label(value: Any) -> str:
    if not finite(value):
        return "nan"
    return f"{float(value):g}"


def fmt(value: Any, digits: int = 4) -> str:
    if not finite(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def fmt_sci(value: Any, digits: int = 3) -> str:
    if not finite(value):
        return "nan"
    return f"{float(value):.{digits}e}"


def mean_std(values: list[float]) -> tuple[float, float, int]:
    array = np.asarray([value for value in values if finite(value)], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan"), 0
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0, int(array.size)


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


def pct_lower(candidate: float, control: float) -> float:
    if not finite(candidate) or not finite(control) or control == 0.0:
        return float("nan")
    return 100.0 * (control - candidate) / abs(control)


def fmt_pct(value: Any) -> str:
    if not finite(value):
        return "nan"
    return f"{float(value):+.1f}%"


def trainer_default_prior_weight() -> float:
    text = TRAINER.read_text(encoding="utf-8")
    match = re.search(r"prior_weight:\s*float\s*=\s*([0-9.eE+-]+)", text)
    if not match:
        return float("nan")
    return float(match.group(1))


def successful_runs(results: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        run for run in results.get("runs", {}).values()
        if run.get("status") == "ok" and not run.get("failure_flag")
    ]


def grouped_runs(results: dict[str, Any], default_weight: float | None = None) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in successful_runs(results):
        config = run.get("config", {})
        topology = str(config.get("topology"))
        prior = str(config.get("prior"))
        if topology != TOPOLOGY or prior not in PRIORS:
            continue
        prior_weight = config.get("prior_weight", default_weight)
        grouped[(topology, prior, lambda_label(prior_weight))].append(run)
    return grouped


def rollout_values(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    prior: str,
    lambda_value: str,
    horizon: str,
) -> list[float]:
    return [
        float(run.get("diagnostics", {}).get("rollout_errors", {}).get(horizon, float("nan")))
        for run in grouped.get((TOPOLOGY, prior, lambda_value), [])
    ]


def rollout_mean(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    prior: str,
    lambda_value: str,
    horizon: str,
) -> float:
    return mean_std(rollout_values(grouped, prior, lambda_value, horizon))[0]


def seed_rollout_map(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    prior: str,
    lambda_value: str,
    horizon: str,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for run in grouped.get((TOPOLOGY, prior, lambda_value), []):
        config = run.get("config", {})
        value = run.get("diagnostics", {}).get("rollout_errors", {}).get(horizon)
        if finite(value):
            values[int(config.get("seed", -1))] = float(value)
    return values


def paired_deltas(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    control_prior: str,
    lambda_value: str,
    horizon: str,
) -> list[float]:
    graph_values = seed_rollout_map(grouped, "graph", lambda_value, horizon)
    control_values = seed_rollout_map(grouped, control_prior, lambda_value, horizon)
    return [
        control_values[seed] - graph_values[seed]
        for seed in SEEDS
        if seed in graph_values and seed in control_values
    ]


def paired_stats(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    control_prior: str,
    lambda_value: str,
    horizon: str,
) -> dict[str, float | int]:
    deltas = paired_deltas(grouped, control_prior, lambda_value, horizon)
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


def specificity_supported(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    lambda_value: str,
    horizon: str = "32",
) -> tuple[bool, str]:
    perm = paired_stats(grouped, "permuted_graph", lambda_value, horizon)
    rand = paired_stats(grouped, "random_graph", lambda_value, horizon)
    supported = all(
        int(stats["n"]) == len(SEEDS)
        and float(stats["mean_delta"]) > 0.0
        and int(stats["win_count"]) >= 4
        and float(stats["ci_low"]) > 0.0
        for stats in [perm, rand]
    )
    reason = (
        f"perm delta={fmt(float(perm['mean_delta']))}, wins={int(perm['win_count'])}/{int(perm['n'])}, "
        f"CI=[{fmt(float(perm['ci_low']))}, {fmt(float(perm['ci_high']))}]; "
        f"rand delta={fmt(float(rand['mean_delta']))}, wins={int(rand['win_count'])}/{int(rand['n'])}, "
        f"CI=[{fmt(float(rand['ci_low']))}, {fmt(float(rand['ci_high']))}]"
    )
    return supported, reason


def check_expected(results: dict[str, Any]) -> tuple[list[tuple[str, str, str, int]], list[str], list[str]]:
    observed = set()
    failures = []
    nan_runs = []
    for name, run in results.get("runs", {}).items():
        config = run.get("config", {})
        topology = str(config.get("topology"))
        prior = str(config.get("prior"))
        prior_weight = lambda_label(config.get("prior_weight"))
        seed = int(config.get("seed", -1))
        observed.add((topology, prior, prior_weight, seed))
        if run.get("status") != "ok" or run.get("failure_flag"):
            failures.append(str(name))
            continue
        rollout = run.get("diagnostics", {}).get("rollout_errors", {})
        for horizon in HORIZONS:
            if not finite(rollout.get(horizon)):
                nan_runs.append(str(name))
                break
    return sorted(EXPECTED - observed), sorted(failures), sorted(set(nan_runs))


def cycle4_lambda_audit(cycle4_results: dict[str, Any]) -> list[str]:
    if not cycle4_results:
        return ["Cycle 4 results file not found; expected sweep was 0.001, 0.005, 0.01, 0.05."]
    lambdas = sorted({
        lambda_label(run.get("config", {}).get("prior_weight"))
        for run in successful_runs(cycle4_results)
        if run.get("config", {}).get("topology") in {"lattice", "scalefree"}
    }, key=float)
    includes = "YES" if "0.1" in lambdas else "NO"
    return [f"Cycle 4 observed lambdas: {', '.join(lambdas)}. Includes 0.1? {includes}."]


def metadata_summary(results: dict[str, Any]) -> tuple[int, int]:
    graph_runs = [
        run for run in successful_runs(results)
        if run.get("config", {}).get("prior") in PRIORS
    ]
    with_hash = sum(1 for run in graph_runs if run.get("config_hash") and run.get("git_commit"))
    with_graph = sum(
        1 for run in graph_runs
        if isinstance(run.get("graph_metadata"), dict)
        and finite(run["graph_metadata"].get("graph_laplacian_trace"))
        and finite(run["graph_metadata"].get("graph_laplacian_frobenius_norm"))
        and finite(run["graph_metadata"].get("graph_laplacian_largest_eigenvalue"))
    )
    return with_hash, with_graph


def cycle3_cycle5_comparison(
    cycle3_grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    cycle5_grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> list[str]:
    lines = [
        "## Cycle 3 vs Cycle 5 Lambda=0.1 Comparison",
        "",
        "| source | prior | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source, grouped in [("Cycle 3 implicit", cycle3_grouped), ("Cycle 5 explicit", cycle5_grouped)]:
        for prior in PRIORS:
            cells = [fmt_mean_std(rollout_values(grouped, prior, "0.1", horizon)) for horizon in HORIZONS]
            lines.append(f"| {source} | {prior} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "| source | control | H=32 paired delta +/- std | graph wins | bootstrap 95% CI |",
        "|---|---|---:|---:|---:|",
    ])
    for source, grouped in [("Cycle 3 implicit", cycle3_grouped), ("Cycle 5 explicit", cycle5_grouped)]:
        for control in ["permuted_graph", "random_graph"]:
            stats = paired_stats(grouped, control, "0.1", "32")
            lines.append(
                f"| {source} | {control} | {fmt(float(stats['mean_delta']))} +/- {fmt(float(stats['std_delta']))} | "
                f"{int(stats['win_count'])}/{int(stats['n'])} | "
                f"[{fmt(float(stats['ci_low']))}, {fmt(float(stats['ci_high']))}] |"
            )
    return lines


def conclusion(
    cycle3_grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    cycle5_grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> str:
    c3_supported, c3_reason = specificity_supported(cycle3_grouped, "0.1")
    c5_supported, c5_reason = specificity_supported(cycle5_grouped, "0.1")
    cycle5_support = {lambda_value: specificity_supported(cycle5_grouped, lambda_value)[0] for lambda_value in LAMBDAS}
    completed = all(int(paired_stats(cycle5_grouped, "permuted_graph", lambda_value, "32")["n"]) == len(SEEDS) for lambda_value in LAMBDAS)
    if not completed:
        return (
            "Cycle 5 is not complete enough to make the final bridge conclusion. "
            f"Cycle 3 lambda=0.1 specificity support: {'YES' if c3_supported else 'NO'} ({c3_reason})."
        )

    robust = all(cycle5_support.values())
    high_only = bool(cycle5_support.get("0.1")) and not any(cycle5_support[value] for value in LAMBDAS if value != "0.1")
    cycle3_h32 = [
        rollout_mean(cycle3_grouped, prior, "0.1", "32")
        for prior in PRIORS
    ]
    cycle5_h32 = [
        rollout_mean(cycle5_grouped, prior, "0.1", "32")
        for prior in PRIORS
    ]
    same_h32 = all(
        finite(left) and finite(right) and abs(left - right) < 5e-7
        for left, right in zip(cycle3_h32, cycle5_h32)
    )

    if c3_supported and c5_supported and same_h32:
        reproduce = "YES"
    elif c3_supported and c5_supported:
        reproduce = "YES, qualitatively"
    elif c3_supported and not c5_supported:
        reproduce = "NO"
    else:
        reproduce = "NOT ESTABLISHED"

    if robust:
        paper_claim = "A true topology-specific claim is supported for lattice across this lambda bridge."
    elif high_only:
        paper_claim = "Do not claim lambda-robust true topology specificity; at most describe high-lambda lattice specificity and frame lower-lambda gains as generic graph smoothing."
    else:
        paper_claim = "The safer paper claim is generic smoothing/regularization, not stable true topology specificity."

    return (
        f"Does Cycle 3 lattice specificity reproduce? {reproduce}. "
        f"Cycle 3 evidence: {c3_reason}. Cycle 5 lambda=0.1 evidence: {c5_reason}\n"
        f"Is specificity lambda-robust? {'YES' if robust else 'NO'}.\n"
        f"Is specificity high-lambda-only? {'YES' if high_only else 'NO'}.\n"
        f"Paper stance: {paper_claim}"
    )


def build_report(
    cycle5_results: dict[str, Any],
    cycle5_path: Path,
    cycle3_results: dict[str, Any],
    cycle4_results: dict[str, Any],
) -> str:
    default_weight = trainer_default_prior_weight()
    cycle5_grouped = grouped_runs(cycle5_results)
    cycle3_grouped = grouped_runs(cycle3_results, default_weight=default_weight)
    missing, failures, nan_runs = check_expected(cycle5_results)
    success_count = sum(1 for run in successful_runs(cycle5_results) if run.get("config", {}).get("topology") == TOPOLOGY)
    metadata_hash_count, metadata_graph_count = metadata_summary(cycle5_results)
    result_label = cycle5_path.relative_to(ROOT) if cycle5_path.is_absolute() and cycle5_path.is_relative_to(ROOT) else cycle5_path

    lines = [
        "# Cycle 5 HO Lattice Bridge Report",
        "",
        f"Results file: `{result_label}`",
        f"Schema version: `{cycle5_results.get('schema_version', 'missing')}`",
        "",
        "## Cycle 3 Default Lambda Audit",
        "",
        f"- `scripts/train_cycle3_ho_networks.py` default `prior_weight` = {default_weight:g}.",
        "- Cycle 3 configs omitted `prior_weight`, so Cycle 3 lattice graph-prior runs used the trainer default lambda=0.1.",
    ]
    lines.extend(f"- {line}" for line in cycle4_lambda_audit(cycle4_results))
    lines.extend([
        "",
        "## Run Integrity",
        "",
        f"Success/failure count: {success_count} ok / {len(failures)} failed",
        f"Missing expected runs: {len(missing)}",
        f"Runs with non-finite rollout diagnostics: {nan_runs or 'none'}",
        f"Runs with config_hash + git_commit: {metadata_hash_count}",
        f"Runs with graph Laplacian metadata: {metadata_graph_count}",
        "",
        "## Rollout Error Mean +/- Std",
        "",
        "| lambda | prior | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for lambda_value in LAMBDAS:
        for prior in PRIORS:
            cells = [fmt_mean_std(rollout_values(cycle5_grouped, prior, lambda_value, horizon)) for horizon in HORIZONS]
            lines.append(f"| {lambda_value} | {prior} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Paired Deltas And Win Counts",
        "",
        "Deltas are control rollout error minus graph rollout error. Positive values mean graph has lower error.",
        "",
        "| lambda | control | horizon | mean delta +/- std | graph wins | bootstrap 95% CI |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for lambda_value in LAMBDAS:
        for control in ["permuted_graph", "random_graph"]:
            for horizon in HORIZONS:
                stats = paired_stats(cycle5_grouped, control, lambda_value, horizon)
                lines.append(
                    f"| {lambda_value} | {control} | H={horizon} | "
                    f"{fmt(float(stats['mean_delta']))} +/- {fmt(float(stats['std_delta']))} | "
                    f"{int(stats['win_count'])}/{int(stats['n'])} | "
                    f"[{fmt(float(stats['ci_low']))}, {fmt(float(stats['ci_high']))}] |"
                )

    lines.extend([""])
    lines.extend(cycle3_cycle5_comparison(cycle3_grouped, cycle5_grouped))
    lines.extend([
        "",
        "## Specificity Summary",
        "",
        "Support requires H=32 paired evidence against both controls: positive mean delta, at least 4/5 graph wins, and bootstrap CI lower bound above zero.",
        "",
        "| lambda | graph_specificity_supported? | H=32 evidence |",
        "|---:|---|---|",
    ])
    for lambda_value in LAMBDAS:
        supported, reason = specificity_supported(cycle5_grouped, lambda_value)
        lines.append(f"| {lambda_value} | {'YES' if supported else 'NO'} | {reason} |")

    lines.extend([
        "",
        "## Conclusion",
        "",
        conclusion(cycle3_grouped, cycle5_grouped),
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Cycle 5 HO lattice lambda bridge.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--cycle3-results", type=Path, default=DEFAULT_CYCLE3_RESULTS)
    parser.add_argument("--cycle4-results", type=Path, default=DEFAULT_CYCLE4_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    report = build_report(
        load_json(args.results),
        args.results,
        load_json(args.cycle3_results),
        load_json(args.cycle4_results),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
