from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "experiments" / "results" / "cycle3_ho_networks" / "cycle3_ho_networks_results.json"
DEFAULT_REPORT = ROOT / "analysis_out" / "CYCLE3_HO_NETWORKS_REPORT.md"
TOPOLOGIES = ["lattice", "random", "scalefree"]
PRIORS = ["none", "covariance", "graph", "permuted_graph", "random_graph"]
SEEDS = [0, 1, 2, 3, 4]
HORIZONS = ["1", "2", "4", "8", "16", "32"]
EXPECTED = {
    (topology, prior, seed)
    for topology in TOPOLOGIES
    for prior in PRIORS
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


def pct_improvement(baseline: float, candidate: float) -> float:
    if not finite(baseline) or not finite(candidate) or baseline == 0.0:
        return float("nan")
    return 100.0 * (baseline - candidate) / abs(baseline)


def pct_lower(candidate: float, control: float) -> float:
    if not finite(candidate) or not finite(control) or control == 0.0:
        return float("nan")
    return 100.0 * (control - candidate) / abs(control)


def fmt_delta(pct: float) -> str:
    if not finite(pct):
        return "nan"
    label = "gain" if pct > 0 else "loss"
    return f"{pct:+.1f}% {label}"


def group_successful_runs(runs: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs.values():
        if run.get("status") != "ok" or run.get("failure_flag"):
            continue
        config = run.get("config", {})
        grouped[(str(config.get("topology")), str(config.get("prior")))].append(run)
    return grouped


def rollout_values(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    topology: str,
    prior: str,
    horizon: str,
) -> list[float]:
    return [
        float(run.get("diagnostics", {}).get("rollout_errors", {}).get(horizon, float("nan")))
        for run in grouped.get((topology, prior), [])
    ]


def rollout_mean(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    topology: str,
    prior: str,
    horizon: str = "32",
) -> float:
    return mean_std(rollout_values(grouped, topology, prior, horizon))[0]


def compare_cell(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    topology: str,
    left_prior: str,
    right_prior: str,
    horizon: str,
) -> str:
    left_mean = rollout_mean(grouped, topology, left_prior, horizon)
    right_mean = rollout_mean(grouped, topology, right_prior, horizon)
    return f"{fmt(left_mean)} -> {fmt(right_mean)} ({fmt_delta(pct_improvement(left_mean, right_mean))})"


def specificity_cell(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    topology: str,
    control_prior: str,
    horizon: str,
) -> str:
    graph_mean = rollout_mean(grouped, topology, "graph", horizon)
    control_mean = rollout_mean(grouped, topology, control_prior, horizon)
    return f"{fmt(graph_mean)} vs {fmt(control_mean)} ({fmt_delta(pct_lower(graph_mean, control_mean))})"


def seed_rollout_map(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    topology: str,
    prior: str,
    horizon: str,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for run in grouped.get((topology, prior), []):
        config = run.get("config", {})
        value = run.get("diagnostics", {}).get("rollout_errors", {}).get(horizon)
        if finite(value):
            values[int(config.get("seed", -1))] = float(value)
    return values


def paired_delta_values(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    topology: str,
    control_prior: str,
    horizon: str,
) -> list[float]:
    graph_values = seed_rollout_map(grouped, topology, "graph", horizon)
    control_values = seed_rollout_map(grouped, topology, control_prior, horizon)
    deltas = []
    for seed in SEEDS:
        if seed in graph_values and seed in control_values:
            deltas.append(control_values[seed] - graph_values[seed])
    return deltas


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


def paired_specificity_stats(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    topology: str,
    control_prior: str,
    horizon: str,
) -> dict[str, float | int]:
    deltas = paired_delta_values(grouped, topology, control_prior, horizon)
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


def paired_specificity_cell(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    topology: str,
    control_prior: str,
    horizon: str,
) -> str:
    stats = paired_specificity_stats(grouped, topology, control_prior, horizon)
    n = int(stats["n"])
    wins = int(stats["win_count"])
    return (
        f"{fmt(float(stats['mean_delta']))} +/- {fmt(float(stats['std_delta']))}; "
        f"wins={wins}/{n}; "
        f"95% CI [{fmt(float(stats['ci_low']))}, {fmt(float(stats['ci_high']))}]"
    )


def check_expected(runs: dict[str, Any]) -> tuple[list[tuple[str, str, int]], list[str], list[str]]:
    observed = set()
    failures = []
    nan_runs = []
    for name, run in runs.items():
        config = run.get("config", {})
        observed.add((str(config.get("topology")), str(config.get("prior")), int(config.get("seed", -1))))
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
    graph_priors = {"graph", "permuted_graph", "random_graph"}
    graph_runs = [
        run for run in runs.values()
        if run.get("status") == "ok" and run.get("config", {}).get("prior") in graph_priors
    ]
    return bool(graph_runs) and all(
        run.get("prior_implementation", {}).get("graph_prior_nodewise") is True
        and run.get("prior_implementation", {}).get("graph_prior_form") == "nodewise_trace_HtLH"
        and run.get("prior_implementation", {}).get("uses_latent_projected_laplacian") is False
        for run in graph_runs
    )


def topology_metrics(grouped: dict[tuple[str, str], list[dict[str, Any]]], topology: str, horizon: str = "32") -> dict[str, float]:
    none = rollout_mean(grouped, topology, "none", horizon)
    covariance = rollout_mean(grouped, topology, "covariance", horizon)
    graph = rollout_mean(grouped, topology, "graph", horizon)
    permuted = rollout_mean(grouped, topology, "permuted_graph", horizon)
    random = rollout_mean(grouped, topology, "random_graph", horizon)
    return {
        "none": none,
        "covariance": covariance,
        "graph": graph,
        "permuted_graph": permuted,
        "random_graph": random,
        "covariance_gain_pct": pct_improvement(none, covariance),
        "graph_gain_pct": pct_improvement(none, graph),
        "permuted_gain_pct": pct_improvement(none, permuted),
        "random_gain_pct": pct_improvement(none, random),
        "graph_vs_permuted_pct": pct_lower(graph, permuted),
        "graph_vs_random_pct": pct_lower(graph, random),
        "graph_vs_covariance_pct": pct_lower(graph, covariance),
    }


def classify_topology(metrics: dict[str, float]) -> tuple[str, str]:
    required = ["none", "covariance", "graph", "permuted_graph", "random_graph"]
    if any(not finite(metrics[key]) for key in required):
        return "incomplete", "missing one or more H=32 means"

    graph_gain = metrics["graph_gain_pct"]
    cov_gain = metrics["covariance_gain_pct"]
    perm_gain = metrics["permuted_gain_pct"]
    rand_gain = metrics["random_gain_pct"]
    spec_perm = metrics["graph_vs_permuted_pct"]
    spec_rand = metrics["graph_vs_random_pct"]
    graph_vs_cov = metrics["graph_vs_covariance_pct"]

    if graph_gain < -5.0:
        return "prior-hurts/mismatch", f"graph is worse than none at H=32 ({fmt_delta(graph_gain)})"
    if cov_gain >= graph_gain - 5.0 and cov_gain >= 5.0 and graph_vs_cov <= 5.0:
        return "covariance-explained", "covariance matches or beats graph within 5%"
    if graph_gain >= 5.0 and spec_perm >= 10.0 and spec_rand >= 10.0:
        return "true-graph-specific", "true graph beats both graph controls by >=10%"
    if graph_gain >= 5.0 and abs(spec_perm) <= 10.0 and abs(spec_rand) <= 10.0:
        return "random-control-equivalent", "true, permuted, and random graph controls are within +/-10%"
    if graph_gain >= 10.0 and (perm_gain >= 5.0 or rand_gain >= 5.0):
        return "graph-smoothing-dominated", "graph controls improve over none without true-graph specificity"
    return "random-control-equivalent", "no topology-specific advantage under H=32 thresholds"


def recommendation(classifications: dict[str, tuple[str, str]], failures: list[str], missing: list[tuple[str, str, int]]) -> str:
    labels = [classifications[topology][0] for topology in TOPOLOGIES]
    n_specific = labels.count("true-graph-specific")
    n_smoothing = labels.count("graph-smoothing-dominated")
    n_equivalent = labels.count("random-control-equivalent")
    n_cov = labels.count("covariance-explained")

    lines = []
    if missing or failures:
        lines.append("The sweep is incomplete, so recommendations are provisional.")

    if n_specific >= 2:
        lines.append("rMD17 top-up recommendation: top up the flagged Cycle 2 rMD17 molecules before ISO17.")
        lines.append("ISO17 recommendation: run ISO17 after rMD17 top-up, because controlled HO supports topology-specific structure.")
        lines.append("Paper thesis: emphasize topology-specific relational structure, with graph smoothing as a secondary mechanism.")
    elif n_smoothing + n_equivalent + n_cov >= 2:
        lines.append("rMD17 top-up recommendation: do not top up rMD17 seeds yet; first resolve the generic-control mechanism.")
        lines.append("ISO17 recommendation: do not run ISO17 yet; it would likely amplify the same ambiguity at higher cost.")
        lines.append("Paper thesis: emphasize generic smoothing/conditioning unless a later controlled result recovers true-topology specificity.")
    else:
        lines.append("rMD17 top-up recommendation: top up only if a downstream claim needs molecule-level stability intervals.")
        lines.append("ISO17 recommendation: wait until the controlled HO ambiguity is resolved.")
        lines.append("Paper thesis: keep both generic smoothing and topology-specific relational structure as live hypotheses.")
    return "\n".join(lines)


def build_report(results: dict[str, Any], results_path: Path) -> str:
    runs = results.get("runs", {})
    grouped = group_successful_runs(runs)
    missing, failures, nan_runs = check_expected(runs)
    graph_ok = graph_metadata_ok(runs)
    success_count = sum(1 for run in runs.values() if run.get("status") == "ok" and not run.get("failure_flag"))

    result_label = results_path.relative_to(ROOT) if results_path.is_absolute() and results_path.is_relative_to(ROOT) else results_path
    lines = [
        "# Cycle 3 Controlled HO Networks Report",
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
        "| topology | prior | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for topology in TOPOLOGIES:
        for prior in PRIORS:
            cells = [fmt_mean_std(rollout_values(grouped, topology, prior, horizon)) for horizon in HORIZONS]
            lines.append(f"| {topology} | {prior} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Graph Prior Gain Beyond GNN None",
        "",
        "| topology | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for topology in TOPOLOGIES:
        cells = [compare_cell(grouped, topology, "none", "graph", horizon) for horizon in HORIZONS]
        lines.append(f"| {topology} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Graph Specificity: Graph vs Permuted Graph",
        "",
        "| topology | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for topology in TOPOLOGIES:
        cells = [specificity_cell(grouped, topology, "permuted_graph", horizon) for horizon in HORIZONS]
        lines.append(f"| {topology} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Graph Specificity: Graph vs Random Graph",
        "",
        "| topology | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for topology in TOPOLOGIES:
        cells = [specificity_cell(grouped, topology, "random_graph", horizon) for horizon in HORIZONS]
        lines.append(f"| {topology} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Paired Graph-Specificity Analysis",
        "",
        "Paired deltas are computed seed-wise as control rollout error minus graph rollout error. Positive values mean the true graph has lower error than the control.",
        "",
        "| topology | control | horizon | mean delta +/- std | graph wins | bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for topology in TOPOLOGIES:
        for control_prior in ["permuted_graph", "random_graph"]:
            for horizon in HORIZONS:
                stats = paired_specificity_stats(grouped, topology, control_prior, horizon)
                n = int(stats["n"])
                wins = int(stats["win_count"])
                lines.append(
                    f"| {topology} | {control_prior} | H={horizon} | "
                    f"{fmt(float(stats['mean_delta']))} +/- {fmt(float(stats['std_delta']))} | "
                    f"{wins}/{n} | "
                    f"[{fmt(float(stats['ci_low']))}, {fmt(float(stats['ci_high']))}] |"
                )

    lines.extend([
        "",
        "## Euclidean Conditioning: Covariance vs Graph Controls",
        "",
        "| topology | covariance vs graph H=32 | covariance vs permuted H=32 | covariance vs random H=32 |",
        "|---|---:|---:|---:|",
    ])
    for topology in TOPOLOGIES:
        cells = [
            compare_cell(grouped, topology, "covariance", "graph", "32"),
            compare_cell(grouped, topology, "covariance", "permuted_graph", "32"),
            compare_cell(grouped, topology, "covariance", "random_graph", "32"),
        ]
        lines.append(f"| {topology} | " + " | ".join(cells) + " |")

    classifications: dict[str, tuple[str, str]] = {}
    lines.extend([
        "",
        "## Topology Classification",
        "",
        "| topology | class | covariance gain H=32 | graph gain H=32 | graph vs permuted H=32 | graph vs random H=32 | rationale |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for topology in TOPOLOGIES:
        metrics = topology_metrics(grouped, topology)
        label, reason = classify_topology(metrics)
        classifications[topology] = (label, reason)
        lines.append(
            f"| {topology} | {label} | "
            f"{fmt_delta(metrics['covariance_gain_pct'])} | "
            f"{fmt_delta(metrics['graph_gain_pct'])} | "
            f"{fmt_delta(metrics['graph_vs_permuted_pct'])} | "
            f"{fmt_delta(metrics['graph_vs_random_pct'])} | "
            f"{reason} |"
        )

    lines.extend([
        "",
        "## Recommendation",
        "",
        recommendation(classifications, failures, missing),
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Cycle 3 controlled HO network sweep.")
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
