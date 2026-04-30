from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "experiments" / "results" / "cycle1_aspirin_pilot" / "cycle1_aspirin_pilot_results.json"
DEFAULT_REPORT = ROOT / "analysis_out" / "CYCLE1_ASPIRIN_PILOT_REPORT.md"
HORIZONS = ["1", "2", "4", "8", "16", "32"]
DIAGNOSTICS = [
    "effective_rank",
    "covariance_condition_number",
    "projection_gaussianity_statistic",
    "prior_loss_mean",
]
MLP_PRIORS = ["none", "variance", "covariance", "sigreg"]
GNN_PRIORS = ["none", "variance", "covariance", "sigreg", "graph", "permuted_graph", "random_graph"]
EXPECTED = {
    (encoder, prior, seed)
    for encoder, priors in [("mlp_global", MLP_PRIORS), ("gnn_node", GNN_PRIORS)]
    for prior in priors
    for seed in range(5)
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


def group_successful_runs(runs: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs.values():
        if run.get("status") != "ok" or run.get("failure_flag"):
            continue
        config = run.get("config", {})
        grouped[(config.get("encoder"), config.get("prior"))].append(run)
    return grouped


def rollout_values(grouped: dict[tuple[str, str], list[dict[str, Any]]], encoder: str, prior: str, horizon: str) -> list[float]:
    return [
        float(run.get("diagnostics", {}).get("rollout_errors", {}).get(horizon, float("nan")))
        for run in grouped.get((encoder, prior), [])
    ]


def diagnostic_values(grouped: dict[tuple[str, str], list[dict[str, Any]]], encoder: str, prior: str, metric: str) -> list[float]:
    return [
        float(run.get("diagnostics", {}).get(metric, float("nan")))
        for run in grouped.get((encoder, prior), [])
    ]


def comparison_line(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    label: str,
    left: tuple[str, str],
    right: tuple[str, str],
    metric: str,
    lower_is_better: bool = True,
) -> str:
    left_values = diagnostic_values(grouped, *left, metric) if metric not in HORIZONS else rollout_values(grouped, *left, metric)
    right_values = diagnostic_values(grouped, *right, metric) if metric not in HORIZONS else rollout_values(grouped, *right, metric)
    left_mean, _, _ = mean_std(left_values)
    right_mean, _, _ = mean_std(right_values)
    if finite(left_mean) and finite(right_mean) and left_mean != 0.0:
        pct = 100.0 * (left_mean - right_mean) / abs(left_mean)
        direction = "gain" if (pct > 0 if lower_is_better else pct < 0) else "loss"
        delta = f"{pct:+.1f}% {direction}"
    else:
        delta = "nan"
    return f"- {label}: {fmt(left_mean)} -> {fmt(right_mean)} ({delta})"


def check_expected(runs: dict[str, Any]) -> tuple[list[tuple[str, str, int]], list[str], list[str]]:
    observed = set()
    failures = []
    nan_runs = []
    for name, run in runs.items():
        config = run.get("config", {})
        observed.add((config.get("encoder"), config.get("prior"), int(config.get("seed", -1))))
        if run.get("status") != "ok" or run.get("failure_flag"):
            failures.append(name)
            continue
        diagnostics = run.get("diagnostics", {})
        for horizon in HORIZONS:
            if not finite(diagnostics.get("rollout_errors", {}).get(horizon)):
                nan_runs.append(name)
        for metric in DIAGNOSTICS:
            if not finite(diagnostics.get(metric)):
                nan_runs.append(name)
    missing = sorted(EXPECTED - observed)
    return missing, sorted(failures), sorted(set(nan_runs))


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


def recommend(grouped: dict[tuple[str, str], list[dict[str, Any]]], graph_ok: bool, failures: list[str], missing: list[tuple[str, str, int]]) -> str:
    graph_h32 = mean_std(rollout_values(grouped, "gnn_node", "graph", "32"))[0]
    perm_h32 = mean_std(rollout_values(grouped, "gnn_node", "permuted_graph", "32"))[0]
    rand_h32 = mean_std(rollout_values(grouped, "gnn_node", "random_graph", "32"))[0]
    gnn_none_h32 = mean_std(rollout_values(grouped, "gnn_node", "none", "32"))[0]
    mlp_none_h32 = mean_std(rollout_values(grouped, "mlp_global", "none", "32"))[0]
    mlp_cov_cond = mean_std(diagnostic_values(grouped, "mlp_global", "covariance", "covariance_condition_number"))[0]
    mlp_none_cond = mean_std(diagnostic_values(grouped, "mlp_global", "none", "covariance_condition_number"))[0]
    gnn_cov_cond = mean_std(diagnostic_values(grouped, "gnn_node", "covariance", "covariance_condition_number"))[0]
    gnn_none_cond = mean_std(diagnostic_values(grouped, "gnn_node", "none", "covariance_condition_number"))[0]

    if failures or missing or not graph_ok:
        return "Cycle 2 is not justified until the pilot is complete and graph metadata is valid."
    graph_specific = (
        finite(graph_h32)
        and finite(perm_h32)
        and finite(rand_h32)
        and graph_h32 < 0.9 * perm_h32
        and graph_h32 < 0.9 * rand_h32
    )
    graph_smoothing = (
        finite(graph_h32)
        and finite(gnn_none_h32)
        and graph_h32 < gnn_none_h32
        and finite(perm_h32)
        and finite(rand_h32)
        and abs(graph_h32 - rand_h32) / max(abs(graph_h32), 1e-12) < 0.10
    )
    architecture_effect = finite(gnn_none_h32) and finite(mlp_none_h32) and gnn_none_h32 < mlp_none_h32
    conditioning_effect = (
        finite(mlp_cov_cond) and finite(mlp_none_cond) and mlp_cov_cond < mlp_none_cond
    ) or (
        finite(gnn_cov_cond) and finite(gnn_none_cond) and gnn_cov_cond < gnn_none_cond
    )

    if graph_specific:
        return (
            "Cycle 2 multi-molecule expansion is justified for GNN graph, permuted_graph, "
            "random_graph, plus GNN none controls. Evidence points most directly to true graph specificity."
        )
    if graph_smoothing:
        return (
            "Cycle 2 multi-molecule expansion is justified, but as a controlled test rather than a graph-specific claim. "
            "Expand GNN none, GNN graph, GNN permuted_graph, GNN random_graph, and MLP none. "
            "Current evidence points to a strong GNN graph-prior/generic smoothing effect beyond GNN, "
            "with weak true graph specificity because permuted/random graph controls are close to the true graph."
        )
    if architecture_effect:
        return (
            "Cycle 2 is justified as an architecture-controlled expansion: expand MLP none, GNN none, "
            "and GNN graph controls. Current evidence points more to architecture/generic smoothing than true graph specificity."
        )
    if conditioning_effect:
        return (
            "Cycle 2 should expand covariance/SIGReg conditioning controls before graph claims. "
            "Evidence points primarily to Euclidean conditioning."
        )
    return (
        "Cycle 2 is only weakly justified; expand a minimal aspirin-to-multimolecule panel with all controls. "
        "Current evidence is inconclusive between generic smoothing and noise."
    )


def build_report(results: dict[str, Any]) -> str:
    runs = results.get("runs", {})
    grouped = group_successful_runs(runs)
    missing, failures, nan_runs = check_expected(runs)
    graph_ok = graph_metadata_ok(runs)
    success_count = sum(1 for run in runs.values() if run.get("status") == "ok" and not run.get("failure_flag"))
    failure_count = len(failures)

    lines = [
        "# Cycle 1 Aspirin Pilot Report",
        "",
        f"Results file: `{DEFAULT_RESULTS.relative_to(ROOT)}`",
        f"Schema version: `{results.get('schema_version')}`",
        f"Success/failure count: {success_count} ok / {failure_count} failed",
        f"Missing expected runs: {missing or 'none'}",
        f"Runs with non-finite diagnostics: {nan_runs or 'none'}",
        f"Graph prior metadata node-wise: {'YES' if graph_ok else 'NO'}",
        "",
        "## Rollout Error Mean +/- Std",
        "",
        "| encoder | prior | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for encoder, priors in [("mlp_global", MLP_PRIORS), ("gnn_node", GNN_PRIORS)]:
        for prior in priors:
            cells = [fmt_mean_std(rollout_values(grouped, encoder, prior, horizon)) for horizon in HORIZONS]
            lines.append(f"| {encoder} | {prior} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Diagnostics Mean +/- Std",
        "",
        "| encoder | prior | effective_rank | condition_number | projection_gaussianity | prior_loss_mean |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for encoder, priors in [("mlp_global", MLP_PRIORS), ("gnn_node", GNN_PRIORS)]:
        for prior in priors:
            cells = [fmt_mean_std(diagnostic_values(grouped, encoder, prior, metric)) for metric in DIAGNOSTICS]
            lines.append(f"| {encoder} | {prior} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Euclidean Conditioning Gains",
        "",
        comparison_line(grouped, "MLP none vs MLP covariance condition number", ("mlp_global", "none"), ("mlp_global", "covariance"), "covariance_condition_number"),
        comparison_line(grouped, "MLP none vs MLP SIGReg condition number", ("mlp_global", "none"), ("mlp_global", "sigreg"), "covariance_condition_number"),
        comparison_line(grouped, "GNN none vs GNN covariance condition number", ("gnn_node", "none"), ("gnn_node", "covariance"), "covariance_condition_number"),
        comparison_line(grouped, "GNN none vs GNN SIGReg condition number", ("gnn_node", "none"), ("gnn_node", "sigreg"), "covariance_condition_number"),
        "",
        "## Graph Specificity",
        "",
        comparison_line(grouped, "GNN graph vs GNN permuted_graph H=32", ("gnn_node", "graph"), ("gnn_node", "permuted_graph"), "32"),
        comparison_line(grouped, "GNN graph vs GNN random_graph H=32", ("gnn_node", "graph"), ("gnn_node", "random_graph"), "32"),
        "",
        "## Architecture Gain",
        "",
        comparison_line(grouped, "MLP none vs GNN none H=32", ("mlp_global", "none"), ("gnn_node", "none"), "32"),
        "",
        "## Graph Prior Gain Beyond GNN",
        "",
        comparison_line(grouped, "GNN none vs GNN graph H=32", ("gnn_node", "none"), ("gnn_node", "graph"), "32"),
        "",
        "## Recommendation",
        "",
        recommend(grouped, graph_ok, failures, missing),
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Cycle 1 aspirin pilot results.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    results = load_json(args.results)
    report = build_report(results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
