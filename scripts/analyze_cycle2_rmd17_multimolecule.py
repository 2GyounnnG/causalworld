from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "experiments" / "results" / "cycle2_rmd17_multimolecule" / "cycle2_rmd17_multimolecule_results.json"
DEFAULT_REPORT = ROOT / "analysis_out" / "CYCLE2_RMD17_MULTIMOLECULE_REPORT.md"
MOLECULES = ["aspirin", "ethanol", "malonaldehyde", "naphthalene", "toluene"]
HORIZONS = ["1", "2", "4", "8", "16", "32"]
SETTINGS = [
    ("mlp_global", "none"),
    ("gnn_node", "none"),
    ("gnn_node", "graph"),
    ("gnn_node", "permuted_graph"),
    ("gnn_node", "random_graph"),
]
SEEDS = [0, 1, 2]
EXPECTED = {
    (molecule, encoder, prior, seed)
    for molecule in MOLECULES
    for encoder, prior in SETTINGS
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
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return float(array.mean()), std, int(array.size)


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


def group_successful_runs(runs: dict[str, Any]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs.values():
        if run.get("status") != "ok" or run.get("failure_flag"):
            continue
        config = run.get("config", {})
        grouped[
            (
                str(config.get("molecule")),
                str(config.get("encoder")),
                str(config.get("prior")),
            )
        ].append(run)
    return grouped


def rollout_values(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    molecule: str,
    encoder: str,
    prior: str,
    horizon: str,
) -> list[float]:
    return [
        float(run.get("diagnostics", {}).get("rollout_errors", {}).get(horizon, float("nan")))
        for run in grouped.get((molecule, encoder, prior), [])
    ]


def rollout_mean(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    molecule: str,
    encoder: str,
    prior: str,
    horizon: str = "32",
) -> float:
    return mean_std(rollout_values(grouped, molecule, encoder, prior, horizon))[0]


def compare_cell(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    molecule: str,
    left: tuple[str, str],
    right: tuple[str, str],
    horizon: str,
) -> str:
    left_mean = rollout_mean(grouped, molecule, *left, horizon=horizon)
    right_mean = rollout_mean(grouped, molecule, *right, horizon=horizon)
    return f"{fmt(left_mean)} -> {fmt(right_mean)} ({fmt_delta(pct_improvement(left_mean, right_mean))})"


def specificity_cell(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    molecule: str,
    control_prior: str,
    horizon: str,
) -> str:
    graph_mean = rollout_mean(grouped, molecule, "gnn_node", "graph", horizon=horizon)
    control_mean = rollout_mean(grouped, molecule, "gnn_node", control_prior, horizon=horizon)
    return f"{fmt(graph_mean)} vs {fmt(control_mean)} ({fmt_delta(pct_lower(graph_mean, control_mean))})"


def check_expected(runs: dict[str, Any]) -> tuple[list[tuple[str, str, str, int]], list[str], list[str]]:
    observed = set()
    failures = []
    nan_runs = []
    for name, run in runs.items():
        config = run.get("config", {})
        observed.add(
            (
                str(config.get("molecule")),
                str(config.get("encoder")),
                str(config.get("prior")),
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


def molecule_metrics(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    molecule: str,
    horizon: str = "32",
) -> dict[str, float]:
    mlp_none = rollout_mean(grouped, molecule, "mlp_global", "none", horizon=horizon)
    gnn_none = rollout_mean(grouped, molecule, "gnn_node", "none", horizon=horizon)
    graph = rollout_mean(grouped, molecule, "gnn_node", "graph", horizon=horizon)
    permuted = rollout_mean(grouped, molecule, "gnn_node", "permuted_graph", horizon=horizon)
    random = rollout_mean(grouped, molecule, "gnn_node", "random_graph", horizon=horizon)
    return {
        "mlp_none": mlp_none,
        "gnn_none": gnn_none,
        "graph": graph,
        "permuted_graph": permuted,
        "random_graph": random,
        "architecture_gain_pct": pct_improvement(mlp_none, gnn_none),
        "graph_gain_pct": pct_improvement(gnn_none, graph),
        "permuted_gain_pct": pct_improvement(gnn_none, permuted),
        "random_gain_pct": pct_improvement(gnn_none, random),
        "specificity_vs_permuted_pct": pct_lower(graph, permuted),
        "specificity_vs_random_pct": pct_lower(graph, random),
    }


def classify_molecule(metrics: dict[str, float]) -> tuple[str, str]:
    required = [
        "mlp_none",
        "gnn_none",
        "graph",
        "permuted_graph",
        "random_graph",
    ]
    if any(not finite(metrics[key]) for key in required):
        return "incomplete", "missing one or more H=32 means"

    arch_gain = metrics["architecture_gain_pct"]
    graph_gain = metrics["graph_gain_pct"]
    perm_gain = metrics["permuted_gain_pct"]
    rand_gain = metrics["random_gain_pct"]
    spec_perm = metrics["specificity_vs_permuted_pct"]
    spec_rand = metrics["specificity_vs_random_pct"]

    if graph_gain < -5.0:
        return "prior-hurts / mismatch", f"graph is worse than GNN none at H=32 ({fmt_delta(graph_gain)})"
    if graph_gain >= 5.0 and spec_perm >= 10.0 and spec_rand >= 10.0:
        return "graph-specific", "true graph beats both permuted and random controls by >=10%"
    if graph_gain >= 5.0 and abs(spec_perm) <= 10.0 and abs(spec_rand) <= 10.0:
        return "random-control-equivalent", "true, permuted, and random graph controls are within +/-10%"
    if graph_gain >= 10.0 and (perm_gain >= 5.0 or rand_gain >= 5.0):
        return "graph-smoothing-dominated", "graph controls improve over GNN none without true-graph specificity"
    if arch_gain >= 10.0 and graph_gain < 10.0:
        return "architecture-dominated", "GNN none improves over MLP none more than graph improves over GNN none"
    if graph_gain < 5.0 and arch_gain < 5.0:
        return "random-control-equivalent", "effects are below the 5% practical threshold"
    return "architecture-dominated", "dominant effect is not graph-specific under the H=32 thresholds"


def top_up_recommendations(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    classifications: dict[str, tuple[str, str]],
) -> tuple[list[str], list[str]]:
    top_up = []
    reasons = []
    for molecule in MOLECULES:
        label, reason = classifications[molecule]
        metrics = molecule_metrics(grouped, molecule)
        graph_values = rollout_values(grouped, molecule, "gnn_node", "graph", "32")
        graph_mean, graph_std, n = mean_std(graph_values)
        cv = graph_std / abs(graph_mean) if finite(graph_mean) and graph_mean != 0.0 else float("nan")
        near_specificity_boundary = (
            finite(metrics["specificity_vs_permuted_pct"])
            and finite(metrics["specificity_vs_random_pct"])
            and (
                abs(metrics["specificity_vs_permuted_pct"] - 10.0) <= 5.0
                or abs(metrics["specificity_vs_random_pct"] - 10.0) <= 5.0
            )
        )
        unstable = finite(cv) and cv >= 0.20
        if label in {"graph-specific", "prior-hurts / mismatch"} or near_specificity_boundary or unstable:
            top_up.append(molecule)
            details = [label, reason]
            if near_specificity_boundary:
                details.append("near graph-specificity threshold")
            if unstable:
                details.append(f"graph H=32 CV={cv:.2f} over n={n}")
            reasons.append(f"- {molecule}: " + "; ".join(details))
    return top_up, reasons


def recommendation(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
    classifications: dict[str, tuple[str, str]],
    missing: list[tuple[str, str, str, int]],
    failures: list[str],
    graph_ok: bool,
) -> str:
    top_up, top_up_reasons = top_up_recommendations(grouped, classifications)
    labels = [classifications[molecule][0] for molecule in MOLECULES]
    n_specific = labels.count("graph-specific")
    n_equivalent = labels.count("random-control-equivalent")
    n_smoothing = labels.count("graph-smoothing-dominated")

    lines = []
    if missing or failures:
        lines.append("The sweep is incomplete, so recommendations are provisional.")
    if not graph_ok:
        lines.append("Do not interpret graph-specificity until graph-prior metadata is valid.")

    if top_up:
        lines.append("Top up from 3 to 5 seeds: " + ", ".join(top_up) + ".")
        lines.extend(top_up_reasons)
    else:
        lines.append("Top up from 3 to 5 seeds: none recommended from current H=32 thresholds.")

    if n_specific >= 3:
        lines.append("True graph specificity is supported across the panel.")
    elif n_specific > 0:
        lines.append("True graph specificity is molecule-dependent rather than panel-wide.")
    elif n_equivalent + n_smoothing >= 3:
        lines.append("True graph specificity is not supported; the evidence favors generic graph smoothing or control equivalence.")
    else:
        lines.append("True graph specificity is inconclusive from the current 3-seed panel.")

    if n_specific >= 2:
        lines.append("Cycle 3 recommendation: go to ISO17 next, using the specific molecules as anchors.")
    elif n_smoothing + n_equivalent >= 3:
        lines.append("Cycle 3 recommendation: run controlled HO networks before ISO17 to separate graph smoothing from true graph semantics.")
    else:
        lines.append("Cycle 3 recommendation: top up the flagged rMD17 molecules before choosing ISO17 versus controlled HO networks.")
    return "\n".join(lines)


def build_report(results: dict[str, Any], results_path: Path) -> str:
    runs = results.get("runs", {})
    grouped = group_successful_runs(runs)
    missing, failures, nan_runs = check_expected(runs)
    graph_ok = graph_metadata_ok(runs)
    success_count = sum(1 for run in runs.values() if run.get("status") == "ok" and not run.get("failure_flag"))
    failure_count = len(failures)

    lines = [
        "# Cycle 2 rMD17 Multi-Molecule Attribution Report",
        "",
        f"Results file: `{results_path.relative_to(ROOT) if results_path.is_relative_to(ROOT) else results_path}`",
        f"Schema version: `{results.get('schema_version')}`",
        f"Success/failure count: {success_count} ok / {failure_count} failed",
        f"Missing expected runs: {len(missing)}",
        f"Runs with non-finite rollout diagnostics: {nan_runs or 'none'}",
        f"Graph prior metadata node-wise: {'YES' if graph_ok else 'NO'}",
        "",
        "## Rollout Error Mean +/- Std",
        "",
        "| molecule | encoder | prior | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for molecule in MOLECULES:
        for encoder, prior in SETTINGS:
            cells = [fmt_mean_std(rollout_values(grouped, molecule, encoder, prior, horizon)) for horizon in HORIZONS]
            lines.append(f"| {molecule} | {encoder} | {prior} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Graph Prior Gain Beyond GNN None",
        "",
        "| molecule | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for molecule in MOLECULES:
        cells = [
            compare_cell(grouped, molecule, ("gnn_node", "none"), ("gnn_node", "graph"), horizon)
            for horizon in HORIZONS
        ]
        lines.append(f"| {molecule} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Graph Specificity: Graph vs Permuted Graph",
        "",
        "| molecule | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for molecule in MOLECULES:
        cells = [specificity_cell(grouped, molecule, "permuted_graph", horizon) for horizon in HORIZONS]
        lines.append(f"| {molecule} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Graph Specificity: Graph vs Random Graph",
        "",
        "| molecule | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for molecule in MOLECULES:
        cells = [specificity_cell(grouped, molecule, "random_graph", horizon) for horizon in HORIZONS]
        lines.append(f"| {molecule} | " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "## Architecture Gain: MLP None vs GNN None",
        "",
        "| molecule | H=1 | H=2 | H=4 | H=8 | H=16 | H=32 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for molecule in MOLECULES:
        cells = [
            compare_cell(grouped, molecule, ("mlp_global", "none"), ("gnn_node", "none"), horizon)
            for horizon in HORIZONS
        ]
        lines.append(f"| {molecule} | " + " | ".join(cells) + " |")

    classifications: dict[str, tuple[str, str]] = {}
    lines.extend([
        "",
        "## Molecule Classification",
        "",
        "| molecule | class | arch gain H=32 | graph gain H=32 | graph vs permuted H=32 | graph vs random H=32 | rationale |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for molecule in MOLECULES:
        metrics = molecule_metrics(grouped, molecule)
        label, reason = classify_molecule(metrics)
        classifications[molecule] = (label, reason)
        lines.append(
            f"| {molecule} | {label} | "
            f"{fmt_delta(metrics['architecture_gain_pct'])} | "
            f"{fmt_delta(metrics['graph_gain_pct'])} | "
            f"{fmt_delta(metrics['specificity_vs_permuted_pct'])} | "
            f"{fmt_delta(metrics['specificity_vs_random_pct'])} | "
            f"{reason} |"
        )

    lines.extend([
        "",
        "## Recommendation",
        "",
        recommendation(grouped, classifications, missing, failures, graph_ok),
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Cycle 2 rMD17 multi-molecule attribution sweep.")
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
