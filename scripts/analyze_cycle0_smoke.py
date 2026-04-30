from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "experiments" / "results" / "cycle0_aspirin_smoke" / "cycle0_aspirin_smoke_results.json"
DEFAULT_REPORT = ROOT / "analysis_out" / "CYCLE0_SMOKE_REPORT.md"
EXPECTED_RUNS = {
    "mlp_none",
    "mlp_covariance",
    "mlp_sigreg",
    "gnn_none",
    "gnn_graph",
    "gnn_permuted_graph",
    "gnn_random_graph",
}
EXPECTED_HORIZONS = {"1", "2", "4", "8", "16", "32"}
EXPECTED_DIAGNOSTICS = {
    "rollout_errors",
    "effective_rank",
    "covariance_condition_number",
    "projection_gaussianity_statistic",
    "graph_stationarity",
    "final_train_loss",
    "prior_loss_mean",
    "nan_detected",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def finite_or_marked_unavailable(value: Any) -> bool:
    if isinstance(value, dict) and value.get("available") is False:
        return True
    if isinstance(value, dict):
        return all(finite_or_marked_unavailable(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_or_marked_unavailable(v) for v in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def run_has_all_diagnostics(run: dict[str, Any]) -> bool:
    diagnostics = run.get("diagnostics", {})
    if not EXPECTED_DIAGNOSTICS.issubset(diagnostics.keys()):
        return False
    rollout = diagnostics.get("rollout_errors", {})
    return EXPECTED_HORIZONS.issubset(set(rollout.keys()))


def build_report(results: dict[str, Any]) -> str:
    runs = results.get("runs", {})
    missing_runs = sorted(EXPECTED_RUNS.difference(runs.keys()))
    failed_runs = sorted(name for name, run in runs.items() if run.get("failure_flag") or run.get("status") != "ok")
    nonfinite_runs = sorted(
        name
        for name, run in runs.items()
        if not finite_or_marked_unavailable(run.get("diagnostics", {}))
        or bool(run.get("diagnostics", {}).get("nan_detected", False))
    )
    missing_diag_runs = sorted(name for name, run in runs.items() if not run_has_all_diagnostics(run))
    graph_runs = [
        run
        for run in runs.values()
        if run.get("status") == "ok"
        and run.get("config", {}).get("prior") in {"graph", "permuted_graph", "random_graph"}
    ]
    graph_nodewise = bool(graph_runs) and all(
        run.get("prior_implementation", {}).get("graph_prior_nodewise") is True
        and run.get("prior_implementation", {}).get("graph_prior_form") == "nodewise_trace_HtLH"
        and run.get("prior_implementation", {}).get("uses_latent_projected_laplacian") is False
        for run in graph_runs
    )
    mlp_graph_disabled = all(
        not (
            run.get("config", {}).get("encoder") == "mlp_global"
            and run.get("config", {}).get("prior") in {"graph", "permuted_graph", "random_graph"}
        )
        for run in runs.values()
    )
    schema_stable = (
        results.get("schema_version") == "cycle0_smoke_v1"
        and not missing_runs
        and not missing_diag_runs
    )
    all_finished = not missing_runs and not failed_runs and not nonfinite_runs
    cycle1_ready = all_finished and graph_nodewise and mlp_graph_disabled and not missing_diag_runs and schema_stable

    lines = [
        "# Cycle 0 Smoke Report",
        "",
        f"Results file: `{DEFAULT_RESULTS.relative_to(ROOT)}`",
        f"Schema version: `{results.get('schema_version')}`",
        f"Runs observed: {len(runs)} / {len(EXPECTED_RUNS)}",
        "",
        "## Required Questions",
        "",
        f"- Did all runs finish without NaN? {'YES' if all_finished else 'NO'}",
        f"- Does graph prior use node-wise H^T L H? {'YES' if graph_nodewise else 'NO'}",
        f"- Are MLP graph priors disabled unless mathematically defined? {'YES' if mlp_graph_disabled else 'NO'}",
        f"- Are all diagnostics saved? {'YES' if not missing_diag_runs else 'NO'}",
        f"- Is the JSON schema stable? {'YES' if schema_stable else 'NO'}",
        f"- Is Cycle 1 ready? {'YES' if cycle1_ready else 'NO'}",
        "",
        "## Run Table",
        "",
        "| run | encoder | prior | status | final_loss | prior_loss_mean | H=1 | H=32 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for name in sorted(runs):
        run = runs[name]
        diagnostics = run.get("diagnostics", {})
        rollout = diagnostics.get("rollout_errors", {})
        lines.append(
            "| {name} | {encoder} | {prior} | {status} | {loss:.6g} | {prior_loss:.6g} | {h1:.6g} | {h32:.6g} |".format(
                name=name,
                encoder=run.get("config", {}).get("encoder"),
                prior=run.get("config", {}).get("prior"),
                status=run.get("status"),
                loss=float(diagnostics.get("final_train_loss", float("nan"))),
                prior_loss=float(diagnostics.get("prior_loss_mean", float("nan"))),
                h1=float(rollout.get("1", float("nan"))),
                h32=float(rollout.get("32", float("nan"))),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Missing runs: {missing_runs or 'none'}",
            f"- Failed runs: {failed_runs or 'none'}",
            f"- Non-finite diagnostic runs: {nonfinite_runs or 'none'}",
            f"- Runs missing diagnostics: {missing_diag_runs or 'none'}",
            "- Graph stationarity is explicitly marked unavailable in this smoke path.",
            "- Cycle 1 should not start from this script; this report only gates readiness.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Cycle 0 smoke results.")
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
