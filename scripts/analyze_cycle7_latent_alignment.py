from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

RESULT_PATHS = {
    "cycle2_rmd17_multimolecule": ROOT
    / "experiments/results/cycle2_rmd17_multimolecule/cycle2_rmd17_multimolecule_results.json",
    "cycle3_ho_networks": ROOT / "experiments/results/cycle3_ho_networks/cycle3_ho_networks_results.json",
    "cycle4_ho_lambda_robustness": ROOT
    / "experiments/results/cycle4_ho_lambda_robustness/cycle4_ho_lambda_robustness_results.json",
    "cycle5_ho_lattice_bridge": ROOT
    / "experiments/results/cycle5_ho_lattice_bridge/cycle5_ho_lattice_bridge_results.json",
}

REPORT_PATH = ROOT / "analysis_out/CYCLE7_LATENT_ALIGNMENT_REPORT.md"
SUMMARY_CSV = ROOT / "analysis_out/cycle7_latent_alignment_summary.csv"

GRAPH_PRIORS = {"graph", "permuted_graph", "random_graph"}
HORIZONS = ("16", "32")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def runs_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    runs = payload.get("runs", {})
    return list(runs.values()) if isinstance(runs, dict) else list(runs)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean(values: list[float]) -> float:
    values = [float(value) for value in values if finite(value)]
    return sum(values) / len(values) if values else float("nan")


def fmt(value: Any, digits: int = 4) -> str:
    if not finite(value):
        return "NA"
    value = float(value)
    if abs(value) >= 1:
        return f"{value:.{digits}f}"
    return f"{value:.{max(digits, 6)}f}"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def domain_item(config: dict[str, Any]) -> tuple[str, str]:
    if "topology" in config:
        return "HO", str(config.get("topology", ""))
    return "rMD17", str(config.get("molecule", ""))


def prior_weight_for(source: str, config: dict[str, Any]) -> float:
    if source == "cycle3_ho_networks":
        return 0.1
    return float(config.get("prior_weight", 0.1))


def checkpoint_path_from_run(run: dict[str, Any]) -> Path | None:
    for key in ("checkpoint_path", "checkpoint", "model_checkpoint"):
        value = run.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            return path if path.is_absolute() else ROOT / path
    config = run.get("config", {})
    for key in ("checkpoint_path", "checkpoint", "model_checkpoint"):
        value = config.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            return path if path.is_absolute() else ROOT / path
    return None


def discover_checkpoint_inventory() -> dict[str, Any]:
    checkpoint_files = sorted(ROOT.glob("checkpoints/**/*.pt"))
    names = [path.name for path in checkpoint_files]
    encoder_tokens = Counter()
    for name in names:
        if "_gnn_" in name or "_gnn_node_" in name:
            encoder_tokens["gnn_or_gnn_node"] += 1
        elif "_flat_" in name:
            encoder_tokens["flat"] += 1
        elif "_mlp_" in name:
            encoder_tokens["mlp"] += 1
        else:
            encoder_tokens["other"] += 1
    return {
        "n_checkpoint_files": len(checkpoint_files),
        "encoder_tokens": dict(encoder_tokens),
        "examples": [rel(path) for path in checkpoint_files[:8]],
    }


def build_summary_rows(results: dict[str, dict[str, Any]], torch_available: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, payload in results.items():
        for run in runs_from_payload(payload):
            config = run.get("config", {})
            prior = config.get("prior")
            if run.get("status") != "ok" or config.get("encoder") != "gnn_node" or prior not in GRAPH_PRIORS:
                continue
            domain, item = domain_item(config)
            checkpoint = checkpoint_path_from_run(run)
            if checkpoint is None:
                status = "missing_checkpoint_path"
                reason = "Cycle result does not record checkpoint_path; trained node-wise H cannot be reconstructed without retraining."
                checkpoint_text = ""
            elif not checkpoint.exists():
                status = "missing_checkpoint_file"
                reason = "Cycle result records checkpoint_path, but file is absent."
                checkpoint_text = rel(checkpoint)
            elif not torch_available:
                status = "torch_unavailable"
                reason = "Checkpoint exists, but this Python environment cannot import torch to load the model."
                checkpoint_text = rel(checkpoint)
            else:
                status = "checkpoint_available_not_loaded"
                reason = (
                    "Checkpoint loading is intentionally not attempted here because no requested Cycle 2-5 "
                    "node-wise checkpoint is referenced by the current result payloads."
                )
                checkpoint_text = rel(checkpoint)
            row = {
                "source": source,
                "run_name": run.get("run_name", config.get("run_name", "")),
                "domain": domain,
                "item": item,
                "prior": prior,
                "prior_weight": prior_weight_for(source, config),
                "seed": config.get("seed", ""),
                "checkpoint_status": status,
                "checkpoint_path": checkpoint_text,
                "skip_reason": reason,
                "D_H": "",
                "D_dH": "",
                "D_H_norm": "",
                "D_dH_norm": "",
                "R_low_H_2": "",
                "R_low_H_4": "",
                "R_low_H_8": "",
                "R_low_dH_2": "",
                "R_low_dH_4": "",
                "R_low_dH_8": "",
            }
            rows.append(row)
    return rows


def observed_specificity(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, payload in results.items():
        grouped: dict[tuple[str, str, float, str], list[dict[str, Any]]] = defaultdict(list)
        for run in runs_from_payload(payload):
            config = run.get("config", {})
            prior = config.get("prior")
            if run.get("status") != "ok" or config.get("encoder") != "gnn_node" or prior not in GRAPH_PRIORS:
                continue
            domain, item = domain_item(config)
            grouped[(domain, item, prior_weight_for(source, config), str(prior))].append(run)
        for domain, item, prior_weight, prior in list(grouped):
            if prior != "graph":
                continue
            graph_runs = grouped[(domain, item, prior_weight, "graph")]
            for control in ("permuted_graph", "random_graph"):
                control_runs = grouped.get((domain, item, prior_weight, control), [])
                if not control_runs:
                    continue
                for horizon in HORIZONS:
                    graph_mean = mean(
                        [
                            float(run.get("diagnostics", {}).get("rollout_errors", {}).get(horizon))
                            for run in graph_runs
                            if finite(run.get("diagnostics", {}).get("rollout_errors", {}).get(horizon))
                        ]
                    )
                    control_mean = mean(
                        [
                            float(run.get("diagnostics", {}).get("rollout_errors", {}).get(horizon))
                            for run in control_runs
                            if finite(run.get("diagnostics", {}).get("rollout_errors", {}).get(horizon))
                        ]
                    )
                    rows.append(
                        {
                            "source": source,
                            "domain": domain,
                            "item": item,
                            "prior_weight": prior_weight,
                            "horizon": int(horizon),
                            "control": control,
                            "graph_mean": graph_mean,
                            "control_mean": control_mean,
                            "S_graph": control_mean - graph_mean
                            if finite(graph_mean) and finite(control_mean)
                            else float("nan"),
                        }
                    )
    return rows


def write_summary_csv(rows: list[dict[str, Any]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "run_name",
        "domain",
        "item",
        "prior",
        "prior_weight",
        "seed",
        "checkpoint_status",
        "checkpoint_path",
        "skip_reason",
        "D_H",
        "D_dH",
        "D_H_norm",
        "D_dH_norm",
        "R_low_H_2",
        "R_low_H_4",
        "R_low_H_8",
        "R_low_dH_2",
        "R_low_dH_4",
        "R_low_dH_8",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def build_report(
    summary_rows: list[dict[str, Any]],
    specificity_rows: list[dict[str, Any]],
    inventory: dict[str, Any],
    torch_available: bool,
) -> str:
    status_counts = Counter(row["checkpoint_status"] for row in summary_rows)
    source_counts = Counter(row["source"] for row in summary_rows)
    source_table = [[source, count] for source, count in sorted(source_counts.items())]
    status_table = [[status, count] for status, count in sorted(status_counts.items())]
    h32_positive = [
        row
        for row in specificity_rows
        if int(row["horizon"]) == 32 and finite(row["S_graph"]) and float(row["S_graph"]) > 0
    ]
    interesting = [
        row
        for row in specificity_rows
        if int(row["horizon"]) == 32
        and (
            (row["source"] == "cycle5_ho_lattice_bridge" and row["item"] == "lattice")
            or row["source"] == "cycle2_rmd17_multimolecule"
        )
    ]
    specificity_table = [
        [
            row["source"],
            row["domain"],
            row["item"],
            f"{float(row['prior_weight']):g}",
            row["control"],
            fmt(row["S_graph"]),
        ]
        for row in interesting[:20]
    ]
    lines = [
        "# Cycle 7 Latent Alignment Diagnostics",
        "",
        "Analysis only: no training, ISO17, or rMD17 top-up commands are run.",
        "",
        "## Checkpoint Audit",
        "",
        f"Python torch import available: {'YES' if torch_available else 'NO'}",
        f"Checkpoint files discovered under `checkpoints/`: {inventory['n_checkpoint_files']}",
        "Checkpoint filename encoder tokens: "
        + ", ".join(f"{key}={value}" for key, value in sorted(inventory["encoder_tokens"].items())),
        "",
        "Eligible Cycle 2-5 graph-prior node-wise runs:",
    ]
    lines.extend(markdown_table(["source", "eligible runs"], source_table))
    lines.extend(["", "Checkpoint status:"])
    lines.extend(markdown_table(["status", "runs"], status_table))
    lines.extend(
        [
            "",
            "Result: no requested Cycle 2/3/4/5 node-wise GNN run records a checkpoint path. Existing `.pt` files are not referenced by these cycle result payloads, and the visible checkpoint naming is dominated by older `flat`/spectral runs rather than the node-wise GNN Cycle 2-5 runs.",
            "",
            "## Observed Specificity Available For Alignment",
            "",
            f"H=32 positive true-graph specificity comparisons in existing results: {len(h32_positive)}.",
        ]
    )
    lines.extend(
        markdown_table(
            ["source", "domain", "item", "lambda", "control", "S_graph H=32"],
            specificity_table,
        )
    )
    lines.extend(
        [
            "",
            "## Diagnostic Answers",
            "",
            "- Does latent-space `D_dH` explain high-lambda lattice specificity? Not with the current artifacts. Cycle 5 contains the high-lambda lattice specificity result, but the trained node-wise models were not checkpointed, so `H_t` and `H_{t+1}` cannot be recovered without retraining.",
            "- Is latent alignment more predictive than raw-coordinate alignment from Cycle 6? Unknown. The needed latent-state measurements are unavailable for the requested cycles.",
            "- Does true graph produce lower latent Dirichlet energy than permuted/random controls? Not evaluated. The summary CSV records every eligible run as missing a recoverable checkpoint.",
            "- Should the paper include latent alignment as mechanism or only as negative diagnostic? Do not include it as positive mechanism from current artifacts. At most, mention that raw-coordinate alignment was a negative diagnostic and that latent-state alignment would require checkpointed reruns or future checkpointing.",
            "",
            "## Decision",
            "",
            "Cycle 7 is blocked by missing checkpoint artifacts, not by a statistical result. The paper should not claim a latent-alignment mechanism unless node-wise checkpoints are produced by a future analysis-only rerun of saved models or by newly checkpointed experiments.",
            "",
            "## Files",
            "",
            f"- `{rel(SUMMARY_CSV)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Cycle 7 latent graph-dynamics alignment availability.")
    parser.parse_args()
    results = {source: load_json(path) for source, path in RESULT_PATHS.items() if path.exists()}
    torch_available = importlib.util.find_spec("torch") is not None
    inventory = discover_checkpoint_inventory()
    summary_rows = build_summary_rows(results, torch_available=torch_available)
    specificity_rows = observed_specificity(results)
    write_summary_csv(summary_rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        build_report(summary_rows, specificity_rows, inventory, torch_available=torch_available),
        encoding="utf-8",
    )
    print(f"Wrote {rel(REPORT_PATH)}")
    print(f"Wrote {rel(SUMMARY_CSV)} ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
