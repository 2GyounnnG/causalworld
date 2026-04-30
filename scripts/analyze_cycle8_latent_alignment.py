from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = (
    ROOT
    / "experiments/results/cycle8_checkpointed_lattice_latent_alignment/cycle8_checkpointed_lattice_latent_alignment_results.json"
)
DEFAULT_REPORT = ROOT / "analysis_out/CYCLE8_CHECKPOINTED_LATENT_ALIGNMENT_REPORT.md"
DEFAULT_SUMMARY_CSV = ROOT / "analysis_out/cycle8_latent_alignment_summary.csv"

PRIORS = ["graph", "permuted_graph", "random_graph"]
SEEDS = [0, 1, 2, 3, 4]
HORIZONS = ["1", "2", "4", "8", "16", "32"]


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


def mean_std(values: list[float]) -> tuple[float, float, int]:
    array = np.asarray([value for value in values if finite(value)], dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan"), 0
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0, int(array.size)


def fmt(value: Any, digits: int = 4) -> str:
    if not finite(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def fmt_mean_std(values: list[float], digits: int = 4) -> str:
    mean, std, n = mean_std(values)
    return f"{fmt(mean, digits)} +/- {fmt(std, digits)} (n={n})"


def pearson(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 3 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_values[j] == sorted_values[i]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 3:
        return float("nan")
    return pearson(rankdata(x_arr[mask]), rankdata(y_arr[mask]))


def dirichlet_batch(matrices: torch.Tensor, laplacians: torch.Tensor) -> tuple[float, float]:
    if laplacians.ndim == 2:
        laplacians = laplacians.unsqueeze(0).expand(matrices.shape[0], -1, -1)
    values = torch.einsum("sni,snm,smi->s", matrices, laplacians, matrices)
    norms = torch.sum(matrices * matrices, dim=(1, 2)).clamp_min(1e-12)
    return float(values.mean().item()), float((values / norms).mean().item())


def low_frequency_ratio_batch(matrices: torch.Tensor, laplacians: torch.Tensor, k: int) -> float:
    ratios: list[float] = []
    if laplacians.ndim == 2:
        laplacians = laplacians.unsqueeze(0).expand(matrices.shape[0], -1, -1)
    for matrix, laplacian in zip(matrices, laplacians):
        eigvals, eigvecs = torch.linalg.eigh(laplacian)
        order = torch.argsort(eigvals)
        eigvecs = eigvecs[:, order]
        kk = min(k, eigvecs.shape[1] - 1)
        coeff = eigvecs.T @ matrix
        energy = torch.sum(coeff * coeff, dim=1)
        total = torch.sum(energy).item()
        if total > 1e-12:
            ratios.append(float((torch.sum(energy[:kk]) / torch.sum(energy)).item()))
    return float(np.mean(ratios)) if ratios else float("nan")


def safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def artifact_metrics(artifact: dict[str, Any]) -> dict[str, float]:
    delta_h = artifact["Delta_H"].to(dtype=torch.float32)
    h_t = artifact["H_t"].to(dtype=torch.float32)
    true_l = artifact["true_laplacian"].to(dtype=torch.float32)
    prior_l = artifact["prior_laplacians"].to(dtype=torch.float32)
    random_l = artifact["random_laplacians"].to(dtype=torch.float32)
    prior_perm = artifact["prior_permutation_indices"].to(dtype=torch.long)
    control_perm = artifact["control_permutation_indices"].to(dtype=torch.long)

    prior_delta = delta_h
    if str(artifact.get("prior")) == "permuted_graph":
        prior_delta = torch.stack([matrix[perm] for matrix, perm in zip(delta_h, prior_perm)], dim=0)
    perm_delta = torch.stack([matrix[perm] for matrix, perm in zip(delta_h, control_perm)], dim=0)

    d_true, d_true_norm = dirichlet_batch(delta_h, true_l)
    d_prior, d_prior_norm = dirichlet_batch(prior_delta, prior_l)
    d_perm, d_perm_norm = dirichlet_batch(perm_delta, true_l)
    d_rand, d_rand_norm = dirichlet_batch(delta_h, random_l)
    d_h_true, d_h_true_norm = dirichlet_batch(h_t, true_l)
    out = {
        "D_true_Delta_H": d_true,
        "D_true_Delta_H_norm": d_true_norm,
        "D_prior_Delta_H": d_prior,
        "D_prior_Delta_H_norm": d_prior_norm,
        "D_perm_Delta_H": d_perm,
        "D_perm_Delta_H_norm": d_perm_norm,
        "D_rand_Delta_H": d_rand,
        "D_rand_Delta_H_norm": d_rand_norm,
        "D_true_H": d_h_true,
        "D_true_H_norm": d_h_true_norm,
    }
    for k in (2, 4, 8):
        out[f"R_low_true_Delta_H_{k}"] = low_frequency_ratio_batch(delta_h, true_l, k)
        out[f"R_low_prior_Delta_H_{k}"] = low_frequency_ratio_batch(prior_delta, prior_l, k)
        out[f"R_low_perm_Delta_H_{k}"] = low_frequency_ratio_batch(perm_delta, true_l, k)
        out[f"R_low_rand_Delta_H_{k}"] = low_frequency_ratio_batch(delta_h, random_l, k)
    return out


def successful_runs(results: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        run
        for run in results.get("runs", {}).values()
        if run.get("status") == "ok" and not run.get("failure_flag")
    ]


def build_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in successful_runs(results):
        config = run.get("config", {})
        artifact_path = ROOT / str(run.get("latent_trace_path", ""))
        checkpoint_path = ROOT / str(run.get("checkpoint_path", ""))
        row: dict[str, Any] = {
            "run_name": run.get("run_name"),
            "prior": config.get("prior"),
            "prior_weight": float(config.get("prior_weight", 0.1)),
            "seed": int(config.get("seed", -1)),
            "checkpoint_available": checkpoint_path.exists(),
            "artifact_available": artifact_path.exists(),
            "H16_rollout": float(run.get("diagnostics", {}).get("rollout_errors", {}).get("16", float("nan"))),
            "H32_rollout": float(run.get("diagnostics", {}).get("rollout_errors", {}).get("32", float("nan"))),
        }
        if artifact_path.exists():
            row.update(artifact_metrics(safe_torch_load(artifact_path)))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "prior",
        "prior_weight",
        "seed",
        "checkpoint_available",
        "artifact_available",
        "H16_rollout",
        "H32_rollout",
        "D_true_Delta_H",
        "D_true_Delta_H_norm",
        "D_prior_Delta_H",
        "D_prior_Delta_H_norm",
        "D_perm_Delta_H",
        "D_perm_Delta_H_norm",
        "D_rand_Delta_H",
        "D_rand_Delta_H_norm",
        "D_true_H",
        "D_true_H_norm",
        "R_low_true_Delta_H_2",
        "R_low_true_Delta_H_4",
        "R_low_true_Delta_H_8",
        "R_low_prior_Delta_H_2",
        "R_low_prior_Delta_H_4",
        "R_low_prior_Delta_H_8",
        "R_low_perm_Delta_H_2",
        "R_low_perm_Delta_H_4",
        "R_low_perm_Delta_H_8",
        "R_low_rand_Delta_H_2",
        "R_low_rand_Delta_H_4",
        "R_low_rand_Delta_H_8",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def rows_by_prior(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("prior"))].append(row)
    return grouped


def paired_model_delta(rows: list[dict[str, Any]], control_prior: str, metric: str) -> list[float]:
    grouped = rows_by_prior(rows)
    graph_by_seed = {int(row["seed"]): float(row[metric]) for row in grouped.get("graph", []) if finite(row.get(metric))}
    control_by_seed = {int(row["seed"]): float(row[metric]) for row in grouped.get(control_prior, []) if finite(row.get(metric))}
    return [control_by_seed[seed] - graph_by_seed[seed] for seed in SEEDS if seed in graph_by_seed and seed in control_by_seed]


def markdown_table(headers: list[str], table_rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table_rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def build_report(results: dict[str, Any], rows: list[dict[str, Any]], summary_csv: Path) -> str:
    failures = [name for name, run in results.get("runs", {}).items() if run.get("status") != "ok" or run.get("failure_flag")]
    checkpoints = sum(1 for row in rows if row.get("checkpoint_available"))
    artifacts = sum(1 for row in rows if row.get("artifact_available"))
    grouped = rows_by_prior(rows)
    rollout_table = []
    energy_table = []
    low_table = []
    for prior in PRIORS:
        prior_rows = grouped.get(prior, [])
        rollout_table.append(
            [
                prior,
                fmt_mean_std([row.get("H16_rollout", float("nan")) for row in prior_rows]),
                fmt_mean_std([row.get("H32_rollout", float("nan")) for row in prior_rows]),
            ]
        )
        energy_table.append(
            [
                prior,
                fmt_mean_std([row.get("D_true_Delta_H", float("nan")) for row in prior_rows]),
                fmt_mean_std([row.get("D_prior_Delta_H", float("nan")) for row in prior_rows]),
                fmt_mean_std([row.get("D_perm_Delta_H", float("nan")) for row in prior_rows]),
                fmt_mean_std([row.get("D_rand_Delta_H", float("nan")) for row in prior_rows]),
                fmt_mean_std([row.get("D_true_Delta_H_norm", float("nan")) for row in prior_rows]),
            ]
        )
        low_table.append(
            [
                prior,
                fmt_mean_std([row.get("R_low_true_Delta_H_2", float("nan")) for row in prior_rows]),
                fmt_mean_std([row.get("R_low_true_Delta_H_4", float("nan")) for row in prior_rows]),
                fmt_mean_std([row.get("R_low_true_Delta_H_8", float("nan")) for row in prior_rows]),
            ]
        )

    paired_perm = paired_model_delta(rows, "permuted_graph", "D_true_Delta_H_norm")
    paired_rand = paired_model_delta(rows, "random_graph", "D_true_Delta_H_norm")
    corr_metrics = ["D_true_Delta_H_norm", "D_prior_Delta_H_norm", "D_perm_Delta_H_norm", "D_rand_Delta_H_norm"]
    corr_table = []
    for metric in corr_metrics:
        x = [float(row.get(metric, float("nan"))) for row in rows]
        y = [float(row.get("H32_rollout", float("nan"))) for row in rows]
        corr_table.append([metric, len([1 for a, b in zip(x, y) if finite(a) and finite(b)]), fmt(pearson(x, y)), fmt(spearman(x, y))])

    graph_h32 = mean_std([row["H32_rollout"] for row in grouped.get("graph", [])])[0]
    perm_h32 = mean_std([row["H32_rollout"] for row in grouped.get("permuted_graph", [])])[0]
    rand_h32 = mean_std([row["H32_rollout"] for row in grouped.get("random_graph", [])])[0]
    graph_d = mean_std([row["D_true_Delta_H_norm"] for row in grouped.get("graph", [])])[0]
    perm_d = mean_std([row["D_true_Delta_H_norm"] for row in grouped.get("permuted_graph", [])])[0]
    rand_d = mean_std([row["D_true_Delta_H_norm"] for row in grouped.get("random_graph", [])])[0]
    explains = (
        finite(graph_h32)
        and graph_h32 < perm_h32
        and graph_h32 < rand_h32
        and finite(graph_d)
        and graph_d < perm_d
        and graph_d < rand_d
    )

    lines = [
        "# Cycle 8 Checkpointed Latent Alignment Report",
        "",
        "Experiment: `cycle8_checkpointed_lattice_latent_alignment`",
        "Analysis includes only HO lattice, GNN encoder, priors graph/permuted_graph/random_graph, lambda=0.1, seeds 0-4.",
        "",
        "## Run Integrity",
        "",
        f"Success/failure count: {len(rows)} ok / {len(failures)} failed",
        f"Checkpoint availability: {checkpoints}/{len(rows)}",
        f"Latent artifact availability: {artifacts}/{len(rows)}",
        "",
        "## Rollout Error",
        "",
    ]
    lines.extend(markdown_table(["prior", "H=16", "H=32"], rollout_table))
    lines.extend(["", "## Latent Dirichlet Energy", ""])
    lines.extend(
        markdown_table(
            ["prior", "D_true(Delta_H)", "D_prior(Delta_H)", "D_perm(Delta_H)", "D_rand(Delta_H)", "D_true_norm(Delta_H)"],
            energy_table,
        )
    )
    lines.extend(["", "## Low-Frequency Ratio On True L", ""])
    lines.extend(markdown_table(["prior", "R_low K=2", "R_low K=4", "R_low K=8"], low_table))
    lines.extend(["", "## Paired Latent Alignment", ""])
    lines.extend(
        markdown_table(
            ["control model", "paired delta D_true_norm(control - graph)", "graph lower count"],
            [
                ["permuted_graph", fmt_mean_std(paired_perm), f"{sum(1 for value in paired_perm if value > 0)}/{len(paired_perm)}"],
                ["random_graph", fmt_mean_std(paired_rand), f"{sum(1 for value in paired_rand if value > 0)}/{len(paired_rand)}"],
            ],
        )
    )
    lines.extend(["", "## Correlation With H=32 Rollout", ""])
    lines.extend(markdown_table(["latent metric", "n", "Pearson r", "Spearman rho"], corr_table))
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"Can latent alignment explain high-lambda lattice specificity? {'YES' if explains else 'NO / MIXED'}. "
            "The strict explanation criterion used here is that the graph-prior model has both lower H=32 rollout error and lower `D_true_norm(Delta_H)` than both control-prior models.",
            "If the paired latent-energy deltas are positive and the rollout table reproduces the graph advantage, latent alignment is a plausible mechanism. If rollout specificity reproduces without lower true-graph latent energy, latent alignment should remain future work or a negative diagnostic.",
            "",
            "## Files",
            "",
            f"- `{summary_csv.relative_to(ROOT)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Cycle 8 checkpointed lattice latent alignment.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    args = parser.parse_args()

    results = load_json(args.results)
    rows = build_rows(results)
    write_csv(args.summary_csv, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(results, rows, args.summary_csv), encoding="utf-8")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.summary_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
