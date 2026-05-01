from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

RMD17_MOLECULES = ["aspirin", "ethanol", "malonaldehyde", "naphthalene", "toluene"]
HO_TOPOLOGIES = ["lattice", "random", "scalefree"]
GRAPH_TYPES = ["true_graph", "permuted_graph", "random_graph"]
CONTROLS = {"permuted_graph": "permuted_graph", "random_graph": "random_graph"}
HORIZONS = ["16", "32"]

DEFAULT_RMD17_ROOT = ROOT / "data" / "rmd17_raw" / "rmd17" / "npz_data"
DEFAULT_HO_ROOT = ROOT / "data" / "ho_raw"

RESULTS = {
    "cycle2_rmd17_multimolecule": ROOT
    / "experiments"
    / "results"
    / "cycle2_rmd17_multimolecule"
    / "cycle2_rmd17_multimolecule_results.json",
    "cycle3_ho_networks": ROOT
    / "experiments"
    / "results"
    / "cycle3_ho_networks"
    / "cycle3_ho_networks_results.json",
    "cycle4_ho_lambda_robustness": ROOT
    / "experiments"
    / "results"
    / "cycle4_ho_lambda_robustness"
    / "cycle4_ho_lambda_robustness_results.json",
    "cycle5_ho_lattice_bridge": ROOT
    / "experiments"
    / "results"
    / "cycle5_ho_lattice_bridge"
    / "cycle5_ho_lattice_bridge_results.json",
}

SUMMARY_CSV = ROOT / "analysis_out" / "cycle6_alignment_summary.csv"
SPECIFICITY_CSV = ROOT / "analysis_out" / "cycle6_alignment_vs_specificity.csv"
REPORT = ROOT / "analysis_out" / "CYCLE6_GRAPH_DYNAMICS_ALIGNMENT_REPORT.md"


@dataclass(frozen=True)
class DatasetSpec:
    domain: str
    item: str
    data_root: Path
    n_frames: int
    n_nodes: int
    coords: np.ndarray
    edges: np.ndarray | None
    atomic_numbers: np.ndarray | None = None
    cutoff: float = 5.0


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean_std(values: list[float]) -> tuple[float, float, int]:
    values = [float(value) for value in values if finite(value)]
    if not values:
        return float("nan"), float("nan"), 0
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(np.mean(values)), std, len(values)


def fmt(value: Any, digits: int = 4) -> str:
    if not finite(value):
        return "NA"
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.{digits}f}"
    return f"{value:.{max(digits, 6)}f}"


def fmt_pct(value: Any) -> str:
    if not finite(value):
        return "NA"
    return f"{float(value):+.1f}%"


def pct_change(base: float, new: float) -> float:
    if not finite(base) or not finite(new) or float(base) == 0.0:
        return float("nan")
    return 100.0 * (float(new) - float(base)) / float(base)


def centered(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    return coords - coords.mean(axis=0, keepdims=True)


def sampled_frame_indices(n_frames: int, n_transitions: int, stride: int, horizon: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    max_start = n_frames - horizon - 1
    candidates = np.arange(0, max_start, stride)
    if n_transitions > len(candidates):
        n_transitions = len(candidates)
    chosen = rng.choice(candidates, size=n_transitions, replace=False)
    chosen.sort()
    return [int(value) for value in chosen]


def rmd17_edges_and_weights(coords: np.ndarray, cutoff: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    n_nodes = int(coords.shape[0])
    pairs: list[tuple[int, int]] = []
    weights: list[float] = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            distance = float(dist[i, j])
            if distance < cutoff:
                pairs.append((i, j))
                weights.append(1.0 / max(distance, 1e-6))
    return np.asarray(pairs, dtype=np.int64), np.asarray(weights, dtype=np.float64)


def laplacian_from_edges(n_nodes: int, edges: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    laplacian = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    if weights is None:
        weights = np.ones(int(edges.shape[0]), dtype=np.float64)
    for (src, dst), weight in zip(edges.astype(np.int64), weights.astype(np.float64)):
        if int(src) == int(dst):
            continue
        laplacian[int(src), int(src)] += float(weight)
        laplacian[int(dst), int(dst)] += float(weight)
        laplacian[int(src), int(dst)] -= float(weight)
        laplacian[int(dst), int(src)] -= float(weight)
    return laplacian


def random_edge_pairs_np(n_nodes: int, n_edges: int, seed: int) -> np.ndarray:
    if n_edges <= 0 or n_nodes < 2:
        return np.empty((0, 2), dtype=np.int64)
    rng = np.random.default_rng(seed)
    all_pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    choice = rng.choice(len(all_pairs), size=min(n_edges, len(all_pairs)), replace=False)
    return np.asarray([all_pairs[int(index)] for index in choice], dtype=np.int64)


def deterministic_permutation(n_nodes: int, seed: int) -> np.ndarray:
    # The training prior uses torch.randperm for this seed formula. This
    # analysis stays dependency-light and uses NumPy for the same role:
    # deterministic spectrum-preserving node-label scrambling.
    rng = np.random.default_rng(int(seed))
    return rng.permutation(n_nodes).astype(np.int64)


def laplacian_metrics(laplacian: np.ndarray) -> dict[str, float]:
    eigvals = np.linalg.eigvalsh(laplacian)
    eigvals = np.clip(eigvals, 0.0, None)
    positive = eigvals[eigvals > 1e-10]
    gap = float(eigvals[1]) if eigvals.size > 1 else float("nan")
    spread = float(positive[-1] / positive[0]) if positive.size else float("nan")
    return {
        "trace": float(np.trace(laplacian)),
        "frobenius_norm": float(np.linalg.norm(laplacian, ord="fro")),
        "lambda_max": float(eigvals[-1]) if eigvals.size else float("nan"),
        "spectral_gap": gap,
        "condition_spread": spread,
    }


def low_frequency_ratios(matrix: np.ndarray, laplacian: np.ndarray) -> dict[int, float]:
    n_nodes = int(matrix.shape[0])
    max_k = max(2, n_nodes // 4)
    ks = [k for k in (2, 4, 8) if k <= max_k and k < n_nodes]
    if not ks:
        return {}
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order]
    coeff = eigvecs.T @ matrix
    energy_by_mode = np.sum(coeff * coeff, axis=1)
    total = float(np.sum(energy_by_mode))
    if total <= 1e-18:
        return {k: float("nan") for k in ks}
    return {k: float(np.sum(energy_by_mode[:k]) / total) for k in ks}


def dirichlet(matrix: np.ndarray, laplacian: np.ndarray) -> float:
    return float(np.trace(matrix.T @ laplacian @ matrix))


def load_rmd17_spec(molecule: str, data_root: Path) -> DatasetSpec | None:
    path = data_root / f"rmd17_{molecule}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    coords = data["coords"].astype(np.float64)
    atomic_numbers = data["nuclear_charges"].astype(np.int64)
    return DatasetSpec(
        domain="rMD17",
        item=molecule,
        data_root=data_root,
        n_frames=int(coords.shape[0]),
        n_nodes=int(coords.shape[1]),
        coords=coords,
        edges=None,
        atomic_numbers=atomic_numbers,
    )


def load_ho_spec(topology: str, data_root: Path) -> DatasetSpec | None:
    path = data_root / f"ho_{topology}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    coords = data["coords"].astype(np.float64)
    return DatasetSpec(
        domain="HO",
        item=topology,
        data_root=data_root,
        n_frames=int(coords.shape[0]),
        n_nodes=int(coords.shape[1]),
        coords=coords,
        edges=data["edges"].astype(np.int64),
        atomic_numbers=data["nuclear_charges"].astype(np.int64),
    )


def collect_existing_sampling_specs(results: dict[str, dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, int]]]:
    specs: dict[tuple[str, str], dict[tuple[int, int, int, int], dict[str, int]]] = defaultdict(dict)
    for source, payload in results.items():
        runs = payload.get("runs", {})
        if isinstance(runs, list):
            iterable = runs
        else:
            iterable = runs.values()
        for run in iterable:
            if run.get("status") != "ok":
                continue
            config = run.get("config", {})
            prior = config.get("prior")
            encoder = config.get("encoder")
            if encoder != "gnn_node" or prior not in {"graph", "permuted_graph", "random_graph"}:
                continue
            if "molecule" in config:
                key = ("rMD17", str(config["molecule"]))
            elif "topology" in config:
                key = ("HO", str(config["topology"]))
            else:
                continue
            seed = int(config.get("seed", 0))
            n_transitions = int(config.get("n_transitions", 96))
            stride = int(config.get("stride", 10))
            horizon = int(config.get("horizon", 1))
            specs[key][(seed, n_transitions, stride, horizon)] = {
                "seed": seed,
                "n_transitions": n_transitions,
                "stride": stride,
                "horizon": horizon,
            }
    return {key: sorted(value.values(), key=lambda item: (item["seed"], item["n_transitions"], item["stride"], item["horizon"])) for key, value in specs.items()}


def graph_laplacian_for_sample(
    spec: DatasetSpec,
    graph_type: str,
    frame_idx: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if spec.domain == "HO":
        assert spec.edges is not None
        true_edges = spec.edges
        true_weights = np.ones(true_edges.shape[0], dtype=np.float64)
    else:
        true_edges, true_weights = rmd17_edges_and_weights(spec.coords[frame_idx], cutoff=spec.cutoff)
    true_laplacian = laplacian_from_edges(spec.n_nodes, true_edges, true_weights)
    if graph_type == "true_graph":
        return true_laplacian, None
    if graph_type == "permuted_graph":
        perm_seed = 2003 + int(seed) * 100000 + int(frame_idx)
        return true_laplacian, deterministic_permutation(spec.n_nodes, perm_seed)
    if graph_type == "random_graph":
        random_seed = 3001 + int(seed) * 100000 + int(frame_idx)
        random_edges = random_edge_pairs_np(spec.n_nodes, int(true_edges.shape[0]), seed=random_seed)
        random_weight = float(np.mean(true_weights)) if true_weights.size else 1.0
        random_weights = np.full(int(random_edges.shape[0]), random_weight, dtype=np.float64)
        return laplacian_from_edges(spec.n_nodes, random_edges, random_weights), None
    raise ValueError(f"Unknown graph_type {graph_type!r}")


def compute_alignment_for_graph_type(
    spec: DatasetSpec,
    graph_type: str,
    sampling_specs: list[dict[str, int]],
) -> dict[str, Any]:
    d_x_values: list[float] = []
    d_dx_values: list[float] = []
    x_norm_values: list[float] = []
    dx_norm_values: list[float] = []
    r_x: dict[int, list[float]] = defaultdict(list)
    r_dx: dict[int, list[float]] = defaultdict(list)
    metric_values: dict[str, list[float]] = defaultdict(list)
    n_edges_values: list[int] = []
    sample_count = 0
    seen_samples: set[tuple[int, int]] = set()
    for sample_spec in sampling_specs:
        seed = int(sample_spec["seed"])
        frame_indices = sampled_frame_indices(
            spec.n_frames,
            int(sample_spec["n_transitions"]),
            int(sample_spec["stride"]),
            int(sample_spec["horizon"]),
            seed,
        )
        for frame_idx in frame_indices:
            # True-graph diagnostics do not depend on seed; avoid overweighting duplicate
            # frames seen through multiple lambda sweeps. Controls are seed-dependent.
            dedupe_key = (seed if graph_type != "true_graph" else -1, frame_idx)
            if dedupe_key in seen_samples:
                continue
            seen_samples.add(dedupe_key)
            if frame_idx + 1 >= spec.n_frames:
                continue
            x = centered(spec.coords[frame_idx])
            x_next = centered(spec.coords[frame_idx + 1])
            dx = x_next - x
            laplacian, perm = graph_laplacian_for_sample(spec, graph_type, frame_idx, seed)
            x_eval = x[perm] if perm is not None else x
            dx_eval = dx[perm] if perm is not None else dx
            d_x_values.append(dirichlet(x_eval, laplacian))
            d_dx_values.append(dirichlet(dx_eval, laplacian))
            x_norm_values.append(float(np.sum(x_eval * x_eval)))
            dx_norm_values.append(float(np.sum(dx_eval * dx_eval)))
            for key, value in laplacian_metrics(laplacian).items():
                metric_values[key].append(value)
            n_edges_values.append(int(round(float(np.trace(laplacian) / 2.0))))
            for k, value in low_frequency_ratios(x_eval, laplacian).items():
                r_x[k].append(value)
            for k, value in low_frequency_ratios(dx_eval, laplacian).items():
                r_dx[k].append(value)
            sample_count += 1
    d_x = float(np.mean(d_x_values)) if d_x_values else float("nan")
    d_dx = float(np.mean(d_dx_values)) if d_dx_values else float("nan")
    x_norm = float(np.mean(x_norm_values)) if x_norm_values else float("nan")
    dx_norm = float(np.mean(dx_norm_values)) if dx_norm_values else float("nan")
    out: dict[str, Any] = {
        "domain": spec.domain,
        "item": spec.item,
        "graph_type": graph_type,
        "n_nodes": spec.n_nodes,
        "n_edges_mean": float(np.mean(n_edges_values)) if n_edges_values else float("nan"),
        "n_samples": sample_count,
        "D_X": d_x,
        "D_dX": d_dx,
        "D_X_norm": d_x / x_norm if finite(d_x) and x_norm > 0 else float("nan"),
        "D_dX_norm": d_dx / dx_norm if finite(d_dx) and dx_norm > 0 else float("nan"),
    }
    for key, values in metric_values.items():
        out[key] = float(np.mean(values)) if values else float("nan")
    for k in (2, 4, 8):
        out[f"R_low_X_{k}"] = float(np.mean(r_x[k])) if r_x.get(k) else float("nan")
        out[f"R_low_dX_{k}"] = float(np.mean(r_dx[k])) if r_dx.get(k) else float("nan")
    return out


def load_existing_results() -> dict[str, dict[str, Any]]:
    return {source: load_json(path) for source, path in RESULTS.items() if path.exists()}


def graph_prior_runs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    runs = payload.get("runs", {})
    iterable = runs if isinstance(runs, list) else runs.values()
    return [
        run
        for run in iterable
        if run.get("status") == "ok"
        and run.get("config", {}).get("encoder") == "gnn_node"
        and run.get("config", {}).get("prior") in {"graph", "permuted_graph", "random_graph"}
    ]


def group_rollouts(runs: list[dict[str, Any]], source: str) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        config = run.get("config", {})
        if "molecule" in config:
            domain = "rMD17"
            item = config["molecule"]
            prior_weight = float(config.get("prior_weight", 0.1))
        else:
            domain = "HO"
            item = config["topology"]
            prior_weight = float(config.get("prior_weight", 0.1))
        if source == "cycle3_ho_networks":
            prior_weight = 0.1
        key = (domain, item, float(prior_weight), config.get("prior"))
        grouped[key].append(run)
    return grouped


def observed_specificity_rows(results: dict[str, dict[str, Any]], alignment_by_key: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, payload in results.items():
        grouped = group_rollouts(graph_prior_runs(payload), source)
        base_keys = [key for key in grouped if key[-1] == "graph"]
        for domain, item, prior_weight, _prior in sorted(base_keys):
            graph_runs = grouped[(domain, item, prior_weight, "graph")]
            graph_prior_loss_values = [
                float(run.get("diagnostics", {}).get("prior_loss_mean"))
                for run in graph_runs
                if finite(run.get("diagnostics", {}).get("prior_loss_mean"))
            ]
            graph_prior_loss_mean = mean_std(graph_prior_loss_values)[0]
            for control_prior, graph_type in CONTROLS.items():
                control_key = (domain, item, prior_weight, control_prior)
                if control_key not in grouped:
                    continue
                for horizon in HORIZONS:
                    graph_values = [
                        float(run.get("diagnostics", {}).get("rollout_errors", {}).get(horizon))
                        for run in graph_runs
                        if finite(run.get("diagnostics", {}).get("rollout_errors", {}).get(horizon))
                    ]
                    control_values = [
                        float(run.get("diagnostics", {}).get("rollout_errors", {}).get(horizon))
                        for run in grouped[control_key]
                        if finite(run.get("diagnostics", {}).get("rollout_errors", {}).get(horizon))
                    ]
                    graph_mean, _, graph_n = mean_std(graph_values)
                    control_mean, _, control_n = mean_std(control_values)
                    if not graph_n or not control_n:
                        continue
                    true_align = alignment_by_key.get((domain, item, "true_graph"))
                    control_align = alignment_by_key.get((domain, item, graph_type))
                    if not true_align or not control_align:
                        continue
                    row = {
                        "source": source,
                        "domain": domain,
                        "item": item,
                        "prior_weight": prior_weight,
                        "horizon": int(horizon),
                        "control": control_prior,
                        "graph_mean": graph_mean,
                        "control_mean": control_mean,
                        "S_graph": control_mean - graph_mean,
                        "graph_prior_loss_mean": graph_prior_loss_mean,
                        "D_dX_norm_true": true_align["D_dX_norm"],
                        "D_dX_norm_control": control_align["D_dX_norm"],
                        "gap_D_dX_norm_control_minus_true": control_align["D_dX_norm"] - true_align["D_dX_norm"],
                        "D_X_norm_true": true_align["D_X_norm"],
                        "D_X_norm_control": control_align["D_X_norm"],
                        "gap_D_X_norm_control_minus_true": control_align["D_X_norm"] - true_align["D_X_norm"],
                        "gap_R_low_dX_2_true_minus_control": true_align["R_low_dX_2"] - control_align["R_low_dX_2"],
                        "gap_R_low_dX_4_true_minus_control": true_align["R_low_dX_4"] - control_align["R_low_dX_4"],
                        "gap_R_low_dX_8_true_minus_control": true_align["R_low_dX_8"] - control_align["R_low_dX_8"],
                        "gap_R_low_X_2_true_minus_control": true_align["R_low_X_2"] - control_align["R_low_X_2"],
                        "gap_R_low_X_4_true_minus_control": true_align["R_low_X_4"] - control_align["R_low_X_4"],
                        "gap_R_low_X_8_true_minus_control": true_align["R_low_X_8"] - control_align["R_low_X_8"],
                        "gap_trace_control_minus_true": control_align["trace"] - true_align["trace"],
                        "gap_frobenius_control_minus_true": control_align["frobenius_norm"] - true_align["frobenius_norm"],
                        "gap_lambda_max_control_minus_true": control_align["lambda_max"] - true_align["lambda_max"],
                        "gap_spectral_gap_control_minus_true": control_align["spectral_gap"] - true_align["spectral_gap"],
                    }
                    rows.append(row)
    return rows


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


def pearson(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 3 or np.std(x_arr) <= 0 or np.std(y_arr) <= 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def spearman(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(np.sum(valid)) < 3:
        return float("nan")
    return pearson(rankdata(x_arr[valid]), rankdata(y_arr[valid]))


def correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictors = [
        "gap_D_dX_norm_control_minus_true",
        "gap_D_X_norm_control_minus_true",
        "gap_R_low_dX_2_true_minus_control",
        "gap_R_low_dX_4_true_minus_control",
        "gap_R_low_dX_8_true_minus_control",
        "gap_R_low_X_2_true_minus_control",
        "gap_R_low_X_4_true_minus_control",
        "gap_R_low_X_8_true_minus_control",
        "gap_trace_control_minus_true",
        "gap_frobenius_control_minus_true",
        "gap_lambda_max_control_minus_true",
        "gap_spectral_gap_control_minus_true",
        "graph_prior_loss_mean",
    ]
    scopes: list[tuple[str, list[dict[str, Any]]]] = [("all", rows)]
    for horizon in (16, 32):
        scopes.append((f"H={horizon}", [row for row in rows if int(row["horizon"]) == horizon]))
    for domain in ("rMD17", "HO"):
        scopes.append((domain, [row for row in rows if row["domain"] == domain]))
    out: list[dict[str, Any]] = []
    for scope, scope_rows in scopes:
        y = [row["S_graph"] for row in scope_rows if finite(row.get("S_graph"))]
        for predictor in predictors:
            pairs = [
                (float(row[predictor]), float(row["S_graph"]))
                for row in scope_rows
                if finite(row.get(predictor)) and finite(row.get("S_graph"))
            ]
            if len(pairs) < 3:
                continue
            x_vals = [pair[0] for pair in pairs]
            y_vals = [pair[1] for pair in pairs]
            out.append(
                {
                    "scope": scope,
                    "predictor": predictor,
                    "n": len(pairs),
                    "pearson_r": pearson(x_vals, y_vals),
                    "spearman_rho": spearman(x_vals, y_vals),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float) and not math.isfinite(value):
                    value = ""
                formatted[key] = value
            writer.writerow(formatted)


def markdown_table(rows: list[list[Any]], headers: list[str]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return out


def build_report(
    alignment_rows: list[dict[str, Any]],
    specificity_rows: list[dict[str, Any]],
    corr_rows: list[dict[str, Any]],
    missing: list[str],
) -> str:
    by_key = {(row["domain"], row["item"], row["graph_type"]): row for row in alignment_rows}
    gap_rows = []
    for domain, item in sorted({(row["domain"], row["item"]) for row in alignment_rows}):
        true = by_key.get((domain, item, "true_graph"))
        perm = by_key.get((domain, item, "permuted_graph"))
        rand = by_key.get((domain, item, "random_graph"))
        if not true or not perm or not rand:
            continue
        gap_rows.append(
            [
                domain,
                item,
                fmt(perm["D_dX_norm"] - true["D_dX_norm"]),
                fmt(rand["D_dX_norm"] - true["D_dX_norm"]),
                fmt(true["R_low_dX_2"] - perm["R_low_dX_2"]),
                fmt(true["R_low_dX_2"] - rand["R_low_dX_2"]),
                "YES" if perm["D_dX_norm"] > true["D_dX_norm"] and rand["D_dX_norm"] > true["D_dX_norm"] else "NO",
            ]
        )
    selected_corr = [
        row
        for row in corr_rows
        if row["scope"] in {"all", "H=32", "HO", "rMD17"}
        and row["predictor"]
        in {
            "gap_D_dX_norm_control_minus_true",
            "gap_D_X_norm_control_minus_true",
            "gap_R_low_dX_2_true_minus_control",
            "gap_frobenius_control_minus_true",
            "gap_lambda_max_control_minus_true",
            "graph_prior_loss_mean",
        }
    ]
    corr_table = [
        [
            row["scope"],
            row["predictor"],
            row["n"],
            fmt(row["pearson_r"]),
            fmt(row["spearman_rho"]),
        ]
        for row in selected_corr
    ]
    h32_rows = [row for row in specificity_rows if int(row["horizon"]) == 32]
    ddx_spearman = next(
        (
            row["spearman_rho"]
            for row in corr_rows
            if row["scope"] == "H=32" and row["predictor"] == "gap_D_dX_norm_control_minus_true"
        ),
        float("nan"),
    )
    dx_spearman = next(
        (
            row["spearman_rho"]
            for row in corr_rows
            if row["scope"] == "H=32" and row["predictor"] == "gap_D_X_norm_control_minus_true"
        ),
        float("nan"),
    )
    rlow_spearman = next(
        (
            row["spearman_rho"]
            for row in corr_rows
            if row["scope"] == "H=32" and row["predictor"] == "gap_R_low_dX_2_true_minus_control"
        ),
        float("nan"),
    )
    fro_spearman = next(
        (
            row["spearman_rho"]
            for row in corr_rows
            if row["scope"] == "H=32" and row["predictor"] == "gap_frobenius_control_minus_true"
        ),
        float("nan"),
    )
    lambda_spearman = next(
        (
            row["spearman_rho"]
            for row in corr_rows
            if row["scope"] == "H=32" and row["predictor"] == "gap_lambda_max_control_minus_true"
        ),
        float("nan"),
    )
    positive_alignment = sum(
        1
        for row in h32_rows
        if finite(row["gap_D_dX_norm_control_minus_true"]) and float(row["gap_D_dX_norm_control_minus_true"]) > 0
    )
    positive_specificity = sum(1 for row in h32_rows if float(row["S_graph"]) > 0)
    both_positive = sum(
        1
        for row in h32_rows
        if float(row["S_graph"]) > 0
        and finite(row["gap_D_dX_norm_control_minus_true"])
        and float(row["gap_D_dX_norm_control_minus_true"]) > 0
    )
    lines = [
        "# Cycle 6 Graph-Dynamics Alignment Diagnostics",
        "",
        "Analysis only: no training, ISO17, rMD17 top-up, or new experiment launch commands are run.",
        "",
        "Inputs:",
    ]
    for path in RESULTS.values():
        lines.append(f"- `{path.relative_to(ROOT)}`")
    lines.extend(
        [
            "",
            "Method:",
            "- `X_t` is the centered raw coordinate matrix for each frame.",
            "- `dX_t` is the centered one-step coordinate change, `X_{t+1} - X_t`.",
            "- HO true graphs use the stored unit-weight topology edges.",
            "- rMD17 true graphs use the same cutoff graph and inverse-distance weights as the node-wise graph prior.",
            "- `permuted_graph` uses the true Laplacian with the training permutation seed formula and a deterministic NumPy permutation; this preserves the scale-matched control interpretation.",
            "- `random_graph` uses the same random-edge seed formula used during training.",
            "",
        ]
    )
    if missing:
        lines.append("Missing or skipped datasets:")
        lines.extend(f"- {item}" for item in missing)
        lines.append("")
    lines.extend(
        [
            "## Alignment Gap Summary",
            "",
            "Positive `D_dX` gaps mean the true graph is smoother than the control graph for temporal changes. Positive `R_low` gaps mean temporal changes put more energy in the true graph low-frequency basis.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            gap_rows,
            [
                "domain",
                "item",
                "D_dX gap perm-control",
                "D_dX gap random-control",
                "R_low_dX2 gap perm",
                "R_low_dX2 gap random",
                "true smoother vs both?",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Correlations With Observed Specificity",
            "",
            "`S_graph = E_control - E_graph`; positive values mean the true graph prior has lower rollout error than the control.",
            "",
        ]
    )
    lines.extend(markdown_table(corr_table, ["scope", "predictor", "n", "Pearson r", "Spearman rho"]))
    lines.extend(
        [
            "",
            "## Diagnostic Answers",
            "",
            f"1. Are true dynamics smoother on true graph than on controls? At H=32 comparison granularity, {positive_alignment}/{len(h32_rows)} rows have positive temporal-change alignment gaps, and {both_positive}/{positive_specificity} positive-specificity rows also have positive temporal-change gaps. The answer is mixed, not universal.",
            f"2. Does temporal-change smoothness predict observed specificity? Weakly at best in the pooled table: H=32 Spearman rho for `gap_D_dX_norm_control_minus_true` is {fmt(ddx_spearman)}.",
            f"3. Is `D_dX` more predictive than static `D_X`? H=32 Spearman rho is {fmt(ddx_spearman)} for `D_dX` versus {fmt(dx_spearman)} for `D_X`; use the sign and magnitude together, because source/lambda reuse creates repeated alignment rows.",
            f"4. Is low-frequency energy ratio predictive? H=32 Spearman rho for `R_low_dX(K=2)` gap is {fmt(rlow_spearman)}; this is not strong enough to stand alone as a decision rule.",
            f"5. Are graph spectrum-only metrics insufficient? Yes. H=32 Spearman rho for Frobenius/lambda_max gaps is {fmt(fro_spearman)}/{fmt(lambda_spearman)}, and permuted controls are spectrum-matched by construction, so spectrum-only metrics cannot explain permuted-control outcomes.",
            "6. Decision rule: the proposed `only when D_dX is smoother on the true graph` rule is not validated by these raw-coordinate diagnostics. It would correctly warn against many generic-smoothing cases, but it would also reject the Cycle 5 high-lambda lattice specificity point. A safer rule is: claim topology specificity only with paired rollout evidence against scale-matched controls; use positive `D_dX` alignment as supporting mechanistic evidence when present, not as a sufficient or currently necessary condition.",
            "",
            "Key caveat: these diagnostics are computed on raw centered coordinates, while the prior acts on learned node states. A learned encoder can rotate, rescale, or filter dynamics before the Laplacian penalty is applied, so raw-coordinate alignment is a mechanism probe rather than a complete explanation.",
            "",
            "## Files",
            "",
            f"- `{SUMMARY_CSV.relative_to(ROOT)}`",
            f"- `{SPECIFICITY_CSV.relative_to(ROOT)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze graph-dynamics alignment for Cycle 6 diagnostics.")
    parser.add_argument("--rmd17-root", type=Path, default=DEFAULT_RMD17_ROOT)
    parser.add_argument("--ho-root", type=Path, default=DEFAULT_HO_ROOT)
    args = parser.parse_args()

    results = load_existing_results()
    sampling_by_dataset = collect_existing_sampling_specs(results)
    specs: list[DatasetSpec] = []
    missing: list[str] = []
    for topology in HO_TOPOLOGIES:
        spec = load_ho_spec(topology, args.ho_root)
        if spec is None:
            missing.append(f"HO {topology}: missing data")
        else:
            specs.append(spec)
    for molecule in RMD17_MOLECULES:
        spec = load_rmd17_spec(molecule, args.rmd17_root)
        if spec is None:
            missing.append(f"rMD17 {molecule}: missing data")
        else:
            specs.append(spec)

    alignment_rows: list[dict[str, Any]] = []
    for spec in specs:
        sampling_specs = sampling_by_dataset.get((spec.domain, spec.item))
        if not sampling_specs:
            missing.append(f"{spec.domain} {spec.item}: no existing graph-prior sampling specs")
            continue
        for graph_type in GRAPH_TYPES:
            alignment_rows.append(compute_alignment_for_graph_type(spec, graph_type, sampling_specs))

    alignment_by_key = {
        (str(row["domain"]), str(row["item"]), str(row["graph_type"])): row
        for row in alignment_rows
    }
    specificity_rows = observed_specificity_rows(results, alignment_by_key)
    corr_rows = correlation_rows(specificity_rows)

    alignment_fields = [
        "domain",
        "item",
        "graph_type",
        "n_nodes",
        "n_edges_mean",
        "n_samples",
        "D_X",
        "D_dX",
        "D_X_norm",
        "D_dX_norm",
        "R_low_X_2",
        "R_low_X_4",
        "R_low_X_8",
        "R_low_dX_2",
        "R_low_dX_4",
        "R_low_dX_8",
        "trace",
        "frobenius_norm",
        "lambda_max",
        "spectral_gap",
        "condition_spread",
    ]
    specificity_fields = [
        "source",
        "domain",
        "item",
        "prior_weight",
        "horizon",
        "control",
        "graph_mean",
        "control_mean",
        "S_graph",
        "graph_prior_loss_mean",
        "D_dX_norm_true",
        "D_dX_norm_control",
        "gap_D_dX_norm_control_minus_true",
        "D_X_norm_true",
        "D_X_norm_control",
        "gap_D_X_norm_control_minus_true",
        "gap_R_low_dX_2_true_minus_control",
        "gap_R_low_dX_4_true_minus_control",
        "gap_R_low_dX_8_true_minus_control",
        "gap_R_low_X_2_true_minus_control",
        "gap_R_low_X_4_true_minus_control",
        "gap_R_low_X_8_true_minus_control",
        "gap_trace_control_minus_true",
        "gap_frobenius_control_minus_true",
        "gap_lambda_max_control_minus_true",
        "gap_spectral_gap_control_minus_true",
    ]
    write_csv(SUMMARY_CSV, alignment_rows, alignment_fields)
    write_csv(SPECIFICITY_CSV, specificity_rows, specificity_fields)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(build_report(alignment_rows, specificity_rows, corr_rows, missing), encoding="utf-8")
    print(f"Wrote {REPORT.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_CSV.relative_to(ROOT)} ({len(alignment_rows)} rows)")
    print(f"Wrote {SPECIFICITY_CSV.relative_to(ROOT)} ({len(specificity_rows)} rows)")


if __name__ == "__main__":
    main()
