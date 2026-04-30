from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "experiments" / "results" / "cycle4_ho_lambda_robustness" / "cycle4_ho_lambda_robustness_results.json"
DEFAULT_CONFIG_DIR = ROOT / "experiments" / "configs" / "cycle4_ho_lambda_robustness"
DEFAULT_DATA_ROOT = ROOT / "data" / "ho_raw"
DEFAULT_REPORT = ROOT / "analysis_out" / "CYCLE4_GRAPH_SPECTRA_AND_LOSS_DIAGNOSTICS.md"

GRAPH_PRIORS = ("graph", "permuted_graph", "random_graph")
H32_KEY = "32"


@dataclass(frozen=True)
class RunRecord:
    run_name: str
    topology: str
    prior: str
    prior_weight: float
    seed: int
    h32_rollout_error: float
    final_train_loss: float
    prior_loss_mean: float
    status_ok: bool
    config: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def as_float(value: Any, default: float = float("nan")) -> float:
    if finite(value):
        return float(value)
    return default


def fmt(value: Any, digits: int = 4) -> str:
    if not finite(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def fmt_sci(value: Any, digits: int = 3) -> str:
    if not finite(value):
        return "nan"
    return f"{float(value):.{digits}e}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    if not finite(value):
        return "nan"
    return f"{float(value):+.{digits}f}%"


def mean(values: Iterable[float]) -> float:
    array = np.asarray([value for value in values if finite(value)], dtype=float)
    return float(array.mean()) if array.size else float("nan")


def std(values: Iterable[float]) -> float:
    array = np.asarray([value for value in values if finite(value)], dtype=float)
    if array.size == 0:
        return float("nan")
    return float(array.std(ddof=1)) if array.size > 1 else 0.0


def mean_std(values: Iterable[float]) -> tuple[float, float, int]:
    array = np.asarray([value for value in values if finite(value)], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan"), 0
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0, int(array.size)


def pct_lower(candidate: float, control: float) -> float:
    if not finite(candidate) or not finite(control) or control == 0.0:
        return float("nan")
    return 100.0 * (control - candidate) / abs(control)


def pearson(x_values: list[float], y_values: list[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(x_values, y_values) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return float("nan")
    x = np.asarray([pair[0] for pair in pairs], dtype=float)
    y = np.asarray([pair[1] for pair in pairs], dtype=float)
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman(x_values: list[float], y_values: list[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(x_values, y_values) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return float("nan")
    x = rankdata(np.asarray([pair[0] for pair in pairs], dtype=float))
    y = rankdata(np.asarray([pair[1] for pair in pairs], dtype=float))
    return pearson(x.tolist(), y.tolist())


def sampled_frame_indices(n_frames: int, n_transitions: int, stride: int, horizon: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    max_start = n_frames - horizon - 1
    candidates = np.arange(0, max_start, stride)
    if n_transitions > len(candidates):
        raise ValueError(f"Requested {n_transitions}, but only {len(candidates)} candidates exist")
    chosen = rng.choice(candidates, size=n_transitions, replace=False)
    chosen.sort()
    return [int(value) for value in chosen]


def random_edge_pairs(n_nodes: int, n_edges: int, seed: int) -> np.ndarray:
    if n_edges <= 0 or n_nodes < 2:
        return np.empty((0, 2), dtype=np.int64)
    rng = np.random.default_rng(seed)
    all_pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    choice = rng.choice(len(all_pairs), size=min(n_edges, len(all_pairs)), replace=False)
    return np.asarray([all_pairs[int(index)] for index in choice], dtype=np.int64)


def laplacian_from_edges(n_nodes: int, edges: np.ndarray, weight: float = 1.0) -> np.ndarray:
    laplacian = np.zeros((n_nodes, n_nodes), dtype=float)
    for src, dst in np.asarray(edges, dtype=np.int64):
        if int(src) == int(dst):
            continue
        laplacian[int(src), int(src)] += weight
        laplacian[int(dst), int(dst)] += weight
        laplacian[int(src), int(dst)] -= weight
        laplacian[int(dst), int(src)] -= weight
    return laplacian


def laplacian_metrics(laplacian: np.ndarray, edge_count: int) -> dict[str, float]:
    degrees = np.diag(laplacian)
    eigvals = np.linalg.eigvalsh(laplacian)
    eigvals = np.clip(eigvals, 0.0, None)
    positive = eigvals[eigvals > 1e-10]
    lambda_max = float(eigvals[-1]) if eigvals.size else float("nan")
    spectral_gap = float(eigvals[1]) if eigvals.size > 1 else float("nan")
    condition_number = float(lambda_max / spectral_gap) if spectral_gap > 1e-10 else float("inf")
    eigen_spread = float(lambda_max / positive[0]) if positive.size else float("inf")
    return {
        "n_nodes": float(laplacian.shape[0]),
        "n_edges": float(edge_count),
        "degree_mean": float(degrees.mean()) if degrees.size else float("nan"),
        "degree_std": float(degrees.std(ddof=0)) if degrees.size else float("nan"),
        "laplacian_trace": float(np.trace(laplacian)),
        "frobenius_norm": float(np.linalg.norm(laplacian, ord="fro")),
        "lambda_max": lambda_max,
        "spectral_gap": spectral_gap,
        "condition_number": condition_number,
        "eigen_spread": eigen_spread,
    }


def average_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for metrics in metric_dicts for key in metrics})
    return {key: mean(metrics.get(key, float("nan")) for metrics in metric_dicts) for key in keys}


def metric_dict_stds(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for metrics in metric_dicts for key in metrics})
    return {key: std(metrics.get(key, float("nan")) for metrics in metric_dicts) for key in keys}


def load_topology_graph(topology: str, data_root: Path) -> tuple[int, int, int, np.ndarray]:
    npz_path = data_root / f"ho_{topology}.npz"
    with np.load(npz_path) as data:
        n_frames = int(data["coords"].shape[0])
        n_nodes = int(data["coords"].shape[1])
        edges = np.asarray(data["edges"], dtype=np.int64)
    return n_frames, n_nodes, int(edges.shape[0]), edges


def extract_run_records(results: dict[str, Any]) -> list[RunRecord]:
    records: list[RunRecord] = []
    for run_name, run in sorted(results.get("runs", {}).items()):
        config = run.get("config", {})
        diagnostics = run.get("diagnostics", {})
        records.append(
            RunRecord(
                run_name=str(run_name),
                topology=str(config.get("topology")),
                prior=str(config.get("prior")),
                prior_weight=as_float(config.get("prior_weight")),
                seed=int(config.get("seed", -1)),
                h32_rollout_error=as_float(diagnostics.get("rollout_errors", {}).get(H32_KEY)),
                final_train_loss=as_float(diagnostics.get("final_train_loss")),
                prior_loss_mean=as_float(diagnostics.get("prior_loss_mean")),
                status_ok=run.get("status") == "ok" and not bool(run.get("failure_flag")),
                config=dict(config),
            )
        )
    return records


def load_config_fallbacks(config_dir: Path) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for path in sorted(config_dir.glob("*.json")):
        try:
            raw = load_json(path)
        except json.JSONDecodeError:
            continue
        run_name = str(raw.get("run_name", path.stem))
        configs[run_name] = raw
    return configs


def config_for_record(record: RunRecord, config_fallbacks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merged = dict(config_fallbacks.get(record.run_name, {}))
    merged.update(record.config)
    return merged


def graph_metric_samples_for_run(
    record: RunRecord,
    config: dict[str, Any],
    data_root: Path,
) -> list[dict[str, float]]:
    n_frames, n_nodes, n_edges, true_edges = load_topology_graph(record.topology, data_root)
    true_laplacian = laplacian_from_edges(n_nodes, true_edges, weight=1.0)
    if record.prior in {"graph", "permuted_graph"}:
        return [laplacian_metrics(true_laplacian, n_edges)]
    if record.prior != "random_graph":
        return []

    frame_indices = sampled_frame_indices(
        n_frames=n_frames,
        n_transitions=int(config.get("n_transitions", 96)),
        stride=int(config.get("stride", 10)),
        horizon=int(config.get("horizon", 1)),
        seed=int(record.seed),
    )
    samples = []
    for frame_idx in frame_indices:
        edges = random_edge_pairs(
            n_nodes=n_nodes,
            n_edges=n_edges,
            seed=3001 + int(record.seed) * 100000 + int(frame_idx),
        )
        laplacian = laplacian_from_edges(n_nodes, edges, weight=1.0)
        samples.append(laplacian_metrics(laplacian, int(edges.shape[0])))
    return samples


def build_metric_index(
    records: list[RunRecord],
    config_fallbacks: dict[str, dict[str, Any]],
    data_root: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[tuple[str, str], dict[str, float]]]:
    run_metrics: dict[str, dict[str, float]] = {}
    run_metric_stds: dict[str, dict[str, float]] = {}
    grouped_samples: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)

    for record in records:
        if record.prior not in GRAPH_PRIORS:
            continue
        config = config_for_record(record, config_fallbacks)
        samples = graph_metric_samples_for_run(record, config, data_root)
        if not samples:
            continue
        avg = average_metric_dicts(samples)
        run_metrics[record.run_name] = avg
        run_metric_stds[record.run_name] = metric_dict_stds(samples)
        grouped_samples[(record.topology, record.prior)].extend(samples)

    group_metrics = {
        key: average_metric_dicts(samples)
        for key, samples in grouped_samples.items()
    }
    return run_metrics, run_metric_stds, group_metrics


def grouped_records(records: list[RunRecord]) -> dict[tuple[str, str, float], list[RunRecord]]:
    grouped: dict[tuple[str, str, float], list[RunRecord]] = defaultdict(list)
    for record in records:
        if record.status_ok and record.prior in GRAPH_PRIORS:
            grouped[(record.topology, record.prior, record.prior_weight)].append(record)
    return grouped


def grouped_by_topology_prior(records: list[RunRecord]) -> dict[tuple[str, str], list[RunRecord]]:
    grouped: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        if record.status_ok and record.prior in GRAPH_PRIORS:
            grouped[(record.topology, record.prior)].append(record)
    return grouped


def h32_mean(records: list[RunRecord]) -> float:
    return mean(record.h32_rollout_error for record in records)


def best_control_summary(records: list[RunRecord]) -> list[tuple[str, float, str, float, float]]:
    grouped = grouped_records(records)
    out = []
    for (topology, _prior, prior_weight), graph_records in sorted(grouped.items()):
        if _prior != "graph":
            continue
        graph_mean = h32_mean(graph_records)
        for control in ("permuted_graph", "random_graph"):
            control_mean = h32_mean(grouped.get((topology, control, prior_weight), []))
            out.append((topology, prior_weight, control, graph_mean, control_mean))
    return out


def correlation_rows(records: list[RunRecord], run_metrics: dict[str, dict[str, float]]) -> list[tuple[str, str, int, float, float]]:
    rows = []
    scopes: list[tuple[str, list[RunRecord]]] = [("all", [record for record in records if record.status_ok and record.prior in GRAPH_PRIORS])]
    topologies = sorted({record.topology for record in records if record.status_ok and record.prior in GRAPH_PRIORS})
    for topology in topologies:
        scopes.append((topology, [record for record in records if record.status_ok and record.prior in GRAPH_PRIORS and record.topology == topology]))

    predictors = [
        ("lambda", lambda record: record.prior_weight),
        ("lambda_eff_fro", lambda record: record.prior_weight * run_metrics.get(record.run_name, {}).get("frobenius_norm", float("nan"))),
        ("lambda_eff_lambda_max", lambda record: record.prior_weight * run_metrics.get(record.run_name, {}).get("lambda_max", float("nan"))),
        ("prior_loss_mean", lambda record: record.prior_loss_mean),
        ("final_train_loss", lambda record: record.final_train_loss),
    ]
    for scope, scope_records in scopes:
        y = [record.h32_rollout_error for record in scope_records]
        for predictor_name, fn in predictors:
            x = [fn(record) for record in scope_records]
            rows.append((scope, predictor_name, len(scope_records), pearson(x, y), spearman(x, y)))
    return rows


def scale_mismatch_summary(group_metrics: dict[tuple[str, str], dict[str, float]]) -> list[str]:
    lines = []
    topologies = sorted({topology for topology, _prior in group_metrics})
    for topology in topologies:
        graph = group_metrics.get((topology, "graph"), {})
        permuted = group_metrics.get((topology, "permuted_graph"), {})
        random = group_metrics.get((topology, "random_graph"), {})
        graph_fro = graph.get("frobenius_norm", float("nan"))
        graph_lmax = graph.get("lambda_max", float("nan"))
        perm_fro_delta = 100.0 * (permuted.get("frobenius_norm", float("nan")) - graph_fro) / graph_fro if graph_fro else float("nan")
        rand_fro_delta = 100.0 * (random.get("frobenius_norm", float("nan")) - graph_fro) / graph_fro if graph_fro else float("nan")
        perm_lmax_delta = 100.0 * (permuted.get("lambda_max", float("nan")) - graph_lmax) / graph_lmax if graph_lmax else float("nan")
        rand_lmax_delta = 100.0 * (random.get("lambda_max", float("nan")) - graph_lmax) / graph_lmax if graph_lmax else float("nan")
        lines.append(
            f"{topology}: permuted vs true fro {fmt_pct(perm_fro_delta)}, lambda_max {fmt_pct(perm_lmax_delta)}; "
            f"random vs true fro {fmt_pct(rand_fro_delta)}, lambda_max {fmt_pct(rand_lmax_delta)}."
        )
    return lines


def narrative_answers(
    records: list[RunRecord],
    group_metrics: dict[tuple[str, str], dict[str, float]],
    corr_rows: list[tuple[str, str, int, float, float]],
) -> list[str]:
    mismatch_lines = scale_mismatch_summary(group_metrics)
    outperform = [
        (topology, prior_weight, control, graph_mean, control_mean)
        for topology, prior_weight, control, graph_mean, control_mean in best_control_summary(records)
        if finite(graph_mean) and finite(control_mean) and control_mean < graph_mean
    ]
    all_corr = [row for row in corr_rows if row[0] == "all"]
    abs_spearman = {name: abs(rho) for _scope, name, _n, _r, rho in all_corr if finite(rho)}
    best_predictor = max(abs_spearman, key=abs_spearman.get) if abs_spearman else "unknown"
    lambda_rho = next((rho for scope, name, _n, _r, rho in corr_rows if scope == "all" and name == "lambda"), float("nan"))
    fro_rho = next((rho for scope, name, _n, _r, rho in corr_rows if scope == "all" and name == "lambda_eff_fro"), float("nan"))
    max_rho = next((rho for scope, name, _n, _r, rho in corr_rows if scope == "all" and name == "lambda_eff_lambda_max"), float("nan"))

    lines = [
        "## Diagnostic Answers",
        "",
        "- Are true/permuted/random graph priors matched in Laplacian scale? Partly. True and permuted are exactly matched in scale because permutation relabels node states against the same Laplacian spectrum. Random controls match node count, edge count, and trace, but not degree dispersion, Frobenius norm, lambda_max, or spectral gap.",
    ]
    lines.extend(f"  {line}" for line in mismatch_lines)
    if outperform:
        examples = "; ".join(
            f"{topology} lambda={prior_weight:g} {control} {fmt(control_mean)} < graph {fmt(graph_mean)}"
            for topology, prior_weight, control, graph_mean, control_mean in outperform[:8]
        )
        lines.append(
            "- Could random/permuted controls outperform because effective smoothing strength differs? Random controls can, in principle, because their spectral scale is not matched beyond edge count and trace. Permuted controls cannot be explained by Laplacian scale: their spectrum is identical to the true graph, so their wins point to node-label specificity being weak or to optimization/regularization effects. Observed H=32 control wins include: "
            + examples
            + "."
        )
    else:
        lines.append(
            "- Could random/permuted controls outperform because effective smoothing strength differs? Random controls could differ in effective strength, but this result set has no H=32 mean control wins over true graph."
        )
    lines.append(
        f"- Does H=32 correlate better with lambda or lambda_eff? Across all Cycle 4 graph-prior runs, Spearman rho is lambda={fmt(lambda_rho)}, lambda*fro={fmt(fro_rho)}, lambda*lambda_max={fmt(max_rho)}. By rank correlation the strongest listed predictor is `{best_predictor}`; the differences should be read cautiously because topology and lambda are confounded and the response is non-monotone."
    )
    lines.append(
        "- Is lambda sensitivity likely topology-specific or regularization-scale driven? The evidence points more toward regularization scale plus graph-control mismatch than a stable true-topology effect. Topology matters because scale metrics differ by graph family, but true specificity does not survive lambda changes and permuted controls sometimes win at identical Laplacian scale."
    )
    return lines


def build_report(results: dict[str, Any], results_path: Path, config_dir: Path, data_root: Path) -> str:
    records = extract_run_records(results)
    config_fallbacks = load_config_fallbacks(config_dir)
    run_metrics, run_metric_stds, group_metrics = build_metric_index(records, config_fallbacks, data_root)
    corr = correlation_rows(records, run_metrics)
    graph_records = [record for record in records if record.prior in GRAPH_PRIORS]
    ok_count = sum(1 for record in graph_records if record.status_ok)
    failures = [record.run_name for record in graph_records if not record.status_ok]
    result_label = results_path.relative_to(ROOT) if results_path.is_absolute() and results_path.is_relative_to(ROOT) else results_path

    lines = [
        "# Cycle 4 Graph Spectra And Loss Diagnostics",
        "",
        f"Results file: `{result_label}`",
        f"Schema version: `{results.get('schema_version')}`",
        f"Graph-prior success/failure count: {ok_count} ok / {len(failures)} failed",
        f"Config directory: `{config_dir.relative_to(ROOT) if config_dir.is_absolute() and config_dir.is_relative_to(ROOT) else config_dir}`",
        f"Data root: `{data_root.relative_to(ROOT) if data_root.is_absolute() and data_root.is_relative_to(ROOT) else data_root}`",
        "",
        "Analysis only: no training or ISO17 commands are run.",
        "",
        "## Extracted Run Fields",
        "",
        "| topology | prior | lambda | seed | H=32 rollout | final_train_loss | prior_loss_mean |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for record in sorted(graph_records, key=lambda item: (item.topology, item.prior, item.prior_weight, item.seed)):
        lines.append(
            f"| {record.topology} | {record.prior} | {record.prior_weight:g} | {record.seed} | "
            f"{fmt(record.h32_rollout_error)} | {fmt_sci(record.final_train_loss)} | {fmt(record.prior_loss_mean)} |"
        )

    lines.extend([
        "",
        "## Laplacian Scale Diagnostics",
        "",
        "For `graph` and `permuted_graph`, the table is the static topology Laplacian. For `random_graph`, it is averaged over the reconstructed training-frame random graphs used by the prior loss.",
        "",
        "| topology | prior | n_nodes | n_edges | degree mean | degree std | trace | fro norm | lambda_max | spectral gap | cond/spread |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for topology, prior in sorted(group_metrics):
        metrics = group_metrics[(topology, prior)]
        lines.append(
            f"| {topology} | {prior} | {fmt(metrics.get('n_nodes'), 0)} | {fmt(metrics.get('n_edges'), 0)} | "
            f"{fmt(metrics.get('degree_mean'))} | {fmt(metrics.get('degree_std'))} | "
            f"{fmt(metrics.get('laplacian_trace'))} | {fmt(metrics.get('frobenius_norm'))} | "
            f"{fmt(metrics.get('lambda_max'))} | {fmt(metrics.get('spectral_gap'))} | "
            f"{fmt(metrics.get('condition_number'))} |"
        )

    lines.extend([
        "",
        "## Effective Prior Strength",
        "",
        "| topology | prior | lambda | H=32 mean +/- std | prior_loss_mean +/- std | lambda*fro | lambda*lambda_max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    grouped = grouped_records(records)
    for (topology, prior, prior_weight), group in sorted(grouped.items()):
        h32_mu, h32_sd, h32_n = mean_std(record.h32_rollout_error for record in group)
        loss_mu, loss_sd, _loss_n = mean_std(record.prior_loss_mean for record in group)
        scale = mean(run_metrics.get(record.run_name, {}).get("frobenius_norm", float("nan")) for record in group)
        lmax = mean(run_metrics.get(record.run_name, {}).get("lambda_max", float("nan")) for record in group)
        lines.append(
            f"| {topology} | {prior} | {prior_weight:g} | "
            f"{fmt(h32_mu)} +/- {fmt(h32_sd)} (n={h32_n}) | "
            f"{fmt(loss_mu)} +/- {fmt(loss_sd)} | "
            f"{fmt(prior_weight * scale)} | {fmt(prior_weight * lmax)} |"
        )

    lines.extend([
        "",
        "## H=32 Control Comparisons",
        "",
        "| topology | lambda | control | graph mean | control mean | graph advantage |",
        "|---|---:|---|---:|---:|---:|",
    ])
    for topology, prior_weight, control, graph_mean, control_mean in best_control_summary(records):
        lines.append(
            f"| {topology} | {prior_weight:g} | {control} | {fmt(graph_mean)} | {fmt(control_mean)} | {fmt_pct(pct_lower(graph_mean, control_mean))} |"
        )

    lines.extend([
        "",
        "## Correlation With H=32 Rollout",
        "",
        "Correlations use successful graph-prior runs. Positive values mean larger predictor values accompany larger H=32 rollout error.",
        "",
        "| scope | predictor | n | Pearson r | Spearman rho |",
        "|---|---|---:|---:|---:|",
    ])
    for scope, predictor, n, r, rho in corr:
        lines.append(f"| {scope} | {predictor} | {n} | {fmt(r)} | {fmt(rho)} |")

    lines.extend([""])
    lines.extend(narrative_answers(records, group_metrics, corr))

    lines.extend([
        "",
        "## Implementation Notes",
        "",
        "- `permuted_graph` is analyzed as a relabeled true-graph Laplacian: spectral scale is unchanged even though node-label alignment is destroyed in the loss.",
        "- `random_graph` uses the same random-edge seed formula as training: `3001 + seed * 100000 + frame_idx`, with frame indices reconstructed from `n_transitions`, `stride`, `horizon`, and `seed`.",
        "- HO graph edge weights are unit weights in the training loader, so trace is `2 * n_edges` for all true/permuted/random graphs with the same topology edge count.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Cycle 4 graph spectra, effective prior strength, and loss diagnostics.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    results = load_json(args.results)
    report = build_report(results, args.results, args.config_dir, args.data_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
