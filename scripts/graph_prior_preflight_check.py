from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_cycle8_latent_alignment import artifact_metrics
from scripts.train_cycle0 import (
    Cycle0LatentDynamics,
    collect_transitions,
    evaluate_rollout,
    latent_diagnostics,
    move_obs,
    select_device,
    set_seed,
    train_batch,
)
from scripts.train_cycle3_ho_networks import (
    Cycle3HOConfig,
    HOTrajectory,
    config_hash,
    get_git_commit,
    laplacian_from_edges,
    random_edge_pairs_np,
    sampled_frame_indices,
)


DEFAULT_REPORT = ROOT / "analysis_out" / "GRAPH_PRIOR_PREFLIGHT_REPORT.md"
DEFAULT_SUMMARY = ROOT / "analysis_out" / "graph_prior_preflight_summary.csv"
DEFAULT_ARTIFACT_DIR = ROOT / "experiments" / "artifacts" / "graph_prior_preflight"
DEFAULT_DATA_ROOT = ROOT / "data" / "ho_raw"
SCHEMA_VERSION = "graph_prior_preflight_v2"
TRACE_SEED = 4242


@dataclass(frozen=True)
class GraphControl:
    graph_type: str
    laplacian: np.ndarray
    permutation: np.ndarray | None = None
    seed: int | None = None


class DatasetAdapter(Protocol):
    """Adapter for node-wise dynamics datasets used by this preflight tool."""

    name: str
    n_nodes: int
    n_frames: int

    def sample_transitions(
        self,
        *,
        n_transitions: int,
        stride: int,
        horizon: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        ...

    def true_laplacian(self) -> np.ndarray:
        ...

    def metadata(self) -> dict[str, Any]:
        ...

    def coordinates(self, frame_idx: int) -> np.ndarray:
        ...

    def training_config(
        self,
        args: argparse.Namespace,
        *,
        prior: str,
        seed: int,
    ) -> Any:
        ...

    def __getitem__(self, idx: int) -> Any:
        ...


class HOAdapter:
    """HO topology adapter backed by the existing HOTrajectory/Cycle3 trainer path."""

    def __init__(self, dataset: str, data_root: Path):
        if not dataset.startswith("ho_"):
            raise ValueError(f"HO adapter expected dataset name like 'ho_lattice', got {dataset!r}")
        topology = dataset.removeprefix("ho_")
        if topology not in {"lattice", "random", "scalefree"}:
            raise ValueError("supported HO datasets are ho_lattice, ho_random, and ho_scalefree")
        self.name = dataset
        self.topology = topology
        self.traj = HOTrajectory(topology, data_root=data_root)
        self.n_nodes = int(self.traj.n_atoms)
        self.n_frames = int(self.traj.n_frames)
        self.n_edges = int(self.traj.edges.shape[0])

    def __getitem__(self, idx: int) -> Any:
        return self.traj[idx]

    def sample_transitions(
        self,
        *,
        n_transitions: int,
        stride: int,
        horizon: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        return collect_transitions(
            self.traj,
            n_transitions=n_transitions,
            stride=stride,
            horizon=horizon,
            seed=seed,
        )

    def true_laplacian(self) -> np.ndarray:
        return laplacian_from_edges(self.n_nodes, self.traj.edges)

    def coordinates(self, frame_idx: int) -> np.ndarray:
        return np.asarray(self.traj.coords[frame_idx], dtype=np.float64)

    def training_config(
        self,
        args: argparse.Namespace,
        *,
        prior: str,
        seed: int,
    ) -> Cycle3HOConfig:
        return Cycle3HOConfig(
            run_name=(
                f"preflight_{self.name}_gnn_{prior}_"
                f"lambda{str(args.prior_weight).replace('.', 'p')}_seed{seed}"
            ),
            topology=self.topology,
            encoder="gnn_node",
            prior=prior,
            seed=int(seed),
            num_epochs=int(args.epochs),
            n_transitions=int(args.train_transitions),
            eval_n_transitions=int(args.eval_transitions),
            stride=int(args.train_stride),
            eval_stride=int(args.eval_stride),
            horizon=1,
            eval_horizons=tuple(int(value) for value in args.horizons),
            latent_dim=16,
            node_dim=16,
            hidden_dim=64,
            transition_hidden_dim=64,
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            prior_weight=float(args.prior_weight),
            device=args.device,
            data_root=str(args.data_root),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "dataset": self.name,
            "domain": "HO",
            "topology": self.topology,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_frames": self.n_frames,
            "npz_path": str(self.traj.npz_path),
        }


def grid_edges(n_nodes: int) -> np.ndarray:
    side = int(round(math.sqrt(n_nodes)))
    if side * side != n_nodes:
        raise ValueError("lattice graph_heat topology requires N to be a perfect square")
    edges: list[tuple[int, int]] = []
    for row in range(side):
        for col in range(side):
            idx = row * side + col
            if col + 1 < side:
                edges.append((idx, idx + 1))
            if row + 1 < side:
                edges.append((idx, idx + side))
    return np.asarray(edges, dtype=np.int64)


def graph_heat_edges(topology: str, n_nodes: int, seed: int) -> np.ndarray:
    if topology == "chain":
        return np.asarray([(idx, idx + 1) for idx in range(n_nodes - 1)], dtype=np.int64)
    if topology == "ring":
        edges = [(idx, idx + 1) for idx in range(n_nodes - 1)]
        edges.append((n_nodes - 1, 0))
        return np.asarray(edges, dtype=np.int64)
    if topology == "lattice":
        return grid_edges(n_nodes)
    if topology == "random":
        n_edges = max(n_nodes - 1, 2 * n_nodes)
        return random_edge_pairs_np(n_nodes, n_edges, seed=seed)
    if topology == "scalefree":
        rng = np.random.default_rng(seed)
        edges: set[tuple[int, int]] = {(0, 1), (1, 2), (0, 2)}
        degrees = np.zeros(n_nodes, dtype=np.float64)
        for src, dst in edges:
            degrees[src] += 1.0
            degrees[dst] += 1.0
        for node in range(3, n_nodes):
            probs = degrees[:node] + 1.0
            probs = probs / probs.sum()
            targets = rng.choice(node, size=min(2, node), replace=False, p=probs)
            for target in targets:
                pair = (int(target), node) if int(target) < node else (node, int(target))
                edges.add(pair)
                degrees[pair[0]] += 1.0
                degrees[pair[1]] += 1.0
        return np.asarray(sorted(edges), dtype=np.int64)
    raise ValueError("graph_heat topology must be one of chain, ring, lattice, random, scalefree")


class GraphHeatAdapter:
    """Synthetic graph heat dynamics: X[t+1] = X[t] - tau L X[t] + noise eps."""

    def __init__(
        self,
        *,
        topology: str,
        n_nodes: int = 36,
        n_dim: int = 4,
        n_frames: int = 512,
        tau: float = 0.05,
        noise: float = 0.01,
        seed: int = 0,
    ):
        if n_dim != 4:
            raise ValueError("graph_heat currently requires d=4 to match the existing GNN input width")
        self.name = f"graph_heat_{topology}"
        self.topology = topology
        self.n_nodes = int(n_nodes)
        self.n_dim = int(n_dim)
        self.n_frames = int(n_frames)
        self.tau = float(tau)
        self.noise = float(noise)
        self.seed = int(seed)
        self.edges = graph_heat_edges(topology, self.n_nodes, self.seed)
        self.laplacian = laplacian_from_edges(self.n_nodes, self.edges)
        self.coords = self._simulate()
        self.energies = np.sum(self.coords * self.coords, axis=(1, 2)).astype(np.float32)

    def _simulate(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        coords = np.zeros((self.n_frames, self.n_nodes, self.n_dim), dtype=np.float32)
        coords[0] = rng.normal(size=(self.n_nodes, self.n_dim)).astype(np.float32)
        for frame_idx in range(self.n_frames - 1):
            drift = self.laplacian @ coords[frame_idx]
            eps = rng.normal(size=(self.n_nodes, self.n_dim)).astype(np.float32)
            coords[frame_idx + 1] = coords[frame_idx] - self.tau * drift.astype(np.float32) + self.noise * eps
        return coords

    def __getitem__(self, idx: int) -> HeteroData:
        x = torch.from_numpy(self.coords[idx])
        pos = x[:, :3].to(dtype=torch.float32)
        atomic_numbers = (10.0 * x[:, 3]).to(dtype=torch.float32)
        directed = np.concatenate([self.edges, self.edges[:, ::-1]], axis=0)
        edge_index = torch.from_numpy(directed.T.copy()).to(dtype=torch.long)
        edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float32)

        data = HeteroData()
        data["atom"].pos = pos
        data["atom"].atomic_number = atomic_numbers
        data["atom"].x = torch.cat([atomic_numbers.view(-1, 1) / 10.0, pos], dim=-1)
        data["atom", "bonded", "atom"].edge_index = edge_index
        data["atom", "bonded", "atom"].edge_attr = edge_attr
        return data

    def sample_transitions(
        self,
        *,
        n_transitions: int,
        stride: int,
        horizon: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        frame_indices = sampled_frame_indices(self.n_frames, n_transitions, stride, horizon, seed)
        transitions = []
        for frame_idx in frame_indices:
            obs, next_obs, energy, _force = self.get_pair(frame_idx, horizon=horizon)
            transitions.append(
                {
                    "obs": obs,
                    "next_obs": next_obs,
                    "frame_idx": int(frame_idx),
                    "horizon": int(horizon),
                    "energy": float(energy),
                    "molecule": self.name,
                }
            )
        return transitions

    def get_pair(self, idx: int, horizon: int = 1) -> tuple[HeteroData, HeteroData, float, None]:
        if idx + horizon >= self.n_frames:
            raise IndexError(f"idx+horizon={idx + horizon} >= n_frames={self.n_frames}")
        return self[idx], self[idx + horizon], float(self.energies[idx]), None

    def true_laplacian(self) -> np.ndarray:
        return self.laplacian

    def coordinates(self, frame_idx: int) -> np.ndarray:
        return np.asarray(self.coords[frame_idx], dtype=np.float64)

    def training_config(
        self,
        args: argparse.Namespace,
        *,
        prior: str,
        seed: int,
    ) -> Cycle3HOConfig:
        return Cycle3HOConfig(
            run_name=(
                f"preflight_{self.name}_gnn_{prior}_"
                f"lambda{str(args.prior_weight).replace('.', 'p')}_seed{seed}"
            ),
            topology=self.topology,
            encoder="gnn_node",
            prior=prior,
            seed=int(seed),
            num_epochs=int(args.epochs),
            n_transitions=int(args.train_transitions),
            eval_n_transitions=int(args.eval_transitions),
            stride=int(args.train_stride),
            eval_stride=int(args.eval_stride),
            horizon=1,
            eval_horizons=tuple(int(value) for value in args.horizons),
            latent_dim=16,
            node_dim=16,
            hidden_dim=64,
            transition_hidden_dim=64,
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            prior_weight=float(args.prior_weight),
            device=args.device,
            data_root=str(args.data_root),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "dataset": self.name,
            "domain": "graph_heat",
            "topology": self.topology,
            "n_nodes": self.n_nodes,
            "n_edges": int(self.edges.shape[0]),
            "n_frames": self.n_frames,
            "n_dim": self.n_dim,
            "tau": self.tau,
            "noise": self.noise,
            "seed": self.seed,
        }


def build_adapter(args: argparse.Namespace) -> DatasetAdapter:
    dataset = str(args.dataset)
    if dataset.startswith("ho_"):
        return HOAdapter(dataset, args.data_root)
    if dataset == "graph_heat" or dataset.startswith("graph_heat_"):
        topology = args.topology or (dataset.removeprefix("graph_heat_") if dataset.startswith("graph_heat_") else "lattice")
        return GraphHeatAdapter(
            topology=topology,
            n_nodes=args.graph_heat_n,
            n_dim=args.graph_heat_d,
            n_frames=args.graph_heat_t,
            tau=args.graph_heat_tau,
            noise=args.graph_heat_noise,
            seed=args.graph_heat_seed,
        )
    raise ValueError(
        f"Unknown dataset {dataset!r}. Available: ho_lattice, ho_random, ho_scalefree, "
        "graph_heat, graph_heat_<topology>"
    )


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean(values: list[float]) -> float:
    vals = [float(value) for value in values if finite(value)]
    return float(np.mean(vals)) if vals else float("nan")


def mean_std(values: list[float]) -> tuple[float, float, int]:
    vals = np.asarray([float(value) for value in values if finite(value)], dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan"), 0
    return float(vals.mean()), float(vals.std(ddof=1)) if vals.size > 1 else 0.0, int(vals.size)


def fmt(value: Any, digits: int = 4) -> str:
    if not finite(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_mean_std(values: list[float], digits: int = 4) -> str:
    m, s, n = mean_std(values)
    return f"{fmt(m, digits)} +/- {fmt(s, digits)} (n={n})"


def fmt_pct(value: Any) -> str:
    if not finite(value):
        return "NA"
    return f"{float(value):+.1f}%"


def rel_to_root(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def pct_gain(base: float, candidate: float) -> float:
    if not finite(base) or not finite(candidate) or float(base) == 0.0:
        return float("nan")
    return 100.0 * (float(base) - float(candidate)) / float(base)


def pearson(x: list[float], y: list[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 3 or np.std(x_arr) <= 0 or np.std(y_arr) <= 0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def centered(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix - matrix.mean(axis=0, keepdims=True)


def torch_permutation(n_nodes: int, seed: int) -> np.ndarray:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(n_nodes, generator=generator).numpy().astype(np.int64)


def edge_count_from_laplacian(laplacian: np.ndarray) -> int:
    off_diag = np.triu(np.asarray(laplacian) < -1e-12, k=1)
    return int(np.count_nonzero(off_diag))


def weighted_laplacian_from_pairs(n_nodes: int, edges: np.ndarray, weight: float = 1.0) -> np.ndarray:
    laplacian = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for src, dst in np.asarray(edges, dtype=np.int64):
        if int(src) == int(dst):
            continue
        laplacian[int(src), int(src)] += float(weight)
        laplacian[int(dst), int(dst)] += float(weight)
        laplacian[int(src), int(dst)] -= float(weight)
        laplacian[int(dst), int(src)] -= float(weight)
    return laplacian


def mean_edge_weight_from_laplacian(laplacian: np.ndarray) -> float:
    weights = -np.asarray(laplacian, dtype=np.float64)[np.triu_indices_from(laplacian, k=1)]
    weights = weights[weights > 1e-12]
    return float(weights.mean()) if weights.size else 1.0


def generate_graph_controls(
    adapter: DatasetAdapter,
    *,
    n_permuted: int,
    n_random: int,
    include_random: bool = True,
) -> list[GraphControl]:
    true_laplacian = adapter.true_laplacian()
    n_nodes = int(adapter.n_nodes)
    n_edges = edge_count_from_laplacian(true_laplacian)
    mean_weight = mean_edge_weight_from_laplacian(true_laplacian)
    controls = [GraphControl("true_graph", true_laplacian)]
    controls.extend(
        GraphControl(
            "permuted_graph",
            true_laplacian,
            permutation=torch_permutation(n_nodes, 2003 + int(seed) * 100000),
            seed=int(seed),
        )
        for seed in range(int(n_permuted))
    )
    if include_random:
        controls.extend(
            GraphControl(
                "random_graph",
                weighted_laplacian_from_pairs(
                    n_nodes,
                    random_edge_pairs_np(n_nodes, n_edges, seed=3001 + int(seed)),
                    weight=mean_weight,
                ),
                seed=int(seed),
            )
            for seed in range(int(n_random))
        )
    return controls


def normalize_laplacian(laplacian: np.ndarray, mode: str) -> np.ndarray:
    laplacian = np.asarray(laplacian, dtype=np.float64)
    if mode == "none":
        return laplacian
    if mode == "trace":
        scale = float(np.trace(laplacian))
    elif mode == "lambda_max":
        eigvals = np.linalg.eigvalsh(laplacian)
        scale = float(np.max(eigvals)) if eigvals.size else float("nan")
    else:
        raise ValueError(f"unknown Laplacian normalization {mode!r}")
    if not finite(scale) or abs(scale) < 1e-12:
        return laplacian
    return laplacian / scale


def dirichlet(matrix: np.ndarray, laplacian: np.ndarray) -> float:
    return float(np.trace(matrix.T @ laplacian @ matrix))


def low_frequency_ratios(matrix: np.ndarray, laplacian: np.ndarray) -> dict[int, float]:
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order]
    coeff = eigvecs.T @ matrix
    energy = np.sum(coeff * coeff, axis=1)
    total = float(np.sum(energy))
    out: dict[int, float] = {}
    for k in (2, 4, 8):
        if k < eigvecs.shape[1] and total > 1e-12:
            out[k] = float(np.sum(energy[:k]) / total)
        else:
            out[k] = float("nan")
    return out


def graph_metrics(laplacian: np.ndarray) -> dict[str, float]:
    eigvals = np.linalg.eigvalsh(laplacian)
    eigvals = np.clip(eigvals, 0.0, None)
    return {
        "trace": float(np.trace(laplacian)),
        "frobenius_norm": float(np.linalg.norm(laplacian, ord="fro")),
        "lambda_max": float(eigvals[-1]) if eigvals.size else float("nan"),
        "spectral_gap": float(eigvals[1]) if eigvals.size > 1 else float("nan"),
    }


def raw_alignment_for_controls(
    adapter: DatasetAdapter,
    *,
    controls: list[GraphControl],
    frame_indices: list[int],
    normalization: str,
) -> dict[str, Any]:
    d_dx_values: list[float] = []
    dx_norm_values: list[float] = []
    r_low: dict[int, list[float]] = defaultdict(list)
    metrics: dict[str, list[float]] = defaultdict(list)
    if not controls:
        raise ValueError("raw alignment requires at least one graph control")
    for control in controls:
        laplacian = normalize_laplacian(control.laplacian, normalization)
        for key, value in graph_metrics(laplacian).items():
            metrics[key].append(value)
        for frame_idx in frame_indices:
            x = centered(adapter.coordinates(frame_idx))
            x_next = centered(adapter.coordinates(frame_idx + 1))
            dx = x_next - x
            dx_eval = dx[control.permutation] if control.permutation is not None else dx
            d_dx_values.append(dirichlet(dx_eval, laplacian))
            dx_norm_values.append(float(np.sum(dx_eval * dx_eval)))
            for k, value in low_frequency_ratios(dx_eval, laplacian).items():
                r_low[k].append(value)
    d_dx = mean(d_dx_values)
    dx_norm = mean(dx_norm_values)
    out: dict[str, Any] = {
        "stage": "stage0_raw",
        "dataset": adapter.name,
        "graph_type": controls[0].graph_type,
        "normalization": normalization,
        "n_graphs": len(controls),
        "n_frames": len(frame_indices),
        "D_dX_norm": d_dx / dx_norm if finite(d_dx) and finite(dx_norm) and dx_norm > 0 else float("nan"),
    }
    for key, values in metrics.items():
        out[key] = mean(values)
    for k in (2, 4, 8):
        out[f"R_low_dX_{k}"] = mean(r_low[k])
    return out


def stage0_raw_diagnostics(args: argparse.Namespace, adapter: DatasetAdapter) -> list[dict[str, Any]]:
    frame_indices = sampled_frame_indices(
        adapter.n_frames,
        args.raw_transitions,
        args.raw_stride,
        1,
        args.raw_seed,
    )
    controls = generate_graph_controls(
        adapter,
        n_permuted=args.n_permuted,
        n_random=args.n_random,
        include_random=not args.skip_random_graph,
    )
    grouped_controls: dict[str, list[GraphControl]] = defaultdict(list)
    for control in controls:
        grouped_controls[control.graph_type].append(control)
    rows = [
        raw_alignment_for_controls(
            adapter,
            controls=grouped_controls[graph_type],
            frame_indices=frame_indices,
            normalization=args.laplacian_normalization,
        )
        for graph_type in ("true_graph", "permuted_graph", "random_graph")
        if grouped_controls.get(graph_type)
    ]

    by_type = {row["graph_type"]: row for row in rows}
    true = by_type["true_graph"]
    for control in ("permuted_graph", "random_graph"):
        if control not in by_type:
            continue
        row = by_type[control]
        row["gap_D_dX_norm_control_minus_true"] = row["D_dX_norm"] - true["D_dX_norm"]
        for k in (2, 4, 8):
            row[f"gap_R_low_dX_{k}_true_minus_control"] = true[f"R_low_dX_{k}"] - row[f"R_low_dX_{k}"]
    true["gap_D_dX_norm_control_minus_true"] = 0.0
    for k in (2, 4, 8):
        true[f"gap_R_low_dX_{k}_true_minus_control"] = 0.0
    return rows


def mini_train(
    config: Cycle3HOConfig,
    adapter: DatasetAdapter,
) -> tuple[dict[str, Any], Cycle0LatentDynamics | None, torch.device]:
    set_seed(config.seed)
    device = select_device(config.device)
    train_transitions = adapter.sample_transitions(
        n_transitions=config.n_transitions,
        stride=config.stride,
        horizon=config.horizon,
        seed=config.seed,
    )
    eval_transitions = adapter.sample_transitions(
        n_transitions=config.eval_n_transitions,
        stride=config.eval_stride,
        horizon=max(config.eval_horizons),
        seed=config.seed + 1000,
    )
    model = Cycle0LatentDynamics(config, n_atoms=adapter.n_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    epoch_losses: list[float] = []
    prior_values: list[float] = []
    transition_values: list[float] = []
    start_time = time.time()
    for epoch in range(config.num_epochs):
        model.train()
        order = np.random.permutation(len(train_transitions))
        epoch_total: list[float] = []
        for start_idx in range(0, len(order), config.batch_size):
            batch = [train_transitions[int(index)] for index in order[start_idx : start_idx + config.batch_size]]
            optimizer.zero_grad()
            total, transition_loss, prior_loss, _z_batch = train_batch(model, batch, config, device)
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss in {config.run_name} at epoch {epoch + 1}")
            total.backward()
            optimizer.step()
            epoch_total.append(float(total.detach().cpu()))
            prior_values.append(float(prior_loss.detach().cpu()))
            transition_values.append(float(transition_loss.detach().cpu()))
        epoch_losses.append(mean(epoch_total))
        print(
            f"[preflight|{config.run_name}] epoch {epoch + 1}/{config.num_epochs} "
            f"loss={epoch_losses[-1]:.6f}",
            flush=True,
        )
    rollout_errors = evaluate_rollout(model, adapter, eval_transitions, config.eval_horizons, device, config.latent_dim)  # type: ignore[arg-type]
    z_diag = []
    with torch.no_grad():
        model.eval()
        for transition in train_transitions[: min(32, len(train_transitions))]:
            z, _h = model.encode(move_obs(transition["obs"], device))
            z_diag.append(z.detach().cpu().numpy())
    diagnostics = latent_diagnostics(np.stack(z_diag, axis=0), seed=config.seed)
    diagnostics.update(
        {
            "rollout_errors": rollout_errors,
            "final_train_loss": float(epoch_losses[-1]) if epoch_losses else float("nan"),
            "prior_loss_mean": mean(prior_values),
            "transition_loss_mean": mean(transition_values),
            "final_transition_loss": mean(transition_values[-max(1, math.ceil(len(transition_values) / max(1, config.num_epochs))) :]),
        }
    )
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "run_name": config.run_name,
            "stage": "stage1_mini_train",
            "dataset": adapter.name,
            "status": "ok",
            "topology": config.topology,
            "encoder": config.encoder,
            "prior": config.prior,
            "prior_weight": config.prior_weight,
            "seed": config.seed,
            "config_hash": config_hash(config),
            "git_commit": get_git_commit(),
            "config": asdict(config),
            "diagnostics": diagnostics,
            **{f"H{horizon}": rollout_errors.get(str(horizon)) for horizon in config.eval_horizons},
            "final_train_loss": diagnostics["final_train_loss"],
            "prior_loss_mean": diagnostics["prior_loss_mean"],
            "wall_time_sec": time.time() - start_time,
        },
        model,
        device,
    )


def stage1_mini_training(
    args: argparse.Namespace,
    adapter: DatasetAdapter,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], Cycle0LatentDynamics], dict[int, torch.device]]:
    rows: list[dict[str, Any]] = []
    models: dict[tuple[str, int], Cycle0LatentDynamics] = {}
    devices: dict[int, torch.device] = {}
    for seed in args.seeds:
        for prior in ("none", "graph", "permuted_graph"):
            config = adapter.training_config(args, prior=prior, seed=int(seed))
            try:
                row, model, device = mini_train(config, adapter)
                rows.append(row)
                if prior in {"graph", "permuted_graph"}:
                    assert model is not None
                    models[(prior, int(seed))] = model
                    devices[int(seed)] = device
            except Exception as exc:
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_name": config.run_name,
                        "stage": "stage1_mini_train",
                        "dataset": adapter.name,
                        "status": "failed",
                        "topology": config.topology,
                        "encoder": config.encoder,
                        "prior": prior,
                        "prior_weight": config.prior_weight,
                        "seed": int(seed),
                        "error": str(exc),
                        "H16": float("nan"),
                        "H32": float("nan"),
                    }
                )
                print(f"FAILED {config.run_name}: {exc}", flush=True)
    return rows, models, devices


def grouped_rollout_means(rows: list[dict[str, Any]], horizon: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    key = f"H{horizon}"
    for row in rows:
        if row.get("stage") == "stage1_mini_train" and row.get("status") == "ok":
            values[str(row["prior"])].append(float(row.get(key, float("nan"))))
    return {prior: mean(vals) for prior, vals in values.items()}


def should_run_latent_audit(stage1_rows: list[dict[str, Any]]) -> bool:
    h32 = grouped_rollout_means(stage1_rows, "32")
    return finite(h32.get("graph")) and finite(h32.get("permuted_graph")) and h32["graph"] < h32["permuted_graph"]


def laplacian_tensor(laplacian: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(laplacian, dtype=np.float32))


def collect_preflight_latent_trace(
    model: Cycle0LatentDynamics,
    config: Cycle3HOConfig,
    adapter: DatasetAdapter,
    device: torch.device,
) -> dict[str, Any]:
    transitions = adapter.sample_transitions(
        n_transitions=config.eval_n_transitions,
        stride=config.eval_stride,
        horizon=1,
        seed=TRACE_SEED,
    )
    true_l = adapter.true_laplacian()
    true_l_tensor = laplacian_tensor(true_l)
    n_nodes = int(adapter.n_nodes)
    n_edges = edge_count_from_laplacian(true_l)
    mean_weight = mean_edge_weight_from_laplacian(true_l)

    model.eval()
    h_t_values: list[torch.Tensor] = []
    h_tp1_values: list[torch.Tensor] = []
    frame_indices: list[int] = []
    prior_laplacians: list[torch.Tensor] = []
    random_laplacians: list[torch.Tensor] = []
    prior_perms: list[torch.Tensor] = []
    control_perms: list[torch.Tensor] = []
    identity = torch.arange(n_nodes, dtype=torch.long)

    with torch.no_grad():
        for transition in transitions:
            frame_idx = int(transition["frame_idx"])
            obs = move_obs(transition["obs"], device)
            next_obs = move_obs(transition["next_obs"], device)
            _z_t, h_t = model.encode(obs)
            _z_tp1, h_tp1 = model.encode(next_obs)
            if h_t is None or h_tp1 is None:
                raise RuntimeError("latent audit requires an encoder that emits node states")
            h_t_values.append(h_t.detach().cpu())
            h_tp1_values.append(h_tp1.detach().cpu())
            frame_indices.append(frame_idx)

            perm = torch.from_numpy(torch_permutation(n_nodes, 2003 + int(config.seed) * 100000 + frame_idx))
            random_edges = random_edge_pairs_np(n_nodes, n_edges, seed=3001 + int(config.seed) * 100000 + frame_idx)
            random_l = laplacian_tensor(weighted_laplacian_from_pairs(n_nodes, random_edges, weight=mean_weight))
            random_laplacians.append(random_l)
            control_perms.append(perm)
            if config.prior == "permuted_graph":
                prior_laplacians.append(true_l_tensor)
                prior_perms.append(perm)
            elif config.prior == "random_graph":
                prior_laplacians.append(random_l)
                prior_perms.append(identity)
            else:
                prior_laplacians.append(true_l_tensor)
                prior_perms.append(identity)

    h_t = torch.stack(h_t_values, dim=0)
    h_tp1 = torch.stack(h_tp1_values, dim=0)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": config.run_name,
        "dataset": adapter.name,
        "topology": config.topology,
        "prior": config.prior,
        "prior_weight": float(config.prior_weight),
        "seed": int(config.seed),
        "trace_seed": TRACE_SEED,
        "frame_idx": torch.tensor(frame_indices, dtype=torch.long),
        "H_t": h_t,
        "H_tplus1": h_tp1,
        "Delta_H": h_tp1 - h_t,
        "true_laplacian": true_l_tensor,
        "prior_laplacians": torch.stack(prior_laplacians, dim=0),
        "random_laplacians": torch.stack(random_laplacians, dim=0),
        "prior_permutation_indices": torch.stack(prior_perms, dim=0),
        "control_permutation_indices": torch.stack(control_perms, dim=0),
        "dataset_metadata": adapter.metadata(),
    }


def stage2_latent_audit(
    args: argparse.Namespace,
    adapter: DatasetAdapter,
    models: dict[tuple[str, int], Cycle0LatentDynamics],
    devices: dict[int, torch.device],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for prior in ("graph", "permuted_graph"):
            model = models.get((prior, int(seed)))
            device = devices.get(int(seed))
            if model is None or device is None:
                continue
            config = adapter.training_config(args, prior=prior, seed=int(seed))
            artifact = collect_preflight_latent_trace(model, config, adapter, device)
            artifact_path = args.artifact_dir / f"{config.run_name}_latents.pt"
            torch.save(artifact, artifact_path)
            metrics = artifact_metrics(artifact)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "stage": "stage2_latent_audit",
                    "run_name": config.run_name,
                    "dataset": adapter.name,
                    "status": "ok",
                    "topology": config.topology,
                    "prior": prior,
                    "prior_weight": config.prior_weight,
                    "seed": int(seed),
                    "latent_artifact_path": rel_to_root(artifact_path),
                    **metrics,
                }
            )
    return rows


def classify(stage1_rows: list[dict[str, Any]], stage2_rows: list[dict[str, Any]]) -> tuple[str, dict[str, float]]:
    h16 = grouped_rollout_means(stage1_rows, "16")
    h32 = grouped_rollout_means(stage1_rows, "32")
    graph_gain = pct_gain(h32.get("none", float("nan")), h32.get("graph", float("nan")))
    specificity_gain = pct_gain(h32.get("permuted_graph", float("nan")), h32.get("graph", float("nan")))
    metrics = {
        "graph_gain_h32_pct": graph_gain,
        "true_vs_permuted_gain_h32_pct": specificity_gain,
        "none_h32": h32.get("none", float("nan")),
        "graph_h32": h32.get("graph", float("nan")),
        "permuted_h32": h32.get("permuted_graph", float("nan")),
        "none_h16": h16.get("none", float("nan")),
        "graph_h16": h16.get("graph", float("nan")),
        "permuted_h16": h16.get("permuted_graph", float("nan")),
    }
    if not finite(graph_gain) or graph_gain <= 0.0:
        return "no_graph_gain", metrics
    if not finite(specificity_gain) or specificity_gain <= 0.0:
        return "generic_smoothing", metrics
    if not stage2_rows:
        return "candidate_topology_specific", metrics
    by_prior: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage2_rows:
        by_prior[str(row.get("prior"))].append(row)
    graph_d = mean([row["D_true_Delta_H_norm"] for row in by_prior.get("graph", [])])
    perm_d = mean([row["D_true_Delta_H_norm"] for row in by_prior.get("permuted_graph", [])])
    graph_r = mean([row["R_low_true_Delta_H_4"] for row in by_prior.get("graph", [])])
    perm_r = mean([row["R_low_true_Delta_H_4"] for row in by_prior.get("permuted_graph", [])])
    metrics.update(
        {
            "graph_D_true_Delta_H_norm": graph_d,
            "permuted_D_true_Delta_H_norm": perm_d,
            "graph_R_low_true_Delta_H_4": graph_r,
            "permuted_R_low_true_Delta_H_4": perm_r,
        }
    )
    if finite(graph_d) and finite(perm_d) and graph_d < perm_d and finite(graph_r) and finite(perm_r) and graph_r > perm_r:
        return "topology_aligned_latent_smoothing", metrics
    return "candidate_topology_specific", metrics


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_summary_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def graph_heat_oracle_metrics(args: argparse.Namespace, adapter: DatasetAdapter) -> dict[str, Any]:
    meta = adapter.metadata()
    if meta.get("domain") != "graph_heat":
        return {}
    horizons = tuple(sorted({1, *(int(value) for value in args.horizons)}))
    max_horizon = max(horizons)
    frame_indices = sampled_frame_indices(
        adapter.n_frames,
        args.eval_transitions,
        args.eval_stride,
        max_horizon,
        int(args.seeds[0]) + 1000,
    )
    laplacian = adapter.true_laplacian()
    tau = float(meta["tau"])
    mse_by_horizon: dict[int, list[float]] = {horizon: [] for horizon in horizons}
    for frame_idx in frame_indices:
        pred = np.asarray(adapter.coordinates(frame_idx), dtype=np.float64).copy()
        for step in range(1, max_horizon + 1):
            pred = pred - tau * (laplacian @ pred)
            if step in mse_by_horizon:
                target = np.asarray(adapter.coordinates(frame_idx + step), dtype=np.float64)
                mse_by_horizon[step].append(float(np.mean((pred - target) ** 2)))
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "oracle_heat_baseline",
        "dataset": adapter.name,
        "topology": meta.get("topology"),
        "n_eval_transitions": len(frame_indices),
        "tau": tau,
        "noise": float(meta["noise"]),
    }
    for horizon in horizons:
        row[f"H{horizon}_mse"] = mean(mse_by_horizon[horizon])
    return row


def graph_heat_lambda_sanity(args: argparse.Namespace, adapter: DatasetAdapter) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if adapter.metadata().get("domain") != "graph_heat":
        return rows
    for lambda_value in (0.001, 0.01, 0.1):
        lambda_args = copy.copy(args)
        lambda_args.prior_weight = float(lambda_value)
        config = adapter.training_config(lambda_args, prior="graph", seed=0)
        try:
            row, _model, _device = mini_train(config, adapter)
            row["stage"] = "graph_heat_lambda_sanity"
            row["lambda_sanity"] = True
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "stage": "graph_heat_lambda_sanity",
                    "dataset": adapter.name,
                    "status": "failed",
                    "topology": adapter.metadata().get("topology"),
                    "encoder": "gnn_node",
                    "prior": "graph",
                    "prior_weight": float(lambda_value),
                    "seed": 0,
                    "error": str(exc),
                    "H16": float("nan"),
                    "H32": float("nan"),
                }
            )
    return rows


def report_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_report(
    path: Path,
    args: argparse.Namespace,
    raw_rows: list[dict[str, Any]],
    stage1_rows: list[dict[str, Any]],
    stage2_rows: list[dict[str, Any]],
    classification: str,
    class_metrics: dict[str, float],
    oracle_row: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_by_type = {str(row["graph_type"]): row for row in raw_rows}
    train_by_prior: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage1_rows:
        if row.get("status") == "ok":
            train_by_prior[str(row["prior"])].append(row)
    latent_by_prior: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage2_rows:
        if row.get("status") == "ok":
            latent_by_prior[str(row["prior"])].append(row)

    lines: list[str] = [
        "# Graph Prior Preflight Report",
        "",
        f"Created: `{datetime.now(timezone.utc).isoformat()}`",
        f"Schema version: `{SCHEMA_VERSION}`",
        "",
        "Scope: lightweight dataset-adapter preflight using the existing GNN trainer, graph prior loss, and rollout evaluator.",
        "No ISO17, rMD17 top-up, full sweeps, or large experiments were run.",
        "",
        "## Configuration",
        "",
        *report_table(
            ["field", "value"],
            [
                ["dataset", args.dataset],
                ["topology", getattr(args, "topology", "") or ""],
                ["prior_weight", args.prior_weight],
                ["seeds", ",".join(str(seed) for seed in args.seeds)],
                ["epochs", args.epochs],
                ["train transitions", args.train_transitions],
                ["eval transitions", args.eval_transitions],
                ["horizons", ",".join(str(value) for value in args.horizons)],
                ["raw normalization", args.laplacian_normalization],
                ["N permuted graphs", args.n_permuted],
                ["M random graphs", 0 if args.skip_random_graph else args.n_random],
            ],
        ),
        "",
        "## Stage 0: Raw Graph-Dynamics Diagnostic",
        "",
        *report_table(
            ["graph type", "D_dX_norm", "R_low K=2", "R_low K=4", "R_low K=8", "D gap control-true"],
            [
                [
                    graph_type,
                    fmt(row.get("D_dX_norm"), 6),
                    fmt(row.get("R_low_dX_2"), 4),
                    fmt(row.get("R_low_dX_4"), 4),
                    fmt(row.get("R_low_dX_8"), 4),
                    fmt(row.get("gap_D_dX_norm_control_minus_true"), 6),
                ]
                for graph_type, row in raw_by_type.items()
            ],
        ),
        "",
        "Positive `D gap control-true` means raw temporal changes are smoother on the true graph than on that control.",
        "",
        "## Stage 1: Mini Training",
        "",
        *report_table(
            ["prior", "H=16", "H=32", "final train loss", "prior loss mean"],
            [
                [
                    prior,
                    fmt_mean_std([row.get("H16") for row in rows], 4),
                    fmt_mean_std([row.get("H32") for row in rows], 4),
                    fmt_mean_std([row.get("final_train_loss") for row in rows], 6),
                    fmt_mean_std([row.get("prior_loss_mean") for row in rows], 4),
                ]
                for prior, rows in train_by_prior.items()
            ],
        ),
        "",
        f"Graph gain vs none at H=32: `{fmt_pct(class_metrics.get('graph_gain_h32_pct'))}`",
        f"True-vs-permuted gain at H=32: `{fmt_pct(class_metrics.get('true_vs_permuted_gain_h32_pct'))}`",
        "",
    ]
    if oracle_row:
        lines.extend(
            [
                "## Graph Heat Oracle Baseline",
                "",
                *report_table(
                    ["metric", "value"],
                    [
                        ["one-step MSE", fmt(oracle_row.get("H1_mse"), 8)],
                        ["H=16 rollout MSE", fmt(oracle_row.get("H16_mse"), 8)],
                        ["H=32 rollout MSE", fmt(oracle_row.get("H32_mse"), 8)],
                    ],
                ),
                "",
            ]
        )
    lines.extend(["## Stage 2: Latent Audit", ""])
    if not stage2_rows:
        lines.extend(
            [
                "Skipped because the mini true-graph run did not beat the permuted control at H=32.",
                "",
            ]
        )
    else:
        lines.extend(
            report_table(
                ["prior", "D_true(Delta_H) norm", "R_low true K=2", "R_low true K=4", "R_low true K=8"],
                [
                    [
                        prior,
                        fmt_mean_std([row.get("D_true_Delta_H_norm") for row in rows], 4),
                        fmt_mean_std([row.get("R_low_true_Delta_H_2") for row in rows], 4),
                        fmt_mean_std([row.get("R_low_true_Delta_H_4") for row in rows], 4),
                        fmt_mean_std([row.get("R_low_true_Delta_H_8") for row in rows], 4),
                    ]
                    for prior, rows in latent_by_prior.items()
                ],
            )
        )
        lines.extend([""])
        if "graph" in latent_by_prior and "permuted_graph" in latent_by_prior:
            graph_rows = sorted(latent_by_prior["graph"], key=lambda row: int(row["seed"]))
            perm_rows = sorted(latent_by_prior["permuted_graph"], key=lambda row: int(row["seed"]))
            paired = [
                perm["D_true_Delta_H_norm"] - graph["D_true_Delta_H_norm"]
                for graph, perm in zip(graph_rows, perm_rows)
                if int(graph["seed"]) == int(perm["seed"])
            ]
            lines.extend(
                [
                    f"Paired `D_true_norm(permuted - graph)`: `{fmt_mean_std(paired, 4)}`",
                    f"Graph lower latent-energy count: `{sum(1 for value in paired if value > 0)}/{len(paired)}`",
                    "",
                ]
            )
    if stage2_rows:
        h32_by_run = {
            (str(row["prior"]), int(row["seed"])): float(row["H32"])
            for row in stage1_rows
            if row.get("status") == "ok" and finite(row.get("H32"))
        }
        d_values = []
        h32_values = []
        for row in stage2_rows:
            key = (str(row["prior"]), int(row["seed"]))
            if key in h32_by_run:
                d_values.append(float(row["D_true_Delta_H_norm"]))
                h32_values.append(h32_by_run[key])
        lines.extend(
            [
                "## Latent-Energy Correlation",
                "",
                f"Pearson r between `D_true_norm(Delta_H)` and H=32 rollout over audited runs: `{fmt(pearson(d_values, h32_values), 4)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Final Classification",
            "",
            f"`{classification}`",
            "",
            "Decision rules:",
            "- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.",
            "- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.",
            "- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.",
            "- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.",
            "",
            "## Files",
            "",
            f"- `{rel_to_root(args.summary)}`",
        ]
    )
    if stage2_rows:
        lines.append(f"- latent artifacts under `{rel_to_root(args.artifact_dir)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rows_for_stage(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("stage") == stage and row.get("status", "ok") == "ok"]


def first_prior_row(rows: list[dict[str, Any]], prior: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("prior") == prior:
            return row
    return None


def write_graph_heat_failure_audit(
    path: Path,
    *,
    args: argparse.Namespace,
    raw_rows: list[dict[str, Any]],
    oracle_row: dict[str, Any],
    existing_rows: list[dict[str, Any]],
    lambda_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_by_type = {str(row.get("graph_type")): row for row in raw_rows}
    stage1_rows = rows_for_stage(existing_rows, "stage1_mini_train")
    none_row = first_prior_row(stage1_rows, "none")
    permuted_row = first_prior_row(stage1_rows, "permuted_graph")
    graph_row = first_prior_row(stage1_rows, "graph")

    true_d = raw_by_type.get("true_graph", {}).get("D_dX_norm")
    perm_d = raw_by_type.get("permuted_graph", {}).get("D_dX_norm")
    rand_d = raw_by_type.get("random_graph", {}).get("D_dX_norm")
    raw_smoother = finite(true_d) and finite(perm_d) and finite(rand_d) and float(true_d) < float(perm_d) and float(true_d) < float(rand_d)

    lambda_ok_rows = [row for row in lambda_rows if row.get("status") == "ok"]
    best_lambda = min(lambda_ok_rows, key=lambda row: float(row.get("H32", float("inf"))), default=None)
    graph_01_h32 = next(
        (float(row.get("H32")) for row in lambda_ok_rows if abs(float(row.get("prior_weight", float("nan"))) - 0.1) < 1e-12),
        float(graph_row.get("H32", float("nan"))) if graph_row else float("nan"),
    )
    none_h32 = float(none_row.get("H32", float("nan"))) if none_row else float("nan")
    perm_h32 = float(permuted_row.get("H32", float("nan"))) if permuted_row else float("nan")
    best_h32 = float(best_lambda.get("H32", float("nan"))) if best_lambda else float("nan")
    high_lambda_hurts = finite(best_h32) and finite(graph_01_h32) and best_h32 < graph_01_h32
    high_lambda_explains = high_lambda_hurts and finite(none_h32) and finite(perm_h32) and best_h32 < none_h32 and best_h32 < perm_h32
    oracle_one = oracle_row.get("H1_mse")
    noise = oracle_row.get("noise")
    oracle_accurate = finite(oracle_one) and finite(noise) and float(oracle_one) <= 5.0 * (float(noise) ** 2)

    final_call = (
        "Treat as a lambda-sensitive candidate under this latent model."
        if high_lambda_explains
        else "Treat as a no-graph-gain case under this latent model, despite the oracle generator being available."
    )

    lines = [
        "# Graph Heat Failure Audit",
        "",
        f"Created: `{datetime.now(timezone.utc).isoformat()}`",
        "Scope: graph_heat_lattice only; no ISO17, rMD17 top-up, or full sweeps were run.",
        "",
        "## Stage 0 Raw Diagnostics",
        "",
        *report_table(
            ["graph type", "D_dX_norm", "R_low K=2", "R_low K=4", "R_low K=8"],
            [
                [
                    graph_type,
                    fmt(row.get("D_dX_norm"), 6),
                    fmt(row.get("R_low_dX_2"), 4),
                    fmt(row.get("R_low_dX_4"), 4),
                    fmt(row.get("R_low_dX_8"), 4),
                ]
                for graph_type, row in raw_by_type.items()
            ],
        ),
        "",
        f"Raw dynamics smoother on true graph than both controls: `{'YES' if raw_smoother else 'NO'}`.",
        "",
        "## Oracle Heat Baseline",
        "",
        *report_table(
            ["metric", "MSE"],
            [
                ["one-step", fmt(oracle_row.get("H1_mse"), 8)],
                ["H=16 rollout", fmt(oracle_row.get("H16_mse"), 8)],
                ["H=32 rollout", fmt(oracle_row.get("H32_mse"), 8)],
            ],
        ),
        "",
        f"Oracle one-step MSE is within `5 * noise^2`: `{'YES' if oracle_accurate else 'NO'}`.",
        "",
        "## Existing Preflight Comparison",
        "",
        *report_table(
            ["prior", "lambda", "H=16", "H=32"],
            [
                ["none", "0", fmt(none_row.get("H16") if none_row else float("nan"), 4), fmt(none_h32, 4)],
                ["graph", "0.1", fmt(graph_row.get("H16") if graph_row else float("nan"), 4), fmt(graph_row.get("H32") if graph_row else float("nan"), 4)],
                ["permuted_graph", "0.1", fmt(permuted_row.get("H16") if permuted_row else float("nan"), 4), fmt(perm_h32, 4)],
            ],
        ),
        "",
        "## Lambda Sanity Check",
        "",
        *report_table(
            ["prior", "lambda", "H=16", "H=32", "prior loss mean"],
            [
                [
                    row.get("prior"),
                    fmt(row.get("prior_weight"), 3),
                    fmt(row.get("H16"), 4),
                    fmt(row.get("H32"), 4),
                    fmt(row.get("prior_loss_mean"), 6),
                ]
                for row in lambda_rows
            ],
        ),
        "",
        f"Best graph lambda by H=32: `{fmt(best_lambda.get('prior_weight') if best_lambda else float('nan'), 3)}` with H=32 `{fmt(best_h32, 4)}`.",
        f"Lower lambda improves over graph lambda=0.1: `{'YES' if high_lambda_hurts else 'NO'}`.",
        f"High lambda / over-smoothing fully explains the original miss: `{'YES' if high_lambda_explains else 'NO'}`.",
        "",
        "## Answers",
        "",
        f"- Is raw dynamics smoother on true graph than controls? `{'YES' if raw_smoother else 'NO'}`.",
        f"- Is the oracle heat model accurate? `{'YES' if oracle_accurate else 'NO'}` for one-step prediction; see rollout MSE above.",
        f"- Is true graph prior hurt due to high lambda / over-smoothing? `{'YES' if high_lambda_explains else 'PARTLY' if high_lambda_hurts else 'NO'}`.",
        f"- Should graph_heat be treated as positive-control failure or no-graph-gain under this latent model? {final_call}",
        "",
        "## Files",
        "",
        f"- `{rel_to_root(args.summary)}`",
        f"- `{rel_to_root(path)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight graph-prior preflight check.")
    parser.add_argument("--dataset", default=None, help="Dataset adapter name, e.g. ho_lattice.")
    parser.add_argument(
        "--topology",
        default=None,
        choices=["chain", "ring", "lattice", "random", "scalefree"],
        help="Topology alias for HO or graph_heat datasets.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--laplacian-normalization", default="lambda_max", choices=["lambda_max", "trace", "none"])
    parser.add_argument("--n-permuted", type=int, default=8)
    parser.add_argument("--n-random", type=int, default=8)
    parser.add_argument("--skip-random-graph", action="store_true")
    parser.add_argument("--raw-transitions", type=int, default=64)
    parser.add_argument("--raw-stride", type=int, default=10)
    parser.add_argument("--raw-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seeds", type=parse_seeds, default=[0])
    parser.add_argument("--train-transitions", type=int, default=48)
    parser.add_argument("--eval-transitions", type=int, default=12)
    parser.add_argument("--horizons", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--eval-stride", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-latent-audit", action="store_true")
    parser.add_argument("--graph-heat-failure-audit-only", action="store_true")
    parser.add_argument("--graph-heat-n", type=int, default=36)
    parser.add_argument("--graph-heat-d", type=int, default=4)
    parser.add_argument("--graph-heat-t", type=int, default=512)
    parser.add_argument("--graph-heat-tau", type=float, default=0.05)
    parser.add_argument("--graph-heat-noise", type=float, default=0.01)
    parser.add_argument("--graph-heat-seed", type=int, default=0)
    args = parser.parse_args()
    if args.dataset is None:
        args.dataset = f"ho_{args.topology}" if args.topology else "ho_lattice"
    if str(args.dataset).startswith("ho_"):
        dataset_topology = str(args.dataset).removeprefix("ho_")
        if args.topology is not None and args.topology != dataset_topology:
            raise ValueError(f"--dataset {args.dataset!r} conflicts with --topology {args.topology!r}")
        args.topology = dataset_topology
    if str(args.dataset).startswith("graph_heat_"):
        dataset_topology = str(args.dataset).removeprefix("graph_heat_")
        if args.topology is not None and args.topology != dataset_topology:
            raise ValueError(f"--dataset {args.dataset!r} conflicts with --topology {args.topology!r}")
        args.topology = dataset_topology
    if str(args.dataset) == "graph_heat" and args.topology is None:
        args.topology = "lattice"
    if str(args.dataset).startswith("graph_heat"):
        if args.raw_stride == 10:
            args.raw_stride = 5
        if args.train_stride == 10:
            args.train_stride = 5
        if args.eval_stride == 20:
            args.eval_stride = 10
    if 32 not in [int(value) for value in args.horizons]:
        raise ValueError("--horizons must include 32 because classification is defined at H=32")
    if args.out_dir is not None:
        args.report = args.report or (args.out_dir / "preflight_report.md")
        args.summary = args.summary or (args.out_dir / "summary.csv")
        args.artifact_dir = args.artifact_dir or (args.out_dir / "artifacts")
    else:
        args.report = args.report or DEFAULT_REPORT
        args.summary = args.summary or DEFAULT_SUMMARY
        args.artifact_dir = args.artifact_dir or DEFAULT_ARTIFACT_DIR
    return args


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    if args.out_dir is not None:
        args.out_dir = args.out_dir.resolve()
    args.report = args.report.resolve()
    args.summary = args.summary.resolve()
    args.artifact_dir = args.artifact_dir.resolve()

    adapter = build_adapter(args)
    if args.graph_heat_failure_audit_only:
        raw_rows = stage0_raw_diagnostics(args, adapter)
        oracle_row = graph_heat_oracle_metrics(args, adapter)
        existing_rows = read_summary_csv(args.summary)
        lambda_rows = graph_heat_lambda_sanity(args, adapter)
        audit_path = (args.out_dir or args.summary.parent) / "FAILURE_AUDIT.md"
        write_graph_heat_failure_audit(
            audit_path.resolve(),
            args=args,
            raw_rows=raw_rows,
            oracle_row=oracle_row,
            existing_rows=existing_rows,
            lambda_rows=lambda_rows,
        )
        print(f"Wrote {rel_to_root(audit_path.resolve())}")
        return

    raw_rows = stage0_raw_diagnostics(args, adapter)
    stage1_rows, models, devices = stage1_mini_training(args, adapter)
    audit = bool(args.force_latent_audit or should_run_latent_audit(stage1_rows))
    stage2_rows = stage2_latent_audit(args, adapter, models, devices) if audit else []
    classification, class_metrics = classify(stage1_rows, stage2_rows)
    oracle_row = graph_heat_oracle_metrics(args, adapter)

    all_rows: list[dict[str, Any]] = []
    all_rows.extend(raw_rows)
    all_rows.extend(stage1_rows)
    all_rows.extend(stage2_rows)
    if oracle_row:
        all_rows.append(oracle_row)
    all_rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "classification",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": adapter.name,
            "topology": args.topology,
            "classification": classification,
            **class_metrics,
            **{f"metadata_{key}": value for key, value in adapter.metadata().items()},
        }
    )
    write_summary_csv(args.summary, all_rows)
    write_report(args.report, args, raw_rows, stage1_rows, stage2_rows, classification, class_metrics, oracle_row)
    print(f"Wrote {rel_to_root(args.report)}")
    print(f"Wrote {rel_to_root(args.summary)} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
