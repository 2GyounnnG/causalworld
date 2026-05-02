from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pickle
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
    graph_prior_loss,
    latent_diagnostics,
    move_obs,
    node_feature_tensor,
    select_device,
    set_seed,
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
DEFAULT_METR_LA_ROOT = ROOT / "data" / "metr_la"
SCHEMA_VERSION = "graph_prior_preflight_v2"
TRACE_SEED = 4242
CALIBRATION_EPS = 1e-12


def log_progress(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


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
        raise ValueError("lattice topology requires N to be a perfect square")
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
    raise ValueError("topology must be one of chain, ring, lattice, random, scalefree")


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


class GraphLowFreqAdapter:
    """Synthetic dynamics constrained to low-frequency eigenvectors of the true graph."""

    def __init__(
        self,
        *,
        topology: str,
        n_nodes: int = 36,
        n_dim: int = 4,
        n_frames: int = 512,
        low_k: int = 4,
        noise: float = 0.01,
        coefficient_decay: float = 0.95,
        seed: int = 0,
    ):
        if n_dim != 4:
            raise ValueError("graph_lowfreq currently requires d=4 to match the existing GNN input width")
        self.name = f"graph_lowfreq_{topology}"
        self.topology = topology
        self.n_nodes = int(n_nodes)
        self.n_dim = int(n_dim)
        self.n_frames = int(n_frames)
        self.low_k = int(low_k)
        self.noise = float(noise)
        self.coefficient_decay = float(coefficient_decay)
        self.seed = int(seed)
        self.edges = graph_heat_edges(topology, self.n_nodes, self.seed)
        self.laplacian = laplacian_from_edges(self.n_nodes, self.edges)
        self.coords = self._simulate()
        self.energies = np.sum(self.coords * self.coords, axis=(1, 2)).astype(np.float32)

    def _low_basis(self) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(self.laplacian)
        order = np.argsort(eigvals)
        eigvecs = eigvecs[:, order]
        k = min(self.low_k, max(1, self.n_nodes - 1))
        return eigvecs[:, 1 : k + 1]

    def _simulate(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        basis = self._low_basis()
        coeffs = np.zeros((self.n_frames, basis.shape[1], self.n_dim), dtype=np.float64)
        coeffs[0] = rng.normal(size=(basis.shape[1], self.n_dim))
        for frame_idx in range(self.n_frames - 1):
            coeff_noise = rng.normal(size=(basis.shape[1], self.n_dim))
            coeffs[frame_idx + 1] = self.coefficient_decay * coeffs[frame_idx] + self.noise * coeff_noise
        coords = np.einsum("nk,tkd->tnd", basis, coeffs)
        return coords.astype(np.float32)

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
            "domain": "graph_lowfreq",
            "topology": self.topology,
            "n_nodes": self.n_nodes,
            "n_edges": int(self.edges.shape[0]),
            "n_frames": self.n_frames,
            "n_dim": self.n_dim,
            "low_k": self.low_k,
            "noise": self.noise,
            "coefficient_decay": self.coefficient_decay,
            "seed": self.seed,
        }


class SpringMassAdapter:
    """Second-order spring-mass dynamics on a known interaction graph."""

    def __init__(
        self,
        *,
        topology: str,
        n_nodes: int = 36,
        position_dim: int = 1,
        n_frames: int = 512,
        spring_k: float = 1.0,
        damping: float = 0.05,
        dt: float = 0.05,
        noise: float = 0.001,
        seed: int = 0,
    ):
        if position_dim < 1:
            raise ValueError("--spring-position-dim must be at least 1")
        if 2 * int(position_dim) > 4:
            raise ValueError(
                "spring_mass supports --spring-position-dim up to 2 with the current 4-channel GNN input"
            )
        self.name = f"spring_mass_{topology}"
        self.topology = topology
        self.n_nodes = int(n_nodes)
        self.position_dim = int(position_dim)
        self.n_dim = 2 * self.position_dim
        self.n_frames = int(n_frames)
        self.spring_k = float(spring_k)
        self.damping = float(damping)
        self.dt = float(dt)
        self.noise = float(noise)
        self.seed = int(seed)
        self.edges = graph_heat_edges(topology, self.n_nodes, self.seed)
        self.laplacian = laplacian_from_edges(self.n_nodes, self.edges)
        self.coords = self._simulate()
        self.energies = self._energy_series().astype(np.float32)

    def _simulate(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        positions = np.zeros((self.n_frames, self.n_nodes, self.position_dim), dtype=np.float64)
        velocities = np.zeros_like(positions)
        positions[0] = rng.normal(scale=0.5, size=(self.n_nodes, self.position_dim))
        velocities[0] = rng.normal(scale=0.1, size=(self.n_nodes, self.position_dim))
        for frame_idx in range(self.n_frames - 1):
            spring_acc = -self.spring_k * (self.laplacian @ positions[frame_idx])
            damping_acc = -self.damping * velocities[frame_idx]
            eps = self.noise * rng.normal(size=(self.n_nodes, self.position_dim))
            acceleration = spring_acc + damping_acc + eps
            velocities[frame_idx + 1] = velocities[frame_idx] + self.dt * acceleration
            positions[frame_idx + 1] = positions[frame_idx] + self.dt * velocities[frame_idx + 1]
        coords = np.concatenate([positions, velocities], axis=-1)
        return coords.astype(np.float32)

    def _energy_series(self) -> np.ndarray:
        positions = self.coords[:, :, : self.position_dim].astype(np.float64)
        velocities = self.coords[:, :, self.position_dim :].astype(np.float64)
        kinetic = 0.5 * np.sum(velocities * velocities, axis=(1, 2))
        potential = np.zeros(self.n_frames, dtype=np.float64)
        for src, dst in self.edges:
            displacement = positions[:, int(src), :] - positions[:, int(dst), :]
            potential += 0.5 * self.spring_k * np.sum(displacement * displacement, axis=1)
        return kinetic + potential

    def _model_features(self, idx: int) -> torch.Tensor:
        state = torch.from_numpy(self.coords[idx]).to(dtype=torch.float32)
        if state.shape[1] < 4:
            padding = torch.zeros((self.n_nodes, 4 - state.shape[1]), dtype=torch.float32)
            state = torch.cat([state, padding], dim=-1)
        return state[:, :4]

    def __getitem__(self, idx: int) -> HeteroData:
        features = self._model_features(idx)
        atomic_numbers = 10.0 * features[:, 0]
        pos = features[:, 1:4]
        directed = np.concatenate([self.edges, self.edges[:, ::-1]], axis=0)
        edge_index = torch.from_numpy(directed.T.copy()).to(dtype=torch.long)
        edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float32)

        data = HeteroData()
        data["atom"].pos = pos
        data["atom"].atomic_number = atomic_numbers
        data["atom"].x = features
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
            "domain": "spring_mass",
            "topology": self.topology,
            "n_nodes": self.n_nodes,
            "n_edges": int(self.edges.shape[0]),
            "n_frames": self.n_frames,
            "n_dim": self.n_dim,
            "position_dim": self.position_dim,
            "trajectory_shape": list(self.coords.shape),
            "k": self.spring_k,
            "damping": self.damping,
            "dt": self.dt,
            "noise": self.noise,
            "seed": self.seed,
        }


class GraphWaveAdapter:
    """Graph-discretized damped wave equation on a known interaction graph."""

    def __init__(
        self,
        *,
        topology: str,
        n_nodes: int = 36,
        n_frames: int = 512,
        wave_c: float = 1.0,
        damping: float = 0.02,
        dt: float = 0.05,
        noise: float = 0.001,
        seed: int = 0,
    ):
        self.name = f"graph_wave_{topology}"
        self.topology = topology
        self.n_nodes = int(n_nodes)
        self.n_dim = 2
        self.n_frames = int(n_frames)
        self.wave_c = float(wave_c)
        self.damping = float(damping)
        self.dt = float(dt)
        self.noise = float(noise)
        self.seed = int(seed)
        self.edges = graph_heat_edges(topology, self.n_nodes, self.seed)
        self.laplacian = laplacian_from_edges(self.n_nodes, self.edges)
        self.coords = self._simulate()
        self.energies = self._energy_series().astype(np.float32)

    def _simulate(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        displacement = np.zeros((self.n_frames, self.n_nodes, 1), dtype=np.float64)
        velocity = np.zeros_like(displacement)
        displacement[0] = rng.normal(scale=0.5, size=(self.n_nodes, 1))
        velocity[0] = rng.normal(scale=0.1, size=(self.n_nodes, 1))
        for frame_idx in range(self.n_frames - 1):
            acceleration = -(self.wave_c**2) * (self.laplacian @ displacement[frame_idx])
            acceleration = acceleration - self.damping * velocity[frame_idx]
            eps = self.noise * rng.normal(size=(self.n_nodes, 1))
            velocity[frame_idx + 1] = velocity[frame_idx] + self.dt * acceleration + eps
            displacement[frame_idx + 1] = displacement[frame_idx] + self.dt * velocity[frame_idx + 1]
        coords = np.concatenate([displacement, velocity], axis=-1)
        return coords.astype(np.float32)

    def _energy_series(self) -> np.ndarray:
        displacement = self.coords[:, :, :1].astype(np.float64)
        velocity = self.coords[:, :, 1:].astype(np.float64)
        kinetic = 0.5 * np.sum(velocity * velocity, axis=(1, 2))
        potential = np.zeros(self.n_frames, dtype=np.float64)
        for src, dst in self.edges:
            diff = displacement[:, int(src), :] - displacement[:, int(dst), :]
            potential += 0.5 * (self.wave_c**2) * np.sum(diff * diff, axis=1)
        return kinetic + potential

    def _model_features(self, idx: int) -> torch.Tensor:
        state = torch.from_numpy(self.coords[idx]).to(dtype=torch.float32)
        padding = torch.zeros((self.n_nodes, 2), dtype=torch.float32)
        return torch.cat([state, padding], dim=-1)

    def __getitem__(self, idx: int) -> HeteroData:
        features = self._model_features(idx)
        atomic_numbers = 10.0 * features[:, 0]
        pos = features[:, 1:4]
        directed = np.concatenate([self.edges, self.edges[:, ::-1]], axis=0)
        edge_index = torch.from_numpy(directed.T.copy()).to(dtype=torch.long)
        edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float32)

        data = HeteroData()
        data["atom"].pos = pos
        data["atom"].atomic_number = atomic_numbers
        data["atom"].x = features
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
            "domain": "graph_wave",
            "topology": self.topology,
            "n_nodes": self.n_nodes,
            "n_edges": int(self.edges.shape[0]),
            "n_frames": self.n_frames,
            "n_dim": self.n_dim,
            "trajectory_shape": list(self.coords.shape),
            "c": self.wave_c,
            "damping": self.damping,
            "dt": self.dt,
            "noise": self.noise,
            "seed": self.seed,
        }


class NBodyDistanceAdapter:
    """Long-range n-body dynamics with a fixed distance-kNN candidate graph."""

    def __init__(
        self,
        *,
        n_nodes: int = 36,
        position_dim: int = 2,
        n_frames: int = 512,
        gravitational_constant: float = 0.1,
        dt: float = 0.02,
        softening: float = 0.1,
        damping: float = 0.0,
        noise: float = 0.0005,
        graph_k: int = 8,
        seed: int = 0,
    ):
        if position_dim < 1:
            raise ValueError("--nbody-position-dim must be at least 1")
        if n_nodes < 2:
            raise ValueError("--nbody-n must be at least 2")
        self.name = "nbody_distance"
        self.topology = "distance_knn"
        self.graph_source = "distance_knn"
        self.n_nodes = int(n_nodes)
        self.position_dim = int(position_dim)
        self.n_dim = 2 * self.position_dim + 1
        self.n_frames = int(n_frames)
        self.gravitational_constant = float(gravitational_constant)
        self.dt = float(dt)
        self.softening = float(softening)
        self.damping = float(damping)
        self.noise = float(noise)
        self.graph_k = min(max(1, int(graph_k)), self.n_nodes - 1)
        self.seed = int(seed)
        self.coords = self._simulate()
        self.adjacency, self.sigma = self._distance_knn_adjacency(self.coords[0, :, : self.position_dim])
        self.edges, self.edge_weights = self._edge_pairs_and_weights()
        self.laplacian = self._laplacian_from_adjacency(self.adjacency)
        self.energies = self._energy_series().astype(np.float32)

    def _initial_state(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = rng.normal(scale=1.0, size=(self.n_nodes, self.position_dim))
        positions = positions - positions.mean(axis=0, keepdims=True)
        velocities = rng.normal(scale=0.05, size=(self.n_nodes, self.position_dim))
        velocities = velocities - velocities.mean(axis=0, keepdims=True)
        masses = rng.uniform(0.5, 1.5, size=(self.n_nodes, 1))
        return positions, velocities, masses

    def _accelerations(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        delta = positions[None, :, :] - positions[:, None, :]
        dist_sq = np.sum(delta * delta, axis=-1) + self.softening**2
        inv_dist3 = np.power(dist_sq, -1.5)
        np.fill_diagonal(inv_dist3, 0.0)
        pair_scale = self.gravitational_constant * masses.reshape(1, self.n_nodes) * inv_dist3
        acceleration = np.einsum("ij,ijd->id", pair_scale, delta)
        acceleration = acceleration - self.damping * velocities
        if self.noise > 0.0:
            acceleration = acceleration + self.noise * rng.normal(size=velocities.shape)
        return acceleration

    def _simulate(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        positions = np.zeros((self.n_frames, self.n_nodes, self.position_dim), dtype=np.float64)
        velocities = np.zeros_like(positions)
        masses = np.zeros((self.n_frames, self.n_nodes, 1), dtype=np.float64)
        positions[0], velocities[0], masses[0] = self._initial_state(rng)
        for frame_idx in range(self.n_frames - 1):
            acceleration = self._accelerations(positions[frame_idx], velocities[frame_idx], masses[frame_idx], rng)
            velocities[frame_idx + 1] = velocities[frame_idx] + self.dt * acceleration
            positions[frame_idx + 1] = positions[frame_idx] + self.dt * velocities[frame_idx + 1]
            masses[frame_idx + 1] = masses[frame_idx]
        coords = np.concatenate([positions, velocities, masses], axis=-1)
        return coords.astype(np.float32)

    def _distance_knn_adjacency(self, initial_positions: np.ndarray) -> tuple[np.ndarray, float]:
        diff = initial_positions[:, None, :] - initial_positions[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(distances, np.inf)
        k = min(self.graph_k, self.n_nodes - 1)
        neighbor_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        neighbor_distances = distances[np.arange(self.n_nodes)[:, None], neighbor_idx]
        finite_distances = neighbor_distances[np.isfinite(neighbor_distances)]
        sigma = float(np.median(finite_distances)) if finite_distances.size else 1.0
        sigma = max(sigma, 1e-6)
        adjacency = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        weights = np.exp(-(neighbor_distances**2) / (sigma**2))
        row_idx = np.broadcast_to(np.arange(self.n_nodes)[:, None], neighbor_idx.shape)
        adjacency[row_idx, neighbor_idx] = weights
        adjacency = np.maximum(adjacency, adjacency.T)
        np.fill_diagonal(adjacency, 0.0)
        return adjacency, sigma

    def _edge_pairs_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        src, dst = np.where(np.triu(self.adjacency, k=1) > 0)
        edges = np.stack([src, dst], axis=1).astype(np.int64)
        weights = self.adjacency[src, dst].astype(np.float32)
        return edges, weights

    @staticmethod
    def _laplacian_from_adjacency(adjacency: np.ndarray) -> np.ndarray:
        weights = np.maximum(np.asarray(adjacency, dtype=np.float64), 0.0)
        return np.diag(weights.sum(axis=1)) - weights

    def _energy_series(self) -> np.ndarray:
        positions = self.coords[:, :, : self.position_dim].astype(np.float64)
        velocities = self.coords[:, :, self.position_dim : 2 * self.position_dim].astype(np.float64)
        masses = self.coords[:, :, -1].astype(np.float64)
        kinetic = 0.5 * np.sum(masses[:, :, None] * velocities * velocities, axis=(1, 2))
        potential = np.zeros(self.n_frames, dtype=np.float64)
        for src in range(self.n_nodes):
            for dst in range(src + 1, self.n_nodes):
                diff = positions[:, dst, :] - positions[:, src, :]
                distance = np.sqrt(np.sum(diff * diff, axis=1) + self.softening**2)
                potential -= self.gravitational_constant * masses[:, src] * masses[:, dst] / distance
        return kinetic + potential

    def _model_features(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        state = torch.from_numpy(self.coords[idx]).to(dtype=torch.float32)
        mass = state[:, -1]
        dynamic = state[:, :-1]
        if dynamic.shape[1] < 3:
            padding = torch.zeros((self.n_nodes, 3 - dynamic.shape[1]), dtype=torch.float32)
            pos = torch.cat([dynamic, padding], dim=-1)
        else:
            pos = dynamic[:, :3]
        features = torch.cat([mass.view(-1, 1), pos], dim=-1)
        return features, mass

    def __getitem__(self, idx: int) -> HeteroData:
        features, mass = self._model_features(idx)
        directed = np.concatenate([self.edges, self.edges[:, ::-1]], axis=0)
        directed_weights = np.concatenate([self.edge_weights, self.edge_weights], axis=0)
        edge_index = torch.from_numpy(directed.T.copy()).to(dtype=torch.long)
        edge_attr = torch.from_numpy((1.0 / np.maximum(directed_weights, 1e-6)).reshape(-1, 1)).to(dtype=torch.float32)

        data = HeteroData()
        data["atom"].pos = features[:, 1:4]
        data["atom"].atomic_number = 10.0 * mass
        data["atom"].x = features
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
            "domain": "nbody_distance",
            "topology": self.topology,
            "graph_source": self.graph_source,
            "n_nodes": self.n_nodes,
            "n_edges": int(self.edges.shape[0]),
            "n_frames": self.n_frames,
            "n_dim": self.n_dim,
            "position_dim": self.position_dim,
            "trajectory_shape": list(self.coords.shape),
            "G": self.gravitational_constant,
            "dt": self.dt,
            "softening": self.softening,
            "damping": self.damping,
            "noise": self.noise,
            "graph_k": self.graph_k,
            "sigma": self.sigma,
            "seed": self.seed,
        }


def read_metr_la_csv(path: Path, max_timesteps: int | None = None) -> tuple[np.ndarray, list[str]]:
    if max_timesteps is not None and max_timesteps < 2:
        raise ValueError("--metr-la-max-timesteps must be at least 2 when set")
    log_progress(f"METR-LA: loading CSV {rel_to_root(path)}")
    try:
        import pandas as pd
    except Exception:
        pd = None
    if pd is not None:
        frame = pd.read_csv(path, nrows=max_timesteps)
        if frame.shape[1] >= 2 and not np.issubdtype(frame.dtypes.iloc[0], np.number):
            frame = frame.iloc[:, 1:]
        values = frame.apply(lambda col: pd.to_numeric(col, errors="coerce")).to_numpy(dtype=np.float32)
        sensor_ids = [str(col) for col in frame.columns]
    else:
        with path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader)
            rows = []
            for row_idx, row in enumerate(reader):
                if max_timesteps is not None and row_idx >= max_timesteps:
                    break
                rows.append(row)
        start_col = 1
        try:
            float(rows[0][0])
            start_col = 0
        except (ValueError, IndexError):
            start_col = 1
        sensor_ids = [str(value) for value in header[start_col:]]
        values = np.asarray([[float(value) if value else np.nan for value in row[start_col:]] for row in rows], dtype=np.float32)
    n_rows = values.shape[0] if values.ndim >= 1 else 0
    n_sensors = values.shape[1] if values.ndim >= 2 else 0
    log_progress(f"METR-LA: CSV shape after loading rows={n_rows} sensors={n_sensors}")
    if max_timesteps is not None:
        log_progress(f"METR-LA: applying --metr-la-max-timesteps {max_timesteps}")
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError(f"METR-LA CSV must contain a [time, sensor] table, got shape {values.shape}")
    col_means = np.nanmean(values, axis=0)
    row_idx, col_idx = np.where(~np.isfinite(values))
    if row_idx.size:
        values[row_idx, col_idx] = col_means[col_idx]
    return values[:, :, None].astype(np.float32), sensor_ids


def read_metr_la_adj_pkl(path: Path, sensor_ids: list[str] | None = None) -> np.ndarray:
    with path.open("rb") as file:
        try:
            payload = pickle.load(file)
        except UnicodeDecodeError:
            file.seek(0)
            payload = pickle.load(file, encoding="latin1")
    if isinstance(payload, tuple) and len(payload) >= 3:
        pkl_sensor_ids, sensor_id_to_ind, adj = payload[:3]
        adj = np.asarray(adj, dtype=np.float64)
        if sensor_ids:
            indices = [int(sensor_id_to_ind[str(sensor_id)]) for sensor_id in sensor_ids if str(sensor_id) in sensor_id_to_ind]
            if len(indices) == len(sensor_ids):
                adj = adj[np.ix_(indices, indices)]
        return adj
    if isinstance(payload, dict):
        for key in ("adj_mx", "adjacency", "adj", "W"):
            if key in payload:
                return np.asarray(payload[key], dtype=np.float64)
    return np.asarray(payload, dtype=np.float64)


def infer_metr_la_csv_path(root: Path, csv_path: Path | None) -> Path:
    csv_candidates = [
        csv_path,
        root / "metr-la.csv",
        root / "METR-LA.csv",
        root / "metr_la.csv",
        root / "METR_LA.csv",
    ]
    found_csv = next((path for path in csv_candidates if path is not None and path.exists()), None)
    if found_csv is None:
        raise FileNotFoundError(
            "METR-LA CSV not found. Expected a speed CSV, e.g. "
            f"{root / 'metr-la.csv'}, or pass --metr-la-csv."
        )
    return found_csv


def infer_metr_la_adj_path(root: Path, adj_path: Path | None) -> Path:
    adj_candidates = [
        adj_path,
        root / "adj_mx.pkl",
        root / "adjacency.pkl",
        root / "metr-la-adj.pkl",
        root / "METR-LA-adj.pkl",
    ]
    found_adj = next((path for path in adj_candidates if path is not None and path.exists()), None)
    if found_adj is None:
        raise FileNotFoundError(
            "METR-LA adjacency pkl not found. Expected e.g. "
            f"{root / 'adj_mx.pkl'}, or pass --metr-la-adj-pkl."
        )
    return found_adj


def metr_la_corr_topk_cache_path(root: Path, top_k: int, corr_mode: str, max_timesteps: int | None) -> Path:
    timestep_label = str(int(max_timesteps)) if max_timesteps is not None else "all"
    return root / f"processed_corr_topk_k{int(top_k)}_{corr_mode}_T{timestep_label}.pt"


def load_metr_la_corr_topk_cache(cache_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    if not cache_path.exists():
        log_progress(f"METR-LA: cache not found {rel_to_root(cache_path)}")
        return None
    log_progress(f"METR-LA: loading cache {rel_to_root(cache_path)}")
    payload = torch.load(cache_path, map_location="cpu")
    coords = payload["coords"]
    adjacency = payload["adjacency"]
    sensor_ids = payload.get("sensor_ids", [])
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    if isinstance(adjacency, torch.Tensor):
        adjacency = adjacency.cpu().numpy()
    return np.asarray(coords, dtype=np.float32), np.asarray(adjacency, dtype=np.float64), [str(value) for value in sensor_ids]


def save_metr_la_corr_topk_cache(
    cache_path: Path,
    *,
    coords: np.ndarray,
    adjacency: np.ndarray,
    sensor_ids: list[str],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log_progress(f"METR-LA: saving cache {rel_to_root(cache_path)}")
    torch.save(
        {
            "coords": torch.from_numpy(np.asarray(coords, dtype=np.float32)),
            "adjacency": torch.from_numpy(np.asarray(adjacency, dtype=np.float64)),
            "sensor_ids": sensor_ids,
        },
        cache_path,
    )


def correlation_topk_adjacency(
    coords: np.ndarray,
    *,
    top_k: int,
    mode: str,
    train_fraction: float = 0.7,
) -> np.ndarray:
    if mode not in {"positive", "absolute"}:
        raise ValueError("--metr-la-corr-mode must be positive or absolute")
    n_train = max(2, int(coords.shape[0] * train_fraction))
    series = np.asarray(coords[:n_train, :, 0], dtype=np.float64)
    log_progress(f"METR-LA: computing correlation over {n_train} timesteps x {series.shape[1]} sensors")
    corr = np.corrcoef(series, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    weights = np.maximum(corr, 0.0) if mode == "positive" else np.abs(corr)
    np.fill_diagonal(weights, 0.0)
    k = min(max(0, int(top_k)), max(0, weights.shape[0] - 1))
    log_progress(f"METR-LA: building top-k graph k={k} mode={mode}")
    adjacency = np.zeros_like(weights)
    if k > 0:
        top_idx = np.argpartition(-weights, kth=k - 1, axis=1)[:, :k]
        row_idx = np.broadcast_to(np.arange(weights.shape[0])[:, None], top_idx.shape)
        top_weights = weights[row_idx, top_idx]
        keep = top_weights > 0.0
        adjacency[row_idx[keep], top_idx[keep]] = top_weights[keep]
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def load_metr_la_from_pyg_temporal() -> tuple[np.ndarray, np.ndarray, str]:
    try:
        from torch_geometric_temporal.dataset import METRLADatasetLoader
    except Exception as exc:
        raise ImportError(f"PyTorch Geometric Temporal METRLADatasetLoader unavailable: {exc}") from exc
    loader = METRLADatasetLoader()
    dataset = loader.get_dataset()
    features: list[np.ndarray] = []
    edge_index = None
    edge_weight = None
    for snapshot in dataset:
        x = np.asarray(snapshot.x, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim == 2:
            x = x[:, :1]
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)[:, :1]
        features.append(x)
        if edge_index is None:
            edge_index = np.asarray(snapshot.edge_index, dtype=np.int64)
            edge_weight = np.asarray(getattr(snapshot, "edge_attr", np.ones(edge_index.shape[1])), dtype=np.float64).reshape(-1)
    if not features or edge_index is None:
        raise RuntimeError("PyG Temporal METR-LA loader returned no snapshots")
    coords = np.stack(features, axis=0).astype(np.float32)
    n_nodes = coords.shape[1]
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for idx in range(edge_index.shape[1]):
        src = int(edge_index[0, idx])
        dst = int(edge_index[1, idx])
        if src == dst:
            continue
        weight = float(edge_weight[idx]) if edge_weight is not None and idx < edge_weight.shape[0] else 1.0
        adjacency[src, dst] = max(adjacency[src, dst], weight)
        adjacency[dst, src] = max(adjacency[dst, src], weight)
    return coords, adjacency, "pyg_temporal"


class METRLAAdapter:
    """METR-LA traffic speed forecasting adapter."""

    def __init__(
        self,
        *,
        root: Path,
        csv_path: Path | None = None,
        adj_path: Path | None = None,
        graph_source: str = "adjacency_pkl",
        top_k: int = 8,
        corr_mode: str = "positive",
        max_timesteps: int | None = None,
        prefer_pyg_temporal: bool = True,
    ):
        self.name = "metr_la"
        self.topology = "traffic"
        if graph_source not in {"adjacency_pkl", "correlation_topk"}:
            raise ValueError("--metr-la-graph-source must be adjacency_pkl or correlation_topk")
        self.source = "local"
        self.graph_source = graph_source
        self.top_k = int(top_k)
        self.correlation_mode = corr_mode
        self.max_timesteps = max_timesteps
        self.csv_path: Path | None = None
        self.adj_path: Path | None = None
        errors: list[str] = []
        coords: np.ndarray | None = None
        adjacency: np.ndarray | None = None
        if prefer_pyg_temporal:
            try:
                log_progress("METR-LA: loading PyG Temporal dataset")
                coords, adjacency, self.source = load_metr_la_from_pyg_temporal()
                log_progress(f"METR-LA: PyG dataset shape after loading timesteps={coords.shape[0]} sensors={coords.shape[1]}")
                if max_timesteps is not None:
                    log_progress(f"METR-LA: applying --metr-la-max-timesteps {max_timesteps}")
                    coords = coords[:max_timesteps]
            except Exception as exc:
                errors.append(str(exc))
        if coords is None or adjacency is None:
            try:
                self.csv_path = infer_metr_la_csv_path(root, csv_path)
                if graph_source == "correlation_topk":
                    if max_timesteps is not None:
                        log_progress(f"METR-LA: applying --metr-la-max-timesteps {max_timesteps}")
                    cache_path = metr_la_corr_topk_cache_path(root, self.top_k, self.correlation_mode, max_timesteps)
                    cached = load_metr_la_corr_topk_cache(cache_path)
                    if cached is not None:
                        coords, adjacency, sensor_ids = cached
                    else:
                        coords, sensor_ids = read_metr_la_csv(self.csv_path, max_timesteps=max_timesteps)
                        adjacency = correlation_topk_adjacency(coords, top_k=self.top_k, mode=self.correlation_mode)
                        save_metr_la_corr_topk_cache(
                            cache_path,
                            coords=coords,
                            adjacency=adjacency,
                            sensor_ids=sensor_ids,
                        )
                    self.graph_source = "correlation_topk"
                else:
                    coords, sensor_ids = read_metr_la_csv(self.csv_path, max_timesteps=max_timesteps)
                    self.adj_path = infer_metr_la_adj_path(root, adj_path)
                    adjacency = read_metr_la_adj_pkl(self.adj_path, sensor_ids)
                    self.graph_source = "adjacency_pkl"
                self.source = "local_csv_pkl"
            except Exception as exc:
                errors.append(str(exc))
                raise FileNotFoundError("Could not load METR-LA. " + " | ".join(errors)) from exc
        adjacency = np.asarray(adjacency, dtype=np.float64)
        adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
        adjacency = 0.5 * (adjacency + adjacency.T)
        np.fill_diagonal(adjacency, 0.0)
        if adjacency.shape[0] != coords.shape[1] or adjacency.shape[1] != coords.shape[1]:
            raise ValueError(f"METR-LA adjacency shape {adjacency.shape} does not match coords nodes {coords.shape[1]}")
        self.coords = coords.astype(np.float32)
        self.adjacency = adjacency
        self.laplacian = self._laplacian_from_adjacency(adjacency)
        self.n_frames = int(self.coords.shape[0])
        self.n_nodes = int(self.coords.shape[1])
        self.n_dim = int(self.coords.shape[2])
        self.mean = float(np.mean(self.coords))
        self.std = float(np.std(self.coords) + 1e-6)
        self.energies = np.mean(self.coords * self.coords, axis=(1, 2)).astype(np.float32)

    @staticmethod
    def _laplacian_from_adjacency(adjacency: np.ndarray) -> np.ndarray:
        weights = np.maximum(np.asarray(adjacency, dtype=np.float64), 0.0)
        return np.diag(weights.sum(axis=1)) - weights

    def _edge_pairs_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        src, dst = np.where(np.triu(self.adjacency, k=1) > 0)
        edges = np.stack([src, dst], axis=1).astype(np.int64)
        weights = self.adjacency[src, dst].astype(np.float32)
        return edges, weights

    def __getitem__(self, idx: int) -> HeteroData:
        values = torch.from_numpy(((self.coords[idx, :, 0] - self.mean) / self.std).astype(np.float32))
        pos = torch.zeros((self.n_nodes, 3), dtype=torch.float32)
        atomic_numbers = 10.0 * values
        edges, weights = self._edge_pairs_and_weights()
        directed = np.concatenate([edges, edges[:, ::-1]], axis=0)
        directed_weights = np.concatenate([weights, weights], axis=0)
        edge_index = torch.from_numpy(directed.T.copy()).to(dtype=torch.long)
        edge_attr = torch.from_numpy((1.0 / np.maximum(directed_weights, 1e-6)).reshape(-1, 1)).to(dtype=torch.float32)
        data = HeteroData()
        data["atom"].pos = pos
        data["atom"].atomic_number = atomic_numbers
        data["atom"].x = torch.cat([values.view(-1, 1), pos], dim=-1)
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
            topology="metr_la",
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
            "domain": "traffic",
            "source": self.source,
            "graph_source": self.graph_source,
            "top_k": self.top_k if self.graph_source == "correlation_topk" else "",
            "correlation_mode": self.correlation_mode if self.graph_source == "correlation_topk" else "",
            "max_timesteps": self.max_timesteps or "",
            "topology": self.topology,
            "n_nodes": self.n_nodes,
            "n_edges": int(np.count_nonzero(np.triu(self.adjacency, k=1))),
            "n_frames": self.n_frames,
            "n_dim": self.n_dim,
            "csv_path": str(self.csv_path) if self.csv_path else "",
            "adj_path": str(self.adj_path) if self.adj_path else "",
        }


def build_adapter(args: argparse.Namespace) -> DatasetAdapter:
    dataset = str(args.dataset)
    if dataset.startswith("ho_"):
        return HOAdapter(dataset, args.data_root)
    if dataset == "metr_la":
        return METRLAAdapter(
            root=args.metr_la_root,
            csv_path=args.metr_la_csv,
            adj_path=args.metr_la_adj_pkl,
            graph_source=args.metr_la_graph_source,
            top_k=args.metr_la_top_k,
            corr_mode=args.metr_la_corr_mode,
            max_timesteps=args.metr_la_max_timesteps,
            prefer_pyg_temporal=not args.metr_la_no_pyg_temporal,
        )
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
    if dataset == "graph_lowfreq" or dataset.startswith("graph_lowfreq_"):
        topology = args.topology or (dataset.removeprefix("graph_lowfreq_") if dataset.startswith("graph_lowfreq_") else "lattice")
        return GraphLowFreqAdapter(
            topology=topology,
            n_nodes=args.lowfreq_n,
            n_dim=args.lowfreq_d,
            n_frames=args.lowfreq_t,
            low_k=args.lowfreq_k,
            noise=args.lowfreq_noise,
            coefficient_decay=args.lowfreq_decay,
            seed=args.lowfreq_seed,
        )
    if dataset == "spring_mass" or dataset.startswith("spring_mass_"):
        topology = args.topology or (dataset.removeprefix("spring_mass_") if dataset.startswith("spring_mass_") else "lattice")
        return SpringMassAdapter(
            topology=topology,
            n_nodes=args.spring_n,
            position_dim=args.spring_position_dim,
            n_frames=args.spring_t,
            spring_k=args.spring_k,
            damping=args.spring_damping,
            dt=args.spring_dt,
            noise=args.spring_noise,
            seed=args.spring_seed,
        )
    if dataset == "graph_wave" or dataset.startswith("graph_wave_"):
        topology = args.topology or (dataset.removeprefix("graph_wave_") if dataset.startswith("graph_wave_") else "lattice")
        return GraphWaveAdapter(
            topology=topology,
            n_nodes=args.wave_n,
            n_frames=args.wave_t,
            wave_c=args.wave_c,
            damping=args.wave_damping,
            dt=args.wave_dt,
            noise=args.wave_noise,
            seed=args.wave_seed,
        )
    if dataset == "nbody_distance":
        return NBodyDistanceAdapter(
            n_nodes=args.nbody_n,
            position_dim=args.nbody_position_dim,
            n_frames=args.nbody_t,
            gravitational_constant=args.nbody_g,
            dt=args.nbody_dt,
            softening=args.nbody_softening,
            damping=args.nbody_damping,
            noise=args.nbody_noise,
            graph_k=args.nbody_graph_k,
            seed=args.nbody_seed,
        )
    raise ValueError(
        f"Unknown dataset {dataset!r}. Available: ho_lattice, ho_random, ho_scalefree, "
        "graph_heat, graph_heat_<topology>, graph_lowfreq, graph_lowfreq_<topology>, "
        "spring_mass, spring_mass_<topology>, graph_wave, graph_wave_<topology>, nbody_distance"
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


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    return value


def write_run_config(args: argparse.Namespace) -> Path:
    output_dir = args.out_dir or args.report.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "args": json_ready(vars(args)),
    }
    log_progress(f"Writing run config: {rel_to_root(config_path)}")
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path


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
    log_progress("Stage 0: raw graph-dynamics diagnostics start")
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
    log_progress("Stage 0: raw graph-dynamics diagnostics end")
    return rows


def prior_metadata(prior: str) -> dict[str, Any]:
    if prior == "temporal_smooth":
        return {
            "prior_type": "temporal_smooth",
            "graph_used_for_prior": False,
            "prior_family": "temporal",
        }
    if prior in {"graph", "permuted_graph", "random_graph"}:
        return {
            "prior_type": prior,
            "graph_used_for_prior": True,
            "prior_family": "graph",
        }
    return {
        "prior_type": prior,
        "graph_used_for_prior": False,
        "prior_family": "none" if prior == "none" else "global",
    }


def apply_transition_args(config: Cycle3HOConfig, args: argparse.Namespace | None) -> Cycle3HOConfig:
    if args is None:
        return config
    config.transition_type = str(getattr(args, "transition_type", "mlp_pooled"))
    config.transition_gnn_layers = int(getattr(args, "transition_gnn_layers", 1))
    config.decode_loss_weight = float(getattr(args, "decode_loss_weight", 1.0))
    config.node_feature_dim = int(getattr(args, "node_feature_dim", 4))
    return config


def preflight_train_batch(
    model: Cycle0LatentDynamics,
    transitions: list[dict[str, Any]],
    config: Cycle3HOConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    transition_losses: list[torch.Tensor] = []
    prior_losses: list[torch.Tensor] = []
    z_values: list[torch.Tensor] = []
    for transition in transitions:
        obs = move_obs(transition["obs"], device)
        next_obs = move_obs(transition["next_obs"], device)
        z, h = model.encode(obs)
        if config.prior == "temporal_smooth":
            z_next, h_next = model.encode(next_obs)
        else:
            with torch.no_grad():
                z_next, h_next = model.encode(next_obs)
        if model.transition_type == "graph_node_sage":
            if h is None or h_next is None:
                raise ValueError("graph_node_sage transition requires node-wise encoder states")
            edge_index = obs["atom", "bonded", "atom"].edge_index
            h_pred = model.step_node_states(h, edge_index)
            latent_loss = torch.nn.functional.mse_loss(h_pred, h_next.detach())
            x_pred = model.decode_node_features(h_pred)
            x_next = node_feature_tensor(next_obs).to(device=x_pred.device, dtype=x_pred.dtype)
            if x_pred.shape != x_next.shape:
                raise ValueError(f"Decoded node feature shape {tuple(x_pred.shape)} does not match target {tuple(x_next.shape)}")
            decode_weight = float(getattr(config, "decode_loss_weight", 1.0))
            transition_losses.append(latent_loss + decode_weight * torch.nn.functional.mse_loss(x_pred, x_next))
        else:
            z_pred = model.step_latent(z)
            transition_losses.append(torch.nn.functional.mse_loss(z_pred, z_next.detach()))
        z_values.append(z)
        if config.prior in {"graph", "permuted_graph", "random_graph"}:
            prior_losses.append(
                graph_prior_loss(
                    config.prior,
                    obs,
                    h,
                    seed=config.seed,
                    frame_idx=int(transition["frame_idx"]),
                )
            )
        elif config.prior == "temporal_smooth":
            if h is None or h_next is None:
                raise ValueError("temporal_smooth prior requires an encoder that emits node-wise states")
            prior_losses.append(torch.mean((h_next - h) ** 2))

    transition_loss = torch.stack(transition_losses).mean()
    z_batch = torch.stack(z_values, dim=0)
    if config.prior in {"graph", "permuted_graph", "random_graph", "temporal_smooth"}:
        prior_loss = torch.stack(prior_losses).mean()
    else:
        prior_loss = transition_loss.new_tensor(0.0)
    total = transition_loss + config.prior_weight * prior_loss
    return total, transition_loss, prior_loss, z_batch


def estimate_initial_prior_losses(
    model: Cycle0LatentDynamics,
    transitions: list[dict[str, Any]],
    config: Cycle3HOConfig,
    device: torch.device,
) -> dict[str, float]:
    batch = transitions[: max(1, min(len(transitions), int(config.batch_size), 16))]
    losses: dict[str, float] = {}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for prior in ("graph", "permuted_graph", "temporal_smooth"):
            probe_config = copy.copy(config)
            probe_config.prior = prior
            _total, _transition_loss, prior_loss, _z_batch = preflight_train_batch(model, batch, probe_config, device)
            losses[prior] = float(prior_loss.detach().cpu())
    if was_training:
        model.train()
    return losses


def calibration_metadata(
    *,
    prior: str,
    nominal_prior_weight: float,
    effective_prior_weight: float,
    initial_prior_loss: float,
    target_regularization_strength: float,
    calibration_reference_prior: str,
) -> dict[str, Any]:
    return {
        "nominal_prior_weight": nominal_prior_weight,
        "effective_prior_weight": effective_prior_weight,
        "initial_prior_loss": initial_prior_loss,
        "target_regularization_strength": target_regularization_strength,
        "calibration_reference_prior": calibration_reference_prior,
        "effective_prior_contribution_mean": float("nan"),
    }


def mini_train(
    config: Cycle3HOConfig,
    adapter: DatasetAdapter,
    args: argparse.Namespace | None = None,
) -> tuple[dict[str, Any], Cycle0LatentDynamics | None, torch.device]:
    config = apply_transition_args(config, args)
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
    nominal_prior_weight = float(config.prior_weight)
    effective_prior_weight = nominal_prior_weight
    initial_prior_loss = float("nan")
    target_regularization_strength = float("nan")
    calibration_reference_prior = getattr(args, "calibration_reference_prior", "graph") if args is not None else "graph"
    if args is not None and bool(getattr(args, "calibrate_prior_strength", False)):
        initial_losses = estimate_initial_prior_losses(model, train_transitions, config, device)
        reference_loss = initial_losses.get(str(calibration_reference_prior), float("nan"))
        target_regularization_strength = (
            float(getattr(args, "calibration_target_ratio", 1.0)) * nominal_prior_weight * reference_loss
            if finite(reference_loss)
            else float("nan")
        )
        if config.prior in initial_losses:
            initial_prior_loss = initial_losses[config.prior]
            effective_prior_weight = (
                target_regularization_strength / (initial_prior_loss + CALIBRATION_EPS)
                if finite(target_regularization_strength) and finite(initial_prior_loss)
                else nominal_prior_weight
            )
            config.prior_weight = float(effective_prior_weight)
        else:
            initial_prior_loss = 0.0
            effective_prior_weight = nominal_prior_weight
    else:
        config.prior_weight = nominal_prior_weight
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
            total, transition_loss, prior_loss, _z_batch = preflight_train_batch(model, batch, config, device)
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
            **prior_metadata(config.prior),
            **calibration_metadata(
                prior=config.prior,
                nominal_prior_weight=nominal_prior_weight,
                effective_prior_weight=effective_prior_weight,
                initial_prior_loss=initial_prior_loss,
                target_regularization_strength=target_regularization_strength,
                calibration_reference_prior=str(calibration_reference_prior),
            ),
            "seed": config.seed,
            "config_hash": config_hash(config),
            "git_commit": get_git_commit(),
            "config": asdict(config),
            "diagnostics": diagnostics,
            **{f"H{horizon}": rollout_errors.get(str(horizon)) for horizon in config.eval_horizons},
            **{f"H{horizon}_node_fro": rollout_errors.get(f"{horizon}_node_fro", float("nan")) for horizon in config.eval_horizons},
            **{f"H{horizon}_pooled": rollout_errors.get(f"{horizon}_pooled", float("nan")) for horizon in config.eval_horizons},
            **{f"X{horizon}_rmse": rollout_errors.get(f"{horizon}_x_rmse", float("nan")) for horizon in config.eval_horizons},
            "final_train_loss": diagnostics["final_train_loss"],
            "prior_loss_mean": diagnostics["prior_loss_mean"],
            "effective_prior_contribution_mean": float(effective_prior_weight) * float(diagnostics["prior_loss_mean"]),
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
    priors = ["none", "graph", "permuted_graph"]
    if args.include_temporal_prior:
        priors.append("temporal_smooth")
    for seed in args.seeds:
        for prior in priors:
            config = adapter.training_config(args, prior=prior, seed=int(seed))
            log_progress(f"Mini-training start: {config.run_name}")
            try:
                row, model, device = mini_train(config, adapter, args)
                rows.append(row)
                if prior in {"graph", "permuted_graph"}:
                    assert model is not None
                    models[(prior, int(seed))] = model
                    devices[int(seed)] = device
                log_progress(f"Mini-training end: {config.run_name}")
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
                        **prior_metadata(prior),
                        **calibration_metadata(
                            prior=prior,
                            nominal_prior_weight=float(args.prior_weight),
                            effective_prior_weight=float(config.prior_weight),
                            initial_prior_loss=float("nan"),
                            target_regularization_strength=float("nan"),
                            calibration_reference_prior=str(args.calibration_reference_prior),
                        ),
                        "seed": int(seed),
                        "error": str(exc),
                        "H16": float("nan"),
                        "H32": float("nan"),
                        "H16_node_fro": float("nan"),
                        "H32_node_fro": float("nan"),
                        "H16_pooled": float("nan"),
                        "H32_pooled": float("nan"),
                        "X16_rmse": float("nan"),
                        "X32_rmse": float("nan"),
                    }
                )
                print(f"FAILED {config.run_name}: {exc}", flush=True)
                log_progress(f"Mini-training end: {config.run_name} failed")
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
    temporal_gain = pct_gain(h32.get("none", float("nan")), h32.get("temporal_smooth", float("nan")))
    graph_vs_temporal_gain = pct_gain(h32.get("temporal_smooth", float("nan")), h32.get("graph", float("nan")))
    metrics = {
        "graph_gain_h32_pct": graph_gain,
        "true_vs_permuted_gain_h32_pct": specificity_gain,
        "temporal_gain_h32_pct": temporal_gain,
        "graph_vs_temporal_gain_h32_pct": graph_vs_temporal_gain,
        "none_h32": h32.get("none", float("nan")),
        "graph_h32": h32.get("graph", float("nan")),
        "permuted_h32": h32.get("permuted_graph", float("nan")),
        "temporal_smooth_h32": h32.get("temporal_smooth", float("nan")),
        "none_h16": h16.get("none", float("nan")),
        "graph_h16": h16.get("graph", float("nan")),
        "permuted_h16": h16.get("permuted_graph", float("nan")),
        "temporal_smooth_h16": h16.get("temporal_smooth", float("nan")),
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


def strict_rollout_label(args: argparse.Namespace, metrics: dict[str, float]) -> str:
    none_h32 = metrics.get("none_h32", float("nan"))
    graph_h32 = metrics.get("graph_h32", float("nan"))
    permuted_h32 = metrics.get("permuted_h32", float("nan"))
    temporal_h32 = metrics.get("temporal_smooth_h32", float("nan"))
    mode = str(getattr(args, "mode", "") or getattr(args, "analysis_mode", "")).lower()
    epochs = int(getattr(args, "epochs", 0) or 0)

    if not finite(graph_h32):
        return "inconclusive"
    if finite(none_h32) and float(graph_h32) >= float(none_h32):
        return "no_graph_gain"
    if finite(permuted_h32) and float(graph_h32) >= float(permuted_h32):
        return "generic_smoothing"
    if finite(temporal_h32) and float(graph_h32) >= float(temporal_h32):
        return "temporal_sufficient"
    if mode == "quick" or epochs <= 5:
        return "quick_topology_signal"
    if mode == "standard" or epochs >= 20:
        return "candidate_topology_specific_candidate"
    return "inconclusive"


def dataset_family(args: argparse.Namespace, metadata: dict[str, Any]) -> str:
    dataset = str(metadata.get("dataset") or getattr(args, "dataset", "")).lower()
    domain = str(metadata.get("domain", "")).lower()
    topology = str(metadata.get("topology") or getattr(args, "topology", "")).lower()
    if dataset == "metr_la" or domain == "traffic":
        return "metr_la"
    if dataset.startswith("graph_heat") or domain == "graph_heat":
        return "graph_heat"
    if dataset == "nbody_distance" or domain == "nbody_distance":
        return "nbody_distance"
    if dataset == "ho_lattice" and topology == "lattice":
        return "ho_lattice"
    return dataset


def diagnostic_interpretation(
    args: argparse.Namespace,
    classification: str,
    metrics: dict[str, float],
    metadata: dict[str, Any],
) -> dict[str, str]:
    strict_label = strict_rollout_label(args, metrics)
    family = dataset_family(args, metadata)
    graph_h32 = metrics.get("graph_h32", float("nan"))
    none_h32 = metrics.get("none_h32", float("nan"))

    failure_by_label = {
        "no_graph_gain": "candidate_prior_not_useful_under_tested_condition",
        "generic_smoothing": "topology_not_isolated_from_spectral_smoothing",
        "temporal_sufficient": "graph_free_temporal_smoothing_explains_gain",
        "quick_topology_signal": "quick_budget_candidate_signal_requires_persistence_check",
        "candidate_topology_specific_candidate": "standard_budget_candidate_graph_favorable_requires_robustness_and_audit",
        "inconclusive": "missing_controls_or_insufficient_evidence",
    }
    next_by_label = {
        "no_graph_gain": "skip_prior_or_test_alternative_graph_construction",
        "generic_smoothing": "test_simpler_smoothing_controls_or_alternative_graph_construction",
        "temporal_sufficient": "prefer_calibrated_temporal_smoothing_unless_topology_attribution_is_goal",
        "quick_topology_signal": "run_standard_budget_persistence_check_with_same_controls",
        "candidate_topology_specific_candidate": "run_multi_seed_validation_and_optional_latent_audit",
        "inconclusive": "run_missing_controls_or_collect_required_artifacts",
    }
    boundary_by_label = {
        "no_graph_gain": "No evidence that the tested prior is useful under this model condition; does not rule out other priors or budgets.",
        "generic_smoothing": "Graph-style regularization helps, but topology-specific attribution is not isolated.",
        "temporal_sufficient": "Graph-free temporal smoothing explains the gain; graph structure is not necessary under this condition.",
        "quick_topology_signal": "Screening outcome only; not final topology-specific attribution.",
        "candidate_topology_specific_candidate": "Candidate graph is favorable under this construction and budget; does not prove true physical graph and requires robustness/audit.",
        "inconclusive": "Missing controls or artifacts prevent a stronger claim.",
    }

    diagnostic_failure_mode = failure_by_label.get(strict_label, "missing_controls_or_insufficient_evidence")
    recommended_next_experiment = next_by_label.get(strict_label, "run_missing_controls_or_collect_required_artifacts")
    claim_boundary = boundary_by_label.get(strict_label, "Missing controls or artifacts prevent a stronger claim.")

    if family == "metr_la":
        if strict_label in {"no_graph_gain", "generic_smoothing"}:
            diagnostic_failure_mode = "candidate_graph_construction_risk"
        recommended_next_experiment = "test_official_road_adjacency_road_distance_or_learned_traffic_graph"
        claim_boundary = "Applies only to the tested correlation-derived graph, not all traffic graphs."
    elif family == "graph_heat":
        if strict_label == "no_graph_gain" or classification == "overconstrained" or (
            finite(graph_h32) and finite(none_h32) and float(graph_h32) > float(none_h32)
        ):
            diagnostic_failure_mode = "prior_form_operator_mismatch_possible"
        recommended_next_experiment = "test_operator_aligned_heat_prior_or_first_order_heat_residual"
        claim_boundary = "True graph provenance alone is insufficient; the tested prior form may be mismatched to the physical operator."
    elif family == "nbody_distance":
        nbody_failure = {
            "no_graph_gain": "distance_knn_graph_not_useful_for_this_k_and_budget",
            "generic_smoothing": "distance_knn_topology_not_isolated_from_spectral_control",
            "temporal_sufficient": "temporal_smoothing_explains_gain_for_this_k_and_budget",
            "quick_topology_signal": "quick_signal_requires_standard_and_graph_construction_validation",
            "candidate_topology_specific_candidate": "candidate_graph_favorable_for_this_k_and_budget_only",
            "inconclusive": "missing_controls_or_insufficient_evidence",
        }
        diagnostic_failure_mode = nbody_failure.get(strict_label, diagnostic_failure_mode)
        recommended_next_experiment = "run_multi_seed_validation_and_soft_radial_or_inverse_distance_graph_sweep"
        claim_boundary = "Applies only to this distance-kNN construction, seed, budget, and model condition."
    elif family == "ho_lattice" and classification == "topology_aligned_latent_smoothing":
        diagnostic_failure_mode = "controlled_audit_positive_case"
        recommended_next_experiment = "use_as_controlled_positive_case_then_test_unknown_topology_audit_cases"
        claim_boundary = "Controlled positive case, not evidence that audit triggers reliably in unknown-topology regimes."

    return {
        "strict_label": strict_label,
        "diagnostic_failure_mode": diagnostic_failure_mode,
        "recommended_next_experiment": recommended_next_experiment,
        "claim_boundary": claim_boundary,
    }


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
    diagnostic_fields: dict[str, str],
    oracle_row: dict[str, Any] | None = None,
) -> None:
    log_progress(f"Report writing start: {rel_to_root(path)}")
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
                ["transition type", args.transition_type],
                ["transition GNN layers", args.transition_gnn_layers],
                ["decode loss weight", args.decode_loss_weight],
                ["node feature dim", args.node_feature_dim],
                ["raw normalization", args.laplacian_normalization],
                ["N permuted graphs", args.n_permuted],
                ["M random graphs", 0 if args.skip_random_graph else args.n_random],
                ["include temporal prior", bool(args.include_temporal_prior)],
                ["calibrate prior strength", bool(args.calibrate_prior_strength)],
                ["calibration reference prior", args.calibration_reference_prior],
                ["calibration target ratio", args.calibration_target_ratio],
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
            [
                "prior",
                "H=16",
                "H=32",
                "nominal lambda",
                "effective lambda",
                "initial prior loss",
                "prior loss mean",
                "effective prior contribution",
                "final train loss",
            ],
            [
                [
                    prior,
                    fmt_mean_std([row.get("H16") for row in rows], 4),
                    fmt_mean_std([row.get("H32") for row in rows], 4),
                    fmt_mean_std([row.get("nominal_prior_weight") for row in rows], 6),
                    fmt_mean_std([row.get("effective_prior_weight") for row in rows], 6),
                    fmt_mean_std([row.get("initial_prior_loss") for row in rows], 6),
                    fmt_mean_std([row.get("prior_loss_mean") for row in rows], 4),
                    fmt_mean_std([row.get("effective_prior_contribution_mean") for row in rows], 6),
                    fmt_mean_std([row.get("final_train_loss") for row in rows], 6),
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
            "## Strict Protocol Interpretation",
            "",
            *report_table(
                ["field", "value"],
                [
                    ["Legacy automatic classification", classification],
                    ["Strict manuscript label", diagnostic_fields.get("strict_label", "inconclusive")],
                    ["Diagnostic failure mode", diagnostic_fields.get("diagnostic_failure_mode", "missing_controls_or_insufficient_evidence")],
                    ["Recommended next experiment", diagnostic_fields.get("recommended_next_experiment", "run_missing_controls_or_collect_required_artifacts")],
                    ["Claim boundary", diagnostic_fields.get("claim_boundary", "Missing controls or artifacts prevent a stronger claim.")],
                ],
            ),
            "",
            "The strict label follows the staged manuscript hierarchy and should be preferred over the legacy automatic classification when they disagree.",
            "",
            "## Final Classification",
            "",
            f"`{classification}`",
            "",
            *temporal_prior_interpretation(stage1_rows),
            "Decision rules:",
            "- `no_graph_gain`: true graph mini run does not beat GNN none at H=32.",
            "- `generic_smoothing`: true graph beats none but does not beat the permuted graph at H=32.",
            "- `candidate_topology_specific`: true graph beats both none and permuted, but latent alignment is absent or not audited.",
            "- `topology_aligned_latent_smoothing`: true graph beats both rollout controls and its learned `Delta_H` is smoother/more low-frequency in the true graph basis.",
            "- `temporal_smooth`: graph-free temporal baseline; it does not change the graph classification by itself.",
            "",
            "## Files",
            "",
            f"- `{rel_to_root(args.summary)}`",
        ]
    )
    if stage2_rows:
        lines.append(f"- latent artifacts under `{rel_to_root(args.artifact_dir)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log_progress(f"Report writing end: {rel_to_root(path)}")


def rows_for_stage(rows: list[dict[str, Any]], stage: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("stage") == stage and row.get("status", "ok") == "ok"]


def first_prior_row(rows: list[dict[str, Any]], prior: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("prior") == prior:
            return row
    return None


def temporal_prior_interpretation(stage1_rows: list[dict[str, Any]]) -> list[str]:
    h32 = grouped_rollout_means(stage1_rows, "32")
    none = h32.get("none", float("nan"))
    graph = h32.get("graph", float("nan"))
    permuted = h32.get("permuted_graph", float("nan"))
    temporal = h32.get("temporal_smooth", float("nan"))
    if not finite(temporal):
        return []
    lines = [
        "## Temporal Prior Interpretation",
        "",
        (
            "H=32 temporal_smooth rollout: "
            f"`{fmt(temporal, 4)}`; temporal gain vs none: `{fmt_pct(pct_gain(none, temporal))}`; "
            f"graph gain vs temporal_smooth: `{fmt_pct(pct_gain(temporal, graph))}`."
        ),
        "",
    ]
    if finite(graph) and finite(none) and graph >= none and temporal < none:
        lines.append("Interpretation: graph-free temporal smoothing may be preferable.")
    elif finite(graph) and graph < none and temporal <= graph:
        lines.append("Interpretation: graph gain may be temporal-smoothing-like rather than topology-specific.")
    elif finite(graph) and finite(permuted) and graph < temporal and graph < permuted:
        lines.append(
            "Interpretation: graph beats temporal_smooth and permuted_graph, which is stronger evidence for topology-specific graph usefulness."
        )
    else:
        lines.append(
            "Interpretation: temporal_smooth is included as a graph-free baseline; use it to separate temporal regularization from graph-specific effects."
        )
    lines.append("")
    return lines


def write_graph_heat_failure_audit(
    path: Path,
    *,
    args: argparse.Namespace,
    raw_rows: list[dict[str, Any]],
    oracle_row: dict[str, Any],
    existing_rows: list[dict[str, Any]],
    lambda_rows: list[dict[str, Any]],
) -> None:
    log_progress(f"Report writing start: {rel_to_root(path)}")
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
    log_progress(f"Report writing end: {rel_to_root(path)}")


def graph_lowfreq_failure_analysis(args: argparse.Namespace, adapter: DatasetAdapter) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    training_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    for epoch_count in (20, 50):
        epoch_args = copy.copy(args)
        epoch_args.epochs = int(epoch_count)
        for prior in ("none", "graph", "permuted_graph"):
            config = adapter.training_config(epoch_args, prior=prior, seed=0)
            config.run_name = f"{config.run_name}_epochs{epoch_count}"
            try:
                row, model, device = mini_train(config, adapter)
                row["stage"] = "graph_lowfreq_training_analysis"
                row["analysis_epochs"] = int(epoch_count)
                training_rows.append(row)
                if prior in {"graph", "permuted_graph"}:
                    artifact = collect_preflight_latent_trace(model, config, adapter, device)
                    artifact_path = args.artifact_dir / f"{config.run_name}_latents.pt"
                    torch.save(artifact, artifact_path)
                    metrics = artifact_metrics(artifact)
                    latent_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "stage": "graph_lowfreq_latent_analysis",
                            "run_name": config.run_name,
                            "dataset": adapter.name,
                            "status": "ok",
                            "topology": config.topology,
                            "prior": prior,
                            "prior_weight": config.prior_weight,
                            "seed": 0,
                            "analysis_epochs": int(epoch_count),
                            "latent_artifact_path": rel_to_root(artifact_path),
                            **metrics,
                        }
                    )
            except Exception as exc:
                training_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "stage": "graph_lowfreq_training_analysis",
                        "dataset": adapter.name,
                        "status": "failed",
                        "topology": adapter.metadata().get("topology"),
                        "encoder": "gnn_node",
                        "prior": prior,
                        "prior_weight": float(epoch_args.prior_weight),
                        "seed": 0,
                        "analysis_epochs": int(epoch_count),
                        "error": str(exc),
                        "H16": float("nan"),
                        "H32": float("nan"),
                    }
                )
    return training_rows, latent_rows


def row_for_epochs_prior(rows: list[dict[str, Any]], epoch_count: int, prior: str) -> dict[str, Any] | None:
    for row in rows:
        if int(row.get("analysis_epochs", -1)) == int(epoch_count) and str(row.get("prior")) == prior:
            return row
    return None


def write_graph_lowfreq_failure_analysis(
    path: Path,
    *,
    args: argparse.Namespace,
    raw_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    latent_rows: list[dict[str, Any]],
) -> None:
    log_progress(f"Report writing start: {rel_to_root(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_by_type = {str(row.get("graph_type")): row for row in raw_rows}
    comparison_rows: list[list[Any]] = []
    latent_table_rows: list[list[Any]] = []
    overtake_flags: list[bool] = []
    latent_smoother_flags: list[bool] = []
    for epoch_count in (20, 50):
        none_row = row_for_epochs_prior(training_rows, epoch_count, "none")
        graph_row = row_for_epochs_prior(training_rows, epoch_count, "graph")
        perm_row = row_for_epochs_prior(training_rows, epoch_count, "permuted_graph")
        none_h32 = float(none_row.get("H32", float("nan"))) if none_row else float("nan")
        graph_h32 = float(graph_row.get("H32", float("nan"))) if graph_row else float("nan")
        perm_h32 = float(perm_row.get("H32", float("nan"))) if perm_row else float("nan")
        overtake = finite(graph_h32) and finite(perm_h32) and graph_h32 < perm_h32
        overtake_flags.append(overtake)
        comparison_rows.append(
            [
                epoch_count,
                fmt(none_row.get("H16") if none_row else float("nan"), 4),
                fmt(none_h32, 4),
                fmt(graph_row.get("H16") if graph_row else float("nan"), 4),
                fmt(graph_h32, 4),
                fmt(perm_row.get("H16") if perm_row else float("nan"), 4),
                fmt(perm_h32, 4),
                "YES" if overtake else "NO",
            ]
        )
        graph_latent = row_for_epochs_prior(latent_rows, epoch_count, "graph")
        perm_latent = row_for_epochs_prior(latent_rows, epoch_count, "permuted_graph")
        graph_d = float(graph_latent.get("D_true_Delta_H_norm", float("nan"))) if graph_latent else float("nan")
        perm_d = float(perm_latent.get("D_true_Delta_H_norm", float("nan"))) if perm_latent else float("nan")
        graph_r4 = float(graph_latent.get("R_low_true_Delta_H_4", float("nan"))) if graph_latent else float("nan")
        perm_r4 = float(perm_latent.get("R_low_true_Delta_H_4", float("nan"))) if perm_latent else float("nan")
        latent_smoother = finite(graph_d) and finite(perm_d) and graph_d < perm_d and finite(graph_r4) and finite(perm_r4) and graph_r4 > perm_r4
        latent_smoother_flags.append(latent_smoother)
        latent_table_rows.append(
            [
                epoch_count,
                fmt(graph_d, 4),
                fmt(perm_d, 4),
                fmt(graph_r4, 4),
                fmt(perm_r4, 4),
                "YES" if latent_smoother else "NO",
            ]
        )

    any_overtake = any(overtake_flags)
    any_latent_smooth = any(latent_smoother_flags)
    if any_overtake:
        diagnosis = "longer mini-training is enough for the true graph to overtake the permuted control."
    elif any_latent_smooth:
        diagnosis = "the graph prior induces more true-graph-smooth latent deltas, but rollout still favors the permuted control; this points to latent-transition mismatch."
    else:
        diagnosis = "generic smoothing dominance remains under this latent model; longer mini-training did not expose topology specificity."

    lines = [
        "# Graph Low-Frequency Failure Analysis",
        "",
        f"Created: `{datetime.now(timezone.utc).isoformat()}`",
        "Scope: graph_lowfreq_lattice only; seed 0; prior weight 0.1; epochs 20 and 50.",
        "No ISO17, rMD17 top-up, or broad sweeps were run.",
        "",
        "## Raw Alignment",
        "",
        *report_table(
            ["graph type", "D_dX_norm", "R_low K=4", "D gap control-true"],
            [
                [
                    graph_type,
                    fmt(row.get("D_dX_norm"), 6),
                    fmt(row.get("R_low_dX_4"), 4),
                    fmt(row.get("gap_D_dX_norm_control_minus_true"), 6),
                ]
                for graph_type, row in raw_by_type.items()
            ],
        ),
        "",
        "## Longer Mini-Training Rollout",
        "",
        *report_table(
            ["epochs", "none H16", "none H32", "graph H16", "graph H32", "permuted H16", "permuted H32", "graph beats permuted H32"],
            comparison_rows,
        ),
        "",
        "## Forced Latent Audit",
        "",
        *report_table(
            ["epochs", "graph D_true_norm(Delta_H)", "permuted D_true_norm(Delta_H)", "graph R_low K=4", "permuted R_low K=4", "graph latent smoother"],
            latent_table_rows,
        ),
        "",
        "## Answers",
        "",
        f"1. Does true graph overtake permuted with longer mini-training? `{'YES' if any_overtake else 'NO'}`.",
        f"2. Does graph prior induce more true-graph-smooth latent dynamics even when rollout is worse? `{'YES' if any_latent_smooth else 'NO'}`.",
        f"3. Main diagnosis: {diagnosis}",
        "",
        "## Files",
        "",
        f"- `{rel_to_root(path)}`",
        f"- latent artifacts under `{rel_to_root(args.artifact_dir)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log_progress(f"Report writing end: {rel_to_root(path)}")


def parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight graph-prior preflight check.")
    parser.add_argument("--dataset", default=None, help="Dataset adapter name, e.g. ho_lattice.")
    parser.add_argument(
        "--topology",
        default=None,
        choices=["chain", "ring", "lattice", "random", "scalefree"],
        help="Topology alias for HO, graph_heat, graph_lowfreq, spring_mass, or graph_wave datasets.",
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
    parser.add_argument("--transition-type", default="mlp_pooled", choices=["mlp_pooled", "graph_node_sage"])
    parser.add_argument("--transition-gnn-layers", type=int, default=1)
    parser.add_argument("--decode-loss-weight", type=float, default=1.0)
    parser.add_argument("--node-feature-dim", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-latent-audit", action="store_true")
    parser.add_argument("--include-temporal-prior", action="store_true")
    parser.add_argument("--calibrate-prior-strength", action="store_true")
    parser.add_argument(
        "--calibration-reference-prior",
        default="graph",
        choices=["graph", "permuted_graph", "temporal_smooth"],
    )
    parser.add_argument("--calibration-target-ratio", type=float, default=1.0)
    parser.add_argument("--graph-heat-failure-audit-only", action="store_true")
    parser.add_argument("--graph-lowfreq-failure-analysis-only", action="store_true")
    parser.add_argument("--graph-heat-n", type=int, default=36)
    parser.add_argument("--graph-heat-d", type=int, default=4)
    parser.add_argument("--graph-heat-t", type=int, default=512)
    parser.add_argument("--graph-heat-tau", type=float, default=0.05)
    parser.add_argument("--graph-heat-noise", type=float, default=0.01)
    parser.add_argument("--graph-heat-seed", type=int, default=0)
    parser.add_argument("--lowfreq-n", type=int, default=36)
    parser.add_argument("--lowfreq-d", type=int, default=4)
    parser.add_argument("--lowfreq-t", type=int, default=512)
    parser.add_argument("--lowfreq-k", type=int, default=4)
    parser.add_argument("--lowfreq-noise", type=float, default=0.01)
    parser.add_argument("--lowfreq-decay", type=float, default=0.95)
    parser.add_argument("--lowfreq-seed", type=int, default=0)
    parser.add_argument("--spring-n", type=int, default=36)
    parser.add_argument("--spring-position-dim", type=int, default=1)
    parser.add_argument("--spring-t", type=int, default=512)
    parser.add_argument("--spring-k", type=float, default=1.0)
    parser.add_argument("--spring-damping", type=float, default=0.05)
    parser.add_argument("--spring-dt", type=float, default=0.05)
    parser.add_argument("--spring-noise", type=float, default=0.001)
    parser.add_argument("--spring-seed", type=int, default=0)
    parser.add_argument("--wave-n", type=int, default=36)
    parser.add_argument("--wave-t", type=int, default=512)
    parser.add_argument("--wave-c", type=float, default=1.0)
    parser.add_argument("--wave-damping", type=float, default=0.02)
    parser.add_argument("--wave-dt", type=float, default=0.05)
    parser.add_argument("--wave-noise", type=float, default=0.001)
    parser.add_argument("--wave-seed", type=int, default=0)
    parser.add_argument("--nbody-n", type=int, default=36)
    parser.add_argument("--nbody-position-dim", type=int, default=2)
    parser.add_argument("--nbody-t", type=int, default=512)
    parser.add_argument("--nbody-g", type=float, default=0.1)
    parser.add_argument("--nbody-dt", type=float, default=0.02)
    parser.add_argument("--nbody-softening", type=float, default=0.1)
    parser.add_argument("--nbody-damping", type=float, default=0.0)
    parser.add_argument("--nbody-noise", type=float, default=0.0005)
    parser.add_argument("--nbody-graph-k", type=int, default=8)
    parser.add_argument("--nbody-seed", type=int, default=0)
    parser.add_argument("--metr-la-root", type=Path, default=DEFAULT_METR_LA_ROOT)
    parser.add_argument("--metr-la-csv", type=Path, default=None)
    parser.add_argument("--metr-la-adj-pkl", type=Path, default=None)
    parser.add_argument("--metr-la-no-pyg-temporal", action="store_true")
    parser.add_argument("--metr-la-graph-source", choices=["adjacency_pkl", "correlation_topk"], default="adjacency_pkl")
    parser.add_argument("--metr-la-top-k", type=int, default=8)
    parser.add_argument("--metr-la-corr-mode", choices=["positive", "absolute"], default="positive")
    parser.add_argument("--metr-la-max-timesteps", type=int, default=None)
    args = parser.parse_args()
    if args.metr_la_max_timesteps is not None and args.metr_la_max_timesteps < 2:
        raise ValueError("--metr-la-max-timesteps must be at least 2 when set")
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
    if str(args.dataset).startswith("graph_lowfreq_"):
        dataset_topology = str(args.dataset).removeprefix("graph_lowfreq_")
        if args.topology is not None and args.topology != dataset_topology:
            raise ValueError(f"--dataset {args.dataset!r} conflicts with --topology {args.topology!r}")
        args.topology = dataset_topology
    if str(args.dataset) == "graph_lowfreq" and args.topology is None:
        args.topology = "lattice"
    if str(args.dataset).startswith("spring_mass_"):
        dataset_topology = str(args.dataset).removeprefix("spring_mass_")
        if args.topology is not None and args.topology != dataset_topology:
            raise ValueError(f"--dataset {args.dataset!r} conflicts with --topology {args.topology!r}")
        args.topology = dataset_topology
    if str(args.dataset) == "spring_mass" and args.topology is None:
        args.topology = "lattice"
    if str(args.dataset).startswith("graph_wave_"):
        dataset_topology = str(args.dataset).removeprefix("graph_wave_")
        if args.topology is not None and args.topology != dataset_topology:
            raise ValueError(f"--dataset {args.dataset!r} conflicts with --topology {args.topology!r}")
        args.topology = dataset_topology
    if str(args.dataset) == "graph_wave" and args.topology is None:
        args.topology = "lattice"
    if str(args.dataset) == "nbody_distance":
        args.topology = "distance_knn"
    if (
        str(args.dataset).startswith("graph_heat")
        or str(args.dataset).startswith("graph_lowfreq")
        or str(args.dataset).startswith("spring_mass")
        or str(args.dataset).startswith("graph_wave")
        or str(args.dataset) == "nbody_distance"
    ):
        if args.raw_stride == 10:
            args.raw_stride = 5
        if args.train_stride == 10:
            args.train_stride = 5
        if args.eval_stride == 20:
            args.eval_stride = 10
    args.metr_la_root = args.metr_la_root.resolve()
    if args.metr_la_csv is not None:
        args.metr_la_csv = args.metr_la_csv.resolve()
    if args.metr_la_adj_pkl is not None:
        args.metr_la_adj_pkl = args.metr_la_adj_pkl.resolve()
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

    output_dir = args.out_dir or args.report.parent
    log_progress(f"Creating output directory: {rel_to_root(output_dir)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(args)

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

    if args.graph_lowfreq_failure_analysis_only:
        raw_rows = stage0_raw_diagnostics(args, adapter)
        training_rows, latent_rows = graph_lowfreq_failure_analysis(args, adapter)
        analysis_path = (args.out_dir or args.summary.parent) / "FAILURE_ANALYSIS.md"
        write_graph_lowfreq_failure_analysis(
            analysis_path.resolve(),
            args=args,
            raw_rows=raw_rows,
            training_rows=training_rows,
            latent_rows=latent_rows,
        )
        write_summary_csv((args.out_dir or args.summary.parent) / "failure_analysis_summary.csv", [*raw_rows, *training_rows, *latent_rows])
        print(f"Wrote {rel_to_root(analysis_path.resolve())}")
        return

    raw_rows = stage0_raw_diagnostics(args, adapter)
    stage1_rows, models, devices = stage1_mini_training(args, adapter)
    audit = bool(args.force_latent_audit or should_run_latent_audit(stage1_rows))
    stage2_rows = stage2_latent_audit(args, adapter, models, devices) if audit else []
    classification, class_metrics = classify(stage1_rows, stage2_rows)
    diagnostic_fields = diagnostic_interpretation(args, classification, class_metrics, adapter.metadata())
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
            **diagnostic_fields,
            **class_metrics,
            **{f"metadata_{key}": value for key, value in adapter.metadata().items()},
        }
    )
    write_summary_csv(args.summary, all_rows)
    write_report(args.report, args, raw_rows, stage1_rows, stage2_rows, classification, class_metrics, diagnostic_fields, oracle_row)
    print(f"Wrote {rel_to_root(args.report)}")
    print(f"Wrote {rel_to_root(args.summary)} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
