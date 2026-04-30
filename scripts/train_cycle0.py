from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rmd17_loader import RMD17Trajectory


CONFIG_DIR = ROOT / "experiments" / "configs" / "cycle0_aspirin_smoke"
RESULT_DIR = ROOT / "experiments" / "results" / "cycle0_aspirin_smoke"
DEFAULT_OUTPUT = RESULT_DIR / "cycle0_aspirin_smoke_results.json"
DEFAULT_DATA_ROOT = ROOT / "data" / "rmd17_raw" / "rmd17" / "npz_data"
SCHEMA_VERSION = "cycle0_smoke_v1"
GRAPH_PRIORS = {"graph", "permuted_graph", "random_graph"}
GLOBAL_PRIORS = {"none", "variance", "covariance", "sigreg"}


@dataclass
class Cycle0Config:
    run_name: str
    molecule: str = "aspirin"
    encoder: str = "mlp_global"
    prior: str = "none"
    seed: int = 0
    num_epochs: int = 3
    n_transitions: int = 96
    eval_n_transitions: int = 24
    stride: int = 10
    eval_stride: int = 20
    horizon: int = 1
    eval_horizons: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    latent_dim: int = 16
    node_dim: int = 16
    hidden_dim: int = 64
    transition_hidden_dim: int = 64
    batch_size: int = 16
    lr: float = 1e-3
    prior_weight: float = 0.1
    sigreg_num_slices: int = 8
    device: str = "auto"
    data_root: str = str(DEFAULT_DATA_ROOT)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    raise ValueError("device must be 'auto', 'cpu', or 'cuda'")


def load_config(path: Path) -> Cycle0Config:
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if "eval_horizons" in raw:
        raw["eval_horizons"] = tuple(int(value) for value in raw["eval_horizons"])
    return Cycle0Config(**raw)


def save_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)
    os.replace(tmp, path)


def load_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runs": {},
        }
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("runs", {})
    return payload


def collect_transitions(
    traj: RMD17Trajectory,
    *,
    n_transitions: int,
    stride: int,
    horizon: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    max_start = traj.n_frames - horizon - 1
    candidates = np.arange(0, max_start, stride)
    if n_transitions > len(candidates):
        raise ValueError(f"Requested {n_transitions}, but only {len(candidates)} candidates exist")
    chosen = rng.choice(candidates, size=n_transitions, replace=False)
    chosen.sort()
    transitions = []
    for frame_idx in chosen:
        obs, next_obs, energy, _force = traj.get_pair(int(frame_idx), horizon=horizon)
        transitions.append(
            {
                "obs": obs,
                "next_obs": next_obs,
                "frame_idx": int(frame_idx),
                "horizon": int(horizon),
                "energy": float(energy),
                "molecule": traj.molecule,
            }
        )
    return transitions


def move_obs(obs: HeteroData, device: torch.device) -> HeteroData:
    return obs.clone().to(device)


class MLPGlobalEncoder(nn.Module):
    """Global MLP encoder: positions plus atomic numbers -> z in R^D."""

    emits_node_states = False

    def __init__(self, n_atoms: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.n_atoms = int(n_atoms)
        self.latent_dim = int(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(self.n_atoms * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, obs: HeteroData) -> tuple[Tensor, Tensor | None]:
        pos = obs["atom"].pos
        z_atom = obs["atom"].atomic_number.to(dtype=pos.dtype, device=pos.device).view(-1, 1) / 10.0
        features = torch.cat([z_atom, pos], dim=-1).reshape(-1)
        if features.numel() != self.n_atoms * 4:
            raise ValueError(f"Expected {self.n_atoms} atoms, got flattened size {features.numel()}")
        return self.net(features), None


class GNNNodeEncoder(nn.Module):
    """Node-wise graph encoder: emits H in R^{N x d_node}, z = mean_pool(H)."""

    emits_node_states = True

    def __init__(self, latent_dim: int, node_dim: int, hidden_dim: int):
        super().__init__()
        if latent_dim != node_dim:
            raise ValueError("Cycle 0 uses z=mean(H), so latent_dim must equal node_dim")
        self.input_proj = nn.Linear(4, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, node_dim)

    def forward(self, obs: HeteroData) -> tuple[Tensor, Tensor | None]:
        pos = obs["atom"].pos
        z_atom = obs["atom"].atomic_number.to(dtype=pos.dtype, device=pos.device).view(-1, 1) / 10.0
        x = torch.cat([z_atom, pos], dim=-1)
        edge_index = obs["atom", "bonded", "atom"].edge_index
        h = F.relu(self.input_proj(x))
        h = F.relu(self.conv1(h, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        node_states = self.out(h)
        return node_states.mean(dim=0), node_states


class Cycle0LatentDynamics(nn.Module):
    def __init__(self, config: Cycle0Config, n_atoms: int):
        super().__init__()
        if config.encoder == "mlp_global":
            self.encoder = MLPGlobalEncoder(n_atoms, config.latent_dim, config.hidden_dim)
        elif config.encoder == "gnn_node":
            self.encoder = GNNNodeEncoder(config.latent_dim, config.node_dim, config.hidden_dim)
        else:
            raise ValueError("encoder must be 'mlp_global' or 'gnn_node'")
        self.transition = nn.Sequential(
            nn.Linear(config.latent_dim + 1, config.transition_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.transition_hidden_dim, config.transition_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.transition_hidden_dim, config.latent_dim),
        )

    @property
    def emits_node_states(self) -> bool:
        return bool(getattr(self.encoder, "emits_node_states", False))

    def encode(self, obs: HeteroData) -> tuple[Tensor, Tensor | None]:
        return self.encoder(obs)

    def step_latent(self, z: Tensor, action: Tensor | None = None) -> Tensor:
        if action is None:
            action = z.new_zeros((1,))
        if action.dim() == 0:
            action = action.view(1)
        return z + self.transition(torch.cat([z, action.to(device=z.device, dtype=z.dtype)], dim=-1))

    def rollout_latent(self, z0: Tensor, horizon: int) -> Tensor:
        latents = [z0]
        z = z0
        for _ in range(horizon):
            z = self.step_latent(z)
            latents.append(z)
        return torch.stack(latents, dim=0)


def covariance_penalty(z_batch: Tensor) -> Tensor:
    if z_batch.shape[0] < 2:
        return z_batch.new_tensor(0.0)
    zc = z_batch - z_batch.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (z_batch.shape[0] - 1)
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    return torch.norm(cov - eye, p="fro")


def variance_penalty(z_batch: Tensor, gamma: float = 1.0) -> Tensor:
    if z_batch.shape[0] < 2:
        return z_batch.new_tensor(0.0)
    std = torch.sqrt(z_batch.var(dim=0, unbiased=False) + 1e-6)
    return F.relu(gamma - std).mean()


def projection_gaussianity_penalty(z_batch: Tensor, num_slices: int, seed: int) -> Tensor:
    if z_batch.shape[0] < 4:
        return z_batch.new_tensor(0.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1009 + int(seed))
    directions = torch.randn((z_batch.shape[1], num_slices), generator=generator)
    directions = F.normalize(directions, dim=0).to(device=z_batch.device, dtype=z_batch.dtype)
    proj = z_batch @ directions
    centered = proj - proj.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-6)
    normalized = centered / std.view(1, -1)
    skew = (normalized ** 3).mean(dim=0)
    kurt = (normalized ** 4).mean(dim=0)
    return (proj.mean(dim=0).pow(2) + (std - 1.0).pow(2) + skew.pow(2) + (kurt - 3.0).pow(2)).mean()


def projection_gaussianity_statistic_np(z: np.ndarray, num_slices: int = 16, seed: int = 0) -> float:
    if z.shape[0] < 4:
        return float("nan")
    rng = np.random.default_rng(1009 + int(seed))
    directions = rng.normal(size=(z.shape[1], num_slices))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True) + 1e-12
    proj = z @ directions
    centered = proj - proj.mean(axis=0, keepdims=True)
    std = np.sqrt(np.var(centered, axis=0) + 1e-6)
    normalized = centered / std.reshape(1, -1)
    skew = np.mean(normalized ** 3, axis=0)
    kurt = np.mean(normalized ** 4, axis=0)
    stat = np.mean(np.mean(proj, axis=0) ** 2 + (std - 1.0) ** 2 + skew ** 2 + (kurt - 3.0) ** 2)
    return float(stat)


def undirected_edges_and_weights(obs: HeteroData) -> tuple[Tensor, Tensor]:
    edge_index = obs["atom", "bonded", "atom"].edge_index
    edge_attr = obs["atom", "bonded", "atom"].edge_attr
    pairs: list[tuple[int, int]] = []
    weights: list[float] = []
    seen: set[tuple[int, int]] = set()
    for k in range(edge_index.shape[1]):
        i = int(edge_index[0, k].item())
        j = int(edge_index[1, k].item())
        if i == j:
            continue
        pair = (i, j) if i < j else (j, i)
        if pair in seen:
            continue
        seen.add(pair)
        distance = float(edge_attr[k].view(-1)[0].item()) if edge_attr.numel() else 1.0
        pairs.append(pair)
        weights.append(1.0 / max(distance, 1e-6))
    device = obs["atom"].pos.device
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty((0,), device=device)
    return (
        torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous(),
        torch.tensor(weights, dtype=obs["atom"].pos.dtype, device=device),
    )


def build_node_laplacian(n_nodes: int, edge_pairs: Tensor, weights: Tensor, dtype: torch.dtype) -> Tensor:
    """Build weighted node Laplacian L in R^{N_nodes x N_nodes}."""
    device = edge_pairs.device
    laplacian = torch.zeros((n_nodes, n_nodes), dtype=dtype, device=device)
    if edge_pairs.numel() == 0:
        return laplacian
    src = edge_pairs[0]
    dst = edge_pairs[1]
    laplacian.index_put_((src, src), weights, accumulate=True)
    laplacian.index_put_((dst, dst), weights, accumulate=True)
    laplacian.index_put_((src, dst), -weights, accumulate=True)
    laplacian.index_put_((dst, src), -weights, accumulate=True)
    return laplacian


def node_laplacian_trace(node_states: Tensor, laplacian: Tensor) -> Tensor:
    """True graph prior: Tr(H^T L H) = sum_(i,j in E) w_ij ||h_i - h_j||^2."""
    assert node_states is not None
    assert node_states.ndim == 2
    assert laplacian.ndim == 2
    assert laplacian.shape[0] == laplacian.shape[1] == node_states.shape[0]
    return torch.trace(node_states.T @ laplacian @ node_states)


def random_edge_pairs(n_nodes: int, n_edges: int, seed: int, device: torch.device) -> Tensor:
    if n_edges <= 0 or n_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    rng = np.random.default_rng(seed)
    all_pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    choice = rng.choice(len(all_pairs), size=min(n_edges, len(all_pairs)), replace=False)
    pairs = np.asarray([all_pairs[int(index)] for index in choice], dtype=np.int64)
    return torch.from_numpy(pairs).to(device=device).t().contiguous()


def graph_prior_loss(
    prior: str,
    obs: HeteroData,
    node_states: Tensor | None,
    *,
    seed: int,
    frame_idx: int,
) -> Tensor:
    assert node_states is not None
    assert node_states.ndim == 2
    edge_pairs, weights = undirected_edges_and_weights(obs)
    if prior == "graph":
        laplacian = build_node_laplacian(
            node_states.shape[0],
            edge_pairs,
            weights,
            dtype=node_states.dtype,
        )
        assert laplacian.ndim == 2
        assert laplacian.shape[0] == laplacian.shape[1] == node_states.shape[0]
        return node_laplacian_trace(node_states, laplacian)
    if prior == "permuted_graph":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(2003 + int(seed) * 100000 + int(frame_idx))
        perm = torch.randperm(node_states.shape[0], generator=generator).to(node_states.device)
        laplacian = build_node_laplacian(
            node_states.shape[0],
            edge_pairs,
            weights,
            dtype=node_states.dtype,
        )
        assert laplacian.ndim == 2
        assert laplacian.shape[0] == laplacian.shape[1] == node_states.shape[0]
        return node_laplacian_trace(node_states[perm], laplacian)
    if prior == "random_graph":
        random_edges = random_edge_pairs(
            node_states.shape[0],
            int(edge_pairs.shape[1]),
            seed=3001 + int(seed) * 100000 + int(frame_idx),
            device=node_states.device,
        )
        random_weights = weights.mean().expand(random_edges.shape[1]) if weights.numel() else weights
        laplacian = build_node_laplacian(
            node_states.shape[0],
            random_edges,
            random_weights,
            dtype=node_states.dtype,
        )
        assert laplacian.ndim == 2
        assert laplacian.shape[0] == laplacian.shape[1] == node_states.shape[0]
        return node_laplacian_trace(node_states, laplacian)
    raise ValueError(f"Unknown graph prior {prior!r}")


def validate_config(config: Cycle0Config) -> None:
    if config.prior in GRAPH_PRIORS and config.encoder != "gnn_node":
        raise ValueError(
            "Cycle 0 graph priors are defined only for encoders that emit node-wise H. "
            "Use encoder='gnn_node' or choose a global-z prior."
        )
    if config.encoder == "mlp_global" and config.prior in GRAPH_PRIORS:
        raise ValueError("MLP graph priors are disabled unless MLP explicitly emits node-wise H")
    if config.prior not in {*GLOBAL_PRIORS, *GRAPH_PRIORS}:
        raise ValueError(f"Unknown prior {config.prior!r}")


def train_batch(
    model: Cycle0LatentDynamics,
    transitions: list[dict[str, Any]],
    config: Cycle0Config,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    transition_losses: list[Tensor] = []
    prior_losses: list[Tensor] = []
    z_values: list[Tensor] = []
    for transition in transitions:
        obs = move_obs(transition["obs"], device)
        next_obs = move_obs(transition["next_obs"], device)
        z, h = model.encode(obs)
        with torch.no_grad():
            z_next, _ = model.encode(next_obs)
        z_pred = model.step_latent(z)
        transition_losses.append(F.mse_loss(z_pred, z_next))
        z_values.append(z)
        if config.prior in GRAPH_PRIORS:
            prior_losses.append(
                graph_prior_loss(
                    config.prior,
                    obs,
                    h,
                    seed=config.seed,
                    frame_idx=int(transition["frame_idx"]),
                )
            )

    transition_loss = torch.stack(transition_losses).mean()
    z_batch = torch.stack(z_values, dim=0)
    if config.prior == "variance":
        prior_loss = variance_penalty(z_batch)
    elif config.prior == "covariance":
        prior_loss = covariance_penalty(z_batch)
    elif config.prior == "sigreg":
        prior_loss = projection_gaussianity_penalty(z_batch, config.sigreg_num_slices, config.seed)
    elif config.prior in GRAPH_PRIORS:
        prior_loss = torch.stack(prior_losses).mean()
    else:
        prior_loss = transition_loss.new_tensor(0.0)
    total = transition_loss + config.prior_weight * prior_loss
    return total, transition_loss, prior_loss, z_batch


def evaluate_rollout(
    model: Cycle0LatentDynamics,
    traj: RMD17Trajectory,
    eval_transitions: list[dict[str, Any]],
    horizons: tuple[int, ...],
    device: torch.device,
    latent_dim: int,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    out: dict[str, float] = {}
    max_h = max(horizons)
    with torch.no_grad():
        for horizon in horizons:
            errors: list[float] = []
            for transition in eval_transitions:
                frame_idx = int(transition["frame_idx"])
                if frame_idx + max_h >= traj.n_frames:
                    continue
                obs0 = move_obs(traj[frame_idx], device)
                obs_h = move_obs(traj[frame_idx + horizon], device)
                z0, _ = model.encode(obs0)
                z_pred = model.rollout_latent(z0, horizon)[-1]
                z_true, _ = model.encode(obs_h)
                errors.append(float((torch.norm(z_pred - z_true, p=2) / math.sqrt(latent_dim)).detach().cpu()))
            out[str(horizon)] = float(np.mean(errors)) if errors else float("nan")
    if was_training:
        model.train()
    return out


def collect_latents(
    model: Cycle0LatentDynamics,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for sample in samples:
            z, _ = model.encode(move_obs(sample["obs"], device))
            values.append(z.detach().cpu().numpy())
    return np.stack(values, axis=0)


def latent_diagnostics(z: np.ndarray, seed: int) -> dict[str, Any]:
    if z.shape[0] < 2:
        return {
            "effective_rank": float("nan"),
            "covariance_condition_number": float("nan"),
            "projection_gaussianity_statistic": float("nan"),
        }
    centered = z - z.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(eigvals.sum())
    if total <= 0.0:
        effective_rank = 0.0
    else:
        probs = eigvals / total
        probs = probs[probs > 0.0]
        effective_rank = float(np.exp(-(probs * np.log(probs)).sum()))
    positive = eigvals[eigvals > 1e-10]
    condition = float(positive.max() / positive.min()) if positive.size else float("inf")
    return {
        "effective_rank": effective_rank,
        "covariance_condition_number": condition,
        "projection_gaussianity_statistic": projection_gaussianity_statistic_np(z, seed=seed),
    }


def train_one(config: Cycle0Config) -> dict[str, Any]:
    validate_config(config)
    set_seed(config.seed)
    device = select_device(config.device)
    traj = RMD17Trajectory(config.molecule, data_root=Path(config.data_root))
    max_eval_horizon = max(config.eval_horizons)
    train_transitions = collect_transitions(
        traj,
        n_transitions=config.n_transitions,
        stride=config.stride,
        horizon=config.horizon,
        seed=config.seed,
    )
    eval_transitions = collect_transitions(
        traj,
        n_transitions=config.eval_n_transitions,
        stride=config.eval_stride,
        horizon=max_eval_horizon,
        seed=config.seed + 1000,
    )
    model = Cycle0LatentDynamics(config, n_atoms=traj.n_atoms).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    epoch_losses: list[float] = []
    prior_values: list[float] = []
    transition_values: list[float] = []
    nan_detected = False
    start = time.time()
    for epoch in range(config.num_epochs):
        model.train()
        order = np.random.permutation(len(train_transitions))
        epoch_total: list[float] = []
        for start_idx in range(0, len(order), config.batch_size):
            batch = [train_transitions[int(index)] for index in order[start_idx : start_idx + config.batch_size]]
            optimizer.zero_grad()
            total, transition_loss, prior_loss, _z_batch = train_batch(model, batch, config, device)
            if not torch.isfinite(total):
                nan_detected = True
                raise FloatingPointError(f"Non-finite loss in {config.run_name} at epoch {epoch + 1}")
            total.backward()
            optimizer.step()
            epoch_total.append(float(total.detach().cpu()))
            prior_values.append(float(prior_loss.detach().cpu()))
            transition_values.append(float(transition_loss.detach().cpu()))
        epoch_loss = float(np.mean(epoch_total)) if epoch_total else float("nan")
        epoch_losses.append(epoch_loss)
        print(
            f"[cycle0|{config.run_name}] epoch {epoch + 1}/{config.num_epochs} "
            f"loss={epoch_loss:.6f}",
            flush=True,
        )

    rollout_errors = evaluate_rollout(
        model,
        traj,
        eval_transitions,
        config.eval_horizons,
        device,
        config.latent_dim,
    )
    z_diag = collect_latents(model, train_transitions[: min(64, len(train_transitions))], device)
    diagnostics = latent_diagnostics(z_diag, seed=config.seed)
    diagnostics.update(
        {
            "rollout_errors": rollout_errors,
            "graph_stationarity": {
                "available": False,
                "reason": "Cycle 0 smoke does not estimate temporal graph-stationarity statistics.",
            },
            "final_train_loss": float(epoch_losses[-1]) if epoch_losses else float("nan"),
            "prior_loss_mean": float(np.mean(prior_values)) if prior_values else float("nan"),
            "transition_loss_mean": float(np.mean(transition_values)) if transition_values else float("nan"),
            "nan_detected": bool(nan_detected),
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": config.run_name,
        "status": "ok",
        "failure_flag": False,
        "config": asdict(config),
        "diagnostics": diagnostics,
        "prior_implementation": {
            "graph_prior_form": "nodewise_trace_HtLH" if config.prior in GRAPH_PRIORS else None,
            "graph_prior_nodewise": config.prior in GRAPH_PRIORS,
            "uses_latent_projected_laplacian": False,
            "uses_old_latent_projected_laplacian": False,
            "node_wise_laplacian_formula": (
                "Tr(H^T L H) = sum_(i,j in E) w_ij ||h_i - h_j||^2"
                if config.prior in GRAPH_PRIORS
                else None
            ),
            "graph_prior_requires_node_states": config.prior in GRAPH_PRIORS,
            "encoder_emits_node_states": model.emits_node_states,
            "mlp_graph_priors_disabled": True,
        },
        "timing": {"wall_time_sec": time.time() - start},
    }


def failed_result(config: Cycle0Config, exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": config.run_name,
        "status": "failed",
        "failure_flag": True,
        "config": asdict(config),
        "diagnostics": {
            "rollout_errors": {str(h): float("nan") for h in config.eval_horizons},
            "effective_rank": float("nan"),
            "covariance_condition_number": float("nan"),
            "projection_gaussianity_statistic": float("nan"),
            "graph_stationarity": {"available": False, "reason": "run failed"},
            "final_train_loss": float("nan"),
            "prior_loss_mean": float("nan"),
            "nan_detected": True,
        },
        "prior_implementation": {
            "graph_prior_form": "nodewise_trace_HtLH" if config.prior in GRAPH_PRIORS else None,
            "graph_prior_nodewise": config.prior in GRAPH_PRIORS,
            "uses_latent_projected_laplacian": False,
            "uses_old_latent_projected_laplacian": False,
            "graph_prior_requires_node_states": config.prior in GRAPH_PRIORS,
            "encoder_emits_node_states": config.encoder == "gnn_node",
            "mlp_graph_priors_disabled": True,
        },
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def summarize_results(payload: dict[str, Any]) -> dict[str, Any]:
    runs = payload.get("runs", {})
    required_horizons = {"1", "2", "4", "8", "16", "32"}
    failures = [name for name, run in runs.items() if run.get("failure_flag") or run.get("status") != "ok"]
    nan_runs = []
    missing_diag = []
    for name, run in runs.items():
        diagnostics = run.get("diagnostics", {})
        rollout = set(diagnostics.get("rollout_errors", {}).keys())
        required_keys = {
            "effective_rank",
            "covariance_condition_number",
            "projection_gaussianity_statistic",
            "graph_stationarity",
            "final_train_loss",
            "prior_loss_mean",
        }
        if not required_horizons.issubset(rollout) or not required_keys.issubset(diagnostics.keys()):
            missing_diag.append(name)
        for value in flatten_numeric(diagnostics):
            if isinstance(value, float) and math.isnan(value):
                nan_runs.append(name)
                break
    return {
        "n_runs": len(runs),
        "n_failures": len(failures),
        "failures": failures,
        "runs_with_nan_diagnostics": sorted(set(nan_runs)),
        "runs_missing_diagnostics": missing_diag,
        "schema_version": payload.get("schema_version"),
    }


def flatten_numeric(obj: Any) -> list[float]:
    values: list[float] = []
    if isinstance(obj, dict):
        for value in obj.values():
            values.extend(flatten_numeric(value))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(flatten_numeric(value))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        values.append(float(obj))
    return values


def run_configs(config_paths: list[Path], output_path: Path, schema_version: str = SCHEMA_VERSION) -> dict[str, Any]:
    payload = load_results(output_path)
    payload["schema_version"] = schema_version
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    for config_path in config_paths:
        config = load_config(config_path)
        print(f"\n=== Cycle 0 smoke run: {config.run_name} ===", flush=True)
        try:
            result = train_one(config)
        except Exception as exc:
            result = failed_result(config, exc)
            print(f"FAILED {config.run_name}: {exc}", flush=True)
            print(result["traceback"], flush=True)
        payload["runs"][config.run_name] = result
        payload["summary"] = summarize_results(payload)
        save_json_atomic(payload, output_path)
        print(f"saved {output_path}", flush=True)
    payload["summary"] = summarize_results(payload)
    save_json_atomic(payload, output_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cycle 0 rMD17 aspirin smoke experiments.")
    parser.add_argument("--config", action="append", type=Path, help="Config JSON path. May be repeated.")
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--schema-version", default=SCHEMA_VERSION)
    parser.add_argument("--all", action="store_true", help="Run all configs in --config-dir.")
    args = parser.parse_args()

    if args.all:
        config_paths = sorted(args.config_dir.glob("*.json"))
    else:
        config_paths = args.config or []
    if not config_paths:
        raise SystemExit("No configs selected. Use --all or --config PATH.")
    run_configs(config_paths, args.output, schema_version=args.schema_version)


if __name__ == "__main__":
    main()
