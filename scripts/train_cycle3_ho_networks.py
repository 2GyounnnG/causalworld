from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_cycle0 import (
    Cycle0LatentDynamics,
    GRAPH_PRIORS,
    GLOBAL_PRIORS,
    collect_latents,
    collect_transitions,
    evaluate_rollout,
    latent_diagnostics,
    save_json_atomic,
    select_device,
    set_seed,
    train_batch,
)


CONFIG_DIR = ROOT / "experiments" / "configs" / "cycle3_ho_networks"
RESULT_DIR = ROOT / "experiments" / "results" / "cycle3_ho_networks"
DEFAULT_OUTPUT = RESULT_DIR / "cycle3_ho_networks_results.json"
DEFAULT_DATA_ROOT = ROOT / "data" / "ho_raw"
SCHEMA_VERSION = "cycle3_ho_networks_v1"
TOPOLOGIES = ["lattice", "random", "scalefree"]


@dataclass
class Cycle3HOConfig:
    run_name: str
    topology: str = "lattice"
    encoder: str = "gnn_node"
    prior: str = "none"
    seed: int = 0
    num_epochs: int = 30
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
    batch_size: int = 32
    lr: float = 1e-3
    prior_weight: float = 0.1
    sigreg_num_slices: int = 8
    device: str = "auto"
    data_root: str = str(DEFAULT_DATA_ROOT)


class HOTrajectory:
    def __init__(self, topology: str, data_root: Path | None = None):
        root = data_root or DEFAULT_DATA_ROOT
        self.topology = topology
        self.molecule = f"ho_{topology}"
        self.npz_path = root / f"ho_{topology}.npz"
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Missing HO topology file: {self.npz_path}")
        data = np.load(self.npz_path)
        self.coords = data["coords"].astype(np.float32)
        self.atomic_numbers = data["nuclear_charges"].astype(np.int64)
        self.energies = data["energies"].astype(np.float32)
        self.edges = data["edges"].astype(np.int64)
        self.n_frames = int(self.coords.shape[0])
        self.n_atoms = int(self.coords.shape[1])

    def __getitem__(self, idx: int) -> HeteroData:
        coords = torch.from_numpy(self.coords[idx])
        atomic_numbers = torch.from_numpy(self.atomic_numbers)
        atom_x = torch.cat([atomic_numbers.to(dtype=torch.float32).view(-1, 1), coords], dim=-1)

        edge_pairs = self.edges
        directed = np.concatenate([edge_pairs, edge_pairs[:, ::-1]], axis=0)
        edge_index = torch.from_numpy(directed.T.copy()).to(dtype=torch.long)
        edge_attr = torch.ones((edge_index.shape[1], 1), dtype=torch.float32)

        data = HeteroData()
        data["atom"].pos = coords
        data["atom"].atomic_number = atomic_numbers
        data["atom"].x = atom_x
        data["atom", "bonded", "atom"].edge_index = edge_index
        data["atom", "bonded", "atom"].edge_attr = edge_attr
        return data

    def get_pair(self, idx: int, horizon: int = 1):
        if idx + horizon >= self.n_frames:
            raise IndexError(f"idx+horizon={idx + horizon} >= n_frames={self.n_frames}")
        obs_t = self[idx]
        obs_next = self[idx + horizon]
        return obs_t, obs_next, float(self.energies[idx]), None


def load_config(path: Path) -> Cycle3HOConfig:
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if "eval_horizons" in raw:
        raw["eval_horizons"] = tuple(int(value) for value in raw["eval_horizons"])
    return Cycle3HOConfig(**raw)


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


def validate_config(config: Cycle3HOConfig) -> None:
    if config.topology not in TOPOLOGIES:
        raise ValueError(f"Unknown topology {config.topology!r}. Valid: {TOPOLOGIES}")
    if config.encoder != "gnn_node":
        raise ValueError("Cycle 3 HO uses GNN encoder only: encoder must be 'gnn_node'")
    if config.prior not in {*GLOBAL_PRIORS, *GRAPH_PRIORS}:
        raise ValueError(f"Unknown prior {config.prior!r}")
    if config.prior in {"variance", "sigreg"}:
        raise ValueError("Cycle 3 HO only enables none, covariance, graph, permuted_graph, random_graph")


def train_one(config: Cycle3HOConfig) -> dict[str, Any]:
    validate_config(config)
    set_seed(config.seed)
    device = select_device(config.device)
    traj = HOTrajectory(config.topology, data_root=Path(config.data_root))
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
        horizon=max(config.eval_horizons),
        seed=config.seed + 1000,
    )
    model = Cycle0LatentDynamics(config, n_atoms=traj.n_atoms).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    epoch_losses: list[float] = []
    prior_values: list[float] = []
    transition_values: list[float] = []
    nan_detected = False
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
            f"[cycle3-ho|{config.run_name}] epoch {epoch + 1}/{config.num_epochs} "
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
            "graph_edge_weights": "unit",
        },
        "timing": {"wall_time_sec": time.time() - start_time},
    }


def failed_result(config: Cycle3HOConfig, exc: BaseException) -> dict[str, Any]:
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
            "final_train_loss": float("nan"),
            "prior_loss_mean": float("nan"),
            "transition_loss_mean": float("nan"),
            "nan_detected": True,
        },
        "prior_implementation": {
            "graph_prior_form": "nodewise_trace_HtLH" if config.prior in GRAPH_PRIORS else None,
            "graph_prior_nodewise": config.prior in GRAPH_PRIORS,
            "uses_latent_projected_laplacian": False,
            "uses_old_latent_projected_laplacian": False,
            "graph_prior_requires_node_states": config.prior in GRAPH_PRIORS,
            "encoder_emits_node_states": config.encoder == "gnn_node",
            "graph_edge_weights": "unit",
        },
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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
            "final_train_loss",
            "prior_loss_mean",
            "transition_loss_mean",
        }
        if not required_horizons.issubset(rollout) or not required_keys.issubset(diagnostics.keys()):
            missing_diag.append(name)
        for horizon in required_horizons:
            if not finite(diagnostics.get("rollout_errors", {}).get(horizon)):
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


def run_configs(config_paths: list[Path], output_path: Path, schema_version: str = SCHEMA_VERSION) -> dict[str, Any]:
    payload = load_results(output_path)
    payload["schema_version"] = schema_version
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    for config_path in config_paths:
        config = load_config(config_path)
        print(f"\n=== Cycle 3 HO run: {config.run_name} ===", flush=True)
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
    parser = argparse.ArgumentParser(description="Run Cycle 3 controlled HO network experiments.")
    parser.add_argument("--config", action="append", type=Path, help="Config JSON path. May be repeated.")
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--schema-version", default=SCHEMA_VERSION)
    parser.add_argument("--all", action="store_true", help="Run all configs in --config-dir.")
    args = parser.parse_args()

    config_paths = sorted(args.config_dir.glob("*.json")) if args.all else (args.config or [])
    if not config_paths:
        raise SystemExit("No configs selected. Use --all or --config PATH.")
    run_configs(config_paths, args.output, schema_version=args.schema_version)


if __name__ == "__main__":
    main()
