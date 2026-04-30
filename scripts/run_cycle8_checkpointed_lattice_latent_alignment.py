from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_cycle0 import (
    Cycle0LatentDynamics,
    collect_transitions,
    evaluate_rollout,
    latent_diagnostics,
    move_obs,
    save_json_atomic,
    select_device,
    set_seed,
    train_batch,
)
from scripts.train_cycle3_ho_networks import (
    Cycle3HOConfig,
    HOTrajectory,
    config_hash,
    get_git_commit,
    graph_laplacian_metadata,
    laplacian_from_edges,
    load_config,
    random_edge_pairs_np,
)


SCHEMA_VERSION = "cycle8_checkpointed_lattice_latent_alignment_v1"
DEFAULT_CONFIG_DIR = ROOT / "experiments/configs/cycle8_checkpointed_lattice_latent_alignment"
DEFAULT_RESULT = (
    ROOT
    / "experiments/results/cycle8_checkpointed_lattice_latent_alignment/cycle8_checkpointed_lattice_latent_alignment_results.json"
)
DEFAULT_CHECKPOINT_DIR = ROOT / "experiments/checkpoints/cycle8_checkpointed_lattice"
DEFAULT_ARTIFACT_DIR = ROOT / "experiments/artifacts/cycle8_checkpointed_lattice"
TRACE_SEED = 4242


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


def rel_to_root(path: Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        return str(path)
    return str(path.relative_to(ROOT))


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def torch_permutation(n_nodes: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(n_nodes, generator=generator)


def laplacian_tensor(laplacian: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(laplacian, dtype=np.float32))


def build_trace_laplacians(config: Cycle3HOConfig, traj: HOTrajectory, frame_indices: list[int]) -> dict[str, torch.Tensor]:
    n_nodes = int(traj.n_atoms)
    n_edges = int(traj.edges.shape[0])
    true_l = laplacian_from_edges(n_nodes, traj.edges)
    true_l_tensor = laplacian_tensor(true_l)

    prior_laplacians: list[torch.Tensor] = []
    random_laplacians: list[torch.Tensor] = []
    prior_perms: list[torch.Tensor] = []
    control_perms: list[torch.Tensor] = []
    for frame_idx in frame_indices:
        perm = torch_permutation(n_nodes, 2003 + int(config.seed) * 100000 + int(frame_idx))
        random_edges = random_edge_pairs_np(n_nodes, n_edges, seed=3001 + int(config.seed) * 100000 + int(frame_idx))
        random_l = laplacian_tensor(laplacian_from_edges(n_nodes, random_edges))
        random_laplacians.append(random_l)
        control_perms.append(perm)
        if config.prior == "random_graph":
            prior_laplacians.append(random_l)
            prior_perms.append(torch.arange(n_nodes, dtype=torch.long))
        elif config.prior == "permuted_graph":
            prior_laplacians.append(true_l_tensor)
            prior_perms.append(perm)
        else:
            prior_laplacians.append(true_l_tensor)
            prior_perms.append(torch.arange(n_nodes, dtype=torch.long))

    return {
        "true_laplacian": true_l_tensor,
        "prior_laplacians": torch.stack(prior_laplacians, dim=0),
        "random_laplacians": torch.stack(random_laplacians, dim=0),
        "prior_permutation_indices": torch.stack(prior_perms, dim=0),
        "control_permutation_indices": torch.stack(control_perms, dim=0),
    }


def collect_latent_trace(
    model: Cycle0LatentDynamics,
    config: Cycle3HOConfig,
    traj: HOTrajectory,
    device: torch.device,
) -> dict[str, Any]:
    transitions = collect_transitions(
        traj,
        n_transitions=config.eval_n_transitions,
        stride=config.eval_stride,
        horizon=1,
        seed=TRACE_SEED,
    )
    model.eval()
    h_t_values: list[torch.Tensor] = []
    h_tp1_values: list[torch.Tensor] = []
    z_t_values: list[torch.Tensor] = []
    z_tp1_values: list[torch.Tensor] = []
    frame_indices: list[int] = []
    with torch.no_grad():
        for transition in transitions:
            obs = move_obs(transition["obs"], device)
            next_obs = move_obs(transition["next_obs"], device)
            z_t, h_t = model.encode(obs)
            z_tp1, h_tp1 = model.encode(next_obs)
            assert h_t is not None and h_tp1 is not None
            h_t_values.append(h_t.detach().cpu())
            h_tp1_values.append(h_tp1.detach().cpu())
            z_t_values.append(z_t.detach().cpu())
            z_tp1_values.append(z_tp1.detach().cpu())
            frame_indices.append(int(transition["frame_idx"]))
    h_t = torch.stack(h_t_values, dim=0)
    h_tp1 = torch.stack(h_tp1_values, dim=0)
    z_t = torch.stack(z_t_values, dim=0)
    z_tp1 = torch.stack(z_tp1_values, dim=0)
    laplacians = build_trace_laplacians(config, traj, frame_indices)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": config.run_name,
        "topology": config.topology,
        "prior": config.prior,
        "prior_weight": float(config.prior_weight),
        "seed": int(config.seed),
        "trace_seed": TRACE_SEED,
        "frame_idx": torch.tensor(frame_indices, dtype=torch.long),
        "H_t": h_t,
        "H_tplus1": h_tp1,
        "Delta_H": h_tp1 - h_t,
        "z_t": z_t,
        "z_tplus1": z_tp1,
        "Delta_z": z_tp1 - z_t,
        **laplacians,
        "laplacian_metadata": graph_laplacian_metadata(config, traj),
    }


def summarize_results(payload: dict[str, Any]) -> dict[str, Any]:
    failures = [
        name
        for name, run in payload.get("runs", {}).items()
        if run.get("status") != "ok" or run.get("failure_flag")
    ]
    missing_artifacts = []
    for name, run in payload.get("runs", {}).items():
        if run.get("status") != "ok":
            continue
        checkpoint = ROOT / str(run.get("checkpoint_path", ""))
        artifact = ROOT / str(run.get("latent_trace_path", ""))
        if not checkpoint.exists() or not artifact.exists():
            missing_artifacts.append(name)
    return {
        "schema_version": payload.get("schema_version"),
        "n_runs": len(payload.get("runs", {})),
        "n_failures": len(failures),
        "failures": sorted(failures),
        "runs_missing_checkpoint_or_latent_artifact": sorted(missing_artifacts),
    }


def result_from_saved_artifacts(config: Cycle3HOConfig, checkpoint_path: Path, artifact_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    diagnostics = checkpoint.get("diagnostics", {})
    graph_metadata = checkpoint.get("graph_metadata")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": config.run_name,
        "status": "ok",
        "failure_flag": False,
        **top_level_result_fields(config, diagnostics, graph_metadata),
        "config": asdict(config),
        "checkpoint_path": rel_to_root(checkpoint_path),
        "latent_artifact_path": rel_to_root(artifact_path),
        "latent_trace_path": rel_to_root(artifact_path),
        "config_hash": checkpoint.get("config_hash", config_hash(config)),
        "git_commit": checkpoint.get("git_commit", get_git_commit()),
        "graph_metadata": graph_metadata,
        "diagnostics": diagnostics,
        "prior_implementation": {
            "graph_prior_form": "nodewise_trace_HtLH",
            "graph_prior_nodewise": True,
            "uses_latent_projected_laplacian": False,
            "uses_old_latent_projected_laplacian": False,
            "node_wise_laplacian_formula": "Tr(H^T L H) = sum_(i,j in E) w_ij ||h_i - h_j||^2",
            "graph_prior_requires_node_states": True,
            "encoder_emits_node_states": True,
            "graph_edge_weights": "unit",
        },
        "timing": checkpoint.get("timing", {}),
        "restored_from_checkpoint_artifact": True,
    }


def top_level_result_fields(
    config: Cycle3HOConfig,
    diagnostics: dict[str, Any],
    graph_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    graph_metadata = graph_metadata or {}
    rollout_errors = diagnostics.get("rollout_errors", {})
    return {
        "topology": config.topology,
        "encoder": config.encoder,
        "prior": config.prior,
        "prior_weight": config.prior_weight,
        "lambda": config.prior_weight,
        "seed": config.seed,
        "training_seed": config.seed,
        "data_generation_seed": graph_metadata.get("data_generation_seed"),
        "graph_seed": graph_metadata.get("graph_seed"),
        "permutation_seed": graph_metadata.get("permutation_seed"),
        "permutation_seed_formula": graph_metadata.get("permutation_seed_formula"),
        "random_graph_seed": graph_metadata.get("random_graph_seed"),
        "random_graph_seed_formula": graph_metadata.get("random_graph_seed_formula"),
        "random_graph_seed_min": graph_metadata.get("random_graph_seed_min"),
        "random_graph_seed_max": graph_metadata.get("random_graph_seed_max"),
        "rollout_errors": rollout_errors,
        "final_train_loss": diagnostics.get("final_train_loss"),
        "final_transition_loss": diagnostics.get("final_transition_loss"),
        "prior_loss_mean": diagnostics.get("prior_loss_mean"),
        "transition_loss_mean": diagnostics.get("transition_loss_mean"),
    }


def train_one(
    config: Cycle3HOConfig,
    *,
    checkpoint_dir: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    if config.topology != "lattice" or config.encoder != "gnn_node" or config.prior not in {"graph", "permuted_graph", "random_graph"}:
        raise ValueError("Cycle 8 is restricted to lattice GNN graph/permuted_graph/random_graph runs.")
    set_seed(config.seed)
    device = select_device(config.device)
    traj = HOTrajectory(config.topology, data_root=Path(config.data_root))
    graph_metadata = graph_laplacian_metadata(config, traj)
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
        print(f"[cycle8|{config.run_name}] epoch {epoch + 1}/{config.num_epochs} loss={epoch_loss:.6f}", flush=True)

    rollout_errors = evaluate_rollout(model, traj, eval_transitions, config.eval_horizons, device, config.latent_dim)
    z_diag = []
    with torch.no_grad():
        model.eval()
        for transition in train_transitions[: min(64, len(train_transitions))]:
            z, _h = model.encode(move_obs(transition["obs"], device))
            z_diag.append(z.detach().cpu().numpy())
    diagnostics = latent_diagnostics(np.stack(z_diag, axis=0), seed=config.seed)
    diagnostics.update(
        {
            "rollout_errors": rollout_errors,
            "final_train_loss": float(epoch_losses[-1]) if epoch_losses else float("nan"),
            "prior_loss_mean": float(np.mean(prior_values)) if prior_values else float("nan"),
            "transition_loss_mean": float(np.mean(transition_values)) if transition_values else float("nan"),
            "final_transition_loss": float(np.mean(transition_values[-max(1, math.ceil(len(transition_values) / max(1, config.num_epochs))) :]))
            if transition_values
            else float("nan"),
            "nan_detected": bool(nan_detected),
        }
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{config.run_name}.pt"
    artifact_path = artifact_dir / f"{config.run_name}_latents.pt"
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "diagnostics": diagnostics,
        "graph_metadata": graph_metadata,
        "config_hash": config_hash(config),
        "git_commit": get_git_commit(),
    }
    torch.save(checkpoint, checkpoint_path)
    torch.save(collect_latent_trace(model, config, traj, device), artifact_path)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": config.run_name,
        "status": "ok",
        "failure_flag": False,
        **top_level_result_fields(config, diagnostics, graph_metadata),
        "config": asdict(config),
        "checkpoint_path": rel_to_root(checkpoint_path),
        "latent_artifact_path": rel_to_root(artifact_path),
        "latent_trace_path": rel_to_root(artifact_path),
        "config_hash": config_hash(config),
        "git_commit": get_git_commit(),
        "graph_metadata": graph_metadata,
        "diagnostics": diagnostics,
        "prior_implementation": {
            "graph_prior_form": "nodewise_trace_HtLH",
            "graph_prior_nodewise": True,
            "uses_latent_projected_laplacian": False,
            "uses_old_latent_projected_laplacian": False,
            "node_wise_laplacian_formula": "Tr(H^T L H) = sum_(i,j in E) w_ij ||h_i - h_j||^2",
            "graph_prior_requires_node_states": True,
            "encoder_emits_node_states": True,
            "graph_edge_weights": "unit",
        },
        "timing": {"wall_time_sec": time.time() - start_time},
    }


def failed_result(config: Cycle3HOConfig, exc: BaseException) -> dict[str, Any]:
    diagnostics = {
        "rollout_errors": {str(h): float("nan") for h in config.eval_horizons},
        "final_train_loss": float("nan"),
        "prior_loss_mean": float("nan"),
        "transition_loss_mean": float("nan"),
        "final_transition_loss": float("nan"),
        "nan_detected": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": config.run_name,
        "status": "failed",
        "failure_flag": True,
        **top_level_result_fields(config, diagnostics, None),
        "config": asdict(config),
        "checkpoint_path": None,
        "latent_artifact_path": None,
        "latent_trace_path": None,
        "config_hash": config_hash(config),
        "git_commit": get_git_commit(),
        "graph_metadata": None,
        "diagnostics": diagnostics,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def run_configs(config_paths: list[Path], output_path: Path, checkpoint_dir: Path, artifact_dir: Path, *, force: bool = False) -> dict[str, Any]:
    payload = load_results(output_path)
    payload["schema_version"] = SCHEMA_VERSION
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    for config_path in config_paths:
        config = load_config(config_path)
        checkpoint_path = checkpoint_dir / f"{config.run_name}.pt"
        artifact_path = artifact_dir / f"{config.run_name}_latents.pt"
        if checkpoint_path.exists() and artifact_path.exists() and not force:
            print(f"RESTORE complete {config.run_name}", flush=True)
            payload["runs"][config.run_name] = result_from_saved_artifacts(config, checkpoint_path, artifact_path)
            payload["summary"] = summarize_results(payload)
            save_json_atomic(payload, output_path)
            continue
        existing = payload.get("runs", {}).get(config.run_name)
        if existing and existing.get("status") == "ok" and not force:
            existing_checkpoint_path = ROOT / str(existing.get("checkpoint_path", ""))
            existing_artifact_path = ROOT / str(existing.get("latent_trace_path", ""))
            if existing_checkpoint_path.exists() and existing_artifact_path.exists():
                print(f"SKIP complete {config.run_name}", flush=True)
                continue
        print(f"\n=== Cycle 8 run: {config.run_name} ===", flush=True)
        try:
            result = train_one(config, checkpoint_dir=checkpoint_dir, artifact_dir=artifact_dir)
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
    parser = argparse.ArgumentParser(description="Run Cycle 8 checkpointed HO lattice latent-alignment experiment.")
    parser.add_argument("--config", action="append", type=Path)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_paths = sorted(args.config_dir.glob("*.json")) if args.all else (args.config or [])
    if not config_paths:
        raise SystemExit("No configs selected. Use --all or --config PATH.")
    run_configs(
        config_paths,
        args.output,
        args.checkpoint_dir,
        args.artifact_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
