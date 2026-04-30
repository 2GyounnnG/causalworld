import os
import json
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

from scripts.rmd17_loader import (
    RMD17Trajectory,
    collect_rmd17_transitions,
    MOLECULES,
)
from model import (
    WorldModel,
    build_causal_laplacian,
    euclidean_cov_penalty,
    variance_only_penalty,
    sigreg_gauss_penalty,
    permuted_laplacian,
    random_laplacian,
    identity_quadratic_penalty,
)


def build_molecular_laplacian(obs: HeteroData, latent_dim: int) -> torch.Tensor:
    """
    Build a (latent_dim, latent_dim) Laplacian from the atom-bond graph
    of the current frame. Reuses the same projection strategy as
    build_causal_laplacian in model.py.
    """
    edge_index = obs["atom", "bonded", "atom"].edge_index
    edge_attr = obs["atom", "bonded", "atom"].edge_attr
    n_atoms = obs["atom"].x.shape[0]

    graph = nx.Graph()
    graph.add_nodes_from(range(n_atoms))
    ei = edge_index.cpu().numpy()
    ea = edge_attr.cpu().numpy().flatten()
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        if i < j:
            weight = 1.0 / max(float(ea[k]), 1e-6)
            graph.add_edge(i, j, weight=weight)

    graph_di = graph.to_directed()
    return build_causal_laplacian(graph_di, latent_dim)


def molecular_graph_shape(obs: HeteroData) -> tuple[int, int]:
    edge_index = obs["atom", "bonded", "atom"].edge_index
    n_atoms = obs["atom"].x.shape[0]
    n_bonds = 0
    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        if i < j:
            n_bonds += 1
    return int(n_atoms), n_bonds


def build_graph_source_laplacian(
    obs: HeteroData,
    latent_dim: int,
    graph_source: str,
    seed: int,
    frame_idx: int,
    device: torch.device,
) -> torch.Tensor:
    if graph_source == "bond":
        return build_molecular_laplacian(obs, latent_dim).to(device)
    if graph_source == "identity":
        return torch.eye(latent_dim).to(device)

    n_atoms, n_bonds = molecular_graph_shape(obs)
    if graph_source == "random":
        rng_seed = seed * 100000 + int(frame_idx)
        graph = nx.gnm_random_graph(n_atoms, n_bonds, seed=rng_seed)
    elif graph_source == "complete":
        graph = nx.complete_graph(n_atoms)
    else:
        raise ValueError("graph_source must be one of 'bond', 'random', 'complete', or 'identity'")

    return build_causal_laplacian(graph.to_directed(), latent_dim).to(device)


def adapt_obs_for_world_model(obs: HeteroData) -> HeteroData:
    """Map atom/bonded graphs into the node/hyperedge schema expected by WorldModel."""
    data = HeteroData()
    data["atom"].pos = obs["atom"].pos
    atomic_numbers = obs["atom"].atomic_number.view(-1, 1).to(dtype=torch.float32)
    data["node"].x = atomic_numbers
    data["node"].node_id = torch.arange(atomic_numbers.shape[0], dtype=torch.long)

    edge_index = obs["atom", "bonded", "atom"].edge_index
    edge_attr = obs["atom", "bonded", "atom"].edge_attr
    undirected_pairs: List[tuple[int, int]] = []
    distances: List[float] = []
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
        undirected_pairs.append(pair)
        distances.append(float(edge_attr[k].view(-1)[0].item()))

    if undirected_pairs:
        hyperedge_x = torch.tensor(
            [[2.0, -1.0] for _ in undirected_pairs],
            dtype=torch.float32,
        )
        incidence_pairs = []
        for bond_index, (i, j) in enumerate(undirected_pairs):
            incidence_pairs.append((i, bond_index))
            incidence_pairs.append((j, bond_index))
        incidence_index = torch.tensor(incidence_pairs, dtype=torch.long).t().contiguous()
    else:
        hyperedge_x = torch.empty((0, 2), dtype=torch.float32)
        incidence_index = torch.empty((2, 0), dtype=torch.long)

    data["hyperedge"].x = hyperedge_x
    data["hyperedge"].distance = torch.tensor(distances, dtype=torch.float32) if distances else torch.empty((0,), dtype=torch.float32)
    data["hyperedge"].edge_tuple = undirected_pairs
    data["node", "member_of", "hyperedge"].edge_index = incidence_index
    data["hyperedge", "has_member", "node"].edge_index = incidence_index.flip(0)
    return data


def move_obs_to_device(obs: HeteroData, device: torch.device) -> HeteroData:
    return adapt_obs_for_world_model(obs).to(device)


@dataclass
class Config:
    molecule: str = "aspirin"
    encoder: str = "flat"
    prior: str = "spectral"
    graph_source: str = "bond"
    prior_weight: float = 0.1
    latent_dim: int = 16
    hidden_dim: int = 32
    mlp_hidden_dim: int = 128
    n_transitions: int = 2000
    stride: int = 10
    horizon: int = 1
    eval_horizons: tuple = (1, 2, 4, 8, 16)
    num_epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 0
    device: str = "cuda"
    transition_hidden_dim: int = 128
    laplacian_mode: str = "per_frame"
    save_checkpoint: bool = False
    checkpoint_dir: str = "checkpoints/rmd17"
    save_frame_indices: bool = True
    disjoint_eval: bool = False
    eval_n_transitions: int = 200
    variance_gamma: float = 1.0
    sigreg_num_slices: int = 8
    sigreg_sigma: float = 1.0
    control_edge_density: float = 0.3
    control_seed_offset: int = 1000


def is_particle_dataset(dataset_id: str) -> bool:
    return dataset_id.startswith(("lj_", "ho_", "3bpa"))


def is_iso17_dataset(dataset_id: str) -> bool:
    return dataset_id.startswith("iso17")


def get_dataset_trajectory(dataset_id: str):
    if is_particle_dataset(dataset_id):
        from scripts.particle_loader import ParticleTrajectory

        return ParticleTrajectory(dataset_id)
    if is_iso17_dataset(dataset_id):
        from scripts.iso17_loader import ISO17Trajectory

        return ISO17Trajectory(dataset_id)
    return RMD17Trajectory(dataset_id)


def collect_dataset_transitions(dataset_id: str, **kwargs) -> List[Dict]:
    if is_particle_dataset(dataset_id):
        from scripts.particle_loader import collect_particle_transitions

        return collect_particle_transitions(dataset_id=dataset_id, **kwargs)
    if is_iso17_dataset(dataset_id):
        from scripts.iso17_loader import collect_iso17_transitions

        return collect_iso17_transitions(dataset_id=dataset_id, **kwargs)
    return collect_rmd17_transitions(molecule=dataset_id, **kwargs)


def infer_model_n_atoms(config: Config, transitions: list[dict] | None = None) -> int | None:
    if config.encoder != "mlp":
        return None
    if transitions:
        return int(transitions[0]["obs"]["atom"].pos.shape[0])
    return int(get_dataset_trajectory(config.molecule).n_atoms)


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def checkpoint_path(config: Config) -> Path:
    prior_label = config.prior
    if config.prior == "spectral" and config.graph_source != "bond":
        prior_label = f"{config.prior}_{config.graph_source}"
    filename = (
        f"{config.molecule}_{config.encoder}_{prior_label}_"
        f"seed{config.seed}_w{config.prior_weight}_mode{config.laplacian_mode}.pt"
    )
    checkpoint_dir = Path(config.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = ROOT / checkpoint_dir
    return checkpoint_dir / filename


def transition_used_frames(transitions: list[dict], horizons: tuple[int, ...] | list[int]) -> list[int]:
    used: set[int] = set()
    for transition in transitions:
        if "frame_idx" not in transition:
            continue
        frame_idx = int(transition["frame_idx"])
        used.add(frame_idx)
        for horizon in horizons:
            used.add(frame_idx + int(horizon))
    return sorted(used)


def collect_disjoint_eval_transitions(
    *,
    molecule: str,
    n_transitions: int,
    stride: int,
    eval_horizons: tuple[int, ...] | list[int],
    seed: int,
    forbidden_frame_idx: set[int],
    cutoff: float = 5.0,
) -> list[dict]:
    """Sample eval transitions whose start/target frames avoid training frames."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    max_horizon = max(int(horizon) for horizon in eval_horizons)
    traj = get_dataset_trajectory(molecule)
    rng = np.random.default_rng(seed)
    max_start = traj.n_frames - max_horizon - 1
    candidates = np.arange(0, max_start, stride)
    valid_candidates = []
    for frame_idx in candidates:
        used = {int(frame_idx)}
        used.update(int(frame_idx) + int(horizon) for horizon in eval_horizons)
        if used.isdisjoint(forbidden_frame_idx):
            valid_candidates.append(int(frame_idx))

    if n_transitions > len(valid_candidates):
        raise ValueError(
            f"Requested {n_transitions} disjoint eval transitions but only "
            f"{len(valid_candidates)} candidates remain at stride={stride}"
        )

    chosen = rng.choice(np.asarray(valid_candidates, dtype=int), size=n_transitions, replace=False)
    chosen.sort()

    transitions = []
    for frame_idx in chosen:
        obs_t, obs_next, energy, _force = traj.get_pair(int(frame_idx), max_horizon)
        transitions.append(
            {
                "obs": obs_t,
                "next_obs": obs_next,
                "horizon": max_horizon,
                "frame_idx": int(frame_idx),
                "energy": energy,
                "molecule": molecule,
            }
        )
    return transitions


def frame_index_metadata(
    transitions: list[dict],
    eval_transitions: list[dict],
    *,
    train_horizon: int,
    eval_horizons: tuple[int, ...] | list[int],
) -> dict:
    train_frame_idx = [int(transition["frame_idx"]) for transition in transitions if "frame_idx" in transition]
    eval_frame_idx = [int(transition["frame_idx"]) for transition in eval_transitions if "frame_idx" in transition]
    train_used_frame_idx = transition_used_frames(transitions, [train_horizon])
    eval_used_frame_idx = transition_used_frames(eval_transitions, eval_horizons)
    start_overlap = sorted(set(train_frame_idx).intersection(eval_frame_idx))
    used_overlap = sorted(set(train_used_frame_idx).intersection(eval_used_frame_idx))
    return {
        "train_frame_idx": train_frame_idx,
        "eval_frame_idx": eval_frame_idx,
        "train_used_frame_idx": train_used_frame_idx,
        "eval_used_frame_idx": eval_used_frame_idx,
        "train_eval_start_overlap_count": len(start_overlap),
        "train_eval_start_overlap_examples": start_overlap[:20],
        "train_eval_overlap_count": len(used_overlap),
        "train_eval_overlap_examples": used_overlap[:20],
    }


def save_checkpoint(config: Config, model: WorldModel, final_loss: float, rollout_errors: dict) -> Path:
    path = checkpoint_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "config": dict(config.__dict__),
            "final_loss": float(final_loss),
            "rollout_errors": rollout_errors,
            "git_commit": get_git_commit(),
        },
        path,
    )
    return path


def load_checkpoint_model(path: str | Path, device: torch.device | str | None = None) -> tuple[Config, WorldModel, dict]:
    checkpoint = torch.load(path, map_location=device or "cpu")
    config = Config(**checkpoint["config"])
    target_device = torch.device(device or config.device)
    model = WorldModel(
        encoder=config.encoder,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        action_dim=1,
        transition_hidden_dim=config.transition_hidden_dim,
        mlp_hidden_dim=config.mlp_hidden_dim,
        n_atoms=infer_model_n_atoms(config),
    ).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return config, model, checkpoint


def evaluate_rollout(model, eval_transitions, horizons, device, latent_dim) -> Dict:
    traj_cache = {}
    errors_by_h = {H: [] for H in horizons}
    zero_actions = [0.0] * max(horizons)
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for transition in eval_transitions:
            molecule = transition["molecule"]
            if molecule not in traj_cache:
                traj_cache[molecule] = get_dataset_trajectory(molecule)
            traj = traj_cache[molecule]
            idx = transition["frame_idx"]

            if idx + max(horizons) >= traj.n_frames:
                continue

            obs_0 = move_obs_to_device(traj[idx], device)
            z0 = model.encode(obs_0)

            for horizon in horizons:
                z_rollout = model.rollout_latent(z0, zero_actions[:horizon])
                z_pred_h = z_rollout[-1]
                obs_h = move_obs_to_device(traj[idx + horizon], device)
                z_true = model.encode(obs_h)
                err = torch.norm(z_pred_h - z_true, p=2) / (latent_dim ** 0.5)
                errors_by_h[horizon].append(float(err.detach().cpu()))

    if was_training:
        model.train()
    return {H: float(np.mean(v)) if v else float("nan") for H, v in errors_by_h.items()}


def quick_rollout_eval(model, samples, horizon, device, latent_dim) -> float:
    was_training = model.training
    model.eval()
    errs = []
    traj_cache = {}

    with torch.no_grad():
        for transition in samples:
            molecule = transition["molecule"]
            if molecule not in traj_cache:
                traj_cache[molecule] = get_dataset_trajectory(molecule)
            traj = traj_cache[molecule]
            idx = transition["frame_idx"]
            if idx + horizon >= traj.n_frames:
                continue

            obs_0 = move_obs_to_device(traj[idx], device)
            z0 = model.encode(obs_0)
            z_rollout = model.rollout_latent(z0, [0.0] * horizon)
            z_pred = z_rollout[-1]
            z_true = model.encode(move_obs_to_device(traj[idx + horizon], device))
            errs.append(float((torch.norm(z_pred - z_true, p=2) / (latent_dim ** 0.5)).detach().cpu()))

    if was_training:
        model.train()
    return float(np.mean(errs)) if errs else float("nan")


def train_one_seed(config: Config) -> Dict:
    """
    Returns a dict with:
      "rollout_errors": {H: float}  (per eval horizon)
      "final_loss": float
      "config": asdict
      "wall_time_sec": float
    """
    t0 = time.time()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    control_generator = torch.Generator(device="cpu")
    control_generator.manual_seed(config.seed + config.control_seed_offset)

    transitions = collect_dataset_transitions(
        dataset_id=config.molecule,
        n_transitions=config.n_transitions,
        stride=config.stride,
        horizon=config.horizon,
        seed=config.seed,
    )

    if config.disjoint_eval:
        train_used_frame_idx = set(transition_used_frames(transitions, [config.horizon]))
        eval_transitions = collect_disjoint_eval_transitions(
            molecule=config.molecule,
            n_transitions=config.eval_n_transitions,
            stride=config.stride * 10,
            eval_horizons=config.eval_horizons,
            seed=config.seed + 1000,
            forbidden_frame_idx=train_used_frame_idx,
        )
    else:
        eval_transitions = collect_dataset_transitions(
            dataset_id=config.molecule,
            n_transitions=config.eval_n_transitions,
            stride=config.stride * 10,
            horizon=max(config.eval_horizons),
            seed=config.seed + 1000,
        )

    device = torch.device(config.device)
    if config.laplacian_mode not in {"per_frame", "fixed_frame0", "fixed_mean"}:
        raise ValueError("laplacian_mode must be one of 'per_frame', 'fixed_frame0', or 'fixed_mean'")
    spectral_priors = {"spectral", "permuted_spectral", "random_spectral"}
    if config.prior in spectral_priors and config.graph_source not in {"bond", "random", "complete", "identity"}:
        raise ValueError("graph_source must be one of 'bond', 'random', 'complete', or 'identity'")

    fixed_laplacian = None
    if config.prior in spectral_priors and config.laplacian_mode != "per_frame":
        traj = get_dataset_trajectory(config.molecule)
        if config.laplacian_mode == "fixed_frame0":
            fixed_laplacian = build_graph_source_laplacian(
                traj[0],
                config.latent_dim,
                config.graph_source,
                config.seed,
                0,
                device,
            )
        elif config.laplacian_mode == "fixed_mean":
            n_frames = min(500, traj.n_frames)
            if config.graph_source == "bond":
                frame_laplacians = [
                    build_molecular_laplacian(traj[frame_idx], config.latent_dim)
                    for frame_idx in range(n_frames)
                ]
            else:
                frame_laplacians = [
                    build_graph_source_laplacian(
                        traj[frame_idx],
                        config.latent_dim,
                        config.graph_source,
                        config.seed,
                        frame_idx,
                        device,
                    )
                    for frame_idx in range(n_frames)
                ]
            fixed_laplacian = torch.stack(frame_laplacians, dim=0).mean(dim=0).to(device)

    model = WorldModel(
        encoder=config.encoder,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        action_dim=1,
        transition_hidden_dim=config.transition_hidden_dim,
        mlp_hidden_dim=config.mlp_hidden_dim,
        n_atoms=infer_model_n_atoms(config, transitions),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    epoch_loss = float("nan")
    for epoch in range(config.num_epochs):
        model.train()
        order = np.random.permutation(len(transitions))
        total_values = []
        transition_values = []
        reward_values = []
        prior_values = []

        for batch_start in range(0, len(transitions), config.batch_size):
            batch_idx = order[batch_start : batch_start + config.batch_size]
            optimizer.zero_grad()
            batch_transitions = [transitions[int(index)] for index in batch_idx]
            obs_raw_batch = [transition["obs"] for transition in batch_transitions]
            next_obs_raw_batch = [transition["next_obs"] for transition in batch_transitions]
            adapted_obs_list = [adapt_obs_for_world_model(o) for o in obs_raw_batch]
            adapted_next_obs_list = [adapt_obs_for_world_model(o) for o in next_obs_raw_batch]
            obs = Batch.from_data_list(adapted_obs_list).to(device)
            next_obs = Batch.from_data_list(adapted_next_obs_list).to(device)
            batch_size = len(batch_transitions)
            action_batch = torch.zeros(batch_size, dtype=torch.float32, device=device)
            reward_batch = torch.zeros(batch_size, dtype=torch.float32, device=device)
            done_batch = torch.zeros(batch_size, dtype=torch.float32, device=device)

            if config.prior == "spectral":
                B = len(batch_transitions)
                if fixed_laplacian is None:
                    laplacian = torch.stack(
                        [
                            build_graph_source_laplacian(
                                obs_raw,
                                config.latent_dim,
                                config.graph_source,
                                config.seed,
                                int(transition.get("frame_idx", int(index))),
                                device,
                            )
                            for index, transition, obs_raw in zip(
                                batch_idx, batch_transitions, obs_raw_batch
                            )
                        ],
                        dim=0,
                    )
                else:
                    laplacian = fixed_laplacian.unsqueeze(0).expand(B, -1, -1).contiguous()
                loss_dict = model.loss(
                    observation=obs,
                    action=action_batch,
                    next_observation=next_obs,
                    reward=reward_batch,
                    done=done_batch,
                    prior="spectral",
                    prior_weight=config.prior_weight,
                    laplacian=laplacian,
                )
            elif config.prior == "euclidean":
                loss_dict = model.loss(
                    observation=obs,
                    action=action_batch,
                    next_observation=next_obs,
                    reward=reward_batch,
                    done=done_batch,
                    prior="none",
                    prior_weight=0.0,
                )
                batch_latents = model.encode(obs)
            elif config.prior == "variance":
                loss_dict = model.loss(
                    observation=obs,
                    action=action_batch,
                    next_observation=next_obs,
                    reward=reward_batch,
                    done=done_batch,
                    prior="none",
                    prior_weight=0.0,
                )
                batch_latents = model.encode(obs)
            elif config.prior == "identity_quadratic":
                loss_dict = model.loss(
                    observation=obs,
                    action=action_batch,
                    next_observation=next_obs,
                    reward=reward_batch,
                    done=done_batch,
                    prior="none",
                    prior_weight=0.0,
                )
                batch_latents = model.encode(obs)
            elif config.prior == "sigreg":
                loss_dict = model.loss(
                    observation=obs,
                    action=action_batch,
                    next_observation=next_obs,
                    reward=reward_batch,
                    done=done_batch,
                    prior="none",
                    prior_weight=0.0,
                )
                batch_latents = model.encode(obs)
            elif config.prior == "permuted_spectral":
                base_laplacians = []
                for index, transition, obs_raw in zip(
                    batch_idx, batch_transitions, obs_raw_batch
                ):
                    if fixed_laplacian is None:
                        base = build_graph_source_laplacian(
                            obs_raw,
                            config.latent_dim,
                            config.graph_source,
                            config.seed,
                            int(transition.get("frame_idx", int(index))),
                            device,
                        )
                    else:
                        base = fixed_laplacian
                    base_laplacians.append(base)
                laplacian = torch.stack(
                    [
                        permuted_laplacian(base, generator=control_generator)
                        for base in base_laplacians
                    ],
                    dim=0,
                )
                loss_dict = model.loss(
                    observation=obs,
                    action=action_batch,
                    next_observation=next_obs,
                    reward=reward_batch,
                    done=done_batch,
                    prior="spectral",
                    prior_weight=config.prior_weight,
                    laplacian=laplacian,
                )
            elif config.prior == "random_spectral":
                laplacian = torch.stack(
                    [
                        random_laplacian(
                            config.latent_dim,
                            config.control_edge_density,
                            device=device,
                            dtype=torch.float32,
                            generator=control_generator,
                        )
                        for _ in batch_transitions
                    ],
                    dim=0,
                )
                loss_dict = model.loss(
                    observation=obs,
                    action=action_batch,
                    next_observation=next_obs,
                    reward=reward_batch,
                    done=done_batch,
                    prior="spectral",
                    prior_weight=config.prior_weight,
                    laplacian=laplacian,
                )
            elif config.prior == "none":
                loss_dict = model.loss(
                    observation=obs,
                    action=action_batch,
                    next_observation=next_obs,
                    reward=reward_batch,
                    done=done_batch,
                    prior="none",
                    prior_weight=0.0,
                )
            else:
                raise ValueError(
                    "prior must be one of 'none', 'euclidean', 'spectral', 'variance', "
                    "'sigreg', 'identity_quadratic', 'permuted_spectral', 'random_spectral'"
                )

            total = loss_dict["total"]
            if config.prior == "euclidean":
                prior_loss = euclidean_cov_penalty(batch_latents)
                total = total + config.prior_weight * prior_loss
            elif config.prior == "variance":
                prior_loss = variance_only_penalty(
                    batch_latents,
                    gamma=config.variance_gamma,
                )
                total = total + config.prior_weight * prior_loss
            elif config.prior == "identity_quadratic":
                prior_loss = identity_quadratic_penalty(batch_latents)
                total = total + config.prior_weight * prior_loss
            elif config.prior == "sigreg":
                prior_loss = sigreg_gauss_penalty(
                    batch_latents,
                    num_slices=config.sigreg_num_slices,
                    sigma=config.sigreg_sigma,
                )
                total = total + config.prior_weight * prior_loss
            elif config.prior == "spectral":
                prior_loss = loss_dict["prior"]
            elif config.prior in {"permuted_spectral", "random_spectral"}:
                prior_loss = loss_dict["prior"]
            else:
                prior_loss = total.new_tensor(0.0)

            total.backward()
            optimizer.step()

            total_values.append(float(total.detach().cpu()))
            transition_values.append(float(loss_dict["transition"].detach().cpu()))
            reward_values.append(float(loss_dict["reward"].detach().cpu()))
            prior_values.append(float(prior_loss.detach().cpu()))

        epoch_loss = float(np.mean(total_values)) if total_values else float("nan")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            mid_err = quick_rollout_eval(
                model,
                eval_transitions[:20],
                horizon=8,
                device=device,
                latent_dim=config.latent_dim,
            )
            print(
                f"[{config.molecule}|{config.encoder}|{config.prior}|seed={config.seed}] "
                f"epoch {epoch + 1:3d}/{config.num_epochs} "
                f"loss={epoch_loss:.4f} H8_eval={mid_err:.4f}",
                flush=True,
            )

    rollout_errors = evaluate_rollout(
        model,
        eval_transitions,
        horizons=config.eval_horizons,
        device=device,
        latent_dim=config.latent_dim,
    )
    metadata = (
        frame_index_metadata(
            transitions,
            eval_transitions,
            train_horizon=config.horizon,
            eval_horizons=config.eval_horizons,
        )
        if config.save_frame_indices
        else {}
    )
    config_dict = dict(config.__dict__)
    checkpoint = None
    if config.save_checkpoint:
        checkpoint = save_checkpoint(config, model, float(epoch_loss), rollout_errors)

    result = {
        "rollout_errors": rollout_errors,
        "final_loss": float(epoch_loss),
        "config": config_dict,
        "wall_time_sec": time.time() - t0,
    }
    if metadata:
        result["metadata"] = metadata
    if checkpoint is not None:
        try:
            result["checkpoint_path"] = str(checkpoint.relative_to(ROOT))
        except ValueError:
            result["checkpoint_path"] = str(checkpoint)
    from diagnostics_latent import compute_all_diagnostics

    diagnostic_batch_size = min(config.batch_size, len(transitions))
    with torch.no_grad():
        z_list = []
        for k in range(diagnostic_batch_size):
            obs_k = transitions[k]["obs"]
            obs_k_device = move_obs_to_device(obs_k, device)
            z_list.append(model.encode(obs_k_device))
        z_batch = torch.stack(z_list, dim=0)

    diag_L = None
    if config.prior in {"spectral", "permuted_spectral", "random_spectral"}:
        diag_L = build_graph_source_laplacian(
            transitions[0]["obs"],
            config.latent_dim,
            config.graph_source,
            config.seed,
            0,
            device,
        )

    result["final_diagnostics"] = compute_all_diagnostics(z_batch, L=diag_L)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", default="aspirin", choices=MOLECULES)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick 1-seed 3-epoch test for pipeline verification",
    )
    args = parser.parse_args()

    if args.smoke:
        configs = [
            Config(
                molecule=args.molecule,
                prior="spectral",
                num_epochs=3,
                n_transitions=200,
                seed=0,
            )
        ]
    else:
        configs = []
        for prior in ["none", "euclidean", "spectral"]:
            for seed in [0, 1, 2]:
                configs.append(Config(molecule=args.molecule, prior=prior, seed=seed))

    if not torch.cuda.is_available():
        print("WARNING: CUDA unavailable, falling back to CPU")
        for config in configs:
            config.device = "cpu"
    else:
        print(f"device=cuda ({torch.cuda.get_device_name(0)})")

    all_results = {}
    for config in configs:
        key = f"{config.molecule}|{config.encoder}|{config.prior}|seed={config.seed}"
        print(f"\n=== Starting {key} ===", flush=True)
        result = train_one_seed(config)
        all_results[key] = result

        out_path = ROOT / f"rmd17_{args.molecule}_results.json"
        if args.smoke:
            out_path = ROOT / f"rmd17_{args.molecule}_smoke.json"
        with out_path.open("w", encoding="utf-8") as file:
            json.dump(all_results, file, indent=2, default=str)
        print(f"  -> saved partial results to {out_path.name}")

    print("\n=== Summary ===")
    print(f"molecule: {args.molecule}")
    print(f"{'prior':12s} {'seed':5s} " + " ".join(f"H={H:<4d}" for H in [1, 2, 4, 8, 16]))
    for key, result in all_results.items():
        parts = key.split("|")
        prior = parts[2]
        seed = parts[3].split("=")[1]
        errs = result["rollout_errors"]
        print(
            f"{prior:12s} {seed:5s} "
            + " ".join(f"{errs.get(H, float('nan')):.4f}" for H in [1, 2, 4, 8, 16])
        )


if __name__ == "__main__":
    main()
