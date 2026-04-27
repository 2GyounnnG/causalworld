from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from model import WorldModel, euclidean_cov_penalty
from scripts.iso17_loader import (
    DATA_ROOT,
    ISO17Trajectory,
    SPLITS,
    collect_iso17_transitions,
)
from scripts.train_rmd17 import (
    build_graph_source_laplacian,
    frame_index_metadata,
    get_git_commit,
    move_obs_to_device,
    transition_used_frames,
)


@dataclass
class Config:
    split: str = "reference"
    eval_split: str = "reference"
    eval_splits: tuple = ()
    isomer: str = "all"
    eval_isomer: str = "all"
    encoder: str = "flat"
    prior: str = "spectral"
    graph_source: str = "bond"
    prior_weight: float = 0.1
    latent_dim: int = 16
    hidden_dim: int = 32
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
    checkpoint_dir: str = "checkpoints/iso17"
    save_frame_indices: bool = True
    disjoint_eval: bool = False
    eval_n_transitions: int = 200
    data_root: str = ""


def _data_root(config: Config) -> Path | None:
    return Path(config.data_root) if config.data_root else None


def _dataset_label(split: str, isomer: str) -> str:
    return f"{split}_{isomer}".replace("/", "-")


def _eval_label(config: Config) -> str:
    if config.eval_splits:
        return "+".join(str(split) for split in config.eval_splits)
    return config.eval_split


def checkpoint_path(config: Config) -> Path:
    prior_label = config.prior
    if config.prior == "spectral" and config.graph_source != "bond":
        prior_label = f"{config.prior}_{config.graph_source}"
    filename = (
        f"iso17_{_dataset_label(config.split, config.isomer)}_"
        f"eval-{_dataset_label(_eval_label(config), config.eval_isomer)}_"
        f"{config.encoder}_{prior_label}_seed{config.seed}_"
        f"w{config.prior_weight}_mode{config.laplacian_mode}.pt"
    )
    checkpoint_dir = Path(config.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = ROOT / checkpoint_dir
    return checkpoint_dir / filename


def collect_disjoint_eval_transitions(
    *,
    split: str,
    isomer: str,
    n_transitions: int,
    stride: int,
    eval_horizons: tuple[int, ...] | list[int],
    seed: int,
    forbidden_frame_idx: set[int],
    cutoff: float = 5.0,
    data_root: Path | None = None,
) -> list[dict]:
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    max_horizon = max(int(horizon) for horizon in eval_horizons)
    traj = ISO17Trajectory(split=split, isomer=isomer, cutoff=cutoff, data_root=data_root)
    rng = np.random.default_rng(seed)
    max_start = traj.n_frames - max_horizon - 1
    valid_candidates = []
    for frame_idx in range(0, max_start, stride):
        used = {int(frame_idx)}
        used.update(int(frame_idx) + int(horizon) for horizon in eval_horizons)
        if used.isdisjoint(forbidden_frame_idx) and traj.can_pair(int(frame_idx), max_horizon):
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
                "dataset": "iso17",
                "molecule": "iso17",
                "split": split,
                "isomer": traj.isomer,
            }
        )
    return transitions


def collect_eval_transitions_for_config(
    config: Config,
    *,
    eval_split: str,
    data_root: Path | None,
    train_used_frame_idx: set[int],
) -> list[dict]:
    same_eval_source = config.split == eval_split and config.isomer == config.eval_isomer
    if config.disjoint_eval and same_eval_source:
        return collect_disjoint_eval_transitions(
            split=eval_split,
            isomer=config.eval_isomer,
            n_transitions=config.eval_n_transitions,
            stride=config.stride * 10,
            eval_horizons=config.eval_horizons,
            seed=config.seed + 1000,
            forbidden_frame_idx=train_used_frame_idx,
            data_root=data_root,
        )

    return collect_iso17_transitions(
        split=eval_split,
        isomer=config.eval_isomer,
        n_transitions=config.eval_n_transitions,
        stride=config.stride * 10,
        horizon=max(config.eval_horizons),
        seed=config.seed + 1000,
        data_root=data_root,
    )


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


def evaluate_rollout(
    model: WorldModel,
    eval_transitions: list[dict],
    horizons,
    device: torch.device,
    latent_dim: int,
    data_root: Path | None,
) -> Dict:
    traj_cache = {}
    errors_by_h = {H: [] for H in horizons}
    zero_actions = [0.0] * max(horizons)
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for transition in eval_transitions:
            split = transition.get("split", "reference")
            isomer = transition.get("isomer", "all")
            cache_key = (split, isomer)
            if cache_key not in traj_cache:
                traj_cache[cache_key] = ISO17Trajectory(split=split, isomer=isomer, data_root=data_root)
            traj = traj_cache[cache_key]
            idx = int(transition["frame_idx"])

            if idx + max(horizons) >= traj.n_frames:
                continue

            obs_0 = move_obs_to_device(traj[idx], device)
            z0 = model.encode(obs_0)

            for horizon in horizons:
                if not traj.can_pair(idx, int(horizon)):
                    continue
                z_rollout = model.rollout_latent(z0, zero_actions[:horizon])
                z_pred_h = z_rollout[-1]
                obs_h = move_obs_to_device(traj[idx + horizon], device)
                z_true = model.encode(obs_h)
                err = torch.norm(z_pred_h - z_true, p=2) / (latent_dim ** 0.5)
                errors_by_h[horizon].append(float(err.detach().cpu()))

    if was_training:
        model.train()
    return {H: float(np.mean(v)) if v else float("nan") for H, v in errors_by_h.items()}


def quick_rollout_eval(
    model: WorldModel,
    samples: list[dict],
    horizon: int,
    device: torch.device,
    latent_dim: int,
    data_root: Path | None,
) -> float:
    was_training = model.training
    model.eval()
    errs = []
    traj_cache = {}

    with torch.no_grad():
        for transition in samples:
            split = transition.get("split", "reference")
            isomer = transition.get("isomer", "all")
            cache_key = (split, isomer)
            if cache_key not in traj_cache:
                traj_cache[cache_key] = ISO17Trajectory(split=split, isomer=isomer, data_root=data_root)
            traj = traj_cache[cache_key]
            idx = int(transition["frame_idx"])
            if idx + horizon >= traj.n_frames or not traj.can_pair(idx, horizon):
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
    t0 = time.time()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    data_root = _data_root(config)
    transitions = collect_iso17_transitions(
        split=config.split,
        isomer=config.isomer,
        n_transitions=config.n_transitions,
        stride=config.stride,
        horizon=config.horizon,
        seed=config.seed,
        data_root=data_root,
    )

    eval_splits = tuple(config.eval_splits) if config.eval_splits else (config.eval_split,)
    train_used_frame_idx = set(transition_used_frames(transitions, [config.horizon]))
    eval_transition_sets = {
        eval_split: collect_eval_transitions_for_config(
            config,
            eval_split=eval_split,
            data_root=data_root,
            train_used_frame_idx=train_used_frame_idx,
        )
        for eval_split in eval_splits
    }
    progress_eval_split = eval_splits[0]
    progress_eval_transitions = eval_transition_sets[progress_eval_split]

    device = torch.device(config.device)
    if config.laplacian_mode not in {"per_frame", "fixed_frame0", "fixed_mean"}:
        raise ValueError("laplacian_mode must be one of 'per_frame', 'fixed_frame0', or 'fixed_mean'")
    if config.prior == "spectral" and config.graph_source not in {"bond", "random", "complete", "identity"}:
        raise ValueError("graph_source must be one of 'bond', 'random', 'complete', or 'identity'")

    fixed_laplacian = None
    if config.prior == "spectral" and config.laplacian_mode != "per_frame":
        traj = ISO17Trajectory(config.split, isomer=config.isomer, data_root=data_root)
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
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    action_zero = torch.zeros(1, dtype=torch.float32, device=device)

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
            base_totals = []
            transition_losses = []
            reward_losses = []
            prior_losses = []
            batch_latents = []

            for index in batch_idx:
                transition = transitions[int(index)]
                obs_raw = transition["obs"]
                next_obs_raw = transition["next_obs"]
                obs = move_obs_to_device(obs_raw, device)
                next_obs = move_obs_to_device(next_obs_raw, device)

                if config.prior == "spectral":
                    if fixed_laplacian is None:
                        frame_idx = int(transition.get("frame_idx", int(index)))
                        laplacian = build_graph_source_laplacian(
                            obs_raw,
                            config.latent_dim,
                            config.graph_source,
                            config.seed,
                            frame_idx,
                            device,
                        )
                    else:
                        laplacian = fixed_laplacian
                    loss_dict = model.loss(
                        observation=obs,
                        action=action_zero,
                        next_observation=next_obs,
                        reward=0.0,
                        done=0.0,
                        prior="spectral",
                        prior_weight=config.prior_weight,
                        laplacian=laplacian,
                    )
                elif config.prior == "euclidean":
                    loss_dict = model.loss(
                        observation=obs,
                        action=action_zero,
                        next_observation=next_obs,
                        reward=0.0,
                        done=0.0,
                        prior="none",
                        prior_weight=0.0,
                    )
                    batch_latents.append(model.encode(obs))
                elif config.prior == "none":
                    loss_dict = model.loss(
                        observation=obs,
                        action=action_zero,
                        next_observation=next_obs,
                        reward=0.0,
                        done=0.0,
                        prior="none",
                        prior_weight=0.0,
                    )
                else:
                    raise ValueError("prior must be one of 'none', 'euclidean', or 'spectral'")

                base_totals.append(loss_dict["total"])
                transition_losses.append(loss_dict["transition"])
                reward_losses.append(loss_dict["reward"])
                prior_losses.append(loss_dict["prior"])

            total = torch.stack(base_totals).mean()
            if config.prior == "euclidean":
                prior_loss = euclidean_cov_penalty(torch.stack(batch_latents, dim=0))
                total = total + config.prior_weight * prior_loss
            elif config.prior == "spectral":
                prior_loss = torch.stack(prior_losses).mean()
            else:
                prior_loss = total.new_tensor(0.0)

            total.backward()
            optimizer.step()

            total_values.append(float(total.detach().cpu()))
            transition_values.append(float(torch.stack(transition_losses).mean().detach().cpu()))
            reward_values.append(float(torch.stack(reward_losses).mean().detach().cpu()))
            prior_values.append(float(prior_loss.detach().cpu()))

        epoch_loss = float(np.mean(total_values)) if total_values else float("nan")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            mid_err = quick_rollout_eval(
                model,
                progress_eval_transitions[:20],
                horizon=8,
                device=device,
                latent_dim=config.latent_dim,
                data_root=data_root,
            )
            print(
                f"[iso17:{config.split}->{progress_eval_split}|{config.encoder}|"
                f"{config.prior}|seed={config.seed}] "
                f"epoch {epoch + 1:3d}/{config.num_epochs} "
                f"loss={epoch_loss:.4f} H8_eval={mid_err:.4f}",
                flush=True,
            )

    rollout_error_sets = {
        eval_split: evaluate_rollout(
            model,
            eval_transitions,
            horizons=config.eval_horizons,
            device=device,
            latent_dim=config.latent_dim,
            data_root=data_root,
        )
        for eval_split, eval_transitions in eval_transition_sets.items()
    }
    metadata_sets = {}
    if config.save_frame_indices:
        metadata_sets = {
            eval_split: frame_index_metadata(
                transitions,
                eval_transitions,
                train_horizon=config.horizon,
                eval_horizons=config.eval_horizons,
            )
            for eval_split, eval_transitions in eval_transition_sets.items()
        }

    config_dict = dict(config.__dict__)
    result = {
        "final_loss": float(epoch_loss),
        "config": config_dict,
        "wall_time_sec": time.time() - t0,
    }
    if len(eval_splits) == 1:
        only_split = eval_splits[0]
        result["rollout_errors"] = rollout_error_sets[only_split]
        if metadata_sets:
            result["metadata"] = metadata_sets[only_split]
    else:
        for eval_split in eval_splits:
            result[f"rollout_errors_{eval_split}"] = rollout_error_sets[eval_split]
            if metadata_sets:
                result[f"metadata_{eval_split}"] = metadata_sets[eval_split]

    checkpoint = None
    if config.save_checkpoint:
        checkpoint = save_checkpoint(config, model, float(epoch_loss), rollout_error_sets)

    if checkpoint is not None:
        try:
            result["checkpoint_path"] = str(checkpoint.relative_to(ROOT))
        except ValueError:
            result["checkpoint_path"] = str(checkpoint)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="reference", choices=SPLITS)
    parser.add_argument("--eval-split", default="reference", choices=SPLITS)
    parser.add_argument("--isomer", default="all")
    parser.add_argument("--eval-isomer", default="all")
    parser.add_argument("--data-root", default="")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick 1-seed 3-epoch test for pipeline verification",
    )
    args = parser.parse_args()

    if args.smoke:
        configs = [
            Config(
                split=args.split,
                eval_split=args.eval_split,
                isomer=args.isomer,
                eval_isomer=args.eval_isomer,
                data_root=args.data_root,
                prior="spectral",
                num_epochs=3,
                n_transitions=200,
                eval_n_transitions=50,
                seed=0,
                disjoint_eval=True,
            )
        ]
    else:
        configs = []
        for prior in ["none", "euclidean", "spectral"]:
            for seed in [0, 1, 2]:
                configs.append(
                    Config(
                        split=args.split,
                        eval_split=args.eval_split,
                        isomer=args.isomer,
                        eval_isomer=args.eval_isomer,
                        data_root=args.data_root,
                        prior=prior,
                        seed=seed,
                    )
                )

    if not torch.cuda.is_available():
        print("WARNING: CUDA unavailable, falling back to CPU")
        for config in configs:
            config.device = "cpu"
    else:
        print(f"device=cuda ({torch.cuda.get_device_name(0)})")

    all_results = {}
    for config in configs:
        key = (
            f"iso17:{config.split}->{config.eval_split}|{config.encoder}|"
            f"{config.prior}|seed={config.seed}"
        )
        print(f"\n=== Starting {key} ===", flush=True)
        result = train_one_seed(config)
        all_results[key] = result

        out_path = ROOT / "iso17_results.json"
        if args.smoke:
            out_path = ROOT / "iso17_smoke.json"
        with out_path.open("w", encoding="utf-8") as file:
            json.dump(all_results, file, indent=2, default=str)
        print(f"  -> saved partial results to {out_path.name}")

    print("\n=== Summary ===")
    print(f"train split: {args.split}; eval split: {args.eval_split}")
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
