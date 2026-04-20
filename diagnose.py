"""Diagnostics for the Phase-3 branching-profile pilot.

This script intentionally writes only diagnostic artifacts and leaves the
main pilot outputs untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from model import WorldModel, build_causal_laplacian, euclidean_cov_penalty
from train import (
    Config,
    HORIZONS,
    build_environment,
    collect_episodes,
    evaluate_rollout,
    flatten_transitions,
    move_obs_to_device,
    select_device,
    set_seed,
)


OUT_DIR = "/mnt/d/causalworld"
DIAGNOSTICS_MD = os.path.join(OUT_DIR, "diagnostics.md")
WEIGHT_SWEEP_PNG = os.path.join(OUT_DIR, "weight_sweep.png")
DIAGNOSTICS_JSON = os.path.join(OUT_DIR, "diagnostics_results.json")

DEFAULT_SEEDS = [0, 1, 2]
MAX_STEPS = 16
N_TRAIN = 200
N_EVAL = 50
BATCH_SIZE = 32
EPOCHS = 200
LATENT_DIM = 16
TRANSITION_HIDDEN_DIM = 128
ENV_PROFILE = "branching"


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std())


def _format_float(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:.4f}"


def _collect(seed: int, n_train: int = N_TRAIN, n_eval: int = N_EVAL) -> Tuple[List[dict], List[dict]]:
    rule, initial_state, max_steps = build_environment(ENV_PROFILE, seed)
    train_episodes = collect_episodes(
        rule,
        initial_state,
        n_train,
        max_steps,
        seed=seed,
        env_profile=ENV_PROFILE,
    )
    eval_episodes = collect_episodes(
        rule,
        initial_state,
        n_eval,
        max_steps,
        seed=10_000 + seed,
        env_profile=ENV_PROFILE,
    )
    return train_episodes, eval_episodes


def train_quiet(
    config: Config,
    train_transitions: List[dict],
    device: torch.device,
) -> WorldModel:
    set_seed(config.seed)
    model = WorldModel(
        encoder=config.encoder,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        action_dim=1,
        transition_hidden_dim=config.transition_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    prepared_transitions = []
    for transition in train_transitions:
        prepared = {
            "obs": move_obs_to_device(transition["obs"], device),
            "next_obs": move_obs_to_device(transition["next_obs"], device),
            "action": torch.tensor([float(transition["action"])], device=device),
            "reward": torch.tensor(float(transition["reward"]), device=device),
            "done": torch.tensor(float(transition["done"]), device=device),
        }
        if config.prior == "spectral":
            prepared["laplacian"] = build_causal_laplacian(
                transition["causal_graph"],
                config.latent_dim,
            ).to(device)
        prepared_transitions.append(prepared)

    for _ in range(config.num_epochs):
        model.train()
        epoch_transitions = list(prepared_transitions)
        random.shuffle(epoch_transitions)

        for start in range(0, len(epoch_transitions), config.batch_size):
            batch = epoch_transitions[start : start + config.batch_size]
            optimizer.zero_grad()
            base_totals: List[torch.Tensor] = []
            prior_losses: List[torch.Tensor] = []
            batch_latents: List[torch.Tensor] = []

            for transition in batch:
                obs = transition["obs"]
                next_obs = transition["next_obs"]
                action = transition["action"]
                reward = transition["reward"]
                done = transition["done"]

                if config.prior == "spectral":
                    loss_dict = model.loss(
                        obs,
                        action,
                        next_obs,
                        reward,
                        done,
                        prior="spectral",
                        prior_weight=config.prior_weight,
                        laplacian=transition["laplacian"],
                    )
                    prior_losses.append(loss_dict["prior"])
                elif config.prior == "euclidean":
                    loss_dict = model.loss(
                        obs,
                        action,
                        next_obs,
                        reward,
                        done,
                        prior="none",
                        prior_weight=0.0,
                    )
                    batch_latents.append(model.encode(obs))
                elif config.prior == "none":
                    loss_dict = model.loss(
                        obs,
                        action,
                        next_obs,
                        reward,
                        done,
                        prior="none",
                        prior_weight=0.0,
                    )
                else:
                    raise ValueError("prior must be one of none, euclidean, spectral")

                base_totals.append(loss_dict["total"])

            total = torch.stack(base_totals).mean()
            if config.prior == "euclidean":
                total = total + config.prior_weight * euclidean_cov_penalty(
                    torch.stack(batch_latents)
                )
            elif config.prior == "spectral":
                # The spectral prior is already included in per-transition totals.
                _ = prior_losses

            total.backward()
            optimizer.step()

    return model


def flat_features(obs) -> torch.Tensor:
    node_x = obs["node"].x
    edge_x = obs["hyperedge"].x
    num_nodes = float(node_x.shape[0])
    num_edges = float(edge_x.shape[0])

    if num_edges > 0:
        arities = edge_x[:, 0]
        mean_arity = float(arities.mean().item())
        max_arity = float(arities.max().item())
        origins = edge_x[:, 1]
        known = (origins >= 0).float()
        frac_known = float(known.mean().item())
        num_events = (
            float(origins[origins >= 0].max().item() + 1)
            if known.sum() > 0
            else 0.0
        )
    else:
        mean_arity = 0.0
        max_arity = 0.0
        frac_known = 0.0
        num_events = 0.0

    return torch.tensor(
        [num_nodes, num_edges, mean_arity, max_arity, frac_known, num_events],
        dtype=torch.float,
    )


def prepare_flat_transitions(
    transitions: List[dict],
    latent_dim: int,
) -> Dict[str, torch.Tensor]:
    return {
        "obs": torch.stack([flat_features(t["obs"]) for t in transitions]),
        "next_obs": torch.stack([flat_features(t["next_obs"]) for t in transitions]),
        "actions": torch.tensor(
            [[float(t["action"])] for t in transitions],
            dtype=torch.float,
        ),
        "rewards": torch.tensor(
            [float(t["reward"]) for t in transitions],
            dtype=torch.float,
        ),
        "dones": torch.tensor(
            [float(t["done"]) for t in transitions],
            dtype=torch.float,
        ),
        "laplacians": torch.stack(
            [build_causal_laplacian(t["causal_graph"], latent_dim) for t in transitions]
        ),
    }


def train_flat_quick(
    config: Config,
    train_transitions: List[dict],
    device: torch.device,
) -> WorldModel:
    set_seed(config.seed)
    model = WorldModel(
        encoder="flat",
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        action_dim=1,
        transition_hidden_dim=config.transition_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    prepared = {
        key: value.to(device)
        for key, value in prepare_flat_transitions(train_transitions, config.latent_dim).items()
    }
    n = prepared["obs"].shape[0]

    for _ in range(config.num_epochs):
        model.train()
        permutation = torch.randperm(n, device=device)
        for start in range(0, n, config.batch_size):
            index = permutation[start : start + config.batch_size]
            obs_features = prepared["obs"][index]
            next_features = prepared["next_obs"][index]
            actions = prepared["actions"][index]
            rewards = prepared["rewards"][index]
            dones = prepared["dones"][index]

            optimizer.zero_grad()
            z = model.encoder.mlp(obs_features)
            z_next = model.encoder.mlp(next_features).detach()
            z_pred, reward_pred, done_logit = model.transition(z, actions)

            transition_loss = F.mse_loss(z_pred, z_next)
            reward_loss = F.mse_loss(reward_pred, rewards)
            done_loss = F.binary_cross_entropy_with_logits(done_logit, dones)
            reg_l2 = z.pow(2).mean()
            reg_smooth = (z_pred - z).pow(2).mean()
            total = transition_loss + reward_loss + done_loss + 1e-4 * reg_l2 + 1e-3 * reg_smooth

            if config.prior == "euclidean":
                total = total + config.prior_weight * euclidean_cov_penalty(z)
            elif config.prior == "spectral":
                laplacians = prepared["laplacians"][index]
                prior = torch.einsum("bi,bij,bj->b", z, laplacians, z).mean()
                total = total + config.prior_weight * prior
            elif config.prior != "none":
                raise ValueError("prior must be one of none, euclidean, spectral")

            total.backward()
            optimizer.step()

    return model


def evaluate_flat_rollout(
    model: WorldModel,
    episodes: List[dict],
    horizons: List[int],
    device: torch.device,
) -> Dict[int, float]:
    was_training = model.training
    model.eval()
    metrics: Dict[int, float] = {}

    with torch.no_grad():
        for horizon in horizons:
            errors: List[float] = []
            for episode in episodes:
                if len(episode["actions"]) < horizon or len(episode["observations"]) <= horizon:
                    continue
                obs0 = flat_features(episode["observations"][0]).to(device)
                obsH = flat_features(episode["observations"][horizon]).to(device)
                z0 = model.encoder.mlp(obs0)
                z_rollout = model.rollout_latent(z0, episode["actions"][:horizon])
                z_pred_H = z_rollout[-1]
                z_true_H = model.encoder.mlp(obsH)
                error = torch.norm(z_pred_H - z_true_H, p=2) / math.sqrt(model.latent_dim)
                errors.append(float(error.detach().cpu()))
            metrics[horizon] = float(np.mean(errors)) if errors else float("nan")

    if was_training:
        model.train()
    return metrics


def flat_one_step_error_by_step(
    model: WorldModel,
    episodes: List[dict],
    device: torch.device,
    steps: List[int],
) -> Dict[int, float]:
    was_training = model.training
    model.eval()
    errors_by_step: Dict[int, List[float]] = {step: [] for step in steps}

    with torch.no_grad():
        for episode in episodes:
            transitions = episode["transitions"]
            for step in steps:
                index = step - 1
                if index >= len(transitions):
                    continue
                transition = transitions[index]
                obs_features = flat_features(transition["obs"]).to(device)
                next_features = flat_features(transition["next_obs"]).to(device)
                action = torch.tensor([[float(transition["action"])]], device=device)
                z = model.encoder.mlp(obs_features)
                pred, _, _ = model.transition(z, action)
                target = model.encoder.mlp(next_features)
                error = torch.norm(pred - target, p=2) / math.sqrt(model.latent_dim)
                errors_by_step[step].append(float(error.detach().cpu()))

    if was_training:
        model.train()
    return {
        step: float(np.mean(values)) if values else float("nan")
        for step, values in errors_by_step.items()
    }


def encode_hypergraph_batch(
    model: WorldModel,
    observations: List[object],
    device: torch.device,
) -> torch.Tensor:
    data = Batch.from_data_list([obs.clone() if hasattr(obs, "clone") else obs for obs in observations]).to(device)
    encoder = model.encoder
    node_x = data["node"].x
    edge_x = data["hyperedge"].x
    if node_x.shape[0] == 0 or edge_x.shape[0] == 0:
        return torch.zeros((len(observations), model.latent_dim), device=device)

    x_dict = {
        "node": F.relu(encoder.node_proj(node_x)),
        "hyperedge": F.relu(encoder.edge_proj(edge_x)),
    }
    edge_index_dict = {
        ("node", "member_of", "hyperedge"): data["node", "member_of", "hyperedge"].edge_index,
        ("hyperedge", "has_member", "node"): data["hyperedge", "has_member", "node"].edge_index,
    }
    x_dict = {key: F.relu(value) for key, value in encoder.conv1(x_dict, edge_index_dict).items()}
    x_dict = {key: F.relu(value) for key, value in encoder.conv2(x_dict, edge_index_dict).items()}
    node_pool = global_mean_pool(x_dict["node"], data["node"].batch)
    edge_pool = global_mean_pool(x_dict["hyperedge"], data["hyperedge"].batch)
    return encoder.out(torch.cat([node_pool, edge_pool], dim=-1))


def train_hypergraph_quick(
    config: Config,
    train_transitions: List[dict],
    device: torch.device,
) -> WorldModel:
    set_seed(config.seed)
    model = WorldModel(
        encoder="hypergraph",
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        action_dim=1,
        transition_hidden_dim=config.transition_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    prepared = [
        {
            "obs": transition["obs"],
            "next_obs": transition["next_obs"],
            "action": float(transition["action"]),
            "reward": float(transition["reward"]),
            "done": float(transition["done"]),
            "laplacian": build_causal_laplacian(
                transition["causal_graph"],
                config.latent_dim,
            ),
        }
        for transition in train_transitions
    ]

    for _ in range(config.num_epochs):
        model.train()
        random.shuffle(prepared)
        for start in range(0, len(prepared), config.batch_size):
            batch = prepared[start : start + config.batch_size]
            optimizer.zero_grad()
            z = encode_hypergraph_batch(model, [item["obs"] for item in batch], device)
            z_next = encode_hypergraph_batch(model, [item["next_obs"] for item in batch], device).detach()
            actions = torch.tensor([[item["action"]] for item in batch], dtype=torch.float, device=device)
            rewards = torch.tensor([item["reward"] for item in batch], dtype=torch.float, device=device)
            dones = torch.tensor([item["done"] for item in batch], dtype=torch.float, device=device)
            z_pred, reward_pred, done_logit = model.transition(z, actions)

            transition_loss = F.mse_loss(z_pred, z_next)
            reward_loss = F.mse_loss(reward_pred, rewards)
            done_loss = F.binary_cross_entropy_with_logits(done_logit, dones)
            reg_l2 = z.pow(2).mean()
            reg_smooth = (z_pred - z).pow(2).mean()
            total = transition_loss + reward_loss + done_loss + 1e-4 * reg_l2 + 1e-3 * reg_smooth

            if config.prior == "spectral":
                laplacians = torch.stack([item["laplacian"] for item in batch]).to(device)
                prior = torch.einsum("bi,bij,bj->b", z, laplacians, z).mean()
                total = total + config.prior_weight * prior
            elif config.prior == "euclidean":
                total = total + config.prior_weight * euclidean_cov_penalty(z)
            elif config.prior != "none":
                raise ValueError("prior must be one of none, euclidean, spectral")

            total.backward()
            optimizer.step()

    return model


def evaluate_hypergraph_rollout_quick(
    model: WorldModel,
    episodes: List[dict],
    horizons: List[int],
    device: torch.device,
) -> Dict[int, float]:
    was_training = model.training
    model.eval()
    metrics: Dict[int, float] = {}

    with torch.no_grad():
        for horizon in horizons:
            valid = [
                episode
                for episode in episodes
                if len(episode["actions"]) >= horizon and len(episode["observations"]) > horizon
            ]
            if not valid:
                metrics[horizon] = float("nan")
                continue
            z = encode_hypergraph_batch(
                model,
                [episode["observations"][0] for episode in valid],
                device,
            )
            for step in range(horizon):
                actions = torch.tensor(
                    [[float(episode["actions"][step])] for episode in valid],
                    dtype=torch.float,
                    device=device,
                )
                z, _, _ = model.transition(z, actions)
            z_true = encode_hypergraph_batch(
                model,
                [episode["observations"][horizon] for episode in valid],
                device,
            )
            errors = torch.norm(z - z_true, p=2, dim=-1) / math.sqrt(model.latent_dim)
            metrics[horizon] = float(errors.mean().detach().cpu())

    if was_training:
        model.train()
    return metrics


def one_step_error_by_step(
    model: WorldModel,
    episodes: List[dict],
    device: torch.device,
    steps: List[int],
) -> Dict[int, float]:
    was_training = model.training
    model.eval()
    errors_by_step: Dict[int, List[float]] = {step: [] for step in steps}

    with torch.no_grad():
        for episode in episodes:
            transitions = episode["transitions"]
            for step in steps:
                index = step - 1
                if index >= len(transitions):
                    continue
                transition = transitions[index]
                obs = move_obs_to_device(transition["obs"], device)
                next_obs = move_obs_to_device(transition["next_obs"], device)
                action = torch.tensor([float(transition["action"])], device=device)
                pred = model.forward(obs, action)["next_latent_pred"]
                target = model.encode(next_obs)
                error = torch.norm(pred - target, p=2) / math.sqrt(model.latent_dim)
                errors_by_step[step].append(float(error.detach().cpu()))

    if was_training:
        model.train()
    return {
        step: float(np.mean(values)) if values else float("nan")
        for step, values in errors_by_step.items()
    }


def causal_graph_size_audit(device: torch.device) -> Dict[str, object]:
    steps = [1, 2, 4, 8, 16]
    train_episodes, eval_episodes = _collect(seed=0, n_train=N_TRAIN, n_eval=N_EVAL)
    transitions_by_step: Dict[int, List[dict]] = {step: [] for step in steps}
    for episode in train_episodes:
        for step in steps:
            index = step - 1
            if index < len(episode["transitions"]):
                transitions_by_step[step].append(episode["transitions"][index])

    train_transitions = flatten_transitions(train_episodes)
    config = Config(
        encoder="flat",
        prior="spectral",
        prior_weight=0.01,
        latent_dim=LATENT_DIM,
        hidden_dim=32,
        transition_hidden_dim=TRANSITION_HIDDEN_DIM,
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seed=0,
    )
    model = train_flat_quick(config, train_transitions, device)
    h1_errors = flat_one_step_error_by_step(model, train_episodes, device, steps)

    rows = []
    for step in steps:
        transitions = transitions_by_step[step]
        node_counts = [t["causal_graph"].number_of_nodes() for t in transitions]
        edge_counts = [t["causal_graph"].number_of_edges() for t in transitions]
        rows.append(
            {
                "step_in_episode": step,
                "mean_nodes": float(np.mean(node_counts)) if node_counts else float("nan"),
                "mean_edges": float(np.mean(edge_counts)) if edge_counts else float("nan"),
                "mean_H1_error_of_spectral_prior": h1_errors[step],
            }
        )

    eval_h16 = evaluate_flat_rollout(model, eval_episodes, [16], device)[16]
    step16_nodes = next(row["mean_nodes"] for row in rows if row["step_in_episode"] == 16)
    identity_like_flag = bool(math.isfinite(step16_nodes) and step16_nodes < 10.0)
    print(
        "diagnostic 1/4 causal_graph_size: "
        f"step16_mean_nodes={step16_nodes:.2f} "
        f"flat_spectral_eval_H16={eval_h16:.4f} "
        f"identity_like_flag={identity_like_flag}"
    )
    return {
        "rows": rows,
        "step16_identity_like_flag": identity_like_flag,
        "flat_spectral_seed0_eval_H16": eval_h16,
    }


def weight_sensitivity(device: torch.device) -> Dict[str, object]:
    weights = [0.01, 0.03, 0.1, 0.3, 1.0]
    priors = ["euclidean", "spectral"]
    seeds = DEFAULT_SEEDS
    results: Dict[str, Dict[str, List[float]]] = {
        prior: {str(weight): [] for weight in weights} for prior in priors
    }

    for seed in seeds:
        train_episodes, eval_episodes = _collect(seed=seed)
        train_transitions = flatten_transitions(train_episodes)
        for prior in priors:
            for weight in weights:
                config = Config(
                    encoder="flat",
                    prior=prior,
                    prior_weight=weight,
                    latent_dim=LATENT_DIM,
                    hidden_dim=32,
                    transition_hidden_dim=TRANSITION_HIDDEN_DIM,
                    num_epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    seed=seed,
                )
                model = train_flat_quick(config, train_transitions, device)
                h16 = evaluate_flat_rollout(model, eval_episodes, [16], device)[16]
                results[prior][str(weight)].append(h16)
                print(
                    "weight_sweep_cell "
                    f"prior={prior} weight={weight:g} seed={seed} H16={h16:.4f}",
                    flush=True,
                )

    plt.figure(figsize=(7, 4.5))
    x = np.asarray(weights, dtype=float)
    for prior in priors:
        means = []
        stds = []
        for weight in weights:
            mean, std = _mean_std(results[prior][str(weight)])
            means.append(mean)
            stds.append(std)
        y = np.asarray(means, dtype=float)
        yerr = np.asarray(stds, dtype=float)
        plt.plot(x, y, marker="o", linewidth=1.8, label=prior)
        plt.fill_between(x, np.maximum(y - yerr, 1e-8), y + yerr, alpha=0.18)
    plt.xscale("log")
    plt.xlabel("Prior weight")
    plt.ylabel("H=16 rollout error")
    plt.title("Flat encoder prior-weight sensitivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(WEIGHT_SWEEP_PNG, dpi=150)
    plt.close()

    spectral_wins = []
    for weight in weights:
        euc_mean, _ = _mean_std(results["euclidean"][str(weight)])
        spec_mean, _ = _mean_std(results["spectral"][str(weight)])
        spectral_wins.append(bool(spec_mean < euc_mean))
    print(
        "diagnostic 2/4 weight_sensitivity: "
        f"spectral_wins={sum(spectral_wins)}/{len(weights)} "
        f"plot={WEIGHT_SWEEP_PNG}"
    )
    return {
        "weights": weights,
        "results": results,
        "spectral_wins_by_weight": {
            str(weight): spectral_wins[index] for index, weight in enumerate(weights)
        },
    }


def encoder_capacity(device: torch.device) -> Dict[str, object]:
    hidden_dims = [32, 64, 128]
    seeds = DEFAULT_SEEDS
    results: Dict[str, List[float]] = {str(hidden_dim): [] for hidden_dim in hidden_dims}

    for seed in seeds:
        train_episodes, eval_episodes = _collect(seed=seed)
        train_transitions = flatten_transitions(train_episodes)
        for hidden_dim in hidden_dims:
            config = Config(
                encoder="hypergraph",
                prior="spectral",
                prior_weight=0.01,
                latent_dim=LATENT_DIM,
                hidden_dim=hidden_dim,
                transition_hidden_dim=TRANSITION_HIDDEN_DIM,
                num_epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                seed=seed,
            )
            model = train_hypergraph_quick(config, train_transitions, device)
            h16 = evaluate_hypergraph_rollout_quick(model, eval_episodes, [16], device)[16]
            results[str(hidden_dim)].append(h16)
            print(
                "capacity_cell "
                f"hidden_dim={hidden_dim} seed={seed} H16={h16:.4f}",
                flush=True,
            )

    means = {hidden_dim: _mean_std(values)[0] for hidden_dim, values in results.items()}
    monotonic_drop = means["32"] > means["64"] > means["128"]
    print(
        "diagnostic 3/4 encoder_capacity: "
        f"H16_means={{{', '.join(f'{k}: {v:.4f}' for k, v in means.items())}}} "
        f"monotonic_drop={monotonic_drop}"
    )
    return {
        "hidden_dims": hidden_dims,
        "results": results,
        "mean_H16": means,
        "std_H16": {hidden_dim: _mean_std(values)[1] for hidden_dim, values in results.items()},
        "monotonic_drop": monotonic_drop,
    }


def seed_stability(device: torch.device) -> Dict[str, object]:
    seeds = list(range(10))
    h16_values: List[float] = []
    for seed in seeds:
        train_episodes, eval_episodes = _collect(seed=seed)
        train_transitions = flatten_transitions(train_episodes)
        config = Config(
            encoder="flat",
            prior="spectral",
            prior_weight=0.01,
            latent_dim=LATENT_DIM,
            hidden_dim=32,
            transition_hidden_dim=TRANSITION_HIDDEN_DIM,
            num_epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            seed=seed,
        )
        model = train_flat_quick(config, train_transitions, device)
        h16 = evaluate_flat_rollout(model, eval_episodes, [16], device)[16]
        h16_values.append(h16)
        print(f"seed_stability_cell seed={seed} H16={h16:.4f}", flush=True)

    mean, std = _mean_std(h16_values)
    ratio = std / mean if math.isfinite(mean) and mean != 0 else float("nan")
    insufficient = bool(math.isfinite(ratio) and ratio > 0.3)
    print(
        "diagnostic 4/4 seed_stability: "
        f"H16_mean={mean:.4f} std={std:.4f} min={min(h16_values):.4f} "
        f"max={max(h16_values):.4f} std_over_mean={ratio:.4f} "
        f"three_seed_insufficient={insufficient}"
    )
    return {
        "seeds": seeds,
        "H16_values": h16_values,
        "mean": mean,
        "std": std,
        "min": float(min(h16_values)),
        "max": float(max(h16_values)),
        "std_over_mean": ratio,
        "three_seed_insufficient": insufficient,
    }


def _markdown_table(headers: List[str], rows: List[List[object]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join([header, sep, *body])


def write_diagnostics_md(results: Dict[str, object]) -> None:
    causal = results["causal_graph_size"]
    weight = results["weight_sensitivity"]
    capacity = results["encoder_capacity"]
    stability = results["seed_stability"]

    lines: List[str] = ["# Phase-3 Pilot Diagnostics", ""]

    lines.extend(["## Causal Graph Size", ""])
    lines.append(
        _markdown_table(
            [
                "step_in_episode",
                "mean_nodes",
                "mean_edges",
                "mean_H1_error_of_spectral_prior",
            ],
            [
                [
                    row["step_in_episode"],
                    _format_float(row["mean_nodes"]),
                    _format_float(row["mean_edges"]),
                    _format_float(row["mean_H1_error_of_spectral_prior"]),
                ]
                for row in causal["rows"]
            ],
        )
    )
    lines.append("")
    if causal["step16_identity_like_flag"]:
        lines.append(
            "**Flag:** mean causal graph nodes at step 16 is under 10, so the "
            "Laplacian may be mostly identity-like. Treat the spectral advantage "
            "as potentially acting like a disguised L2 penalty rather than a "
            "genuine graph-spectral prior."
        )
    else:
        lines.append(
            "Mean causal graph nodes at step 16 is at least 10, so this audit does "
            "not by itself indicate an identity-like Laplacian."
        )
    lines.append("")

    lines.extend(["## Weight Sensitivity", ""])
    weight_rows = []
    for prior in ["euclidean", "spectral"]:
        for prior_weight in weight["weights"]:
            values = weight["results"][prior][str(prior_weight)]
            mean, std = _mean_std(values)
            weight_rows.append(
                [
                    prior,
                    prior_weight,
                    _format_float(mean),
                    _format_float(std),
                    ", ".join(_format_float(value) for value in values),
                ]
            )
    lines.append(
        _markdown_table(
            ["prior", "weight", "H16_mean", "H16_std", "seed_values"],
            weight_rows,
        )
    )
    spectral_wins = sum(weight["spectral_wins_by_weight"].values())
    lines.append("")
    lines.append(
        f"Spectral beats euclidean at {spectral_wins}/{len(weight['weights'])} "
        f"tested weights. Plot: `weight_sweep.png`."
    )
    lines.append("")

    lines.extend(["## Encoder Capacity", ""])
    capacity_rows = []
    for hidden_dim in capacity["hidden_dims"]:
        values = capacity["results"][str(hidden_dim)]
        capacity_rows.append(
            [
                hidden_dim,
                _format_float(capacity["mean_H16"][str(hidden_dim)]),
                _format_float(capacity["std_H16"][str(hidden_dim)]),
                ", ".join(_format_float(value) for value in values),
            ]
        )
    lines.append(
        _markdown_table(
            ["hidden_dim", "H16_mean", "H16_std", "seed_values"],
            capacity_rows,
        )
    )
    lines.append("")
    if capacity["monotonic_drop"]:
        lines.append(
            "H=16 error drops monotonically with hidden_dim, so the original "
            "hypergraph result appears capacity-sensitive."
        )
    else:
        lines.append(
            "H=16 error does not drop monotonically with hidden_dim, so this "
            "diagnostic does not support a simple capacity-limited explanation."
        )
    lines.append("")

    lines.extend(["## Seed Stability", ""])
    lines.append(
        _markdown_table(
            ["metric", "value"],
            [
                ["H16_mean", _format_float(stability["mean"])],
                ["H16_std", _format_float(stability["std"])],
                ["H16_min", _format_float(stability["min"])],
                ["H16_max", _format_float(stability["max"])],
                ["std_over_mean", _format_float(stability["std_over_mean"])],
            ],
        )
    )
    lines.append("")
    lines.append(
        "Seed H=16 values: "
        + ", ".join(_format_float(value) for value in stability["H16_values"])
    )
    lines.append("")
    if stability["three_seed_insufficient"]:
        lines.append(
            "**Flag:** std/mean is above 0.3, so 3-seed averaging is insufficient "
            "for the main grid. Rerun the main grid with 10 seeds before treating "
            "effect sizes as stable."
        )
    else:
        lines.append(
            "std/mean is not above 0.3, so this diagnostic does not flag the "
            "3-seed mean as unstable."
        )
    lines.append("")

    flat_spectral_h16 = stability["mean"]
    capacity_best = min(capacity["mean_H16"].values())
    weight_spectral_wide = spectral_wins >= 4
    graph_ok = not causal["step16_identity_like_flag"]
    stable = not stability["three_seed_insufficient"]
    if graph_ok and weight_spectral_wide and stable:
        verdict = (
            "The Phase-3 pilot result is reasonably trustworthy as directional "
            "evidence: the causal graphs are not identity-like by this audit, "
            "the spectral prior wins across most weights, and seed variability "
            "does not exceed the warning threshold."
        )
    else:
        concerns = []
        if not graph_ok:
            concerns.append("causal graphs may be too small for a genuine spectral effect")
        if not weight_spectral_wide:
            concerns.append("the spectral win is not broad across prior weights")
        if not stable:
            concerns.append("10-seed variability is too high for 3-seed claims")
        if capacity_best < flat_spectral_h16:
            concerns.append("a larger hypergraph encoder can beat flat+spectral")
        verdict = (
            "The Phase-3 pilot remains suggestive but should not yet be treated "
            "as settled evidence because "
            + "; ".join(concerns)
            + "."
        )

    lines.extend(["## Can I Trust The Phase-3 Pilot Result?", "", verdict, ""])

    with open(DIAGNOSTICS_MD, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def save_json(results: Dict[str, object]) -> None:
    with open(DIAGNOSTICS_JSON, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"diagnostics_device={device}")
    config = {
            "env_profile": ENV_PROFILE,
            "n_train": N_TRAIN,
            "n_eval": N_EVAL,
            "max_steps": MAX_STEPS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "latent_dim": LATENT_DIM,
            "transition_hidden_dim": TRANSITION_HIDDEN_DIM,
            "device": str(device),
        }
    results: Dict[str, object] = {"config": config}
    if os.path.exists(DIAGNOSTICS_JSON):
        with open(DIAGNOSTICS_JSON, "r", encoding="utf-8") as file:
            loaded = json.load(file)
        if loaded.get("config") == config:
            results.update(loaded)
            print(f"resuming diagnostics from {DIAGNOSTICS_JSON}", flush=True)

    if "causal_graph_size" not in results:
        results["causal_graph_size"] = causal_graph_size_audit(device)
        save_json(results)
    else:
        print("diagnostic 1/4 causal_graph_size: reused checkpoint", flush=True)

    if "weight_sensitivity" not in results:
        results["weight_sensitivity"] = weight_sensitivity(device)
        save_json(results)
    else:
        print("diagnostic 2/4 weight_sensitivity: reused checkpoint", flush=True)

    if "encoder_capacity" not in results:
        results["encoder_capacity"] = encoder_capacity(device)
        save_json(results)
    else:
        print("diagnostic 3/4 encoder_capacity: reused checkpoint", flush=True)

    if "seed_stability" not in results:
        results["seed_stability"] = seed_stability(device)
        save_json(results)
    else:
        print("diagnostic 4/4 seed_stability: reused checkpoint", flush=True)
    save_json(results)
    write_diagnostics_md(results)
    print(f"diagnostics written: {DIAGNOSTICS_MD}")
    print(f"diagnostics json: {DIAGNOSTICS_JSON}")
    print(f"weight sweep plot: {WEIGHT_SWEEP_PNG}")


if __name__ == "__main__":
    main()
