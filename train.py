"""Phase-3 pilot horizon-scaling experiment for causal hypergraph models."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData

from hypergraph_env import CausalWorldEnv, HypergraphState, RewriteRule
from model import WorldModel, build_causal_laplacian, euclidean_cov_penalty


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HORIZONS = [1, 2, 4, 8, 16]


@dataclass
class Config:
    encoder: str
    prior: str
    prior_weight: float
    latent_dim: int = 16
    hidden_dim: int = 32
    transition_hidden_dim: int = 128
    num_epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_obs_to_device(obs: HeteroData, device: torch.device) -> HeteroData:
    if hasattr(obs, "clone"):
        return obs.clone().to(device)
    return copy.deepcopy(obs).to(device)


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    raise ValueError("device must be one of 'auto', 'cpu', or 'cuda'")


def print_device_diagnostics(device: torch.device) -> None:
    print(f"torch.__version__={torch.__version__}")
    print(f"torch.version.cuda={torch.version.cuda}")
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.current_device_name={torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("torch.cuda.current_device_name=None")
    print(f"selected DEVICE={device}")


def summarize_episodes(episodes: List[dict], label: str) -> None:
    lengths = [len(episode["actions"]) for episode in episodes]
    transitions = flatten_transitions(episodes)
    actions = [transition["action"] for transition in transitions]
    action_histogram: Dict[int, int] = {}
    for action in actions:
        action_histogram[action] = action_histogram.get(action, 0) + 1

    causal_node_counts = [
        transition["causal_graph"].number_of_nodes()
        for transition in transitions
    ]
    causal_edge_counts = [
        transition["causal_graph"].number_of_edges()
        for transition in transitions
    ]
    available_matches = [
        transition.get("available_matches", -1)
        for transition in transitions
    ]
    available_matches_before = [
        transition.get("available_matches_before", transition.get("available_matches", -1))
        for transition in transitions
    ]
    available_matches_after = [
        transition.get("available_matches_after", transition.get("available_matches", -1))
        for transition in transitions
    ]

    def stats(values: List[int]) -> Tuple[float, float, float]:
        if not values:
            return float("nan"), float("nan"), float("nan")
        return float(min(values)), float(np.mean(values)), float(max(values))

    length_min, length_mean, length_max = stats(lengths)
    node_min, node_mean, node_max = stats(causal_node_counts)
    edge_min, edge_mean, edge_max = stats(causal_edge_counts)
    match_min, match_mean, match_max = stats(available_matches)
    match_before_min, match_before_mean, match_before_max = stats(available_matches_before)
    match_after_min, match_after_mean, match_after_max = stats(available_matches_after)
    print(f"[{label}] episodes={len(episodes)} transitions={len(transitions)}")
    print(
        f"[{label}] episode_length min/mean/max="
        f"{length_min:.0f}/{length_mean:.2f}/{length_max:.0f}"
    )
    print(f"[{label}] unique_actions={len(set(actions))} action_histogram={action_histogram}")
    print(
        f"[{label}] causal_nodes min/mean/max="
        f"{node_min:.0f}/{node_mean:.2f}/{node_max:.0f}"
    )
    print(
        f"[{label}] causal_edges min/mean/max="
        f"{edge_min:.0f}/{edge_mean:.2f}/{edge_max:.0f}"
    )
    print(
        f"[{label}] available_matches min/mean/max="
        f"{match_min:.0f}/{match_mean:.2f}/{match_max:.0f}"
    )
    print(
        f"[{label}] available_matches_before min/mean/max="
        f"{match_before_min:.0f}/{match_before_mean:.2f}/{match_before_max:.0f}"
    )
    print(
        f"[{label}] available_matches_after min/mean/max="
        f"{match_after_min:.0f}/{match_after_mean:.2f}/{match_after_max:.0f}"
    )


def make_initial_state(profile: str, seed: int) -> HypergraphState:
    rng = random.Random(seed)
    if profile == "minimal":
        return HypergraphState([(0, 1), (1, 2)], seed=seed)
    if profile == "branching":
        chain_len = 7 + rng.randrange(3)
        edges = [(node, node + 1) for node in range(chain_len)]
        next_node = chain_len + 1

        for center in range(1, chain_len):
            incoming = next_node
            outgoing = next_node + 1
            next_node += 2
            edges.append((incoming, center))
            edges.append((center, outgoing))

        for _ in range(3):
            start = rng.randrange(0, chain_len)
            edges.append((start, next_node))
            edges.append((next_node, min(start + 1, chain_len)))
            next_node += 1

        rng.shuffle(edges)
        return HypergraphState(edges, seed=seed)
    raise ValueError("profile must be either 'minimal' or 'branching'")


def build_environment(profile: str = "minimal", seed: int = 0) -> Tuple[RewriteRule, HypergraphState, int]:
    rule = RewriteRule(
        lhs=[("x", "y"), ("y", "z")],
        rhs=[("x", "z"), ("z", "w")],
        name="{x,y},{y,z}->{x,z},{z,w}",
    )
    initial_state = make_initial_state(profile, seed)
    return rule, initial_state, 16


def collect_episodes(
    rule: RewriteRule,
    initial_state: HypergraphState,
    num_episodes: int,
    max_steps: int,
    seed: int,
    env_profile: str = "minimal",
) -> List[dict]:
    episodes: List[dict] = []
    for episode_index in range(num_episodes):
        episode_seed = seed + episode_index
        episode_initial_state = make_initial_state(env_profile, episode_seed)
        env = CausalWorldEnv(
            rule,
            episode_initial_state,
            max_steps=max_steps,
            seed=episode_seed,
        )
        obs = env.reset(seed=episode_seed)
        observations = [obs]
        actions: List[int] = []
        transitions: List[dict] = []
        done = False

        while not done and len(actions) < max_steps:
            available_matches_before = len(rule.find_matches(env.state))
            action = env.sample_action()
            if action is None:
                break

            next_obs, reward, done, info = env.step(action)
            available_matches_after = int(info.get("available_matches", -1))
            causal_graph_snapshot = env.state.causal_graph().copy()
            transitions.append(
                {
                    "obs": obs,
                    "action": int(action),
                    "next_obs": next_obs,
                    "reward": float(reward),
                    "done": float(done),
                    "causal_graph": causal_graph_snapshot,
                    "available_matches": available_matches_after,
                    "available_matches_before": int(available_matches_before),
                    "available_matches_after": available_matches_after,
                }
            )
            actions.append(int(action))
            observations.append(next_obs)
            obs = next_obs

        episodes.append(
            {
                "observations": observations,
                "actions": actions,
                "transitions": transitions,
            }
        )
    return episodes


def flatten_transitions(episodes: Iterable[dict]) -> List[dict]:
    return [
        transition
        for episode in episodes
        for transition in episode["transitions"]
    ]


def evaluate_rollout(
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

                obs0 = move_obs_to_device(episode["observations"][0], device)
                obsH = move_obs_to_device(episode["observations"][horizon], device)
                actions = episode["actions"][:horizon]

                z0 = model.encode(obs0)
                z_rollout = model.rollout_latent(z0, actions)
                z_pred_H = z_rollout[-1]
                z_true_H = model.encode(obsH)
                error = torch.norm(z_pred_H - z_true_H, p=2) / math.sqrt(model.latent_dim)
                errors.append(float(error.detach().cpu()))

            metrics[horizon] = float(np.mean(errors)) if errors else float("nan")

    if was_training:
        model.train()
    return metrics


def train_one(
    config: Config,
    train_transitions: List[dict],
    eval_episodes: List[dict],
    device: torch.device,
    horizons: List[int] = HORIZONS,
) -> Tuple[WorldModel, dict]:
    set_seed(config.seed)
    model = WorldModel(
        encoder=config.encoder,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        action_dim=1,
        transition_hidden_dim=config.transition_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    history = {
        "total_loss": [],
        "transition_loss": [],
        "reward_loss": [],
        "prior_loss": [],
        "rollout": [],
    }

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_transitions = list(train_transitions)
        random.shuffle(epoch_transitions)
        total_values: List[float] = []
        transition_values: List[float] = []
        reward_values: List[float] = []
        prior_values: List[float] = []

        for start in range(0, len(epoch_transitions), config.batch_size):
            batch = epoch_transitions[start:start + config.batch_size]
            optimizer.zero_grad()
            base_totals: List[torch.Tensor] = []
            transition_losses: List[torch.Tensor] = []
            reward_losses: List[torch.Tensor] = []
            prior_losses: List[torch.Tensor] = []
            batch_latents: List[torch.Tensor] = []

            for transition in batch:
                obs = move_obs_to_device(transition["obs"], device)
                next_obs = move_obs_to_device(transition["next_obs"], device)
                action = torch.tensor([float(transition["action"])], device=device)
                reward = torch.tensor(float(transition["reward"]), device=device)
                done = torch.tensor(float(transition["done"]), device=device)

                if config.prior == "spectral":
                    laplacian = build_causal_laplacian(
                        transition["causal_graph"],
                        config.latent_dim,
                    ).to(device)
                    loss_dict = model.loss(
                        obs,
                        action,
                        next_obs,
                        reward,
                        done,
                        prior="spectral",
                        prior_weight=config.prior_weight,
                        laplacian=laplacian,
                    )
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
                    raise ValueError("prior must be one of 'none', 'euclidean', or 'spectral'")

                base_totals.append(loss_dict["total"])
                transition_losses.append(loss_dict["transition"])
                reward_losses.append(loss_dict["reward"])
                prior_losses.append(loss_dict["prior"])

            total = torch.stack(base_totals).mean()
            if config.prior == "euclidean":
                latent_batch = torch.stack(batch_latents)
                prior_loss = euclidean_cov_penalty(latent_batch)
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

        eval_metrics = evaluate_rollout(model, eval_episodes, horizons, device)
        rollout_H8 = eval_metrics.get(8, float("nan"))
        mean_total = float(np.mean(total_values)) if total_values else float("nan")
        mean_transition = float(np.mean(transition_values)) if transition_values else float("nan")
        mean_reward = float(np.mean(reward_values)) if reward_values else float("nan")
        mean_prior = float(np.mean(prior_values)) if prior_values else float("nan")

        history["total_loss"].append(mean_total)
        history["transition_loss"].append(mean_transition)
        history["reward_loss"].append(mean_reward)
        history["prior_loss"].append(mean_prior)
        history["rollout"].append(eval_metrics)

        print(
            f"[{config.encoder}|{config.prior}|seed={config.seed}] "
            f"epoch {epoch}/{config.num_epochs} "
            f"total={mean_total:.3f} "
            f"trans={mean_transition:.3f} "
            f"prior={mean_prior:.3f} "
            f"rollout_H8={rollout_H8:.2f}"
        )

    return model, history


def _mean_std(values: List[float]) -> Tuple[float, float]:
    finite = np.array([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std())


def _save_results(results: Dict[str, Dict[str, List[float]]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, sort_keys=True)


def _plot_results(results: Dict[str, Dict[str, List[float]]], path: str) -> None:
    plt.figure(figsize=(8, 5))
    horizons = HORIZONS
    x = np.array(horizons, dtype=float)

    for key, horizon_values in sorted(results.items()):
        means: List[float] = []
        stds: List[float] = []
        for horizon in horizons:
            mean, std = _mean_std(horizon_values[str(horizon)])
            means.append(mean)
            stds.append(std)

        y = np.array(means, dtype=float)
        yerr = np.array(stds, dtype=float)
        plt.plot(x, y, marker="o", linewidth=1.6, label=key)
        lower = np.maximum(y - yerr, 1e-8)
        upper = np.maximum(y + yerr, 1e-8)
        plt.fill_between(x, lower, upper, alpha=0.15)

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(horizons, [str(horizon) for horizon in horizons])
    plt.xlabel("Horizon")
    plt.ylabel("Mean rollout error")
    plt.title("Horizon scaling pilot")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _print_table(results: Dict[str, Dict[str, List[float]]]) -> None:
    print("encoder       prior        H=1     H=2     H=4     H=8    H=16")
    for key in sorted(results):
        encoder, prior = key.split("|", 1)
        row = [encoder.ljust(12), prior.ljust(12)]
        for horizon in HORIZONS:
            mean, _ = _mean_std(results[key][str(horizon)])
            row.append(f"{mean:>6.2f}" if math.isfinite(mean) else "   nan")
        print(" ".join(row))


def _mean_results(results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    return {
        key: {
            str(horizon): _mean_std(horizon_values[str(horizon)])[0]
            for horizon in HORIZONS
        }
        for key, horizon_values in results.items()
    }


def _std_results(results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    return {
        key: {
            str(horizon): _mean_std(horizon_values[str(horizon)])[1]
            for horizon in HORIZONS
        }
        for key, horizon_values in results.items()
    }


def _spectral_advantage(
    mean_results: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    advantage: Dict[str, Dict[str, Dict[str, float]]] = {}
    for encoder in ["flat", "hypergraph"]:
        spectral = mean_results[f"{encoder}|spectral"]
        none = mean_results[f"{encoder}|none"]
        euclidean = mean_results[f"{encoder}|euclidean"]
        advantage[encoder] = {
            "vs_none": {
                str(horizon): none[str(horizon)] - spectral[str(horizon)]
                for horizon in HORIZONS
            },
            "vs_euclidean": {
                str(horizon): euclidean[str(horizon)] - spectral[str(horizon)]
                for horizon in HORIZONS
            },
        }
    return advantage


def _print_advantage_table(
    advantage: Dict[str, Dict[str, Dict[str, float]]]
) -> None:
    print("encoder       comparison      H=1     H=2     H=4     H=8    H=16")
    for encoder in ["flat", "hypergraph"]:
        rows = [
            ("spec-vs-none", advantage[encoder]["vs_none"]),
            ("spec-vs-eucl", advantage[encoder]["vs_euclidean"]),
        ]
        for comparison, values in rows:
            row = [encoder.ljust(12), comparison.ljust(14)]
            for horizon in HORIZONS:
                value = values[str(horizon)]
                row.append(f"{value:>6.2f}" if math.isfinite(value) else "   nan")
            print(" ".join(row))


def _build_per_seed_results(
    results: Dict[str, Dict[str, List[float]]],
    seeds: List[int],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    per_seed: Dict[str, Dict[str, Dict[str, float]]] = {
        f"seed_{seed}": {} for seed in seeds
    }
    for key, horizon_values in results.items():
        for seed_index, seed in enumerate(seeds):
            per_seed[f"seed_{seed}"][key] = {
                str(horizon): horizon_values[str(horizon)][seed_index]
                for horizon in HORIZONS
                if seed_index < len(horizon_values[str(horizon)])
            }
    return per_seed


def _save_experiment_summary(
    results: Dict[str, Dict[str, List[float]]],
    seeds: List[int],
    config: Dict[str, object],
    path: str,
) -> Dict[str, object]:
    mean_results = _mean_results(results)
    std_results = _std_results(results)
    summary = {
        "config": config,
        "per_seed_results": _build_per_seed_results(results, seeds),
        "mean_results": mean_results,
        "std_results": std_results,
        "spectral_advantage": _spectral_advantage(mean_results),
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
    return summary


def run_experiment(
    smoke: bool = False,
    seed: int = 0,
    diagnostics: bool = False,
    epochs: int | None = None,
    n_train_override: int | None = None,
    n_eval_override: int | None = None,
    batch_size_override: int | None = None,
    device_name: str = "auto",
    env_profile: str = "minimal",
) -> Dict[str, Dict[str, List[float]]]:
    global DEVICE
    selected_device = select_device(device_name)
    DEVICE = selected_device
    if diagnostics:
        print_device_diagnostics(selected_device)

    set_seed(seed)
    rule, initial_state, default_max_steps = build_environment(env_profile, seed)

    if smoke:
        n_train = 12
        n_eval = 6
        max_steps = default_max_steps
        num_epochs = 1
        seeds = [seed]
    else:
        n_train = 200
        n_eval = 50
        max_steps = default_max_steps
        num_epochs = 200
        seeds = [0, 1, 2]

    if n_train_override is not None:
        n_train = n_train_override
    if n_eval_override is not None:
        n_eval = n_eval_override
    if epochs is not None:
        num_epochs = epochs
    batch_size = batch_size_override if batch_size_override is not None else 32

    encoders = ["hypergraph", "flat"]
    priors = ["none", "euclidean", "spectral"]
    prior_weights = {
        "none": 0.0,
        "euclidean": 1e-2,
        "spectral": 1e-2,
    }
    results = {
        f"{encoder}|{prior}": {str(horizon): [] for horizon in HORIZONS}
        for encoder in encoders
        for prior in priors
    }
    data_cache: Dict[int, Tuple[List[dict], List[dict], List[dict]]] = {}

    print(f"device={selected_device}")
    for current_seed in seeds:
        train_episodes = collect_episodes(
            rule,
            initial_state,
            n_train,
            max_steps,
            seed=current_seed,
            env_profile=env_profile,
        )
        eval_episodes = collect_episodes(
            rule,
            initial_state,
            n_eval,
            max_steps,
            seed=10_000 + current_seed,
            env_profile=env_profile,
        )
        train_transitions = flatten_transitions(train_episodes)
        data_cache[current_seed] = (train_episodes, eval_episodes, train_transitions)
        if smoke or diagnostics:
            summarize_episodes(train_episodes, f"train|seed={current_seed}")
            summarize_episodes(eval_episodes, f"eval|seed={current_seed}")

        for encoder in encoders:
            for prior in priors:
                config = Config(
                    encoder=encoder,
                    prior=prior,
                    prior_weight=prior_weights[prior],
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    seed=current_seed,
                )
                _, eval_episodes_cached, train_transitions_cached = data_cache[current_seed]
                model, _ = train_one(
                    config,
                    train_transitions_cached,
                    eval_episodes_cached,
                    selected_device,
                    horizons=HORIZONS,
                )
                final_metrics = evaluate_rollout(
                    model,
                    eval_episodes_cached,
                    HORIZONS,
                    selected_device,
                )
                result_key = f"{encoder}|{prior}"
                for horizon in HORIZONS:
                    results[result_key][str(horizon)].append(final_metrics[horizon])

    out_dir = "/mnt/d/causalworld"
    _save_results(results, os.path.join(out_dir, "results.json"))
    _plot_results(results, os.path.join(out_dir, "horizon_gap.png"))
    summary_config = {
        "smoke": smoke,
        "env_profile": env_profile,
        "device": str(selected_device),
        "seeds": seeds,
        "n_train": n_train,
        "n_eval": n_eval,
        "max_steps": max_steps,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "horizons": HORIZONS,
        "encoders": encoders,
        "priors": priors,
        "prior_weights": prior_weights,
    }
    summary = _save_experiment_summary(
        results,
        seeds,
        summary_config,
        os.path.join(out_dir, "midscale_summary.json"),
    )
    _save_experiment_summary(
        results,
        seeds,
        summary_config,
        os.path.join(out_dir, "full_summary.json"),
    )
    _print_table(results)
    _print_advantage_table(summary["spectral_advantage"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-eval", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--env-profile", choices=["minimal", "branching"], default="minimal")
    args = parser.parse_args()
    run_experiment(
        smoke=args.smoke,
        seed=args.seed,
        diagnostics=args.diagnostics,
        epochs=args.epochs,
        n_train_override=args.n_train,
        n_eval_override=args.n_eval,
        batch_size_override=args.batch_size,
        device_name=args.device,
        env_profile=args.env_profile,
    )


if __name__ == "__main__":
    main()
