"""Interactive 10-seed flat validation run at 200 epochs.

Launch manually from tmux with the activated causalworld environment:
    python scripts/run_10seed_flat_200ep.py

The training loop itself is reused from train.py. This script only handles
configuration, progress formatting, partial result persistence, and plotting.
"""

from __future__ import annotations

import builtins
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch

import train
from train import Config, build_environment, collect_episodes, flatten_transitions, select_device


ENCODER = "flat"
PRIORS = ["none", "euclidean", "spectral"]
SEEDS = list(range(10))
NUM_EPOCHS = 200
MAX_STEPS = 16
HORIZONS = [1, 2, 4, 8, 16]
OUTPUT_JSON = ROOT / "validation_10seed_flat_200ep.json"
OUTPUT_PNG = ROOT / "validation_10seed_flat_200ep.png"


def load_pilot_config() -> dict[str, Any]:
    results_path = ROOT / "results.json"
    summary_path = ROOT / "full_summary.json"

    with results_path.open("r", encoding="utf-8") as file:
        results = json.load(file)
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    expected_keys = {f"{encoder}|{prior}" for encoder in ("flat", "hypergraph") for prior in PRIORS}
    missing = sorted(expected_keys.difference(results))
    if missing:
        raise RuntimeError(f"{results_path} is missing expected pilot result keys: {missing}")

    config = summary["config"]
    prior_weights = config["prior_weights"]
    if not math.isclose(float(prior_weights["euclidean"]), float(prior_weights["spectral"])):
        raise RuntimeError(f"Pilot euclidean/spectral prior weights differ: {prior_weights}")

    return {
        "prior_weight": float(prior_weights["spectral"]),
        "prior_weights": {
            "none": float(prior_weights["none"]),
            "euclidean": float(prior_weights["euclidean"]),
            "spectral": float(prior_weights["spectral"]),
        },
        "batch_size": int(config["batch_size"]),
        "latent_dim": 16,
        "hidden_dim": 32,
        "transition_hidden_dim": 128,
        "lr": 1e-3,
        "optimizer": "Adam",
        "n_train": int(config["n_train"]),
        "n_eval": int(config["n_eval"]),
        "env_profile": str(config.get("env_profile", "branching")),
        "prior_weight_source": "full_summary.json config.prior_weights; results.json confirmed pilot result keys",
    }


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is false")
    device = select_device("cuda")
    print(f"device=cuda", flush=True)
    print(f"gpu={torch.cuda.get_device_name(torch.cuda.current_device())}", flush=True)
    return device


def empty_results() -> dict[str, dict[str, list[float]]]:
    return {
        f"{ENCODER}|{prior}": {str(horizon): [] for horizon in HORIZONS}
        for prior in PRIORS
    }


def save_payload(payload: dict[str, Any]) -> None:
    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def mean_std(values: list[float]) -> tuple[float, float]:
    finite = np.array([value for value in values if math.isfinite(float(value))], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std())


def plot_results(results: dict[str, dict[str, list[float]]]) -> None:
    x = np.array(HORIZONS, dtype=float)
    plt.figure(figsize=(8, 5))
    for prior in PRIORS:
        key = f"{ENCODER}|{prior}"
        means = []
        stds = []
        for horizon in HORIZONS:
            mean, std = mean_std(results[key][str(horizon)])
            means.append(mean)
            stds.append(std)

        y = np.array(means, dtype=float)
        yerr = np.array(stds, dtype=float)
        plt.plot(x, y, marker="o", linewidth=1.8, label=prior)
        plt.fill_between(x, np.maximum(y - yerr, 1e-8), np.maximum(y + yerr, 1e-8), alpha=0.15)

    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(HORIZONS, [str(horizon) for horizon in HORIZONS])
    plt.xlabel("Rollout horizon H")
    plt.ylabel("Mean rollout error")
    plt.title("10-seed flat validation, 200 epochs")
    plt.grid(True, which="both", linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=160)
    plt.close()


def print_epoch_line_factory(config: Config, latest_eval: dict[int, float]) -> Any:
    pattern = re.compile(
        r"epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)\s+"
        r"total=(?P<total>[-+0-9.eEnNaA]+)\s+"
        r"trans=(?P<trans>[-+0-9.eEnNaA]+)\s+"
        r"prior=(?P<prior_loss>[-+0-9.eEnNaA]+)"
    )

    def patched_print(*args: Any, **kwargs: Any) -> None:
        text = " ".join(str(arg) for arg in args)
        match = pattern.search(text)
        if match is None:
            return
        h16 = latest_eval.get(16, float("nan"))
        builtins._original_print(  # type: ignore[attr-defined]
            f"[{config.encoder}|{config.prior}|seed={config.seed}] "
            f"epoch {int(match.group('epoch'))}/{match.group('total_epochs')} "
            f"total={float(match.group('total')):.3f} "
            f"prior={float(match.group('prior_loss')):.3f} "
            f"H16_eval={h16:.2f}",
            flush=True,
        )

    return patched_print


def train_with_requested_progress(
    config: Config,
    train_transitions: list[dict[str, Any]],
    eval_episodes: list[dict[str, Any]],
    device: torch.device,
) -> dict[int, float]:
    latest_eval: dict[int, float] = {}
    original_evaluate_rollout = train.evaluate_rollout
    original_print = builtins.print
    builtins._original_print = original_print  # type: ignore[attr-defined]

    def wrapped_evaluate_rollout(*args: Any, **kwargs: Any) -> dict[int, float]:
        metrics = original_evaluate_rollout(*args, **kwargs)
        latest_eval.clear()
        latest_eval.update(metrics)
        return metrics

    try:
        train.evaluate_rollout = wrapped_evaluate_rollout
        builtins.print = print_epoch_line_factory(config, latest_eval)
        model, _ = train.train_one(config, train_transitions, eval_episodes, device, horizons=HORIZONS)
    finally:
        train.evaluate_rollout = original_evaluate_rollout
        builtins.print = original_print
        delattr(builtins, "_original_print")

    return original_evaluate_rollout(model, eval_episodes, HORIZONS, device)


def main() -> None:
    device = require_cuda()
    pilot = load_pilot_config()
    start_time = time.time()

    payload: dict[str, Any] = {
        "task": "flat_10seed_200ep",
        "encoder": ENCODER,
        "max_steps": MAX_STEPS,
        "prior_weight": pilot["prior_weight"],
        "config": {
            "seeds": SEEDS,
            "priors": PRIORS,
            "horizons": HORIZONS,
            "num_epochs": NUM_EPOCHS,
            "batch_size": pilot["batch_size"],
            "latent_dim": pilot["latent_dim"],
            "hidden_dim": pilot["hidden_dim"],
            "transition_hidden_dim": pilot["transition_hidden_dim"],
            "optimizer": pilot["optimizer"],
            "lr": pilot["lr"],
            "n_train": pilot["n_train"],
            "n_eval": pilot["n_eval"],
            "env_profile": pilot["env_profile"],
            "prior_weights": pilot["prior_weights"],
            "prior_weight_source": pilot["prior_weight_source"],
        },
        "results": empty_results(),
        "wall_time_sec": 0.0,
    }
    save_payload(payload)

    for seed in SEEDS:
        rule, initial_state, _ = build_environment(pilot["env_profile"], seed)
        train_episodes = collect_episodes(
            rule,
            initial_state,
            pilot["n_train"],
            MAX_STEPS,
            seed=seed,
            env_profile=pilot["env_profile"],
        )
        eval_episodes = collect_episodes(
            rule,
            initial_state,
            pilot["n_eval"],
            MAX_STEPS,
            seed=10_000 + seed,
            env_profile=pilot["env_profile"],
        )
        train_transitions = flatten_transitions(train_episodes)

        for prior in PRIORS:
            config = Config(
                encoder=ENCODER,
                prior=prior,
                prior_weight=pilot["prior_weights"][prior],
                latent_dim=pilot["latent_dim"],
                hidden_dim=pilot["hidden_dim"],
                transition_hidden_dim=pilot["transition_hidden_dim"],
                num_epochs=NUM_EPOCHS,
                batch_size=pilot["batch_size"],
                lr=pilot["lr"],
                seed=seed,
            )
            final_metrics = train_with_requested_progress(config, train_transitions, eval_episodes, device)
            key = f"{ENCODER}|{prior}"
            for horizon in HORIZONS:
                payload["results"][key][str(horizon)].append(float(final_metrics[horizon]))

        payload["wall_time_sec"] = time.time() - start_time
        save_payload(payload)

    payload["wall_time_sec"] = time.time() - start_time
    save_payload(payload)
    plot_results(payload["results"])


if __name__ == "__main__":
    main()
