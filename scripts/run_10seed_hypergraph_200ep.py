"""Interactive 10-seed hypergraph validation run at 200 epochs."""

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


ENCODER = "hypergraph"
PRINT_ENCODER = "hyper"
HIDDEN_DIM = 32
PRIORS = ["none", "euclidean", "spectral"]
SEEDS = list(range(10))
NUM_EPOCHS = 200
MAX_STEPS = 16
HORIZONS = [1, 2, 4, 8, 16]
OUTPUT_JSON = ROOT / "validation_10seed_hypergraph_200ep.json"
OUTPUT_PNG = ROOT / "validation_10seed_hypergraph_200ep.png"


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
        "hidden_dim": HIDDEN_DIM,
        "transition_hidden_dim": 128,
        "lr": 1e-3,
        "optimizer": "Adam",
        "n_train": int(config["n_train"]),
        "n_eval": int(config["n_eval"]),
        "env_profile": str(config.get("env_profile", "branching")),
        "prior_weight_source": "results.json confirmed pilot keys; full_summary.json supplied prior_weights",
    }


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is false")
    device = select_device("cuda")
    print("device=cuda", flush=True)
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
        means: list[float] = []
        stds: list[float] = []
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
    plt.title("10-seed hypergraph validation, 200 epochs")
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
            f"[{PRINT_ENCODER}|{config.prior}|seed={config.seed}] "
            f"epoch {int(match.group('epoch'))}/{match.group('total_epochs')} "
            f"total={float(match.group('total')):.2f} "
            f"prior={float(match.group('prior_loss')):.2f} "
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


def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda out of memory" in text


def run_seed(
    seed: int,
    device: torch.device,
    pilot: dict[str, Any],
    batch_size: int,
) -> dict[str, dict[str, float]]:
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
    seed_results: dict[str, dict[str, float]] = {}

    for prior in PRIORS:
        config = Config(
            encoder=ENCODER,
            prior=prior,
            prior_weight=pilot["prior_weights"][prior],
            latent_dim=pilot["latent_dim"],
            hidden_dim=HIDDEN_DIM,
            transition_hidden_dim=pilot["transition_hidden_dim"],
            num_epochs=NUM_EPOCHS,
            batch_size=batch_size,
            lr=pilot["lr"],
            seed=seed,
        )
        final_metrics = train_with_requested_progress(config, train_transitions, eval_episodes, device)
        seed_results[prior] = {str(horizon): float(final_metrics[horizon]) for horizon in HORIZONS}

    return seed_results


def main() -> None:
    device = require_cuda()
    pilot = load_pilot_config()
    start_time = time.time()

    payload: dict[str, Any] = {
        "task": "hypergraph_10seed_200ep",
        "encoder": ENCODER,
        "hidden_dim": HIDDEN_DIM,
        "max_steps": MAX_STEPS,
        "prior_weight": pilot["prior_weight"],
        "config": {
            "seeds": SEEDS,
            "priors": PRIORS,
            "horizons": HORIZONS,
            "num_epochs": NUM_EPOCHS,
            "batch_size": pilot["batch_size"],
            "latent_dim": pilot["latent_dim"],
            "hidden_dim": HIDDEN_DIM,
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
        batch_size = pilot["batch_size"]
        retried = False
        while True:
            try:
                seed_results = run_seed(seed, device, pilot, batch_size)
                break
            except RuntimeError as exc:
                if not is_oom(exc) or retried:
                    raise
                retried = True
                batch_size = max(1, batch_size // 2)
                print(
                    f"[{PRINT_ENCODER}|seed={seed}] OOM encountered, retrying seed once with batch_size={batch_size}",
                    flush=True,
                )
                torch.cuda.empty_cache()

        for prior in PRIORS:
            key = f"{ENCODER}|{prior}"
            for horizon in HORIZONS:
                payload["results"][key][str(horizon)].append(seed_results[prior][str(horizon)])

        payload["wall_time_sec"] = time.time() - start_time
        save_payload(payload)
        torch.cuda.empty_cache()

    payload["wall_time_sec"] = time.time() - start_time
    save_payload(payload)
    plot_results(payload["results"])


if __name__ == "__main__":
    main()
