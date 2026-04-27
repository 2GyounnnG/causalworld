from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import (
    Config,
    build_environment,
    collect_episodes,
    evaluate_rollout,
    flatten_transitions,
    select_device,
    train_one,
)


TASK = "wolfram_clean_mechanism"
ENVIRONMENT = "wolfram"
ENCODER = "flat"
PRIORS = ["none", "spectral"]
PRIOR_WEIGHTS = [0.001, 0.005, 0.01]
LAPLACIAN_MODES = ["per_step", "fixed_initial", "fixed_average"]
SEEDS = [0, 1, 2, 3, 4]
NUM_EPOCHS = 200
MAX_STEPS = 16
HORIZONS = [1, 2, 4, 8, 16]
ENV_PROFILE = "minimal"
N_TRAIN = 200
N_EVAL = 50
OUTPUT_JSON = ROOT / "validation_wolfram_clean_mechanism.json"


def make_key(prior: str, weight: float, laplacian_mode: str, seed: int) -> str:
    return (
        f"{ENVIRONMENT}|{ENCODER}|{prior}|"
        f"w={weight}|mode={laplacian_mode}|seed={seed}"
    )


def experiment_grid() -> list[tuple[str, float, str, int]]:
    runs: list[tuple[str, float, str, int]] = []
    for seed in SEEDS:
        runs.append(("none", 0.0, "per_step", seed))

    for weight in PRIOR_WEIGHTS:
        for laplacian_mode in LAPLACIAN_MODES:
            for seed in SEEDS:
                runs.append(("spectral", weight, laplacian_mode, seed))

    return runs


def make_payload() -> dict[str, Any]:
    return {
        "task": TASK,
        "environment": ENVIRONMENT,
        "encoder": ENCODER,
        "priors": PRIORS,
        "weights": PRIOR_WEIGHTS,
        "laplacian_modes": LAPLACIAN_MODES,
        "seeds": SEEDS,
        "num_epochs": NUM_EPOCHS,
        "horizons": HORIZONS,
        "results": {},
    }


def load_payload() -> dict[str, Any]:
    if not OUTPUT_JSON.exists():
        return make_payload()
    with OUTPUT_JSON.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    payload.setdefault("results", {})
    return payload


def save_payload(payload: dict[str, Any]) -> None:
    tmp_path = OUTPUT_JSON.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    tmp_path.replace(OUTPUT_JSON)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run, but torch.cuda.is_available() is false")
    device = select_device("cuda")
    print(f"device={device}", flush=True)
    print(f"gpu={torch.cuda.get_device_name(torch.cuda.current_device())}", flush=True)
    return device


def print_every_5_epochs_factory(original_print: Any) -> Any:
    pattern = re.compile(
        r"\[(?P<label>[^\]]+)\]\s+epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)\s+"
        r"total=(?P<total>[-+0-9.eEnNaA]+)\s+"
        r"trans=(?P<trans>[-+0-9.eEnNaA]+)\s+"
        r"prior=(?P<prior_loss>[-+0-9.eEnNaA]+)\s+"
        r"rollout_H8=(?P<h8>[-+0-9.eEnNaA]+)"
    )

    def patched_print(*args: Any, **kwargs: Any) -> None:
        text = " ".join(str(arg) for arg in args)
        match = pattern.search(text)
        if match is None:
            original_print(*args, **kwargs)
            return

        epoch = int(match.group("epoch"))
        if epoch % 5 == 0:
            kwargs["flush"] = True
            original_print(*args, **kwargs)

    return patched_print


def train_with_sparse_progress(
    config: Config,
    train_transitions: list[dict[str, Any]],
    eval_episodes: list[dict[str, Any]],
    device: torch.device,
) -> dict[int, float]:
    original_print = builtins.print
    builtins.print = print_every_5_epochs_factory(original_print)
    try:
        model, _ = train_one(
            config,
            train_transitions,
            eval_episodes,
            device,
            horizons=HORIZONS,
        )
    finally:
        builtins.print = original_print

    return evaluate_rollout(model, eval_episodes, HORIZONS, device)


def build_datasets(seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rule, initial_state, default_max_steps = build_environment(ENV_PROFILE, seed)
    if default_max_steps != MAX_STEPS:
        raise RuntimeError(
            f"Expected Wolfram max_steps={MAX_STEPS}, but build_environment returned {default_max_steps}"
        )

    train_episodes = collect_episodes(
        rule,
        initial_state,
        N_TRAIN,
        MAX_STEPS,
        seed=seed,
        env_profile=ENV_PROFILE,
    )
    eval_episodes = collect_episodes(
        rule,
        initial_state,
        N_EVAL,
        MAX_STEPS,
        seed=10_000 + seed,
        env_profile=ENV_PROFILE,
    )
    return flatten_transitions(train_episodes), eval_episodes


def main() -> None:
    device = require_cuda()
    payload = load_payload()
    datasets_by_seed: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}

    for prior, weight, laplacian_mode, seed in experiment_grid():
        key = make_key(prior, weight, laplacian_mode, seed)
        if key in payload["results"]:
            print(f"SKIP existing {key}", flush=True)
            continue

        if seed not in datasets_by_seed:
            datasets_by_seed[seed] = build_datasets(seed)
        train_transitions, eval_episodes = datasets_by_seed[seed]

        config = Config(
            encoder=ENCODER,
            prior=prior,
            prior_weight=weight,
            num_epochs=NUM_EPOCHS,
            seed=seed,
            laplacian_mode=laplacian_mode,
        )

        print(f"\n=== Starting {key} ===", flush=True)
        start_time = time.time()
        rollout_errors = train_with_sparse_progress(
            config,
            train_transitions,
            eval_episodes,
            device,
        )
        wall_time_sec = time.time() - start_time

        payload["results"][key] = {
            "rollout_errors": {
                str(horizon): rollout_errors[horizon]
                for horizon in HORIZONS
            },
            "wall_time_sec": wall_time_sec,
        }
        save_payload(payload)
        print(f"[{key}] H16={rollout_errors[16]:.3f}", flush=True)


if __name__ == "__main__":
    main()
