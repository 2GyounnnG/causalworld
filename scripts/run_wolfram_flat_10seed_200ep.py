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

import train
from train import (
    Config,
    build_environment,
    collect_episodes,
    evaluate_rollout,
    flatten_transitions,
    select_device,
    train_one,
)


ENCODER = "flat"
PRIORS = ["none", "euclidean", "spectral"]
SEEDS = list(range(10))
NUM_EPOCHS = 200
MAX_STEPS = 16
HORIZONS = [1, 2, 4, 8, 16]
N_TRAIN = 200
N_EVAL = 50
ENV_PROFILE = "minimal"
OUTPUT_JSON = ROOT / "validation_wolfram_flat_10seed_200ep.json"


def load_pilot_prior_weight() -> float:
    results_path = ROOT / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    expected_keys = {f"{ENCODER}|{prior}" for prior in PRIORS}
    missing = sorted(expected_keys.difference(results.keys()))
    if missing:
        raise RuntimeError(f"{results_path} is missing expected pilot keys: {missing}")

    train_source = (ROOT / "train.py").read_text(encoding="utf-8")
    match = re.search(
        r'prior_weights\s*=\s*\{\s*"none":\s*([0-9.eE+-]+),\s*"euclidean":\s*([0-9.eE+-]+),\s*"spectral":\s*([0-9.eE+-]+),\s*\}',
        train_source,
        re.DOTALL,
    )
    if match is None:
        raise RuntimeError("Could not locate pilot prior_weights in train.py")

    euclidean_weight = float(match.group(2))
    spectral_weight = float(match.group(3))
    if euclidean_weight != spectral_weight:
        raise RuntimeError(
            f"Pilot euclidean/spectral prior weights differ in train.py: {euclidean_weight} vs {spectral_weight}"
        )
    return euclidean_weight


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run, but torch.cuda.is_available() is false")
    device = select_device("cuda")
    print(f"device={device}")
    print(f"gpu={torch.cuda.get_device_name(torch.cuda.current_device())}")
    return device


def empty_results() -> dict[str, dict[str, list[float]]]:
    return {
        f"{ENCODER}|{prior}": {str(horizon): [] for horizon in HORIZONS}
        for prior in PRIORS
    }


def save_payload(payload: dict[str, Any]) -> None:
    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def print_every_20_epochs_factory() -> Any:
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
            return

        epoch = int(match.group("epoch"))
        if epoch % 20 != 0:
            return
        builtins._original_print(text, flush=True)  # type: ignore[attr-defined]

    return patched_print


def train_with_sparse_progress(
    config: Config,
    train_transitions: list[dict[str, Any]],
    eval_episodes: list[dict[str, Any]],
    device: torch.device,
) -> dict[int, float]:
    original_print = builtins.print
    builtins._original_print = original_print  # type: ignore[attr-defined]

    try:
        builtins.print = print_every_20_epochs_factory()
        model, _ = train_one(
            config,
            train_transitions,
            eval_episodes,
            device,
            horizons=HORIZONS,
        )
    finally:
        builtins.print = original_print
        delattr(builtins, "_original_print")

    return evaluate_rollout(model, eval_episodes, HORIZONS, device)


def main() -> None:
    device = require_cuda()
    prior_weight = load_pilot_prior_weight()
    start_time = time.time()

    payload: dict[str, Any] = {
        "task": "wolfram_flat_10seed_200ep",
        "encoder": ENCODER,
        "max_steps": MAX_STEPS,
        "prior_weight": prior_weight,
        "config": {
            "seeds": SEEDS,
            "priors": PRIORS,
            "horizons": HORIZONS,
            "num_epochs": NUM_EPOCHS,
            "n_train": N_TRAIN,
            "n_eval": N_EVAL,
            "max_steps": MAX_STEPS,
            "env_profile": ENV_PROFILE,
            "prior_weights": {
                "none": 0.0,
                "euclidean": prior_weight,
                "spectral": prior_weight,
            },
        },
        "results": empty_results(),
        "completed_seeds": [],
        "wall_time_sec": 0.0,
    }
    save_payload(payload)

    for seed in SEEDS:
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
        train_transitions = flatten_transitions(train_episodes)

        for prior in PRIORS:
            config = Config(
                encoder=ENCODER,
                prior=prior,
                prior_weight=0.0 if prior == "none" else prior_weight,
                num_epochs=NUM_EPOCHS,
                seed=seed,
            )
            final_metrics = train_with_sparse_progress(
                config,
                train_transitions,
                eval_episodes,
                device,
            )
            result_key = f"{ENCODER}|{prior}"
            for horizon in HORIZONS:
                payload["results"][result_key][str(horizon)].append(final_metrics[horizon])

        payload["completed_seeds"].append(seed)
        payload["wall_time_sec"] = time.time() - start_time
        save_payload(payload)
        print(f"[seed={seed}] saved partial results to {OUTPUT_JSON.name}", flush=True)


if __name__ == "__main__":
    main()
