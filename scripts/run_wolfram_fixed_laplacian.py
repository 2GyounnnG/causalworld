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


TASK = "wolfram_fixed_laplacian_ablation"
ENCODER = "flat"
PRIOR = "spectral"
PRIOR_WEIGHT = 0.01
LAPLACIAN_MODES = ["per_step", "fixed_initial", "fixed_average"]
SEEDS = [0, 1, 2, 3, 4]
NUM_EPOCHS = 200
MAX_STEPS = 16
HORIZONS = [1, 2, 4, 8, 16]
ENV_PROFILE = "minimal"
N_TRAIN = 200
N_EVAL = 50
OUTPUT_JSON = ROOT / "validation_wolfram_fixed_laplacian.json"


def make_key(mode: str, seed: int) -> str:
    return f"{ENCODER}|{PRIOR}|mode={mode}|seed={seed}"


def make_payload() -> dict[str, Any]:
    return {
        "task": TASK,
        "modes": LAPLACIAN_MODES,
        "seeds": SEEDS,
        "num_epochs": NUM_EPOCHS,
        "results": {},
    }


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
    payload = make_payload()
    save_payload(payload)

    for mode in LAPLACIAN_MODES:
        for seed in SEEDS:
            train_transitions, eval_episodes = build_datasets(seed)
            config = Config(
                encoder=ENCODER,
                prior=PRIOR,
                prior_weight=PRIOR_WEIGHT,
                num_epochs=NUM_EPOCHS,
                seed=seed,
                laplacian_mode=mode,
            )

            start_time = time.time()
            rollout_errors = train_with_sparse_progress(
                config,
                train_transitions,
                eval_episodes,
                device,
            )
            wall_time_sec = time.time() - start_time

            key = make_key(mode, seed)
            payload["results"][key] = {
                "rollout_errors": {
                    str(horizon): rollout_errors[horizon]
                    for horizon in HORIZONS
                },
                "wall_time_sec": wall_time_sec,
            }
            save_payload(payload)
            print(f"[mode={mode}|seed={seed}] H16={rollout_errors[16]:.3f}", flush=True)


if __name__ == "__main__":
    main()
