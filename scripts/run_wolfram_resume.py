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


def load_payload() -> dict[str, Any]:
    if not OUTPUT_JSON.exists():
        raise FileNotFoundError(f"Missing resume target: {OUTPUT_JSON}")
    with OUTPUT_JSON.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_payload(payload: dict[str, Any]) -> None:
    tmp_path = OUTPUT_JSON.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    tmp_path.replace(OUTPUT_JSON)


def verify_expected_startup_state(payload: dict[str, Any]) -> None:
    completed_seeds = payload.get("completed_seeds")
    flat_none_h1 = payload.get("results", {}).get("flat|none", {}).get("1")

    expected_prefix = list(range(len(completed_seeds))) if isinstance(completed_seeds, list) else None
    if (
        not isinstance(completed_seeds, list)
        or completed_seeds != expected_prefix
        or any(not isinstance(seed, int) or not 0 <= seed <= 9 for seed in completed_seeds)
    ):
        print(
            "ERROR: unexpected completed_seeds state; expected a sorted prefix of [0..9], "
            f"found {completed_seeds!r}",
            flush=True,
        )
        raise SystemExit(1)

    if not isinstance(flat_none_h1, list) or len(flat_none_h1) != len(completed_seeds):
        print(
            "ERROR: unexpected results['flat|none']['1'] length; expected "
            f"{len(completed_seeds)}, "
            f"found {None if flat_none_h1 is None else len(flat_none_h1)}",
            flush=True,
        )
        raise SystemExit(1)


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
    payload = load_payload()
    verify_expected_startup_state(payload)
    completed_seeds = payload["completed_seeds"]
    seeds = [s for s in range(10) if s not in completed_seeds]

    start_time = time.time() - float(payload.get("wall_time_sec", 0.0))

    for seed in seeds:
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
