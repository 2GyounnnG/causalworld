from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rmd17 import Config, train_one_seed


DATASET = "lj_N256"
PRIOR = "spectral"
WEIGHTS = [0.001, 0.01, 0.1, 1.0]
SEEDS = list(range(5))
NUM_EPOCHS = 50
N_TRANSITIONS = 800
STRIDE = 1
EVAL_N_TRANSITIONS = 80
ENCODER = "flat"
GRAPH_SOURCE = "bond"
LAPLACIAN_MODE = "per_frame"
SAVE_CHECKPOINT = True
OUTPUT_PATH = ROOT / "lj_weight_sweep.json"


def make_key(weight: float, seed: int) -> str:
    return f"lj_N256|{ENCODER}|spectral|w={weight}|seed={seed}"


def save_atomic(all_results: dict[str, Any]) -> None:
    tmp = OUTPUT_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2, default=str)
    os.replace(tmp, OUTPUT_PATH)


def load_results() -> dict[str, Any]:
    if not OUTPUT_PATH.exists():
        return {}
    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def finite_value(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def already_done(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    rollout_errors = result.get("rollout_errors")
    if not isinstance(rollout_errors, dict):
        return False
    return any(finite_value(value) for value in rollout_errors.values())


def get_metric(mapping: dict[str, Any], key: int | str) -> float:
    value = mapping.get(key, mapping.get(str(key), float("nan")))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_values(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def mean_std(values: list[float]) -> tuple[float, float]:
    array = finite_values(values)
    if array.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return mean, std


def format_float(value: Any, digits: int) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(numeric):
        return "nan"
    return f"{numeric:.{digits}f}"


def build_summary(all_results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for weight in WEIGHTS:
        successful = [
            all_results.get(make_key(weight, seed))
            for seed in SEEDS
            if isinstance(all_results.get(make_key(weight, seed)), dict)
            and "error" not in all_results.get(make_key(weight, seed), {})
            and already_done(all_results.get(make_key(weight, seed)))
        ]
        weight_summary: dict[str, Any] = {
            "prior": PRIOR,
            "weight": weight,
            "n_success": len(successful),
            "rollout_errors": {},
            "diagnostics": {},
        }
        for horizon in [1, 8, 16]:
            mean, std = mean_std([get_metric(result.get("rollout_errors", {}), horizon) for result in successful])
            weight_summary["rollout_errors"][str(horizon)] = {"mean": mean, "std": std}
        for diagnostic in ["effective_rank", "condition_number", "projection_gaussianity", "spectral_alignment"]:
            mean, std = mean_std([get_metric(result.get("final_diagnostics", {}), diagnostic) for result in successful])
            weight_summary["diagnostics"][diagnostic] = {"mean": mean, "std": std}
        summary[str(weight)] = weight_summary
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== Summary ===", flush=True)
    print(
        f"{'weight':>8s} {'n':>3s} {'H=1 mean+-std':>18s} "
        f"{'H=8 mean+-std':>18s} {'H=16 mean+-std':>18s} {'r_eff':>8s} {'cond':>10s}",
        flush=True,
    )
    for weight in WEIGHTS:
        weight_summary = summary[str(weight)]
        rollouts = weight_summary["rollout_errors"]
        diagnostics = weight_summary["diagnostics"]
        print(
            f"{format_float(weight, 3):>8s} {weight_summary['n_success']:3d} "
            f"{format_float(rollouts['1']['mean'], 4)}+-{format_float(rollouts['1']['std'], 4):>8s} "
            f"{format_float(rollouts['8']['mean'], 4)}+-{format_float(rollouts['8']['std'], 4):>8s} "
            f"{format_float(rollouts['16']['mean'], 4)}+-{format_float(rollouts['16']['std'], 4):>8s} "
            f"{format_float(diagnostics['effective_rank']['mean'], 2):>8s} "
            f"{format_float(diagnostics['condition_number']['mean'], 2):>10s}",
            flush=True,
        )


def main() -> None:
    all_results = load_results()
    pending_pairs = []
    for weight in WEIGHTS:
        for seed in SEEDS:
            key = make_key(weight, seed)
            if already_done(all_results.get(key)):
                print(f"[SKIP] {key}", flush=True)
            else:
                pending_pairs.append((weight, seed))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"device=cuda ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print("WARNING: CUDA unavailable, falling back to CPU", flush=True)
    print(f"Resume: {len(pending_pairs)}/{len(WEIGHTS) * len(SEEDS)} runs pending", flush=True)
    print(f"output: {OUTPUT_PATH}", flush=True)

    for i, (weight, seed) in enumerate(pending_pairs, start=1):
        key = make_key(weight, seed)
        print(f"\n=== Starting {key} ({i}/{len(pending_pairs)}) ===", flush=True)
        t0 = time.time()
        config = Config(
            molecule=DATASET,
            encoder=ENCODER,
            prior=PRIOR,
            graph_source=GRAPH_SOURCE,
            laplacian_mode=LAPLACIAN_MODE,
            seed=seed,
            num_epochs=NUM_EPOCHS,
            n_transitions=N_TRANSITIONS,
            stride=STRIDE,
            eval_n_transitions=EVAL_N_TRANSITIONS,
            prior_weight=weight,
            save_checkpoint=SAVE_CHECKPOINT,
            device=device,
        )
        try:
            result = train_one_seed(config)
            result["user_facing_prior"] = PRIOR
            result["wall_time_sec"] = time.time() - t0
            all_results[key] = result
            rollout = result.get("rollout_errors", {})
            diag = result.get("final_diagnostics", {})
            print(
                f"=== Finished {key} | H1={get_metric(rollout, 1):.4f} "
                f"H8={get_metric(rollout, 8):.4f} H16={get_metric(rollout, 16):.4f} | "
                f"r_eff={get_metric(diag, 'effective_rank'):.2f} | wall={time.time() - t0:.1f}s ===",
                flush=True,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            all_results[key] = {"error": str(exc), "traceback": tb, "user_facing_prior": PRIOR, "wall_time_sec": time.time() - t0}
            print(f"=== FAILED {key}: {exc} ===", flush=True)
            print(tb, flush=True)
        save_atomic(all_results)
        print(f"  -> saved partial results to {OUTPUT_PATH.name}", flush=True)

    print_summary(build_summary(all_results))


if __name__ == "__main__":
    main()
