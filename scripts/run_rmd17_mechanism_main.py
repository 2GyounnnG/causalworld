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


PRIORS = [
    "none",
    "variance",
    "covariance",
    "sigreg",
    "spectral",
    "permuted_spectral",
    "random_spectral",
]
MOLECULES = ["aspirin"]
SEEDS = list(range(5))
NUM_EPOCHS = 50
N_TRANSITIONS = 2000
PRIOR_WEIGHT = 0.1
ENCODER = "flat"
GRAPH_SOURCE = "bond"
LAPLACIAN_MODE = "per_frame"
SAVE_CHECKPOINT = True
DISJOINT_EVAL = False
OUTPUT_PATH = ROOT / "rmd17_mechanism_main.json"
SUMMARY_PATH = ROOT / "rmd17_mechanism_main_summary.json"


def config_prior(prior: str) -> str:
    return "euclidean" if prior == "covariance" else prior


def make_key(molecule: str, prior: str, seed: int) -> str:
    return f"{molecule}|{ENCODER}|{prior}|seed={seed}"


def save_atomic(all_results: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    os.replace(tmp, path)


def load_results() -> dict[str, Any]:
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def finite_rollout_value(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def already_done(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if "rollout_errors" not in result:
        return False
    rollout_errors = result["rollout_errors"]
    if not isinstance(rollout_errors, dict):
        return False
    return any(finite_rollout_value(value) for value in rollout_errors.values())


def get_metric(mapping: dict[str, Any], key: int | str, default: float = float("nan")) -> float:
    value = mapping.get(key, mapping.get(str(key), default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def nan_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if n > 1 else 0.0
    ci = 1.96 * std / math.sqrt(n)
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - ci,
        "ci_high": mean + ci,
    }


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
    horizons = [1, 2, 4, 8, 16]
    diagnostics = [
        "effective_rank",
        "condition_number",
        "projection_gaussianity",
        "spectral_alignment",
    ]

    for prior in PRIORS:
        prior_results = []
        for molecule in MOLECULES:
            for seed in SEEDS:
                key = make_key(molecule, prior, seed)
                result = all_results.get(key)
                if not isinstance(result, dict) or "error" in result:
                    continue
                if already_done(result):
                    prior_results.append(result)

        rollout_summary = {}
        for horizon in horizons:
            values = [
                get_metric(result.get("rollout_errors", {}), horizon)
                for result in prior_results
            ]
            rollout_summary[str(horizon)] = nan_stats(values)

        diagnostic_summary = {}
        for diagnostic in diagnostics:
            values = [
                get_metric(result.get("final_diagnostics", {}), diagnostic)
                for result in prior_results
            ]
            stats = nan_stats(values)
            diagnostic_summary[diagnostic] = {
                "mean": stats["mean"],
                "std": stats["std"],
            }

        summary[prior] = {
            "n_success": len(prior_results),
            "rollout_errors": rollout_summary,
            "diagnostics": diagnostic_summary,
        }

    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== Summary by prior ===", flush=True)
    print(
        "prior | n | H=1_mean | H=8_mean | H=16_mean | r_eff | cond | proj_g | spec_a",
        flush=True,
    )
    for prior in PRIORS:
        prior_summary = summary[prior]
        rollouts = prior_summary["rollout_errors"]
        diagnostics = prior_summary["diagnostics"]
        print(
            f"{prior} | "
            f"{prior_summary['n_success']} | "
            f"{format_float(rollouts['1']['mean'], 4)} | "
            f"{format_float(rollouts['8']['mean'], 4)} | "
            f"{format_float(rollouts['16']['mean'], 4)} | "
            f"{format_float(diagnostics['effective_rank']['mean'], 2)} | "
            f"{format_float(diagnostics['condition_number']['mean'], 2)} | "
            f"{format_float(diagnostics['projection_gaussianity']['mean'], 2)} | "
            f"{format_float(diagnostics['spectral_alignment']['mean'], 2)}",
            flush=True,
        )


def main() -> None:
    script_start_time = time.time()
    tasks = [
        (molecule, prior, seed)
        for molecule in MOLECULES
        for prior in PRIORS
        for seed in SEEDS
    ]
    all_results = load_results()

    pending_tasks = []
    for molecule, prior, seed in tasks:
        key = make_key(molecule, prior, seed)
        if already_done(all_results.get(key)):
            print(f"[SKIP] {key}", flush=True)
        else:
            pending_tasks.append((molecule, prior, seed))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"device=cuda ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print("WARNING: CUDA unavailable, falling back to CPU", flush=True)
    print(f"Resume: {len(pending_tasks)}/{len(tasks)} runs pending", flush=True)
    print(f"output: {OUTPUT_PATH}", flush=True)

    for i, (molecule, prior, seed) in enumerate(pending_tasks, start=1):
        key = make_key(molecule, prior, seed)
        print(f"\n=== Starting {key} ({i}/{len(pending_tasks)}) ===", flush=True)
        t0 = time.time()
        config = Config(
            molecule=molecule,
            encoder=ENCODER,
            prior=config_prior(prior),
            graph_source=GRAPH_SOURCE,
            laplacian_mode=LAPLACIAN_MODE,
            seed=seed,
            num_epochs=NUM_EPOCHS,
            n_transitions=N_TRANSITIONS,
            prior_weight=PRIOR_WEIGHT,
            save_checkpoint=SAVE_CHECKPOINT,
            disjoint_eval=DISJOINT_EVAL,
        )
        if device == "cpu":
            config.device = "cpu"

        try:
            result = train_one_seed(config)
            if "config" in result:
                result["config"]["prior"] = prior
            result["user_facing_prior"] = prior
            result["wall_time_sec"] = time.time() - t0
            all_results[key] = result

            rollout = result.get("rollout_errors", {})
            diag = result.get("final_diagnostics", {})
            print(
                f"=== Finished {key} | "
                f"H1={get_metric(rollout, 1):.4f} "
                f"H8={get_metric(rollout, 8):.4f} "
                f"H16={get_metric(rollout, 16):.4f} | "
                f"r_eff={get_metric(diag, 'effective_rank'):.2f} | "
                f"wall={time.time() - t0:.1f}s ===",
                flush=True,
            )
        except Exception as exc:
            tb = traceback.format_exc()
            all_results[key] = {
                "error": str(exc),
                "traceback": tb,
                "user_facing_prior": prior,
                "wall_time_sec": time.time() - t0,
            }
            print(f"=== FAILED {key}: {exc} ===", flush=True)
            print(tb, flush=True)

        save_atomic(all_results, OUTPUT_PATH)

        completed_so_far = len(
            [
                k
                for k, v in all_results.items()
                if isinstance(k, str) and isinstance(v, dict) and "rollout_errors" in v
            ]
        )
        avg = (time.time() - script_start_time) / max(completed_so_far, 1)
        remaining = len(pending_tasks) - i
        eta_sec = avg * remaining
        eta_min = eta_sec / 60
        print(
            f"  -> saved partial; avg={avg:.0f}s/run; "
            f"ETA={eta_min:.0f}min for {remaining} remaining",
            flush=True,
        )

    summary = build_summary(all_results)
    save_atomic(summary, SUMMARY_PATH)
    print_summary(summary)
    print(f"\nWrote {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
