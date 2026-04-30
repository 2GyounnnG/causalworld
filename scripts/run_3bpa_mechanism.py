from __future__ import annotations

from scripts.run_iso17_mechanism import (
    ENCODER,
    GRAPH_SOURCE,
    LAPLACIAN_MODE,
    NUM_EPOCHS,
    PRIOR_WEIGHT,
    PRIORS,
    ROOT,
    SAVE_CHECKPOINT,
    SEEDS,
    already_done,
    config_prior,
    format_float,
    format_mean_std,
    get_metric,
    nan_stats,
    save_atomic,
)

import json
import time
import traceback
from typing import Any

import torch

from scripts.train_rmd17 import Config, train_one_seed


DATASET = "3bpa"
N_TRANSITIONS = 400
STRIDE = 1
EVAL_N_TRANSITIONS = 30
DISJOINT_EVAL = False
OUTPUT_PATH = ROOT / "3bpa_mechanism.json"
SUMMARY_PATH = ROOT / "3bpa_mechanism_summary.json"


def make_key(prior: str, seed: int) -> str:
    return f"3bpa|{ENCODER}|{prior}|seed={seed}"


def load_results() -> dict[str, Any]:
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_summary(all_results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {DATASET: {}}
    horizons = [1, 2, 4, 8, 16]
    diagnostics = ["effective_rank", "condition_number", "projection_gaussianity", "spectral_alignment"]
    for prior in PRIORS:
        cell_results = [
            all_results.get(make_key(prior, seed))
            for seed in SEEDS
            if isinstance(all_results.get(make_key(prior, seed)), dict)
            and "error" not in all_results.get(make_key(prior, seed), {})
            and already_done(all_results.get(make_key(prior, seed)))
        ]
        rollout_summary = {
            str(horizon): nan_stats([get_metric(result.get("rollout_errors", {}), horizon) for result in cell_results])
            for horizon in horizons
        }
        diagnostic_summary = {}
        for diagnostic in diagnostics:
            stats = nan_stats([get_metric(result.get("final_diagnostics", {}), diagnostic) for result in cell_results])
            diagnostic_summary[diagnostic] = {"mean": stats["mean"], "std": stats["std"]}
        summary[DATASET][prior] = {
            "n_success": len(cell_results),
            "rollout_errors": rollout_summary,
            "diagnostics": diagnostic_summary,
        }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== Summary by dataset and prior ===", flush=True)
    print("dataset | prior | n | H=1 | H=8 | H=16 | r_eff | spec_align", flush=True)
    for prior in PRIORS:
        cell_summary = summary[DATASET][prior]
        rollouts = cell_summary["rollout_errors"]
        diagnostics = cell_summary["diagnostics"]
        print(
            f"{DATASET} | {prior} | {cell_summary['n_success']} | "
            f"{format_mean_std(rollouts['1'])} | {format_mean_std(rollouts['8'])} | "
            f"{format_mean_std(rollouts['16'])} | "
            f"{format_float(diagnostics['effective_rank']['mean'], 2)} | "
            f"{format_float(diagnostics['spectral_alignment']['mean'], 2)}",
            flush=True,
        )


def main() -> None:
    script_start_time = time.time()
    tasks = [(prior, seed) for prior in PRIORS for seed in SEEDS]
    all_results = load_results()
    pending_tasks = []
    for prior, seed in tasks:
        key = make_key(prior, seed)
        if already_done(all_results.get(key)):
            print(f"[SKIP] {key}", flush=True)
        else:
            pending_tasks.append((prior, seed))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"device=cuda ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print("WARNING: CUDA unavailable, falling back to CPU", flush=True)
    print(f"Resume: {len(pending_tasks)}/{len(tasks)} runs pending", flush=True)
    print(f"output: {OUTPUT_PATH}", flush=True)

    for i, (prior, seed) in enumerate(pending_tasks, start=1):
        key = make_key(prior, seed)
        print(f"\n=== Starting {key} ({i}/{len(pending_tasks)}) ===", flush=True)
        t0 = time.time()
        config = Config(
            molecule=DATASET,
            encoder=ENCODER,
            prior=config_prior(prior),
            graph_source=GRAPH_SOURCE,
            laplacian_mode=LAPLACIAN_MODE,
            seed=seed,
            num_epochs=NUM_EPOCHS,
            n_transitions=N_TRANSITIONS,
            stride=STRIDE,
            eval_n_transitions=EVAL_N_TRANSITIONS,
            prior_weight=PRIOR_WEIGHT,
            save_checkpoint=SAVE_CHECKPOINT,
            disjoint_eval=DISJOINT_EVAL,
            device=device,
        )
        try:
            result = train_one_seed(config)
            if not isinstance(result.get("config"), dict):
                result["config"] = {}
            result["config"]["prior"] = prior
            result["user_facing_prior"] = prior
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
            all_results[key] = {"error": str(exc), "traceback": tb, "user_facing_prior": prior, "wall_time_sec": time.time() - t0}
            print(f"=== FAILED {key}: {exc} ===", flush=True)
            print(tb, flush=True)

        save_atomic(all_results, OUTPUT_PATH)
        avg = (time.time() - script_start_time) / max(i, 1)
        print(f"  -> saved partial; avg={avg:.0f}s/run; ETA={(avg * (len(pending_tasks) - i)) / 60:.0f}min", flush=True)

    summary = build_summary(all_results)
    save_atomic(summary, SUMMARY_PATH)
    print_summary(summary)
    print(f"\nWrote {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
