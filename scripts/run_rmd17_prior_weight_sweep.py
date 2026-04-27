from __future__ import annotations

import builtins
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rmd17 import Config, train_one_seed


EXPERIMENT_NAME = "rmd17_aspirin_prior_weight_sweep_full"
MOLECULE = "aspirin"
ENCODER = "flat"
PRIOR_VARIANTS = [("none", "bond"), ("spectral", "bond")]
PRIOR_WEIGHTS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
SEEDS = [0, 1, 2, 3, 4]
NUM_EPOCHS = 50
HORIZONS = [1, 2, 4, 8, 16]
DISJOINT_EVAL = True
OUTPUT_PATH = ROOT / "rmd17_aspirin_prior_weight_sweep_full.json"
CHECKPOINT_DIR = "checkpoints/rmd17_aspirin_prior_weight_sweep_full"


def make_payload() -> dict[str, Any]:
    return {
        "experiment_name": EXPERIMENT_NAME,
        "molecule": MOLECULE,
        "encoder": ENCODER,
        "prior_variants": PRIOR_VARIANTS,
        "prior_weights": PRIOR_WEIGHTS,
        "seeds": SEEDS,
        "epochs": NUM_EPOCHS,
        "horizons": HORIZONS,
        "strict_disjoint_eval": DISJOINT_EVAL,
        "checkpoint_dir": CHECKPOINT_DIR,
        "results": {},
    }


def load_payload() -> dict[str, Any]:
    if not OUTPUT_PATH.exists():
        return make_payload()
    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    payload.setdefault("results", {})
    return payload


def save_payload(payload: dict[str, Any]) -> None:
    tmp_path = OUTPUT_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)
    tmp_path.replace(OUTPUT_PATH)


def make_key(prior: str, prior_weight: float, seed: int) -> str:
    return f"{MOLECULE}|{ENCODER}|{prior}|w={prior_weight}|seed={seed}"


def experiment_grid() -> list[tuple[str, str, float, int]]:
    runs: list[tuple[str, str, float, int]] = []
    for prior, graph_source in PRIOR_VARIANTS:
        weights = [0.0] if prior == "none" else PRIOR_WEIGHTS
        for weight in weights:
            for seed in SEEDS:
                runs.append((prior, graph_source, weight, seed))
    return runs


def is_complete(result: Any) -> bool:
    return (
        isinstance(result, dict)
        and result.get("status", "ok") == "ok"
        and isinstance(result.get("rollout_errors"), dict)
    )


def print_every_5_epochs_factory(original_print: Any) -> Any:
    pattern = re.compile(r"\[[^\]]+\]\s+epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)")

    def patched_print(*args: Any, **kwargs: Any) -> None:
        text = " ".join(str(arg) for arg in args)
        match = pattern.search(text)
        if match is None or int(match.group("epoch")) % 5 == 0:
            kwargs["flush"] = True
            original_print(*args, **kwargs)

    return patched_print


def train_with_sparse_progress(config: Config) -> dict[str, Any]:
    original_print = builtins.print
    builtins.print = print_every_5_epochs_factory(original_print)
    try:
        return train_one_seed(config)
    finally:
        builtins.print = original_print


def h16_value(result: dict[str, Any]) -> float:
    rollout_errors = result.get("rollout_errors", {})
    return float(rollout_errors.get(16, rollout_errors.get("16", float("nan"))))


def main() -> None:
    payload = load_payload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"device=cuda ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print("WARNING: CUDA unavailable, falling back to CPU", flush=True)

    for prior, graph_source, prior_weight, seed in experiment_grid():
        key = make_key(prior, prior_weight, seed)
        if is_complete(payload["results"].get(key)):
            print(f"SKIP existing {key}", flush=True)
            continue

        config = Config(
            molecule=MOLECULE,
            encoder=ENCODER,
            prior=prior,
            graph_source=graph_source,
            prior_weight=prior_weight,
            num_epochs=NUM_EPOCHS,
            eval_horizons=tuple(HORIZONS),
            seed=seed,
            device=device,
            save_checkpoint=True,
            save_frame_indices=True,
            disjoint_eval=DISJOINT_EVAL,
            checkpoint_dir=CHECKPOINT_DIR,
        )

        print(f"\n=== Starting {key} ===", flush=True)
        try:
            result = train_with_sparse_progress(config)
            result["status"] = "ok"
        except Exception as exc:
            result = {
                "status": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "config": dict(config.__dict__),
            }
            payload["results"][key] = result
            save_payload(payload)
            raise

        payload["results"][key] = result
        save_payload(payload)
        print(f"[{key}] H16={h16_value(result):.3f}", flush=True)
        print(f"  -> saved partial results to {OUTPUT_PATH.name}", flush=True)

    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
