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

from scripts.analyze_rmd17_disjoint_checkpointed import analyze_file
from scripts.train_rmd17 import Config, train_one_seed


EXPERIMENT_NAME = "rmd17_aspirin_disjoint_checkpointed"
MOLECULE = "aspirin"
ENCODER = "flat"
PRIORS = ["none", "euclidean", "spectral"]
SEEDS = [0, 1, 2, 3, 4]
NUM_EPOCHS = 50
HORIZONS = [1, 2, 4, 8, 16]
OUTPUT_PATH = ROOT / "rmd17_aspirin_disjoint_checkpointed.json"
CHECKPOINT_DIR = "checkpoints/rmd17_aspirin_disjoint_checkpointed"
AGGREGATE_PATH = ROOT / "analysis_out/aggregate_rmd17_aspirin_disjoint_checkpointed.csv"
REPORT_PATH = ROOT / "analysis_out/RMD17_ASPIRIN_DISJOINT_CHECKPOINTED.md"


def make_payload() -> dict[str, Any]:
    return {
        "experiment_name": EXPERIMENT_NAME,
        "molecule": MOLECULE,
        "encoder": ENCODER,
        "priors": PRIORS,
        "seeds": SEEDS,
        "epochs": NUM_EPOCHS,
        "horizons": HORIZONS,
        "strict_disjoint_eval": True,
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


def make_key(prior: str, seed: int) -> str:
    return f"{MOLECULE}|{ENCODER}|{prior}|seed={seed}"


def is_complete(result: Any) -> bool:
    return isinstance(result, dict) and result.get("status", "ok") == "ok" and isinstance(result.get("rollout_errors"), dict)


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


def refresh_analysis() -> None:
    analyze_file(
        input_path=OUTPUT_PATH,
        aggregate_path=AGGREGATE_PATH,
        report_path=REPORT_PATH,
        title="rMD17 Aspirin Disjoint Checkpointed",
        expected_molecules=[MOLECULE],
        expected_seeds=SEEDS,
    )


def main() -> None:
    payload = load_payload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"device=cuda ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print("WARNING: CUDA unavailable, falling back to CPU", flush=True)

    for prior in PRIORS:
        for seed in SEEDS:
            key = make_key(prior, seed)
            if is_complete(payload["results"].get(key)):
                print(f"SKIP existing {key}", flush=True)
                continue

            config = Config(
                molecule=MOLECULE,
                encoder=ENCODER,
                prior=prior,
                prior_weight=0.1,
                num_epochs=NUM_EPOCHS,
                eval_horizons=tuple(HORIZONS),
                seed=seed,
                device=device,
                save_checkpoint=True,
                save_frame_indices=True,
                disjoint_eval=True,
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
                refresh_analysis()
                raise

            payload["results"][key] = result
            save_payload(payload)
            refresh_analysis()
            print(f"  -> saved partial results to {OUTPUT_PATH.name}", flush=True)

    refresh_analysis()
    print(f"wrote {AGGREGATE_PATH.relative_to(ROOT)}", flush=True)
    print(f"wrote {REPORT_PATH.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
