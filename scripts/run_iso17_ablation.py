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

from scripts.iso17_loader import DATA_ROOT, ISO17Trajectory
from scripts.train_iso17 import Config, train_one_seed


EXPERIMENT_NAME = "iso17_split_eval_ablation"
TRAIN_SPLIT = "reference"
EVAL_SPLITS = ["test_within", "test_other"]
ISOMER = "all"
ENCODER = "flat"
PRIOR_VARIANTS = [("none", "bond"), ("euclidean", "bond"), ("spectral", "bond")]
PRIOR_WEIGHT = 0.1
NUM_EPOCHS = 50
SEEDS = [0, 1, 2, 3, 4]
HORIZONS = [1, 2, 4, 8, 16]
SAVE_CHECKPOINT = True
SAVE_FRAME_INDICES = True
DISJOINT_EVAL = True
CHECKPOINT_DIR = "checkpoints/iso17_split_eval_ablation"
OUTPUT_PATH = ROOT / "iso17_split_eval_ablation.json"


def make_payload() -> dict[str, Any]:
    return {
        "experiment_name": EXPERIMENT_NAME,
        "train_split": TRAIN_SPLIT,
        "eval_splits": EVAL_SPLITS,
        "isomer": ISOMER,
        "encoder": ENCODER,
        "prior_variants": [
            {"prior": prior, "graph_source": graph_source}
            for prior, graph_source in PRIOR_VARIANTS
        ],
        "prior_weight": PRIOR_WEIGHT,
        "epochs": NUM_EPOCHS,
        "seeds": SEEDS,
        "horizons": HORIZONS,
        "save_checkpoint": SAVE_CHECKPOINT,
        "save_frame_indices": SAVE_FRAME_INDICES,
        "strict_disjoint_eval": DISJOINT_EVAL,
        "checkpoint_dir": CHECKPOINT_DIR,
        "data_root": str(DATA_ROOT),
        "total_runs": len(PRIOR_VARIANTS) * len(SEEDS),
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
    return f"iso17|{TRAIN_SPLIT}|{prior}|seed={seed}"


def is_complete(result: Any) -> bool:
    return (
        isinstance(result, dict)
        and result.get("status", "ok") == "ok"
        and isinstance(result.get("rollout_errors_test_within"), dict)
        and isinstance(result.get("rollout_errors_test_other"), dict)
    )


def verify_inputs() -> None:
    train = ISO17Trajectory(split=TRAIN_SPLIT, isomer=ISOMER)
    print(f"Verified ISO17 train source: {train}", flush=True)
    for eval_split in EVAL_SPLITS:
        eval_traj = ISO17Trajectory(split=eval_split, isomer=ISOMER)
        print(f"Verified ISO17 eval source:  {eval_traj}", flush=True)


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


def h16_value(result: dict[str, Any], split: str) -> float:
    rollout_errors = result.get(f"rollout_errors_{split}", {})
    return float(rollout_errors.get(16, rollout_errors.get("16", float("nan"))))


def main() -> None:
    verify_inputs()
    payload = load_payload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"device=cuda ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print("WARNING: CUDA unavailable, falling back to CPU", flush=True)

    print(
        f"Planned runs: {len(PRIOR_VARIANTS)} priors x {len(SEEDS)} seeds = "
        f"{len(PRIOR_VARIANTS) * len(SEEDS)}",
        flush=True,
    )

    for prior, graph_source in PRIOR_VARIANTS:
        for seed in SEEDS:
            key = make_key(prior, seed)
            if is_complete(payload["results"].get(key)):
                print(f"SKIP existing {key}", flush=True)
                continue

            config = Config(
                split=TRAIN_SPLIT,
                eval_split=EVAL_SPLITS[0],
                eval_splits=tuple(EVAL_SPLITS),
                isomer=ISOMER,
                eval_isomer=ISOMER,
                encoder=ENCODER,
                prior=prior,
                graph_source=graph_source,
                prior_weight=PRIOR_WEIGHT,
                num_epochs=NUM_EPOCHS,
                eval_horizons=tuple(HORIZONS),
                seed=seed,
                device=device,
                save_checkpoint=SAVE_CHECKPOINT,
                save_frame_indices=SAVE_FRAME_INDICES,
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
            h16_parts = " ".join(
                f"{split}_H16={h16_value(result, split):.3f}" for split in EVAL_SPLITS
            )
            print(f"[{key}] {h16_parts}", flush=True)
            print(f"  -> saved partial results to {OUTPUT_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
