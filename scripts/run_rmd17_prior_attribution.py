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


TASK = "prior_attribution"
MOLECULES = ["aspirin", "ethanol", "malonaldehyde"]
ENCODER = "flat"
PRIOR_VARIANTS = [
    ("spectral", "random"),
    ("spectral", "complete"),
    ("spectral", "identity"),
]
PRIOR_WEIGHT = 0.1
NUM_EPOCHS = 50
SEEDS = [0, 1, 2, 3, 4]
HORIZONS = [1, 2, 4, 8, 16]
CHECKPOINT_DIR = "checkpoints/rmd17_prior_attribution"
OUTPUT_PATH = ROOT / "rmd17_prior_attribution.json"


def make_payload() -> dict[str, Any]:
    return {
        "task": TASK,
        "molecules": MOLECULES,
        "encoder": ENCODER,
        "variants": [
            {
                "label": f"{prior}_{graph_source}_graph"
                if graph_source != "identity"
                else f"{prior}_{graph_source}",
                "prior": prior,
                "graph_source": graph_source,
            }
            for prior, graph_source in PRIOR_VARIANTS
        ],
        "seeds": SEEDS,
        "epochs": NUM_EPOCHS,
        "horizons": HORIZONS,
        "prior_weight": PRIOR_WEIGHT,
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


def make_key(molecule: str, prior: str, graph_source: str, seed: int) -> str:
    return f"{molecule}|{ENCODER}|{prior}|graph={graph_source}|seed={seed}"


def is_complete(result: Any) -> bool:
    return isinstance(result, dict) and result.get("status", "ok") == "ok" and isinstance(result.get("rollout_errors"), dict)


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for prior attribution, but torch.cuda.is_available() is false")
    print(f"device=cuda ({torch.cuda.get_device_name(0)})", flush=True)


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
    require_cuda()
    payload = load_payload()

    for molecule in MOLECULES:
        for prior, graph_source in PRIOR_VARIANTS:
            for seed in SEEDS:
                key = make_key(molecule, prior, graph_source, seed)
                if is_complete(payload["results"].get(key)):
                    print(f"SKIP existing {key}", flush=True)
                    continue

                config = Config(
                    molecule=molecule,
                    encoder=ENCODER,
                    prior=prior,
                    graph_source=graph_source,
                    prior_weight=PRIOR_WEIGHT,
                    num_epochs=NUM_EPOCHS,
                    eval_horizons=tuple(HORIZONS),
                    seed=seed,
                    device="cuda",
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
                    raise

                payload["results"][key] = result
                save_payload(payload)
                print(f"[{molecule}|graph={graph_source}|seed={seed}] H16={h16_value(result):.3f}", flush=True)
                print(f"  -> saved partial results to {OUTPUT_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
