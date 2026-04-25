from __future__ import annotations

import builtins
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rmd17 import Config, build_molecular_laplacian, train_one_seed


MOLECULE = "aspirin"
ENCODER = "flat"
PRIOR = "spectral"
PRIOR_WEIGHT = 0.1
MODES = ["per_frame", "fixed_frame0", "fixed_mean"]
SEEDS = [0, 1, 2, 3, 4]
NUM_EPOCHS = 50
HORIZONS = [1, 2, 4, 8, 16]
OUTPUT_PATH = ROOT / "rmd17_aspirin_laplacian_ablation.json"


def make_key(mode: str, seed: int) -> str:
    return f"{MOLECULE}|{ENCODER}|{PRIOR}|mode={mode}|seed={seed}"


def make_payload() -> dict[str, Any]:
    return {
        "task": "laplacian_ablation",
        "modes": MODES,
        "seeds": SEEDS,
        "results": {},
    }


def save_payload(payload: dict[str, Any]) -> None:
    tmp_path = OUTPUT_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)
    tmp_path.replace(OUTPUT_PATH)


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ablation, but torch.cuda.is_available() is false")
    print(f"device=cuda ({torch.cuda.get_device_name(0)})", flush=True)


def print_every_5_epochs_factory(original_print: Any) -> Any:
    pattern = re.compile(
        r"\[(?P<label>[^\]]+)\]\s+epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)\s+"
        r"loss=(?P<loss>[-+0-9.eEnNaA]+)\s+H8_eval=(?P<h8>[-+0-9.eEnNaA]+)"
    )

    def patched_print(*args: Any, **kwargs: Any) -> None:
        text = " ".join(str(arg) for arg in args)
        match = pattern.search(text)
        if match is None:
            original_print(*args, **kwargs)
            return

        epoch = int(match.group("epoch"))
        if epoch % 5 == 0:
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


def main() -> None:
    require_cuda()
    payload = make_payload()

    # Keep the requested import visible to readers of this ablation script.
    _ = build_molecular_laplacian

    for mode in MODES:
        for seed in SEEDS:
            key = make_key(mode, seed)
            config = Config(
                molecule=MOLECULE,
                encoder=ENCODER,
                prior=PRIOR,
                prior_weight=PRIOR_WEIGHT,
                num_epochs=NUM_EPOCHS,
                eval_horizons=tuple(HORIZONS),
                seed=seed,
                device="cuda",
                laplacian_mode=mode,
            )

            print(f"\n=== Starting {key} ===", flush=True)
            result = train_with_sparse_progress(config)
            payload["results"][key] = result
            save_payload(payload)
            print(f"  -> saved partial results to {OUTPUT_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
