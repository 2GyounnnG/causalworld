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

from scripts.train_rmd17 import Config, train_one_seed


MOLECULE = "aspirin"
ENCODER = "flat"
PRIOR = "euclidean"
SEEDS = list(range(10))
NUM_EPOCHS = 50
OUTPUT_PATH = ROOT / "rmd17_aspirin_10seed_results.json"


def make_key(prior: str, seed: int) -> str:
    return f"{MOLECULE}|{ENCODER}|{prior}|seed={seed}"


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this redo, but torch.cuda.is_available() is false")
    print(f"device=cuda ({torch.cuda.get_device_name(0)})", flush=True)


def load_results() -> dict[str, Any]:
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Missing results file: {OUTPUT_PATH}")
    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_results(data: dict[str, Any]) -> None:
    tmp_path = OUTPUT_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)
    tmp_path.replace(OUTPUT_PATH)


def validate_bug_state(data: dict[str, Any]) -> None:
    none_key = make_key("none", 0)
    euclidean_key = make_key(PRIOR, 0)

    none_loss = data[none_key]["final_loss"]
    euclidean_loss = data[euclidean_key]["final_loss"]

    if none_loss == euclidean_loss:
        print(
            "WARNING: confirmed stale euclidean bug state before redo: "
            f"{none_key}.final_loss == {euclidean_key}.final_loss == {none_loss:.15f}",
            flush=True,
        )
        return

    print(
        "ERROR: seed=0 none/euclidean final_loss values already differ; "
        f"refusing to overwrite {OUTPUT_PATH.name}\n"
        f"  {none_key}.final_loss      = {none_loss:.15f}\n"
        f"  {euclidean_key}.final_loss = {euclidean_loss:.15f}",
        flush=True,
    )
    raise SystemExit(1)


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
            original_print(*args, **kwargs)

    return patched_print


def train_with_sparse_progress(config: Config) -> dict[str, Any]:
    original_print = builtins.print
    builtins.print = print_every_5_epochs_factory(original_print)
    try:
        return train_one_seed(config)
    finally:
        builtins.print = original_print


def verify_redo(data: dict[str, Any]) -> None:
    equal_seeds: list[int] = []
    for seed in SEEDS:
        none_loss = data[make_key("none", seed)]["final_loss"]
        euclidean_loss = data[make_key(PRIOR, seed)]["final_loss"]
        if none_loss == euclidean_loss:
            equal_seeds.append(seed)

    if equal_seeds:
        raise RuntimeError(
            "Redo verification failed: euclidean final_loss still exactly equals none "
            f"for seeds {equal_seeds}"
        )

    print("verification passed: euclidean final_loss differs from none for all 10 seeds", flush=True)


def main() -> None:
    require_cuda()
    data = load_results()
    validate_bug_state(data)

    for seed in SEEDS:
        key = make_key(PRIOR, seed)
        print(f"\n=== Starting {key} ===", flush=True)
        config = Config(
            molecule=MOLECULE,
            encoder=ENCODER,
            prior=PRIOR,
            seed=seed,
            num_epochs=NUM_EPOCHS,
        )
        result = train_with_sparse_progress(config)
        data[key] = result
        save_results(data)
        print(f"  -> saved partial results to {OUTPUT_PATH.name}", flush=True)

    verify_redo(data)


if __name__ == "__main__":
    main()
