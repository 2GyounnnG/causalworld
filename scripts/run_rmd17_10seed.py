import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rmd17 import Config, train_one_seed


ENCODER = "flat"
MOLECULES = ["aspirin"]
PRIORS = ["none", "euclidean", "spectral"]
SEEDS = list(range(10))
NUM_EPOCHS = 50
OUTPUT_PATH = ROOT / "rmd17_aspirin_10seed_results.json"


def main() -> None:
    configs = [
        Config(
            molecule=molecule,
            encoder=ENCODER,
            prior=prior,
            seed=seed,
            num_epochs=NUM_EPOCHS,
        )
        for molecule in MOLECULES
        for prior in PRIORS
        for seed in SEEDS
    ]

    if not torch.cuda.is_available():
        print("WARNING: CUDA unavailable, falling back to CPU")
        for config in configs:
            config.device = "cpu"
    else:
        print(f"device=cuda ({torch.cuda.get_device_name(0)})")

    all_results = {}
    for config in configs:
        key = f"{config.molecule}|{config.encoder}|{config.prior}|seed={config.seed}"
        print(f"\n=== Starting {key} ===", flush=True)
        result = train_one_seed(config)
        all_results[key] = result

        with OUTPUT_PATH.open("w", encoding="utf-8") as file:
            json.dump(all_results, file, indent=2, default=str)
        print(f"  -> saved partial results to {OUTPUT_PATH.name}", flush=True)

    print("\n=== Summary ===")
    print(f"{'molecule':14s} {'prior':12s} {'seed':5s} " + " ".join(f"H={h:<4d}" for h in [1, 2, 4, 8, 16]))
    for key, result in all_results.items():
        parts = key.split("|")
        molecule = parts[0]
        prior = parts[2]
        seed = parts[3].split("=")[1]
        errs = result["rollout_errors"]
        print(
            f"{molecule:14s} {prior:12s} {seed:5s} "
            + " ".join(f"{errs.get(h, float('nan')):.4f}" for h in [1, 2, 4, 8, 16])
        )


if __name__ == "__main__":
    main()
