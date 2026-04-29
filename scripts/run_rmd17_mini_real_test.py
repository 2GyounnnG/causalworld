import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rmd17 import Config, train_one_seed


ENCODER = "flat"
MOLECULE = "aspirin"
PRIORS = [
    "none",
    "variance",
    "covariance",
    "sigreg",
    "spectral",
    "permuted_spectral",
    "random_spectral",
]
SEEDS = [0, 1]
NUM_EPOCHS = 5
N_TRANSITIONS = 400
PRIOR_WEIGHT = 0.1
OUTPUT_PATH = ROOT / "rmd17_mini_real_test.json"


def config_prior(prior: str) -> str:
    if prior == "covariance":
        return "euclidean"
    return prior


def main() -> None:
    configs = [
        (
            prior,
            seed,
            Config(
                molecule=MOLECULE,
                encoder=ENCODER,
                prior=config_prior(prior),
                seed=seed,
                num_epochs=NUM_EPOCHS,
                n_transitions=N_TRANSITIONS,
                prior_weight=PRIOR_WEIGHT,
            ),
        )
        for prior in PRIORS
        for seed in SEEDS
    ]

    if not torch.cuda.is_available():
        print("WARNING: CUDA unavailable, falling back to CPU")
        for _, _, config in configs:
            config.device = "cpu"
    else:
        print(f"device=cuda ({torch.cuda.get_device_name(0)})")

    all_results = {}
    total = len(configs)
    for i, (prior, seed, config) in enumerate(configs, start=1):
        print(f"\n=== Starting prior={prior} seed={seed} ({i}/{total}) ===", flush=True)
        result = train_one_seed(config)
        if "final_diagnostics" not in result:
            print(f"WARNING: final_diagnostics missing for prior={prior} seed={seed}", flush=True)
        key = f"{MOLECULE}|{ENCODER}|{prior}|seed={seed}"
        all_results[key] = result

        with OUTPUT_PATH.open("w", encoding="utf-8") as file:
            json.dump(all_results, file, indent=2, default=str)
        print(f"  -> saved partial results to {OUTPUT_PATH.name}", flush=True)

    print("\n=== Summary ===")
    print(
        f"{'prior':18s} {'seed':5s} {'H=1':>8s} {'H=8':>8s} {'H=16':>8s} "
        f"{'r_eff':>8s} {'cond':>8s} {'proj_gauss':>10s} {'spec_align':>10s}"
    )
    for key, result in all_results.items():
        parts = key.split("|")
        prior = parts[2]
        seed = parts[3].split("=")[1]
        errs = result["rollout_errors"]
        diagnostics = result.get("final_diagnostics", {})
        print(
            f"{prior:18s} {seed:5s} "
            f"{errs.get(1, float('nan')):8.4f} "
            f"{errs.get(8, float('nan')):8.4f} "
            f"{errs.get(16, float('nan')):8.4f} "
            f"{diagnostics.get('effective_rank', float('nan')):8.2f} "
            f"{diagnostics.get('condition_number', float('nan')):8.2f} "
            f"{diagnostics.get('projection_gaussianity', float('nan')):10.2f} "
            f"{diagnostics.get('spectral_alignment', float('nan')):10.2f}"
        )


if __name__ == "__main__":
    main()
