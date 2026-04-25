import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rmd17 import Config, train_one_seed

OUT = ROOT / "rmd17_aspirin_10seed_results.json"

if OUT.exists():
    with OUT.open("r", encoding="utf-8") as f:
        all_results = json.load(f)
else:
    all_results = {}

print("resume target:", OUT, flush=True)
print("existing keys:", len(all_results), flush=True)

if torch.cuda.is_available():
    print(f"device=cuda ({torch.cuda.get_device_name(0)})", flush=True)
else:
    print("WARNING: CUDA unavailable; using CPU", flush=True)

for seed in range(10):
    key = f"aspirin|flat|spectral|seed={seed}"

    if key in all_results:
        print(f"SKIP existing {key}", flush=True)
        continue

    print(f"\n=== Starting {key} ===", flush=True)
    cfg = Config(
        molecule="aspirin",
        encoder="flat",
        prior="spectral",
        seed=seed,
        num_epochs=50,
    )

    if not torch.cuda.is_available():
        cfg.device = "cpu"

    result = train_one_seed(cfg)
    all_results[key] = result

    tmp = OUT.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    tmp.replace(OUT)

    print(f"  -> saved partial results to {OUT.name}", flush=True)

print("DONE rmd17 aspirin spectral resume", flush=True)
