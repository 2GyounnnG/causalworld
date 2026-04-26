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

from scripts.rmd17_loader import DATA_ROOT, MOLECULES as RMD17_MOLECULES
from scripts.train_rmd17 import Config, train_one_seed


EXPERIMENT_NAME = "rmd17_extended_molecules"
MOLECULES = ["benzene", "toluene", "naphthalene", "salicylic"]
ENCODER = "flat"
PRIOR_VARIANTS = [("none", "bond"), ("spectral", "bond")]
PRIOR_WEIGHT = 0.1
NUM_EPOCHS = 50
SEEDS = [0, 1, 2, 3, 4]
HORIZONS = [1, 2, 4, 8, 16]
SAVE_CHECKPOINT = True
SAVE_FRAME_INDICES = True
DISJOINT_EVAL = True
CHECKPOINT_DIR = "checkpoints/rmd17_extended_molecules"
OUTPUT_PATH = ROOT / "rmd17_extended_molecules.json"


def make_payload() -> dict[str, Any]:
    return {
        "experiment_name": EXPERIMENT_NAME,
        "molecules": MOLECULES,
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
        "total_runs": len(MOLECULES) * len(PRIOR_VARIANTS) * len(SEEDS),
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


def make_key(molecule: str, prior: str, seed: int) -> str:
    return f"{molecule}|{ENCODER}|{prior}|seed={seed}"


def is_complete(result: Any) -> bool:
    return (
        isinstance(result, dict)
        and result.get("status", "ok") == "ok"
        and isinstance(result.get("rollout_errors"), dict)
    )


def verify_inputs() -> None:
    missing_molecules = [molecule for molecule in MOLECULES if molecule not in RMD17_MOLECULES]
    if missing_molecules:
        raise ValueError(
            "Missing molecules in scripts.rmd17_loader.MOLECULES: "
            + ", ".join(missing_molecules)
        )

    missing_npz = []
    for molecule in MOLECULES:
        npz_path = DATA_ROOT / f"rmd17_{molecule}.npz"
        if not npz_path.exists():
            missing_npz.append(str(npz_path))
    if missing_npz:
        raise FileNotFoundError("Missing rMD17 npz files:\n" + "\n".join(missing_npz))

    print(
        "Verified molecules in scripts.rmd17_loader.MOLECULES: "
        + ", ".join(MOLECULES),
        flush=True,
    )
    print(f"Verified npz files in {DATA_ROOT}:", flush=True)
    for molecule in MOLECULES:
        print(f"  - rmd17_{molecule}.npz", flush=True)


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
    verify_inputs()
    payload = load_payload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"device=cuda ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print("WARNING: CUDA unavailable, falling back to CPU", flush=True)

    print(
        f"Planned runs: {len(MOLECULES)} molecules x "
        f"{len(PRIOR_VARIANTS)} priors x {len(SEEDS)} seeds = "
        f"{len(MOLECULES) * len(PRIOR_VARIANTS) * len(SEEDS)}",
        flush=True,
    )

    for molecule in MOLECULES:
        for prior, graph_source in PRIOR_VARIANTS:
            for seed in SEEDS:
                key = make_key(molecule, prior, seed)
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
                print(f"[{key}] H16={h16_value(result):.3f}", flush=True)
                print(f"  -> saved partial results to {OUTPUT_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
