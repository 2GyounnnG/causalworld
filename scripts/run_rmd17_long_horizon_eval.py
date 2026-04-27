from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rmd17 import (
    Config,
    checkpoint_path,
    collect_disjoint_eval_transitions,
    collect_rmd17_transitions,
    evaluate_rollout,
    load_checkpoint_model,
    transition_used_frames,
)


EXPERIMENT_NAME = "rmd17_long_horizon_eval"
MOLECULES = ["aspirin", "ethanol", "malonaldehyde"]
ENCODER = "flat"
PRIOR_VARIANTS = [("none", "bond"), ("spectral", "bond")]
SEEDS = [0, 1, 2, 3, 4]
PRIOR_WEIGHT = 0.1
NEW_HORIZONS = [32, 64]
EXISTING_HORIZONS_IN_METADATA = [1, 2, 4, 8, 16]
OUTPUT_PATH = ROOT / "rmd17_long_horizon_eval.json"
ASPIRIN_CHECKPOINT_DIR = "checkpoints/rmd17_aspirin_disjoint_checkpointed"
MULTIMOLECULE_CHECKPOINT_DIR = "checkpoints/rmd17_multimolecule_disjoint_checkpointed"


def checkpoint_dir_for_molecule(molecule: str) -> str:
    if molecule == "aspirin":
        return ASPIRIN_CHECKPOINT_DIR
    return MULTIMOLECULE_CHECKPOINT_DIR


def checkpoint_for_run(molecule: str, prior: str, graph_source: str, seed: int) -> Path:
    config = Config(
        molecule=molecule,
        encoder=ENCODER,
        prior=prior,
        graph_source=graph_source,
        prior_weight=PRIOR_WEIGHT,
        seed=seed,
        checkpoint_dir=checkpoint_dir_for_molecule(molecule),
    )
    return checkpoint_path(config)


def make_key(molecule: str, prior: str, seed: int) -> str:
    return f"{molecule}|{ENCODER}|{prior}|seed={seed}"


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def make_payload() -> dict[str, Any]:
    return {
        "experiment_name": EXPERIMENT_NAME,
        "molecules": MOLECULES,
        "encoder": ENCODER,
        "prior_variants": PRIOR_VARIANTS,
        "seeds": SEEDS,
        "prior_weight": PRIOR_WEIGHT,
        "new_horizons": NEW_HORIZONS,
        "existing_horizons_in_metadata": EXISTING_HORIZONS_IN_METADATA,
        "checkpoint_dirs": {
            "aspirin": ASPIRIN_CHECKPOINT_DIR,
            "ethanol": MULTIMOLECULE_CHECKPOINT_DIR,
            "malonaldehyde": MULTIMOLECULE_CHECKPOINT_DIR,
        },
        "checkpoint_filename_pattern": (
            "{molecule}_{encoder}_{prior_label}_"
            "seed{seed}_w{prior_weight}_mode{laplacian_mode}.pt"
        ),
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
        json.dump(payload, file, indent=2, sort_keys=True)
    tmp_path.replace(OUTPUT_PATH)


def is_complete(result: Any) -> bool:
    return (
        isinstance(result, dict)
        and result.get("status") == "ok"
        and isinstance(result.get("rollout_errors_extended"), dict)
    )


def collect_eval_transitions(config: Config) -> list[dict[str, Any]]:
    if config.disjoint_eval:
        train_transitions = collect_rmd17_transitions(
            molecule=config.molecule,
            n_transitions=config.n_transitions,
            stride=config.stride,
            horizon=config.horizon,
            seed=config.seed,
        )
        train_used_frame_idx = set(transition_used_frames(train_transitions, [config.horizon]))
        return collect_disjoint_eval_transitions(
            molecule=config.molecule,
            n_transitions=config.eval_n_transitions,
            stride=config.stride * 10,
            eval_horizons=tuple(EXISTING_HORIZONS_IN_METADATA + NEW_HORIZONS),
            seed=config.seed + 1000,
            forbidden_frame_idx=train_used_frame_idx,
        )

    return collect_rmd17_transitions(
        molecule=config.molecule,
        n_transitions=config.eval_n_transitions,
        stride=config.stride * 10,
        horizon=max(NEW_HORIZONS),
        seed=config.seed + 1000,
    )


def main() -> None:
    payload = load_payload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"device=cuda ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print("WARNING: CUDA unavailable, falling back to CPU", flush=True)

    for molecule in MOLECULES:
        for prior, graph_source in PRIOR_VARIANTS:
            for seed in SEEDS:
                key = make_key(molecule, prior, seed)
                if is_complete(payload["results"].get(key)):
                    print(f"SKIP existing {key}", flush=True)
                    continue

                path = checkpoint_for_run(molecule, prior, graph_source, seed)
                path_text = rel_path(path)
                if not path.exists():
                    result = {
                        "status": "missing_checkpoint",
                        "checkpoint_path": path_text,
                        "molecule": molecule,
                        "prior": prior,
                        "seed": seed,
                    }
                    payload["results"][key] = result
                    save_payload(payload)
                    print(f"SKIP missing checkpoint {path_text}", flush=True)
                    continue

                print(f"\n=== Evaluating {key} from {path_text} ===", flush=True)
                config, model, _checkpoint = load_checkpoint_model(path, device=device)
                eval_transitions = collect_eval_transitions(config)
                rollout_errors = evaluate_rollout(
                    model,
                    eval_transitions,
                    horizons=NEW_HORIZONS,
                    device=torch.device(device),
                    latent_dim=config.latent_dim,
                )

                result = {
                    "rollout_errors_extended": {
                        str(horizon): rollout_errors[horizon]
                        for horizon in NEW_HORIZONS
                    },
                    "checkpoint_path": path_text,
                    "status": "ok",
                    "molecule": molecule,
                    "prior": prior,
                    "seed": seed,
                }
                payload["results"][key] = result
                save_payload(payload)
                print(
                    f"[{key}] H32={rollout_errors[32]:.4f} H64={rollout_errors[64]:.4f}",
                    flush=True,
                )

    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
