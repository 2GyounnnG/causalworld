"""Sanity check for Wolfram prior variants."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import (
    Config,
    build_environment,
    collect_episodes,
    flatten_transitions,
    select_device,
    set_seed,
    train_one,
)


PRIORS = [
    "none",
    "variance",
    "covariance",
    "sigreg",
    "spectral",
    "permuted_spectral",
    "random_spectral",
]
ENCODER = "flat"
ENV_PROFILE = "minimal"
NUM_EPOCHS = 3
BATCH_SIZE = 16
LATENT_DIM = 16
PRIOR_WEIGHT = 0.1
N_TRAIN = 24
N_EVAL = 4


def run_prior(prior: str, device: torch.device) -> tuple[float, float]:
    set_seed(0)
    rule, initial_state, max_steps = build_environment(ENV_PROFILE, seed=0)
    train_episodes = collect_episodes(
        rule,
        initial_state,
        N_TRAIN,
        max_steps,
        seed=0,
        env_profile=ENV_PROFILE,
    )
    eval_episodes = collect_episodes(
        rule,
        initial_state,
        N_EVAL,
        max_steps,
        seed=10_000,
        env_profile=ENV_PROFILE,
    )
    train_transitions = flatten_transitions(train_episodes)
    config = Config(
        seed=0,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        latent_dim=LATENT_DIM,
        encoder=ENCODER,
        prior=prior,
        prior_weight=PRIOR_WEIGHT,
    )
    _, history = train_one(config, train_transitions, eval_episodes, device)
    losses = history["total_loss"]
    if len(losses) != NUM_EPOCHS:
        raise RuntimeError(f"expected {NUM_EPOCHS} losses, got {len(losses)}")

    epoch0_loss = float(losses[0])
    epoch3_loss = float(losses[-1])
    if not math.isfinite(epoch3_loss):
        raise RuntimeError(f"final train loss is not finite: {epoch3_loss}")
    if not epoch3_loss < epoch0_loss:
        raise RuntimeError(
            f"final train loss did not decrease: epoch0={epoch0_loss:.6f}, "
            f"epoch3={epoch3_loss:.6f}"
        )
    return epoch0_loss, epoch3_loss


def main() -> None:
    device = select_device("auto")
    failed: list[str] = []
    print(f"device={device}", flush=True)

    for prior in PRIORS:
        try:
            epoch0_loss, epoch3_loss = run_prior(prior, device)
        except Exception as exc:  # pragma: no cover - diagnostic script.
            failed.append(prior)
            print(f"[FAIL] prior={prior}: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            continue
        print(
            f"[OK] prior={prior} epoch0_loss={epoch0_loss:.4f} "
            f"epoch3_loss={epoch3_loss:.4f}",
            flush=True,
        )

    if failed:
        print(
            f"Sanity check failed for {len(failed)} priors: {', '.join(failed)}",
            flush=True,
        )
        sys.exit(1)

    print("All 7 priors passed sanity check.", flush=True)


if __name__ == "__main__":
    main()
