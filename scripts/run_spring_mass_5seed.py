"""Run the spring-mass lattice 5-seed preflight package."""

from __future__ import annotations

from pathlib import Path

from preflight_multiseed_common import ROOT, run_regime_launcher


OUT_ROOT = ROOT / "analysis_out" / "preflight_runs" / "spring_mass_5seed"
AGGREGATE = ROOT / "scripts" / "aggregate_spring_mass_5seed.py"
COMMON = {
    "dataset": "spring_mass_lattice",
    "train_transitions": 96,
    "eval_transitions": 32,
    "raw_transitions": 64,
    "train_stride": 5,
    "eval_stride": 10,
    "raw_stride": 5,
    "prior_weight": 0.1,
    "batch_size": 32,
    "device": "auto",
    "extra_args": [
        ("--spring-n", 36),
        ("--spring-position-dim", 1),
        ("--spring-t", 512),
        ("--spring-k", 1.0),
        ("--spring-damping", 0.05),
        ("--spring-dt", 0.05),
        ("--spring-noise", 0.001),
        ("--spring-seed", 0),
    ],
}
CONFIGS = [
    {**COMMON, "name": "quick_ep5", "epochs": 5},
    {**COMMON, "name": "standard_ep20", "epochs": 20},
]


def main() -> None:
    run_regime_launcher(
        regime_name="spring_mass_5seed",
        default_out_root=OUT_ROOT,
        configs=CONFIGS,
        aggregate_script=AGGREGATE,
        default_seeds=[0, 1, 2, 3, 4],
    )


if __name__ == "__main__":
    main()
