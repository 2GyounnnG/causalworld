"""Run the graph-wave lattice 5-seed preflight package."""

from __future__ import annotations

from preflight_multiseed_common import ROOT, run_regime_launcher


OUT_ROOT = ROOT / "analysis_out" / "preflight_runs" / "graph_wave_5seed"
AGGREGATE = ROOT / "scripts" / "aggregate_graph_wave_5seed.py"
COMMON = {
    "dataset": "graph_wave_lattice",
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
        ("--wave-n", 36),
        ("--wave-t", 512),
        ("--wave-c", 1.0),
        ("--wave-damping", 0.02),
        ("--wave-dt", 0.05),
        ("--wave-noise", 0.001),
        ("--wave-seed", 0),
    ],
}
CONFIGS = [
    {**COMMON, "name": "quick_ep5", "epochs": 5},
    {**COMMON, "name": "standard_ep20", "epochs": 20},
]


def main() -> None:
    run_regime_launcher(
        regime_name="graph_wave_5seed",
        default_out_root=OUT_ROOT,
        configs=CONFIGS,
        aggregate_script=AGGREGATE,
        default_seeds=[0, 1, 2, 3, 4],
    )


if __name__ == "__main__":
    main()
