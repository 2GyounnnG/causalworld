"""Run the graph-heat lattice 3-seed preflight package."""

from __future__ import annotations

from preflight_multiseed_common import ROOT, run_regime_launcher


OUT_ROOT = ROOT / "analysis_out" / "preflight_runs" / "graph_heat_3seed"
AGGREGATE = ROOT / "scripts" / "aggregate_graph_heat_3seed.py"
CONFIGS = [
    {
        "name": "quick_ep5",
        "dataset": "graph_heat_lattice",
        "epochs": 5,
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
            ("--graph-heat-n", 36),
            ("--graph-heat-d", 4),
            ("--graph-heat-t", 512),
            ("--graph-heat-tau", 0.05),
            ("--graph-heat-noise", 0.01),
            ("--graph-heat-seed", 0),
        ],
    }
]


def main() -> None:
    run_regime_launcher(
        regime_name="graph_heat_3seed",
        default_out_root=OUT_ROOT,
        configs=CONFIGS,
        aggregate_script=AGGREGATE,
        default_seeds=[0, 1, 2],
    )


if __name__ == "__main__":
    main()
