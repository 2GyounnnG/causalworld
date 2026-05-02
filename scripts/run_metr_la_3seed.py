"""Run the METR-LA correlation-top-k 3-seed preflight package."""

from __future__ import annotations

from preflight_multiseed_common import ROOT, run_regime_launcher


OUT_ROOT = ROOT / "analysis_out" / "preflight_runs" / "metr_la_3seed"
AGGREGATE = ROOT / "scripts" / "aggregate_metr_la_3seed.py"
CONFIGS = [
    {
        "name": "corr_T2000_train160_ep5",
        "dataset": "metr_la",
        "epochs": 5,
        "train_transitions": 160,
        "eval_transitions": 64,
        "raw_transitions": 64,
        "train_stride": 10,
        "eval_stride": 20,
        "raw_stride": 10,
        "prior_weight": 0.1,
        "batch_size": 32,
        "device": "auto",
        "extra_args": [
            ("--metr-la-root", ROOT / "data" / "metr_la"),
            ("--metr-la-csv", ROOT / "data" / "metr_la" / "metr-la.csv"),
            ("--metr-la-graph-source", "correlation_topk"),
            ("--metr-la-top-k", 8),
            ("--metr-la-corr-mode", "positive"),
            ("--metr-la-max-timesteps", 2000),
        ],
        "extra_flags": ["--metr-la-no-pyg-temporal"],
    }
]


def main() -> None:
    run_regime_launcher(
        regime_name="metr_la_3seed",
        default_out_root=OUT_ROOT,
        configs=CONFIGS,
        aggregate_script=AGGREGATE,
        default_seeds=[0, 1, 2],
    )


if __name__ == "__main__":
    main()
