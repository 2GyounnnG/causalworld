"""Aggregate graph-wave 5-seed preflight outputs."""

from __future__ import annotations

from preflight_multiseed_common import ROOT, run_regime_aggregator


def main() -> None:
    run_regime_aggregator(
        title="GRAPH_WAVE_5SEED_REPORT",
        default_out_root=ROOT / "analysis_out" / "preflight_runs" / "graph_wave_5seed",
        config_names=["quick_ep5", "standard_ep20"],
        default_seeds=[0, 1, 2, 3, 4],
        default_report=ROOT / "analysis_out" / "GRAPH_WAVE_5SEED_REPORT.md",
        default_table=ROOT / "paper" / "tables" / "graph_wave_5seed_summary.md",
        default_summary_csv=ROOT / "analysis_out" / "graph_wave_5seed_summary.csv",
        seed_count=5,
    )


if __name__ == "__main__":
    main()
