"""Aggregate graph-heat 3-seed preflight outputs."""

from __future__ import annotations

from preflight_multiseed_common import ROOT, run_regime_aggregator


def main() -> None:
    run_regime_aggregator(
        title="GRAPH_HEAT_3SEED_REPORT",
        default_out_root=ROOT / "analysis_out" / "preflight_runs" / "graph_heat_3seed",
        config_names=["quick_ep5"],
        default_seeds=[0, 1, 2],
        default_report=ROOT / "analysis_out" / "GRAPH_HEAT_3SEED_REPORT.md",
        default_table=ROOT / "paper" / "tables" / "graph_heat_3seed_summary.md",
        default_summary_csv=ROOT / "analysis_out" / "graph_heat_3seed_summary.csv",
        seed_count=3,
    )


if __name__ == "__main__":
    main()
