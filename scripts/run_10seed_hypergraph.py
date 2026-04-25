"""Task A+ wrapper for the overnight hypergraph validation."""

from __future__ import annotations

from run_10seed_flat import DEFAULT_HORIZONS, DEFAULT_SEEDS, TaskSettings, main_for_settings


def main() -> int:
    settings = TaskSettings(
        task="A+",
        encoder="hypergraph",
        output_json="validation_10seed_hypergraph.json",
        output_png="validation_10seed_hypergraph.png",
        seeds=DEFAULT_SEEDS,
        horizons=DEFAULT_HORIZONS,
        max_steps=16,
        hidden_dim=32,
    )
    return main_for_settings(settings)


if __name__ == "__main__":
    raise SystemExit(main())
