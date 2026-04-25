"""Task B wrapper for the overnight long-horizon flat validation."""

from __future__ import annotations

from run_10seed_flat import TaskSettings, main_for_settings


def main() -> int:
    settings = TaskSettings(
        task="B",
        encoder="flat",
        output_json="long_horizon.json",
        output_png="long_horizon.png",
        seeds=[0, 1, 2],
        horizons=[1, 2, 4, 8, 16, 32, 64],
        max_steps=64,
        hidden_dim=32,
    )
    return main_for_settings(settings)


if __name__ == "__main__":
    raise SystemExit(main())
