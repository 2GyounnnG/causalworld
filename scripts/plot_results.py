from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HORIZONS = [1, 2, 4, 8, 16]
COLORS = {
    "none": "#4c4c4c",
    "euclidean": "#2f6db3",
    "spectral": "#b63f3f",
    "per_frame": "#b63f3f",
    "fixed_frame0": "#3f7f5f",
    "fixed_mean": "#6b4da8",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def f(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def save_both(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def setup_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", color="#dddddd", linewidth=0.8)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5, alpha=0.5)


def plot_horizon_scaling(rows: list[dict[str, str]], out_base: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    setup_axis(ax, title, "Rollout horizon", "Latent rollout error")
    any_data = False
    labels = sorted({row.get("prior", "") for row in rows if row.get("prior")})
    order = [label for label in ["none", "euclidean", "spectral"] if label in labels] + [label for label in labels if label not in {"none", "euclidean", "spectral"}]
    for prior in order:
        xs, ys, low, high = [], [], [], []
        for horizon in HORIZONS:
            match = next((row for row in rows if row.get("prior") == prior and row.get("horizon") == str(horizon)), None)
            if not match:
                continue
            mean = f(match.get("mean"))
            ci_low = f(match.get("ci95_low"))
            ci_high = f(match.get("ci95_high"))
            if mean is None:
                continue
            xs.append(horizon)
            ys.append(mean)
            low.append(ci_low if ci_low is not None else mean)
            high.append(ci_high if ci_high is not None else mean)
        if xs:
            any_data = True
            color = COLORS.get(prior, None)
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=prior, color=color)
            ax.fill_between(xs, low, high, color=color, alpha=0.16)
    ax.set_xscale("log", base=2)
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels([str(h) for h in HORIZONS])
    if any_data:
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
    save_both(fig, out_base)


def plot_weight_sweep(rows: list[dict[str, str]], out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    setup_axis(ax, "rMD17 Aspirin Prior-Weight Sweep", "Prior weight", "H=16 rollout error")
    h16 = [row for row in rows if row.get("horizon") == "16"]
    weights = sorted({f(row.get("prior_weight")) for row in h16 if f(row.get("prior_weight")) is not None})
    any_data = False
    for prior in ["none", "euclidean", "spectral"]:
        xs, ys, yerr_low, yerr_high = [], [], [], []
        for weight in weights:
            match = next(
                (
                    row
                    for row in h16
                    if row.get("prior") == prior
                    and f(row.get("prior_weight")) is not None
                    and abs(f(row.get("prior_weight")) - weight) < 1e-12
                ),
                None,
            )
            if not match:
                continue
            mean = f(match.get("mean"))
            ci_low = f(match.get("ci95_low"))
            ci_high = f(match.get("ci95_high"))
            if mean is None:
                continue
            xs.append(weight)
            ys.append(mean)
            yerr_low.append(mean - (ci_low if ci_low is not None else mean))
            yerr_high.append((ci_high if ci_high is not None else mean) - mean)
        if xs:
            any_data = True
            ax.errorbar(
                xs,
                ys,
                yerr=[yerr_low, yerr_high],
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=prior,
                color=COLORS.get(prior, None),
            )
    ax.set_xscale("log")
    if any_data:
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
    save_both(fig, out_base)


def plot_laplacian_ablation(rows: list[dict[str, str]], out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    setup_axis(ax, "rMD17 Aspirin Laplacian Ablation", "Rollout horizon", "Latent rollout error")
    modes = ["per_frame", "fixed_frame0", "fixed_mean"]
    any_data = False
    for mode in modes:
        xs, ys, low, high = [], [], [], []
        for horizon in HORIZONS:
            match = next((row for row in rows if row.get("laplacian_mode") == mode and row.get("horizon") == str(horizon)), None)
            if not match:
                continue
            mean = f(match.get("mean"))
            ci_low = f(match.get("ci95_low"))
            ci_high = f(match.get("ci95_high"))
            if mean is None:
                continue
            xs.append(horizon)
            ys.append(mean)
            low.append(ci_low if ci_low is not None else mean)
            high.append(ci_high if ci_high is not None else mean)
        if xs:
            any_data = True
            color = COLORS.get(mode, None)
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=mode, color=color)
            ax.fill_between(xs, low, high, color=color, alpha=0.16)
    ax.set_xscale("log", base=2)
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels([str(h) for h in HORIZONS])
    if any_data:
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
    save_both(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", default="analysis_out")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    analysis = Path(args.analysis)
    plots = analysis / "plots"
    outputs = [
        plots / "rmd17_aspirin_horizon_scaling",
        plots / "rmd17_weight_sweep",
        plots / "rmd17_laplacian_ablation",
        plots / "wolfram_horizon_scaling",
    ]
    if args.dry_run:
        for output in outputs:
            print(f"would write {output.with_suffix('.png')} and {output.with_suffix('.pdf')}")
        return

    plot_horizon_scaling(
        read_csv(analysis / "aggregate_by_prior_horizon.csv"),
        plots / "rmd17_aspirin_horizon_scaling",
        "rMD17 Aspirin Horizon Scaling",
    )
    plot_weight_sweep(
        read_csv(analysis / "aggregate_weight_sweep.csv"),
        plots / "rmd17_weight_sweep",
    )
    plot_laplacian_ablation(
        read_csv(analysis / "aggregate_laplacian_ablation.csv"),
        plots / "rmd17_laplacian_ablation",
    )
    plot_horizon_scaling(
        read_csv(analysis / "aggregate_wolfram.csv"),
        plots / "wolfram_horizon_scaling",
        "Wolfram Flat 200-Epoch Horizon Scaling",
    )
    print(f"wrote plots to {plots}")


if __name__ == "__main__":
    main()
