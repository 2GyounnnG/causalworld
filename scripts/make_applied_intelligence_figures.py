from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch


OUT_DIR = Path("paper/applied_intelligence/figures")
COLORS = {
    "none": "#4c4c4c",
    "euclidean": "#1f77b4",
    "spectral": "#d62728",
    "per_frame": "#d62728",
    "fixed_mean": "#2ca02c",
    "fixed_frame0": "#9467bd",
}
MARKERS = {
    "none": "o",
    "euclidean": "s",
    "spectral": "^",
    "per_frame": "o",
    "fixed_mean": "s",
    "fixed_frame0": "^",
}


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def to_float(value: str) -> float:
    return float(value) if value not in {"", None} else float("nan")


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_line_with_ci(ax: plt.Axes, rows: list[dict[str, str]], label_field: str, label: str) -> None:
    selected = sorted([row for row in rows if row.get(label_field) == label], key=lambda row: to_float(row["horizon"]))
    x = [to_float(row["horizon"]) for row in selected]
    y = [to_float(row["mean"]) for row in selected]
    low = [to_float(row["ci95_low"]) for row in selected]
    high = [to_float(row["ci95_high"]) for row in selected]
    lower = [max(0.0, mean - lo) for mean, lo in zip(y, low)]
    upper = [hi - mean for mean, hi in zip(y, high)]
    ax.errorbar(
        x,
        y,
        yerr=[lower, upper],
        marker=MARKERS.get(label, "o"),
        color=COLORS.get(label, None),
        linewidth=1.8,
        markersize=5,
        capsize=3,
        label=label,
    )


def fig1_conceptual_priors() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))

    ax = axes[0]
    points = [(-0.9, -0.15), (-0.5, 0.35), (0.05, -0.25), (0.45, 0.2), (0.85, -0.05)]
    for x, y in points:
        ax.add_patch(Circle((x, y), 0.055, color=COLORS["euclidean"]))
    ax.add_patch(Ellipse((0, 0), 2.1, 0.85, angle=0, fill=False, edgecolor=COLORS["euclidean"], linewidth=2))
    ax.add_patch(Ellipse((0, 0), 1.35, 0.55, angle=0, fill=False, edgecolor=COLORS["euclidean"], linewidth=1, linestyle="--"))
    ax.text(0, 0.75, "Euclidean covariance prior", ha="center", weight="bold")
    ax.text(0, -0.75, "Regularizes latent cloud\nwithout graph structure", ha="center")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[1]
    nodes = {
        0: (-0.9, -0.25),
        1: (-0.45, 0.35),
        2: (0.1, 0.1),
        3: (0.55, 0.45),
        4: (0.9, -0.25),
    }
    edges = [(0, 1), (1, 2), (2, 3), (2, 4), (0, 2)]
    for i, j in edges:
        ax.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]], color="#777777", linewidth=1.5)
    for idx, (x, y) in nodes.items():
        ax.add_patch(Circle((x, y), 0.075, color=COLORS["spectral"]))
        ax.text(x, y - 0.22, f"v{idx}", ha="center", fontsize=8)
    ax.add_patch(FancyArrowPatch((-0.95, -0.72), (0.95, -0.72), arrowstyle="<->", mutation_scale=12, color=COLORS["spectral"]))
    ax.text(0, 0.82, "Graph-spectral prior", ha="center", weight="bold")
    ax.text(0, -0.95, "Uses Laplacian geometry\nto shape latent variation", ha="center")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.1, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle("Latent prior geometry", y=1.02)
    save_figure(fig, "fig1_conceptual_priors")


def fig2_rmd17_horizon_scaling() -> None:
    rows = read_csv("analysis_out/aggregate_by_prior_horizon.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for prior in ["none", "euclidean", "spectral"]:
        plot_line_with_ci(ax, rows, "prior", prior)
    ax.set_title("rMD17 aspirin horizon scaling")
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel("Rollout error (lower is better)")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "fig2_rmd17_horizon_scaling")


def fig3_laplacian_ablation() -> None:
    rows = read_csv("analysis_out/aggregate_laplacian_ablation.csv")
    main_rows = read_csv("analysis_out/aggregate_by_prior_horizon.csv")
    baseline = next(
        to_float(row["mean"])
        for row in main_rows
        if row.get("prior") == "none" and row.get("horizon") == "16"
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for mode in ["per_frame", "fixed_mean", "fixed_frame0"]:
        plot_line_with_ci(ax, rows, "laplacian_mode", mode)
    ax.axhline(baseline, color=COLORS["none"], linestyle="--", linewidth=1.4, label="none H=16 baseline")
    ax.set_title("rMD17 spectral Laplacian ablation")
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel("Rollout error (lower is better)")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, "fig3_laplacian_ablation")


def fig4_weight_sweep_h16() -> None:
    rows = [row for row in read_csv("analysis_out/aggregate_weight_sweep.csv") if row.get("horizon") == "16"]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    for prior in ["none", "euclidean", "spectral"]:
        selected = sorted(
            [row for row in rows if row.get("prior") == prior],
            key=lambda row: to_float(row["prior_weight"]),
        )
        x = [to_float(row["prior_weight"]) for row in selected]
        y = [to_float(row["mean"]) for row in selected]
        yerr = [to_float(row["stderr"]) for row in selected]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=MARKERS[prior],
            color=COLORS[prior],
            linewidth=1.8,
            markersize=5,
            capsize=3,
            label=prior,
        )
    ax.annotate(
        "spectral w=0.01\nunstable seeds",
        xy=(0.01, 1.908429248),
        xytext=(0.018, 0.82),
        arrowprops={"arrowstyle": "->", "color": COLORS["spectral"]},
        color=COLORS["spectral"],
        fontsize=9,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("rMD17 weight sweep at H=16")
    ax.set_xlabel("Prior weight")
    ax.set_ylabel("H=16 rollout error (log scale)")
    ax.set_xticks([0.001, 0.01, 0.1, 1.0])
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1.0"])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "fig4_weight_sweep_h16")


def fig5_wolfram_instability() -> None:
    ratio_rows = [
        row
        for row in read_csv("analysis_out/wolfram_horizon_ratios.csv")
        if row.get("level") == "seed" and row.get("prior") == "spectral"
    ]
    ratio_rows = sorted(ratio_rows, key=lambda row: to_float(row["h16"]), reverse=True)
    agg = read_csv("analysis_out/aggregate_wolfram.csv")
    spectral_h16 = next(row for row in agg if row.get("prior") == "spectral" and row.get("horizon") == "16")
    mean = to_float(spectral_h16["mean"])
    median = to_float(spectral_h16["median"])

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    seeds = [row["seed"] for row in ratio_rows]
    values = [to_float(row["h16"]) for row in ratio_rows]
    colors = [COLORS["spectral"] if value > 1.0 else "#999999" for value in values]
    ax.bar(range(len(values)), values, color=colors, width=0.72)
    ax.axhline(mean, color="#222222", linestyle="-", linewidth=1.4, label=f"spectral mean={mean:.2f}")
    ax.axhline(median, color="#222222", linestyle="--", linewidth=1.4, label=f"spectral median={median:.3f}")
    ax.set_yscale("log")
    ax.set_title("Wolfram spectral H=16 heavy-tail instability")
    ax.set_xlabel("Spectral seed (sorted by H=16)")
    ax.set_ylabel("H=16 rollout error (log scale)")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.98,
        0.92,
        "Seeds 2, 8, 1 dominate mean",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=COLORS["spectral"],
        fontsize=9,
    )
    save_figure(fig, "fig5_wolfram_instability")


def main() -> None:
    setup_style()
    fig1_conceptual_priors()
    fig2_rmd17_horizon_scaling()
    fig3_laplacian_ablation()
    fig4_weight_sweep_h16()
    fig5_wolfram_instability()
    print(f"wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
