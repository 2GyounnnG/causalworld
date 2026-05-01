#!/usr/bin/env python3
"""Generate JCS manuscript figures from prepared figure_data CSV files.

This script is an asset-generation utility only. It reads from paper/figure_data
and writes PNGs to paper/figures. It does not run experiments or touch model code.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-causalworld-jcs")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PAPER_DIR / "figure_data"
FIG_DIR = PAPER_DIR / "figures"


PRIOR_ORDER = ["none", "graph_laplacian", "permuted_graph", "temporal_smooth", "random_graph"]
PRIOR_LABEL = {
    "none": "None",
    "graph_laplacian": "Graph",
    "permuted_graph": "Permuted",
    "temporal_smooth": "Temporal",
    "random_graph": "Random",
}
PRIOR_COLOR = {
    "none": "#6b7280",
    "graph_laplacian": "#2563eb",
    "permuted_graph": "#f59e0b",
    "temporal_smooth": "#059669",
    "random_graph": "#7c3aed",
}
LABEL_COLOR = {
    "topology_aligned_latent_smoothing": "#2563eb",
    "candidate_topology_specific": "#1d4ed8",
    "temporal_smoothing_sufficient": "#059669",
    "graph_generic_smoothing": "#f59e0b",
    "low_budget_only": "#d97706",
    "no_prior_gain": "#dc2626",
    "overconstrained": "#991b1b",
    "inconclusive": "#6b7280",
}
FIG3_SHORT_LABEL = {
    "HO lattice": "HO lattice",
    "Graph heat lattice": "Heat lattice",
    "Graph low-frequency lattice": "Low-frequency\nlattice",
    "Spring-mass quick + temporal": "Spring-mass\nquick + temporal",
    "Spring-mass standard": "Spring-mass\nstandard",
    "Graph-wave quick + temporal": "Graph-wave\nquick + temporal",
    "Graph-wave standard": "Graph-wave\nstandard",
    "N-body quick + temporal": "N-body\nquick + temporal",
    "N-body standard": "N-body\nstandard",
    "METR-LA correlation": "METR-LA\ncorrelation",
}
FIG4_SHORT_LABEL = {
    "Spring-mass quick": "Spring-mass\nquick",
    "Graph-wave quick": "Graph-wave\nquick",
    "N-body quick": "N-body\nquick",
}
LABEL_LEGEND = {
    "topology_aligned_latent_smoothing": "Topology-aligned audit",
    "candidate_topology_specific": "Candidate topology-specific",
    "temporal_smoothing_sufficient": "Temporal sufficient",
    "graph_generic_smoothing": "Generic graph smoothing",
    "low_budget_only": "Low-budget only",
    "no_prior_gain": "No prior gain",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str | None) -> float | None:
    if value is None or value == "" or value == "missing":
        return None
    return float(value)


def savefig(fig: plt.Figure, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def make_fig3() -> None:
    rows = read_csv(DATA_DIR / "fig3_preflight_classification_overview.csv")
    rows = list(reversed(rows))
    labels = [FIG3_SHORT_LABEL.get(r["display_case"], r["display_case"]) for r in rows]
    gains = [as_float(r["graph_gain_vs_none_pct"]) or 0.0 for r in rows]
    colors = [LABEL_COLOR.get(r["protocol_label"], "#6b7280") for r in rows]

    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    y = list(range(len(rows)))
    ax.barh(y, gains, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Graph gain vs no prior at H=32 (%)", fontsize=10)
    ax.set_title("Preflight labels separate utility from attribution", fontsize=12, pad=12)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.tick_params(axis="x", labelsize=9)

    legend_labels = [
        "topology_aligned_latent_smoothing",
        "candidate_topology_specific",
        "temporal_smoothing_sufficient",
        "graph_generic_smoothing",
        "low_budget_only",
        "no_prior_gain",
    ]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=LABEL_COLOR[label], label=LABEL_LEGEND[label])
        for label in legend_labels
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    savefig(fig, "fig3_preflight_classification_overview.png")


def make_fig4() -> None:
    rows = read_csv(DATA_DIR / "fig4_prior_family_comparison.csv")
    cases = []
    for row in rows:
        if row["display_case"] not in cases:
            cases.append(row["display_case"])
    values = {(r["display_case"], r["prior_family"]): as_float(r["h32_rollout"]) for r in rows}
    priors = ["none", "graph_laplacian", "permuted_graph", "temporal_smooth"]

    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    x = [i * 1.35 for i in range(len(cases))]
    width = 0.17
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for prior, offset in zip(priors, offsets):
        ys = [values.get((case, prior), 0.0) or 0.0 for case in cases]
        ax.bar(
            [xi + offset for xi in x],
            ys,
            width=width,
            label=PRIOR_LABEL[prior],
            color=PRIOR_COLOR[prior],
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([FIG4_SHORT_LABEL.get(case, case) for case in cases], fontsize=10)
    ax.set_ylabel("H=32 rollout error (lower is better)", fontsize=10)
    ax.set_title("Calibrated temporal smoothing changes prior attribution", fontsize=12, pad=12)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(
        ncol=4,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        frameon=False,
    )
    savefig(fig, "fig4_prior_family_comparison.png")


def make_fig5() -> None:
    rows = read_csv(DATA_DIR / "fig5_training_budget_dependence.csv")
    datasets = []
    for row in rows:
        if row["display_case"] not in datasets:
            datasets.append(row["display_case"])
    priors = ["none", "graph_laplacian", "permuted_graph"]

    fig, axes = plt.subplots(1, len(datasets), figsize=(10.0, 3.8), sharex=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        subset = [r for r in rows if r["display_case"] == dataset]
        for prior in priors:
            prior_rows = sorted(
                [r for r in subset if r["prior_family"] == prior],
                key=lambda r: int(r["epochs"]),
            )
            xs = [int(r["epochs"]) for r in prior_rows]
            ys = [as_float(r["h32_rollout"]) or 0.0 for r in prior_rows]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.8,
                label=PRIOR_LABEL[prior],
                color=PRIOR_COLOR[prior],
            )
        ax.set_title(dataset)
        ax.set_xticks([5, 20])
        ax.set_xlabel("Epochs")
        ax.grid(color="#e5e7eb", linewidth=0.8)
    axes[0].set_ylabel("H=32 rollout error")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Prior recommendations are training-budget dependent", y=1.03)
    savefig(fig, "fig5_training_budget_dependence.png")


def make_fig6() -> None:
    rows = read_csv(DATA_DIR / "fig6_latent_audit_positive_case.csv")
    metrics = ["h32_rollout", "d_true_delta_h_norm", "r_low_k4"]
    priors = ["graph_laplacian", "permuted_graph", "random_graph"]

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.8))
    for ax, metric in zip(axes, metrics):
        subset = [r for r in rows if r["metric"] == metric]
        label = subset[0]["metric_label"] if subset else metric
        means = []
        stds = []
        for prior in priors:
            row = next(r for r in subset if r["prior_family"] == prior)
            means.append(as_float(row["mean"]) or 0.0)
            stds.append(as_float(row["std"]) or 0.0)
        x = list(range(len(priors)))
        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=3,
            color=[PRIOR_COLOR[p] for p in priors],
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([PRIOR_LABEL[p] for p in priors], rotation=20, ha="right")
        ax.set_title(label)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    fig.suptitle("HO lattice audit: rollout gain aligns with latent smoothing", y=1.03)
    savefig(fig, "fig6_latent_audit_positive_case.png")


def main() -> None:
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()


if __name__ == "__main__":
    main()
