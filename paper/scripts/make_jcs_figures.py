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
FIG3_LABELS = [
    ("no_graph_gain", "no graph gain"),
    ("quick_topology_signal", "quick_topology_signal"),
    ("candidate_graph_favorable", "candidate graph favorable\nunder this construction"),
    ("generic_smoothing", "generic_smoothing"),
    ("temporal_sufficient", "temporal_sufficient"),
    ("topology_aligned_audit", "topology_aligned_audit"),
]
FIG3_LABEL_COLOR = {
    "no_graph_gain": "#dc2626",
    "quick_topology_signal": "#60a5fa",
    "candidate_graph_favorable": "#1d4ed8",
    "generic_smoothing": "#f59e0b",
    "temporal_sufficient": "#059669",
    "topology_aligned_audit": "#7c3aed",
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
    "N-body quick": "N-body k=8\nquick",
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


def savefig_many(fig: plt.Figure, filenames: list[str]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for filename in filenames:
        path = FIG_DIR / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def make_fig3() -> None:
    matrix_rows = [
        ("HO lattice audit", ["topology_aligned_audit"], False),
        ("Graph heat lattice", ["no_graph_gain"], False),
        ("METR-LA correlation", ["no_graph_gain"], False),
        ("Spring-mass quick", ["temporal_sufficient"], False),
        ("Spring-mass standard", ["no_graph_gain"], False),
        ("Graph-wave quick", ["temporal_sufficient"], False),
        ("Graph-wave standard", ["no_graph_gain"], False),
        ("N-body k=4 quick", ["no_graph_gain"], True),
        ("N-body k=4 standard", ["candidate_graph_favorable"], True),
        ("N-body k=8 quick", ["quick_topology_signal"], True),
        ("N-body k=8 standard", ["temporal_sufficient", "generic_smoothing"], True),
        ("N-body k=12 quick", ["generic_smoothing"], True),
        ("N-body k=12 standard", ["temporal_sufficient"], True),
    ]

    fig, ax = plt.subplots(figsize=(11.8, 7.6))
    n_rows = len(matrix_rows)
    n_cols = len(FIG3_LABELS)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()

    for row_idx, (_name, active_labels, is_nbody) in enumerate(matrix_rows):
        if is_nbody:
            ax.axhspan(row_idx - 0.5, row_idx + 0.5, color="#eff6ff", zorder=0)
        for col_idx, (label_key, _label_text) in enumerate(FIG3_LABELS):
            rect = plt.Rectangle(
                (col_idx - 0.47, row_idx - 0.38),
                0.94,
                0.76,
                facecolor="#f9fafb",
                edgecolor="#d1d5db",
                linewidth=0.8,
                zorder=1,
            )
            ax.add_patch(rect)
            if label_key in active_labels:
                active = plt.Rectangle(
                    (col_idx - 0.39, row_idx - 0.30),
                    0.78,
                    0.60,
                    facecolor=FIG3_LABEL_COLOR[label_key],
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=2,
                )
                ax.add_patch(active)
                ax.text(
                    col_idx,
                    row_idx,
                    "✓",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color="white",
                    zorder=3,
                )

    ax.set_yticks(list(range(n_rows)))
    ax.set_yticklabels([row[0] for row in matrix_rows], fontsize=9)
    ax.set_xticks(list(range(n_cols)))
    ax.set_xticklabels([label for _key, label in FIG3_LABELS], fontsize=8)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        "Preflight labels are protocol interpretations, not a prior ranking",
        fontsize=12,
        y=0.995,
    )
    ax.set_title(
        "N-body rows vary distance-kNN construction and budget; graph gain alone is not attribution.",
        fontsize=9,
        color="#374151",
        pad=28,
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
    savefig_many(fig, ["fig4_prior_family_comparison.png", "fig4_prior_family_comparison.pdf"])


def make_fig5() -> None:
    priors = ["none", "graph_laplacian", "permuted_graph", "temporal_smooth"]
    rows = [
        {
            "budget": "quick_ep5",
            "k": 4,
            "label": "no graph gain",
            "values": {"none": 0.9198, "graph_laplacian": 0.9690, "permuted_graph": 0.9313, "temporal_smooth": 0.9655},
        },
        {
            "budget": "standard_ep20",
            "k": 4,
            "label": "candidate graph favorable",
            "values": {"none": 0.1494, "graph_laplacian": 0.0560, "permuted_graph": 0.0649, "temporal_smooth": 0.0679},
        },
        {
            "budget": "quick_ep5",
            "k": 8,
            "label": "quick_topology_signal",
            "values": {"none": 0.9150, "graph_laplacian": 0.8727, "permuted_graph": 0.9102, "temporal_smooth": 0.9599},
        },
        {
            "budget": "standard_ep20",
            "k": 8,
            "label": "temporal_sufficient / generic_smoothing",
            "values": {"none": 0.1582, "graph_laplacian": 0.0799, "permuted_graph": 0.0711, "temporal_smooth": 0.0642},
        },
        {
            "budget": "quick_ep5",
            "k": 12,
            "label": "generic_smoothing",
            "values": {"none": 0.9367, "graph_laplacian": 0.8210, "permuted_graph": 0.8017, "temporal_smooth": 0.8834},
        },
        {
            "budget": "standard_ep20",
            "k": 12,
            "label": "temporal_sufficient",
            "values": {"none": 0.1460, "graph_laplacian": 0.0763, "permuted_graph": 0.0946, "temporal_smooth": 0.0487},
        },
    ]
    by_panel = {(row["budget"], row["k"]): row for row in rows}

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6), sharey="row")
    budgets = [("quick_ep5", "quick ep5"), ("standard_ep20", "standard ep20")]
    ks = [4, 8, 12]
    x = list(range(len(priors)))
    for row_idx, (budget_key, budget_label) in enumerate(budgets):
        for col_idx, k_value in enumerate(ks):
            ax = axes[row_idx][col_idx]
            panel = by_panel[(budget_key, k_value)]
            values = [panel["values"][prior] for prior in priors]
            best_idx = min(range(len(values)), key=lambda idx: values[idx])
            edge_widths = [1.5 if idx == best_idx else 0.4 for idx in x]
            ax.bar(
                x,
                values,
                color=[PRIOR_COLOR[prior] for prior in priors],
                edgecolor="black",
                linewidth=edge_widths,
                width=0.72,
            )
            ax.set_title(f"k={k_value}", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(["None", "Graph", "Perm.", "Temp."], fontsize=8)
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
            ax.tick_params(axis="y", labelsize=8)
            ax.text(
                0.04,
                0.94,
                panel["label"].replace(" / ", "\n/ "),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.8,
                color="#111827",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.92},
            )
            if col_idx == 0:
                ax.set_ylabel(f"{budget_label}\nH=32 rollout error", fontsize=9)

    fig.suptitle("N-body recommendations depend on graph construction and budget", y=0.995, fontsize=12)
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
