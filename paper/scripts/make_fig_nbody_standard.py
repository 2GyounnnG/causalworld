"""
Generate Figure 5: N-body standard ep20 raw H32 across k=4/8/12,
4 prior families per panel.

Data source: paper/tables/nbody_robustness_5seed_summary.md
Output: paper/figures/fig_nbody_standard_3panel.pdf (and .png)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Data from nbody_robustness_5seed_summary.md, standard_ep20 rows.
# [None, Graph, Permuted, Temporal] mean +/- std (5 seeds each).
DATA = {
    "k=4": {
        "means": [0.1507, 0.0641, 0.0718, 0.0661],
        "stds": [0.0129, 0.0322, 0.0245, 0.0147],
    },
    "k=8": {
        "means": [0.1522, 0.0814, 0.0759, 0.0695],
        "stds": [0.0115, 0.0332, 0.0276, 0.0240],
    },
    "k=12": {
        "means": [0.1443, 0.0772, 0.0898, 0.0596],
        "stds": [0.0121, 0.0311, 0.0417, 0.0265],
    },
}

LABELS = ["None", "Graph", "Perm.", "Temp."]
COLORS = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]
N_SEEDS = 5
T_CRIT = 2.776  # t_{0.025, df=4} for 95% CI.


def ci95(std):
    """95% CI half-width using t-distribution with df=n-1=4."""
    return T_CRIT * std / np.sqrt(N_SEEDS)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    x = np.arange(4)

    for ax, (k_label, d) in zip(axes, DATA.items()):
        means = np.array(d["means"])
        ci = np.array([ci95(s) for s in d["stds"]])

        ax.bar(
            x,
            means,
            yerr=ci,
            capsize=4,
            color=COLORS,
            edgecolor="black",
            linewidth=0.7,
            error_kw={"elinewidth": 1.0, "ecolor": "black"},
        )

        ax.set_title(f"N-body {k_label}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, 0.20)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(r"$H{=}32$ rollout error (mean $\pm$ 95% CI)", fontsize=10)

    fig.suptitle(
        "N-body standard ep20: graph improves over no-prior, "
        "but does not separate from SMPG or temporal controls",
        fontsize=11,
        y=1.02,
    )

    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / "fig_nbody_standard_3panel.pdf"
    png_path = out_dir / "fig_nbody_standard_3panel.png"

    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=200)

    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
