"""
Figure 2: Decision tree for attribution-class assignment.
Closely modeled on the GPT-generated visual reference.
Output: paper/figures/fig_decision_tree.pdf (and .png)
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ============ Color palette ============
ENTRY_FILL = "#dbe7f3"
ENTRY_EDGE = "#1f4e79"

Q_FILL = "#e7f0fa"
Q_EDGE = "#1f4e79"

C_TOPOLOGY = "#2e7d32"
C_TOPOLOGY_LIGHT = "#e8f4ea"
C_GENERIC = "#d4a017"
C_GENERIC_LIGHT = "#fef6e4"
C_NOEFFECT = "#c43838"
C_NOEFFECT_LIGHT = "#fce4e4"

EDGE_NORMAL = "#222222"
EDGE_YES = "#1f4e79"
TITLE_DARK = "#1a1a1a"


def add_box(ax, cx, cy, w, h, fill, edge, edge_w=1.6, content_lines=None):
    """Generic rounded box. content_lines is list of (text, fontsize, color, weight, style)."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.06",
        linewidth=edge_w,
        edgecolor=edge,
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(box)

    if content_lines:
        n = len(content_lines)
        spacing = 0.24
        total_h = (n - 1) * spacing
        y0 = cy + total_h / 2
        for i, (text, fs, color, weight, style) in enumerate(content_lines):
            ax.text(
                cx,
                y0 - i * spacing,
                text,
                fontsize=fs,
                color=color,
                fontweight=weight,
                fontstyle=style,
                ha="center",
                va="center",
                family="sans-serif",
            )


def add_yes_arrow(ax, x1, y1, x2, y2, label="YES"):
    """Thick blue YES arrow with white-boxed label in middle."""
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=22,
        linewidth=3.0,
        color=EDGE_YES,
        zorder=1,
    )
    ax.add_patch(arrow)
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    ax.text(
        mx,
        my,
        label,
        fontsize=10.5,
        color=EDGE_YES,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor=EDGE_YES,
            linewidth=1.3,
        ),
        zorder=3,
    )


def add_no_arrow(ax, x1, y1, x2, y2, label="NO", label_offset=(0, 0.15)):
    """Thin black NO arrow with optional plain text label above."""
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        color=EDGE_NORMAL,
        zorder=1,
    )
    ax.add_patch(arrow)
    if label is None:
        return
    mx = (x1 + x2) / 2 + label_offset[0]
    my = (y1 + y2) / 2 + label_offset[1]
    ax.text(
        mx,
        my,
        label,
        fontsize=10.5,
        color="#1a1a1a",
        fontweight="bold",
        ha="center",
        va="center",
    )


def main():
    fig = plt.figure(figsize=(9.6, 11.5))
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.94])
    ax.axis("off")

    fig.text(
        0.5,
        0.97,
        "Decision logic for attribution-class assignment",
        fontsize=16,
        color=TITLE_DARK,
        fontweight="bold",
        ha="center",
        va="center",
        family="sans-serif",
    )

    cx_main = 4.15
    cx_right_terminal = 9.1

    y_entry = 11.3
    y_q1 = 9.7
    y_q2 = 7.0
    y_q3 = 4.3
    y_topology = 1.7

    y_no_effect = y_q1
    y_generic = y_q2

    entry_w, entry_h = 5.0, 0.85
    q_w, q_h = 4.65, 1.45
    t_w, t_h = 3.0, 1.45
    top_w, top_h = 4.25, 1.35

    add_box(
        ax,
        cx_main,
        y_entry,
        entry_w,
        entry_h,
        fill=ENTRY_FILL,
        edge=ENTRY_EDGE,
        edge_w=1.6,
        content_lines=[
            ("Run preflight protocol - 13 measurement cells", 11.3, TITLE_DARK, "bold", "normal"),
        ],
    )

    add_box(
        ax,
        cx_main,
        y_q1,
        q_w,
        q_h,
        fill=Q_FILL,
        edge=Q_EDGE,
        edge_w=1.6,
        content_lines=[
            ("Q1. Does graph beat no-prior", 10.9, EDGE_YES, "bold", "normal"),
            (r"on $H{=}32$ rollout?", 10.9, EDGE_YES, "bold", "normal"),
            ("non-overlapping CI vs none", 9.5, "#666", "normal", "italic"),
        ],
    )

    add_box(
        ax,
        cx_main,
        y_q2,
        q_w,
        q_h,
        fill=Q_FILL,
        edge=Q_EDGE,
        edge_w=1.6,
        content_lines=[
            ("Q2. Does graph beat the SMPG control?", 10.9, EDGE_YES, "bold", "normal"),
            ("non-overlapping CI vs SMPG", 9.5, "#666", "normal", "italic"),
        ],
    )

    add_box(
        ax,
        cx_main,
        y_q3,
        q_w,
        q_h,
        fill=Q_FILL,
        edge=Q_EDGE,
        edge_w=1.6,
        content_lines=[
            ("Q3. Do audit-mode latent metrics separate?", 10.9, EDGE_YES, "bold", "normal"),
            (r"$D_{true},  R_{low}$", 10, "#666", "normal", "italic"),
        ],
    )

    add_box(
        ax,
        cx_right_terminal,
        y_no_effect,
        t_w,
        t_h,
        fill=C_NOEFFECT_LIGHT,
        edge=C_NOEFFECT,
        edge_w=1.8,
        content_lines=[
            ("No effect", 12, TITLE_DARK, "bold", "normal"),
            ("4 cells", 10.5, C_NOEFFECT, "bold", "normal"),
            ("spring-mass, graph-wave,", 8.5, "#444", "normal", "italic"),
            ("METR-LA, graph-heat", 8.5, "#444", "normal", "italic"),
        ],
    )

    add_box(
        ax,
        cx_right_terminal,
        y_generic,
        t_w,
        t_h,
        fill=C_GENERIC_LIGHT,
        edge=C_GENERIC,
        edge_w=1.8,
        content_lines=[
            ("Generic regularization", 12, TITLE_DARK, "bold", "normal"),
            ("8 cells", 10.5, C_GENERIC, "bold", "normal"),
            (r"N-body $k{=}4, 8, 12$ +", 8.5, "#444", "normal", "italic"),
            ("5 rMD17 molecules", 8.5, "#444", "normal", "italic"),
        ],
    )

    add_box(
        ax,
        cx_main,
        y_topology,
        top_w,
        top_h,
        fill=C_TOPOLOGY_LIGHT,
        edge=C_TOPOLOGY,
        edge_w=2.0,
        content_lines=[
            ("Topology-aligned support", 12.5, TITLE_DARK, "bold", "normal"),
            ("1 cell", 10.5, C_TOPOLOGY, "bold", "normal"),
            ("HO lattice audit  (controlled audit-positive)", 8.8, "#444", "normal", "italic"),
        ],
    )

    add_no_arrow(
        ax,
        cx_main,
        y_entry - entry_h / 2,
        cx_main,
        y_q1 + q_h / 2,
        label=None,
        label_offset=(0, 0),
    )
    add_yes_arrow(ax, cx_main, y_q1 - q_h / 2, cx_main, y_q2 + q_h / 2)
    add_no_arrow(
        ax,
        cx_main + q_w / 2,
        y_q1,
        cx_right_terminal - t_w / 2,
        y_no_effect,
        label="NO",
        label_offset=(0, 0.18),
    )
    add_yes_arrow(ax, cx_main, y_q2 - q_h / 2, cx_main, y_q3 + q_h / 2)
    add_no_arrow(
        ax,
        cx_main + q_w / 2,
        y_q2,
        cx_right_terminal - t_w / 2,
        y_generic,
        label="NO",
        label_offset=(0, 0.18),
    )
    add_yes_arrow(ax, cx_main, y_q3 - q_h / 2, cx_main, y_topology + top_h / 2)

    ax.text(
        cx_main,
        0.5,
        "Cell counts:  1 (topology-aligned support)  +  8 (generic regularization)  +  4 (no effect)  =  13 measurement cells",
        fontsize=9,
        color="#555",
        fontstyle="italic",
        ha="center",
        va="center",
    )

    ax.set_xlim(-0.1, 11.25)
    ax.set_ylim(0.0, 12.0)
    ax.set_aspect("equal")

    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_stems = ["fig_decision_tree", "fig2_decision_logic"]
    for stem in output_stems:
        pdf_path = out_dir / f"{stem}.pdf"
        png_path = out_dir / f"{stem}.png"
        fig.savefig(pdf_path, bbox_inches="tight", dpi=300, facecolor="white")
        fig.savefig(png_path, bbox_inches="tight", dpi=200, facecolor="white")
        print(f"Wrote: {pdf_path}")
        print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
