"""
Figure 1: closely modeled on GPT visual reference.
Key visual decisions: equal-height boxes, bullet-style body with proper
bullets, large central icons, emphasis arrows on the right half,
"attribution" italic label below the 3->4 arrow.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


# ============ Color palette ============
C_INPUT_FILL = "#eaeef3"
C_INPUT_EDGE = "#7d8a99"
C_PRIOR_FILL = "#ede4f5"
C_PRIOR_EDGE = "#9d6ee5"
C_TRAIN_FILL = "#e1edf7"
C_TRAIN_EDGE = "#5a9fd4"
C_CTRL_FILL = "#fdf3d8"
C_CTRL_EDGE = "#cba84d"
C_OUT_FILL = "#ffffff"
C_OUT_EDGE = "#7d8a99"

EDGE_NORMAL = "#3a3a3a"
EDGE_BOLD = "#1f4e79"
TITLE_DARK = "#1f3a5f"

C_TOPOLOGY = "#2e7d32"
C_GENERIC = "#d4a017"
C_NOEFFECT = "#c43838"


def draw_input_icon(ax, cx, cy, scale=1.0):
    s = scale
    nodes = [
        (cx - 0.32 * s, cy + 0.10 * s),
        (cx + 0.32 * s, cy + 0.10 * s),
        (cx, cy - 0.28 * s),
    ]
    edges = [(0, 1), (0, 2), (1, 2)]
    for i, j in edges:
        ax.plot(
            [nodes[i][0], nodes[j][0]],
            [nodes[i][1], nodes[j][1]],
            color="#222",
            linewidth=1.6,
            zorder=2,
        )
    for x, y in nodes:
        ax.add_patch(
            Circle(
                (x, y),
                0.10 * s,
                facecolor="#cfd8e3",
                edgecolor="#222",
                linewidth=1.6,
                zorder=3,
            )
        )


def draw_prior_icon(ax, cx, cy, scale=1.0):
    s = scale
    rows = 5
    bw, bh = 0.70 * s, 0.08 * s
    gap = 0.03 * s
    total_h = rows * bh + (rows - 1) * gap
    y0 = cy + total_h / 2 - bh
    colors = ["#5a259e", "#7c3aed", "#9d6ee5", "#bf99ee", "#d9bff4"]
    for i in range(rows):
        y = y0 - i * (bh + gap)
        ax.add_patch(
            Rectangle(
                (cx - bw / 2, y),
                bw,
                bh,
                facecolor=colors[i],
                edgecolor="#3a1463",
                linewidth=0.6,
                zorder=2,
            )
        )


def draw_train_icon(ax, cx, cy, scale=1.0):
    s = scale
    ax.plot(
        [cx - 0.36 * s, cx + 0.36 * s],
        [cy - 0.30 * s, cy - 0.30 * s],
        color="#222",
        linewidth=1.2,
        zorder=1,
    )
    ax.plot(
        [cx - 0.36 * s, cx - 0.36 * s],
        [cy - 0.30 * s, cy + 0.32 * s],
        color="#222",
        linewidth=1.2,
        zorder=1,
    )

    xs = np.linspace(cx - 0.32 * s, cx + 0.34 * s, 80)
    ts = np.linspace(0, 1, 80)
    ys = cy + 0.28 * s - 0.54 * s * (1 - np.exp(-3.5 * ts))
    ax.plot(xs, ys, color="#1f4e79", linewidth=2.2, zorder=2)

    ax.text(
        cx - 0.34 * s,
        cy + 0.34 * s,
        "Loss",
        fontsize=7.5,
        color="#222",
        ha="left",
        va="bottom",
        family="sans-serif",
    )
    ax.text(
        cx + 0.34 * s,
        cy - 0.34 * s,
        "Steps",
        fontsize=7.5,
        color="#222",
        ha="right",
        va="top",
        family="sans-serif",
    )


def draw_ctrl_icon(ax, cx, cy, scale=1.0):
    s = scale
    bw = 0.16 * s
    gap = 0.05 * s
    heights = [0.42 * s, 0.46 * s, 0.32 * s]
    err = [0.06 * s, 0.06 * s, 0.06 * s]
    colors = [C_TOPOLOGY, C_GENERIC, C_NOEFFECT]
    total_w = 3 * bw + 2 * gap
    x0 = cx - total_w / 2
    base_y = cy - 0.28 * s
    for i, (h, e, c) in enumerate(zip(heights, err, colors)):
        x = x0 + i * (bw + gap)
        ax.add_patch(
            Rectangle(
                (x, base_y),
                bw,
                h,
                facecolor=c,
                edgecolor="#222",
                linewidth=0.8,
                zorder=2,
            )
        )
        cap_y = base_y + h
        ax.plot(
            [x + bw / 2, x + bw / 2],
            [cap_y, cap_y + e],
            color="#222",
            linewidth=1.0,
            zorder=3,
        )
        ax.plot(
            [x + bw / 2 - 0.04 * s, x + bw / 2 + 0.04 * s],
            [cap_y + e, cap_y + e],
            color="#222",
            linewidth=1.0,
            zorder=3,
        )


def draw_stage_box(
    ax,
    x,
    y,
    w,
    h,
    fill,
    edge,
    edge_w,
    stage_num,
    title,
    body_lines,
    icon_fn,
    stage5=False,
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=edge_w,
        edgecolor=edge,
        facecolor=fill,
        zorder=1,
    )
    ax.add_patch(box)

    ax.text(
        x + 0.18,
        y + h - 0.20,
        f"STAGE {stage_num}",
        fontsize=10,
        color="#888",
        fontweight="bold",
        ha="left",
        va="top",
        family="sans-serif",
    )
    ax.text(
        x + 0.18,
        y + h - 0.50,
        title,
        fontsize=15,
        color="#1a1a1a",
        fontweight="bold",
        ha="left",
        va="top",
        family="sans-serif",
    )

    icon_cx = x + w / 2
    icon_cy = y + h * 0.58
    if icon_fn is not None:
        icon_fn(ax, icon_cx, icon_cy, scale=1.45)
    elif stage5:
        for j, color in enumerate([C_TOPOLOGY, C_GENERIC, C_NOEFFECT]):
            ax.add_patch(
                Circle(
                    (icon_cx, icon_cy + (1 - j) * 0.38),
                    0.15,
                    facecolor=color,
                    edgecolor="#1a1a1a" if j == 0 else "none",
                    linewidth=0.4,
                    alpha=0.95,
                    zorder=3,
                )
            )

    # Keep all bullet rows inside the card. The prior-set card has five rows,
    # so the body block starts higher than the original draft.
    body_y = y + 1.45
    line_h = 0.30
    for line in body_lines:
        if stage5:
            dot_color, text = line
            dot_x = x + 0.30
            ax.add_patch(
                Circle(
                    (dot_x, body_y - 0.05),
                    0.065,
                    facecolor=dot_color,
                    edgecolor="white",
                    linewidth=1.0,
                    zorder=3,
                )
            )
            ax.text(
                x + 0.46,
                body_y,
                text,
                    fontsize=10.5,
                color="#1a1a1a",
                ha="left",
                va="top",
                family="sans-serif",
            )
        else:
            ax.text(
                x + 0.20,
                body_y,
                "\u2022  " + line,
                    fontsize=9.6,
                color="#1a1a1a",
                ha="left",
                va="top",
                family="sans-serif",
            )
        body_y -= line_h


def main():
    fig = plt.figure(figsize=(20, 5.0))
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.89])
    ax.axis("off")

    fig.text(
        0.5,
        0.94,
        "Model-conditioned preflight protocol for prior selection",
        fontsize=20,
        color=TITLE_DARK,
        fontweight="bold",
        ha="center",
        va="center",
        family="sans-serif",
    )

    n_stages = 5
    box_w = 4.4
    box_h = 4.2
    gap = 0.55

    fills = [C_INPUT_FILL, C_PRIOR_FILL, C_TRAIN_FILL, C_CTRL_FILL, C_OUT_FILL]
    edges = [C_INPUT_EDGE, C_PRIOR_EDGE, C_TRAIN_EDGE, C_CTRL_EDGE, C_OUT_EDGE]
    edge_widths = [1.2, 1.2, 1.2, 1.6, 1.2]

    titles = ["Inputs", "Prior set", "Training", "Controls", "Attribution class"]
    bodies = [
        [
            r"Node-wise trajectories $X_t \in \mathbb{R}^{N \times d}$",
            r"Candidate graph Laplacian $L$",
            "Graph provenance metadata",
        ],
        [
            "none (no auxiliary prior)",
            "graph_laplacian",
            "permuted_graph (SMPG)",
            "temporal_smooth (calibrated)",
            "random_graph (optional)",
        ],
        [r"Rollout at $H{=}32$", "5 seeds per cell", "Fixed model condition", "Standardized run artifacts"],
        [
            "Effect size vs no-prior baseline",
            "Effect size vs SMPG control",
            "Effect size vs temporal smoothing",
            r"Audit-mode metrics: $D_{true}, R_{low}$",
        ],
        [
            (C_TOPOLOGY, "topology-aligned support"),
            (C_GENERIC, "generic regularization"),
            (C_NOEFFECT, "no effect"),
        ],
    ]
    icons = [draw_input_icon, draw_prior_icon, draw_train_icon, draw_ctrl_icon, None]

    x_cursor = 0.4
    centers_x = []
    for i in range(n_stages):
        y = 0.25
        draw_stage_box(
            ax,
            x_cursor,
            y,
            box_w,
            box_h,
            fills[i],
            edges[i],
            edge_widths[i],
            i + 1,
            titles[i],
            bodies[i],
            icons[i],
            stage5=i == 4,
        )
        centers_x.append(x_cursor + box_w / 2)
        x_cursor += box_w + gap

    y_mid = 0.25 + box_h / 2

    for i in range(n_stages - 1):
        x_from = centers_x[i] + box_w / 2 + 0.05
        x_to = centers_x[i + 1] - box_w / 2 - 0.05
        emphasis = i >= 2
        arrow = FancyArrowPatch(
            (x_from, y_mid),
            (x_to, y_mid),
            arrowstyle="-|>",
            mutation_scale=28 if emphasis else 18,
            linewidth=2.8 if emphasis else 1.4,
            color=EDGE_BOLD if emphasis else EDGE_NORMAL,
            zorder=4,
        )
        ax.add_patch(arrow)
        if i == 2:
            mid_x = (x_from + x_to) / 2
            ax.text(
                mid_x,
                y_mid - 0.30,
                "attribution",
                fontsize=12,
                color=EDGE_BOLD,
                fontweight="bold",
                ha="center",
                va="top",
                fontstyle="italic",
            )

    ax.set_xlim(0, x_cursor + 0.4)
    ax.set_ylim(0, box_h + 0.70)

    out_dir = Path(__file__).resolve().parent.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_stems = ["fig_protocol_overview", "fig1_protocol_schematic"]
    for stem in output_stems:
        pdf_path = out_dir / f"{stem}.pdf"
        png_path = out_dir / f"{stem}.png"
        fig.savefig(pdf_path, bbox_inches="tight", dpi=300, facecolor="white")
        fig.savefig(png_path, bbox_inches="tight", dpi=200, facecolor="white")
        print(f"Wrote: {pdf_path}")
        print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
