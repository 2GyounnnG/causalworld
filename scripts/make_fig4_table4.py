from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = ROOT / "validation_wolfram_flat_10seed_200ep.json"
TABLE_PATH = ROOT / "paper" / "tables" / "table4_wolfram_main.tex"
FIGURE_DIR = ROOT / "paper" / "figures"
FIGURE_PDF = FIGURE_DIR / "fig4_wolfram_main.pdf"
FIGURE_PNG = FIGURE_DIR / "fig4_wolfram_main.png"

PRIORS = ["none", "euclidean", "spectral"]
HORIZONS = [1, 2, 4, 8, 16]
COLORS = {
    "none": "#888888",
    "euclidean": "#1f77b4",
    "spectral": "#d62728",
}
CATASTROPHIC_THRESHOLD = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Wolfram main-result Table 4 and Figure 4."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"Path to Wolfram 10-seed JSON (default: {DEFAULT_RESULTS_PATH})",
    )
    return parser.parse_args()


def load_results(path: Path) -> dict:
    expanded_path = path.expanduser()
    if not expanded_path.exists():
        raise SystemExit(
            f"Missing Wolfram main results file: {expanded_path}\n"
            "Pass a different path with --input /path/to/validation_wolfram_flat_10seed_200ep.json."
        )
    with expanded_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload["results"]


def result_key(prior: str) -> str:
    return f"flat|{prior}"


def validate_results(results: dict) -> None:
    missing = []
    for prior in PRIORS:
        key = result_key(prior)
        if key not in results:
            missing.append(key)
            continue
        for horizon in HORIZONS:
            h_key = str(horizon)
            if h_key not in results[key]:
                missing.append(f"{key}[{h_key}]")
            elif len(results[key][h_key]) != 10:
                raise ValueError(
                    f"Expected 10 values for {key}[{h_key}], got {len(results[key][h_key])}"
                )
    if missing:
        raise KeyError(f"Missing expected results: {missing}")


def summarize_values(values: list[float], horizon: int) -> dict[str, float | int | list[float]]:
    catastrophic = (
        sum(value > CATASTROPHIC_THRESHOLD for value in values) if horizon == 16 else 0
    )
    return {
        "values": values,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "max": max(values),
        "catastrophic": catastrophic,
    }


def summarize(results: dict) -> dict[str, dict[int, dict[str, float | int | list[float]]]]:
    stats: dict[str, dict[int, dict[str, float | int | list[float]]]] = {}
    for prior in PRIORS:
        stats[prior] = {}
        for horizon in HORIZONS:
            values = [
                float(value)
                for value in results[result_key(prior)][str(horizon)]
            ]
            stats[prior][horizon] = summarize_values(values, horizon)
    return stats


def mean_std_cell(cell: dict[str, float | int | list[float]]) -> str:
    return f"{float(cell['mean']):.3f} ± {float(cell['std']):.3f}"


def markdown_cell(prior: str, horizon: int, cell: dict[str, float | int | list[float]]) -> str:
    if prior == "spectral" and horizon == 16:
        return (
            f"{mean_std_cell(cell)} (median {float(cell['median']):.3f}, "
            f"max {float(cell['max']):.2f}, {int(cell['catastrophic'])} catastrophic)"
        )
    return f"{mean_std_cell(cell)} (median {float(cell['median']):.3f})"


def latex_cell(prior: str, horizon: int, cell: dict[str, float | int | list[float]]) -> str:
    if prior == "spectral" and horizon == 16:
        return (
            f"\\makecell{{{float(cell['mean']):.3f} $\\pm$ {float(cell['std']):.3f}\\\\"
            f"(median {float(cell['median']):.3f}, max {float(cell['max']):.1f})}}"
            "\\textsuperscript{\\dag}"
        )
    return (
        f"\\makecell{{{float(cell['mean']):.3f} $\\pm$ {float(cell['std']):.3f}\\\\"
        f"(median {float(cell['median']):.3f})}}"
    )


def spectral_median_is_best(stats: dict[str, dict[int, dict[str, float | int | list[float]]]]) -> bool:
    medians = {prior: float(stats[prior][16]["median"]) for prior in PRIORS}
    return medians["spectral"] == min(medians.values())


def maybe_bold_spectral(prior: str, text: str, bold_spectral: bool) -> str:
    if prior == "spectral" and bold_spectral:
        return f"\\textbf{{{text}}}"
    return text


def make_latex_table(stats: dict[str, dict[int, dict[str, float | int | list[float]]]]) -> str:
    bold_spectral = spectral_median_is_best(stats)
    spectral_h16 = stats["spectral"][16]
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Prior & H=1 & H=2 & H=4 & H=8 & H=16 \\\\",
        "\\midrule",
    ]
    for prior in PRIORS:
        row = [maybe_bold_spectral(prior, prior, bold_spectral)]
        row.extend(
            maybe_bold_spectral(prior, latex_cell(prior, horizon, stats[prior][horizon]), bold_spectral)
            for horizon in HORIZONS
        )
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            (
                "\\textsuperscript{\\dag} indicates heavy-tail: "
                f"{int(spectral_h16['catastrophic'])} of 10 seeds with H=16 > "
                f"{CATASTROPHIC_THRESHOLD:.1f}, max {float(spectral_h16['max']):.1f}."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def make_markdown_table(stats: dict[str, dict[int, dict[str, float | int | list[float]]]]) -> str:
    lines = [
        "| Prior | H=1 | H=2 | H=4 | H=8 | H=16 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for prior in PRIORS:
        row = [prior]
        row.extend(markdown_cell(prior, horizon, stats[prior][horizon]) for horizon in HORIZONS)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def make_figure(stats: dict[str, dict[int, dict[str, float | int | list[float]]]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    horizons = np.array(HORIZONS, dtype=float)

    for prior in PRIORS:
        values_by_horizon = [
            np.array(stats[prior][horizon]["values"], dtype=float)
            for horizon in HORIZONS
        ]
        medians = np.array([np.median(values) for values in values_by_horizon])
        q25 = np.array([np.percentile(values, 25) for values in values_by_horizon])
        q75 = np.array([np.percentile(values, 75) for values in values_by_horizon])
        color = COLORS[prior]
        ax.fill_between(horizons, q25, q75, color=color, alpha=0.18, linewidth=0)
        ax.plot(horizons, medians, marker="o", linewidth=2, color=color, label=prior)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels([str(horizon) for horizon in HORIZONS])
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel("Latent rollout error")
    ax.set_title("Wolfram CA: latent rollout error by horizon (n=10 seeds, median + IQR)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, bbox_inches="tight")
    plt.close(fig)


def write_outputs(stats: dict[str, dict[int, dict[str, float | int | list[float]]]]) -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(make_latex_table(stats), encoding="utf-8")
    make_figure(stats)


def format_raw_values(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in sorted(values)) + "]"


def sanity_status(stats: dict[str, dict[int, dict[str, float | int | list[float]]]]) -> list[tuple[str, str]]:
    checks = [
        (
            "none mean H=16 in [0.05, 0.10]",
            0.05 <= float(stats["none"][16]["mean"]) <= 0.10,
        ),
        (
            "euclidean mean H=16 in [0.05, 0.10]",
            0.05 <= float(stats["euclidean"][16]["mean"]) <= 0.10,
        ),
        (
            "spectral median H=16 in [0.07, 0.20]",
            0.07 <= float(stats["spectral"][16]["median"]) <= 0.20,
        ),
        (
            "spectral max H=16 > 5.0",
            float(stats["spectral"][16]["max"]) > 5.0,
        ),
    ]
    return [(label, "OK" if passed else "CHECK") for label, passed in checks]


def print_summary(stats: dict[str, dict[int, dict[str, float | int | list[float]]]]) -> None:
    spectral_h16 = stats["spectral"][16]
    print("Generated files:")
    print(f"- {TABLE_PATH.relative_to(ROOT)}")
    print(f"- {FIGURE_PDF.relative_to(ROOT)}")
    print(f"- {FIGURE_PNG.relative_to(ROOT)}")
    print()
    print(make_markdown_table(stats))
    print()
    print("Per-seed spectral H=16:")
    print(f"spectral H=16 per seed (sorted): {format_raw_values(spectral_h16['values'])}")
    print(
        f"- {int(spectral_h16['catastrophic'])} of 10 seeds had H=16 > "
        f"{CATASTROPHIC_THRESHOLD:.1f} (catastrophic threshold)"
    )
    print(f"- Median: {float(spectral_h16['median']):.3f}")
    print(
        f"- Mean: {float(spectral_h16['mean']):.3f} "
        f"(dominated by {int(spectral_h16['catastrophic'])} catastrophic seeds)"
    )
    print()
    print("Sanity check:")
    for label, status in sanity_status(stats):
        print(f"- {status}: {label}")


def main() -> None:
    args = parse_args()
    results = load_results(args.input)
    validate_results(results)
    stats = summarize(results)
    write_outputs(stats)
    print_summary(stats)


if __name__ == "__main__":
    main()
