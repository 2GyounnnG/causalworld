from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = ROOT / "validation_wolfram_clean_mechanism.json"
TABLE_PATH = ROOT / "paper" / "tables" / "table3_wolfram_mechanism.tex"
FIGURE_DIR = ROOT / "paper" / "figures"
FIGURE_PDF = FIGURE_DIR / "fig3_wolfram_mechanism.pdf"
FIGURE_PNG = FIGURE_DIR / "fig3_wolfram_mechanism.png"

MODES = ["per_step", "fixed_initial", "fixed_average"]
WEIGHTS = [0.001, 0.005, 0.01]
SEEDS = range(5)
HORIZON = 16

MODE_COLORS = {
    "per_step": "#2ca02c",
    "fixed_initial": "#d62728",
    "fixed_average": "#ff7f0e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Wolfram mechanism ablation Table 3 and Figure 3."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"Path to mechanism ablation JSON (default: {DEFAULT_RESULTS_PATH})",
    )
    return parser.parse_args()


def load_results(path: Path) -> dict:
    expanded_path = path.expanduser()
    if not expanded_path.exists():
        raise SystemExit(
            f"Missing Wolfram mechanism results file: {expanded_path}\n"
            "Pass a different path with --input /path/to/validation_wolfram_clean_mechanism.json."
        )
    with expanded_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload.get("results", payload)


def spectral_key(mode: str, weight: float, seed: int) -> str:
    return f"wolfram|flat|spectral|w={weight:g}|mode={mode}|seed={seed}"


def baseline_key(seed: int) -> str:
    return f"wolfram|flat|none|w=0.0|mode=per_step|seed={seed}"


def validate_results(results: dict) -> None:
    missing = []
    for seed in SEEDS:
        key = baseline_key(seed)
        if key not in results:
            missing.append(key)
    for mode in MODES:
        for weight in WEIGHTS:
            for seed in SEEDS:
                key = spectral_key(mode, weight, seed)
                if key not in results:
                    missing.append(key)
    if missing:
        raise KeyError(f"Missing expected records: {missing}")

    for key in [baseline_key(seed) for seed in SEEDS]:
        if str(HORIZON) not in results[key].get("rollout_errors", {}):
            raise KeyError(f"Missing rollout_errors[{HORIZON!r}] for {key}")
    for mode in MODES:
        for weight in WEIGHTS:
            for seed in SEEDS:
                key = spectral_key(mode, weight, seed)
                if str(HORIZON) not in results[key].get("rollout_errors", {}):
                    raise KeyError(f"Missing rollout_errors[{HORIZON!r}] for {key}")


def summarize_values(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "max": max(values),
    }


def collect_baseline(results: dict) -> dict[str, float | list[float]]:
    values = [
        float(results[baseline_key(seed)]["rollout_errors"][str(HORIZON)])
        for seed in SEEDS
    ]
    return summarize_values(values)


def collect_spectral(results: dict) -> dict[str, dict[float, dict[str, float | list[float]]]]:
    stats: dict[str, dict[float, dict[str, float | list[float]]]] = {}
    for mode in MODES:
        stats[mode] = {}
        for weight in WEIGHTS:
            values = [
                float(results[spectral_key(mode, weight, seed)]["rollout_errors"][str(HORIZON)])
                for seed in SEEDS
            ]
            stats[mode][weight] = summarize_values(values)
    return stats


def mean_std_text(cell: dict[str, float | list[float]]) -> str:
    return f"{cell['mean']:.3f} ± {cell['std']:.3f}"


def markdown_cell(cell: dict[str, float | list[float]]) -> str:
    return f"{mean_std_text(cell)} (max={cell['max']:.1f})"


def latex_cell(cell: dict[str, float | list[float]]) -> str:
    mean = float(cell["mean"])
    prefix = ""
    if mean > 1.0:
        prefix = "\\cellcolor{red!15}"
    elif mean < 0.1:
        prefix = "\\cellcolor{green!15}"
    return (
        f"{prefix}\\makecell{{{mean:.3f} $\\pm$ {float(cell['std']):.3f}\\\\"
        f"(max={float(cell['max']):.1f})}}"
    )


def make_latex_table(
    spectral_stats: dict[str, dict[float, dict[str, float | list[float]]]],
    baseline: dict[str, float | list[float]],
) -> str:
    baseline_text = f"{float(baseline['mean']):.3f} $\\pm$ {float(baseline['std']):.3f}"
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Mode & $\\lambda=0.001$ & $\\lambda=0.005$ & $\\lambda=0.01$ \\\\",
        "\\midrule",
        (
            "\\multirow{1}{*}{none ($\\lambda=0$)} & "
            f"\\multicolumn{{3}}{{c}}{{\\cellcolor{{green!15}}{baseline_text}}} \\\\"
        ),
        "\\midrule",
    ]
    for mode in MODES:
        row = [mode]
        row.extend(latex_cell(spectral_stats[mode][weight]) for weight in WEIGHTS)
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def make_markdown_table(
    spectral_stats: dict[str, dict[float, dict[str, float | list[float]]]],
    baseline: dict[str, float | list[float]],
) -> str:
    lines = [
        "| Mode | w=0.001 | w=0.005 | w=0.01 |",
        "| --- | --- | --- | --- |",
        (
            "| none (λ=0) | "
            f"{mean_std_text(baseline)} | {mean_std_text(baseline)} | "
            f"{mean_std_text(baseline)} |"
        ),
    ]
    for mode in MODES:
        row = [mode]
        row.extend(markdown_cell(spectral_stats[mode][weight]) for weight in WEIGHTS)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def make_figure(
    spectral_stats: dict[str, dict[float, dict[str, float | list[float]]]],
    baseline: dict[str, float | list[float]],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    x_positions = list(range(len(WEIGHTS)))
    width = 0.24
    offsets = {
        "per_step": -width,
        "fixed_initial": 0.0,
        "fixed_average": width,
    }
    min_positive = 1e-3

    for mode in MODES:
        means = [float(spectral_stats[mode][weight]["mean"]) for weight in WEIGHTS]
        stds = [float(spectral_stats[mode][weight]["std"]) for weight in WEIGHTS]
        lower = [min(std, max(mean - min_positive, min_positive)) for mean, std in zip(means, stds)]
        upper = stds
        ax.bar(
            [x + offsets[mode] for x in x_positions],
            means,
            width=width,
            yerr=[lower, upper],
            capsize=3,
            color=MODE_COLORS[mode],
            edgecolor="black",
            linewidth=0.7,
            label=mode,
        )

    ax.axhline(
        float(baseline["mean"]),
        ls="--",
        color="gray",
        linewidth=1.5,
        label="no prior (baseline)",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{weight:g}" for weight in WEIGHTS])
    ax.set_xlabel("Spectral prior weight")
    ax.set_ylabel("H=16 rollout error (mean ± std)")
    ax.set_yscale("log")
    ax.set_title(
        "Wolfram CA: spectral prior stability by Laplacian mode and weight "
        "(n=5 seeds, H=16)"
    )
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend(loc="upper left")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, bbox_inches="tight")
    plt.close(fig)


def write_outputs(
    spectral_stats: dict[str, dict[float, dict[str, float | list[float]]]],
    baseline: dict[str, float | list[float]],
) -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(make_latex_table(spectral_stats, baseline), encoding="utf-8")
    make_figure(spectral_stats, baseline)


def format_raw_values(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in sorted(values)) + "]"


def print_raw_values(
    spectral_stats: dict[str, dict[float, dict[str, float | list[float]]]],
    baseline: dict[str, float | list[float]],
) -> None:
    print("Per-seed raw H=16 values:")
    print(f"{'none':13s} w=0:     {format_raw_values(baseline['values'])}")
    for mode in MODES:
        for weight in WEIGHTS:
            print(
                f"{mode:13s} w={weight:g}: "
                f"{format_raw_values(spectral_stats[mode][weight]['values'])}"
            )


def sanity_status(
    spectral_stats: dict[str, dict[float, dict[str, float | list[float]]]],
    baseline: dict[str, float | list[float]],
) -> list[tuple[str, str]]:
    checks = [
        (
            "per_step × w=0.005 mean ≈ 0.061",
            abs(float(spectral_stats["per_step"][0.005]["mean"]) - 0.061) <= 0.005,
        ),
        (
            "per_step × w=0.01 mean ≈ 1.89",
            abs(float(spectral_stats["per_step"][0.01]["mean"]) - 1.89) <= 0.05,
        ),
        (
            "fixed_initial × w=0.01 mean ≈ 13046",
            abs(float(spectral_stats["fixed_initial"][0.01]["mean"]) - 13046) <= 5.0,
        ),
        (
            "fixed_initial × w=0.005 mean ≈ 119",
            abs(float(spectral_stats["fixed_initial"][0.005]["mean"]) - 119) <= 1.0,
        ),
        (
            "baseline (none) ≈ 0.066",
            abs(float(baseline["mean"]) - 0.066) <= 0.005,
        ),
    ]
    return [(label, "OK" if passed else "CHECK") for label, passed in checks]


def print_summary(
    spectral_stats: dict[str, dict[float, dict[str, float | list[float]]]],
    baseline: dict[str, float | list[float]],
) -> None:
    print("Generated files:")
    print(f"- {TABLE_PATH.relative_to(ROOT)}")
    print(f"- {FIGURE_PDF.relative_to(ROOT)}")
    print(f"- {FIGURE_PNG.relative_to(ROOT)}")
    print()
    print(make_markdown_table(spectral_stats, baseline))
    print()
    print_raw_values(spectral_stats, baseline)
    print()
    print("Sanity check:")
    for label, status in sanity_status(spectral_stats, baseline):
        print(f"- {status}: {label}")


def main() -> None:
    args = parse_args()
    results = load_results(args.input)
    validate_results(results)
    baseline = collect_baseline(results)
    spectral_stats = collect_spectral(results)
    write_outputs(spectral_stats, baseline)
    print_summary(spectral_stats, baseline)


if __name__ == "__main__":
    main()
