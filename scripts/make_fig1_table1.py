from __future__ import annotations

import json
import math
import os
import re
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "rmd17_aspirin_10seed_results.json"
TABLE_PATH = ROOT / "paper" / "tables" / "table1_rmd17.tex"
FIGURE_DIR = ROOT / "paper" / "figures"
FIGURE_PDF = FIGURE_DIR / "fig1_rmd17_aspirin.pdf"
FIGURE_PNG = FIGURE_DIR / "fig1_rmd17_aspirin.png"

PRIORS = ["none", "euclidean", "spectral"]
HORIZONS = [1, 2, 4, 8, 16]
COLORS = {
    "none": "#888888",
    "euclidean": "#1f77b4",
    "spectral": "#d62728",
}
EXPECTED_H16 = {
    "none": 0.290,
    "euclidean": 0.323,
    "spectral": 0.140,
}


def load_results(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_values(results: dict, prior: str, horizon: int) -> list[float]:
    values = []
    for seed in range(10):
        key = f"aspirin|flat|{prior}|seed={seed}"
        if key not in results:
            raise KeyError(f"Missing result key: {key}")
        rollout_errors = results[key].get("rollout_errors", {})
        h_key = str(horizon)
        if h_key not in rollout_errors:
            raise KeyError(f"Missing rollout_errors[{h_key!r}] for {key}")
        values.append(float(rollout_errors[h_key]))
    return values


def summarize(results: dict) -> dict[str, dict[int, dict[str, float]]]:
    stats = {}
    for prior in PRIORS:
        stats[prior] = {}
        for horizon in HORIZONS:
            values = collect_values(results, prior, horizon)
            n = len(values)
            mean = statistics.mean(values)
            std = statistics.stdev(values) if n > 1 else 0.0
            half_width = 1.96 * std / math.sqrt(n)
            stats[prior][horizon] = {
                "n": n,
                "mean": mean,
                "std": std,
                "ci_low": mean - half_width,
                "ci_high": mean + half_width,
            }
    return stats


def pct_change_vs_none(stats: dict[str, dict[int, dict[str, float]]], prior: str) -> float:
    baseline = stats["none"][16]["mean"]
    mean = stats[prior][16]["mean"]
    return 100.0 * (mean - baseline) / baseline


def mean_std_cell(cell: dict[str, float]) -> str:
    return f"{cell['mean']:.3f} ± {cell['std']:.3f}"


def latex_mean_std_cell(cell: dict[str, float]) -> str:
    return f"{cell['mean']:.3f} $\\pm$ {cell['std']:.3f}"


def maybe_bold(prior: str, text: str) -> str:
    if prior == "spectral":
        return f"\\textbf{{{text}}}"
    return text


def make_latex_table(stats: dict[str, dict[int, dict[str, float]]]) -> str:
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Prior & H=1 & H=2 & H=4 & H=8 & H=16 & \\% $\\Delta$ vs none \\\\",
        "\\midrule",
    ]
    for prior in PRIORS:
        cells = [maybe_bold(prior, prior)]
        cells.extend(maybe_bold(prior, latex_mean_std_cell(stats[prior][h])) for h in HORIZONS)
        cells.append(maybe_bold(prior, f"{pct_change_vs_none(stats, prior):+.1f}\\%"))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def make_markdown_table(stats: dict[str, dict[int, dict[str, float]]]) -> str:
    lines = [
        "| Prior | H=1 | H=2 | H=4 | H=8 | H=16 | % Δ vs none |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for prior in PRIORS:
        row = [prior]
        row.extend(mean_std_cell(stats[prior][h]) for h in HORIZONS)
        row.append(f"{pct_change_vs_none(stats, prior):+.1f}%")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def make_figure(stats: dict[str, dict[int, dict[str, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)
    for prior in PRIORS:
        means = [stats[prior][h]["mean"] for h in HORIZONS]
        lows = [stats[prior][h]["ci_low"] for h in HORIZONS]
        highs = [stats[prior][h]["ci_high"] for h in HORIZONS]
        color = COLORS[prior]
        ax.fill_between(HORIZONS, lows, highs, color=color, alpha=0.2, linewidth=0, zorder=1)
        ax.plot(
            HORIZONS,
            means,
            marker="o",
            linewidth=2,
            color=color,
            label=prior,
            zorder=2,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel("Rollout error")
    ax.set_title("rMD17 aspirin (n=10 seeds, shaded = 95% CI)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, bbox_inches="tight")
    plt.close(fig)


def write_outputs(stats: dict[str, dict[int, dict[str, float]]]) -> str:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    TABLE_PATH.write_text(make_latex_table(stats), encoding="utf-8")
    make_figure(stats)
    return make_markdown_table(stats)


def find_summary_path() -> Path | None:
    candidates = [
        ROOT / "RESULTS_SUMMARY_FOR_PAPER.md",
        ROOT / "analysis_out" / "RESULTS_SUMMARY_FOR_PAPER.md",
        ROOT / "final_results_snapshot" / "RESULTS_SUMMARY_FOR_PAPER.md",
    ]
    return next((path for path in candidates if path.exists()), None)


def parse_summary_h16_means(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"## rMD17 Aspirin 10-Seed H=16(?P<section>.*?)(?:\n## |\Z)",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return {}

    means = {}
    for line in match.group("section").splitlines():
        if not line.startswith("|") or line.startswith("| ---"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) >= 3 and parts[0] in PRIORS:
            try:
                means[parts[0]] = float(parts[2])
            except ValueError:
                pass
    return means


def print_summary(stats: dict[str, dict[int, dict[str, float]]], markdown_table: str) -> None:
    print("Generated files:")
    print(f"- {TABLE_PATH.relative_to(ROOT)}")
    print(f"- {FIGURE_PDF.relative_to(ROOT)}")
    print(f"- {FIGURE_PNG.relative_to(ROOT)}")
    print()
    print(markdown_table)
    print()

    summary_path = find_summary_path()
    summary_means = parse_summary_h16_means(summary_path) if summary_path else {}
    source = summary_path.relative_to(ROOT) if summary_path else "fallback expected values"
    print(f"Sanity check vs {source}:")
    for prior in PRIORS:
        computed = stats[prior][16]["mean"]
        expected = summary_means.get(prior, EXPECTED_H16[prior])
        delta = computed - expected
        status = "OK" if abs(delta) < 0.0015 else "CHECK"
        print(
            f"- {prior}: computed H=16 mean={computed:.3f}; "
            f"expected~{expected:.3f}; delta={delta:+.3f} [{status}]"
        )


def main() -> None:
    results = load_results(RESULTS_PATH)
    stats = summarize(results)
    markdown_table = write_outputs(stats)
    print_summary(stats, markdown_table)


if __name__ == "__main__":
    main()
