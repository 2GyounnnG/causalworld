from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = ROOT / "iso17_split_eval_ablation.json"
TABLE_PATH = ROOT / "paper" / "tables" / "table2_iso17.tex"
FIGURE_DIR = ROOT / "paper" / "figures"
FIGURE_PDF = FIGURE_DIR / "fig2_iso17.pdf"
FIGURE_PNG = FIGURE_DIR / "fig2_iso17.png"

PRIORS = ["none", "euclidean", "spectral"]
SPLITS = ["test_within", "test_other"]
SPLIT_LABELS = {
    "test_within": "test_within",
    "test_other": "test_other",
}
HORIZONS = [1, 2, 4, 8, 16]
EXPECTED_SEEDS = range(5)
OVERLAP_DENOMINATOR = 200

COLORS = {
    "none": "#888888",
    "euclidean": "#1f77b4",
    "spectral": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ISO17 split-eval ablation Table 2 and Figure 2."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"Path to ISO17 split-eval JSON (default: {DEFAULT_RESULTS_PATH})",
    )
    return parser.parse_args()


def load_results(path: Path) -> dict:
    expanded_path = path.expanduser()
    if not expanded_path.exists():
        raise SystemExit(
            f"Missing ISO17 results file: {expanded_path}\n"
            "Pass a different path with --input /path/to/iso17_split_eval_ablation.json."
        )
    with expanded_path.open("r", encoding="utf-8") as file:
        data = json.load(file); return data.get("results", data)


def result_key(prior: str, seed: int) -> str:
    return f"iso17|reference|{prior}|seed={seed}"


def validate_results(results: dict) -> None:
    expected_keys = {result_key(prior, seed) for prior in PRIORS for seed in EXPECTED_SEEDS}
    actual_keys = set(results)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        raise KeyError(f"Missing expected records: {missing}")
    if extra:
        raise KeyError(f"Unexpected records: {extra}")

    for key in sorted(expected_keys):
        record = results[key]
        if record.get("status") != "ok":
            raise ValueError(f"{key} has status={record.get('status')!r}, expected 'ok'")
        for split in SPLITS:
            rollout_field = f"rollout_errors_{split}"
            metadata_field = f"metadata_{split}"
            if rollout_field not in record:
                raise KeyError(f"Missing {rollout_field} for {key}")
            if metadata_field not in record:
                raise KeyError(f"Missing {metadata_field} for {key}")
            for horizon in HORIZONS:
                h_key = str(horizon)
                if h_key not in record[rollout_field]:
                    raise KeyError(f"Missing {rollout_field}[{h_key!r}] for {key}")
            if "train_eval_overlap_count" not in record[metadata_field]:
                raise KeyError(f"Missing train_eval_overlap_count in {metadata_field} for {key}")


def collect_rollout_values(results: dict, prior: str, split: str, horizon: int) -> list[float]:
    values = []
    for seed in EXPECTED_SEEDS:
        record = results[result_key(prior, seed)]
        values.append(float(record[f"rollout_errors_{split}"][str(horizon)]))
    return values


def summarize_rollouts(results: dict) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    stats: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for prior in PRIORS:
        stats[prior] = {}
        for split in SPLITS:
            stats[prior][split] = {}
            for horizon in HORIZONS:
                values = collect_rollout_values(results, prior, split, horizon)
                n = len(values)
                mean = statistics.mean(values)
                std = statistics.stdev(values) if n > 1 else 0.0
                ci95 = 1.96 * std / math.sqrt(n)
                stats[prior][split][horizon] = {
                    "n": n,
                    "mean": mean,
                    "std": std,
                    "ci95": ci95,
                }
    return stats


def summarize_generalization_gap(
    stats: dict[str, dict[str, dict[int, dict[str, float]]]]
) -> dict[str, float]:
    gaps = {}
    for prior in PRIORS:
        within = stats[prior]["test_within"][16]["mean"]
        other = stats[prior]["test_other"][16]["mean"]
        gaps[prior] = other / within
    return gaps


def summarize_overlap(results: dict) -> dict[str, dict]:
    by_prior_split: dict[str, dict[str, dict[str, float]]] = {}
    pooled_values = {split: [] for split in SPLITS}
    for prior in PRIORS:
        by_prior_split[prior] = {}
        for split in SPLITS:
            values = []
            for seed in EXPECTED_SEEDS:
                record = results[result_key(prior, seed)]
                count = float(record[f"metadata_{split}"]["train_eval_overlap_count"])
                values.append(count)
                pooled_values[split].append(count)
            by_prior_split[prior][split] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }

    pooled = {}
    for split, values in pooled_values.items():
        pooled[split] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return {"by_prior_split": by_prior_split, "pooled": pooled}


def mean_std_cell(cell: dict[str, float]) -> str:
    return f"{cell['mean']:.3f} \u00b1 {cell['std']:.3f}"


def latex_mean_std_cell(cell: dict[str, float]) -> str:
    return f"{cell['mean']:.3f} $\\pm$ {cell['std']:.3f}"


def maybe_bold(prior: str, text: str) -> str:
    return f"\\textbf{{{text}}}" if prior == "spectral" else text


def make_latex_table(stats: dict[str, dict[str, dict[int, dict[str, float]]]]) -> str:
    lines = [
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Prior & Split & H=1 & H=2 & H=4 & H=8 & H=16 \\\\",
        "\\midrule",
    ]
    for prior_index, prior in enumerate(PRIORS):
        for split_index, split in enumerate(SPLITS):
            prior_cell = (
                f"\\multirow{{2}}{{*}}{{{maybe_bold(prior, prior)}}}"
                if split_index == 0
                else ""
            )
            row = [prior_cell, maybe_bold(prior, SPLIT_LABELS[split])]
            row.extend(
                maybe_bold(prior, latex_mean_std_cell(stats[prior][split][horizon]))
                for horizon in HORIZONS
            )
            lines.append(" & ".join(row) + " \\\\")
        if prior_index < len(PRIORS) - 1:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def make_markdown_table(stats: dict[str, dict[str, dict[int, dict[str, float]]]]) -> str:
    lines = [
        "| Prior | Split | H=1 | H=2 | H=4 | H=8 | H=16 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for prior in PRIORS:
        for split in SPLITS:
            row = [prior, SPLIT_LABELS[split]]
            row.extend(mean_std_cell(stats[prior][split][horizon]) for horizon in HORIZONS)
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def make_figure(stats: dict[str, dict[str, dict[int, dict[str, float]]]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    x_positions = list(range(len(PRIORS)))
    width = 0.34
    offsets = {"test_within": -width / 2, "test_other": width / 2}

    for split in SPLITS:
        for idx, prior in enumerate(PRIORS):
            cell = stats[prior][split][16]
            ax.bar(
                x_positions[idx] + offsets[split],
                cell["mean"],
                width=width,
                yerr=cell["ci95"],
                color=COLORS[prior],
                edgecolor="black",
                linewidth=0.7,
                capsize=3,
                hatch="///" if split == "test_other" else None,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(PRIORS)
    ax.set_ylabel("H=16 mean rollout error")
    ax.set_title("ISO17 cross-isomer evaluation (n=5 seeds per prior, H=16)")
    ax.grid(axis="y", alpha=0.3)

    legend_handles = [
        mpatches.Patch(facecolor="#bbbbbb", edgecolor="black", label="test_within"),
        mpatches.Patch(
            facecolor="#bbbbbb", edgecolor="black", hatch="///", label="test_other"
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    fig.savefig(FIGURE_PNG, bbox_inches="tight")
    plt.close(fig)


def write_outputs(stats: dict[str, dict[str, dict[int, dict[str, float]]]]) -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(make_latex_table(stats), encoding="utf-8")
    make_figure(stats)


def sanity_status(
    stats: dict[str, dict[str, dict[int, dict[str, float]]]],
    gaps: dict[str, float],
    overlap: dict[str, dict],
) -> list[tuple[str, str]]:
    checks = [
        (
            "spectral test_within H=16 mean",
            abs(stats["spectral"]["test_within"][16]["mean"] - 0.148) <= 0.005,
        ),
        (
            "spectral test_other H=16 mean",
            abs(stats["spectral"]["test_other"][16]["mean"] - 0.150) <= 0.005,
        ),
        ("spectral gen gap", abs(gaps["spectral"] - 1.02) <= 0.05),
    ]
    for split in SPLITS:
        mean_overlap = overlap["pooled"][split]["mean"]
        checks.append(
            (
                f"{split} mean overlap_count per record",
                3.0 <= mean_overlap <= 6.0,
            )
        )
    return [(label, "OK" if passed else "CHECK") for label, passed in checks]


def print_summary(
    stats: dict[str, dict[str, dict[int, dict[str, float]]]],
    gaps: dict[str, float],
    overlap: dict[str, dict],
) -> None:
    print("Generated files:")
    print(f"- {TABLE_PATH.relative_to(ROOT)}")
    print(f"- {FIGURE_PDF.relative_to(ROOT)}")
    print(f"- {FIGURE_PNG.relative_to(ROOT)}")
    print()
    print(make_markdown_table(stats))
    print()

    print("Generalization gap (mean test_other / mean test_within at H=16):")
    for prior in PRIORS:
        print(f"- {prior}: {gaps[prior]:.2f}x")
    print()

    print("Train/eval overlap diagnostics (mean across all priors and seeds):")
    for split in SPLITS:
        mean_overlap = overlap["pooled"][split]["mean"]
        pct = 100.0 * mean_overlap / OVERLAP_DENOMINATOR
        print(
            f"- {split}: mean={mean_overlap:.1f} / {OVERLAP_DENOMINATOR} "
            f"transitions ({pct:.1f}%)"
        )
    print()

    print("Sanity check:")
    for label, status in sanity_status(stats, gaps, overlap):
        print(f"- {status}: {label}")


def main() -> None:
    args = parse_args()
    results = load_results(args.input)
    validate_results(results)
    stats = summarize_rollouts(results)
    gaps = summarize_generalization_gap(stats)
    overlap = summarize_overlap(results)
    write_outputs(stats)
    print_summary(stats, gaps, overlap)


if __name__ == "__main__":
    main()
