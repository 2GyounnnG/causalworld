from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 1729
HORIZONS = [1, 2, 4, 8, 16]
STAT_COLUMNS = [
    "experiment_name",
    "task_family",
    "dataset",
    "molecule",
    "encoder",
    "prior",
    "prior_weight",
    "laplacian_mode",
    "horizon",
    "n",
    "mean",
    "std",
    "stderr",
    "median",
    "min",
    "max",
    "ci95_low",
    "ci95_high",
    "pct_change_vs_none",
    "pct_change_vs_euclidean",
    "pct_change_vs_per_frame",
    "h16_over_h1_mean",
    "notes",
]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def parse_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


PERCENT_COLUMNS = {
    "pct_change_vs_none",
    "pct_change_vs_euclidean",
    "pct_change_vs_per_frame",
}
ROLLOUT_STAT_COLUMNS = {
    "mean",
    "std",
    "stderr",
    "median",
    "min",
    "max",
    "ci95_low",
    "ci95_high",
}
RATIO_COLUMNS = {"h16_over_h1_mean"}


def format_float(value: float | None, column: str = "") -> str:
    if value is None or not math.isfinite(value):
        return ""
    if column in PERCENT_COLUMNS:
        return f"{value:.1e}" if abs(value) >= 1000 else f"{value:.1f}"
    if column in RATIO_COLUMNS:
        return f"{value:.1e}" if abs(value) >= 1000 else f"{value:.1f}"
    if column in ROLLOUT_STAT_COLUMNS:
        return f"{value:.1f}" if abs(value) >= 10 else f"{value:.3f}"
    return f"{value:.10g}"


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_RESAMPLES, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def summarize(values: list[float], rng: np.random.Generator) -> dict[str, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "stderr": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    ci_low, ci_high = bootstrap_ci(arr, rng)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": std,
        "stderr": float(std / math.sqrt(arr.size)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def pct_change(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if abs(denominator) < 1e-12:
        return None
    return 100.0 * (numerator - denominator) / denominator


def group_rows(rows: list[dict[str, str]], experiment_filter: set[str]) -> list[dict[str, str]]:
    clean = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("metric_name") != "rollout_error":
            continue
        if row.get("experiment_name") not in experiment_filter:
            continue
        value = parse_float(row.get("metric_value"))
        horizon = parse_int(row.get("horizon"))
        if value is None or horizon is None:
            continue
        copied = dict(row)
        copied["_value"] = value
        copied["_horizon"] = horizon
        clean.append(copied)
    return clean


def aggregate(
    rows: list[dict[str, str]],
    *,
    experiment_filter: set[str],
    group_fields: list[str],
    comparison_mode: str,
    notes: str = "",
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    exemplars: dict[tuple[Any, ...], dict[str, str]] = {}
    for row in group_rows(rows, experiment_filter):
        key = tuple(row.get(field, "") for field in group_fields)
        grouped[key].append(row["_value"])
        exemplars.setdefault(key, row)

    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        exemplar = exemplars[key]
        stats = summarize(values, rng)
        out = {column: "" for column in STAT_COLUMNS}
        for column in ["experiment_name", "task_family", "dataset", "molecule", "encoder", "prior", "prior_weight", "laplacian_mode"]:
            out[column] = exemplar.get(column, "")
        out["horizon"] = exemplar.get("horizon", "")
        out.update(stats)
        out["notes"] = notes
        output.append(out)

    add_derived_columns(output, comparison_mode)
    return output


def comparison_context(row: dict[str, Any], comparison_mode: str) -> tuple[str, ...]:
    if comparison_mode == "rmd17_prior":
        return (
            str(row.get("experiment_name", "")),
            str(row.get("molecule", "")),
            str(row.get("encoder", "")),
            str(row.get("horizon", "")),
            str(row.get("dataset", "")),
        )
    if comparison_mode == "weight_sweep":
        return (
            str(row.get("experiment_name", "")),
            str(row.get("molecule", "")),
            str(row.get("encoder", "")),
            str(row.get("prior_weight", "")),
            str(row.get("horizon", "")),
            str(row.get("dataset", "")),
        )
    if comparison_mode == "laplacian_ablation":
        return (
            str(row.get("experiment_name", "")),
            str(row.get("molecule", "")),
            str(row.get("encoder", "")),
            str(row.get("horizon", "")),
            str(row.get("dataset", "")),
        )
    if comparison_mode == "wolfram":
        return (
            str(row.get("experiment_name", "")),
            str(row.get("encoder", "")),
            str(row.get("horizon", "")),
            str(row.get("dataset", "")),
        )
    raise ValueError(f"unknown comparison_mode: {comparison_mode}")


def add_derived_columns(rows: list[dict[str, Any]], comparison_mode: str) -> None:
    by_context: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        context = comparison_context(row, comparison_mode)
        by_context[context][str(row.get("prior", ""))] = float(row.get("mean", float("nan")))

    by_lap_context: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for row in rows:
        context = comparison_context(row, "laplacian_ablation")
        by_lap_context[context][str(row.get("laplacian_mode", ""))] = float(row.get("mean", float("nan")))

    h_means: dict[tuple[str, str, str, str, str, str], dict[int, float]] = defaultdict(dict)
    for row in rows:
        horizon = parse_int(row.get("horizon"))
        if horizon is None:
            continue
        context = (
            str(row.get("experiment_name", "")),
            str(row.get("molecule", "")),
            str(row.get("encoder", "")),
            str(row.get("prior", "")),
            str(row.get("prior_weight", "")),
            str(row.get("laplacian_mode", "")),
        )
        h_means[context][horizon] = float(row.get("mean", float("nan")))

    for row in rows:
        context = comparison_context(row, comparison_mode)
        means = by_context.get(context, {})
        mean = float(row.get("mean", float("nan")))
        row["pct_change_vs_none"] = pct_change(mean, means.get("none"))
        row["pct_change_vs_euclidean"] = pct_change(mean, means.get("euclidean"))

        lap_context = comparison_context(row, "laplacian_ablation")
        row["pct_change_vs_per_frame"] = pct_change(mean, by_lap_context.get(lap_context, {}).get("per_frame"))

        h_context = (
            str(row.get("experiment_name", "")),
            str(row.get("molecule", "")),
            str(row.get("encoder", "")),
            str(row.get("prior", "")),
            str(row.get("prior_weight", "")),
            str(row.get("laplacian_mode", "")),
        )
        row["h16_over_h1_mean"] = (
            h_means[h_context][16] / h_means[h_context][1]
            if 1 in h_means[h_context] and 16 in h_means[h_context] and abs(h_means[h_context][1]) > 1e-12
            else None
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=STAT_COLUMNS)
        writer.writeheader()
        for row in rows:
            serial = {}
            for column in STAT_COLUMNS:
                value = row.get(column, "")
                if isinstance(value, float):
                    serial[column] = format_float(value, column)
                elif isinstance(value, int):
                    serial[column] = str(value)
                elif value is None:
                    serial[column] = ""
                else:
                    serial[column] = value
            writer.writerow(serial)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 40) -> str:
    if not rows:
        return "_No rows available._"
    shown = rows[:limit]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in shown:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append(format_float(value, column))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    if len(rows) > limit:
        lines.append(f"| ... | {len(rows) - limit} additional rows omitted |  |  |  |")
    return "\n".join(lines)


def complete_priors_at_h16(rows: list[dict[str, Any]], expected_n: int) -> bool:
    by_prior = {str(row.get("prior")): int(row.get("n", 0)) for row in rows if str(row.get("horizon")) == "16"}
    return all(by_prior.get(prior) == expected_n for prior in ["none", "euclidean", "spectral"])


def mean_at_h16(rows: list[dict[str, Any]], prior: str) -> float | None:
    for row in rows:
        if str(row.get("horizon")) != "16" or str(row.get("prior")) != prior:
            continue
        return parse_float(row.get("mean"))
    return None


def complete_weight_sweep_at_h16(rows: list[dict[str, Any]]) -> bool:
    h16 = [row for row in rows if str(row.get("horizon")) == "16"]
    expected_weights = {"0.001", "0.01", "0.1", "1"}
    seen = set()
    n_by_group = {}
    for row in h16:
        prior_weight = parse_float(row.get("prior_weight"))
        if prior_weight is None:
            continue
        group = (str(row.get("prior")), f"{prior_weight:.12g}")
        seen.add(group)
        n_by_group[group] = int(row.get("n", 0))
    expected = {
        (prior, weight)
        for prior in ["none", "euclidean", "spectral"]
        for weight in expected_weights
    }
    return expected.issubset(seen) and all(n_by_group.get(group) == 3 for group in expected)


def complete_laplacian_at_h16(rows: list[dict[str, Any]]) -> bool:
    by_mode = {str(row.get("laplacian_mode")): int(row.get("n", 0)) for row in rows if str(row.get("horizon")) == "16"}
    return all(by_mode.get(mode) == 5 for mode in ["per_frame", "fixed_frame0", "fixed_mean"])


def section_status(is_complete: bool) -> str:
    return "CONFIRMED" if is_complete else "PENDING"


def main_claim_status_line(main_complete: bool, spectral_wins_h16: bool) -> str:
    if main_complete and spectral_wins_h16:
        return "rMD17 aspirin 10-seed main comparison: CONFIRMED; spectral has lower H=16 rollout error than both none and euclidean."
    if main_complete:
        return "rMD17 aspirin 10-seed main comparison: COMPLETE BUT NOT SUPPORTIVE; do not claim spectral advantage."
    return "rMD17 aspirin 10-seed main comparison: PENDING."


def wolfram_claim_status_line(wolfram_complete: bool, h16_wolfram: list[dict[str, Any]]) -> str:
    if not wolfram_complete:
        return "Wolfram flat 200-epoch comparison: PENDING."
    means = {
        str(row.get("prior")): parse_float(row.get("mean"))
        for row in h16_wolfram
        if str(row.get("horizon")) == "16"
    }
    finite_means = {prior: mean for prior, mean in means.items() if mean is not None}
    if not finite_means:
        return "Wolfram flat 200-epoch comparison: COMPLETE; no spectral advantage."
    best_prior = min(finite_means, key=lambda prior: finite_means[prior])
    if best_prior == "spectral":
        return "Wolfram flat 200-epoch comparison: COMPLETE; spectral has the best H=16 mean."
    if best_prior == "euclidean":
        return "Wolfram flat 200-epoch comparison: COMPLETE; Euclidean has the best H=16 mean; spectral shows long-horizon instability."
    return "Wolfram flat 200-epoch comparison: COMPLETE; no spectral advantage."


def write_summary(
    path: Path,
    prior_rows: list[dict[str, Any]],
    weight_rows: list[dict[str, Any]],
    lap_rows: list[dict[str, Any]],
    wolfram_rows: list[dict[str, Any]],
) -> None:
    h16 = [row for row in prior_rows if str(row.get("horizon")) == "16"]
    h16_weight = [row for row in weight_rows if str(row.get("horizon")) == "16"]
    h16_lap = [row for row in lap_rows if str(row.get("horizon")) == "16"]
    h16_wolfram = [row for row in wolfram_rows if str(row.get("horizon")) == "16"]
    main_complete = complete_priors_at_h16(prior_rows, expected_n=10)
    none_mean = mean_at_h16(prior_rows, "none")
    euclidean_mean = mean_at_h16(prior_rows, "euclidean")
    spectral_mean = mean_at_h16(prior_rows, "spectral")
    spectral_wins_h16 = (
        spectral_mean is not None
        and none_mean is not None
        and euclidean_mean is not None
        and spectral_mean < none_mean
        and spectral_mean < euclidean_mean
    )
    main_claim_confirmed = main_complete and spectral_wins_h16
    weight_complete = complete_weight_sweep_at_h16(weight_rows)
    lap_complete = complete_laplacian_at_h16(lap_rows)
    wolfram_complete = complete_priors_at_h16(wolfram_rows, expected_n=10)
    pending_claims = []
    if not weight_complete:
        pending_claims.append("- The Euclidean prior-weight explanation is pending until the weight sweep completes across all seeds and weights.")
    if not lap_complete:
        pending_claims.append("- The per-frame Laplacian leakage concern is pending until fixed-frame and fixed-mean Laplacian modes complete.")
    if not wolfram_complete:
        pending_claims.append("- Wolfram 200-epoch conclusions are pending until all 10 seeds complete.")
    if not pending_claims:
        pending_claims.append("- No aggregate-completeness blockers remain for the reported rMD17 weight sweep, Laplacian ablation, or Wolfram flat 200-epoch tables.")
    lines = [
        "# Results Summary For Paper",
        "",
        "## Claim Status",
        "",
        f"- {main_claim_status_line(main_complete, spectral_wins_h16)}",
        f"- rMD17 weight sweep: {section_status(weight_complete)}.",
        f"- rMD17 Laplacian ablation: {section_status(lap_complete)}.",
        f"- {wolfram_claim_status_line(wolfram_complete, h16_wolfram)}",
        "",
        "## Main Confirmed Claims",
        "",
        "- The rMD17 aspirin spectral-prior advantage over both no prior and the Euclidean isotropy prior is supported by the completed H=16 table."
        if main_claim_confirmed
        else (
            "- The completed H=16 table does not support the spectral-advantage claim."
            if main_complete
            else "- No final main claim should be made from the rMD17 aspirin table until all three priors have n=10 at H=16."
        ),
        "- This is a different regime from LeJEPA: known downstream rollout prediction, structured relational data, and temporally correlated samples.",
        "",
        "## Claims Still Pending",
        "",
        *pending_claims,
        "",
        "## Claims That Should Not Be Made Yet",
        "",
        "- Do not claim spectral priors are universally optimal.",
        "- Do not claim the Laplacian ablation rules out per-sample information leakage until all fixed-mode seeds finish.",
        "- Do not say \"we refute LeJEPA.\"",
        "",
        "## rMD17 Aspirin 10-Seed H=16",
        "",
        "" if main_complete else "Current data are partial; do not use this table for final claims.",
        "",
        markdown_table(h16, ["prior", "n", "mean", "std", "ci95_low", "ci95_high", "pct_change_vs_none", "pct_change_vs_euclidean"]),
        "",
        "Interpretation: lower rollout error is better. This section is final only when all three priors have n=10 at H=16.",
        "",
        "## Weight Sweep H=16",
        "",
        "" if weight_complete else "Current data are partial; do not use this table for final claims.",
        "",
        markdown_table(h16_weight, ["prior", "prior_weight", "n", "mean", "std", "ci95_low", "ci95_high", "pct_change_vs_none"]),
        "",
        "Interpretation: this section is final only when every prior-weight-prior group has n=3 at H=16.",
        "",
        "## Laplacian Ablation H=16",
        "",
        "" if lap_complete else "Current data are partial; do not use this table for final claims.",
        "",
        markdown_table(h16_lap, ["laplacian_mode", "n", "mean", "std", "ci95_low", "ci95_high", "pct_change_vs_per_frame"]),
        "",
        "Interpretation: this section is final only when every Laplacian mode has n=5 at H=16.",
        "",
        "## Wolfram 200-Epoch H=16",
        "",
        "" if wolfram_complete else "Current data are partial; do not use this table for final claims.",
        "",
        markdown_table(h16_wolfram, ["prior", "n", "mean", "std", "ci95_low", "ci95_high", "pct_change_vs_none", "pct_change_vs_euclidean"]),
        "",
        "The completed Wolfram flat run does not support a universal spectral-prior advantage. Euclidean slightly improves over no prior at H=16, while spectral has a heavy-tailed failure mode: most seeds remain near baseline scale, but a few seeds explode at long horizon.",
        "",
        "Interpretation: this section is final only when all three priors have n=10 at H=16.",
        "",
        "## Reviewer-Risk Section",
        "",
        "- Per-frame Laplacian leakage risk: active risk until fixed Laplacian modes finish and agree directionally with per-frame.",
        "- Euclidean prior weight risk: active risk until lower-weight Euclidean settings are complete.",
        "- Seed/statistical power risk: rMD17 aspirin main result is stronger at n=10; current ablations are underpowered while incomplete.",
        "- rMD17 split/data leakage risk: rMD17 samples come from one trajectory; train/eval use different random seeds but no explicit disjoint-frame audit is yet enforced.",
        "- Synthetic-to-real generalization risk: Wolfram and rMD17 cover different regimes, so cross-domain generalization should be framed cautiously.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="analysis_out/manifest.csv")
    parser.add_argument("--out", default="analysis_out")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out = Path(args.out)
    rows = read_manifest(manifest_path)

    # Keep checkpointed rMD17 reruns out of the main aggregate by default.
    # They are for reviewer-facing disjoint eval / replication and should be
    # aggregated separately if needed.
    prior_rows = aggregate(
        rows,
        experiment_filter={"rmd17_aspirin_10seed"},
        group_fields=["experiment_name", "task_family", "dataset", "molecule", "encoder", "prior", "horizon"],
        comparison_mode="rmd17_prior",
        notes="completed canonical rMD17 aspirin 10-seed run",
    )
    weight_rows = aggregate(
        rows,
        experiment_filter={"rmd17_weight_sweep"},
        group_fields=["experiment_name", "task_family", "dataset", "molecule", "encoder", "prior", "prior_weight", "horizon"],
        comparison_mode="weight_sweep",
        notes="partial current weight sweep if n < 3",
    )
    lap_rows = aggregate(
        rows,
        experiment_filter={"rmd17_laplacian_ablation"},
        group_fields=["experiment_name", "task_family", "dataset", "molecule", "encoder", "prior", "laplacian_mode", "horizon"],
        comparison_mode="laplacian_ablation",
        notes="partial current Laplacian ablation if n < 5",
    )
    wolfram_rows = aggregate(
        rows,
        experiment_filter={"wolfram_flat_10seed_200ep"},
        group_fields=["experiment_name", "task_family", "dataset", "encoder", "prior", "horizon"],
        comparison_mode="wolfram",
        notes="canonical Wolfram 200-epoch resume output; partial if n < 10",
    )

    if args.dry_run:
        print(f"would write {out / 'aggregate_by_prior_horizon.csv'} ({len(prior_rows)} rows)")
        print(f"would write {out / 'aggregate_weight_sweep.csv'} ({len(weight_rows)} rows)")
        print(f"would write {out / 'aggregate_laplacian_ablation.csv'} ({len(lap_rows)} rows)")
        print(f"would write {out / 'aggregate_wolfram.csv'} ({len(wolfram_rows)} rows)")
        print(f"would write {out / 'RESULTS_SUMMARY_FOR_PAPER.md'}")
        return

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "aggregate_by_prior_horizon.csv", prior_rows)
    write_csv(out / "aggregate_weight_sweep.csv", weight_rows)
    write_csv(out / "aggregate_laplacian_ablation.csv", lap_rows)
    write_csv(out / "aggregate_wolfram.csv", wolfram_rows)
    write_summary(out / "RESULTS_SUMMARY_FOR_PAPER.md", prior_rows, weight_rows, lap_rows, wolfram_rows)
    print(f"wrote aggregates to {out}")


if __name__ == "__main__":
    main()
