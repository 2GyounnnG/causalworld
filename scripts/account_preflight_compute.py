from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = ROOT / "analysis_out" / "preflight_runs"
DEFAULT_OUTPUT = ROOT / "paper" / "tables" / "preflight_compute_cost.md"


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def parse_seeds(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if value is None:
        return [0]
    return [int(value)]


def inferred_training_count(config_args: dict[str, Any], summary_rows: list[dict[str, str]]) -> int:
    observed = [
        row for row in summary_rows
        if row.get("stage") in {"stage1_mini_train", "node_order_sanity"}
    ]
    if observed:
        return len(observed)
    seeds = parse_seeds(config_args.get("seeds", [0]))
    if "node_order_sanity" in str(config_args.get("out_dir", "")):
        return len(seeds) * (3 if config_args.get("include_mismatched", True) else 2)
    priors = 3 + (1 if config_args.get("include_temporal_prior") else 0)
    return len(seeds) * priors


def classify_budget(path: Path, epochs: int) -> str:
    text = str(path)
    if "standard" in text or epochs >= 20:
        return "standard check"
    if "node_order_sanity" in text:
        return "audit mode"
    if epochs <= 5:
        return "quick preflight"
    return "intermediate"


def summarize_run(config_path: Path) -> dict[str, Any]:
    payload = load_json(config_path)
    config_args = payload.get("args", {})
    run_dir = config_path.parent
    summary_rows = load_summary(run_dir / "summary.csv")
    epochs = int(config_args.get("epochs", 0) or 0)
    train_transitions = int(config_args.get("train_transitions", 0) or 0)
    trainings = inferred_training_count(config_args, summary_rows)
    cost_units = trainings * epochs * train_transitions
    audited = len([row for row in summary_rows if row.get("stage") == "stage2_latent_audit"])
    return {
        "run_dir": run_dir,
        "dataset": config_args.get("dataset", ""),
        "budget": classify_budget(run_dir, epochs),
        "epochs": epochs,
        "train_transitions": train_transitions,
        "trainings": trainings,
        "audited_runs": audited,
        "cost_units": cost_units,
        "has_summary": bool(summary_rows),
    }


def fmt_int(value: float | int) -> str:
    return f"{int(round(float(value))):,}"


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def hypothetical_full_sweep_cost(args: argparse.Namespace) -> int:
    return (
        args.full_sweep_datasets
        * args.full_sweep_priors
        * args.full_sweep_lambdas
        * args.full_sweep_seeds
        * args.full_sweep_epochs
        * args.full_sweep_train_transitions
    )


def write_report(path: Path, runs: list[dict[str, Any]], args: argparse.Namespace) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["budget"])].append(run)
    group_rows: list[list[str]] = []
    quick_cost = 0
    for budget in ("quick preflight", "standard check", "audit mode", "intermediate"):
        items = grouped.get(budget, [])
        if not items:
            continue
        cost = sum(int(item["cost_units"]) for item in items)
        if budget == "quick preflight":
            quick_cost = cost
        group_rows.append(
            [
                budget,
                fmt_int(len(items)),
                fmt_int(sum(int(item["trainings"]) for item in items)),
                fmt_int(sum(int(item["audited_runs"]) for item in items)),
                fmt_int(cost),
            ]
        )

    full_cost = hypothetical_full_sweep_cost(args)
    denominator = full_cost if full_cost > 0 else 1
    quick_pct = 100.0 * quick_cost / denominator
    all_observed_cost = sum(int(item["cost_units"]) for item in runs)
    lines = [
        "# Preflight Compute Cost Accounting",
        "",
        f"Created: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "Cost unit: one training epoch over one sampled transition. Estimated cost is `number of trainings x epochs x train_transitions`.",
        "This is a relative accounting table for reviewer-facing scale claims; it is not a wall-clock benchmark.",
        "",
        *table(
            ["regime", "run dirs", "trainings", "audited runs", "transition-epoch units"],
            group_rows,
        ),
        "",
        "Hypothetical full sweep comparator:",
        "",
        *table(
            ["datasets", "priors", "lambdas", "seeds", "epochs", "train transitions", "transition-epoch units"],
            [
                [
                    fmt_int(args.full_sweep_datasets),
                    fmt_int(args.full_sweep_priors),
                    fmt_int(args.full_sweep_lambdas),
                    fmt_int(args.full_sweep_seeds),
                    fmt_int(args.full_sweep_epochs),
                    fmt_int(args.full_sweep_train_transitions),
                    fmt_int(full_cost),
                ]
            ],
        ),
        "",
        f"Observed quick-preflight cost is `{quick_pct:.2f}%` of the hypothetical full sweep under the assumptions above.",
        f"All observed/prepared preflight directories together account for `{100.0 * all_observed_cost / denominator:.2f}%` of that full-sweep estimate.",
        "",
        "Protocol interpretation:",
        "- quick mode is triage: it screens whether a prior is worth further attention under a deliberately small budget.",
        "- standard mode is a persistence check: it asks whether the quick signal survives a larger training budget.",
        "- audit mode is a mechanism check: it inspects latent graph-frequency behavior rather than treating rollout gain as sufficient evidence.",
        "- quick candidate topology signal is not final topology-specific evidence.",
        "",
        "Input discovery:",
        f"- scanned `{args.runs_root.relative_to(ROOT)}` for `run_config.json` files",
        f"- output written to `{path.relative_to(ROOT)}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate preflight compute costs from run_config and summary files.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--full-sweep-datasets", type=int, default=4)
    parser.add_argument("--full-sweep-priors", type=int, default=4)
    parser.add_argument("--full-sweep-lambdas", type=int, default=5)
    parser.add_argument("--full-sweep-seeds", type=int, default=5)
    parser.add_argument("--full-sweep-epochs", type=int, default=100)
    parser.add_argument("--full-sweep-train-transitions", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.runs_root = args.runs_root.resolve()
    args.output = args.output.resolve()
    config_paths = sorted(args.runs_root.rglob("run_config.json"))
    runs = [summarize_run(path) for path in config_paths]
    write_report(args.output, runs, args)
    print(f"Wrote {args.output.relative_to(ROOT)} from {len(runs)} run_config.json file(s)")


if __name__ == "__main__":
    main()
