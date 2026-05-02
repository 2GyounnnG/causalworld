"""Aggregate node-order sanity summaries and emit manuscript-ready snippets."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "analysis_out" / "preflight_runs" / "node_order_sanity"
DEFAULT_REPORT = ROOT / "analysis_out" / "NODE_ORDER_SANITY_REPORT.md"
DEFAULT_TABLE = ROOT / "paper" / "tables" / "node_order_sanity_summary.md"
DEFAULT_SNIPPET = ROOT / "paper" / "snippets" / "node_order_sanity_paragraph.md"
CONDITIONS = ("A_baseline", "B_permuted_both", "C_permuted_data_only")


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def stats(values: list[float]) -> tuple[float, float, int]:
    vals = [float(value) for value in values if finite(value)]
    if not vals:
        return float("nan"), float("nan"), 0
    return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0, len(vals)


def fmt(value: object) -> str:
    return "NA" if not finite(value) else f"{float(value):.4f}"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summarize(out_root: Path) -> dict[str, dict[str, float | int]]:
    rows = [
        row
        for row in read_rows(out_root / "summary.csv")
        if row.get("stage") == "node_order_sanity" and row.get("status") == "ok"
    ]
    out = {}
    for condition in CONDITIONS:
        values = [float(row.get("H32", "nan")) for row in rows if row.get("condition") == condition]
        mean, std, n = stats(values)
        out[condition] = {"mean": mean, "std": std, "n": n}
    return out


def decision(summary: dict[str, dict[str, float | int]]) -> str:
    a = summary["A_baseline"]
    b = summary["B_permuted_both"]
    c = summary["C_permuted_data_only"]
    a_mean = float(a["mean"])
    b_mean = float(b["mean"])
    c_mean = float(c["mean"])
    pooled_std = max(float(a["std"]), float(b["std"]), 1e-12)
    data_only_std = max(float(a["std"]), float(c["std"]), 1e-12)
    if finite(a_mean) and finite(c_mean) and c_mean - a_mean < data_only_std:
        return "WARNING: Model appears to ignore graph structure; sanity check inconclusive."
    if finite(a_mean) and finite(b_mean) and abs(a_mean - b_mean) < pooled_std:
        return "PASS: Model is approximately permutation-equivariant; SMPG control is interpretable as a topology-breaking control."
    if finite(a_mean) and finite(b_mean) and b_mean - a_mean > 2.0 * pooled_std:
        return "PARTIAL: Model has node-order dependence; SMPG is a practical topology-breaking stress test rather than an isolated spectral control."
    return "INCONCLUSIVE: Node-order sanity check needs complete seed coverage."


def table(summary: dict[str, dict[str, float | int]]) -> list[str]:
    lines = ["| condition | mean_H32 | std_H32 | n |", "| --- | --- | --- | --- |"]
    for condition in CONDITIONS:
        row = summary[condition]
        lines.append(f"| {condition} | {fmt(row['mean'])} | {fmt(row['std'])} | {row['n']} |")
    return lines


def build_report(summary: dict[str, dict[str, float | int]]) -> str:
    return "\n".join(
        [
            "# Node-Order Sanity Report",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            *table(summary),
            "",
            f"Decision: {decision(summary)}",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate node-order sanity results.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--snippet", type=Path, default=DEFAULT_SNIPPET)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.out_root.resolve() / "summary.csv"
    summary = summarize(args.out_root.resolve())
    if args.dry_run:
        print("Dry run only. No files will be written.")
        print(f"WOULD READ {summary_path} [{'exists' if summary_path.exists() else 'missing'}]")
        print(f"Decision preview: {decision(summary)}")
        return
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.snippet.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(summary)
    args.report.write_text(report, encoding="utf-8")
    args.table.write_text("\n".join(["# Node-Order Sanity Summary", "", *table(summary), ""]), encoding="utf-8")
    args.snippet.write_text(decision(summary) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.table}")
    print(f"Wrote {args.snippet}")


if __name__ == "__main__":
    main()
