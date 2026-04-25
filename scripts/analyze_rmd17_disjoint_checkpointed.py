from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 2, 4, 8, 16]
PRIORS = ["none", "euclidean", "spectral"]
AGG_COLUMNS = [
    "experiment_name",
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
    "notes",
]
ROLLOUT_STAT_COLUMNS = {"mean", "std", "stderr", "median", "min", "max"}
RATIO_COLUMNS = {"h16_over_h1"}


def parse_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def format_float(value: Any, column: str = "") -> str:
    number = parse_float(value)
    if number is None:
        return ""
    if column in RATIO_COLUMNS:
        return f"{number:.1e}" if abs(number) >= 1000 else f"{number:.1f}"
    if column in ROLLOUT_STAT_COLUMNS:
        return f"{number:.1f}" if abs(number) >= 10 else f"{number:.3f}"
    return f"{number:.10g}"


def load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def key_meta(key: str) -> dict[str, Any]:
    parts = key.split("|")
    meta = {"molecule": "", "encoder": "", "prior": "", "seed": ""}
    if len(parts) >= 3:
        meta["molecule"] = parts[0]
        meta["encoder"] = parts[1]
        meta["prior"] = parts[2]
    for part in parts:
        if part.startswith("seed="):
            meta["seed"] = part.split("=", 1)[1]
    return meta


def successful_results(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    results = payload.get("results", {})
    if not isinstance(results, dict):
        return []
    output = []
    for key, result in results.items():
        if not isinstance(result, dict):
            continue
        if result.get("status", "ok") != "ok":
            continue
        if not isinstance(result.get("rollout_errors"), dict):
            continue
        output.append((key, result))
    return output


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": "", "std": "", "stderr": "", "median": "", "min": "", "max": ""}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": std,
        "stderr": float(std / math.sqrt(arr.size)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def aggregate_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str, int], list[float]] = defaultdict(list)
    exemplars: dict[tuple[str, str, str, str, str, str, int], dict[str, Any]] = {}
    experiment_name = payload.get("experiment_name", "")
    for key, result in successful_results(payload):
        meta = key_meta(key)
        config = result.get("config", {})
        prior_weight = str(config.get("prior_weight", ""))
        laplacian_mode = str(config.get("laplacian_mode", ""))
        for horizon, value in result.get("rollout_errors", {}).items():
            number = parse_float(value)
            if number is None:
                continue
            group = (
                str(experiment_name),
                str(meta.get("molecule") or config.get("molecule", "")),
                str(meta.get("encoder") or config.get("encoder", "")),
                str(meta.get("prior") or config.get("prior", "")),
                prior_weight,
                laplacian_mode,
                int(horizon),
            )
            grouped[group].append(number)
            exemplars.setdefault(group, {"config": config})

    rows = []
    for group, values in sorted(grouped.items()):
        experiment_name, molecule, encoder, prior, prior_weight, laplacian_mode, horizon = group
        row = {column: "" for column in AGG_COLUMNS}
        row.update(
            {
                "experiment_name": experiment_name,
                "molecule": molecule,
                "encoder": encoder,
                "prior": prior,
                "prior_weight": prior_weight,
                "laplacian_mode": laplacian_mode,
                "horizon": horizon,
                "notes": "strict disjoint train/eval frame split; checkpointed run",
            }
        )
        row.update(summarize(values))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=AGG_COLUMNS)
        writer.writeheader()
        for row in rows:
            serial = {}
            for column in AGG_COLUMNS:
                value = row.get(column, "")
                if isinstance(value, float):
                    serial[column] = format_float(value, column)
                else:
                    serial[column] = str(value)
            writer.writerow(serial)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available yet._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append(format_float(value, column) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def completion_lines(payload: dict[str, Any], expected_molecules: list[str], expected_seeds: list[int]) -> list[str]:
    observed = set()
    for key, _result in successful_results(payload):
        meta = key_meta(key)
        seed = meta.get("seed")
        if seed == "":
            continue
        observed.add((meta.get("molecule", ""), meta.get("prior", ""), int(seed)))
    lines = []
    for molecule in expected_molecules:
        for prior in PRIORS:
            count = sum((molecule, prior, seed) in observed for seed in expected_seeds)
            lines.append(f"- {molecule} / {prior}: {count}/{len(expected_seeds)} seeds complete")
    return lines


def checkpoint_count(payload: dict[str, Any]) -> int:
    count = 0
    for _key, result in successful_results(payload):
        path = result.get("checkpoint_path")
        if not path:
            continue
        checkpoint_path = ROOT / path
        if checkpoint_path.exists():
            count += 1
    return count


def overlap_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, result in successful_results(payload):
        meta = key_meta(key)
        metadata = result.get("metadata", {})
        rows.append(
            {
                "molecule": meta.get("molecule", ""),
                "prior": meta.get("prior", ""),
                "seed": meta.get("seed", ""),
                "train_eval_overlap_count": metadata.get("train_eval_overlap_count", ""),
                "train_eval_start_overlap_count": metadata.get("train_eval_start_overlap_count", ""),
                "checkpoint_path": result.get("checkpoint_path", ""),
            }
        )
    return sorted(rows, key=lambda row: (row["molecule"], row["prior"], int(row["seed"] or 0)))


def write_report(
    path: Path,
    *,
    payload: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    title: str,
    expected_molecules: list[str],
    expected_seeds: list[int],
) -> None:
    h16_rows = [row for row in aggregate_rows if int(row.get("horizon", 0)) == 16]
    overlap = overlap_rows(payload)
    overlap_bad = [
        row
        for row in overlap
        if str(row.get("train_eval_overlap_count", "")) not in {"", "0"}
        or str(row.get("train_eval_start_overlap_count", "")) not in {"", "0"}
    ]
    lines = [
        f"# {title}",
        "",
        "## Status",
        "",
        *completion_lines(payload, expected_molecules, expected_seeds),
        f"- checkpoints present for completed runs: {checkpoint_count(payload)}",
        f"- frame-overlap audit: {'PASS' if not overlap_bad else 'WARNING'}",
        "",
        "## H=16 Aggregate",
        "",
        markdown_table(h16_rows, ["molecule", "prior", "n", "mean", "std", "median", "min", "max"]),
        "",
        "## Frame-Overlap Audit",
        "",
        markdown_table(
            overlap,
            [
                "molecule",
                "prior",
                "seed",
                "train_eval_start_overlap_count",
                "train_eval_overlap_count",
                "checkpoint_path",
            ],
        ),
        "",
        "## Notes",
        "",
        "- These are new strict disjoint-frame, checkpointed rMD17 runs.",
        "- They do not overwrite old rMD17 aspirin 10-seed result files.",
        "- Lower rollout error is better.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_file(
    *,
    input_path: Path,
    aggregate_path: Path,
    report_path: Path,
    title: str,
    expected_molecules: list[str],
    expected_seeds: list[int],
) -> list[dict[str, Any]]:
    payload = load_payload(input_path)
    rows = aggregate_payload(payload)
    write_csv(aggregate_path, rows)
    write_report(
        report_path,
        payload=payload,
        aggregate_rows=rows,
        title=title,
        expected_molecules=expected_molecules,
        expected_seeds=expected_seeds,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--molecules", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    rows = analyze_file(
        input_path=Path(args.input),
        aggregate_path=Path(args.aggregate),
        report_path=Path(args.report),
        title=args.title,
        expected_molecules=args.molecules,
        expected_seeds=args.seeds,
    )
    print(f"wrote {args.aggregate} ({len(rows)} rows)")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
