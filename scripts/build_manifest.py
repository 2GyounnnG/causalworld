from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HORIZONS = [1, 2, 4, 8, 16]
TARGET_JSON_PATTERNS = (
    "rmd17_*.json",
    "*aspirin*.json",
    "*weight_sweep*.json",
    "*laplacian_ablation*.json",
    "validation*.json",
    "long_horizon.json",
    "*wolfram*.json",
)
EXCLUDED_DIR_NAMES = {
    ".git",
    "__pycache__",
    "analysis_out",
    "venv",
    ".venv",
    "env",
    "envs",
    "miniforge",
    "miniforge3",
    "miniconda",
    "miniconda3",
    "anaconda3",
}
LIKELY_SEARCH_DIRS = ("", "results", "artifacts", "logs")
MANIFEST_COLUMNS = [
    "experiment_name",
    "source_file",
    "task_family",
    "dataset",
    "molecule",
    "prior",
    "encoder",
    "seed",
    "epochs",
    "prior_weight",
    "laplacian_mode",
    "horizon",
    "metric_name",
    "metric_value",
    "status",
    "completed",
    "notes",
]
FAILED_COLUMNS = [
    "source_file",
    "experiment_name",
    "status",
    "completed",
    "expected",
    "observed",
    "missing",
    "notes",
]
DUPLICATE_COLUMNS = [
    "duplicate_key",
    "kept_source_file",
    "removed_source_file",
    "experiment_name",
    "prior",
    "seed",
    "horizon",
    "metric_value",
    "reason",
]
DUPLICATE_ROLLOUT_KEY_COLUMNS = [
    "experiment_name",
    "molecule",
    "encoder",
    "prior",
    "prior_weight",
    "laplacian_mode",
    "seed",
    "horizon",
    "metric_name",
]


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def as_bool(value: bool) -> str:
    return "true" if value else "false"


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def finite_int(value: Any) -> int | None:
    number = finite_float(value)
    if number is None:
        return None
    return int(number)


def weight_key(value: Any) -> str | None:
    number = finite_float(value)
    if number is None:
        return None
    return f"{number:.12g}"


def note_value(notes: Any, name: str) -> str:
    for part in str(notes or "").split(";"):
        part = part.strip()
        if part.startswith(f"{name}="):
            return part.split("=", 1)[1].strip()
    return ""


def duplicate_key_string(key: tuple[str, ...]) -> str:
    return "|".join(f"{column}={value}" for column, value in zip(DUPLICATE_ROLLOUT_KEY_COLUMNS, key))


def parse_key(key: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    parts = key.split("|")
    if len(parts) >= 3:
        if parts[0] in {"aspirin", "ethanol", "malonaldehyde", "benzene", "azobenzene", "naphthalene"}:
            meta["molecule"] = parts[0]
            meta["encoder"] = parts[1]
            meta["prior"] = parts[2]
        else:
            meta["encoder"] = parts[0]
            meta["prior"] = parts[1]
    elif len(parts) == 2:
        meta["encoder"] = parts[0]
        meta["prior"] = parts[1]

    for part in parts:
        if part.startswith("seed="):
            meta["seed"] = int(part.split("=", 1)[1])
        elif part.startswith("w="):
            meta["prior_weight"] = float(part.split("=", 1)[1])
        elif part.startswith("mode="):
            meta["laplacian_mode"] = part.split("=", 1)[1]
    return meta


def classify_file(path: Path) -> tuple[str, str, str, str, str]:
    name = path.name
    if name == "rmd17_aspirin_10seed_checkpointed_results.json":
        return "rmd17_aspirin_10seed_checkpointed", "rmd17", "rmd17", "aspirin", "current"
    if name.startswith("rmd17_aspirin_10seed_results"):
        return "rmd17_aspirin_10seed", "rmd17", "rmd17", "aspirin", "current" if ".pre_" not in name else "snapshot"
    if name == "rmd17_aspirin_weight_sweep.json":
        return "rmd17_weight_sweep", "rmd17", "rmd17", "aspirin", "current"
    if name == "rmd17_aspirin_laplacian_ablation.json":
        return "rmd17_laplacian_ablation", "rmd17", "rmd17", "aspirin", "current"
    if name.startswith("rmd17_aspirin_results"):
        return "rmd17_aspirin_3seed_pilot", "rmd17", "rmd17", "aspirin", "pilot"
    if name.startswith("rmd17_ethanol_results"):
        return "rmd17_ethanol_3seed_pilot", "rmd17", "rmd17", "ethanol", "pilot"
    if name.startswith("rmd17_malonaldehyde_results"):
        return "rmd17_malonaldehyde_3seed_pilot", "rmd17", "rmd17", "malonaldehyde", "pilot"
    if name == "validation_wolfram_flat_10seed_200ep.json":
        return "wolfram_flat_10seed_200ep", "wolfram", "wolfram", "", "current"
    if name == "validation_wolfram_flat_10seed_200ep.pre_restart_20260422_214952.json":
        return "wolfram_flat_10seed_200ep", "wolfram", "wolfram", "", "snapshot"
    if name == "validation_10seed_flat_200ep.json":
        return "wolfram_flat_200ep_3seed_snapshot", "wolfram", "wolfram", "", "snapshot"
    if name.startswith("validation_10seed_flat"):
        return "wolfram_flat_10seed", "wolfram", "wolfram", "", "pilot"
    if name.startswith("validation_10seed_hypergraph"):
        return "wolfram_hypergraph_10seed", "wolfram", "wolfram", "", "pilot"
    if name == "long_horizon.json":
        return "wolfram_long_horizon", "wolfram", "wolfram", "", "pilot"
    return path.stem, "unknown", "", "", "unknown"


def discover_files(root: Path) -> tuple[list[Path], list[Path]]:
    json_paths: set[Path] = set()
    log_paths: set[Path] = set()

    def is_excluded(path: Path) -> bool:
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            parts = path.parts
        return any(part in EXCLUDED_DIR_NAMES for part in parts)

    search_roots: list[Path] = []
    for name in LIKELY_SEARCH_DIRS:
        search_root = root / name if name else root
        if search_root.exists() and search_root.is_dir() and not is_excluded(search_root):
            search_roots.append(search_root)

    for search_root in search_roots:
        for path in search_root.rglob("*"):
            if is_excluded(path) or not path.is_file():
                continue
            lower_name = path.name.lower()
            if path.suffix.lower() == ".json" and any(fnmatch.fnmatch(lower_name, pattern.lower()) for pattern in TARGET_JSON_PATTERNS):
                json_paths.add(path)
            if path.suffix.lower() == ".log" or ("tmux" in lower_name and path.parent.name == "logs"):
                log_paths.add(path)

    return sorted(json_paths), sorted(log_paths)


def row_from_result(
    *,
    path: Path,
    root: Path,
    experiment_name: str,
    task_family: str,
    dataset: str,
    file_molecule: str,
    status: str,
    key: str,
    result: dict[str, Any],
    horizon: int | str,
    metric_name: str,
    metric_value: Any,
    seed_override: int | None = None,
    prior_override: str | None = None,
    encoder_override: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    config = result.get("config", {}) if isinstance(result, dict) else {}
    meta = parse_key(key)
    prior_weight = meta.get("prior_weight", config.get("prior_weight", ""))
    if meta.get("prior_weight") != "" and config.get("prior_weight") not in (None, "", prior_weight):
        notes = "; ".join(filter(None, [notes, f"key_weight={meta.get('prior_weight')} config_weight={config.get('prior_weight')}"]))

    return {
        "experiment_name": experiment_name,
        "source_file": rel(path, root),
        "task_family": task_family,
        "dataset": dataset,
        "molecule": meta.get("molecule", config.get("molecule", file_molecule)),
        "prior": prior_override or meta.get("prior", config.get("prior", "")),
        "encoder": encoder_override or meta.get("encoder", config.get("encoder", "")),
        "seed": seed_override if seed_override is not None else meta.get("seed", config.get("seed", "")),
        "epochs": config.get("num_epochs", ""),
        "prior_weight": prior_weight,
        "laplacian_mode": meta.get("laplacian_mode", config.get("laplacian_mode", "")),
        "horizon": horizon,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "status": status,
        "completed": as_bool(finite_float(metric_value) is not None),
        "notes": notes,
    }


def parse_rollout_result(
    *,
    path: Path,
    root: Path,
    data: dict[str, Any],
    experiment_name: str,
    task_family: str,
    dataset: str,
    file_molecule: str,
    source_state: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results = data.get("results", data)
    if not isinstance(results, dict):
        return rows
    status = "ok" if source_state in {"current", "pilot"} else "duplicate_snapshot"
    for key, result in results.items():
        if not isinstance(result, dict) or "rollout_errors" not in result:
            continue
        for horizon, value in result.get("rollout_errors", {}).items():
            rows.append(
                row_from_result(
                    path=path,
                    root=root,
                    experiment_name=experiment_name,
                    task_family=task_family,
                    dataset=dataset,
                    file_molecule=file_molecule,
                    status=status,
                    key=key,
                    result=result,
                    horizon=horizon,
                    metric_name="rollout_error",
                    metric_value=value,
                    notes=f"source_state={source_state}",
                )
            )
        if "final_loss" in result:
            rows.append(
                row_from_result(
                    path=path,
                    root=root,
                    experiment_name=experiment_name,
                    task_family=task_family,
                    dataset=dataset,
                    file_molecule=file_molecule,
                    status=status,
                    key=key,
                    result=result,
                    horizon="",
                    metric_name="final_loss",
                    metric_value=result.get("final_loss"),
                    notes=f"source_state={source_state}",
                )
            )
    return rows


def parse_wolfram_result(
    *,
    path: Path,
    root: Path,
    data: dict[str, Any],
    experiment_name: str,
    task_family: str,
    dataset: str,
    source_state: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results = data.get("results", {})
    if not isinstance(results, dict):
        return rows
    completed_seeds = data.get("completed_seeds")
    config = data.get("config", {}) if isinstance(data.get("config"), dict) else {}
    seeds = completed_seeds if isinstance(completed_seeds, list) else config.get("seeds", [])
    status = "ok" if source_state in {"current", "pilot"} else "duplicate_snapshot"
    for key, by_horizon in results.items():
        meta = parse_key(key)
        prior = meta.get("prior", "")
        encoder = meta.get("encoder", data.get("encoder", ""))
        if not isinstance(by_horizon, dict):
            continue
        for horizon, values in by_horizon.items():
            if not isinstance(values, list):
                continue
            for index, value in enumerate(values):
                seed = seeds[index] if isinstance(seeds, list) and index < len(seeds) else index
                result = {
                    "config": {
                        "encoder": encoder,
                        "prior": prior,
                        "seed": seed,
                        "num_epochs": config.get("num_epochs", 200 if "200ep" in path.name else ""),
                        "prior_weight": (data.get("prior_weight", "") if prior != "none" else 0.0),
                    }
                }
                rows.append(
                    row_from_result(
                        path=path,
                        root=root,
                        experiment_name=experiment_name,
                        task_family=task_family,
                        dataset=dataset,
                        file_molecule="",
                        status=status,
                        key=key,
                        result=result,
                        horizon=horizon,
                        metric_name="rollout_error",
                        metric_value=value,
                        seed_override=int(seed) if str(seed).isdigit() else None,
                        prior_override=prior,
                        encoder_override=encoder,
                        notes=f"source_state={source_state}",
                    )
                )
    return rows


def expected_run_summary(rows: list[dict[str, Any]], source: str, experiment: str) -> list[dict[str, str]]:
    by_source = [row for row in rows if row["source_file"] == source and row["metric_name"] == "rollout_error"]
    out: list[dict[str, str]] = []

    def add(expected: set[tuple[Any, ...]], present: set[tuple[Any, ...]], notes: str, status_if_missing: str = "incomplete") -> None:
        missing = sorted(expected - present)
        out.append(
            {
                "source_file": source,
                "experiment_name": experiment,
                "status": "complete" if not missing else status_if_missing,
                "completed": as_bool(not missing),
                "expected": str(len(expected)),
                "observed": str(len(expected) - len(missing)),
                "missing": ";".join(str(item) for item in missing[:80]),
                "notes": notes,
            }
        )

    if experiment == "rmd17_aspirin_10seed":
        expected = {
            (prior, seed, h)
            for prior in ["none", "euclidean", "spectral"]
            for seed in range(10)
            for h in HORIZONS
        }
        present = {
            (row["prior"], finite_int(row["seed"]), finite_int(row["horizon"]))
            for row in by_source
            if row["prior"] in {"none", "euclidean", "spectral"}
            and finite_int(row["seed"]) is not None
            and finite_int(row["horizon"]) is not None
        }
        add(expected, present, "expected 3 priors x 10 seeds x 5 horizons; checked by (prior, seed, horizon)")
    elif experiment == "rmd17_weight_sweep":
        unknown_weight_rows = [
            row
            for row in by_source
            if row["prior"] == "none" and weight_key(row.get("prior_weight")) is None
        ]
        if unknown_weight_rows:
            out.append(
                {
                    "source_file": source,
                    "experiment_name": experiment,
                    "status": "unknown",
                    "completed": "false",
                    "expected": str(3 * 4 * 3 * len(HORIZONS)),
                    "observed": str(len(by_source)),
                    "missing": "",
                    "notes": "one or more none-prior weight-sweep rows lack a key/config prior_weight; cannot assign them to sweep weights",
                }
            )
            return out
        expected = {
            (prior, weight_key(weight), seed, h)
            for prior in ["none", "euclidean", "spectral"]
            for seed in [0, 1, 2]
            for weight in [0.001, 0.01, 0.1, 1.0]
            for h in HORIZONS
        }
        present = {
            (row["prior"], weight_key(row.get("prior_weight")), finite_int(row["seed"]), finite_int(row["horizon"]))
            for row in by_source
            if row["prior"] in {"none", "euclidean", "spectral"}
            and weight_key(row.get("prior_weight")) is not None
            and finite_int(row["seed"]) is not None
            and finite_int(row["horizon"]) is not None
        }
        add(expected, present, "expected 3 priors x 4 weights x 3 seeds x 5 horizons; checked by (prior, prior_weight, seed, horizon)")
    elif experiment == "rmd17_laplacian_ablation":
        expected = {
            (mode, seed, h)
            for mode in ["per_frame", "fixed_frame0", "fixed_mean"]
            for seed in [0, 1, 2, 3, 4]
            for h in HORIZONS
        }
        present = {
            (row["laplacian_mode"], finite_int(row["seed"]), finite_int(row["horizon"]))
            for row in by_source
            if row["prior"] == "spectral"
            and row["laplacian_mode"] in {"per_frame", "fixed_frame0", "fixed_mean"}
            and finite_int(row["seed"]) is not None
            and finite_int(row["horizon"]) is not None
        }
        add(expected, present, "expected 3 Laplacian modes x 5 seeds x 5 horizons; checked by (laplacian_mode, seed, horizon)")
    elif experiment == "wolfram_flat_10seed_200ep":
        expected = {
            (prior, seed, h)
            for prior in ["none", "euclidean", "spectral"]
            for seed in range(10)
            for h in HORIZONS
        }
        present = {
            (row["prior"], finite_int(row["seed"]), finite_int(row["horizon"]))
            for row in by_source
            if row["prior"] in {"none", "euclidean", "spectral"}
            and finite_int(row["seed"]) is not None
            and finite_int(row["horizon"]) is not None
        }
        add(expected, present, "expected 3 priors x 10 seeds x 5 horizons; checked by (prior, seed, horizon)")
    return out


def inspect_log(path: Path, root: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lower = text.lower()
    status = "failed" if any(token in lower for token in ["traceback", "error", "killed", "exit=1"]) else "incomplete"
    if "=== finished" in lower and "exit=0" in lower:
        status = "complete"
    starts = len(re.findall(r"^=== Starting ", text, flags=re.MULTILINE))
    saves = len(re.findall(r"saved partial results", text))
    last_line = next((line for line in reversed(text.splitlines()) if line.strip()), "")
    return {
        "source_file": rel(path, root),
        "experiment_name": path.stem,
        "status": status,
        "completed": as_bool(status == "complete"),
        "expected": "",
        "observed": f"starts={starts}; saves={saves}; lines={len(text.splitlines())}",
        "missing": "",
        "notes": last_line[:500],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def dedupe_manifest_rows(manifest_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(manifest_rows):
        if row.get("metric_name") != "rollout_error":
            continue
        key = tuple(str(row.get(column, "")) for column in DUPLICATE_ROLLOUT_KEY_COLUMNS)
        groups[key].append((index, row))

    removed_indices: set[int] = set()
    duplicate_rows: list[dict[str, Any]] = []
    state_rank = {"current": 0, "pilot": 1, "snapshot": 2}

    def rank(item: tuple[int, dict[str, Any]]) -> tuple[int, int, int, int]:
        index, row = item
        status_rank = 0 if row.get("status") == "ok" else 1 if row.get("status") == "duplicate_snapshot" else 2
        source_state = note_value(row.get("notes", ""), "source_state")
        if not source_state and row.get("status") == "duplicate_snapshot":
            source_state = "snapshot"
        source_rank = state_rank.get(source_state, 3)
        return (status_rank, source_rank, len(str(row.get("source_file", ""))), index)

    def reason_for(kept: dict[str, Any], removed: dict[str, Any]) -> str:
        if kept.get("status") != removed.get("status"):
            return f"duplicate rollout_error row; kept status={kept.get('status', '')}"
        kept_state = note_value(kept.get("notes", ""), "source_state")
        removed_state = note_value(removed.get("notes", ""), "source_state")
        if kept_state != removed_state:
            return f"duplicate rollout_error row; kept source_state={kept_state or 'unknown'}"
        if len(str(kept.get("source_file", ""))) != len(str(removed.get("source_file", ""))):
            return "duplicate rollout_error row; kept shortest source_file after status/source_state tie"
        return "duplicate rollout_error row; kept first row after all tie-breakers"

    for key, items in groups.items():
        if len(items) < 2:
            continue
        kept_index, kept = min(items, key=rank)
        for index, removed in items:
            if index == kept_index:
                continue
            removed_indices.add(index)
            duplicate_rows.append(
                {
                    "duplicate_key": duplicate_key_string(key),
                    "kept_source_file": kept.get("source_file", ""),
                    "removed_source_file": removed.get("source_file", ""),
                    "experiment_name": removed.get("experiment_name", ""),
                    "prior": removed.get("prior", ""),
                    "seed": removed.get("seed", ""),
                    "horizon": removed.get("horizon", ""),
                    "metric_value": removed.get("metric_value", ""),
                    "reason": reason_for(kept, removed),
                }
            )

    deduped_rows = [row for index, row in enumerate(manifest_rows) if index not in removed_indices]
    return deduped_rows, duplicate_rows


def write_inventory(
    path: Path,
    root: Path,
    manifest_rows: list[dict[str, Any]],
    findings: list[dict[str, str]],
    tree_lines: list[str],
    duplicate_rows: list[dict[str, Any]],
) -> None:
    counts = Counter(row["experiment_name"] for row in manifest_rows if row["metric_name"] == "rollout_error" and row["status"] == "ok")
    incomplete = [row for row in findings if row["completed"] != "true"]
    lines = [
        "# Run Inventory",
        "",
        f"Repo root inspected: `{root}`",
        "",
        "## Repo Tree",
        "",
        "```",
        *tree_lines[:250],
        "```",
        "",
        "## Current Result Inventory",
        "",
    ]
    for experiment, count in sorted(counts.items()):
        lines.append(f"- `{experiment}`: {count} rollout metric rows")
    lines.extend(["", "## Incomplete, Failed, Or Duplicate Signals", ""])
    affected = sorted({row.get("experiment_name", "") for row in duplicate_rows if row.get("experiment_name", "")})
    if duplicate_rows:
        lines.append(
            f"- Duplicate rollout rows removed from `manifest.csv`: {len(duplicate_rows)} "
            f"(affected experiments: {', '.join(f'`{experiment}`' for experiment in affected)})"
        )
    else:
        lines.append("- Duplicate rollout rows removed from `manifest.csv`: 0 (affected experiments: none)")
    if incomplete:
        for row in incomplete:
            lines.append(
                f"- `{row['source_file']}`: {row['status']} ({row['observed']}/{row['expected'] or '?'}) {row['notes']}"
            )
    else:
        lines.append("- No incomplete or failed target runs detected.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Raw experiment outputs were not modified.",
            "- Snapshot files are retained in the manifest as `duplicate_snapshot` and excluded from aggregate conclusions.",
            "- Current partial experiments are still represented in aggregates, with incomplete status recorded separately.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="analysis_out")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    json_paths, log_paths = discover_files(root)
    manifest_rows: list[dict[str, Any]] = []
    findings: list[dict[str, str]] = []

    for path in json_paths:
        experiment_name, task_family, dataset, molecule, source_state = classify_file(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            findings.append(
                {
                    "source_file": rel(path, root),
                    "experiment_name": experiment_name,
                    "status": "parse_error",
                    "completed": "false",
                    "expected": "",
                    "observed": "",
                    "missing": "",
                    "notes": str(exc),
                }
            )
            continue
        if not isinstance(data, dict):
            continue
        if experiment_name.startswith("wolfram") or path.name.startswith("validation"):
            rows = parse_wolfram_result(
                path=path,
                root=root,
                data=data,
                experiment_name=experiment_name,
                task_family=task_family,
                dataset=dataset,
                source_state=source_state,
            )
            if not rows:
                rows = parse_rollout_result(
                    path=path,
                    root=root,
                    data=data,
                    experiment_name=experiment_name,
                    task_family=task_family,
                    dataset=dataset,
                    file_molecule=molecule,
                    source_state=source_state,
                )
        else:
            rows = parse_rollout_result(
                path=path,
                root=root,
                data=data,
                experiment_name=experiment_name,
                task_family=task_family,
                dataset=dataset,
                file_molecule=molecule,
                source_state=source_state,
            )
        manifest_rows.extend(rows)
        if source_state == "snapshot":
            findings.append(
                {
                    "source_file": rel(path, root),
                    "experiment_name": experiment_name,
                    "status": "duplicate_snapshot",
                    "completed": "false",
                    "expected": "",
                    "observed": str(len(rows)),
                    "missing": "",
                    "notes": "archival snapshot; excluded from aggregate conclusions",
                }
            )
        findings.extend(expected_run_summary(manifest_rows, rel(path, root), experiment_name))

    for path in log_paths:
        log_finding = inspect_log(path, root)
        if log_finding["status"] != "complete":
            findings.append(log_finding)

    tree_lines = [rel(p, root) for p in sorted(root.glob("*"))]
    tree_lines.extend(rel(p, root) for p in sorted(root.glob("*/*")) if ".git" not in p.parts and "__pycache__" not in p.parts)
    tree_lines.extend(rel(p, root) for p in sorted(root.glob("*/*/*")) if ".git" not in p.parts and "__pycache__" not in p.parts)
    deduped_manifest_rows, duplicate_rows = dedupe_manifest_rows(manifest_rows)

    if args.dry_run:
        print(f"would write {out / 'manifest_raw.csv'} with {len(manifest_rows)} rows")
        print(f"would write {out / 'manifest.csv'} with {len(deduped_manifest_rows)} rows")
        print(f"would write {out / 'duplicate_manifest_rows.csv'} with {len(duplicate_rows)} rows")
        print(f"would write {out / 'failed_or_incomplete_runs.csv'} with {len(findings)} rows")
        print(f"would write {out / 'RUN_INVENTORY.md'}")
        return

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "manifest_raw.csv", manifest_rows, MANIFEST_COLUMNS)
    write_csv(out / "manifest.csv", deduped_manifest_rows, MANIFEST_COLUMNS)
    write_csv(out / "duplicate_manifest_rows.csv", duplicate_rows, DUPLICATE_COLUMNS)
    write_csv(out / "failed_or_incomplete_runs.csv", findings, FAILED_COLUMNS)
    write_inventory(out / "RUN_INVENTORY.md", root, deduped_manifest_rows, findings, tree_lines, duplicate_rows)
    print(f"wrote {out / 'manifest_raw.csv'} ({len(manifest_rows)} rows)")
    print(f"wrote {out / 'manifest.csv'} ({len(deduped_manifest_rows)} rows)")
    print(f"wrote {out / 'duplicate_manifest_rows.csv'} ({len(duplicate_rows)} rows)")
    print(f"wrote {out / 'failed_or_incomplete_runs.csv'} ({len(findings)} rows)")
    print(f"wrote {out / 'RUN_INVENTORY.md'}")


if __name__ == "__main__":
    main()
