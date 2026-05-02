"""Resumable launcher for the N-body 5-seed robustness sweep."""

from __future__ import annotations

import argparse
import csv
import os
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "analysis_out" / "preflight_runs" / "nbody_robustness_5seed"
PREFLIGHT_SCRIPT = ROOT / "scripts" / "graph_prior_preflight_check.py"
AGGREGATE_SCRIPT = ROOT / "scripts" / "aggregate_nbody_robustness_5seed.py"
EXPECTED_PRIORS = ("none", "graph", "permuted_graph", "temporal_smooth")


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def summary_appears_complete(summary_path: Path, seeds: list[int]) -> tuple[bool, str]:
    if not summary_path.exists():
        return False, "missing summary.csv"

    try:
        with summary_path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    except csv.Error as exc:
        return False, f"unreadable summary.csv: {exc}"

    found: set[tuple[int, str]] = set()
    has_classification = False
    has_required_classification_metrics = False
    for row in rows:
        if row.get("stage") == "stage1_mini_train" and row.get("status") == "ok":
            try:
                seed = int(str(row.get("seed", "")).strip())
            except ValueError:
                continue
            prior = str(row.get("prior", "")).strip()
            if seed in seeds and prior in EXPECTED_PRIORS and finite(row.get("H32")):
                found.add((seed, prior))
        elif row.get("stage") == "classification":
            has_classification = bool(row.get("classification"))
            has_required_classification_metrics = all(
                finite(row.get(key))
                for key in ("none_h32", "graph_h32", "permuted_h32", "temporal_smooth_h32")
            )

    missing = [
        f"seed{seed}:{prior}"
        for seed in seeds
        for prior in EXPECTED_PRIORS
        if (seed, prior) not in found
    ]
    if missing:
        return False, "missing stage1 H32 rows for " + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
    if not has_classification:
        return False, "missing classification row"
    if not has_required_classification_metrics:
        return False, "classification row missing required H32 metrics"
    return True, "complete"


def interpreter_command(args: argparse.Namespace) -> tuple[list[str], str]:
    if args.conda_env:
        return ["conda", "run", "-n", args.conda_env, "python"], f"conda run -n {args.conda_env}"
    return [args.python_exe], args.python_exe


def out_dir_for(args: argparse.Namespace, *, distance_k: int, budget_label: str, epochs: int) -> Path:
    return args.out_root / f"distance_k_{distance_k:02d}" / f"{budget_label}_ep{epochs}"


def command_for_run(args: argparse.Namespace, *, distance_k: int, budget_label: str, epochs: int) -> list[str]:
    out_dir = out_dir_for(args, distance_k=distance_k, budget_label=budget_label, epochs=epochs)
    python_cmd, _source = interpreter_command(args)
    return [
        *python_cmd,
        str(PREFLIGHT_SCRIPT),
        "--dataset",
        "nbody_distance",
        "--out-dir",
        str(out_dir),
        "--nbody-n",
        str(args.nbody_n),
        "--nbody-graph-k",
        str(distance_k),
        "--nbody-seed",
        str(args.nbody_seed),
        "--epochs",
        str(epochs),
        "--seeds",
        ",".join(str(seed) for seed in args.seeds),
        "--train-transitions",
        str(args.train_transitions),
        "--eval-transitions",
        str(args.eval_transitions),
        "--raw-transitions",
        str(args.raw_transitions),
        "--train-stride",
        str(args.train_stride),
        "--eval-stride",
        str(args.eval_stride),
        "--raw-stride",
        str(args.raw_stride),
        "--horizons",
        *[str(horizon) for horizon in args.horizons],
        "--prior-weight",
        str(args.prior_weight),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--include-temporal-prior",
        "--calibrate-prior-strength",
        "--calibration-reference-prior",
        args.calibration_reference_prior,
        "--calibration-target-ratio",
        str(args.calibration_target_ratio),
        "--skip-random-graph",
    ]


def command_for_aggregate(args: argparse.Namespace) -> list[str]:
    python_cmd, _source = interpreter_command(args)
    return [
        *python_cmd,
        str(AGGREGATE_SCRIPT),
        "--out-root",
        str(args.out_root),
        "--distance-k",
        ",".join(str(value) for value in args.distance_k),
        "--seeds",
        ",".join(str(value) for value in args.seeds),
        "--quick-epochs",
        str(args.quick_epochs),
        "--standard-epochs",
        str(args.standard_epochs),
    ]


def shell_quote(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def display_command(args: argparse.Namespace, command: list[str], *, child: bool = True) -> str:
    prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(str(args.gpu_id))} " if child and args.gpu_id is not None else ""
    return prefix + shell_quote(command)


def child_env(args: argparse.Namespace) -> dict[str, str] | None:
    if args.gpu_id is None:
        return None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    return env


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "N-body distance-kNN 5-seed robustness sweep; old nbody_robustness outputs are untouched.",
        "runs": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the separate 5-seed N-body robustness evidence package.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--distance-k", type=parse_csv_ints, default=[4, 8, 12])
    parser.add_argument("--seeds", type=parse_csv_ints, default=[0, 1, 2, 3, 4], help="Comma-separated seed subset, e.g. 0,1,2 or 3,4.")
    parser.add_argument("--output-suffix", default="", help="Append suffix to the output root directory name, e.g. _m1.")
    parser.add_argument("--gpu-id", default=None, help="Set CUDA_VISIBLE_DEVICES for each child preflight command.")
    parser.add_argument("--nbody-n", type=int, default=36)
    parser.add_argument("--nbody-seed", type=int, default=0)
    parser.add_argument("--quick-epochs", type=int, default=5)
    parser.add_argument("--standard-epochs", type=int, default=20)
    parser.add_argument("--train-transitions", type=int, default=96)
    parser.add_argument("--eval-transitions", type=int, default=32)
    parser.add_argument("--raw-transitions", type=int, default=64)
    parser.add_argument("--train-stride", type=int, default=5)
    parser.add_argument("--eval-stride", type=int, default=10)
    parser.add_argument("--raw-stride", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument(
        "--conda-env",
        default=None,
        help="Run child commands as `conda run -n ENV python ...`; e.g. --conda-env causalworld.",
    )
    parser.add_argument("--calibration-reference-prior", default="graph")
    parser.add_argument("--calibration-target-ratio", type=float, default=1.0)
    parser.add_argument("--force", action="store_true", help="Re-run even if summary.csv appears complete.")
    parser.add_argument("--dry-run", action="store_true", help="Print exact commands without executing them.")
    parser.add_argument("--smoke", action="store_true", help="Run only the first pending command, then aggregate.")
    parser.add_argument("--skip-aggregate", action="store_true", help="Do not run the summary aggregator after runs.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop the launcher after the first child command failure.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_suffix:
        args.out_root = args.out_root.parent / f"{args.out_root.name}{args.output_suffix}"
    args.out_root = args.out_root.resolve()
    _python_cmd, interpreter_source = interpreter_command(args)
    print(f"Interpreter mode: {interpreter_source}")
    if 32 not in set(args.horizons):
        raise ValueError("--horizons must include 32 for robustness labels")

    budgets = [("quick", int(args.quick_epochs)), ("standard", int(args.standard_epochs))]
    manifest_rows: list[dict[str, object]] = []
    commands: list[list[str]] = []
    for distance_k in args.distance_k:
        for budget_label, epochs in budgets:
            out_dir = out_dir_for(args, distance_k=distance_k, budget_label=budget_label, epochs=epochs)
            summary_path = out_dir / "summary.csv"
            complete, reason = summary_appears_complete(summary_path, args.seeds)
            should_run = args.force or not complete
            command = command_for_run(args, distance_k=distance_k, budget_label=budget_label, epochs=epochs)
            manifest_rows.append(
                {
                    "distance_k": distance_k,
                    "budget": f"{budget_label}_ep{epochs}",
                    "out_dir": str(out_dir),
                    "summary": str(summary_path),
                    "complete": complete,
                    "completion_reason": reason,
                    "will_run": should_run,
                    "command": command,
                }
            )
            if should_run:
                commands.append(command)
            else:
                print(f"SKIP complete {out_dir}: {reason}")

    if args.smoke:
        commands = commands[:1]

    if args.dry_run:
        for command in commands:
            print(f"DRY RUN command: {display_command(args, command)}")
        if not args.skip_aggregate:
            print(f"DRY RUN summary command: {shell_quote(command_for_aggregate(args))}")
        print(f"Dry run only. Pending run command(s): {len(commands)}")
        return

    write_manifest(args.out_root / "run_manifest.json", manifest_rows)
    command_failures: list[dict[str, object]] = []
    env = child_env(args)
    for command in commands:
        print(f"RUN command: {display_command(args, command)}", flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            command_failures.append({"returncode": exc.returncode, "command": command})
            print(f"RUN FAILED returncode={exc.returncode}: {display_command(args, command)}", flush=True)
            if args.fail_fast:
                raise

    if not args.skip_aggregate:
        aggregate_command = command_for_aggregate(args)
        print(f"SUMMARY command: {shell_quote(aggregate_command)}", flush=True)
        subprocess.run(aggregate_command, cwd=ROOT, check=True)
    if command_failures:
        failures_path = args.out_root / "command_failures.json"
        failures_path.write_text(json.dumps(command_failures, indent=2) + "\n", encoding="utf-8")
        print(f"Completed with {len(command_failures)} failed child command(s); wrote {failures_path}", flush=True)


if __name__ == "__main__":
    main()
