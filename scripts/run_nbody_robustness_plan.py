from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "analysis_out" / "preflight_runs" / "nbody_robustness"
PREFLIGHT_SCRIPT = ROOT / "scripts" / "graph_prior_preflight_check.py"


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def has_classification(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    with summary_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("stage") == "classification" and row.get("classification"):
                return True
    return False


def interpreter_command(args: argparse.Namespace) -> tuple[list[str], str]:
    if args.conda_env:
        return ["conda", "run", "-n", args.conda_env, "python"], f"conda run -n {args.conda_env}"
    if Path(args.python_exe) == Path(sys.executable):
        return [args.python_exe], f"sys.executable ({args.python_exe})"
    return [args.python_exe], f"--python-exe ({args.python_exe})"


def command_for_run(args: argparse.Namespace, *, distance_k: int, budget_label: str, epochs: int) -> list[str]:
    out_dir = args.out_root / f"distance_k_{distance_k:02d}" / f"{budget_label}_ep{epochs}"
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


def shell_quote(parts: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in parts)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "JCS defensive N-body robustness plan; no existing results are modified.",
        "runs": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable N-body robustness runner for the JCS revision plan."
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--distance-k", type=parse_csv_ints, default=[4, 8, 12])
    parser.add_argument("--seeds", type=parse_csv_ints, default=[0, 1, 2])
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
        help="Run child preflight commands as `conda run -n ENV python ...`; overrides --python-exe.",
    )
    parser.add_argument("--calibration-reference-prior", default="graph")
    parser.add_argument("--calibration-target-ratio", type=float, default=1.0)
    parser.add_argument("--force", action="store_true", help="Re-run completed output directories.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing them.")
    parser.add_argument("--smoke", action="store_true", help="Only run the first quick-budget command.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root = args.out_root.resolve()
    _python_cmd, interpreter_source = interpreter_command(args)
    print(f"Interpreter mode: {interpreter_source}")
    if 32 not in set(args.horizons):
        raise ValueError("--horizons must include 32 for preflight classification")

    budgets = [("quick", int(args.quick_epochs)), ("standard", int(args.standard_epochs))]
    planned: list[dict[str, object]] = []
    commands: list[list[str]] = []
    for distance_k in args.distance_k:
        for budget_label, epochs in budgets:
            command = command_for_run(args, distance_k=distance_k, budget_label=budget_label, epochs=epochs)
            out_dir = args.out_root / f"distance_k_{distance_k:02d}" / f"{budget_label}_ep{epochs}"
            complete = has_classification(out_dir / "summary.csv")
            should_run = args.force or not complete
            planned.append(
                {
                    "distance_k": distance_k,
                    "budget": budget_label,
                    "epochs": epochs,
                    "out_dir": str(out_dir),
                    "complete": complete,
                    "will_run": should_run,
                    "command": command,
                }
            )
            if should_run:
                commands.append(command)

    for command in commands:
        print(shell_quote(command))

    if args.dry_run:
        print(f"Dry run only. Planned {len(commands)} command(s).")
        return

    write_manifest(args.out_root / "run_manifest.json", planned)
    if args.smoke:
        commands = commands[:1]
    for command in commands:
        print(f"Running: {shell_quote(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
