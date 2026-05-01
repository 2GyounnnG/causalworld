from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_SCRIPT = ROOT / "scripts" / "graph_prior_preflight_check.py"
DEFAULT_OUT_DIR = ROOT / "analysis_out" / "preflight_runs" / "stronger_no_prior"


def shell_quote(parts: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in parts)


def interpreter_prefix(args: argparse.Namespace) -> tuple[list[str], str]:
    if args.conda_env:
        return ["conda", "run", "-n", args.conda_env, "python"], f"conda run -n {args.conda_env}"
    if Path(args.python_exe) == Path(sys.executable):
        return [args.python_exe], f"sys.executable ({args.python_exe})"
    return [args.python_exe], f"--python-exe ({args.python_exe})"


def planned_commands(args: argparse.Namespace) -> list[list[str]]:
    prefix, _mode = interpreter_prefix(args)
    out_dir = args.out_dir / args.dataset / f"ep{args.epochs}_train{args.train_transitions}"
    return [
        [
            *prefix,
            str(PREFLIGHT_SCRIPT),
            "--dataset",
            args.dataset,
            "--out-dir",
            str(out_dir),
            "--epochs",
            str(args.epochs),
            "--seeds",
            args.seeds,
            "--train-transitions",
            str(args.train_transitions),
            "--eval-transitions",
            str(args.eval_transitions),
            "--horizons",
            *[str(value) for value in args.horizons],
            "--prior-weight",
            str(args.prior_weight),
            "--device",
            args.device,
        ]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run wrapper for stronger no-prior preflight calibration baselines."
    )
    parser.add_argument("--dataset", default="nbody_distance")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--train-transitions", type=int, default=192)
    parser.add_argument("--eval-transitions", type=int, default=32)
    parser.add_argument("--horizons", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the planned command. This still uses the existing preflight priors and is not a full SOTA benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    _prefix, mode = interpreter_prefix(args)
    print(f"Interpreter mode: {mode}")
    print(
        "Scope: stronger no-prior calibration via existing preflight runner. "
        "This does not implement DCRNN, Graph WaveNet, SchNet, EGNN, or a leaderboard benchmark."
    )
    print(
        "TODO before using as a manuscript result: expose hidden_dim/depth capacity knobs "
        "or document that this is a budget/data-strength baseline only."
    )
    commands = planned_commands(args)
    for command in commands:
        print(shell_quote(command))
    if not args.execute:
        print("Dry run only. Pass --execute intentionally to launch the existing preflight command.")
        return
    for command in commands:
        print(f"Running: {shell_quote(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
