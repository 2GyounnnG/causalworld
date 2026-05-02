"""Launcher for split-host HO lattice 5-seed latent audit runs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "analysis_out" / "preflight_runs" / "ho_audit_5seed"
CYCLE8_SCRIPT = ROOT / "scripts" / "run_cycle8_checkpointed_lattice_latent_alignment.py"
AGGREGATE_SCRIPT = ROOT / "scripts" / "aggregate_ho_audit_5seed.py"
PRIORS = ("graph", "permuted_graph", "random_graph")


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def interpreter_command(args: argparse.Namespace) -> tuple[list[str], str]:
    if args.conda_env:
        return ["conda", "run", "-n", args.conda_env, "python"], f"conda run -n {args.conda_env}"
    return [args.python_exe], args.python_exe


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


def config_for(prior: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_name": f"lattice_gnn_{prior}_lambda0p1_seed{seed}",
        "topology": "lattice",
        "encoder": "gnn_node",
        "prior": prior,
        "seed": int(seed),
        "num_epochs": int(args.epochs),
        "eval_horizons": [1, 2, 4, 8, 16, 32],
        "latent_dim": 16,
        "node_dim": 16,
        "batch_size": int(args.batch_size),
        "prior_weight": float(args.prior_weight),
    }


def config_path(args: argparse.Namespace, prior: str, seed: int) -> Path:
    return args.out_root / "configs" / f"lattice_gnn_{prior}_lambda0p1_seed{seed}.json"


def expected_config_paths(args: argparse.Namespace) -> list[Path]:
    return [config_path(args, prior, seed) for seed in args.seeds for prior in PRIORS]


def write_configs(args: argparse.Namespace) -> None:
    for seed in args.seeds:
        for prior in PRIORS:
            path = config_path(args, prior, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(config_for(prior, seed, args), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def result_path(args: argparse.Namespace) -> Path:
    return args.out_root / "cycle8_ho_audit_results.json"


def run_complete(args: argparse.Namespace) -> tuple[bool, str]:
    path = result_path(args)
    if not path.exists():
        return False, "missing result json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"unreadable result json: {exc}"
    runs = payload.get("runs", {})
    missing = []
    for seed in args.seeds:
        for prior in PRIORS:
            name = f"lattice_gnn_{prior}_lambda0p1_seed{seed}"
            run = runs.get(name)
            if not run or run.get("status") != "ok" or run.get("failure_flag"):
                missing.append(name)
    if missing:
        return False, "missing/failed " + ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
    return True, "complete"


def command_for_run(args: argparse.Namespace) -> list[str]:
    python_cmd, _source = interpreter_command(args)
    command = [
        *python_cmd,
        str(CYCLE8_SCRIPT),
        "--output",
        str(result_path(args)),
        "--checkpoint-dir",
        str(args.out_root / "checkpoints"),
        "--artifact-dir",
        str(args.out_root / "artifacts"),
    ]
    for path in expected_config_paths(args):
        command.extend(["--config", str(path)])
    if args.force:
        command.append("--force")
    return command


def command_for_aggregate(args: argparse.Namespace) -> list[str]:
    python_cmd, _source = interpreter_command(args)
    return [
        *python_cmd,
        str(AGGREGATE_SCRIPT),
        "--out-root",
        str(args.out_root),
        "--seeds",
        ",".join(str(seed) for seed in args.seeds),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated HO lattice 5-seed latent audit configs.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--seeds", type=parse_csv_ints, default=[0, 1, 2, 3, 4], help="Comma-separated seed subset.")
    parser.add_argument("--output-suffix", default="", help="Append suffix to output root directory name, e.g. _m1.")
    parser.add_argument("--gpu-id", default=None, help="Set CUDA_VISIBLE_DEVICES for the child audit command.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--conda-env", default=None, help="Run child commands as `conda run -n ENV python ...`.")
    parser.add_argument("--force", action="store_true", help="Re-run even when the selected seed subset appears complete.")
    parser.add_argument("--dry-run", action="store_true", help="Print generated configs and command without executing.")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_suffix:
        args.out_root = args.out_root.parent / f"{args.out_root.name}{args.output_suffix}"
    args.out_root = args.out_root.resolve()
    _python_cmd, interpreter_source = interpreter_command(args)
    print(f"Interpreter mode: {interpreter_source}")

    complete, reason = run_complete(args)
    command = command_for_run(args)
    should_run = args.force or not complete

    if args.dry_run:
        for path in expected_config_paths(args):
            print(f"DRY RUN config: WOULD WRITE {path}")
        if should_run:
            print(f"DRY RUN command: {display_command(args, command)}")
        else:
            print(f"SKIP complete {args.out_root}: {reason}")
        if not args.skip_aggregate:
            print(f"DRY RUN summary command: {shell_quote(command_for_aggregate(args))}")
        print(f"Dry run only. Pending run command(s): {1 if should_run else 0}")
        return

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "out_root": str(args.out_root),
        "seeds": args.seeds,
        "priors": list(PRIORS),
        "complete": complete,
        "completion_reason": reason,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if should_run:
        write_configs(args)
        print(f"RUN command: {display_command(args, command)}", flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True, env=child_env(args))
        except subprocess.CalledProcessError as exc:
            print(f"RUN FAILED returncode={exc.returncode}: {display_command(args, command)}", flush=True)
            if args.fail_fast:
                raise
    else:
        print(f"SKIP complete {args.out_root}: {reason}", flush=True)

    if not args.skip_aggregate:
        aggregate_command = command_for_aggregate(args)
        print(f"SUMMARY command: {shell_quote(aggregate_command)}", flush=True)
        subprocess.run(aggregate_command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
