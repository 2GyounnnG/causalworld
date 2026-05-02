"""Dry-run friendly node-order sanity package for HO lattice graph controls."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "analysis_out" / "preflight_runs" / "node_order_sanity"
CONDITIONS = {
    "A_baseline": (False, False),
    "B_permuted_both": (True, True),
    "C_permuted_data_only": (True, False),
}


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def shell_quote(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def strip_option(argv: list[str], option: str) -> list[str]:
    out: list[str] = []
    skip_next = False
    for idx, value in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if value == option:
            skip_next = idx + 1 < len(argv)
            continue
        if value.startswith(f"{option}="):
            continue
        out.append(value)
    return out


def maybe_reexec(args: argparse.Namespace) -> bool:
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if not args.conda_env:
        print(f"Interpreter mode: {sys.executable}")
        return False
    print(f"Interpreter mode: conda run -n {args.conda_env}")
    if os.environ.get("CONDA_DEFAULT_ENV") == args.conda_env:
        return False
    forwarded = strip_option(sys.argv[1:], "--conda-env")
    command = ["conda", "run", "-n", args.conda_env, "python", str(Path(__file__).resolve()), *forwarded]
    prefix = f"CUDA_VISIBLE_DEVICES={args.gpu_id} " if args.gpu_id is not None else ""
    print(f"Re-exec command: {prefix}{shell_quote(command)}")
    if args.dry_run:
        print("Dry run only. Target environment will not be launched; continuing locally to print the plan.")
        return False
    subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())
    return True


def finite(value: object) -> bool:
    try:
        import math

        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def completed(summary_path: Path, seeds: list[int]) -> tuple[bool, str]:
    if not summary_path.exists():
        return False, "missing summary.csv"
    with summary_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    found = set()
    for row in rows:
        if row.get("stage") == "node_order_sanity" and row.get("status") == "ok" and finite(row.get("H32")):
            found.add((str(row.get("condition")), int(row.get("seed", -1))))
    missing = [
        f"{condition}:seed{seed}"
        for condition in CONDITIONS
        for seed in seeds
        if (condition, int(seed)) not in found
    ]
    if missing:
        return False, "missing " + ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
    return True, "complete"


def permutation_for(seed: int, base_seed: int, n_nodes: int) -> np.ndarray:
    return np.random.default_rng(int(base_seed) + int(seed) * 100000).permutation(n_nodes).astype(np.int64)


def save_permutation(out_root: Path, condition: str, seed: int, permutation: np.ndarray) -> Path:
    path = out_root / "artifacts" / "permutations" / f"{condition}_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "condition": condition,
        "seed": int(seed),
        "permutation": permutation.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run node-order sanity checks without changing model/prior definitions.")
    parser.add_argument("--dataset", default="ho_lattice", choices=["ho_lattice", "spring_mass_lattice", "nbody_distance"])
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--seeds", type=parse_csv_ints, default=[0, 1, 2])
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-transitions", type=int, default=96)
    parser.add_argument("--eval-transitions", type=int, default=32)
    parser.add_argument("--train-stride", type=int, default=5)
    parser.add_argument("--eval-stride", type=int, default=10)
    parser.add_argument("--horizons", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transition-type", default="mlp_pooled", choices=["mlp_pooled", "graph_node_sage"])
    parser.add_argument("--transition-gnn-layers", type=int, default=1)
    parser.add_argument("--decode-loss-weight", type=float, default=1.0)
    parser.add_argument("--node-feature-dim", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "ho_raw")
    parser.add_argument("--permutation-seed", type=int, default=9103)
    parser.add_argument("--spring-n", type=int, default=36)
    parser.add_argument("--spring-position-dim", type=int, default=1)
    parser.add_argument("--spring-t", type=int, default=512)
    parser.add_argument("--spring-k", type=float, default=1.0)
    parser.add_argument("--spring-damping", type=float, default=0.05)
    parser.add_argument("--spring-dt", type=float, default=0.05)
    parser.add_argument("--spring-noise", type=float, default=0.001)
    parser.add_argument("--spring-seed", type=int, default=0)
    parser.add_argument("--nbody-n", type=int, default=36)
    parser.add_argument("--nbody-position-dim", type=int, default=2)
    parser.add_argument("--nbody-t", type=int, default=512)
    parser.add_argument("--nbody-g", type=float, default=0.1)
    parser.add_argument("--nbody-dt", type=float, default=0.02)
    parser.add_argument("--nbody-softening", type=float, default=0.1)
    parser.add_argument("--nbody-damping", type=float, default=0.0)
    parser.add_argument("--nbody-noise", type=float, default=0.0005)
    parser.add_argument("--nbody-graph-k", type=int, default=8)
    parser.add_argument("--nbody-seed", type=int, default=0)
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_suffix:
        args.out_root = args.out_root.parent / f"{args.out_root.name}{args.output_suffix}"
    args.out_root = args.out_root.resolve()
    if maybe_reexec(args):
        return

    summary_path = args.out_root / "summary.csv"
    complete, reason = completed(summary_path, args.seeds)
    planned = [(condition, int(seed)) for condition in CONDITIONS for seed in args.seeds]
    for condition, seed in planned:
        print(f"PLAN condition={condition} seed={seed} output={args.out_root}")
        print(f"PLAN permutation: {args.out_root / 'artifacts' / 'permutations' / f'{condition}_seed{seed}.json'}")
    if args.dry_run:
        if complete and not args.force:
            print(f"SKIP complete {args.out_root}: {reason}")
            pending = 0
        else:
            pending = len(planned)
        print(f"Dry run only. Planned node-order training(s): {pending}")
        return
    if complete and not args.force:
        print(f"SKIP complete {args.out_root}: {reason}")
        return

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.graph_prior_preflight_check import build_adapter, config_hash, get_git_commit, mini_train, write_summary_csv
    from scripts.run_node_order_sanity_check import NodeOrderAdapter, comparison_rows, load_existing_rows

    args.topology = "lattice"
    args.include_temporal_prior = False
    args.calibrate_prior_strength = False
    args.calibration_reference_prior = "graph"
    args.calibration_target_ratio = 1.0
    args.out_root.mkdir(parents=True, exist_ok=True)
    base = build_adapter(args)
    rows = [row for row in load_existing_rows(summary_path) if row.get("stage") != "node_order_comparison"]

    for seed in args.seeds:
        permutation = permutation_for(int(seed), args.permutation_seed, int(base.n_nodes))
        for condition, (permute_features, permute_graph) in CONDITIONS.items():
            save_permutation(args.out_root, condition, int(seed), permutation)
            adapter = NodeOrderAdapter(
                base,
                condition=condition,
                permutation=permutation if (permute_features or permute_graph) else None,
                permute_features=permute_features,
                permute_graph=permute_graph,
            )
            train_args = copy.copy(args)
            config = adapter.training_config(train_args, prior="graph", seed=int(seed))
            row, _model, _device = mini_train(config, adapter, train_args)
            row.update(
                {
                    "schema_version": "node_order_sanity_v2",
                    "stage": "node_order_sanity",
                    "condition": condition,
                    "dataset": base.name,
                    "permutation_seed": int(args.permutation_seed) + int(seed) * 100000,
                    "config_hash": config_hash(config),
                    "git_commit": get_git_commit(),
                    **{f"metadata_{key}": value for key, value in adapter.metadata().items()},
                }
            )
            rows = [
                existing
                for existing in rows
                if not (
                    existing.get("stage") == "node_order_sanity"
                    and existing.get("condition") == condition
                    and int(existing.get("seed", -1)) == int(seed)
                )
            ]
            rows.append(row)
            write_summary_csv(summary_path, [*rows, *comparison_rows(rows, args.horizons)])
    write_summary_csv(summary_path, [*rows, *comparison_rows(rows, args.horizons)])
    print(f"Wrote {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
