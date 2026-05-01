from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUT_DIR = ROOT / "analysis_out" / "preflight_runs" / "node_order_sanity"


def shell_quote(parts: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in parts)


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


def maybe_reexec_with_requested_interpreter(args: argparse.Namespace) -> bool:
    print(f"Current sys.executable: {sys.executable}", flush=True)
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if args.conda_env:
        print(f"Interpreter mode: conda run -n {args.conda_env}", flush=True)
        if current_env != args.conda_env:
            forwarded = strip_option(strip_option(sys.argv[1:], "--conda-env"), "--python-exe")
            command = ["conda", "run", "-n", args.conda_env, "python", str(Path(__file__).resolve()), *forwarded]
            print(f"Re-exec command: {shell_quote(command)}", flush=True)
            if args.dry_run:
                print("Dry run only. Target environment was not launched.", flush=True)
                return True
            subprocess.run(command, cwd=ROOT, check=True)
            return True
        return False

    if Path(args.python_exe) != Path(sys.executable):
        print(f"Interpreter mode: --python-exe ({args.python_exe})", flush=True)
        forwarded = strip_option(sys.argv[1:], "--python-exe")
        command = [args.python_exe, str(Path(__file__).resolve()), *forwarded]
        print(f"Re-exec command: {shell_quote(command)}", flush=True)
        if args.dry_run:
            print("Dry run only. Target interpreter was not launched.", flush=True)
            return True
        subprocess.run(command, cwd=ROOT, check=True)
        return True

    print(f"Interpreter mode: sys.executable ({sys.executable})", flush=True)
    return False


def require_torch() -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "ERROR: torch is not importable from this Python interpreter.\n"
            f"Current sys.executable: {sys.executable}\n"
            "Run from the causalworld environment, for example:\n"
            "  conda run -n causalworld python scripts/run_node_order_sanity_check.py\n"
            "or activate the environment first:\n"
            "  conda activate causalworld\n"
            "  python scripts/run_node_order_sanity_check.py"
        ) from exc
    print(f"torch import ok: version {torch.__version__}", flush=True)


def mean(values: list[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(vals)) if vals else float("nan")


def parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def permuted_copy(data: Any, *, feature_perm: np.ndarray | None, graph_perm: np.ndarray | None) -> Any:
    import torch

    out = data.clone()
    if feature_perm is not None:
        node_index = torch.as_tensor(feature_perm, dtype=torch.long)
        for attr in ("pos", "atomic_number", "x"):
            if hasattr(out["atom"], attr):
                setattr(out["atom"], attr, getattr(out["atom"], attr)[node_index])
    if graph_perm is not None:
        inverse = np.empty_like(graph_perm)
        inverse[graph_perm] = np.arange(graph_perm.shape[0])
        inverse_t = torch.as_tensor(inverse, dtype=torch.long)
        edge_index = out["atom", "bonded", "atom"].edge_index
        out["atom", "bonded", "atom"].edge_index = inverse_t[edge_index]
    return out


class NodeOrderAdapter:
    def __init__(
        self,
        base: Any,
        *,
        condition: str,
        permutation: np.ndarray | None,
        permute_features: bool,
        permute_graph: bool,
    ):
        self.base = base
        self.condition = condition
        self.permutation = permutation
        self.feature_perm = permutation if permute_features else None
        self.graph_perm = permutation if permute_graph else None
        self.name = f"{base.name}_{condition}"
        self.n_nodes = int(base.n_nodes)
        self.n_frames = int(base.n_frames)
        self.topology = getattr(base, "topology", "")

    def __getitem__(self, idx: int) -> Any:
        return permuted_copy(self.base[idx], feature_perm=self.feature_perm, graph_perm=self.graph_perm)

    def get_pair(self, idx: int, horizon: int = 1) -> tuple[Any, Any, float, None]:
        if idx + horizon >= self.n_frames:
            raise IndexError(f"idx+horizon={idx + horizon} >= n_frames={self.n_frames}")
        obs, next_obs, energy, force = self.base.get_pair(idx, horizon=horizon)
        return (
            permuted_copy(obs, feature_perm=self.feature_perm, graph_perm=self.graph_perm),
            permuted_copy(next_obs, feature_perm=self.feature_perm, graph_perm=self.graph_perm),
            float(energy),
            force,
        )

    def sample_transitions(
        self,
        *,
        n_transitions: int,
        stride: int,
        horizon: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        transitions = self.base.sample_transitions(
            n_transitions=n_transitions,
            stride=stride,
            horizon=horizon,
            seed=seed,
        )
        out: list[dict[str, Any]] = []
        for transition in transitions:
            copied = dict(transition)
            copied["obs"] = permuted_copy(
                transition["obs"],
                feature_perm=self.feature_perm,
                graph_perm=self.graph_perm,
            )
            copied["next_obs"] = permuted_copy(
                transition["next_obs"],
                feature_perm=self.feature_perm,
                graph_perm=self.graph_perm,
            )
            copied["molecule"] = self.name
            copied["node_order_condition"] = self.condition
            out.append(copied)
        return out

    def true_laplacian(self) -> np.ndarray:
        laplacian = np.asarray(self.base.true_laplacian(), dtype=np.float64)
        if self.graph_perm is None:
            return laplacian
        return laplacian[np.ix_(self.graph_perm, self.graph_perm)]

    def coordinates(self, frame_idx: int) -> np.ndarray:
        coords = np.asarray(self.base.coordinates(frame_idx), dtype=np.float64)
        if self.feature_perm is None:
            return coords
        return coords[self.feature_perm]

    def training_config(self, args: argparse.Namespace, *, prior: str, seed: int) -> Any:
        config = self.base.training_config(args, prior=prior, seed=seed)
        config.run_name = (
            f"node_order_{self.base.name}_{self.condition}_gnn_{prior}_"
            f"lambda{str(args.prior_weight).replace('.', 'p')}_seed{seed}"
        )
        return config

    def metadata(self) -> dict[str, Any]:
        return {
            **self.base.metadata(),
            "node_order_condition": self.condition,
            "feature_permuted": self.feature_perm is not None,
            "graph_permuted": self.graph_perm is not None,
            "permutation": self.permutation.tolist() if self.permutation is not None else "",
        }


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def completed_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for row in rows:
        if row.get("stage") == "node_order_sanity" and row.get("status") == "ok":
            out.add((str(row.get("condition")), int(row.get("seed", -1))))
    return out


def comparison_rows(rows: list[dict[str, Any]], horizons: list[int]) -> list[dict[str, Any]]:
    base_rows = [row for row in rows if row.get("stage") == "node_order_sanity" and row.get("status") == "ok"]
    by_key = {(str(row["condition"]), int(row["seed"])): row for row in base_rows}
    conditions = sorted({str(row["condition"]) for row in base_rows if str(row["condition"]) != "original"})
    comparisons: list[dict[str, Any]] = []
    for condition in conditions:
        paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for row in base_rows:
            seed = int(row["seed"])
            if row["condition"] == "original" and (condition, seed) in by_key:
                paired.append((row, by_key[(condition, seed)]))
        comparison: dict[str, Any] = {
            "schema_version": "node_order_sanity_v1",
            "stage": "node_order_comparison",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "condition": condition,
            "baseline_condition": "original",
            "n_paired_seeds": len(paired),
        }
        for horizon in horizons:
            deltas = []
            for original, other in paired:
                try:
                    deltas.append(float(other[f"H{horizon}"]) - float(original[f"H{horizon}"]))
                except (KeyError, TypeError, ValueError):
                    pass
            comparison[f"delta_H{horizon}_condition_minus_original_mean"] = mean(deltas)
            comparison[f"abs_delta_H{horizon}_mean"] = mean([abs(value) for value in deltas])
        comparisons.append(comparison)
    return comparisons


def write_report(path: Path, rows: list[dict[str, Any]], horizons: list[int]) -> None:
    sanity_rows = [row for row in rows if row.get("stage") == "node_order_sanity" and row.get("status") == "ok"]
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in sanity_rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)

    lines = [
        "# Node-Order Sanity Check",
        "",
        f"Created: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "Purpose: test whether the spectrum-matched permuted-graph control is confounded by node-order sensitivity.",
        "The consistent condition permutes node features and graph labels together; a large rollout shift versus original would indicate order sensitivity in the encoder/training path.",
        "The mismatched condition keeps original features with a permuted graph and is an optional unfair-control stress test.",
        "",
        "| condition | seeds | " + " | ".join(f"H={h}" for h in horizons) + " |",
        "| --- | --- | " + " | ".join("---" for _ in horizons) + " |",
    ]
    for condition, condition_rows in sorted(by_condition.items()):
        cells = [condition, str(len(condition_rows))]
        for horizon in horizons:
            values = []
            for row in condition_rows:
                try:
                    values.append(float(row[f"H{horizon}"]))
                except (KeyError, TypeError, ValueError):
                    pass
            cells.append(f"{mean(values):.6g}" if values and math.isfinite(mean(values)) else "NA")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Interpretation:",
            "- quick mode is triage only; it asks whether a prior deserves more attention.",
            "- standard mode is a persistence check under a larger training budget.",
            "- audit mode is a mechanism check on learned latent smoothing.",
            "- a quick candidate topology signal is not final topology-specific evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_run_config(path: Path, args: argparse.Namespace, permutation: np.ndarray) -> None:
    payload = {
        "schema_version": "node_order_sanity_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "permutation": permutation.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Node-order sanity check for graph-prior preflight controls.")
    parser.add_argument("--dataset", default="nbody_distance")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", type=parse_seeds, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-transitions", type=int, default=96)
    parser.add_argument("--eval-transitions", type=int, default=32)
    parser.add_argument("--train-stride", type=int, default=5)
    parser.add_argument("--eval-stride", type=int, default=10)
    parser.add_argument("--horizons", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument(
        "--conda-env",
        default=None,
        help="Re-run this script as `conda run -n ENV python ...`; overrides --python-exe.",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "ho_raw")
    parser.add_argument("--nbody-n", type=int, default=36)
    parser.add_argument("--nbody-graph-k", type=int, default=8)
    parser.add_argument("--nbody-seed", type=int, default=0)
    parser.add_argument("--permutation-seed", type=int, default=9103)
    parser.add_argument("--include-mismatched", action="store_true", default=True)
    parser.add_argument("--no-mismatched", dest="include_mismatched", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if maybe_reexec_with_requested_interpreter(args):
        return
    require_torch()
    args.out_dir = args.out_dir.resolve()
    args.summary = args.out_dir / "summary.csv"
    args.report = args.out_dir / "node_order_sanity_report.md"
    args.artifact_dir = args.out_dir / "artifacts"
    args.topology = "distance_knn" if args.dataset == "nbody_distance" else None
    args.include_temporal_prior = False
    args.calibrate_prior_strength = False
    args.calibration_reference_prior = "graph"
    args.calibration_target_ratio = 1.0

    condition_names = ["original", "consistent_permuted_nodes"]
    if args.include_mismatched:
        condition_names.append("mismatched_original_x_permuted_l")
    planned = [(condition, int(seed)) for condition in condition_names for seed in args.seeds]
    for condition, seed in planned:
        print(f"{condition} seed={seed}")
    if args.dry_run:
        print(f"Dry run only. Planned {len(planned)} node-order training(s).")
        return

    from scripts.graph_prior_preflight_check import (  # noqa: PLC0415
        NBodyDistanceAdapter,
        build_adapter,
        config_hash,
        get_git_commit,
        mini_train,
        write_summary_csv,
    )

    if args.dataset != "nbody_distance":
        base = build_adapter(args)
    else:
        base = NBodyDistanceAdapter(
            n_nodes=args.nbody_n,
            graph_k=args.nbody_graph_k,
            seed=args.nbody_seed,
        )

    rng = np.random.default_rng(args.permutation_seed)
    permutation = rng.permutation(base.n_nodes).astype(np.int64)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(args.out_dir / "run_config.json", args, permutation)

    conditions = [
        NodeOrderAdapter(base, condition="original", permutation=None, permute_features=False, permute_graph=False),
        NodeOrderAdapter(
            base,
            condition="consistent_permuted_nodes",
            permutation=permutation,
            permute_features=True,
            permute_graph=True,
        ),
    ]
    if args.include_mismatched:
        conditions.append(
            NodeOrderAdapter(
                base,
                condition="mismatched_original_x_permuted_l",
                permutation=permutation,
                permute_features=False,
                permute_graph=True,
            )
        )

    rows = [
        row for row in load_existing_rows(args.summary)
        if row.get("stage") != "node_order_comparison"
    ]
    done = completed_keys(rows)
    for condition in conditions:
        for seed in args.seeds:
            key = (condition.condition, int(seed))
            if key in done and not args.force:
                continue
            train_args = copy.copy(args)
            config = condition.training_config(train_args, prior="graph", seed=int(seed))
            row, _model, _device = mini_train(config, condition, train_args)
            row.update(
                {
                    "schema_version": "node_order_sanity_v1",
                    "stage": "node_order_sanity",
                    "condition": condition.condition,
                    "dataset": base.name,
                    "permutation_seed": args.permutation_seed,
                    "config_hash": config_hash(config),
                    "git_commit": get_git_commit(),
                    **{f"metadata_{key}": value for key, value in condition.metadata().items()},
                }
            )
            rows = [
                existing for existing in rows
                if not (
                    existing.get("stage") == "node_order_sanity"
                    and existing.get("condition") == condition.condition
                    and int(existing.get("seed", -1)) == int(seed)
                )
            ]
            rows.append(row)
            write_summary_csv(args.summary, [*rows, *comparison_rows(rows, args.horizons)])

    final_rows = [*rows, *comparison_rows(rows, args.horizons)]
    write_summary_csv(args.summary, final_rows)
    write_report(args.report, final_rows, args.horizons)
    print(f"Wrote {args.report.relative_to(ROOT)}")
    print(f"Wrote {args.summary.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
