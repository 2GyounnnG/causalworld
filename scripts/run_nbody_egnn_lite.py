from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "analysis_out" / "preflight_runs" / "nbody_egnn_lite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TODO stub for an EGNN-lite N-body calibration baseline."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--nbody-n", type=int, default=36)
    parser.add_argument("--horizons", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--dry-run", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = f"conda run -n {args.conda_env}" if args.conda_env else f"sys.executable ({args.python_exe})"
    print(f"Interpreter mode: {mode}")
    print("Dry-run TODO stub: no EGNN-lite training is implemented or launched.")
    print("Feasible local scope:")
    print("- reuse the synthetic nbody_distance adapter for positions, velocities, and masses")
    print("- implement small all-pairs or radius-limited equivariant message passing")
    print("- report coordinate/velocity rollout at H=16,H=32 separately from latent preflight rollout")
    print("- frame as N-body calibration only, not molecular SOTA")
    print("Planned output root:", args.out_dir.resolve())
    print("Smoke command:")
    print("  python scripts/run_nbody_egnn_lite.py --dry-run")
    if not args.dry_run:
        raise SystemExit("EGNN-lite is a TODO stub; no executable training path exists yet.")


if __name__ == "__main__":
    main()
