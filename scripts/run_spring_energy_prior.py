from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "analysis_out" / "preflight_runs" / "spring_energy_prior"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TODO stub for a spring-mass energy-drift prior calibration baseline."
    )
    parser.add_argument("--dataset", default="spring_mass_lattice")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--dry-run", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = f"conda run -n {args.conda_env}" if args.conda_env else f"sys.executable ({args.python_exe})"
    print(f"Interpreter mode: {mode}")
    print("Dry-run TODO stub: no spring energy-prior training is implemented or launched.")
    print("Feasible local scope:")
    print("- reuse the synthetic spring_mass adapter, which already computes per-frame energy")
    print("- define an auxiliary energy prediction head or decoded-state target before applying an energy-drift loss")
    print("- compare against graph and temporal_smooth as a mechanism check, not as SOTA")
    print("- avoid full HNN claims unless the objective is redesigned around state derivatives or Hamiltonian structure")
    print("Planned output root:", args.out_dir.resolve())
    print("Smoke command:")
    print("  python scripts/run_spring_energy_prior.py --dry-run")
    if not args.dry_run:
        raise SystemExit("Spring energy prior is a TODO stub; no executable training path exists yet.")


if __name__ == "__main__":
    main()
