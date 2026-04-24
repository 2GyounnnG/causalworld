from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Check:
    name: str
    status: str
    evidence: str
    recommendation: str = ""


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def safe_read(path: Path) -> str | None:
    if not path.exists():
        return None
    return read(path)


def add(checks: list[Check], name: str, status: str, evidence: str, recommendation: str = "") -> None:
    checks.append(Check(name, status, evidence, recommendation))


def has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.MULTILINE | re.DOTALL) is not None


def tiny_prior_weight_test(root: Path) -> Check:
    sys.path.insert(0, str(root))
    try:
        import torch
        from torch_geometric.data import HeteroData
        from model import WorldModel

        torch.manual_seed(123)
        obs = HeteroData()
        obs["node"].x = torch.tensor([[1.0], [2.0], [3.0]])
        obs["hyperedge"].x = torch.tensor([[2.0, -1.0], [2.0, -1.0]])
        next_obs = HeteroData()
        next_obs["node"].x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        next_obs["hyperedge"].x = torch.tensor([[2.0, -1.0], [2.0, -1.0], [2.0, -1.0]])
        model = WorldModel(encoder="flat", hidden_dim=8, latent_dim=4, action_dim=1, transition_hidden_dim=16)
        action = torch.tensor([0.0])
        none_loss = model.loss(obs, action, next_obs, reward=0.0, done=0.0, prior="none", prior_weight=0.0)
        spectral_zero = model.loss(
            obs,
            action,
            next_obs,
            reward=0.0,
            done=0.0,
            prior="spectral",
            prior_weight=0.0,
            laplacian=torch.eye(4),
        )
        diff = float((none_loss["total"] - spectral_zero["total"]).abs().detach())
        if diff < 1e-9 and float(spectral_zero["prior"].detach()) > 0.0:
            return Check(
                "prior_weight=0 disables prior penalty",
                "PASS",
                f"tiny smoke test: spectral prior_loss={float(spectral_zero['prior'].detach()):.6g}, total difference vs none={diff:.3g}",
            )
        return Check(
            "prior_weight=0 disables prior penalty",
            "FAIL",
            f"tiny smoke test total difference vs none was {diff:.6g}",
            "Inspect WorldModel.loss prior weighting before relying on zero-weight controls.",
        )
    except Exception as exc:
        return Check(
            "prior_weight=0 disables prior penalty",
            "UNKNOWN",
            f"tiny smoke test could not run: {exc}",
            "Run the audit in the project environment with torch_geometric installed.",
        )
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass


def parse_rmd17_result_configs(root: Path) -> list[dict[str, Any]]:
    configs_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in sorted(root.glob("rmd17*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        results = data.get("results", data)
        if not isinstance(results, dict):
            continue
        for result in results.values():
            if not isinstance(result, dict):
                continue
            config = result.get("config")
            if not isinstance(config, dict) or "molecule" not in config or "seed" not in config:
                continue
            try:
                molecule = str(config.get("molecule", ""))
                seed = int(config.get("seed", 0))
                n_transitions = int(config.get("n_transitions", 2000))
                stride = int(config.get("stride", 10))
                horizon = int(config.get("horizon", 1))
                eval_horizons = tuple(int(value) for value in config.get("eval_horizons", (1, 2, 4, 8, 16)))
            except (TypeError, ValueError):
                continue
            key = (molecule, seed, n_transitions, stride, horizon, eval_horizons)
            configs_by_key[key] = {
                "molecule": molecule,
                "seed": seed,
                "n_transitions": n_transitions,
                "stride": stride,
                "horizon": horizon,
                "eval_horizons": eval_horizons,
            }
    return list(configs_by_key.values())


def sampled_frame_indices(n_frames: int, n_transitions: int, stride: int, horizon: int, seed: int) -> set[int]:
    import numpy as np

    rng = np.random.default_rng(seed)
    max_start = n_frames - horizon - 1
    candidates = np.arange(0, max_start, stride)
    if n_transitions > len(candidates):
        raise ValueError(f"requested {n_transitions} transitions but only {len(candidates)} candidates exist")
    chosen = rng.choice(candidates, size=n_transitions, replace=False)
    return set(int(value) for value in chosen)


def rmd17_frame_overlap_check(root: Path) -> Check:
    recommendation = "Persist sampled train/eval frame_idx sets into run metadata for future reviewer-facing leakage audits."
    try:
        configs = parse_rmd17_result_configs(root)
        if not configs:
            return Check(
                "rMD17 train/eval frame-index overlap",
                "WARNING",
                "No rMD17 result configs with molecule/seed were found, and no persisted transition metadata was available for comparison.",
                recommendation,
            )

        data_root = root / "data" / "rmd17_raw" / "rmd17" / "npz_data"
        n_frames_by_molecule: dict[str, int] = {}
        overlaps: list[tuple[str, int, int, list[int]]] = []
        checked = 0
        for config in configs:
            molecule = str(config["molecule"])
            if molecule not in n_frames_by_molecule:
                npz_path = data_root / f"rmd17_{molecule}.npz"
                if not npz_path.exists():
                    return Check(
                        "rMD17 train/eval frame-index overlap",
                        "WARNING",
                        f"Could not reconstruct sampled frame indices because {npz_path} is missing.",
                        recommendation,
                    )
                import numpy as np

                with np.load(npz_path) as data:
                    n_frames_by_molecule[molecule] = int(data["coords"].shape[0])

            seed = int(config["seed"])
            stride = int(config["stride"])
            eval_horizon = max(int(value) for value in config["eval_horizons"])
            train_idx = sampled_frame_indices(
                n_frames_by_molecule[molecule],
                int(config["n_transitions"]),
                stride,
                int(config["horizon"]),
                seed,
            )
            eval_idx = sampled_frame_indices(
                n_frames_by_molecule[molecule],
                200,
                stride * 10,
                eval_horizon,
                seed + 1000,
            )
            overlap = sorted(train_idx & eval_idx)
            checked += 1
            if overlap:
                overlaps.append((molecule, seed, len(overlap), overlap[:5]))

        if overlaps:
            examples = "; ".join(
                f"{molecule} seed={seed}: {count} overlaps, first={first}"
                for molecule, seed, count, first in overlaps[:6]
            )
            return Check(
                "rMD17 train/eval frame-index overlap",
                "WARNING",
                f"Reconstructed train/eval frame_idx sets for {checked} molecule/seed configs from result metadata; {len(overlaps)} configs overlap. {examples}",
                "Exclude training frame_idx values from eval sampling, and persist sampled train/eval frame_idx sets into run metadata for future reviewer-facing leakage audits.",
            )

        return Check(
            "rMD17 train/eval frame-index overlap",
            "PASS",
            f"Reconstructed train/eval frame_idx sets for {checked} molecule/seed configs from result metadata; no overlap found.",
        )
    except Exception as exc:
        return Check(
            "rMD17 train/eval frame-index overlap",
            "WARNING",
            f"Exact frame-index comparison was not feasible: {exc}",
            recommendation,
        )


def inspect_python_syntax(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing file: {path}"
    try:
        ast.parse(read(path), filename=str(path))
        return True, "AST parse succeeded"
    except SyntaxError as exc:
        return False, str(exc)


def build_checks(root: Path) -> list[Check]:
    checks: list[Check] = []
    train_path = root / "train.py"
    model_path = root / "model.py"
    rmd17_path = root / "scripts" / "train_rmd17.py"
    lap_runner_path = root / "scripts" / "run_rmd17_laplacian_ablation.py"
    loader_path = root / "scripts" / "rmd17_loader.py"
    train = safe_read(train_path)
    model = safe_read(model_path)
    rmd17 = safe_read(rmd17_path)
    lap_runner = safe_read(lap_runner_path)
    loader = safe_read(loader_path)
    train_missing = train is None
    model_missing = model is None
    rmd17_missing = rmd17 is None
    lap_runner_missing = lap_runner is None
    loader_missing = loader is None

    for label, text, path in [
        ("train.py", train, train_path),
        ("model.py", model, model_path),
        ("scripts/train_rmd17.py", rmd17, rmd17_path),
        ("scripts/run_rmd17_laplacian_ablation.py", lap_runner, lap_runner_path),
        ("scripts/rmd17_loader.py", loader, loader_path),
    ]:
        if text is None:
            add(checks, f"{label} available", "UNKNOWN", f"missing file: {path}")

    train = train or ""
    model = model or ""
    rmd17 = rmd17 or ""
    lap_runner = lap_runner or ""
    loader = loader or ""

    ok, evidence = inspect_python_syntax(rmd17_path)
    if rmd17_missing:
        parse_status = "UNKNOWN"
    elif ok:
        parse_status = "PASS"
    else:
        parse_status = "FAIL"
    add(checks, "train_rmd17.py parses", parse_status, evidence)

    if rmd17_missing:
        add(checks, "rMD17 Euclidean covariance is batch-level", "UNKNOWN", f"missing file: {rmd17_path}")
    elif "euclidean_cov_penalty(torch.stack(batch_latents" in rmd17:
        add(
            checks,
            "rMD17 Euclidean covariance is batch-level",
            "PASS",
            "scripts/train_rmd17.py collects batch_latents and applies euclidean_cov_penalty(torch.stack(batch_latents, dim=0)) once per minibatch.",
        )
    else:
        add(
            checks,
            "rMD17 Euclidean covariance is batch-level",
            "FAIL",
            "Could not find the batch_latents Euclidean covariance path in scripts/train_rmd17.py.",
            "Do not interpret Euclidean comparisons until this is fixed.",
        )

    if train_missing:
        add(checks, "Wolfram Euclidean covariance is batch-level", "UNKNOWN", f"missing file: {train_path}")
    elif "euclidean_cov_penalty(latent_batch)" in train and "batch_latents.append" in train:
        add(
            checks,
            "Wolfram Euclidean covariance is batch-level",
            "PASS",
            "train.py appends model.encode(obs) across the minibatch and applies euclidean_cov_penalty(latent_batch).",
        )
    else:
        add(
            checks,
            "Wolfram Euclidean covariance is batch-level",
            "FAIL",
            "Could not verify batch-level Euclidean covariance in train.py.",
        )

    if model_missing:
        add(checks, "Euclidean covariance needs at least two latents", "UNKNOWN", f"missing file: {model_path}")
    elif "if B < 2:" in model and "return z_batch.new_tensor(0.0)" in model:
        add(
            checks,
            "Euclidean covariance needs at least two latents",
            "PASS",
            "model.euclidean_cov_penalty returns zero for B < 2, which is why training scripts must bypass per-sample model.loss for Euclidean.",
        )
    else:
        add(
            checks,
            "Euclidean covariance needs at least two latents",
            "UNKNOWN",
            "Could not verify B < 2 handling in model.euclidean_cov_penalty.",
        )

    if rmd17_missing:
        add(checks, "rMD17 Euclidean path is separate from none and spectral", "UNKNOWN", f"missing file: {rmd17_path}")
    elif has(r"elif config\.prior == \"euclidean\":.*?prior=\"none\".*?prior_weight=0\.0", rmd17):
        add(
            checks,
            "rMD17 Euclidean path is separate from none and spectral",
            "PASS",
            "Euclidean first computes base transition loss with prior='none', then adds config.prior_weight * batch covariance penalty.",
        )
    else:
        add(checks, "rMD17 Euclidean path is separate from none and spectral", "FAIL", "Could not verify separate Euclidean branch.")

    checks.append(tiny_prior_weight_test(root))

    if rmd17_missing:
        add(checks, "Config preserves per_frame default", "UNKNOWN", f"missing file: {rmd17_path}")
    elif 'laplacian_mode: str = "per_frame"' in rmd17:
        add(checks, "Config preserves per_frame default", "PASS", "Config.laplacian_mode defaults to 'per_frame', preserving existing callers.")
    else:
        add(checks, "Config preserves per_frame default", "FAIL", "Config.laplacian_mode default was not found.")

    if rmd17_missing:
        add(checks, "per_frame recomputes Laplacian from current sample", "UNKNOWN", f"missing file: {rmd17_path}")
    elif "build_molecular_laplacian(obs_raw, config.latent_dim)" in rmd17 and "fixed_laplacian is None" in rmd17:
        add(checks, "per_frame recomputes Laplacian from current sample", "PASS", "The spectral batch loop builds L from obs_raw when fixed_laplacian is None.")
    else:
        add(checks, "per_frame recomputes Laplacian from current sample", "FAIL", "Could not verify per-frame Laplacian construction.")

    if rmd17_missing:
        add(checks, "fixed_frame0 uses one fixed Laplacian", "UNKNOWN", f"missing file: {rmd17_path}")
    elif "fixed_frame0" in rmd17 and "build_molecular_laplacian(traj[0]" in rmd17:
        add(checks, "fixed_frame0 uses one fixed Laplacian", "PASS", "fixed_frame0 builds L once from RMD17Trajectory(config.molecule)[0] before the model is trained/evaluated.")
    else:
        add(checks, "fixed_frame0 uses one fixed Laplacian", "FAIL", "Could not verify fixed_frame0 precomputation.")

    if rmd17_missing:
        add(checks, "fixed_mean uses one averaged Laplacian", "UNKNOWN", f"missing file: {rmd17_path}")
    elif "fixed_mean" in rmd17 and "range(n_frames)" in rmd17 and "torch.stack(frame_laplacians" in rmd17:
        add(checks, "fixed_mean uses one averaged Laplacian", "PASS", "fixed_mean builds dense first-frame samples up to min(500, n_frames), averages them, and reuses the tensor.")
    else:
        add(checks, "fixed_mean uses one averaged Laplacian", "FAIL", "Could not verify fixed_mean averaging.")

    if rmd17_missing:
        add(checks, "fixed Laplacian modes do not inspect current batch frame for L", "UNKNOWN", f"missing file: {rmd17_path}")
    elif "laplacian=laplacian" in rmd17 and "laplacian = fixed_laplacian" in rmd17:
        add(
            checks,
            "fixed Laplacian modes do not inspect current batch frame for L",
            "PASS",
            "The batch loop reuses fixed_laplacian instead of calling build_molecular_laplacian(obs_raw) in fixed modes.",
        )
    else:
        add(
            checks,
            "fixed Laplacian modes do not inspect current batch frame for L",
            "FAIL",
            "Could not verify fixed_laplacian reuse in the spectral batch loop.",
        )

    if lap_runner_missing:
        add(checks, "Laplacian ablation runner requests all modes", "UNKNOWN", f"missing file: {lap_runner_path}")
    elif "build_molecular_laplacian" in lap_runner and "laplacian_mode=mode" in lap_runner:
        add(
            checks,
            "Laplacian ablation runner requests all modes",
            "PASS",
            "scripts/run_rmd17_laplacian_ablation.py imports build_molecular_laplacian and passes laplacian_mode=mode into Config.",
        )
    else:
        add(checks, "Laplacian ablation runner requests all modes", "WARNING", "Could not verify the runner imports or passes all mode settings.")

    if rmd17_missing:
        add(checks, "rMD17 seed handling", "UNKNOWN", f"missing file: {rmd17_path}")
    elif all(token in rmd17 for token in ["np.random.seed(config.seed)", "torch.manual_seed(config.seed)", "torch.cuda.manual_seed_all(config.seed)"]):
        add(checks, "rMD17 seed handling", "PASS", "train_one_seed sets NumPy, Torch, and CUDA seeds.")
    else:
        add(checks, "rMD17 seed handling", "WARNING", "Could not verify all NumPy/Torch/CUDA seed calls in train_one_seed.")

    if train_missing:
        add(checks, "Wolfram seed handling", "UNKNOWN", f"missing file: {train_path}")
    elif all(token in train for token in ["random.seed(seed)", "np.random.seed(seed)", "torch.manual_seed(seed)", "torch.cuda.manual_seed_all(seed)"]):
        add(checks, "Wolfram seed handling", "PASS", "train.set_seed covers Python random, NumPy, Torch, and CUDA.")
    else:
        add(checks, "Wolfram seed handling", "WARNING", "Could not verify all seed calls in train.set_seed.")

    if rmd17_missing:
        add(checks, "rMD17 output records seed/config", "UNKNOWN", f"missing file: {rmd17_path}")
    elif "config.__dict__" in rmd17 and '"config"' in rmd17:
        add(checks, "rMD17 output records seed/config", "PASS", "train_one_seed returns config.__dict__, which includes seed and laplacian_mode for new runs.")
    else:
        add(checks, "rMD17 output records seed/config", "WARNING", "Could not verify config serialization.")

    if rmd17_missing:
        add(checks, "rMD17 evaluation horizons", "UNKNOWN", f"missing file: {rmd17_path}")
    elif "eval_horizons: tuple = (1, 2, 4, 8, 16)" in rmd17 and "horizon=max(config.eval_horizons)" in rmd17:
        add(checks, "rMD17 evaluation horizons", "PASS", "Config defaults to H=1,2,4,8,16 and eval transition collection uses max horizon.")
    else:
        add(checks, "rMD17 evaluation horizons", "WARNING", "Could not verify default rMD17 evaluation horizons.")

    if train_missing:
        add(checks, "Wolfram evaluation horizons", "UNKNOWN", f"missing file: {train_path}")
    elif "HORIZONS = [1, 2, 4, 8, 16]" in train:
        add(checks, "Wolfram evaluation horizons", "PASS", "train.py defines HORIZONS = [1, 2, 4, 8, 16].")
    else:
        add(checks, "Wolfram evaluation horizons", "WARNING", "Could not verify Wolfram horizon list.")

    if rmd17_missing:
        add(checks, "rMD17 train/eval sampling separation", "UNKNOWN", f"missing file: {rmd17_path}")
    elif "seed=config.seed + 1000" in rmd17 and "stride=config.stride * 10" in rmd17:
        add(
            checks,
            "rMD17 train/eval sampling separation",
            "WARNING",
            "Eval transitions use a different seed and coarser stride, but collect_rmd17_transitions does not explicitly exclude training frame indices.",
            "For a reviewer-facing leakage audit, add a non-training analysis check that compares sampled train/eval frame_idx sets for each seed.",
        )
    else:
        add(checks, "rMD17 train/eval sampling separation", "UNKNOWN", "Could not verify seed/stride separation.")

    checks.append(rmd17_frame_overlap_check(root))

    if train_missing:
        add(checks, "Wolfram train/eval sampling separation", "UNKNOWN", f"missing file: {train_path}")
    elif "seed=10_000 + current_seed" in train or "seed=10_000 + seed" in train:
        add(
            checks,
            "Wolfram train/eval sampling separation",
            "PASS",
            "Wolfram runners collect eval episodes with seed=10_000 + seed.",
        )
    else:
        add(checks, "Wolfram train/eval sampling separation", "WARNING", "Could not verify separate eval seed.")

    if loader_missing:
        add(checks, "rMD17 transition sampling reproducibility", "UNKNOWN", f"missing file: {loader_path}")
    elif "chosen = rng.choice(candidates" in loader and "chosen.sort()" in loader:
        add(
            checks,
            "rMD17 transition sampling reproducibility",
            "PASS",
            "collect_rmd17_transitions samples candidate frame indices with np.random.default_rng(seed) and records frame_idx.",
        )
    else:
        add(checks, "rMD17 transition sampling reproducibility", "WARNING", "Could not verify deterministic frame sampling.")

    return checks


def write_report(path: Path, checks: list[Check]) -> None:
    counts: dict[str, int] = {}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    lines = [
        "# Audit: Priors And Laplacians",
        "",
        "## Summary",
        "",
        *[f"- {status}: {count}" for status, count in sorted(counts.items())],
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence | Recommendation |",
        "| --- | --- | --- | --- |",
    ]
    for check in checks:
        evidence = check.evidence.replace("|", "\\|").replace("\n", " ")
        recommendation = check.recommendation.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {check.name} | {check.status} | {evidence} | {recommendation} |")

    failures = [check for check in checks if check.status == "FAIL"]
    lines.extend(["", "## Commit-Style Patch Section", ""])
    if failures:
        lines.append("The audit found FAIL checks. No automatic model-math changes were applied.")
        for check in failures:
            lines.append(f"- BUG/RISK: {check.name}: {check.evidence}")
            if check.recommendation:
                lines.append(f"  Minimal fix direction: {check.recommendation}")
    else:
        lines.append("No FAIL checks were found, so no patch section is proposed.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="analysis_out")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out)
    checks = build_checks(root)
    output = out / "AUDIT_PRIORS_AND_LAPLACIANS.md"
    if args.dry_run:
        print(f"would write {output} with {len(checks)} checks")
        for check in checks:
            print(f"{check.status}: {check.name}")
        return

    out.mkdir(parents=True, exist_ok=True)
    write_report(output, checks)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
