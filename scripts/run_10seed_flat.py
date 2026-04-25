"""Overnight validation runner for the flat 10-seed task.

This file also holds the shared implementation used by the hypergraph and
long-horizon wrappers. It imports and reuses the existing training objects
from train.py/model.py instead of duplicating the model implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

LOG_PATH = ROOT / "overnight_log.txt"
SUMMARY_PATH = ROOT / "overnight_summary.md"
DEFAULT_PRIORS = ["none", "euclidean", "spectral"]
DEFAULT_SEEDS = list(range(10))
DEFAULT_HORIZONS = [1, 2, 4, 8, 16]
TASK_TIMEOUT_SEC = 3 * 60 * 60


@dataclass(frozen=True)
class TaskSettings:
    task: str
    encoder: str
    output_json: str
    output_png: str
    seeds: list[int]
    horizons: list[int]
    max_steps: int
    hidden_dim: int = 32


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_log(line: str = "") -> None:
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def ensure_run_start(gpu_name: str | None = None) -> None:
    if LOG_PATH.exists():
        text = LOG_PATH.read_text(encoding="utf-8", errors="replace")
        if "RUN_START_ISO=" in text:
            return
    append_log(f"RUN_START_ISO={utc_now_iso()}")
    append_log(f"GPU_NAME={gpu_name or 'NVIDIA GeForce RTX 5060 Ti'}")


def load_training_symbols() -> dict[str, Any]:
    try:
        import torch
        from model import WorldModel, build_causal_laplacian, euclidean_cov_penalty
        from train import (
            Config,
            build_environment,
            collect_episodes,
            evaluate_rollout,
            flatten_transitions,
            move_obs_to_device,
            set_seed,
            train_one,
        )
    except Exception as exc:  # pragma: no cover - exercised only on broken envs.
        write_failure_summary(f"Import failure: {exc}\n\n{traceback.format_exc()}")
        raise
    return {
        "torch": torch,
        "WorldModel": WorldModel,
        "build_causal_laplacian": build_causal_laplacian,
        "euclidean_cov_penalty": euclidean_cov_penalty,
        "Config": Config,
        "build_environment": build_environment,
        "collect_episodes": collect_episodes,
        "evaluate_rollout": evaluate_rollout,
        "flatten_transitions": flatten_transitions,
        "move_obs_to_device": move_obs_to_device,
        "set_seed": set_seed,
        "train_one": train_one,
    }


def cuda_device_or_fail() -> tuple[Any, str]:
    symbols = load_training_symbols()
    torch = symbols["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for overnight validation, but torch.cuda.is_available() is false")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"device=cuda", flush=True)
    print(f"GPU={gpu_name}", flush=True)
    ensure_run_start(gpu_name)
    return device, gpu_name


def load_pilot_defaults() -> dict[str, Any]:
    with (ROOT / "results.json").open("r", encoding="utf-8") as file:
        pilot_results = json.load(file)
    with (ROOT / "full_summary.json").open("r", encoding="utf-8") as file:
        full_summary = json.load(file)

    config = full_summary["config"]
    prior_weights = config["prior_weights"]
    expected_keys = {
        "flat|none",
        "flat|euclidean",
        "flat|spectral",
        "hypergraph|none",
        "hypergraph|euclidean",
        "hypergraph|spectral",
    }
    missing = sorted(expected_keys.difference(pilot_results.keys()))
    if missing:
        raise RuntimeError(f"results.json is missing expected pilot keys: {missing}")
    if not math.isclose(float(prior_weights["euclidean"]), float(prior_weights["spectral"])):
        raise RuntimeError(f"Pilot prior weights differ: {prior_weights}")

    return {
        "prior_weight": float(prior_weights["spectral"]),
        "prior_weights": {
            "none": float(prior_weights["none"]),
            "euclidean": float(prior_weights["euclidean"]),
            "spectral": float(prior_weights["spectral"]),
        },
        "batch_size": int(config["batch_size"]),
        "num_epochs": int(config["num_epochs"]),
        "latent_dim": 16,
        "hidden_dim": 32,
        "transition_hidden_dim": 128,
        "lr": 1e-3,
        "optimizer": "Adam",
        "n_train": int(config["n_train"]),
        "n_eval": int(config["n_eval"]),
        "env_profile": str(config.get("env_profile", "branching")),
    }


def initial_payload(settings: TaskSettings, defaults: dict[str, Any], smoke: bool, num_epochs: int) -> dict[str, Any]:
    return {
        "task": settings.task,
        "encoder": settings.encoder,
        "max_steps": settings.max_steps,
        "prior_weight": defaults["prior_weight"],
        "hidden_dim": settings.hidden_dim,
        "config": {
            "smoke": smoke,
            "seeds": settings.seeds,
            "horizons": settings.horizons,
            "priors": DEFAULT_PRIORS,
            "n_train": defaults["n_train"],
            "n_eval": defaults["n_eval"],
            "num_epochs": num_epochs,
            "batch_size": defaults["batch_size"],
            "latent_dim": defaults["latent_dim"],
            "hidden_dim": settings.hidden_dim,
            "transition_hidden_dim": defaults["transition_hidden_dim"],
            "optimizer": defaults["optimizer"],
            "lr": defaults["lr"],
            "env_profile": defaults["env_profile"],
            "prior_weights": defaults["prior_weights"],
        },
        "results": {
            f"{settings.encoder}|{prior}": {str(horizon): [] for horizon in settings.horizons}
            for prior in DEFAULT_PRIORS
        },
        "wall_time_sec": 0.0,
        "status": "running",
        "errors": [],
    }


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def finite_values(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def mean_std(values: list[Any]) -> tuple[float, float]:
    finite = finite_values(values)
    if not finite:
        return float("nan"), float("nan")
    mean = sum(finite) / len(finite)
    variance = sum((value - mean) ** 2 for value in finite) / len(finite)
    return mean, math.sqrt(variance)


def format_mean_std(values: list[Any]) -> str:
    mean, std = mean_std(values)
    n = len(finite_values(values))
    if not math.isfinite(mean):
        labels = sorted({str(value) for value in values if not isinstance(value, (int, float))})
        return "/".join(labels) if labels else "nan"
    return f"{mean:.4f}+/-{std:.4f} (n={n})"


def table_for_payload(payload: dict[str, Any], title: str | None = None) -> str:
    horizons = payload["config"]["horizons"]
    lines: list[str] = []
    if title:
        lines.append(title)
    header = "| prior | " + " | ".join(f"H{horizon}" for horizon in horizons) + " |"
    sep = "| --- | " + " | ".join("---" for _ in horizons) + " |"
    lines.extend([header, sep])
    for prior in DEFAULT_PRIORS:
        key = f"{payload['encoder']}|{prior}"
        cells = [format_mean_std(payload["results"][key][str(horizon)]) for horizon in horizons]
        lines.append("| " + prior + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def print_and_log_table(payload: dict[str, Any]) -> None:
    title = f"Task {payload['task']} mean+/-std table"
    table = table_for_payload(payload, title=title)
    print(table, flush=True)
    append_log("")
    append_log(table)


def plot_payload(payload: dict[str, Any], output_png: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    horizons = payload["config"]["horizons"]
    x = np.array(horizons, dtype=float)
    plt.figure(figsize=(8, 5))
    for prior in DEFAULT_PRIORS:
        key = f"{payload['encoder']}|{prior}"
        means: list[float] = []
        stds: list[float] = []
        for horizon in horizons:
            mean, std = mean_std(payload["results"][key][str(horizon)])
            means.append(mean)
            stds.append(std)
        y = np.array(means, dtype=float)
        yerr = np.array(stds, dtype=float)
        if not np.isfinite(y).any():
            continue
        plt.plot(x, y, marker="o", linewidth=1.8, label=prior)
        lower = np.maximum(y - yerr, 1e-8)
        upper = np.maximum(y + yerr, 1e-8)
        plt.fill_between(x, lower, upper, alpha=0.15)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(horizons, [str(horizon) for horizon in horizons])
    plt.xlabel("Rollout horizon H")
    plt.ylabel("Mean rollout error")
    plt.title(f"Task {payload['task']}: {payload['encoder']} encoder")
    plt.grid(True, which="both", linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def train_cell(
    *,
    settings: TaskSettings,
    defaults: dict[str, Any],
    prior: str,
    seed: int,
    train_transitions: list[dict[str, Any]],
    eval_episodes: list[dict[str, Any]],
    batch_size: int,
    num_epochs: int,
) -> dict[int, float]:
    symbols = load_training_symbols()
    torch = symbols["torch"]
    Config = symbols["Config"]
    WorldModel = symbols["WorldModel"]
    build_causal_laplacian = symbols["build_causal_laplacian"]
    euclidean_cov_penalty = symbols["euclidean_cov_penalty"]
    evaluate_rollout = symbols["evaluate_rollout"]
    move_obs_to_device = symbols["move_obs_to_device"]
    set_seed = symbols["set_seed"]

    device = torch.device("cuda")
    set_seed(seed)
    config = Config(
        encoder=settings.encoder,
        prior=prior,
        prior_weight=defaults["prior_weights"][prior],
        latent_dim=defaults["latent_dim"],
        hidden_dim=settings.hidden_dim,
        transition_hidden_dim=defaults["transition_hidden_dim"],
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=defaults["lr"],
        seed=seed,
    )
    model = WorldModel(
        encoder=config.encoder,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        action_dim=1,
        transition_hidden_dim=config.transition_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    rng = random.Random(seed)
    label = f"{settings.task}|{settings.encoder}|{prior}|seed={seed}"
    max_horizon = max(settings.horizons)

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_transitions = list(train_transitions)
        rng.shuffle(epoch_transitions)
        total_values: list[float] = []
        prior_values: list[float] = []

        for start in range(0, len(epoch_transitions), config.batch_size):
            batch = epoch_transitions[start:start + config.batch_size]
            optimizer.zero_grad()
            base_totals = []
            prior_losses = []
            batch_latents = []

            for transition in batch:
                obs = move_obs_to_device(transition["obs"], device)
                next_obs = move_obs_to_device(transition["next_obs"], device)
                action = torch.tensor([float(transition["action"])], device=device)
                reward = torch.tensor(float(transition["reward"]), device=device)
                done = torch.tensor(float(transition["done"]), device=device)

                if config.prior == "spectral":
                    laplacian = build_causal_laplacian(transition["causal_graph"], config.latent_dim).to(device)
                    loss_dict = model.loss(
                        obs,
                        action,
                        next_obs,
                        reward,
                        done,
                        prior="spectral",
                        prior_weight=config.prior_weight,
                        laplacian=laplacian,
                    )
                else:
                    loss_dict = model.loss(
                        obs,
                        action,
                        next_obs,
                        reward,
                        done,
                        prior="none",
                        prior_weight=0.0,
                    )
                    if config.prior == "euclidean":
                        batch_latents.append(model.encode(obs))

                base_totals.append(loss_dict["total"])
                prior_losses.append(loss_dict["prior"])

            total = torch.stack(base_totals).mean()
            if config.prior == "euclidean":
                prior_loss = euclidean_cov_penalty(torch.stack(batch_latents))
                total = total + config.prior_weight * prior_loss
            elif config.prior == "spectral":
                prior_loss = torch.stack(prior_losses).mean()
            else:
                prior_loss = total.new_tensor(0.0)

            total.backward()
            optimizer.step()
            total_values.append(float(total.detach().cpu()))
            prior_values.append(float(prior_loss.detach().cpu()))

        if epoch % 20 == 0 or epoch == config.num_epochs:
            eval_metrics = evaluate_rollout(model, eval_episodes, [max_horizon], device)
            mean_total = sum(total_values) / len(total_values) if total_values else float("nan")
            mean_prior = sum(prior_values) / len(prior_values) if prior_values else float("nan")
            print(
                f"[{label}] epoch {epoch:3d}/{config.num_epochs}  "
                f"total={mean_total:.3f}  prior={mean_prior:.3f}  "
                f"H{max_horizon}={eval_metrics.get(max_horizon, float('nan')):.3f}",
                flush=True,
            )

    return evaluate_rollout(model, eval_episodes, settings.horizons, device)


def run_cell_with_oom_retry(
    *,
    settings: TaskSettings,
    defaults: dict[str, Any],
    prior: str,
    seed: int,
    train_transitions: list[dict[str, Any]],
    eval_episodes: list[dict[str, Any]],
    num_epochs: int,
) -> dict[str, Any]:
    symbols = load_training_symbols()
    torch = symbols["torch"]
    batch_sizes = [defaults["batch_size"], max(1, defaults["batch_size"] // 2)]
    started = time.monotonic()
    last_error = ""

    for attempt, batch_size in enumerate(batch_sizes, start=1):
        try:
            metrics = train_cell(
                settings=settings,
                defaults=defaults,
                prior=prior,
                seed=seed,
                train_transitions=train_transitions,
                eval_episodes=eval_episodes,
                batch_size=batch_size,
                num_epochs=num_epochs,
            )
            elapsed = time.monotonic() - started
            done = " ".join(f"H{horizon}={metrics[horizon]:.3f}" for horizon in settings.horizons)
            line = f"[{settings.task}|{settings.encoder}|{prior}|seed={seed}] DONE   {done}  ({elapsed:.0f}s)"
            print(line, flush=True)
            append_log(line)
            return {"status": "ok", "metrics": {str(k): float(v) for k, v in metrics.items()}}
        except Exception as exc:
            last_error = traceback.format_exc()
            if not is_oom(exc) or attempt == len(batch_sizes):
                if is_oom(exc):
                    torch.cuda.empty_cache()
                    elapsed = time.monotonic() - started
                    line = f"[{settings.task}|{settings.encoder}|{prior}|seed={seed}] DONE   OOM  ({elapsed:.0f}s)"
                    print(line, flush=True)
                    append_log(line)
                    return {"status": "OOM", "metrics": {str(horizon): "OOM" for horizon in settings.horizons}, "error": last_error}
                raise
            torch.cuda.empty_cache()
            print(
                f"[{settings.task}|{settings.encoder}|{prior}|seed={seed}] OOM at batch_size={batch_size}; "
                f"retrying once with batch_size={batch_sizes[-1]}",
                flush=True,
            )
    return {"status": "error", "metrics": {str(horizon): "error" for horizon in settings.horizons}, "error": last_error}


def seed_worker(
    settings_data: dict[str, Any],
    defaults: dict[str, Any],
    seed: int,
    num_epochs: int,
    queue: Any,
) -> None:
    try:
        settings = TaskSettings(**settings_data)
        symbols = load_training_symbols()
        torch = symbols["torch"]
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA became unavailable inside worker")
        device = torch.device("cuda")
        build_environment = symbols["build_environment"]
        collect_episodes = symbols["collect_episodes"]
        flatten_transitions = symbols["flatten_transitions"]
        set_seed = symbols["set_seed"]

        set_seed(seed)
        rule, initial_state, _ = build_environment(defaults["env_profile"], seed)
        del initial_state
        train_episodes = collect_episodes(
            rule,
            None,
            defaults["n_train"],
            settings.max_steps,
            seed=seed,
            env_profile=defaults["env_profile"],
        )
        eval_episodes = collect_episodes(
            rule,
            None,
            defaults["n_eval"],
            settings.max_steps,
            seed=10_000 + seed,
            env_profile=defaults["env_profile"],
        )
        train_transitions = flatten_transitions(train_episodes)
        cells: dict[str, Any] = {}

        for prior in DEFAULT_PRIORS:
            key = f"{settings.encoder}|{prior}"
            try:
                cells[key] = run_cell_with_oom_retry(
                    settings=settings,
                    defaults=defaults,
                    prior=prior,
                    seed=seed,
                    train_transitions=train_transitions,
                    eval_episodes=eval_episodes,
                    num_epochs=num_epochs,
                )
            except Exception:
                error = traceback.format_exc()
                line = f"[{settings.task}|{settings.encoder}|{prior}|seed={seed}] DONE   error  ({error.splitlines()[-1]})"
                print(line, flush=True)
                append_log(line)
                cells[key] = {
                    "status": "error",
                    "metrics": {str(horizon): "error" for horizon in settings.horizons},
                    "error": error,
                }
            torch.cuda.empty_cache()
        queue.put({"seed": seed, "cells": cells})
    except Exception:
        queue.put({"seed": seed, "fatal_error": traceback.format_exc()})


def append_seed_results(payload: dict[str, Any], seed_result: dict[str, Any]) -> None:
    horizons = payload["config"]["horizons"]
    if "fatal_error" in seed_result:
        payload["errors"].append({"seed": seed_result["seed"], "error": seed_result["fatal_error"]})
        for prior in DEFAULT_PRIORS:
            key = f"{payload['encoder']}|{prior}"
            for horizon in horizons:
                payload["results"][key][str(horizon)].append("error")
        return

    for prior in DEFAULT_PRIORS:
        key = f"{payload['encoder']}|{prior}"
        cell = seed_result["cells"].get(key, {})
        metrics = cell.get("metrics", {str(horizon): "error" for horizon in horizons})
        if cell.get("status") not in ("ok", None):
            payload["errors"].append({"seed": seed_result["seed"], "cell": key, "status": cell.get("status"), "error": cell.get("error", "")})
        for horizon in horizons:
            payload["results"][key][str(horizon)].append(metrics.get(str(horizon), "error"))


def append_timeout_seed(payload: dict[str, Any], seed: int) -> None:
    horizons = payload["config"]["horizons"]
    payload["errors"].append({"seed": seed, "status": "timeout"})
    for prior in DEFAULT_PRIORS:
        key = f"{payload['encoder']}|{prior}"
        for horizon in horizons:
            payload["results"][key][str(horizon)].append("timeout")


def validate_smoke(payload: dict[str, Any]) -> None:
    for prior in DEFAULT_PRIORS:
        key = f"{payload['encoder']}|{prior}"
        for horizon in payload["config"]["horizons"]:
            values = payload["results"][key][str(horizon)]
            if len(values) != 1 or not isinstance(values[0], (int, float)) or not math.isfinite(float(values[0])):
                raise RuntimeError(f"Smoke validation failed for {key} H{horizon}: {values}")


def run_task(settings: TaskSettings, *, smoke: bool = False, num_epochs_override: int | None = None, task_timeout_sec: int = TASK_TIMEOUT_SEC) -> dict[str, Any]:
    device, gpu_name = cuda_device_or_fail()
    del device
    defaults = load_pilot_defaults()
    seeds = settings.seeds
    num_epochs = num_epochs_override if num_epochs_override is not None else defaults["num_epochs"]
    payload = initial_payload(settings, defaults, smoke, num_epochs)
    output_json = ROOT / settings.output_json
    output_png = ROOT / settings.output_png
    start = time.monotonic()
    save_payload(output_json, payload)

    print(
        f"Starting Task {settings.task}: encoder={settings.encoder} max_steps={settings.max_steps} "
        f"horizons={settings.horizons} seeds={seeds} device=cuda GPU={gpu_name}",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    timed_out = False
    completed_seeds = 0

    for seed in seeds:
        remaining = task_timeout_sec - (time.monotonic() - start)
        if remaining <= 0:
            timed_out = True
            append_timeout_seed(payload, seed)
            save_payload(output_json, payload)
            continue

        queue = ctx.Queue()
        process = ctx.Process(
            target=seed_worker,
            args=(settings.__dict__, defaults, seed, num_epochs, queue),
        )
        process.start()
        process.join(remaining)

        if process.is_alive():
            process.terminate()
            process.join(30)
            timed_out = True
            line = f"[Task {settings.task}] timeout after {task_timeout_sec}s at seed={seed}"
            print(line, flush=True)
            append_log(line)
            append_timeout_seed(payload, seed)
            save_payload(output_json, payload)
            break

        if queue.empty():
            payload["errors"].append({"seed": seed, "status": "worker_exit_without_result", "exitcode": process.exitcode})
            append_timeout_seed(payload, seed)
        else:
            append_seed_results(payload, queue.get())
        completed_seeds += 1
        payload["wall_time_sec"] = time.monotonic() - start
        save_payload(output_json, payload)

    if timed_out:
        for seed in seeds[completed_seeds + 1 :]:
            append_timeout_seed(payload, seed)

    payload["wall_time_sec"] = time.monotonic() - start
    payload["status"] = "timeout" if timed_out else "complete"
    save_payload(output_json, payload)
    plot_payload(payload, output_png)
    print_and_log_table(payload)

    if smoke:
        validate_smoke(payload)

    return payload


def parse_seed_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def read_run_start() -> datetime | None:
    if not LOG_PATH.exists():
        return None
    for line in LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("RUN_START_ISO="):
            try:
                return datetime.fromisoformat(line.split("=", 1)[1])
            except ValueError:
                return None
    return None


def read_gpu_name() -> str:
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("GPU_NAME="):
                return line.split("=", 1)[1]
    return "NVIDIA GeForce RTX 5060 Ti"


def load_payload_if_exists(path: str) -> dict[str, Any] | None:
    full_path = ROOT / path
    if not full_path.exists():
        return None
    with full_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def h_value(payload: dict[str, Any], prior: str, horizon: int) -> tuple[float, float]:
    key = f"{payload['encoder']}|{prior}"
    return mean_std(payload["results"][key][str(horizon)])


def advantage_sentence(payload: dict[str, Any], label: str, horizon: int) -> tuple[str, bool | None]:
    euc_mean, euc_std = h_value(payload, "euclidean", horizon)
    spec_mean, spec_std = h_value(payload, "spectral", horizon)
    if not (math.isfinite(euc_mean) and math.isfinite(spec_mean)):
        return f"{label}: finite euclidean/spectral values at H={horizon} were not available, so no advantage can be assessed.", None
    advantage = euc_mean - spec_mean
    combined_std = math.sqrt(euc_std**2 + spec_std**2)
    beats = advantage > 0
    if advantage > combined_std:
        relation = "larger than"
    else:
        relation = "smaller than or equal to"
    if beats:
        verdict = "spectral is lower"
    else:
        verdict = "spectral is not lower"
    return (
        f"{label}: at H={horizon}, euclidean mean is {euc_mean:.4f} and spectral mean is {spec_mean:.4f}, "
        f"so mean_euc - mean_spec = {advantage:.4f}; the combined std is {combined_std:.4f}, "
        f"making the advantage {relation} the combined std. Numerically, {verdict}.",
        beats,
    )


def long_horizon_sentence(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    advantages: list[tuple[int, float]] = []
    for horizon in payload["config"]["horizons"]:
        euc_mean, _ = h_value(payload, "euclidean", int(horizon))
        spec_mean, _ = h_value(payload, "spectral", int(horizon))
        if math.isfinite(euc_mean) and math.isfinite(spec_mean):
            advantage = euc_mean - spec_mean
            advantages.append((int(horizon), advantage))
            parts.append(f"H{horizon}={advantage:.4f}")
    if not advantages:
        return "Long horizon (B): finite euclidean/spectral values were not available, so the advantage curve cannot be described."
    values = [value for _, value in advantages]
    if any(value <= 0 for value in values) and any(value > 0 for value in values):
        shape = "mixed-sign and non-monotonic"
    elif len(values) >= 3 and all(values[i] <= values[i + 1] for i in range(len(values) - 1)):
        ratios = [value / horizon for horizon, value in advantages if horizon > 0]
        shape = "sublinear" if ratios[-1] < ratios[0] else "roughly linear to superlinear"
    elif len(values) >= 3 and all(values[i] >= values[i + 1] for i in range(len(values) - 1)):
        shape = "decreasing"
    else:
        shape = "non-monotonic"
    return f"Long horizon (B): the spectral advantage curve, measured as mean_euc - mean_spec, is {', '.join(parts)}. Across the measured horizons it looks {shape}; this only describes the observed run."


def write_overnight_summary(extra_error: str | None = None) -> None:
    start = read_run_start()
    end = datetime.now(timezone.utc)
    total = (end - start).total_seconds() if start is not None else float("nan")
    payloads = [
        ("Task A results table (with std bands)", load_payload_if_exists("validation_10seed_flat.json")),
        ("Task A+ results table (with std bands)", load_payload_if_exists("validation_10seed_hypergraph.json")),
        ("Task B results table (with std bands)", load_payload_if_exists("long_horizon.json")),
    ]

    lines = [
        "# Overnight Validation Summary",
        "",
        f"- Start time: {start.isoformat(timespec='seconds') if start else 'unknown'}",
        f"- End time: {end.isoformat(timespec='seconds')}",
        f"- Total wall time: {total:.1f} sec" if math.isfinite(total) else "- Total wall time: unknown",
        f"- GPU: {read_gpu_name()}",
        "",
    ]

    for title, payload in payloads:
        lines.append(f"## {title}")
        lines.append("")
        if payload is None:
            lines.append("No artifact was produced.")
        else:
            lines.append(table_for_payload(payload))
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    task_a = payloads[0][1]
    task_ap = payloads[1][1]
    task_b = payloads[2][1]

    if task_a is None:
        flat_text, flat_beats = "Flat encoder (A): no Task A artifact was available.", None
    else:
        flat_text, flat_beats = advantage_sentence(task_a, "Flat encoder (A)", 16)
    if task_ap is None:
        hyper_text, hyper_beats = "Hypergraph encoder (A+): no Task A+ artifact was available.", None
    else:
        hyper_text, hyper_beats = advantage_sentence(task_ap, "Hypergraph encoder (A+)", 16)

    lines.append(flat_text)
    lines.append("")
    if flat_beats is True and hyper_beats is True:
        presence = "The spectral advantage is present on both encoders by mean."
    elif flat_beats is True and hyper_beats is False:
        presence = "The spectral advantage is present on the flat encoder only by mean."
    elif flat_beats is False and hyper_beats is True:
        presence = "The spectral advantage is present on the hypergraph encoder only by mean."
    elif flat_beats is False and hyper_beats is False:
        presence = "The spectral advantage is present on neither encoder by mean."
    else:
        presence = "The cross-encoder spectral-advantage claim cannot be fully assessed from the available artifacts."
    lines.append(f"{hyper_text} {presence}")
    lines.append("")
    lines.append(long_horizon_sentence(task_b) if task_b is not None else "Long horizon (B): no Task B artifact was available.")
    lines.append("")

    lines.append("## Decisions Made Without Approval")
    lines.append("")
    lines.append("- Used `env_profile=branching` because `full_summary.json` recorded that profile for the pilot whose defaults this run was asked to match.")
    lines.append("- Used the existing activated Python environment and did not install or reinstall dependencies.")
    if extra_error:
        lines.append("")
        lines.append("## Errors")
        lines.append("")
        lines.append("```")
        lines.append(extra_error.rstrip())
        lines.append("```")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_summary(message: str) -> None:
    ensure_run_start("NVIDIA GeForce RTX 5060 Ti")
    write_overnight_summary(extra_error=message)


def main_for_settings(settings: TaskSettings) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=str)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--task-timeout-sec", type=int, default=TASK_TIMEOUT_SEC)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.summary_only:
        write_overnight_summary()
        return 0

    selected_settings = settings
    if args.smoke:
        selected_settings = TaskSettings(
            task=settings.task,
            encoder=settings.encoder,
            output_json=settings.output_json,
            output_png=settings.output_png,
            seeds=[0],
            horizons=settings.horizons,
            max_steps=settings.max_steps,
            hidden_dim=settings.hidden_dim,
        )
    elif args.seeds:
        selected_settings = TaskSettings(
            task=settings.task,
            encoder=settings.encoder,
            output_json=settings.output_json,
            output_png=settings.output_png,
            seeds=parse_seed_list(args.seeds),
            horizons=settings.horizons,
            max_steps=settings.max_steps,
            hidden_dim=settings.hidden_dim,
        )

    num_epochs = 10 if args.smoke and args.num_epochs is None else args.num_epochs
    try:
        run_task(
            selected_settings,
            smoke=args.smoke,
            num_epochs_override=num_epochs,
            task_timeout_sec=args.task_timeout_sec,
        )
        return 0
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        write_failure_summary(error)
        return 1


def main() -> int:
    settings = TaskSettings(
        task="A",
        encoder="flat",
        output_json="validation_10seed_flat.json",
        output_png="validation_10seed_flat.png",
        seeds=DEFAULT_SEEDS,
        horizons=DEFAULT_HORIZONS,
        max_steps=16,
        hidden_dim=32,
    )
    return main_for_settings(settings)


if __name__ == "__main__":
    raise SystemExit(main())
