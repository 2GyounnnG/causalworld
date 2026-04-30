from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "lj_raw"

RHO = 0.8
TEMPERATURE = 1.0
DT = 0.005
CUTOFF = 2.5
CUTOFF2 = CUTOFF * CUTOFF
EPSILON = 1.0
SIGMA = 1.0
PARTICLE_COUNTS = (64, 256, 1024)


def initialize_positions(n_particles: int, box_size: float, rng: np.random.Generator) -> np.ndarray:
    """Initialize particles on a grid with small random perturbations."""
    n_side = int(np.ceil(n_particles ** (1.0 / 3.0)))
    spacing = box_size / n_side
    grid = np.stack(
        np.meshgrid(
            np.arange(n_side, dtype=np.float64),
            np.arange(n_side, dtype=np.float64),
            np.arange(n_side, dtype=np.float64),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    positions = (grid[:n_particles] + 0.5) * spacing
    perturb = rng.uniform(-0.05 * spacing, 0.05 * spacing, size=positions.shape)
    return (positions + perturb) % box_size


def initialize_velocities(n_particles: int, temperature: float, rng: np.random.Generator) -> np.ndarray:
    velocities = rng.normal(0.0, np.sqrt(temperature), size=(n_particles, 3))
    velocities -= velocities.mean(axis=0, keepdims=True)
    return velocities


def kinetic_energy(velocities: np.ndarray) -> float:
    return 0.5 * float(np.sum(velocities * velocities))


def rescale_velocities(velocities: np.ndarray, target_temperature: float) -> np.ndarray:
    current_temperature = 2.0 * kinetic_energy(velocities) / (3.0 * velocities.shape[0] - 3.0)
    if current_temperature <= 0.0:
        return velocities
    return velocities * np.sqrt(target_temperature / current_temperature)


def lj_forces(positions: np.ndarray, box_size: float) -> tuple[np.ndarray, float]:
    """Compute Lennard-Jones forces and potential energy with PBC."""
    delta = positions[:, None, :] - positions[None, :, :]
    delta -= box_size * np.rint(delta / box_size)
    r2 = np.sum(delta * delta, axis=-1)
    mask = (r2 < CUTOFF2) & (r2 > 0.0)

    inv_r2 = np.zeros_like(r2)
    inv_r2[mask] = (SIGMA * SIGMA) / r2[mask]
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2

    pair_scalar = np.zeros_like(r2)
    pair_scalar[mask] = 24.0 * EPSILON * (2.0 * inv_r12[mask] - inv_r6[mask]) / r2[mask]
    forces = np.sum(pair_scalar[:, :, None] * delta, axis=1)

    upper = np.triu(mask, k=1)
    potential = 4.0 * EPSILON * float(np.sum(inv_r12[upper] - inv_r6[upper]))
    return forces, potential


def velocity_verlet_step(
    positions_abs: np.ndarray,
    positions_box: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    box_size: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    positions_abs = positions_abs + velocities * dt + 0.5 * forces * dt * dt
    positions_box = positions_abs % box_size
    new_forces, potential = lj_forces(positions_box, box_size)
    velocities = velocities + 0.5 * (forces + new_forces) * dt
    return positions_abs, positions_box, velocities, new_forces, potential


def run_lj(
    n_particles: int,
    quick: bool = False,
    seed: int = 0,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    rng = np.random.default_rng(seed + n_particles)
    box_size = float((n_particles / RHO) ** (1.0 / 3.0))
    positions_box = initialize_positions(n_particles, box_size, rng)
    positions_abs = positions_box.copy()
    velocities = initialize_velocities(n_particles, TEMPERATURE, rng)
    forces, potential = lj_forces(positions_box, box_size)

    equil_steps = 1000 if quick else 5000
    prod_steps = 1000 if quick else 5000
    sample_every = 5
    n_frames = prod_steps // sample_every

    print(f"N={n_particles}: box={box_size:.6f}, equil={equil_steps}, production={prod_steps}", flush=True)

    for step in range(1, equil_steps + 1):
        positions_abs, positions_box, velocities, forces, potential = velocity_verlet_step(
            positions_abs,
            positions_box,
            velocities,
            forces,
            box_size,
            DT,
        )
        if step <= 2000 and step % 100 == 0:
            velocities = rescale_velocities(velocities, TEMPERATURE)
        if step % 1000 == 0:
            total_energy = kinetic_energy(velocities) + potential
            print(f"N={n_particles}: equil step {step}/{equil_steps}, E={total_energy:.6f}", flush=True)

    coords = np.empty((n_frames, n_particles, 3), dtype=np.float32)
    energies = np.empty((n_frames,), dtype=np.float32)
    frame = 0
    for step in range(1, prod_steps + 1):
        positions_abs, positions_box, velocities, forces, potential = velocity_verlet_step(
            positions_abs,
            positions_box,
            velocities,
            forces,
            box_size,
            DT,
        )
        if step % sample_every == 0:
            coords[frame] = positions_abs.astype(np.float32)
            energies[frame] = np.float32(kinetic_energy(velocities) + potential)
            frame += 1
        if step % 1000 == 0:
            total_energy = kinetic_energy(velocities) + potential
            print(f"N={n_particles}: production step {step}/{prod_steps}, E={total_energy:.6f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"lj_N{n_particles}.npz"
    np.savez_compressed(
        path,
        coords=coords,
        nuclear_charges=np.full((n_particles,), 18, dtype=np.int32),
        energies=energies,
        box_size=np.full((3,), box_size, dtype=np.float32),
    )
    print(f"N={n_particles}: wrote {path}", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Lennard-Jones fluid trajectories.")
    parser.add_argument("--quick", action="store_true", help="Use 1000 equilibration and production steps.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for n_particles in PARTICLE_COUNTS:
        run_lj(n_particles, quick=args.quick, seed=args.seed)


if __name__ == "__main__":
    main()
