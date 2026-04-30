from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "3bpa_raw"
BASE_URL = "https://raw.githubusercontent.com/davkovacs/BOTNet-datasets/main/dataset_3BPA"
FILES = ("train_mixedT.xyz", "test_300K.xyz")


def _energy_from_info(info: dict, atoms=None) -> float:
    for key in ("REF_energy", "energy"):
        if key in info:
            value = np.asarray(info[key]).reshape(-1)
            if value.size:
                return float(value[0])
    if atoms is not None:
        try:
            return float(atoms.get_potential_energy())
        except Exception:
            pass
    raise KeyError(f"No energy found in info keys: {sorted(info.keys())}")


def convert_extxyz(path: Path, split: str | None = None, output_dir: Path = OUTPUT_DIR) -> Path:
    """Convert one 3BPA extxyz trajectory to rMD17-compatible npz."""
    try:
        from ase.io import read
    except ImportError as exc:
        raise ImportError("Converting 3BPA requires ASE. Install ase and retry.") from exc

    atoms_list = read(path, index=":")
    if not atoms_list:
        raise ValueError(f"No frames found in {path}")

    atomic_numbers = atoms_list[0].get_atomic_numbers().astype(np.int32)
    n_atoms = int(atomic_numbers.shape[0])
    coords = np.empty((len(atoms_list), n_atoms, 3), dtype=np.float32)
    energies = np.empty((len(atoms_list),), dtype=np.float32)

    for frame_idx, atoms in enumerate(atoms_list):
        frame_numbers = atoms.get_atomic_numbers().astype(np.int32)
        if frame_numbers.shape != atomic_numbers.shape or not np.array_equal(frame_numbers, atomic_numbers):
            raise ValueError(f"Frame {frame_idx} has inconsistent atomic numbers")
        coords[frame_idx] = atoms.get_positions().astype(np.float32)
        energies[frame_idx] = np.float32(_energy_from_info(atoms.info, atoms))

    if split is None:
        split = path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"3bpa_{split}.npz"
    np.savez_compressed(
        out_path,
        coords=coords,
        nuclear_charges=atomic_numbers,
        energies=energies,
    )
    print(f"wrote {out_path} ({coords.shape[0]} frames, {n_atoms} atoms)", flush=True)
    return out_path


def download_3bpa(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """Download and convert the 3BPA train/test extxyz files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    for filename in FILES:
        xyz_path = output_dir / filename
        url = f"{BASE_URL}/{filename}"
        if not xyz_path.exists():
            subprocess.run(["wget", "-O", str(xyz_path), url], check=True)
        converted.append(convert_extxyz(xyz_path, split=xyz_path.stem, output_dir=output_dir))
    return converted


def main() -> None:
    download_3bpa()


if __name__ == "__main__":
    main()
