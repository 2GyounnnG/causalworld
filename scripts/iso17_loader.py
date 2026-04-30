from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "iso17_raw" / "iso17"
PROCESSED_DIRNAME = "iso17_processed"
COVALENT_SCALE = 1.2
FALLBACK_RADII = {
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    15: 1.07,
    16: 1.05,
    17: 1.02,
}


def _resolve_data_root(data_root: Optional[Path] = None) -> Path:
    if data_root is not None:
        return Path(data_root)
    env = os.environ.get("ISO17_DATA_DIR")
    if env:
        return Path(env)
    return DEFAULT_DATA_ROOT


def _reference_db_path(root: Path) -> Path:
    candidates = [root / "reference.db", root / "iso17" / "reference.db"]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _train_ids_path(root: Path) -> Path:
    candidates = [root / "train_ids.txt", root / "iso17" / "train_ids.txt"]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _processed_dir(root: Path) -> Path:
    if root.name == "iso17":
        return root.parent / PROCESSED_DIRNAME
    return root / PROCESSED_DIRNAME


def _read_train_ids(path: Path) -> list[tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing ISO17 train id file: {path}")
    rows: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip().replace(",", " ")
            if not stripped or stripped.startswith("#"):
                continue
            values = [int(float(part)) for part in stripped.split()]
            if len(values) >= 2:
                rows.append((values[0], values[1]))
            elif len(values) == 1:
                rows.append((-1, values[0]))
    if not rows:
        raise ValueError(f"No ids parsed from {path}")
    return rows


def _row_value(row: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if hasattr(row, key):
            value = getattr(row, key)
            if value is not None:
                return value
        try:
            value = row.get(key)
        except Exception:
            value = None
        if value is not None:
            return value
        for container_name in ("key_value_pairs", "data"):
            container = getattr(row, container_name, None)
            if isinstance(container, dict) and key in container:
                return container[key]
    raise KeyError(f"None of {keys} found in ASE row")


def _row_mol_id(row: Any) -> int | None:
    for keys in (("mol_id", "molecule_id", "isomer_id", "isomer"), ("molecule",)):
        try:
            value = _row_value(row, keys)
        except KeyError:
            continue
        try:
            return int(np.asarray(value).reshape(-1)[0])
        except (TypeError, ValueError):
            return None
    return None


def _db_get_frame(db: Any, frame_id: int) -> Any:
    for row_id in (frame_id, frame_id + 1):
        try:
            return db.get(id=int(row_id))
        except Exception:
            continue
    raise KeyError(f"Could not read ISO17 db row for frame_id={frame_id}")


def _energy_from_row(row: Any) -> float:
    for keys in (("total_energy", "energy"), ("E", "e")):
        try:
            value = _row_value(row, keys)
            return float(np.asarray(value).reshape(-1)[0])
        except KeyError:
            continue
    return float("nan")


def _select_default_mol_id(db: Any, train_ids: list[tuple[int, int]], min_frames: int) -> tuple[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    unknown_ids: list[int] = []
    for mol_id, frame_id in train_ids:
        if mol_id >= 0:
            grouped[int(mol_id)].append(int(frame_id))
        else:
            unknown_ids.append(int(frame_id))

    if unknown_ids:
        for frame_id in unknown_ids:
            row = _db_get_frame(db, frame_id)
            mol_id = _row_mol_id(row)
            if mol_id is None:
                mol_id = 0
            grouped[int(mol_id)].append(int(frame_id))

    for mol_id in sorted(grouped):
        frame_ids = grouped[mol_id]
        if len(frame_ids) >= min_frames:
            return mol_id, frame_ids
    available = {mol_id: len(frame_ids) for mol_id, frame_ids in grouped.items()}
    raise ValueError(f"No ISO17 mol_id has at least {min_frames} train frames. Available counts: {available}")


def _covalent_radii(atomic_numbers: np.ndarray) -> np.ndarray:
    try:
        from ase.data import covalent_radii

        return np.asarray([float(covalent_radii[int(z)]) for z in atomic_numbers], dtype=np.float32)
    except ImportError:
        return np.asarray([FALLBACK_RADII.get(int(z), 0.75) for z in atomic_numbers], dtype=np.float32)


def _bond_edges(coords_t: np.ndarray, atomic_numbers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords_t = np.asarray(coords_t, dtype=np.float32)
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    radii = _covalent_radii(atomic_numbers)
    diff = coords_t[:, None, :] - coords_t[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    cutoffs = COVALENT_SCALE * (radii[:, None] + radii[None, :])
    mask = (dist < cutoffs) & (~np.eye(coords_t.shape[0], dtype=bool))
    src, dst = np.where(mask)
    return np.stack([src, dst], axis=0).astype(np.int64), dist[src, dst].astype(np.float32)


def atoms_to_pyg(coords_t: np.ndarray, atomic_numbers: np.ndarray) -> HeteroData:
    """Convert a single ISO17 frame to the same HeteroData schema as rMD17."""
    coords_t = np.asarray(coords_t, dtype=np.float32)
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    if coords_t.ndim != 2 or coords_t.shape[1] != 3:
        raise ValueError(f"coords_t must have shape (N_atoms, 3), got {coords_t.shape}")
    if atomic_numbers.ndim != 1 or atomic_numbers.shape[0] != coords_t.shape[0]:
        raise ValueError(
            "atomic_numbers must have shape (N_atoms,) matching coords_t; "
            f"got {atomic_numbers.shape} vs {coords_t.shape}"
        )

    edge_index, edge_attr = _bond_edges(coords_t, atomic_numbers)
    data = HeteroData()
    pos = torch.from_numpy(coords_t.astype(np.float32))
    atom_z = torch.from_numpy(atomic_numbers.astype(np.int64))
    atom_x = torch.cat(
        [
            torch.from_numpy(atomic_numbers.astype(np.float32)).unsqueeze(-1),
            pos,
        ],
        dim=-1,
    )
    data["atom"].pos = pos
    data["atom"].atomic_number = atom_z
    data["atom"].atomic_numbers = atom_z
    data["atom"].x = atom_x
    data["atom", "bonded", "atom"].edge_index = torch.from_numpy(edge_index)
    data["atom", "bonded", "atom"].edge_attr = torch.from_numpy(edge_attr).unsqueeze(-1)
    return data


def cache_iso17_default_isomer(
    data_root: Optional[Path] = None,
    min_frames: int = 4000,
) -> Path:
    """Convert the default ISO17 training isomer to an rMD17-style npz cache."""
    try:
        from ase.db import connect
    except ImportError as exc:
        raise ImportError("Reading ISO17 reference.db requires ASE. Install ase and retry.") from exc

    root = _resolve_data_root(data_root)
    db_path = _reference_db_path(root)
    train_ids_path = _train_ids_path(root)
    if not db_path.exists():
        raise FileNotFoundError(f"Missing ISO17 reference database: {db_path}")
    db = connect(db_path)
    train_ids = _read_train_ids(train_ids_path)
    mol_id, frame_ids = _select_default_mol_id(db, train_ids, min_frames=min_frames)

    processed_dir = _processed_dir(root)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = processed_dir / f"iso17_mol{mol_id}.npz"
    if cache_path.exists():
        return cache_path

    coords = []
    energies = []
    atomic_numbers = None
    for frame_id in frame_ids:
        row = _db_get_frame(db, frame_id)
        row_mol_id = _row_mol_id(row)
        if row_mol_id is not None and row_mol_id != mol_id:
            continue
        atoms = row.toatoms()
        numbers = atoms.get_atomic_numbers().astype(np.int32)
        if atomic_numbers is None:
            atomic_numbers = numbers
        elif not np.array_equal(numbers, atomic_numbers):
            raise ValueError(f"Frame {frame_id} has inconsistent atomic numbers for mol_id={mol_id}")
        coords.append(atoms.get_positions().astype(np.float32))
        energies.append(_energy_from_row(row))

    if atomic_numbers is None or len(coords) < min_frames:
        raise ValueError(f"Collected only {len(coords)} frames for ISO17 mol_id={mol_id}")

    np.savez_compressed(
        cache_path,
        coords=np.asarray(coords, dtype=np.float32),
        nuclear_charges=atomic_numbers.astype(np.int32),
        energies=np.asarray(energies, dtype=np.float32),
        mol_id=np.asarray(mol_id, dtype=np.int32),
    )
    return cache_path


def _find_cache(dataset_id: str, data_root: Optional[Path] = None) -> Path:
    root = _resolve_data_root(data_root)
    processed_dir = _processed_dir(root)
    if dataset_id not in {"iso17", "iso17_default"} and dataset_id.startswith("iso17_mol"):
        cache_path = processed_dir / f"{dataset_id}.npz"
        if cache_path.exists():
            return cache_path
    matches = sorted(processed_dir.glob("iso17_mol*.npz")) if processed_dir.exists() else []
    if matches:
        return matches[0]
    return cache_iso17_default_isomer(data_root=data_root)


class ISO17Trajectory(Dataset):
    """Lazy-indexing dataset over one cached ISO17 isomer trajectory."""

    def __init__(self, dataset_id: str = "iso17_default", data_root: Optional[Path] = None):
        self.dataset_id = dataset_id
        self.npz_path = _find_cache(dataset_id, data_root=data_root)
        data = np.load(self.npz_path, allow_pickle=False)
        self.coords = data["coords"]
        self.atomic_numbers = data["nuclear_charges"]
        self.energies = data["energies"]
        self.mol_id = int(np.asarray(data["mol_id"]).reshape(-1)[0]) if "mol_id" in data.files else None
        self.n_frames = int(self.coords.shape[0])
        self.n_atoms = int(self.coords.shape[1])

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx: int) -> HeteroData:
        return atoms_to_pyg(self.coords[idx], self.atomic_numbers)

    def get_pair(self, idx: int, horizon: int = 1):
        """Return (obs_t, obs_{t+horizon}, energy_t, force_t)."""
        if idx + horizon >= self.n_frames:
            raise IndexError(f"idx+horizon={idx+horizon} >= n_frames={self.n_frames}")
        return self[idx], self[idx + horizon], float(self.energies[idx]), None

    def __repr__(self):
        return (
            f"ISO17Trajectory(dataset_id={self.dataset_id!r}, mol_id={self.mol_id}, "
            f"n_frames={self.n_frames}, n_atoms={self.n_atoms}, source={str(self.npz_path)!r})"
        )


def collect_iso17_transitions(
    dataset_id: str = "iso17_default",
    n_transitions: int = 2000,
    stride: int = 10,
    horizon: int = 1,
    seed: int = 0,
    data_root: Optional[Path] = None,
) -> List[Dict]:
    """Sample n_transitions (obs_t, obs_{t+horizon}) pairs from one ISO17 isomer."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    traj = ISO17Trajectory(dataset_id=dataset_id, data_root=data_root)
    rng = np.random.default_rng(seed)
    max_start = traj.n_frames - horizon - 1
    candidates = np.arange(0, max_start, stride)
    if n_transitions > len(candidates):
        raise ValueError(
            f"Requested {n_transitions} but only {len(candidates)} candidates "
            f"for dataset_id={dataset_id!r}, stride={stride}, horizon={horizon}"
        )
    chosen = rng.choice(candidates, size=n_transitions, replace=False)
    chosen.sort()

    transitions = []
    for frame_idx in chosen:
        obs_t, obs_next, energy, _force = traj.get_pair(int(frame_idx), horizon)
        transitions.append(
            {
                "obs": obs_t,
                "next_obs": obs_next,
                "horizon": horizon,
                "frame_idx": int(frame_idx),
                "energy": energy,
                "dataset": dataset_id,
                "molecule": dataset_id,
                "mol_id": traj.mol_id,
            }
        )
    return transitions


def load_iso17_split(
    dataset_id: str = "iso17_default",
    data_dir: Optional[Path] = None,
    n_train: int = 2000,
    n_eval: int = 200,
) -> tuple[list[HeteroData], list[HeteroData], dict]:
    """Load sequential train/eval frame splits for one ISO17 isomer.

    The first n_train transitions use frames [0, n_train]. Evaluation frames
    immediately follow that window.
    """
    traj = ISO17Trajectory(dataset_id=dataset_id, data_root=data_dir)
    train_end = n_train + 1
    eval_end = train_end + n_eval + 1
    if eval_end > traj.n_frames:
        raise ValueError(
            f"Requested {n_train} train and {n_eval} eval transitions, requiring "
            f"{eval_end} frames, but {dataset_id!r} has {traj.n_frames}"
        )
    frames_train = [traj[idx] for idx in range(train_end)]
    frames_eval = [traj[idx] for idx in range(train_end, eval_end)]
    meta = {
        "dataset_id": dataset_id,
        "mol_id": traj.mol_id,
        "n_frames": traj.n_frames,
        "n_atoms": traj.n_atoms,
        "source": str(traj.npz_path),
    }
    return frames_train, frames_eval, meta
