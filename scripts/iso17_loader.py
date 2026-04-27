from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "iso17_raw"
DOWNLOAD_URL = "http://www.quantum-machine.org/datasets/iso17.tar.gz"
SPLITS = ("reference", "test_within", "test_other")

POSITION_KEYS = ("positions", "coords", "coordinates", "R", "x")
ATOMIC_NUMBER_KEYS = ("atomic_numbers", "nuclear_charges", "numbers", "z", "Z")
ENERGY_KEYS = ("energies", "energy", "E", "e", "total_energy")
FORCE_KEYS = ("forces", "F", "f")
ISOMER_KEYS = ("isomer", "isomer_id", "isomer_ids", "molecule", "molecule_id", "mol_id")


def atoms_to_pyg(
    coords_t: np.ndarray,
    atomic_numbers: np.ndarray,
    cutoff: float = 5.0,
) -> HeteroData:
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

    n_atoms = coords_t.shape[0]
    diff = coords_t[:, None, :] - coords_t[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    mask = (dist < cutoff) & (~np.eye(n_atoms, dtype=bool))
    src, dst = np.where(mask)
    edge_attr = dist[src, dst].astype(np.float32)

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
    data["atom"].x = atom_x
    data["atom", "bonded", "atom"].edge_index = torch.from_numpy(
        np.stack([src, dst], axis=0).astype(np.int64)
    )
    data["atom", "bonded", "atom"].edge_attr = torch.from_numpy(edge_attr).unsqueeze(-1)
    return data


def _first_npz_key(data: np.lib.npyio.NpzFile, candidates: tuple[str, ...]) -> str:
    for key in candidates:
        if key in data.files:
            return key
    raise KeyError(f"None of {candidates} found in npz keys: {data.files}")


def _decode_label(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)


def _find_source(root: Path, split: str) -> Path:
    if root.is_file():
        return root

    candidates = [
        root / f"{split}.npz",
        root / f"iso17_{split}.npz",
        root / "iso17" / f"{split}.npz",
        root / "iso17" / f"iso17_{split}.npz",
        root / f"{split}.db",
        root / f"iso17_{split}.db",
        root / "iso17" / f"{split}.db",
        root / "iso17" / f"iso17_{split}.db",
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = sorted(root.rglob(f"*{split}*.npz")) + sorted(root.rglob(f"*{split}*.db"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find ISO17 split {split!r} under {root}. "
        f"Expected files like reference.db, test_within.db, or {split}.npz. "
        f"Download/extract ISO17 from {DOWNLOAD_URL} into {DATA_ROOT}."
    )


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


class ISO17Trajectory(Dataset):
    """Lazy-indexing dataset over an ISO17 split.

    Supports the ASE database files distributed with ISO17 and npz mirrors
    with common key names such as positions/R, atomic_numbers/Z, energies/E,
    and forces/F.
    """

    def __init__(
        self,
        split: str = "reference",
        isomer: str | None = "all",
        cutoff: float = 5.0,
        data_root: Optional[Path] = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"Unknown ISO17 split {split!r}. Valid: {SPLITS}")

        root = Path(data_root) if data_root is not None else DATA_ROOT
        self.source_path = _find_source(root, split)
        self.split = split
        self.isomer = "all" if isomer in (None, "", "all") else str(isomer)
        self.cutoff = cutoff
        self.source_type = self.source_path.suffix.lower().lstrip(".")
        self._db = None
        self._npz = None
        self._isomer_cache: dict[int, str | None] = {}

        if self.source_type == "npz":
            self._load_npz()
        elif self.source_type == "db":
            self._load_db()
        else:
            raise ValueError(f"Unsupported ISO17 source type: {self.source_path}")

    def _load_npz(self) -> None:
        data = np.load(self.source_path, allow_pickle=True, mmap_mode="r")
        self._npz = data
        self.coords = data[_first_npz_key(data, POSITION_KEYS)]
        self.atomic_numbers = data[_first_npz_key(data, ATOMIC_NUMBER_KEYS)]
        self.energies = data[_first_npz_key(data, ENERGY_KEYS)]
        self.forces = data[_first_npz_key(data, FORCE_KEYS)]

        isomer_key = next((key for key in ISOMER_KEYS if key in data.files), None)
        self.isomers = data[isomer_key] if isomer_key is not None else None

        if self.coords.ndim != 3 or self.coords.shape[-1] != 3:
            raise ValueError(f"positions must have shape (n_frames, n_atoms, 3), got {self.coords.shape}")
        if self.atomic_numbers.ndim not in {1, 2}:
            raise ValueError(
                "atomic_numbers must have shape (n_atoms,) or (n_frames, n_atoms), "
                f"got {self.atomic_numbers.shape}"
            )

        frame_indices = np.arange(int(self.coords.shape[0]))
        if self.isomer != "all":
            if self.isomers is None:
                raise ValueError(f"Cannot filter isomer={self.isomer!r}; no isomer key found in npz")
            labels = np.asarray([_decode_label(value) for value in self.isomers])
            frame_indices = frame_indices[labels == self.isomer]
            if frame_indices.size == 0:
                raise ValueError(f"No ISO17 frames found for isomer={self.isomer!r}")

        self.frame_indices = frame_indices.astype(int).tolist()
        self.n_frames = len(self.frame_indices)
        self.n_atoms = int(self.coords.shape[1])

    def _load_db(self) -> None:
        try:
            from ase.db import connect
        except ImportError as exc:
            raise ImportError(
                "Reading ISO17 .db files requires ASE. Install ase or provide an npz mirror."
            ) from exc

        self._db = connect(self.source_path)
        try:
            n_rows = int(self._db.count())
        except Exception:
            n_rows = int(len(self._db))

        if self.isomer == "all":
            self.frame_ids = list(range(1, n_rows + 1))
        else:
            self.frame_ids = []
            for row in self._db.select():
                label = self._row_isomer(row)
                if label == self.isomer:
                    self.frame_ids.append(int(row.id))
            if not self.frame_ids:
                raise ValueError(f"No ISO17 rows found for isomer={self.isomer!r}")

        self.n_frames = len(self.frame_ids)
        first = self._read_db_frame(0)
        self.n_atoms = int(first["positions"].shape[0])

    def _row_isomer(self, row: Any) -> str | None:
        try:
            value = _row_value(row, ISOMER_KEYS)
        except KeyError:
            return None
        return _decode_label(value)

    def _read_db_frame(self, logical_idx: int) -> dict[str, Any]:
        assert self._db is not None
        row_id = self.frame_ids[logical_idx]
        row = self._db.get(id=row_id)
        atoms = row.toatoms()

        try:
            energy = _row_value(row, ENERGY_KEYS)
        except KeyError:
            energy = float("nan")
        try:
            forces = _row_value(row, FORCE_KEYS)
        except KeyError:
            forces = atoms.get_forces()

        return {
            "positions": atoms.get_positions(),
            "atomic_numbers": atoms.get_atomic_numbers(),
            "energy": float(np.asarray(energy).reshape(-1)[0]),
            "forces": np.asarray(forces, dtype=np.float32),
            "isomer": self._row_isomer(row),
        }

    def _read_npz_frame(self, logical_idx: int) -> dict[str, Any]:
        frame_idx = self.frame_indices[logical_idx]
        atomic_numbers = (
            self.atomic_numbers[frame_idx]
            if self.atomic_numbers.ndim == 2
            else self.atomic_numbers
        )
        label = None
        if self.isomers is not None:
            label = _decode_label(self.isomers[frame_idx])
        return {
            "positions": self.coords[frame_idx],
            "atomic_numbers": atomic_numbers,
            "energy": float(np.asarray(self.energies[frame_idx]).reshape(-1)[0]),
            "forces": np.asarray(self.forces[frame_idx], dtype=np.float32),
            "isomer": label,
        }

    def _read_frame(self, logical_idx: int) -> dict[str, Any]:
        if logical_idx < 0 or logical_idx >= self.n_frames:
            raise IndexError(f"idx={logical_idx} outside [0, {self.n_frames})")
        if self.source_type == "npz":
            return self._read_npz_frame(logical_idx)
        return self._read_db_frame(logical_idx)

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx: int) -> HeteroData:
        frame = self._read_frame(idx)
        return atoms_to_pyg(frame["positions"], frame["atomic_numbers"], self.cutoff)

    def _cached_isomer(self, idx: int) -> str | None:
        if idx not in self._isomer_cache:
            self._isomer_cache[idx] = self._read_frame(idx)["isomer"]
        return self._isomer_cache[idx]

    def can_pair(self, idx: int, horizon: int) -> bool:
        """Return whether idx and idx+horizon can be treated as one trajectory pair."""
        if idx + horizon >= self.n_frames:
            return False
        if self.isomer != "all":
            return True
        start_isomer = self._cached_isomer(idx)
        target_isomer = self._cached_isomer(idx + horizon)
        if start_isomer is None or target_isomer is None:
            return True
        return start_isomer == target_isomer

    def get_pair(self, idx: int, horizon: int = 1):
        """Return (obs_t, obs_{t+horizon}, energy_t, force_t)."""
        if not self.can_pair(idx, horizon):
            raise IndexError(f"idx={idx} and idx+horizon={idx+horizon} cross ISO17 isomer boundary")
        frame = self._read_frame(idx)
        obs_t = atoms_to_pyg(frame["positions"], frame["atomic_numbers"], self.cutoff)
        obs_next = self[idx + horizon]
        return obs_t, obs_next, frame["energy"], frame["forces"]

    def __repr__(self):
        return (
            f"ISO17Trajectory(split={self.split!r}, isomer={self.isomer!r}, "
            f"n_frames={self.n_frames}, n_atoms={self.n_atoms}, "
            f"source={str(self.source_path)!r}, cutoff={self.cutoff})"
        )


def collect_iso17_transitions(
    split: str = "reference",
    isomer: str | None = "all",
    n_transitions: int = 2000,
    stride: int = 10,
    horizon: int = 1,
    seed: int = 0,
    cutoff: float = 5.0,
    data_root: Optional[Path] = None,
) -> List[Dict]:
    """Sample n_transitions (obs_t, obs_{t+horizon}) pairs from ISO17."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    traj = ISO17Trajectory(split=split, isomer=isomer, cutoff=cutoff, data_root=data_root)
    rng = np.random.default_rng(seed)
    max_start = traj.n_frames - horizon - 1
    candidates = np.asarray(
        [idx for idx in range(0, max_start, stride) if traj.can_pair(idx, horizon)],
        dtype=int,
    )
    if n_transitions > len(candidates):
        raise ValueError(
            f"Requested {n_transitions} but only {len(candidates)} candidates "
            f"for split={split!r}, isomer={traj.isomer!r}, stride={stride}, horizon={horizon}"
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
                "dataset": "iso17",
                "molecule": "iso17",
                "split": split,
                "isomer": traj.isomer,
            }
        )
    return transitions


def _smoke_test():
    print("=== ISO17 loader smoke test ===")

    if not DATA_ROOT.exists():
        print(f"ERROR: {DATA_ROOT} not found. Download/extract ISO17 from {DOWNLOAD_URL}.")
        return

    try:
        traj = ISO17Trajectory("reference")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return

    print(f"  loaded: {traj}")
    data = traj[0]
    n_atoms = data["atom"].x.shape[0]
    n_edges = data["atom", "bonded", "atom"].edge_index.shape[1]
    print(f"  frame 0: {n_atoms} atoms, {n_edges} edges (cutoff={traj.cutoff} Angstrom)")

    obs_t, obs_next, energy, force = traj.get_pair(0, horizon=1)
    _ = obs_t, obs_next
    print(f"  pair: energy_t={energy:.6f}, force_mag={np.linalg.norm(force):.6f}")

    n = min(10, max(1, traj.n_frames // 100))
    transitions = collect_iso17_transitions(
        "reference",
        n_transitions=n,
        stride=100,
        horizon=1,
        seed=42,
    )
    print(f"  collected {len(transitions)} transitions from ISO17 reference")
    print(f"  first transition frame_idx: {transitions[0]['frame_idx']}")
    print("smoke test passed")


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    _smoke_test()
