from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data"
SPATIAL_CUTOFF = 2.5


def _resolve_data_root(data_root: Optional[Path] = None) -> Path:
    if data_root is not None:
        return Path(data_root)
    env = os.environ.get("PARTICLE_DATA_DIR")
    if env:
        return Path(env)
    return DEFAULT_DATA_ROOT


def _dataset_path(dataset_id: str, data_root: Optional[Path] = None, split: str = "train") -> Path:
    root = _resolve_data_root(data_root)
    if dataset_id.startswith("lj_"):
        return root / "lj_raw" / f"{dataset_id}.npz"
    if dataset_id.startswith("ho_"):
        return root / "ho_raw" / f"{dataset_id}.npz"
    if dataset_id == "3bpa" or dataset_id.startswith("3bpa"):
        filename = "3bpa_test_300K.npz" if split in {"test", "distant_test"} else "3bpa_train_mixedT.npz"
        return root / "3bpa_raw" / filename
    raise ValueError(f"Unknown particle dataset_id {dataset_id!r}")


def _as_directed_edges(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (n_edges, 2), got {edges.shape}")
    edges = edges[edges[:, 0] != edges[:, 1]]
    reverse = edges[:, [1, 0]]
    return np.concatenate([edges, reverse], axis=0)


def _pair_distances(coords_t: np.ndarray, directed_edges: np.ndarray, box_size: np.ndarray | None) -> np.ndarray:
    if directed_edges.size == 0:
        return np.empty((0,), dtype=np.float32)
    delta = coords_t[directed_edges[:, 0]] - coords_t[directed_edges[:, 1]]
    if box_size is not None:
        delta -= box_size * np.rint(delta / box_size)
    return np.linalg.norm(delta, axis=1).astype(np.float32)


def _spatial_edges(coords_t: np.ndarray, cutoff: float, box_size: np.ndarray | None) -> np.ndarray:
    coords_t = np.asarray(coords_t, dtype=np.float64)
    if box_size is not None:
        box = np.asarray(box_size, dtype=np.float64).reshape(3)
        folded = np.mod(coords_t, box)
        try:
            from scipy.spatial import cKDTree

            pairs = cKDTree(folded, boxsize=box).query_pairs(cutoff, output_type="ndarray")
        except ImportError:
            delta = folded[:, None, :] - folded[None, :, :]
            delta -= box * np.rint(delta / box)
            dist = np.linalg.norm(delta, axis=-1)
            pairs = np.argwhere(np.triu((dist < cutoff) & (dist > 0.0), k=1))
    else:
        try:
            from scipy.spatial import cKDTree

            pairs = cKDTree(coords_t).query_pairs(cutoff, output_type="ndarray")
        except ImportError:
            delta = coords_t[:, None, :] - coords_t[None, :, :]
            dist = np.linalg.norm(delta, axis=-1)
            pairs = np.argwhere(np.triu((dist < cutoff) & (dist > 0.0), k=1))

    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return _as_directed_edges(pairs)


def atoms_to_pyg(
    coords_t: np.ndarray,
    atomic_numbers: np.ndarray,
    edge_pairs: np.ndarray | None = None,
    cutoff: float = SPATIAL_CUTOFF,
    box_size: np.ndarray | None = None,
) -> HeteroData:
    """Convert a particle frame to the same HeteroData schema as rMD17.

    Args:
        coords_t: (N_atoms, 3) float positions.
        atomic_numbers: (N_atoms,) int atom or particle type ids.
        edge_pairs: Optional explicit undirected edges with shape (n_edges, 2).
        cutoff: Spatial cutoff for proximity graph construction.
        box_size: Optional PBC box lengths for minimum-image distances.

    Returns:
        HeteroData with atom nodes and ("atom", "bonded", "atom") directed edges.
    """
    coords_t = np.asarray(coords_t, dtype=np.float32)
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
    if coords_t.ndim != 2 or coords_t.shape[1] != 3:
        raise ValueError(f"coords_t must have shape (N_atoms, 3), got {coords_t.shape}")
    if atomic_numbers.ndim != 1 or atomic_numbers.shape[0] != coords_t.shape[0]:
        raise ValueError(
            "atomic_numbers must have shape (N_atoms,) matching coords_t; "
            f"got {atomic_numbers.shape} vs {coords_t.shape}"
        )

    directed_edges = (
        _as_directed_edges(edge_pairs)
        if edge_pairs is not None
        else _spatial_edges(coords_t, cutoff=cutoff, box_size=box_size)
    )
    edge_attr = _pair_distances(coords_t, directed_edges, box_size=box_size)

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
    data["atom", "bonded", "atom"].edge_index = torch.from_numpy(directed_edges.T.astype(np.int64))
    data["atom", "bonded", "atom"].edge_attr = torch.from_numpy(edge_attr).unsqueeze(-1)
    return data


class ParticleTrajectory(Dataset):
    """Lazy-indexing dataset over LJ, harmonic-oscillator, or 3BPA trajectories."""

    def __init__(
        self,
        dataset_id: str,
        cutoff: float = SPATIAL_CUTOFF,
        data_root: Optional[Path] = None,
        split: str = "train",
    ):
        self.dataset_id = dataset_id
        self.cutoff = cutoff
        self.split = split
        self.npz_path = _dataset_path(dataset_id, data_root=data_root, split=split)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Missing: {self.npz_path}")

        data = np.load(self.npz_path, allow_pickle=False)
        self.coords = data["coords"]
        self.atomic_numbers = data["nuclear_charges"]
        self.energies = data["energies"]
        self.explicit_edges = data["edges"].astype(np.int64) if "edges" in data.files else None
        self.box_size = data["box_size"].astype(np.float64) if "box_size" in data.files else None
        self.n_frames = int(self.coords.shape[0])
        self.n_atoms = int(self.coords.shape[1])

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx: int) -> HeteroData:
        return atoms_to_pyg(
            self.coords[idx],
            self.atomic_numbers,
            edge_pairs=self.explicit_edges,
            cutoff=self.cutoff,
            box_size=self.box_size,
        )

    def get_pair(self, idx: int, horizon: int = 1):
        """Return (obs_t, obs_{t+horizon}, energy_t, force_t)."""
        if idx + horizon >= self.n_frames:
            raise IndexError(f"idx+horizon={idx+horizon} >= n_frames={self.n_frames}")
        obs_t = self[idx]
        obs_next = self[idx + horizon]
        return obs_t, obs_next, float(self.energies[idx]), None

    def __repr__(self):
        return (
            f"ParticleTrajectory(dataset_id={self.dataset_id!r}, split={self.split!r}, "
            f"n_frames={self.n_frames}, n_atoms={self.n_atoms}, cutoff={self.cutoff})"
        )


def collect_particle_transitions(
    dataset_id: str,
    n_transitions: int = 2000,
    stride: int = 10,
    horizon: int = 1,
    seed: int = 0,
    cutoff: float = SPATIAL_CUTOFF,
    data_root: Optional[Path] = None,
    split: str = "train",
) -> List[Dict]:
    """Sample n_transitions (obs_t, obs_{t+horizon}) pairs from a particle trajectory."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    traj = ParticleTrajectory(dataset_id, cutoff=cutoff, data_root=data_root, split=split)
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
            }
        )
    return transitions


def load_dataset_split(
    dataset_id: str,
    data_dir: Optional[Path] = None,
    n_train: int = 2000,
    n_eval: int = 200,
    cutoff: float = SPATIAL_CUTOFF,
) -> tuple[list[HeteroData], list[HeteroData], dict]:
    """Load sequential train/eval frame splits for non-rMD17 particle datasets.

    Args:
        dataset_id: Dataset identifier such as "lj_N64", "ho_random", or "3bpa".
        data_dir: Root data directory containing lj_raw/, ho_raw/, and 3bpa_raw/.
        n_train: Number of training transitions, requiring n_train+1 frames.
        n_eval: Number of evaluation transitions after the train window.
        cutoff: Spatial cutoff used when no explicit edges are present.

    Returns:
        (frames_train, frames_eval, dataset_meta), where frame lists contain
        HeteroData objects in the rMD17 atom/bonded schema.
    """
    traj = ParticleTrajectory(dataset_id, cutoff=cutoff, data_root=data_dir)
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
        "n_frames": traj.n_frames,
        "n_atoms": traj.n_atoms,
        "source": str(traj.npz_path),
        "has_explicit_edges": traj.explicit_edges is not None,
        "has_pbc": traj.box_size is not None,
    }
    return frames_train, frames_eval, meta
