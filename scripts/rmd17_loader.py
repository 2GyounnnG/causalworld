import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


DATA_ROOT = Path("/home/user/projects/causalworld/data/rmd17_raw/rmd17/npz_data")
MOLECULES = [
    "aspirin",
    "azobenzene",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "paracetamol",
    "salicylic",
    "toluene",
    "uracil",
]


def atoms_to_pyg(
    coords_t: np.ndarray,
    atomic_numbers: np.ndarray,
    cutoff: float = 5.0,
) -> HeteroData:
    """Convert a single frame to a HeteroData graph.

    Args:
        coords_t: (N_atoms, 3) float, Angstrom
        atomic_numbers: (N_atoms,) int
        cutoff: distance threshold in Angstrom for edges

    Returns HeteroData with:
        data["atom"].x : (N_atoms, 4)  [atomic_number, x, y, z]
        data["atom"].pos : (N_atoms, 3)
        data["atom"].atomic_number : (N_atoms,) int
        data["atom", "bonded", "atom"].edge_index : (2, E) long
        data["atom", "bonded", "atom"].edge_attr  : (E, 1) distance
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


class RMD17Trajectory(Dataset):
    """Lazy-indexing dataset over a single molecule's trajectory."""

    def __init__(self, molecule: str, cutoff: float = 5.0, data_root: Optional[Path] = None):
        if molecule not in MOLECULES:
            raise ValueError(f"Unknown molecule {molecule}. Valid: {MOLECULES}")

        root = data_root or DATA_ROOT
        self.npz_path = root / f"rmd17_{molecule}.npz"
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Missing: {self.npz_path}")

        data = np.load(self.npz_path)
        self.atomic_numbers = data["nuclear_charges"]
        self.coords = data["coords"]
        self.energies = data["energies"]
        self.forces = data["forces"]
        self.molecule = molecule
        self.cutoff = cutoff
        self.n_frames = int(self.coords.shape[0])
        self.n_atoms = int(self.coords.shape[1])

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx: int) -> HeteroData:
        return atoms_to_pyg(self.coords[idx], self.atomic_numbers, self.cutoff)

    def get_pair(self, idx: int, horizon: int = 1):
        """Return (obs_t, obs_{t+horizon}, energy_t, force_t)."""
        if idx + horizon >= self.n_frames:
            raise IndexError(f"idx+horizon={idx+horizon} >= n_frames={self.n_frames}")
        obs_t = self[idx]
        obs_next = self[idx + horizon]
        return obs_t, obs_next, float(self.energies[idx]), self.forces[idx]

    def __repr__(self):
        return (
            f"RMD17Trajectory(molecule={self.molecule!r}, "
            f"n_frames={self.n_frames}, n_atoms={self.n_atoms}, "
            f"cutoff={self.cutoff})"
        )


def collect_rmd17_transitions(
    molecule: str,
    n_transitions: int = 2000,
    stride: int = 10,
    horizon: int = 1,
    seed: int = 0,
    cutoff: float = 5.0,
) -> List[Dict]:
    """Sample n_transitions (obs_t, obs_{t+horizon}) pairs from trajectory.

    stride: only sample starting indices that are multiples of `stride`,
    to reduce temporal autocorrelation (rMD17 authors' warning).
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    traj = RMD17Trajectory(molecule, cutoff=cutoff)
    rng = np.random.default_rng(seed)
    max_start = traj.n_frames - horizon - 1
    candidates = np.arange(0, max_start, stride)
    if n_transitions > len(candidates):
        raise ValueError(
            f"Requested {n_transitions} but only {len(candidates)} candidates at stride={stride}"
        )
    chosen = rng.choice(candidates, size=n_transitions, replace=False)
    chosen.sort()

    transitions = []
    for i in chosen:
        obs_t, obs_next, energy, _force = traj.get_pair(int(i), horizon)
        transitions.append(
            {
                "obs": obs_t,
                "next_obs": obs_next,
                "horizon": horizon,
                "frame_idx": int(i),
                "energy": energy,
                "molecule": molecule,
            }
        )
    return transitions


def _smoke_test():
    print("=== rMD17 loader smoke test ===")

    if not DATA_ROOT.exists():
        print(f"ERROR: {DATA_ROOT} not found. Did you extract rmd17.tar.bz2?")
        return

    traj = RMD17Trajectory("aspirin")
    print(f"  loaded: {traj}")

    data = traj[0]
    n_atoms = data["atom"].x.shape[0]
    n_edges = data["atom", "bonded", "atom"].edge_index.shape[1]
    print(f"  frame 0: {n_atoms} atoms, {n_edges} edges (cutoff={traj.cutoff} Angstrom)")

    obs_t, obs_next, energy, force = traj.get_pair(0, horizon=1)
    _ = obs_t, obs_next
    print(f"  pair: energy_t={energy:.2f} kcal/mol, force_mag={np.linalg.norm(force):.2f}")

    transitions = collect_rmd17_transitions(
        "aspirin",
        n_transitions=10,
        stride=100,
        horizon=1,
        seed=42,
    )
    print(f"  collected {len(transitions)} transitions from aspirin")
    print(f"  first transition frame_idx: {transitions[0]['frame_idx']}")

    azo = RMD17Trajectory("azobenzene")
    print(f"  azobenzene: {azo.n_frames} frames (should be 99988)")

    print("smoke test passed")


if __name__ == "__main__":
    _smoke_test()
