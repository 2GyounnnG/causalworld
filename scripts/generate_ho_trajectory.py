from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "ho_raw"

N_NODES = 64
TEMPERATURE = 0.1
SPRING_K = 1.0
DT = 0.01
N_STEPS = 5000
SAMPLE_EVERY = 5
TOPOLOGIES = ("random", "lattice", "scalefree")


def build_graph(topology: str, seed: int) -> nx.Graph:
    if topology == "random":
        graph = nx.erdos_renyi_graph(N_NODES, 0.05, seed=seed)
        if not nx.is_connected(graph):
            components = [list(component) for component in nx.connected_components(graph)]
            for left, right in zip(components, components[1:]):
                graph.add_edge(left[0], right[0])
        return graph
    if topology == "lattice":
        graph = nx.grid_2d_graph(8, 8)
        return nx.convert_node_labels_to_integers(graph, ordering="sorted")
    if topology == "scalefree":
        return nx.barabasi_albert_graph(N_NODES, 2, seed=seed)
    raise ValueError(f"Unknown topology {topology!r}")


def adjacency_matrix(graph: nx.Graph) -> np.ndarray:
    return nx.to_numpy_array(graph, nodelist=range(N_NODES), dtype=np.float64)


def spring_forces(positions: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    laplacian = np.diag(degree) - adjacency
    return -SPRING_K * laplacian @ positions


def total_energy(positions: np.ndarray, velocities: np.ndarray, edges: np.ndarray) -> float:
    kinetic = 0.5 * float(np.sum(velocities * velocities))
    if edges.size == 0:
        return kinetic
    diff = positions[edges[:, 0]] - positions[edges[:, 1]]
    potential = 0.5 * SPRING_K * float(np.sum(diff * diff))
    return kinetic + potential


def generate_topology(topology: str, seed: int = 0, output_dir: Path = OUTPUT_DIR) -> Path:
    rng = np.random.default_rng(seed)
    graph = build_graph(topology, seed)
    adjacency = adjacency_matrix(graph)
    edges = np.asarray(sorted((min(i, j), max(i, j)) for i, j in graph.edges()), dtype=np.int32)

    positions = rng.uniform(0.0, 1.0, size=(N_NODES, 3))
    velocities = rng.normal(0.0, np.sqrt(TEMPERATURE), size=(N_NODES, 3))
    velocities -= velocities.mean(axis=0, keepdims=True)
    forces = spring_forces(positions, adjacency)

    n_frames = N_STEPS // SAMPLE_EVERY
    coords = np.empty((n_frames, N_NODES, 3), dtype=np.float32)
    energies = np.empty((n_frames,), dtype=np.float32)
    frame = 0

    print(f"{topology}: {N_NODES} nodes, {edges.shape[0]} edges", flush=True)
    for step in range(1, N_STEPS + 1):
        positions = positions + velocities * DT + 0.5 * forces * DT * DT
        new_forces = spring_forces(positions, adjacency)
        velocities = velocities + 0.5 * (forces + new_forces) * DT
        forces = new_forces
        if step % SAMPLE_EVERY == 0:
            coords[frame] = positions.astype(np.float32)
            energies[frame] = np.float32(total_energy(positions, velocities, edges))
            frame += 1
        if step % 1000 == 0:
            print(f"{topology}: step {step}/{N_STEPS}, E={total_energy(positions, velocities, edges):.6f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"ho_{topology}.npz"
    np.savez_compressed(
        path,
        coords=coords,
        nuclear_charges=np.ones((N_NODES,), dtype=np.int32),
        energies=energies,
        edges=edges,
    )
    print(f"{topology}: wrote {path}", flush=True)
    return path


def main() -> None:
    for index, topology in enumerate(TOPOLOGIES):
        generate_topology(topology, seed=1729 + index)


if __name__ == "__main__":
    main()
