"""World model components for causal hypergraph rewriting experiments.

Encoders, transition network, regularizers, and WorldModel wrapper.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool


# Mish activation
class Mish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(F.softplus(x))


def _storage_batch_vector(data: HeteroData, node_type: str) -> Optional[Tensor]:
    batch = getattr(data[node_type], "batch", None)
    return batch if torch.is_tensor(batch) else None


def _num_graphs(data: HeteroData, *batch_vectors: Optional[Tensor]) -> int:
    num_graphs = getattr(data, "num_graphs", None)
    if isinstance(num_graphs, int) and num_graphs > 0:
        return num_graphs
    for batch in batch_vectors:
        if batch is not None and batch.numel() > 0:
            return int(batch.max().item()) + 1
    return 1


# 1. HypergraphEncoder
class HypergraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.node_proj = nn.Linear(1, hidden_dim)
        self.edge_proj = nn.Linear(2, hidden_dim)
        self.conv1 = HeteroConv(
            {
                ("node", "member_of", "hyperedge"): SAGEConv(
                    (hidden_dim, hidden_dim), hidden_dim
                ),
                ("hyperedge", "has_member", "node"): SAGEConv(
                    (hidden_dim, hidden_dim), hidden_dim
                ),
            },
            aggr="mean",
        )
        self.conv2 = HeteroConv(
            {
                ("node", "member_of", "hyperedge"): SAGEConv(
                    (hidden_dim, hidden_dim), hidden_dim
                ),
                ("hyperedge", "has_member", "node"): SAGEConv(
                    (hidden_dim, hidden_dim), hidden_dim
                ),
            },
            aggr="mean",
        )
        self.out = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, data: HeteroData) -> Tensor:
        node_x = data["node"].x
        edge_x = data["hyperedge"].x
        node_batch = _storage_batch_vector(data, "node")
        edge_batch = _storage_batch_vector(data, "hyperedge")
        is_batched = node_batch is not None or edge_batch is not None
        num_graphs = _num_graphs(data, node_batch, edge_batch)

        if node_x.shape[0] == 0 or edge_x.shape[0] == 0:
            device = next(self.parameters()).device
            if is_batched:
                return torch.zeros(num_graphs, self.latent_dim, device=device)
            return torch.zeros(self.latent_dim, device=device)

        ei_n2e = data["node", "member_of", "hyperedge"].edge_index
        ei_e2n = data["hyperedge", "has_member", "node"].edge_index

        x_dict = {
            "node": F.relu(self.node_proj(node_x)),
            "hyperedge": F.relu(self.edge_proj(edge_x)),
        }
        ei_dict = {
            ("node", "member_of", "hyperedge"): ei_n2e,
            ("hyperedge", "has_member", "node"): ei_e2n,
        }

        x_dict = {k: F.relu(v) for k, v in self.conv1(x_dict, ei_dict).items()}
        x_dict = {k: F.relu(v) for k, v in self.conv2(x_dict, ei_dict).items()}

        if node_batch is None:
            node_batch = torch.zeros(
                x_dict["node"].shape[0], dtype=torch.long, device=x_dict["node"].device
            )
        if edge_batch is None:
            edge_batch = torch.zeros(
                x_dict["hyperedge"].shape[0],
                dtype=torch.long,
                device=x_dict["hyperedge"].device,
            )
        node_pool = global_mean_pool(x_dict["node"], node_batch, size=num_graphs)
        edge_pool = global_mean_pool(x_dict["hyperedge"], edge_batch, size=num_graphs)

        combined = torch.cat([node_pool, edge_pool], dim=-1)
        encoded = self.out(combined)
        return encoded if is_batched else encoded.squeeze(0)

    def __repr__(self) -> str:
        return f"HypergraphEncoder(hidden={self.hidden_dim}, latent={self.latent_dim})"


# 2. FlatMLPEncoder
class FlatMLPEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def _extract_features(self, data: HeteroData) -> Tensor:
        node_x = data["node"].x
        edge_x = data["hyperedge"].x
        num_nodes = float(node_x.shape[0])
        num_edges = float(edge_x.shape[0])

        if num_edges > 0:
            arities = edge_x[:, 0]
            mean_arity = arities.mean().item()
            max_arity = arities.max().item()
            origins = edge_x[:, 1]
            known = (origins >= 0).float()
            frac_known = known.mean().item()
            num_events = (
                float(origins[origins >= 0].max().item() + 1)
                if known.sum() > 0
                else 0.0
            )
        else:
            mean_arity = 0.0
            max_arity = 0.0
            frac_known = 0.0
            num_events = 0.0

        features = [
            num_nodes,
            num_edges,
            mean_arity,
            max_arity,
            frac_known,
            num_events,
        ]
        return torch.tensor(features, dtype=torch.float, device=node_x.device)

    def _extract_batched_features(self, data: HeteroData) -> Tensor:
        node_x = data["node"].x
        edge_x = data["hyperedge"].x
        node_batch = _storage_batch_vector(data, "node")
        edge_batch = _storage_batch_vector(data, "hyperedge")
        num_graphs = _num_graphs(data, node_batch, edge_batch)

        if node_batch is None:
            node_batch = torch.zeros(
                node_x.shape[0], dtype=torch.long, device=node_x.device
            )
        if edge_batch is None:
            edge_batch = torch.zeros(
                edge_x.shape[0], dtype=torch.long, device=edge_x.device
            )

        dtype = node_x.dtype
        num_nodes = torch.bincount(node_batch, minlength=num_graphs).to(dtype)
        num_edges = torch.bincount(edge_batch, minlength=num_graphs).to(dtype)

        mean_arity = edge_x.new_zeros(num_graphs)
        max_arity = edge_x.new_zeros(num_graphs)
        frac_known = edge_x.new_zeros(num_graphs)
        num_events = edge_x.new_zeros(num_graphs)
        if edge_x.shape[0] > 0:
            arities = edge_x[:, 0]
            origins = edge_x[:, 1]
            denom = num_edges.clamp_min(1.0)
            mean_arity = mean_arity.scatter_add(0, edge_batch, arities) / denom
            max_arity = max_arity.scatter_reduce(
                0, edge_batch, arities, reduce="amax", include_self=True
            )
            known = (origins >= 0).to(dtype)
            frac_known = frac_known.scatter_add(0, edge_batch, known) / denom
            event_counts = torch.where(
                origins >= 0, origins + 1.0, torch.zeros_like(origins)
            )
            num_events = num_events.scatter_reduce(
                0, edge_batch, event_counts, reduce="amax", include_self=True
            )

        return torch.stack(
            [num_nodes, num_edges, mean_arity, max_arity, frac_known, num_events],
            dim=-1,
        )

    def forward(self, data: HeteroData) -> Tensor:
        is_batched = (
            _storage_batch_vector(data, "node") is not None
            or _storage_batch_vector(data, "hyperedge") is not None
        )
        if is_batched:
            return self.mlp(self._extract_batched_features(data))
        return self.mlp(self._extract_features(data))

    def __repr__(self) -> str:
        return f"FlatMLPEncoder(hidden={self.hidden_dim}, latent={self.latent_dim})"


# 3. MLPEncoder
class MLPEncoder(nn.Module):
    def __init__(
        self,
        n_atoms: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
    ):
        super().__init__()
        if n_atoms <= 0:
            raise ValueError(f"n_atoms must be positive, got {n_atoms}")
        self.n_atoms = int(n_atoms)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.n_atoms * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, data: HeteroData) -> Tensor:
        pos = data["atom"].pos
        atom_batch = _storage_batch_vector(data, "atom")
        if atom_batch is not None:
            num_graphs = _num_graphs(data, atom_batch)
            pos = pos.reshape(num_graphs, self.n_atoms, 3)
            return self.mlp(pos.flatten(start_dim=1))
        assert pos.shape == (
            self.n_atoms,
            3,
        ), f"expected atom positions with shape ({self.n_atoms}, 3), got {tuple(pos.shape)}"
        return self.mlp(pos.flatten())

    def __repr__(self) -> str:
        return (
            f"MLPEncoder(n_atoms={self.n_atoms}, hidden={self.hidden_dim}, "
            f"latent={self.latent_dim})"
        )


# 4. CausalGraphEncoder
class CausalGraphEncoder(nn.Module):
    """Encode event-causal graph statistics into a latent vector."""

    def __init__(self, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(8, hidden_dim),
            Mish(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def _extract_features(self, graph: nx.DiGraph) -> Tensor:
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        if num_nodes == 0:
            return torch.zeros(8, dtype=torch.float)

        in_degrees = np.array([degree for _, degree in graph.in_degree()], dtype=float)
        out_degrees = np.array([degree for _, degree in graph.out_degree()], dtype=float)
        roots = float(np.sum(in_degrees == 0))
        leaves = float(np.sum(out_degrees == 0))
        density = nx.density(graph)

        if nx.is_directed_acyclic_graph(graph):
            longest_path = float(nx.dag_longest_path_length(graph))
        else:
            longest_path = 0.0

        features = [
            float(num_nodes),
            float(num_edges),
            float(in_degrees.mean()),
            float(out_degrees.mean()),
            roots,
            leaves,
            float(density),
            longest_path,
        ]
        return torch.tensor(features, dtype=torch.float)

    def forward(self, graph: nx.DiGraph) -> Tensor:
        device = next(self.parameters()).device
        return self.mlp(self._extract_features(graph).to(device))

    def __repr__(self) -> str:
        return f"CausalGraphEncoder(hidden={self.hidden_dim}, latent={self.latent_dim})"


# 5. TransitionNetwork
class TransitionNetwork(nn.Module):
    """Predict the next latent state and reward from latent state plus action."""

    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 1,
        hidden_dim: int = 128,
        residual: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            Mish(),
        )
        self.delta_head = nn.Linear(hidden_dim, latent_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)

    def _format_action(self, action: Tensor | int | float, latent: Tensor) -> Tensor:
        if not torch.is_tensor(action):
            action = torch.tensor([float(action)], dtype=latent.dtype, device=latent.device)
        else:
            action = action.to(device=latent.device, dtype=latent.dtype)

        if action.dim() == 0:
            action = action.view(1)
        if action.dim() == 1 and self.action_dim == 1:
            action = action.view(-1, 1)
        if action.dim() == 1 and self.action_dim > 1:
            action = action.unsqueeze(0)
        return action

    def forward(
        self, latent: Tensor, action: Tensor | int | float
    ) -> Tuple[Tensor, Tensor, Tensor]:
        squeeze = latent.dim() == 1
        if squeeze:
            latent = latent.unsqueeze(0)

        action_tensor = self._format_action(action, latent)
        if action_tensor.shape[0] == 1 and latent.shape[0] > 1:
            action_tensor = action_tensor.expand(latent.shape[0], -1)
        if action_tensor.shape != (latent.shape[0], self.action_dim):
            raise ValueError(
                "action must be broadcastable to "
                f"({latent.shape[0]}, {self.action_dim}), got {tuple(action_tensor.shape)}"
            )

        hidden = self.trunk(torch.cat([latent, action_tensor], dim=-1))
        delta = self.delta_head(hidden)
        next_latent = latent + delta if self.residual else delta
        reward = self.reward_head(hidden).squeeze(-1)
        done_logit = self.done_head(hidden).squeeze(-1)

        if squeeze:
            return next_latent.squeeze(0), reward.squeeze(0), done_logit.squeeze(0)
        return next_latent, reward, done_logit

    def __repr__(self) -> str:
        return (
            "TransitionNetwork("
            f"latent={self.latent_dim}, action={self.action_dim}, hidden={self.hidden_dim})"
        )


# 6. Regularizers
def latent_l2_regularizer(latent: Tensor) -> Tensor:
    """Encourage compact latent codes."""

    return latent.pow(2).mean()


def temporal_smoothness_regularizer(latent: Tensor, next_latent: Tensor) -> Tensor:
    """Encourage small latent moves between adjacent states."""

    return (next_latent - latent).pow(2).mean()


def variance_regularizer(latents: Tensor, eps: float = 1e-4) -> Tensor:
    """Avoid collapsed representations in batches of latent states."""

    if latents.dim() == 1 or latents.shape[0] < 2:
        return latents.new_tensor(0.0)
    std = torch.sqrt(latents.var(dim=0, unbiased=False) + eps)
    return F.relu(1.0 - std).mean()


def causal_sparsity_regularizer(graph: nx.DiGraph) -> Tensor:
    """Penalize dense event-causal graphs."""

    num_nodes = graph.number_of_nodes()
    if num_nodes <= 1:
        return torch.tensor(0.0)
    max_edges = num_nodes * (num_nodes - 1)
    return torch.tensor(graph.number_of_edges() / max_edges, dtype=torch.float)


def euclidean_cov_penalty(z_batch: Tensor) -> Tensor:
    """Frobenius norm of (Cov(z) - I) over batch (B, D).

    Implements the LeJEPA-style Euclidean isotropy penalty used as the
    baseline prior in the research proposal.
    """

    if z_batch.dim() == 1:
        z_batch = z_batch.unsqueeze(0)
    B, D = z_batch.shape
    if B < 2:
        return z_batch.new_tensor(0.0)
    z_c = z_batch - z_batch.mean(dim=0, keepdim=True)
    C = (z_c.T @ z_c) / (B - 1)
    I = torch.eye(D, device=z_batch.device, dtype=z_batch.dtype)
    return torch.norm(C - I, p="fro")


def spectral_laplacian_penalty(z_batch: Tensor, L: Tensor) -> Tensor:
    """Mean of z_i^T L z_i over batch (B, D). L is (D, D).

    Implements the Laplacian-quadratic spectral prior from the proposal.
    """

    if z_batch.dim() == 1:
        z_batch = z_batch.unsqueeze(0)
    L = L.to(device=z_batch.device, dtype=z_batch.dtype)
    if L.dim() == 3:
        if L.shape[0] == 1 and z_batch.shape[0] > 1:
            L = L.expand(z_batch.shape[0], -1, -1)
        if L.shape[0] != z_batch.shape[0]:
            raise ValueError(
                "batched laplacian must have one matrix per latent vector, "
                f"got {tuple(L.shape)} for z {tuple(z_batch.shape)}"
            )
        return torch.einsum("bi,bij,bj->b", z_batch, L, z_batch).mean()
    return (z_batch @ L * z_batch).sum(dim=-1).mean()


def variance_only_penalty(z_batch: Tensor, gamma: float = 1.0) -> Tensor:
    if z_batch.dim() == 1:
        z_batch = z_batch.unsqueeze(0)
    sigma = torch.sqrt(z_batch.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(gamma - sigma).pow(2).mean()


def sigreg_gauss_penalty(
    z_batch: Tensor,
    num_slices: int = 8,
    sigma: float = 1.0,
) -> Tensor:
    if z_batch.dim() == 1:
        z_batch = z_batch.unsqueeze(0)
    B, D = z_batch.shape
    if B == 0 or num_slices <= 0:
        return z_batch.new_tensor(0.0)

    directions = torch.randn(
        num_slices,
        D,
        device=z_batch.device,
        dtype=z_batch.dtype,
    )
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    projected = z_batch @ directions.T
    t = torch.linspace(-3.0, 3.0, 16, device=z_batch.device, dtype=z_batch.dtype)
    target_cf = torch.exp(-0.5 * t.pow(2))
    weights = torch.exp(-t.pow(2) / (sigma ** 2))
    dt = t[1] - t[0] if t.numel() > 1 else t.new_tensor(1.0)

    losses = []
    target_cf_complex = target_cf.to(torch.complex64)
    for s in projected.T:
        phase = t.unsqueeze(0) * s.unsqueeze(1)
        ecf = torch.exp(1j * phase.to(torch.complex64)).mean(dim=0)
        diff = ecf - target_cf_complex
        slice_loss = (diff.real.pow(2) + diff.imag.pow(2)) * weights
        losses.append(slice_loss.sum() * dt)
    return torch.stack(losses).mean()


def permuted_laplacian(
    L: Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    if L.dim() not in {2, 3}:
        raise ValueError(
            f"laplacian must be 2D or batched 3D, got shape {tuple(L.shape)}"
        )
    latent_dim = L.shape[-1]
    if generator is not None:
        sample_device = generator.device
    else:
        sample_device = torch.device("cpu")
    permutation = torch.randperm(
        latent_dim, device=sample_device, generator=generator
    ).to(L.device)
    if L.dim() == 3:
        return L.index_select(1, permutation).index_select(2, permutation)
    return L.index_select(0, permutation).index_select(1, permutation)


def random_laplacian(
    latent_dim: int,
    edge_density: float,
    device,
    dtype,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    if generator is not None:
        sample_device = generator.device
    else:
        sample_device = torch.device("cpu")
    samples = torch.rand(
        latent_dim,
        latent_dim,
        device=sample_device,
        dtype=dtype,
        generator=generator,
    )
    samples = samples.to(device)
    upper = torch.triu((samples < edge_density).to(dtype), diagonal=1)
    adjacency = upper + upper.T
    degree = torch.diag(adjacency.sum(dim=1))
    return degree - adjacency


def identity_quadratic_penalty(z_batch: Tensor) -> Tensor:
    """
    The simplest possible quadratic penalty: z^T I z = ||z||^2 averaged
    over batch. Equivalent to L2 regularization on latent vectors. Serves
    as a baseline for spectral-quadratic priors: any improvement over
    this baseline indicates that the structured form (Laplacian / random
    graph) is doing more than generic norm shrinkage.
    """
    if z_batch.dim() == 1:
        z_batch = z_batch.unsqueeze(0)
    return z_batch.pow(2).sum(dim=1).mean()


def build_causal_laplacian(causal_graph: nx.DiGraph, latent_dim: int) -> Tensor:
    """Symmetrized normalized Laplacian of the causal event graph.

    The result is projected into ``(latent_dim, latent_dim)``. Strategy:
    symmetrize, compute normalized Laplacian, take top-k eigenvectors
    (k = min(n, latent_dim)), embed into latent_dim with zero-padding, then
    sum lambda_i * v_i v_i^T. If n < 2, return identity.
    """

    n = causal_graph.number_of_nodes()
    if n < 2:
        return torch.eye(latent_dim)
    G_und = causal_graph.to_undirected()
    L_np = nx.normalized_laplacian_matrix(G_und).toarray().astype(np.float32)
    eigvals, eigvecs = np.linalg.eigh(L_np)
    k = min(latent_dim, n)
    L_out = np.zeros((latent_dim, latent_dim), dtype=np.float32)
    for i in range(k):
        v = np.zeros(latent_dim, dtype=np.float32)
        copy_len = min(n, latent_dim)
        v[:copy_len] = eigvecs[:copy_len, -(i + 1)]
        L_out += float(eigvals[-(i + 1)]) * np.outer(v, v)
    return torch.from_numpy(L_out)


# 7. WorldModel wrapper
class WorldModel(nn.Module):
    def __init__(
        self,
        encoder: str = "hypergraph",
        hidden_dim: int = 64,
        latent_dim: int = 32,
        action_dim: int = 1,
        transition_hidden_dim: int = 128,
        mlp_hidden_dim: int = 128,
        n_atoms: int | None = None,
    ):
        super().__init__()
        self.encoder_name = encoder
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        if encoder == "hypergraph":
            self.encoder = HypergraphEncoder(hidden_dim, latent_dim)
        elif encoder == "flat":
            self.encoder = FlatMLPEncoder(hidden_dim, latent_dim)
        elif encoder == "mlp":
            if n_atoms is None:
                raise ValueError("n_atoms must be provided when encoder='mlp'")
            self.encoder = MLPEncoder(
                n_atoms=n_atoms,
                hidden_dim=mlp_hidden_dim,
                latent_dim=latent_dim,
            )
        else:
            raise ValueError("encoder must be one of 'hypergraph', 'flat', or 'mlp'")

        self.transition = TransitionNetwork(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=transition_hidden_dim,
        )

    def encode(self, observation: HeteroData) -> Tensor:
        return self.encoder(observation)

    def forward(
        self, observation: HeteroData, action: Tensor | int | float
    ) -> Dict[str, Tensor]:
        latent = self.encode(observation)
        next_latent_pred, reward_pred, done_logit = self.transition(latent, action)
        return {
            "latent": latent,
            "next_latent_pred": next_latent_pred,
            "reward_pred": reward_pred,
            "done_logit": done_logit,
        }

    def loss(
        self,
        observation: HeteroData,
        action: Tensor | int | float,
        next_observation: HeteroData,
        reward: Tensor | float,
        done: Optional[Tensor | float] = None,
        latent_l2_weight: float = 1e-4,
        smoothness_weight: float = 1e-3,
        prior: str = "none",
        prior_weight: float = 0.0,
        laplacian: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        outputs = self.forward(observation, action)
        target_next_latent = self.encode(next_observation).detach()

        reward_target = (
            reward.to(outputs["reward_pred"].device, dtype=outputs["reward_pred"].dtype)
            if torch.is_tensor(reward)
            else outputs["reward_pred"].new_tensor(reward)
        )
        transition_loss = F.mse_loss(outputs["next_latent_pred"], target_next_latent)
        reward_loss = F.mse_loss(outputs["reward_pred"], reward_target)

        done_loss = outputs["reward_pred"].new_tensor(0.0)
        if done is not None:
            done_target = (
                done.to(outputs["done_logit"].device, dtype=outputs["done_logit"].dtype)
                if torch.is_tensor(done)
                else outputs["done_logit"].new_tensor(done)
            )
            done_loss = F.binary_cross_entropy_with_logits(
                outputs["done_logit"], done_target
            )

        reg_l2 = latent_l2_regularizer(outputs["latent"])
        reg_smooth = temporal_smoothness_regularizer(
            outputs["latent"], outputs["next_latent_pred"]
        )
        total = (
            transition_loss
            + reward_loss
            + done_loss
            + latent_l2_weight * reg_l2
            + smoothness_weight * reg_smooth
        )
        if prior == "euclidean":
            z_batch = (
                outputs["latent"].unsqueeze(0)
                if outputs["latent"].dim() == 1
                else outputs["latent"]
            )
            prior_loss = euclidean_cov_penalty(z_batch)
        elif prior == "spectral":
            assert laplacian is not None, "spectral prior requires laplacian tensor"
            prior_loss = spectral_laplacian_penalty(outputs["latent"], laplacian)
        elif prior == "none":
            prior_loss = outputs["latent"].new_tensor(0.0)
        else:
            raise ValueError("prior must be one of 'none', 'euclidean', or 'spectral'")
        total = total + prior_weight * prior_loss

        return {
            "total": total,
            "transition": transition_loss,
            "reward": reward_loss,
            "done": done_loss,
            "latent_l2": reg_l2,
            "smoothness": reg_smooth,
            "prior": prior_loss,
        }

    def rollout_latent(self, latent: Tensor, actions: List[int | float | Tensor]) -> Tensor:
        """Roll a latent state forward through a sequence of actions."""

        latents = [latent]
        current = latent
        for action in actions:
            current, _, _ = self.transition(current, action)
            latents.append(current)
        return torch.stack(latents, dim=0)


def _demo() -> None:
    from hypergraph_env import CausalWorldEnv, HypergraphState, RewriteRule

    torch.manual_seed(0)
    rule = RewriteRule(
        lhs=[("x", "y"), ("y", "z")],
        rhs=[("x", "z"), ("z", "w")],
        name="{x,y},{y,z}->{x,z},{z,w}",
    )
    initial_state = HypergraphState([(0, 1), (1, 2)], seed=0)
    env = CausalWorldEnv(rule, initial_state, max_steps=5, seed=0)

    obs = env.reset(seed=0)
    model = WorldModel(encoder="hypergraph", hidden_dim=32, latent_dim=16)
    flat_model = WorldModel(encoder="flat", hidden_dim=32, latent_dim=16)
    causal_encoder = CausalGraphEncoder(hidden_dim=32, latent_dim=16)

    action = env.sample_action()
    if action is None:
        raise RuntimeError("demo environment has no available rewrite matches")

    next_obs, reward, done, info = env.step(action)
    outputs = model(obs, action)
    losses = model.loss(obs, action, next_obs, reward, float(done))
    flat_outputs = flat_model(obs, action)
    causal_latent = causal_encoder(env.state.causal_graph())

    print("hypergraph latent:", tuple(outputs["latent"].shape))
    print("next latent prediction:", tuple(outputs["next_latent_pred"].shape))
    print("reward prediction:", float(outputs["reward_pred"].detach()))
    print("flat latent:", tuple(flat_outputs["latent"].shape))
    print("causal latent:", tuple(causal_latent.shape))
    print("loss keys:", sorted(losses.keys()))
    print("env info:", info)

    batch_latents = torch.stack([
        model.encode(obs), model.encode(next_obs)
    ])
    euc = euclidean_cov_penalty(batch_latents)
    L = build_causal_laplacian(env.state.causal_graph(), latent_dim=16)
    spec = spectral_laplacian_penalty(batch_latents, L)
    losses_euc = model.loss(obs, action, next_obs, reward, float(done),
                            prior="euclidean", prior_weight=0.1)
    losses_spec = model.loss(obs, action, next_obs, reward, float(done),
                             prior="spectral", prior_weight=0.1, laplacian=L)
    print(f"euclidean cov penalty: {float(euc.detach()):.4f}")
    print(f"spectral lap penalty:  {float(spec.detach()):.4f}")
    print(f"loss w/ euc prior:  total={float(losses_euc['total'].detach()):.4f} "
          f"prior={float(losses_euc['prior'].detach()):.4f}")
    print(f"loss w/ spec prior: total={float(losses_spec['total'].detach()):.4f} "
          f"prior={float(losses_spec['prior'].detach()):.4f}")
    print(f"causal Laplacian shape: {tuple(L.shape)}")


if __name__ == "__main__":
    _demo()
