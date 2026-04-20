"""A small Wolfram-style hypergraph rewriting environment.

The module defines a minimal synthetic world model based on hypergraph
rewrites. Hyperedges are represented as tuples of integer node ids, while
rewrite rules use strings as pattern variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import copy
import os
import random

import networkx as nx
import torch
from torch_geometric.data import HeteroData


Hyperedge = Tuple[int, ...]
PatternEdge = Tuple[Any, ...]
Bindings = Dict[str, int]


@dataclass(frozen=True)
class RewriteMatch:
    """A concrete match of a rule's left-hand side in a hypergraph state."""

    edge_indices: Tuple[int, ...]
    bindings: Bindings


@dataclass
class RewriteEvent:
    """Bookkeeping for one rewrite, used to recover causal dependencies."""

    event_id: int
    rule: "RewriteRule"
    matched_edge_indices: Tuple[int, ...]
    bindings: Bindings
    parent_events: Tuple[int, ...]
    produced_edges: Tuple[Hyperedge, ...]


class RewriteRule:
    """A hypergraph rewrite rule with variable-based pattern matching.

    Variables are represented by strings. Any variable that appears on the
    right-hand side but not the left-hand side is treated as a new node marker.

    Example:
        ``[("x", "y"), ("y", "z")] -> [("x", "z"), ("z", "w")]``
        creates one new node for ``"w"`` each time the rule is applied.
    """

    def __init__(
        self,
        lhs: Sequence[Sequence[Any]],
        rhs: Sequence[Sequence[Any]],
        name: Optional[str] = None,
    ) -> None:
        self.lhs: List[PatternEdge] = [tuple(edge) for edge in lhs]
        self.rhs: List[PatternEdge] = [tuple(edge) for edge in rhs]
        self.name = name or "rewrite_rule"
        self.lhs_variables = self._variables_in(self.lhs)

    def find_matches(self, state: "HypergraphState") -> List[RewriteMatch]:
        """Return all valid injective edge matches in ``state``."""

        matches: List[RewriteMatch] = []
        used_edge_indices: set[int] = set()

        def backtrack(
            pattern_index: int,
            bindings: Bindings,
            chosen_edges: List[int],
        ) -> None:
            if pattern_index == len(self.lhs):
                matches.append(
                    RewriteMatch(tuple(chosen_edges), dict(bindings))
                )
                return

            pattern_edge = self.lhs[pattern_index]
            for edge_index, state_edge in enumerate(state.hyperedges):
                if edge_index in used_edge_indices:
                    continue
                next_bindings = self._match_edge(
                    pattern_edge, state_edge, bindings
                )
                if next_bindings is None:
                    continue

                used_edge_indices.add(edge_index)
                chosen_edges.append(edge_index)
                backtrack(pattern_index + 1, next_bindings, chosen_edges)
                chosen_edges.pop()
                used_edge_indices.remove(edge_index)

        backtrack(0, {}, [])
        return matches

    @staticmethod
    def _variables_in(pattern: Iterable[PatternEdge]) -> set[str]:
        return {
            item
            for edge in pattern
            for item in edge
            if isinstance(item, str)
        }

    @staticmethod
    def _match_edge(
        pattern_edge: PatternEdge,
        state_edge: Hyperedge,
        bindings: Bindings,
    ) -> Optional[Bindings]:
        if len(pattern_edge) != len(state_edge):
            return None

        next_bindings = dict(bindings)
        for pattern_item, state_node in zip(pattern_edge, state_edge):
            if isinstance(pattern_item, str):
                bound_node = next_bindings.get(pattern_item)
                if bound_node is None:
                    next_bindings[pattern_item] = state_node
                elif bound_node != state_node:
                    return None
            elif pattern_item != state_node:
                return None

        return next_bindings

    def __repr__(self) -> str:
        return f"RewriteRule(name={self.name!r}, lhs={self.lhs!r}, rhs={self.rhs!r})"


class HypergraphState:
    """Mutable hypergraph state with causal rewrite event history."""

    def __init__(
        self,
        hyperedges: Sequence[Sequence[int]],
        seed: Optional[int] = None,
    ) -> None:
        self.hyperedges: List[Hyperedge] = [tuple(edge) for edge in hyperedges]
        self.edge_origins: List[Optional[int]] = [None for _ in self.hyperedges]
        self.events: List[RewriteEvent] = []
        self._next_node_id = self._infer_next_node_id()
        self.rng = random.Random(seed)

    def apply_rule(
        self,
        rule: RewriteRule,
        seed: Optional[int] = None,
    ) -> Optional[RewriteEvent]:
        """Apply one random matching rewrite step, if any match exists."""

        matches = rule.find_matches(self)
        if not matches:
            return None

        rng = random.Random(seed) if seed is not None else self.rng
        return self.apply_match(rule, rng.choice(matches))

    def apply_match(
        self,
        rule: RewriteRule,
        match: RewriteMatch | int,
    ) -> RewriteEvent:
        """Apply ``rule`` to a specific match or match index."""

        if isinstance(match, int):
            matches = rule.find_matches(self)
            if not 0 <= match < len(matches):
                raise IndexError(
                    f"match index {match} out of range for {len(matches)} matches"
                )
            match = matches[match]

        parent_events = sorted(
            {
                origin
                for edge_index in match.edge_indices
                for origin in [self.edge_origins[edge_index]]
                if origin is not None
            }
        )
        new_bindings = self._extend_bindings_with_new_nodes(rule, match.bindings)
        produced_edges = tuple(
            tuple(new_bindings[item] if isinstance(item, str) else item for item in edge)
            for edge in rule.rhs
        )

        event_id = len(self.events)
        matched_edge_set = set(match.edge_indices)

        self.hyperedges = [
            edge
            for edge_index, edge in enumerate(self.hyperedges)
            if edge_index not in matched_edge_set
        ]
        self.edge_origins = [
            origin
            for edge_index, origin in enumerate(self.edge_origins)
            if edge_index not in matched_edge_set
        ]

        self.hyperedges.extend(produced_edges)
        self.edge_origins.extend([event_id for _ in produced_edges])

        event = RewriteEvent(
            event_id=event_id,
            rule=rule,
            matched_edge_indices=match.edge_indices,
            bindings=dict(match.bindings),
            parent_events=tuple(parent_events),
            produced_edges=produced_edges,
        )
        self.events.append(event)
        return event

    def to_pyg_data(self) -> HeteroData:
        """Convert this hypergraph to a PyTorch Geometric ``HeteroData``.

        The conversion uses a bipartite representation:
        node ids live under the ``"node"`` type, hyperedges under
        ``"hyperedge"``, and incidence is represented by
        ``("node", "member_of", "hyperedge")`` plus its reverse.
        """

        data = HeteroData()
        node_ids = sorted(self.node_ids())
        node_to_row = {node_id: row for row, node_id in enumerate(node_ids)}

        if node_ids:
            data["node"].x = torch.tensor(node_ids, dtype=torch.float).view(-1, 1)
            data["node"].node_id = torch.tensor(node_ids, dtype=torch.long)
        else:
            data["node"].x = torch.empty((0, 1), dtype=torch.float)
            data["node"].node_id = torch.empty((0,), dtype=torch.long)

        edge_features = []
        for index, edge in enumerate(self.hyperedges):
            origin = self.edge_origins[index]
            edge_features.append(
                [float(len(edge)), float(origin if origin is not None else -1)]
            )
        if edge_features:
            data["hyperedge"].x = torch.tensor(edge_features, dtype=torch.float)
        else:
            data["hyperedge"].x = torch.empty((0, 2), dtype=torch.float)
        data["hyperedge"].edge_tuple = list(self.hyperedges)

        incidence_pairs = [
            (node_to_row[node], edge_index)
            for edge_index, edge in enumerate(self.hyperedges)
            for node in edge
        ]
        if incidence_pairs:
            edge_index = torch.tensor(incidence_pairs, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        data["node", "member_of", "hyperedge"].edge_index = edge_index
        data["hyperedge", "has_member", "node"].edge_index = edge_index.flip(0)
        return data

    def causal_graph(self) -> nx.DiGraph:
        """Return the causal dependency graph between rewrite events."""

        graph = nx.DiGraph()
        for event in self.events:
            graph.add_node(
                event.event_id,
                rule=event.rule.name,
                bindings=event.bindings,
                produced_edges=event.produced_edges,
            )
            for parent_event in event.parent_events:
                graph.add_edge(parent_event, event.event_id)
        return graph

    def clone(self) -> "HypergraphState":
        """Return a deep copy of this state."""

        return copy.deepcopy(self)

    def node_ids(self) -> set[int]:
        return {node for edge in self.hyperedges for node in edge}

    def _infer_next_node_id(self) -> int:
        nodes = {node for edge in self.hyperedges for node in edge}
        return max(nodes, default=-1) + 1

    def _extend_bindings_with_new_nodes(
        self,
        rule: RewriteRule,
        bindings: Bindings,
    ) -> Bindings:
        next_bindings = dict(bindings)
        for edge in rule.rhs:
            for item in edge:
                if isinstance(item, str) and item not in next_bindings:
                    next_bindings[item] = self._next_node_id
                    self._next_node_id += 1
        return next_bindings


class CausalWorldEnv:
    """A tiny RL-style environment around one hypergraph rewrite rule."""

    def __init__(
        self,
        rule: RewriteRule,
        initial_state: HypergraphState,
        max_steps: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        self.rule = rule
        self.initial_state = initial_state.clone()
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.state = self.initial_state.clone()
        self.step_count = 0

    def reset(self, seed: Optional[int] = None) -> HeteroData:
        """Reset to the initial state and return the initial observation."""

        if seed is not None:
            self.rng.seed(seed)
        self.state = self.initial_state.clone()
        self.step_count = 0
        return self.state.to_pyg_data()

    def step(self, action: int) -> Tuple[HeteroData, float, bool, Dict[str, Any]]:
        """Apply the selected match and return ``(obs, reward, done, info)``."""

        matches = self.rule.find_matches(self.state)
        if not matches:
            self.step_count += 1
            reward = -float(len(self.state.hyperedges))
            done = self.step_count >= self.max_steps
            info = {
                "step_count": self.step_count,
                "available_matches": 0,
                "applied": False,
            }
            return self.state.to_pyg_data(), reward, done, info

        if not 0 <= action < len(matches):
            raise IndexError(
                f"action {action} out of range for {len(matches)} available matches"
            )

        event = self.state.apply_match(self.rule, matches[action])
        self.step_count += 1
        next_matches = self.rule.find_matches(self.state)
        reward = -float(len(self.state.hyperedges))
        done = self.step_count >= self.max_steps
        info = {
            "step_count": self.step_count,
            "available_matches": len(next_matches),
            "applied": True,
            "event_id": event.event_id,
        }
        return self.state.to_pyg_data(), reward, done, info

    def sample_action(self) -> Optional[int]:
        """Sample a valid action from the current state's available matches."""

        matches = self.rule.find_matches(self.state)
        if not matches:
            return None
        return self.rng.randrange(len(matches))


def _run_demo(seed: int = 7) -> None:
    rule = RewriteRule(
        lhs=[("x", "y"), ("y", "z")],
        rhs=[("x", "z"), ("z", "w")],
        name="{x,y},{y,z}->{x,z},{z,w}",
    )
    initial_state = HypergraphState(hyperedges=[(0, 1), (1, 2)], seed=seed)
    env = CausalWorldEnv(rule, initial_state, max_steps=5, seed=seed)
    env.reset(seed=seed)

    for _ in range(5):
        available_matches = rule.find_matches(env.state)
        print(
            f"step={env.step_count} "
            f"hyperedges={len(env.state.hyperedges)} "
            f"available_matches={len(available_matches)}"
        )
        if not available_matches:
            break
        action = env.sample_action()
        if action is None:
            break
        env.step(action)

    available_matches = rule.find_matches(env.state)
    print(
        f"step={env.step_count} "
        f"hyperedges={len(env.state.hyperedges)} "
        f"available_matches={len(available_matches)}"
    )

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import matplotlib.pyplot as plt

    causal_graph = env.state.causal_graph()
    plt.figure(figsize=(6, 4))
    if causal_graph.number_of_nodes() > 0:
        pos = nx.spring_layout(causal_graph, seed=seed)
        nx.draw_networkx(
            causal_graph,
            pos=pos,
            with_labels=True,
            node_color="#92c5de",
            edge_color="#4d4d4d",
            arrows=True,
        )
    else:
        plt.text(0.5, 0.5, "No rewrite events", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("causal_graph.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    _run_demo()
