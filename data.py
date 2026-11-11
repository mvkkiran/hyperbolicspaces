"""Toy dataset: a hierarchy of English statements.

Each child statement is a *more specific* version of its parent (the child
entails the parent). The hierarchy is written as indented text so you can edit
it freely inside the app. Two spaces = one level.

The statements deliberately contain "word traps": sentences from *different*
branches share surface words ("swims in the ocean", "flies in the sky",
"fits in a bag"). A method that only looks at word overlap will be fooled; a
method that embeds the hierarchy should not be.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

DEFAULT_TEXT = """It is something in the world.
  It is a living thing.
    It is an animal.
      It is a mammal that breathes air.
        It is a dog that barks.
          It is a golden retriever that fetches balls.
          It is a chihuahua that fits in a handbag.
        It is a cat that purrs.
        It is a whale that swims in the ocean and breathes air.
      It is a bird with feathers.
        It is a sparrow that flies in the sky.
        It is an eagle that flies and hunts.
        It is a penguin that swims in the ocean but cannot fly.
      It is a fish with gills.
        It is a shark that swims in the ocean with sharp teeth.
        It is a goldfish that swims in a bowl.
    It is a plant.
      It is a tree with a trunk.
        It is an oak tree with acorns.
        It is a pine tree with needles.
      It is a flower.
        It is a rose with thorns.
        It is a sunflower that follows the sun.
  It is a machine made by people.
    It is a vehicle that moves people.
      It is a car that drives on roads.
      It is a bicycle with two wheels.
      It is an airplane that flies in the sky.
    It is a computer that runs programs.
      It is a laptop that fits in a bag.
      It is a phone that fits in a pocket.
"""


@dataclass
class Hierarchy:
    statements: list[str]          # index -> sentence
    graph: nx.Graph                # undirected tree over indices
    parent: dict[int, int | None]  # index -> parent index
    depth: np.ndarray              # index -> depth (root = 0)
    root: int
    dist: np.ndarray               # n x n shortest-path (tree) distances

    @property
    def n(self) -> int:
        return len(self.statements)

    def children(self, i: int) -> list[int]:
        return [j for j, p in self.parent.items() if p == i]

    def ancestor_at_depth(self, i: int, d: int) -> int | None:
        """Ancestor of node i that sits at depth d (None if i is shallower)."""
        if self.depth[i] < d:
            return None
        while self.depth[i] > d:
            i = self.parent[i]
        return i

    def labels_at_depth(self, d: int) -> tuple[np.ndarray, list[int], list[str]]:
        """Group every node by its depth-d ancestor.

        Returns (label_idx per node or -1, list of eligible node indices,
        label names).
        """
        groups: list[int] = []
        lab = np.full(self.n, -1, dtype=int)
        for i in range(self.n):
            a = self.ancestor_at_depth(i, d)
            if a is None:
                continue
            if a not in groups:
                groups.append(a)
            lab[i] = groups.index(a)
        eligible = [i for i in range(self.n) if lab[i] >= 0]
        names = [short(self.statements[g]) for g in groups]
        return lab, eligible, names


def short(sentence: str) -> str:
    """'It is a dog that barks.' -> 'dog that barks'."""
    s = sentence.strip().rstrip(".")
    for prefix in ("It is an ", "It is a ", "It is "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s


def parse_hierarchy(text: str) -> Hierarchy:
    """Parse indented text into a tree. Raises ValueError on bad input."""
    statements: list[str] = []
    parent: dict[int, int | None] = {}
    stack: list[tuple[int, int]] = []  # (indent, node index)
    for raw in text.splitlines():
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        sentence = raw.strip()
        if sentence in statements:
            raise ValueError(f"Duplicate statement: {sentence!r}")
        idx = len(statements)
        statements.append(sentence)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if not stack:
            if idx != 0:
                raise ValueError(
                    "Only one root is allowed: every line except the first must be indented."
                )
            parent[idx] = None
        else:
            parent[idx] = stack[-1][1]
        stack.append((indent, idx))
    if len(statements) < 3:
        raise ValueError("Need at least 3 statements.")

    g = nx.Graph()
    g.add_nodes_from(range(len(statements)))
    for c, p in parent.items():
        if p is not None:
            g.add_edge(p, c)
    depth = np.zeros(len(statements), dtype=int)
    for i in range(len(statements)):
        d, j = 0, i
        while parent[j] is not None:
            j = parent[j]
            d += 1
        depth[i] = d
    sp = dict(nx.all_pairs_shortest_path_length(g))
    n = len(statements)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i, j] = sp[i][j]
    return Hierarchy(statements, g, parent, depth, 0, dist)


def tree_layout(h: Hierarchy) -> dict[int, tuple[float, float]]:
    """Simple top-down layout: leaves spaced evenly, parents centred."""
    pos: dict[int, tuple[float, float]] = {}
    counter = [0.0]

    def place(i: int) -> float:
        kids = h.children(i)
        if not kids:
            x = counter[0]
            counter[0] += 1.0
        else:
            xs = [place(k) for k in kids]
            x = float(np.mean(xs))
        pos[i] = (x, -float(h.depth[i]))
        return x

    place(h.root)
    return pos
