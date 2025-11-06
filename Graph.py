# Graph.py
from __future__ import annotations
"""
Utilities to extract a causal DAG from an SCM, compute its Markov equivalence
class (as a CPDAG), and visualize either with networkx.

Design goals
------------
- Light-weight and dependency-free for core logic (only optional networkx/matplotlib
  for drawing).
- Compatible with this repo's SCM implementation (see SCM.py).
- Clear API that mirrors standard objects: DAG, CPDAG (essential graph).

References: Verma & Pearl (1990); Meek (1995); Chickering (1995, 2002).

Usage
-----
>>> from .SCM import SCM
>>> from .Graph import dag_from_scm, cpdag_from_dag, draw_dag, draw_cpdag
>>> G = dag_from_scm(my_scm)          # networkx DiGraph (optional)
>>> C = cpdag_from_dag(edges_from_scm(my_scm))  # CPDAG object (core)
>>> is_unique = C.is_fully_directed()
>>> # visualization (requires networkx/matplotlib)
>>> draw_cpdag(C, title=f"Unique? {is_unique}")
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, FrozenSet

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover - visualization is optional
    nx = None  # sentinel

try:  # pragma: no cover - optional
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional
    plt = None

from .SCM import SCM

# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

Edge = Tuple[str, str]           # directed (u, v) means u -> v
UEdge = FrozenSet[str]           # undirected {u, v}


def edges_from_scm(scm: SCM) -> Set[Edge]:
    """Return the set of directed edges *u -> v* implied by ``scm.parents``.

    Assumes the SCM is acyclic (enforced in SCM.__init__).
    """
    edges: Set[Edge] = set()
    for v in scm.nodes:
        for u in scm.parents.get(v, []):
            edges.add((u, v))
    return edges


# ---------------------------------------------------------------------------
# CPDAG data structure (essential graph)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CPDAG:
    nodes: Tuple[str, ...]
    directed: FrozenSet[Edge]        # compelled directions u -> v
    undirected: FrozenSet[UEdge]     # reversible edges {u, v}

    def is_fully_directed(self) -> bool:
        return len(self.undirected) == 0

    def as_networkx(self):  # pragma: no cover - visualization helper
        if nx is None:
            raise RuntimeError("networkx is required for as_networkx/drawing")
        G_dir = nx.DiGraph()
        G_dir.add_nodes_from(self.nodes)
        G_dir.add_edges_from(self.directed)
        G_und = nx.Graph()
        G_und.add_nodes_from(self.nodes)
        for uv in self.undirected:
            u, v = tuple(uv)
            G_und.add_edge(u, v)
        return G_dir, G_und


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------

class _PDAG:
    """Internal partially directed graph used during enumeration.

    Representation:
        - adj_und[u] is set of neighbors v with an undirected link u—v
        - adj_dir[u] is set of children v with u -> v
    """

    __slots__ = ("nodes", "adj_und", "adj_dir")

    def __init__(self, nodes: Sequence[str], undirected: Iterable[UEdge], directed: Iterable[Edge]):
        self.nodes: Tuple[str, ...] = tuple(nodes)
        self.adj_und: Dict[str, Set[str]] = {v: set() for v in self.nodes}
        self.adj_dir: Dict[str, Set[str]] = {v: set() for v in self.nodes}
        for u, v in directed:
            self.adj_dir[u].add(v)
        for uv in undirected:
            u, v = tuple(uv)
            self.adj_und[u].add(v)
            self.adj_und[v].add(u)

    # --- graph queries ---
    def has_edge(self, u: str, v: str) -> bool:
        return (v in self.adj_dir.get(u, set())) or (u in self.adj_dir.get(v, set())) or (v in self.adj_und.get(u, set()))

    def neighbors_und(self, v: str) -> Set[str]:
        return set(self.adj_und[v])

    def children(self, u: str) -> Set[str]:
        return set(self.adj_dir[u])

    def parents(self, v: str) -> Set[str]:
        return {u for u in self.nodes if v in self.adj_dir[u]}

    def nonadjacent(self, u: str, v: str) -> bool:
        return (v not in self.adj_dir[u]) and (u not in self.adj_dir[v]) and (v not in self.adj_und[u])

    # --- mutation ---
    def orient(self, u: str, v: str) -> None:
        """Orient an undirected edge u—v into u -> v (assumes it exists)."""
        if v not in self.adj_und[u]:
            return
        self.adj_und[u].remove(v)
        self.adj_und[v].remove(u)
        self.adj_dir[u].add(v)

    def copy(self) -> "_PDAG":
        und = {frozenset({u, v}) for u in self.nodes for v in self.adj_und[u] if u < v}
        dirr = {(u, v) for u in self.nodes for v in self.adj_dir[u]}
        return _PDAG(self.nodes, und, dirr)

    # --- checks ---
    def would_create_cycle(self, u: str, v: str) -> bool:
        """Return True iff adding u -> v would create a directed cycle."""
        # check if v reaches u via directed edges
        stack = [v]
        visited = set()
        while stack:
            x = stack.pop()
            if x == u:
                return True
            if x in visited:
                continue
            visited.add(x)
            stack.extend(self.children(x))
        return False


# --- utilities to compute skeleton and unshielded colliders ---

def _skeleton_from_edges(edges: Set[Edge]) -> Set[UEdge]:
    return {frozenset({u, v}) for (u, v) in edges}


def _immoralities_from_dag(edges: Set[Edge]) -> Set[Tuple[str, str, str]]:
    """Return set of unshielded colliders (x, z, y) meaning x->z<-y and x not adj y."""
    # adjacency for quick lookup
    adj_und = _skeleton_from_edges(edges)
    children: Dict[str, Set[str]] = {}
    parents: Dict[str, Set[str]] = {}
    nodes: Set[str] = set()
    for u, v in edges:
        nodes.add(u); nodes.add(v)
        children.setdefault(u, set()).add(v)
        parents.setdefault(v, set()).add(u)
        children.setdefault(v, set())
        parents.setdefault(u, set())
    imm: Set[Tuple[str, str, str]] = set()
    for z in nodes:
        parz = list(parents.get(z, set()))
        for i in range(len(parz)):
            x = parz[i]
            for j in range(i + 1, len(parz)):
                y = parz[j]
                if frozenset({x, y}) not in adj_und:  # unshielded
                    imm.add((x, z, y))
                    imm.add((y, z, x))  # store both orders for convenience
    return imm


def _unshielded_triples(nodes: Sequence[str], skeleton: Set[UEdge]) -> Set[Tuple[str, str, str]]:
    adj: Dict[str, Set[str]] = {v: set() for v in nodes}
    for uv in skeleton:
        u, v = tuple(uv)
        adj[u].add(v); adj[v].add(u)
    out: Set[Tuple[str, str, str]] = set()
    for z in nodes:
        nz = sorted(adj[z])
        for i in range(len(nz)):
            x = nz[i]
            for j in range(i + 1, len(nz)):
                y = nz[j]
                if frozenset({x, y}) not in skeleton:
                    out.add((x, z, y)); out.add((y, z, x))
    return out


# ---------------------------------------------------------------------------
# CPDAG from a DAG via exact enumeration (safe, small graphs)
# ---------------------------------------------------------------------------

def cpdag_from_dag(edges: Set[Edge]) -> CPDAG:
    """Compute the CPDAG (essential graph) of a DAG given by ``edges``.

    Implementation strategy: enumerate all DAGs in the Markov equivalence class
    (same skeleton + same set of unshielded colliders) and keep edge directions
    that are invariant across the class. This is exact and robust for the small
    graphs typically used in the experiments.
    """
    # Nodes
    nodes: Set[str] = set()
    for u, v in edges:
        nodes.add(u); nodes.add(v)
    nodes_t = tuple(sorted(nodes))

    # Skeleton and collider constraints from the input DAG
    skeleton = _skeleton_from_edges(edges)
    imm = _immoralities_from_dag(edges)                 # required colliders
    U = _unshielded_triples(nodes_t, skeleton)          # all unshielded triples

    # Build initial PDAG where v-structures are already oriented into the center
    undirected: Set[UEdge] = set(skeleton)
    directed: Set[Edge] = set()
    for (x, z, y) in imm:
        if (z, x, y) != (x, z, y):  # keep both copies but only act once
            pass
        # enforce x->z and y->z
        if frozenset({x, z}) in undirected:
            undirected.remove(frozenset({x, z}))
            directed.add((x, z))
        if frozenset({y, z}) in undirected:
            undirected.remove(frozenset({y, z}))
            directed.add((y, z))
    pdag0 = _PDAG(nodes_t, undirected, directed)

    # Prepare list of undecided edges (u—v)
    undecided: List[UEdge] = sorted(list({uv for uv in undirected}), key=lambda e: tuple(sorted(tuple(e))))

    # Backtracking enumeration with constraints
    all_dags: List[Set[Edge]] = []

    def violates_unshielded_collider_constraint(p: _PDAG) -> bool:
        # For any unshielded triple (x, z, y), disallow x->z and y->z unless it is an immorality in the source DAG.
        for (x, z, y) in U:
            if (x, z) in p.adj_dir[x] if False else False:  # placeholder to avoid mypy complaints
                pass
        # Efficient check: iterate over centers z
        for z in p.nodes:
            # collect decided orientations for neighbors of z
            for x in p.parents(z):
                for y in p.parents(z):
                    if x >= y:
                        continue
                    if frozenset({x, y}) not in skeleton:  # unshielded
                        if (x, z, y) not in imm and (y, z, x) not in imm:
                            # found a forbidden new collider
                            return True
        return False

    # Precompute adjacency for cycle checks
    def dfs_has_path(children: Dict[str, Set[str]], src: str, dst: str) -> bool:
        stack = [src]
        seen: Set[str] = set()
        while stack:
            u = stack.pop()
            if u == dst:
                return True
            if u in seen:
                continue
            seen.add(u)
            stack.extend(children[u])
        return False

    def extend(p: _PDAG, idx: int) -> None:
        if violates_unshielded_collider_constraint(p):
            return
        if idx >= len(undecided):
            # all oriented: emit DAG
            dag: Set[Edge] = {(u, v) for u in p.nodes for v in p.children(u)}
            all_dags.append(dag)
            return
        uv = undecided[idx]
        u, v = tuple(uv)
        # Try u -> v if no directed path v -> u exists currently
        if not dfs_has_path(p.adj_dir, v, u):
            p1 = p.copy()
            p1.orient(u, v)
            extend(p1, idx + 1)
        # Try v -> u if no directed path u -> v exists currently
        if not dfs_has_path(p.adj_dir, u, v):
            p2 = p.copy()
            p2.orient(v, u)
            extend(p2, idx + 1)

    extend(pdag0, 0)

    if not all_dags:
        # Fallback: if constraints were inconsistent, at least return the original DAG as a trivial class
        all_dags = [set(edges)]

    # Compute compelled vs reversible orientations
    oriented_same: Dict[UEdge, Optional[Edge]] = {uv: None for uv in skeleton}
    for uv in skeleton:
        u, v = tuple(uv)
        # direction presence across all DAGs
        as_u_v = all((u, v) in D for D in all_dags)
        as_v_u = all((v, u) in D for D in all_dags)
        if as_u_v and not as_v_u:
            oriented_same[uv] = (u, v)
        elif as_v_u and not as_u_v:
            oriented_same[uv] = (v, u)
        else:
            oriented_same[uv] = None

    dir_edges: Set[Edge] = set()
    und_edges: Set[UEdge] = set()
    for uv, orient in oriented_same.items():
        if orient is None:
            und_edges.add(uv)
        else:
            dir_edges.add(orient)

    return CPDAG(nodes=nodes_t, directed=frozenset(dir_edges), undirected=frozenset(und_edges))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cpdag_from_scm(scm: SCM) -> CPDAG:
    """CPDAG for the SCM's induced DAG (fully observed/causally sufficient)."""
    return cpdag_from_dag(edges_from_scm(scm))


def dag_from_scm(scm: SCM):  # pragma: no cover - thin wrapper for visualization
    if nx is None:
        raise RuntimeError("networkx is required for dag_from_scm/drawing")
    G = nx.DiGraph()
    G.add_nodes_from(scm.nodes)
    for v in scm.nodes:
        for u in scm.parents.get(v, []):
            G.add_edge(u, v)
    return G


# ---------------------------------------------------------------------------
# Visualization (optional)
# ---------------------------------------------------------------------------

def _default_pos(nodes: Sequence[str]):  # pragma: no cover - drawing helper
    if nx is None:
        raise RuntimeError("networkx is required for drawing")
    # try to preserve topo order if the caller provided a consistent naming
    try:
        return nx.nx_agraph.graphviz_layout(nx.DiGraph(), prog="dot")  # type: ignore
    except Exception:
        return nx.spring_layout(nx.Graph(list((n, n) for n in nodes)))


def draw_dag(G, *, pos=None, title: Optional[str] = None, ax=None):  # pragma: no cover - drawing
    if nx is None or plt is None:
        raise RuntimeError("networkx and matplotlib are required for drawing")
    if pos is None:
        pos = nx.spring_layout(G)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    nx.draw_networkx_nodes(G, pos, node_color="#f0f0ff", edgecolors="#30304a", node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#2c7fb8", arrows=True, arrowsize=20, width=2.0, ax=ax)
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return ax


def draw_cpdag(C: CPDAG, *, pos=None, title: Optional[str] = None, ax=None):  # pragma: no cover - drawing
    if nx is None or plt is None:
        raise RuntimeError("networkx and matplotlib are required for drawing")
    G_dir, G_und = C.as_networkx()
    if pos is None:
        # layout computed on the union graph for stability
        G_union = nx.Graph()
        G_union.add_nodes_from(C.nodes)
        G_union.add_edges_from([(u, v) for (u, v) in G_dir.edges()])
        for uv in C.undirected:
            u, v = tuple(uv)
            G_union.add_edge(u, v)
        pos = nx.spring_layout(G_union, seed=7)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    # nodes
    nx.draw_networkx_nodes(G_dir, pos, node_color="#f7f7f7", edgecolors="#222", node_size=800, ax=ax)
    nx.draw_networkx_labels(G_dir, pos, font_size=10, ax=ax)
    # undirected edges
    if G_und.number_of_edges() > 0:
        nx.draw_networkx_edges(
            G_und, pos, edge_color="#636363", style="solid", width=2.0, ax=ax
        )
    # directed (compelled) edges
    if G_dir.number_of_edges() > 0:
        nx.draw_networkx_edges(
            G_dir, pos, edge_color="#1f78b4", arrows=True, arrowsize=20, width=2.0, ax=ax
        )
    # legend-like subtitle
    subtitle = "compelled: blue arrows; reversible: gray lines"
    ax.axis("off")
    if title:
        ax.set_title(title + "\n" + subtitle)
    else:
        ax.set_title(subtitle)
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Convenience: end-to-end from SCM
# ---------------------------------------------------------------------------

def graph_or_equivalence_from_scm(scm: SCM) -> Tuple[bool, CPDAG]:
    """Return (is_unique, CPDAG) for the SCM.

    ``is_unique`` is True iff the equivalence class is a singleton (CPDAG has
    no undirected edges). The returned object always encodes the class.
    """
    C = cpdag_from_scm(scm)
    return C.is_fully_directed(), C


# ---------------------------------------------------------------------------
# Minimal smoke test (run as a script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # Tiny example: X -> Z <- Y, Z -> W, X -> W, Y -> W
    nodes = ["X", "Y", "Z", "W"]
    parents = {"X": [], "Y": [], "Z": ["X", "Y"], "W": ["Z", "X", "Y"]}

    def const(v):
        return lambda *_: v

    f = {v: (lambda par, rng, v=v: sum(par.get(u, 0.0) for u in parents[v])) for v in nodes}
    scm = SCM(nodes=nodes, parents=parents, f=f)

    is_unique, C = graph_or_equivalence_from_scm(scm)
    print("Unique DAG?", is_unique)
    print("Directed edges:", sorted(C.directed))
    print("Reversible edges:", sorted(map(tuple, C.undirected)))

    if nx is not None and plt is not None:
        draw_cpdag(C, title=f"Singleton? {is_unique}")
        plt.show()
