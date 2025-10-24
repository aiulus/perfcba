# SCM.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Iterable, Any, Tuple
import numpy as np


# ---------- Interventions ----------

@dataclass(frozen=True)
class Intervention:
    """
    Interventions can:
      - hard-set node values: hard = {"X": 1.0, "Z": 0.0}
      - replace mechanisms (soft intervention): soft = {"Y": new_f_Y}
    """
    name: str
    hard: Dict[str, Any] = field(default_factory=dict)
    soft: Dict[str, Callable[[Dict[str, Any], np.random.Generator], Any]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------- General (functional) SCM ----------

NodeFn = Callable[[Dict[str, Any], np.random.Generator], Any]

class SCM:
    """
    General acyclic structural causal model.
    nodes: list of variable names in any order (we topo-sort internally)
    parents: dict node -> list of parent names
    f: dict node -> structural function f_v(parents_dict, rng) -> value
       (exogenous noise sampling should be inside f_v as needed)
    """

    def __init__(
        self,
        nodes: Iterable[str],
        parents: Dict[str, List[str]],
        f: Dict[str, NodeFn],
        check_dag: bool = True,
    ):
        self.nodes: List[str] = list(nodes)
        self.parents: Dict[str, List[str]] = {v: list(parents.get(v, [])) for v in self.nodes}
        self.f: Dict[str, NodeFn] = dict(f)
        _missing = [v for v in self.nodes if v not in self.f]
        if _missing:
            raise ValueError(f"Missing mechanisms for nodes: {_missing}")
        if check_dag:
            self._topo = self._topological_order()
        else:
            self._topo = list(self.nodes)

    # --- basic graph helpers ---

    def topological_order(self) -> List[str]:
        return list(self._topo)

    def parents_of(self, v: str) -> List[str]:
        return list(self.parents.get(v, []))

    def _topological_order(self) -> List[str]:
        # Kahn's algorithm
        indeg = {v: 0 for v in self.nodes}
        for v in self.nodes:
            for u in self.parents.get(v, []):
                indeg[v] += 1
        q = [v for v in self.nodes if indeg[v] == 0]
        out = []
        while q:
            v = q.pop()
            out.append(v)
            for w in self.nodes:
                if v in self.parents.get(w, []):
                    indeg[w] -= 1
                    if indeg[w] == 0:
                        q.append(w)
        if len(out) != len(self.nodes):
            raise ValueError("Graph has a cycle; SCM must be acyclic.")
        return out

    # --- simulation & expectations ---

    def sample(
        self,
        rng: np.random.Generator,
        intervention: Optional[Intervention] = None,
        return_nodes: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """
        Draw a sample from the interventional distribution P(V | do(intervention)).
        - Hard interventions fix node values.
        - Soft interventions replace selected mechanisms f_v.
        """
        hard = intervention.hard if intervention else {}
        soft = intervention.soft if intervention else {}
        f_eff: Dict[str, NodeFn] = {**self.f, **soft}

        values: Dict[str, Any] = {}
        for v in self._topo:
            if v in hard:
                values[v] = hard[v]
            else:
                par_vals = {u: values[u] for u in self.parents.get(v, [])}
                values[v] = f_eff[v](par_vals, rng)
        if return_nodes is None:
            return values
        ret_nodes = list(return_nodes)
        return {k: values[k] for k in ret_nodes}

    def mean(
        self,
        node: str,
        intervention: Optional[Intervention] = None,
        n_mc: int = 10_000,
        seed: Optional[int] = None,
    ) -> float:
        """Monte-Carlo estimate of E[node | do(intervention)]."""
        rng = np.random.default_rng(seed)
        s = 0.0
        for _ in range(int(n_mc)):
            x = self.sample(rng, intervention=intervention)
            s += float(x[node])
        return s / n_mc


# ---------- Linear–Gaussian SCM (analytic expectations) ----------

class LinearGaussianSCM(SCM):
    r"""
    Linear, additive Gaussian SCM in topological order.

      X = W^T X + c + U,  with U ~ N(mu_U, Σ_U), DAG so (I - W^T) is invertible.

    nodes are ordered topologically. W[i,j] is the weight from node j -> i.
    """

    def __init__(
        self,
        nodes: Iterable[str],
        W: np.ndarray,            # shape (n, n), zero diagonal, strictly triangular after topo order
        c: np.ndarray,            # shape (n,)
        mu_u: np.ndarray,         # shape (n,)
        Sigma_u: np.ndarray,      # shape (n,n)
    ):
        nodes = list(nodes)
        n = len(nodes)
        W = np.asarray(W, dtype=float)
        c = np.asarray(c, dtype=float).reshape(n)
        mu_u = np.asarray(mu_u, dtype=float).reshape(n)
        Sigma_u = np.asarray(Sigma_u, dtype=float).reshape(n, n)

        if W.shape != (n, n) or Sigma_u.shape != (n, n):
            raise ValueError("W and Sigma_u must be (n,n).")
        if np.max(np.abs(np.diag(W))) > 1e-12:
            raise ValueError("Diagonal of W must be zero.")
        self._A = np.eye(n) - W.T  # A X = c + U
        if np.linalg.matrix_rank(self._A) != n:
            raise ValueError("(I - W^T) must be invertible (DAG).")

        # Build parents map from W (edge j->i if W[i,j] != 0)
        parents = {nodes[i]: [nodes[j] for j in range(n) if abs(W[i, j]) > 0] for i in range(n)}

        # structural functions draw U_i then compute node via parents (only needed for simulation fallback)
        def make_f(i: int) -> NodeFn:
            def f_v(par: Dict[str, Any], rng: np.random.Generator) -> float:
                # assemble sum_{j in pa(i)} W[i,j] * x_j + c_i + U_i
                u_i = rng.normal(mu_u[i], np.sqrt(np.maximum(1e-12, Sigma_u[i, i])))
                s = c[i]
                for j, name in enumerate(nodes):
                    if abs(W[i, j]) > 0:
                        s += W[i, j] * par[name]
                return s + u_i
            return f_v

        super().__init__(nodes=nodes, parents=parents, f={nodes[i]: make_f(i) for i in range(n)}, check_dag=False)
        self.nodes = nodes
        self.W, self.c, self.mu_u, self.Sigma_u = W, c, mu_u, Sigma_u
        self._A_inv = np.linalg.inv(self._A)

    def mean_linear_do(self, intervention: Optional[Intervention] = None) -> np.ndarray:
        """
        Analytic E[X | do(intervention)] for hard interventions on any subset S.
        For soft interventions (mechanism replacement), fall back to MC via .mean().
        """
        if intervention is None or (not intervention.soft and not intervention.hard):
            # No intervention: E[X] = A^{-1} (c + mu_U)
            return self._A_inv @ (self.c + self.mu_u)

        if intervention.soft:
            raise NotImplementedError("Analytic mean for soft interventions not provided.")

        n = len(self.nodes)
        # Hard interventions: fix X_S = z, remove incoming edges to S.
        # Partition indices into S (intervened) and R (rest).
        name_to_idx = {v: i for i, v in enumerate(self.nodes)}
        S_idx = np.array(sorted(name_to_idx[v] for v in intervention.hard.keys()), dtype=int)
        if S_idx.size == 0:
            # Degenerate case where intervention has no hard assignments.
            return self._A_inv @ (self.c + self.mu_u)
        R_idx = np.array(sorted(set(range(n)) - set(S_idx)), dtype=int)
        z = np.array([float(intervention.hard[self.nodes[i]]) for i in S_idx], dtype=float)

        # Structural form: X = W X + c + U (rows index children, columns parents).
        # For nodes in R (unintervened):
        #   (I - W_RR) X_R = W_RS z + c_R + mu_R.
        W_RR = self.W[np.ix_(R_idx, R_idx)]
        W_RS = self.W[np.ix_(R_idx, S_idx)]
        c_R = self.c[R_idx]
        mu_R = self.mu_u[R_idx]

        A_R = np.eye(len(R_idx)) - W_RR
        b_R = c_R + mu_R + W_RS @ z

        X_mean = np.zeros(n, dtype=float)
        if R_idx.size:
            X_mean[R_idx] = np.linalg.solve(A_R, b_R)
        X_mean[S_idx] = z
        return X_mean

    def mean(self, node: str, intervention: Optional[Intervention] = None, n_mc: int = 0, seed: Optional[int] = None) -> float:
        """If intervention is hard-only, return analytic mean; else fall back to SCM.mean."""
        if intervention is None or (intervention.hard and not intervention.soft):
            mu_vec = self.mean_linear_do(intervention)
            idx = {v: i for i, v in enumerate(self.nodes)}[node]
            return float(mu_vec[idx])
        # soft/mechanism change -> Monte Carlo
        return super().mean(node=node, intervention=intervention, n_mc=(10_000 if n_mc == 0 else n_mc), seed=seed)
