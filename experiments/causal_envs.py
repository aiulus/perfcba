"""
Utilities for constructing causal bandit environments consistent with the
setting described in the causal bandit study.

This module provides two main building blocks:

1.  Random SCM generation utilities that return fully specified structural
    causal models together with metadata about the reward node and its parent
    set.
2.  Enumerations of combinatorial intervention spaces of the form considered in
    the manuscript (interventions on up to ``m`` variables taking values in
    ``[ell]``).
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ..SCM import SCM, Intervention
from .sampler_cache import ArmKey, SamplerCache, SubsetKey

NodeName = str
Assignment = Tuple[int, ...]

SCM_MODES: Tuple[str, ...] = ("beta_dirichlet", "reference")


# ---------------------------------------------------------------------------
# Configuration and instance containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CausalBanditConfig:
    """High-level configuration for synthetic causal bandit environments."""

    n: int                     # number of observed covariates (X_1, ..., X_n)
    ell: int                   # domain size for each observed covariate
    k: int                     # number of reward parents
    m: int                     # maximum intervention size allowed to the learner
    edge_prob: float = 0.3     # baseline probability of an edge (default for both covariates and reward edges)
    edge_prob_covariates: Optional[float] = None  # optional override for X->X density
    edge_prob_to_reward: Optional[float] = None   # optional override for X->Y density
    reward_alpha: float = 2.0  # parameters of Beta prior for reward Bernoulli means
    reward_beta: float = 2.0
    reward_logit_scale: float = 1.0  # temperature for Bernoulli logits (controls variance)
    scm_mode: str = "beta_dirichlet"  # sampling style for conditional probability tables
    parent_effect: float = 1.0        # reference-mode mixing between base and parent-specific CPDs
    seed: Optional[int] = None
    hard_margin: float = 0.0          # TV distance enforced between parent assignments
    scm_epsilon: float = 0.0          # lower bound enforced on every CPT entry
    scm_delta: float = 0.0            # tighter lower bound for reward CPT entries

    # Gap targeting (optional; keeps hard_margin for backward compatibility)
    target_epsilon: Optional[float] = None
    target_delta: Optional[float] = None
    gap_enforcement_mode: str = "soft"  # one of {"soft", "hard", "reject"}
    max_rejection_attempts: int = 128

    # Reward heterogeneity controls
    arm_heterogeneity_mode: str = "uniform"  # one of {"uniform", "sparse", "clustered"}
    sparse_fraction: float = 0.1
    sparse_separation: float = 0.3
    cluster_count: int = 3

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")
        if not (0 < self.ell):
            raise ValueError("ell must be positive")
        if not (0 < self.k <= self.n):
            raise ValueError("k must satisfy 0 < k <= n")
        if not (0 <= self.m <= self.n):
            raise ValueError("m must be between 0 and n")
        if not (0.0 <= self.edge_prob <= 1.0):
            raise ValueError("edge_prob must be in [0, 1]")
        if self.edge_prob_covariates is not None and not (0.0 <= self.edge_prob_covariates <= 1.0):
            raise ValueError("edge_prob_covariates must be in [0, 1] when provided")
        if self.edge_prob_to_reward is not None and not (0.0 <= self.edge_prob_to_reward <= 1.0):
            raise ValueError("edge_prob_to_reward must be in [0, 1] when provided")
        if self.reward_alpha <= 0 or self.reward_beta <= 0:
            raise ValueError("reward_alpha and reward_beta must be positive")
        if self.reward_logit_scale <= 0:
            raise ValueError("reward_logit_scale must be positive")
        if self.scm_mode not in SCM_MODES:
            raise ValueError(f"scm_mode must be one of {SCM_MODES}, got {self.scm_mode!r}")
        if not (0.0 <= self.parent_effect <= 1.0):
            raise ValueError("parent_effect must be in [0, 1]")
        if not (0.0 <= self.hard_margin < 1.0):
            raise ValueError("hard_margin must lie in [0, 1)")
        if not (0.0 <= self.scm_epsilon < 0.5):
            raise ValueError("scm_epsilon must lie in [0, 0.5).")
        if self.scm_epsilon * self.ell >= 1.0:
            raise ValueError("scm_epsilon must satisfy epsilon * ell < 1.")
        if not (0.0 <= self.scm_delta < 0.5):
            raise ValueError("scm_delta must lie in [0, 0.5).")
        max_covariate_tv = 1.0 - self.ell * self.scm_epsilon if self.scm_epsilon > 0.0 else 1.0
        if self.hard_margin - 1e-9 > max_covariate_tv:
            raise ValueError("hard_margin is incompatible with scm_epsilon.")
        reward_margin = max(self.scm_epsilon, self.scm_delta)
        max_reward_tv = 1.0 - 2.0 * reward_margin if reward_margin > 0.0 else 1.0
        if self.hard_margin - 1e-9 > max_reward_tv:
            raise ValueError("hard_margin is incompatible with scm_delta/scm_epsilon.")
        if self.target_epsilon is not None and self.target_epsilon < 0.0:
            raise ValueError("target_epsilon must be non-negative when provided.")
        if self.target_delta is not None and self.target_delta < 0.0:
            raise ValueError("target_delta must be non-negative when provided.")
        if self.gap_enforcement_mode not in {"soft", "hard", "reject"}:
            raise ValueError("gap_enforcement_mode must be one of {'soft', 'hard', 'reject'}.")
        if self.max_rejection_attempts <= 0:
            raise ValueError("max_rejection_attempts must be positive.")
        if self.arm_heterogeneity_mode not in ("uniform", "sparse", "clustered"):
            raise ValueError("arm_heterogeneity_mode must be one of {'uniform', 'sparse', 'clustered'}")
        if not (0.0 < self.sparse_fraction <= 1.0):
            raise ValueError("sparse_fraction must be in (0, 1]")
        if not (0.0 < self.sparse_separation < 1.0):
            raise ValueError("sparse_separation must be in (0, 1)")
        if self.cluster_count < 1:
            raise ValueError("cluster_count must be at least 1")

    def rng(self) -> np.random.Generator:
        """Return a generator seeded as requested (fresh instance each call)."""

        return np.random.default_rng(self.seed)


@dataclass(frozen=True)
class CausalBanditInstance:
    """Bundle describing a sampled SCM and metadata about the reward node."""

    config: CausalBanditConfig
    scm: SCM
    node_names: Tuple[NodeName, ...]
    reward_node: NodeName
    reward_parents: Tuple[NodeName, ...]
    reward_means: Dict[Assignment, float]  # Bernoulli means conditioned on parents
    sampler_cache: Optional[SamplerCache] = None

    def reward_mean_for(self, parent_assignment: MutableMapping[NodeName, int]) -> float:
        """Return E[Y | parents = parent_assignment]."""

        key = tuple(parent_assignment[name] for name in self.reward_parents)
        return float(self.reward_means[key])

    def parent_indices(self) -> Tuple[int, ...]:
        """Return indices of reward parents relative to ``node_names``."""

        return tuple(self.node_names.index(name) for name in self.reward_parents)

    def sample_reward(
        self,
        arm: "InterventionArm",
        rng: np.random.Generator,
    ) -> float:
        """Draw a single reward sample for the provided intervention arm."""

        def compute() -> float:
            intervention = arm.as_intervention(self.node_names)
            values = self.scm.sample(rng, intervention=intervention)
            return float(values[self.reward_node])

        if self.sampler_cache is None:
            return compute()
        arm_key: ArmKey = (tuple(arm.variables), tuple(arm.values))
        return self.sampler_cache.sample_reward(arm_key=arm_key, rng=rng, compute=compute)

    def estimate_arm_mean(
        self,
        arm: "InterventionArm",
        rng: np.random.Generator,
        n_mc: int = 1024,
    ) -> float:
        """Monte-Carlo estimate of ``E[Y | do(arm)]``."""

        n_samples = max(1, int(n_mc))

        def compute() -> float:
            intervention = arm.as_intervention(self.node_names)
            samples = [
                float(self.scm.sample(rng, intervention=intervention)[self.reward_node])
                for _ in range(n_samples)
            ]
            return float(np.mean(samples))

        if self.sampler_cache is None:
            return compute()
        arm_key: ArmKey = (tuple(arm.variables), tuple(arm.values))
        return self.sampler_cache.estimate_arm_mean(
            arm_key=arm_key,
            n_mc=n_samples,
            rng=rng,
            compute=compute,
        )

    def estimate_subset_mean(
        self,
        subset: Sequence[int],
        assignments: Sequence[int],
        rng: np.random.Generator,
        n_mc: int = 2048,
    ) -> float:
        """Estimate ``E[Y | do(X_subset = assignments)``."""

        n_samples = max(1, int(n_mc))

        def compute() -> float:
            if not subset:
                samples = [float(self.scm.sample(rng)[self.reward_node]) for _ in range(n_samples)]
                return float(np.mean(samples))
            arm = InterventionArm(tuple(subset), tuple(assignments))
            return self.estimate_arm_mean(arm, rng, n_mc=n_samples)

        if self.sampler_cache is None:
            return compute()
        subset_key: SubsetKey = (tuple(subset), tuple(assignments))
        return self.sampler_cache.estimate_subset_mean(
            subset_key=subset_key,
            n_mc=n_samples,
            rng=rng,
            compute=compute,
        )

    # ------------------------------------------------------------------ #
    # Gap estimation utilities
    # ------------------------------------------------------------------ #

    @dataclass(frozen=True)
    class GapEstimates:
        eps: float
        Delta: float
        details: Dict[str, Any]

    def estimate_min_eps_delta(
        self,
        *,
        max_parent_scope_exact: int = 2,
        n_mc: int = 8000,
        alpha: float = 0.05,
        max_hops: int = 3,
        rng: Optional[np.random.Generator] = None,
    ) -> "CausalBanditInstance.GapEstimates":
        """
        Estimate lower bounds on the ancestral (ε) and reward (Δ) gaps.

        The routine enumerates interventions that differ on a single parent/
        ancestor and compares the induced marginals:
        - ε_hat: min over parents X->Y of max_y |P(Y=y|do(Z)) - P(Y=y|do(Z,X))|
        - Δ_hat: min over ancestors X of reward of |E[R|do(Z)] - E[R|do(Z,X)]|

        Exact evaluation is used when the intervention scope is small; otherwise
        Monte Carlo sampling is applied with a conservative CI shrinkage.
        """

        rng = rng or np.random.default_rng(self.config.seed)

        parent_map = {name: list(self.scm.parents_of(name)) for name in self.node_names}
        graph_parents = {name: list(parent_map.get(name, [])) for name in self.node_names}

        def _ancestors(node: NodeName) -> set[NodeName]:
            seen: set[NodeName] = set()
            stack = list(graph_parents.get(node, []))
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                stack.extend(graph_parents.get(cur, []))
            return seen

        def _intervention_key(assignment: MutableMapping[NodeName, int]) -> Tuple[Tuple[NodeName, int], ...]:
            return tuple(sorted((k, int(v)) for k, v in assignment.items()))

        dist_cache: Dict[Tuple[Tuple[NodeName, int], ...], np.ndarray] = {}
        mean_cache: Dict[Tuple[Tuple[NodeName, int], ...], float] = {}

        def _sample_node_distribution(
            node: NodeName, assignment: MutableMapping[NodeName, int], scope_size: int
        ) -> np.ndarray:
            key = _intervention_key(assignment)
            if key in dist_cache:
                return dist_cache[key]

            if scope_size <= max_parent_scope_exact and self.config.ell ** scope_size <= 64:
                # Small scope: exact enumeration over intervened nodes.
                counts = np.zeros(self.config.ell if node != self.reward_node else 2, dtype=float)
                total = 0.0
                iter_rng = np.random.default_rng(int(rng.integers(2**32 - 1)))
                for _ in range(max(1, int(self.config.ell ** max(scope_size, 1)))):
                    draw = self.scm.sample(iter_rng, intervention=Intervention(name="do", hard=dict(assignment)))
                    counts[int(draw[node])] += 1.0
                    total += 1.0
                probs = counts / max(1.0, total)
            else:
                counts = np.zeros(self.config.ell if node != self.reward_node else 2, dtype=float)
                samples = max(1, int(n_mc))
                sample_rng = np.random.default_rng(int(rng.integers(2**32 - 1)))
                for _ in range(samples):
                    draw = self.scm.sample(sample_rng, intervention=Intervention(name="do", hard=dict(assignment)))
                    counts[int(draw[node])] += 1.0
                probs = counts / float(samples)
                # Hoeffding-style shrinkage to get a conservative lower bound on diffs.
                eps_ci = math.sqrt(0.5 * math.log(2.0 / alpha) / float(samples))
                probs = np.clip(probs, 0.0, 1.0)
                # This shrink stretches towards uniform; ensure a valid distribution.
                probs = probs / probs.sum() if probs.sum() > 0 else probs
                probs = np.clip(probs, eps_ci, 1.0)  # avoid zeros for downstream logics
                probs = probs / probs.sum()

            dist_cache[key] = probs
            return probs

        def _sample_reward_mean(assignment: MutableMapping[NodeName, int], scope_size: int) -> float:
            key = _intervention_key(assignment)
            if key in mean_cache:
                return mean_cache[key]
            if scope_size <= max_parent_scope_exact and self.config.ell ** scope_size <= 64:
                # reuse distribution call for Bernoulli reward
                probs = _sample_node_distribution(self.reward_node, assignment, scope_size)
                value = float(probs[1]) if probs.size > 1 else float(probs[0])
            else:
                samples = max(1, int(n_mc))
                sample_rng = np.random.default_rng(int(rng.integers(2**32 - 1)))
                draws = [
                    int(self.scm.sample(sample_rng, intervention=Intervention(name="do", hard=dict(assignment)))[self.reward_node])
                    for _ in range(samples)
                ]
                mean = float(np.mean(draws))
                ci = math.sqrt(0.5 * math.log(2.0 / alpha) / float(samples))
                value = max(0.0, min(1.0, mean - ci))
            mean_cache[key] = value
            return value

        eps_candidates: List[float] = []
        for child in self.node_names:
            if child == self.reward_node:
                continue
            for parent in graph_parents.get(child, []):
                other_parents = [p for p in graph_parents[child] if p != parent]
                if len(other_parents) > max_hops:
                    continue
                for z_vals in _all_assignments(len(other_parents), self.config.ell):
                    assignment = {name: val for name, val in zip(other_parents, z_vals)}
                    base_probs = _sample_node_distribution(child, assignment, len(assignment))
                    for x_val in range(self.config.ell):
                        assignment[parent] = x_val
                        inter_probs = _sample_node_distribution(child, assignment, len(assignment))
                        diff = float(np.max(np.abs(base_probs - inter_probs)))
                        eps_candidates.append(diff)
                        assignment.pop(parent, None)

        delta_candidates: List[float] = []
        reward_parents = graph_parents[self.reward_node]
        reward_ancestors = [a for a in _ancestors(self.reward_node) if a != self.reward_node]
        for anc in reward_ancestors:
            other = [p for p in reward_parents if p != anc]
            if len(other) > max_hops:
                continue
            for z_vals in _all_assignments(len(other), self.config.ell):
                assignment = {name: val for name, val in zip(other, z_vals)}
                base_mean = _sample_reward_mean(assignment, len(assignment))
                for x_val in range(self.config.ell):
                    assignment[anc] = x_val
                    inter_mean = _sample_reward_mean(assignment, len(assignment))
                    delta_candidates.append(abs(base_mean - inter_mean))
                    assignment.pop(anc, None)

        eps_hat = float(min(eps_candidates)) if eps_candidates else 0.0
        delta_hat = float(min(delta_candidates)) if delta_candidates else 0.0

        details = {
            "eps_candidates": eps_candidates,
            "delta_candidates": delta_candidates,
            "config": {
                "max_parent_scope_exact": max_parent_scope_exact,
                "n_mc": n_mc,
                "alpha": alpha,
                "max_hops": max_hops,
            },
        }
        return CausalBanditInstance.GapEstimates(eps=eps_hat, Delta=delta_hat, details=details)


# ---------------------------------------------------------------------------
# SCM generation
# ---------------------------------------------------------------------------


def _random_order(n: int, rng: np.random.Generator) -> List[int]:
    return list(rng.permutation(n))


def _all_assignments(num_parents: int, domain: int) -> Iterable[Assignment]:
    if num_parents == 0:
        yield ()
        return
    yield from itertools.product(range(domain), repeat=num_parents)


_HARD_MARGIN_MAX_ATTEMPTS = 512


def _table_tv_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(a - b)))


def _table_satisfies_margin(table: Dict[Assignment, np.ndarray], margin: float) -> bool:
    if margin <= 0.0 or len(table) <= 1:
        return True
    assignments = list(table.keys())
    for idx, lhs in enumerate(assignments):
        probs_l = table[lhs]
        for rhs in assignments[idx + 1 :]:
            if _table_tv_distance(probs_l, table[rhs]) + 1e-9 < margin:
                return False
    return True


def _apply_probability_margin(probs: np.ndarray, *, margin: float) -> np.ndarray:
    if margin <= 0.0:
        return probs
    ell = probs.shape[0]
    if margin * ell >= 1.0:
        raise ValueError("Margin too large for the provided domain size.")
    scale = 1.0 - ell * margin
    return probs * scale + margin


def _force_margin_table(table: Dict[Assignment, np.ndarray], ell: int, margin: float) -> None:
    if ell <= 1:
        return
    assignments = list(table.keys())
    for idx, assignment in enumerate(assignments):
        anchor = idx % ell
        if margin <= 0.0:
            probs = np.zeros(ell, dtype=float)
            probs[anchor] = 1.0
        else:
            if margin * ell >= 1.0:
                raise ValueError("Margin too large for fallback CPT construction.")
            probs = np.full(ell, margin, dtype=float)
            probs[anchor] = margin + (1.0 - ell * margin)
        table[assignment] = probs


def _sample_covariate_cpt(
    num_parents: int,
    ell: int,
    rng: np.random.Generator,
    *,
    mode: str,
    parent_effect: float,
    hard_margin: float,
    prob_margin: float,
) -> Dict[Assignment, np.ndarray]:
    assignments = list(_all_assignments(num_parents, ell))
    base_alpha = np.ones(ell, dtype=float)
    for _ in range(_HARD_MARGIN_MAX_ATTEMPTS):
        table: Dict[Assignment, np.ndarray] = {}
        if mode == "reference":
            base = rng.dirichlet(np.ones(ell, dtype=float))
            for assignment in assignments:
                rnd = rng.dirichlet(base_alpha)
                probs = (1.0 - parent_effect) * base + parent_effect * rnd
                probs = probs / probs.sum()
                table[assignment] = _apply_probability_margin(probs, margin=prob_margin)
        else:
            for assignment in assignments:
                probs = rng.dirichlet(base_alpha)
                table[assignment] = _apply_probability_margin(probs, margin=prob_margin)
        if _table_satisfies_margin(table, hard_margin):
            return table
    _force_margin_table(table, ell, prob_margin)
    return table


def _sample_reward_cpt(
    num_parents: int,
    ell: int,
    rng: np.random.Generator,
    alpha: float,
    beta: float,
    *,
    mode: str,
    hard_margin: float,
    prob_margin: float,
    reward_margin: float,
) -> Dict[Assignment, np.ndarray]:
    assignments = list(_all_assignments(num_parents, ell))
    for _ in range(_HARD_MARGIN_MAX_ATTEMPTS):
        table: Dict[Assignment, np.ndarray] = {}
        for assignment in assignments:
            if mode == "reference":
                probs = rng.dirichlet(np.ones(2, dtype=float))
            else:
                p = rng.beta(alpha, beta)
                probs = np.array([1.0 - p, p], dtype=float)
            probs = _apply_probability_margin(probs, margin=prob_margin)
            probs = _apply_probability_margin(probs, margin=reward_margin)
            table[assignment] = probs
        if _table_satisfies_margin(table, hard_margin):
            return table
    _force_margin_table(table, 2, max(prob_margin, reward_margin))
    return table


_LOGIT_EPS = 1e-9


def _scale_probability(prob: float, *, logit_scale: float) -> float:
    """Apply a multiplicative scale to the Bernoulli logit to tune variance."""

    if math.isclose(logit_scale, 1.0):
        return float(prob)
    clipped = float(np.clip(prob, _LOGIT_EPS, 1.0 - _LOGIT_EPS))
    logit = math.log(clipped / (1.0 - clipped))
    scaled_logit = logit * logit_scale
    return 1.0 / (1.0 + math.exp(-scaled_logit))


def _apply_reward_logit_scale(
    reward_table: Dict[Assignment, np.ndarray],
    *,
    logit_scale: float,
) -> Dict[Assignment, np.ndarray]:
    if math.isclose(logit_scale, 1.0):
        return reward_table
    scaled: Dict[Assignment, np.ndarray] = {}
    for assignment, probs in reward_table.items():
        p1 = _scale_probability(float(probs[1]), logit_scale=logit_scale)
        scaled[assignment] = np.array([1.0 - p1, p1], dtype=float)
    return scaled


def _sample_reward_cpt_sparse(
    num_parents: int,
    ell: int,
    rng: np.random.Generator,
    *,
    sparse_fraction: float,
    sparse_separation: float,
    prob_margin: float,
    reward_margin: float,
    hard_margin: float,
) -> Dict[Assignment, np.ndarray]:
    """Generate reward CPT where only a small fraction of assignments differ markedly."""

    assignments = list(_all_assignments(num_parents, ell))
    base_mean = 0.5
    base_std = 0.05
    n_special = max(1, int(len(assignments) * sparse_fraction))
    for _ in range(_HARD_MARGIN_MAX_ATTEMPTS):
        special_idxs = set(rng.choice(len(assignments), size=n_special, replace=False).tolist())
        table: Dict[Assignment, np.ndarray] = {}
        for idx, assignment in enumerate(assignments):
            if idx in special_idxs:
                # Pull away from base_mean by at least sparse_separation (low or high).
                if rng.random() < 0.5:
                    low = max(0.01, base_mean - sparse_separation)
                    p1 = rng.uniform(low, max(low, base_mean - 1e-3))
                else:
                    high = min(0.99, base_mean + sparse_separation)
                    p1 = rng.uniform(min(high, base_mean + 1e-3), high)
            else:
                p1 = float(np.clip(rng.normal(base_mean, base_std), 0.01, 0.99))
            probs = np.array([1.0 - p1, p1], dtype=float)
            probs = _apply_probability_margin(probs, margin=prob_margin)
            probs = _apply_probability_margin(probs, margin=reward_margin)
            table[assignment] = probs
        if _table_satisfies_margin(table, hard_margin):
            return table
    _force_margin_table(table, 2, max(prob_margin, reward_margin))
    return table


def _sample_reward_cpt_clustered(
    num_parents: int,
    ell: int,
    rng: np.random.Generator,
    *,
    cluster_count: int,
    prob_margin: float,
    reward_margin: float,
    hard_margin: float,
) -> Dict[Assignment, np.ndarray]:
    """Generate reward CPT with clusters of similar assignments at distinct mean levels."""

    assignments = list(_all_assignments(num_parents, ell))
    for _ in range(_HARD_MARGIN_MAX_ATTEMPTS):
        centers = np.sort(rng.uniform(0.2, 0.8, size=cluster_count))
        cluster_assignments = rng.integers(0, cluster_count, size=len(assignments))
        table: Dict[Assignment, np.ndarray] = {}
        for idx, assignment in enumerate(assignments):
            center = float(centers[cluster_assignments[idx]])
            p1 = float(np.clip(rng.normal(center, 0.03), 0.01, 0.99))
            probs = np.array([1.0 - p1, p1], dtype=float)
            probs = _apply_probability_margin(probs, margin=prob_margin)
            probs = _apply_probability_margin(probs, margin=reward_margin)
            table[assignment] = probs
        if _table_satisfies_margin(table, hard_margin):
            return table
    _force_margin_table(table, 2, max(prob_margin, reward_margin))
    return table


def build_random_scm(config: CausalBanditConfig, *, rng: Optional[np.random.Generator] = None) -> CausalBanditInstance:
    """Construct a random SCM consistent with ``config``."""

    rng = rng or config.rng()
    x_nodes: List[NodeName] = [f"X{i}" for i in range(config.n)]
    reward_node: NodeName = "Y"
    nodes: List[NodeName] = x_nodes + [reward_node]

    parents: Dict[NodeName, List[NodeName]] = {name: [] for name in nodes}

    topo_order = _random_order(config.n, rng)
    cov_density = config.edge_prob if config.edge_prob_covariates is None else config.edge_prob_covariates
    for position, dst_idx in enumerate(topo_order):
        dst_name = x_nodes[dst_idx]
        for src_idx in topo_order[:position]:
            if rng.random() <= cov_density:
                parents[dst_name].append(x_nodes[src_idx])

    # Select reward parents; keep backwards-compatible default of exactly k parents.
    reward_density = config.edge_prob if config.edge_prob_to_reward is None else config.edge_prob_to_reward
    if config.edge_prob_to_reward is None:
        reward_parent_indices = sorted(rng.choice(config.n, size=config.k, replace=False).tolist())
    else:
        candidates = [idx for idx in range(config.n) if rng.random() <= reward_density]
        if len(candidates) < config.k:
            remaining = [idx for idx in range(config.n) if idx not in candidates]
            top_up = rng.choice(remaining, size=config.k - len(candidates), replace=False).tolist()
            candidates.extend(top_up)
        if len(candidates) > config.k:
            candidates = rng.choice(candidates, size=config.k, replace=False).tolist()
        reward_parent_indices = sorted(candidates)
    parents[reward_node] = [x_nodes[idx] for idx in reward_parent_indices]

    structural_functions: Dict[NodeName, Callable[[Dict[NodeName, int], np.random.Generator], int]] = {}
    if config.arm_heterogeneity_mode == "sparse":
        reward_table = _sample_reward_cpt_sparse(
            config.k,
            config.ell,
            rng,
            sparse_fraction=config.sparse_fraction,
            sparse_separation=config.sparse_separation,
            prob_margin=config.scm_epsilon,
            reward_margin=config.scm_delta,
            hard_margin=config.hard_margin,
        )
    elif config.arm_heterogeneity_mode == "clustered":
        reward_table = _sample_reward_cpt_clustered(
            config.k,
            config.ell,
            rng,
            cluster_count=config.cluster_count,
            prob_margin=config.scm_epsilon,
            reward_margin=config.scm_delta,
            hard_margin=config.hard_margin,
        )
    else:
        reward_table = _sample_reward_cpt(
            config.k,
            config.ell,
            rng,
            config.reward_alpha if config.scm_mode != "reference" else 1.0,
            config.reward_beta if config.scm_mode != "reference" else 1.0,
            mode=config.scm_mode,
            hard_margin=config.hard_margin,
            prob_margin=config.scm_epsilon,
            reward_margin=config.scm_delta,
        )
    reward_table = _apply_reward_logit_scale(reward_table, logit_scale=config.reward_logit_scale)

    def make_node_fn(name: NodeName, parent_names: Sequence[NodeName], table: Dict[Assignment, np.ndarray], domain: int):
        parent_tuple = tuple(parent_names)

        def f(par_vals: Dict[NodeName, int], local_rng: np.random.Generator) -> int:
            key = tuple(par_vals[parent] for parent in parent_tuple)
            probs = table[key]
            return int(local_rng.choice(domain, p=probs))

        return f

    for node in x_nodes:
        par_names = parents[node]
        cpt = _sample_covariate_cpt(
            len(par_names),
            config.ell,
            rng,
            mode=config.scm_mode,
            parent_effect=config.parent_effect,
            hard_margin=config.hard_margin,
            prob_margin=config.scm_epsilon,
        )
        structural_functions[node] = make_node_fn(node, par_names, cpt, config.ell)

    structural_functions[reward_node] = make_node_fn(reward_node, parents[reward_node], reward_table, 2)

    scm = SCM(nodes=nodes, parents=parents, f=structural_functions)
    reward_means: Dict[Assignment, float] = {
        assignment: float(probs[1]) for assignment, probs in reward_table.items()
    }

    return CausalBanditInstance(
        config=config,
        scm=scm,
        node_names=tuple(nodes),
        reward_node=reward_node,
        reward_parents=tuple(parents[reward_node]),
        reward_means=reward_means,
    )


def build_random_scm_with_gaps(
    config: CausalBanditConfig,
    *,
    rng: Optional[np.random.Generator] = None,
    tol: float = 0.9,
) -> Tuple[CausalBanditInstance, Dict[str, Any]]:
    """
    Construct a random SCM and (optionally) enforce target epsilon/delta gaps via rejection sampling.

    Returns the instance and a diagnostics dictionary containing measured gaps and attempts.
    """

    rng = rng or config.rng()
    if config.target_epsilon is None and config.target_delta is None:
        return build_random_scm(config, rng=rng), {}

    last_diag: Dict[str, Any] = {}
    last_instance: Optional[CausalBanditInstance] = None
    for attempt in range(1, config.max_rejection_attempts + 1):
        # Disable hard_margin proxy when targeting gaps directly.
        candidate_cfg = dataclasses.replace(config, hard_margin=0.0)
        instance = build_random_scm(candidate_cfg, rng=rng)
        gaps = instance.estimate_min_eps_delta(
            max_parent_scope_exact=2,
            n_mc=8000,
            alpha=0.05,
            max_hops=3,
            rng=rng,
        )
        measured_eps = float(gaps.eps)
        measured_delta = float(gaps.Delta)
        eps_ok = config.target_epsilon is None or measured_eps >= config.target_epsilon * tol
        delta_ok = config.target_delta is None or measured_delta >= config.target_delta * tol
        diag = {
            "measured_epsilon": measured_eps,
            "measured_delta": measured_delta,
            "attempts": attempt,
            "gap_satisfied": bool(eps_ok and delta_ok),
            "target_epsilon": config.target_epsilon,
            "target_delta": config.target_delta,
        }
        last_diag = diag
        last_instance = instance
        if eps_ok and delta_ok:
            return instance, diag
        if config.gap_enforcement_mode == "soft":
            # Accept but keep diagnostics to surface mismatch.
            return instance, diag
        if config.gap_enforcement_mode == "hard":
            continue
        # reject mode: keep looping until success or exhaustion

    if config.gap_enforcement_mode == "hard":
        raise ValueError(
            f"Failed to satisfy gap targets after {config.max_rejection_attempts} attempts "
            f"(last ε={last_diag.get('measured_epsilon', float('nan')):.4f}, "
            f"Δ={last_diag.get('measured_delta', float('nan')):.4f}). "
            "Relax targets or increase max_rejection_attempts."
        )
    if last_instance is None:
        raise RuntimeError("Gap-targeted SCM generation failed unexpectedly.")
    return last_instance, last_diag


# ---------------------------------------------------------------------------
# Intervention enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterventionArm:
    """Representation of an intervention on ``variables`` with categorical values."""

    variables: Tuple[int, ...]
    values: Tuple[int, ...]

    def size(self) -> int:
        return len(self.variables)

    def as_dict(self, node_names: Sequence[NodeName]) -> Dict[NodeName, int]:
        return {node_names[idx]: val for idx, val in zip(self.variables, self.values)}

    def as_intervention(self, node_names: Sequence[NodeName], *, name: Optional[str] = None) -> Intervention:
        hard = self.as_dict(node_names)
        label = name or f"do({', '.join(f'{k}={v}' for k, v in hard.items())})"
        return Intervention(name=label, hard=hard)


class InterventionSpace:
    """Complete enumeration of interventions with size exactly ``m`` (optionally <= m)."""

    def __init__(self, n: int, ell: int, m: int, *, include_lower: bool = False):
        if n <= 0 or ell <= 0:
            raise ValueError("n and ell must be positive")
        if not (0 <= m <= n):
            raise ValueError("m must satisfy 0 <= m <= n")
        self.n = n
        self.ell = ell
        self.m = m
        self.include_lower = include_lower
        self._arms: Tuple[InterventionArm, ...] = tuple(self._generate())
        self._prefix_counts: Dict[int, int] = self._compute_prefix_counts()

    def _sizes(self) -> Iterable[int]:
        if self.include_lower:
            return range(0, self.m + 1)
        return (self.m,)

    def _generate(self) -> Iterable[InterventionArm]:
        for size in self._sizes():
            for variables in itertools.combinations(range(self.n), size):
                if size == 0:
                    yield InterventionArm(tuple(), tuple())
                    continue
                for values in itertools.product(range(self.ell), repeat=size):
                    yield InterventionArm(tuple(variables), tuple(values))

    def _compute_prefix_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for size in self._sizes():
            total = sum(1 for arm in self._arms if arm.size() == size)
            counts[size] = total
        return counts

    def __len__(self) -> int:
        return len(self._arms)

    def __getitem__(self, idx: int) -> InterventionArm:
        return self._arms[idx]

    def counts(self) -> Dict[int, int]:
        """Return counts keyed by intervention size."""

        return dict(self._prefix_counts)

    def sample_indices(
        self,
        sample_size: int,
        rng: np.random.Generator,
        *,
        replace: bool = False,
    ) -> np.ndarray:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if not replace and sample_size > len(self):
            raise ValueError("sample_size cannot exceed population without replacement")
        return rng.choice(len(self), size=sample_size, replace=replace)

    def random_subset(
        self,
        sample_size: int,
        rng: np.random.Generator,
        *,
        replace: bool = False,
    ) -> List[InterventionArm]:
        indices = self.sample_indices(sample_size, rng, replace=replace)
        return [self._arms[i] for i in np.atleast_1d(indices)]

    def arms(self) -> Tuple[InterventionArm, ...]:
        """Return the full tuple of enumerated arms."""

        return self._arms


def arm_to_intervention(
    arm: InterventionArm,
    node_names: Sequence[NodeName],
    *,
    name: Optional[str] = None,
) -> Intervention:
    """Convenience wrapper mirroring ``InterventionArm.as_intervention``."""

    return arm.as_intervention(node_names, name=name)
