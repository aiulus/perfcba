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

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

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
    edge_prob: float = 0.3     # probability of an edge between covariates (respecting acyclicity)
    reward_alpha: float = 2.0  # parameters of Beta prior for reward Bernoulli means
    reward_beta: float = 2.0
    reward_logit_scale: float = 1.0  # temperature for Bernoulli logits (controls variance)
    scm_mode: str = "beta_dirichlet"  # sampling style for conditional probability tables
    parent_effect: float = 1.0        # reference-mode mixing between base and parent-specific CPDs
    seed: Optional[int] = None
    hard_margin: float = 0.0          # TV distance enforced between parent assignments

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


def _force_margin_table(table: Dict[Assignment, np.ndarray], ell: int) -> None:
    if ell <= 1:
        return
    assignments = list(table.keys())
    for idx, assignment in enumerate(assignments):
        probs = np.zeros(ell, dtype=float)
        anchor = idx % ell
        probs[anchor] = 1.0
        table[assignment] = probs


def _sample_covariate_cpt(
    num_parents: int,
    ell: int,
    rng: np.random.Generator,
    *,
    mode: str,
    parent_effect: float,
    hard_margin: float,
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
                table[assignment] = probs
        else:
            for assignment in assignments:
                table[assignment] = rng.dirichlet(base_alpha)
        if _table_satisfies_margin(table, hard_margin):
            return table
    _force_margin_table(table, ell)
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
            table[assignment] = probs
        if _table_satisfies_margin(table, hard_margin):
            return table
    _force_margin_table(table, 2)
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


def build_random_scm(config: CausalBanditConfig, *, rng: Optional[np.random.Generator] = None) -> CausalBanditInstance:
    """Construct a random SCM consistent with ``config``."""

    rng = rng or config.rng()
    x_nodes: List[NodeName] = [f"X{i}" for i in range(config.n)]
    reward_node: NodeName = "Y"
    nodes: List[NodeName] = x_nodes + [reward_node]

    parents: Dict[NodeName, List[NodeName]] = {name: [] for name in nodes}

    topo_order = _random_order(config.n, rng)
    for position, dst_idx in enumerate(topo_order):
        dst_name = x_nodes[dst_idx]
        for src_idx in topo_order[:position]:
            if rng.random() <= config.edge_prob:
                parents[dst_name].append(x_nodes[src_idx])

    reward_parent_indices = sorted(rng.choice(config.n, size=config.k, replace=False).tolist())
    parents[reward_node] = [x_nodes[idx] for idx in reward_parent_indices]

    structural_functions: Dict[NodeName, Callable[[Dict[NodeName, int], np.random.Generator], int]] = {}
    reward_table = _sample_reward_cpt(
        config.k,
        config.ell,
        rng,
        config.reward_alpha if config.scm_mode != "reference" else 1.0,
        config.reward_beta if config.scm_mode != "reference" else 1.0,
        mode=config.scm_mode,
        hard_margin=config.hard_margin,
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
