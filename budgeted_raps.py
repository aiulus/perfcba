"""Budgeted RAPS integration for perfcba.

This module embeds the original RAPS + UCB implementation released by
BorealisAI (https://github.com/BorealisAI/raps) and adapts it to the PCM
interface defined in :mod:`causal_bandits`.  The only intentional behavioural
change is the addition of the ``tau`` budget-allocation parameter that caps the
fraction of the interaction horizon spent on structure discovery; the core
logic otherwise mirrors the reference implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import ceil, floor, log
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

from .causal_bandits import PCM, RAPSParams


# ============================================================================
# UCB primitives (exact copy of the upstream implementation)
# ============================================================================


class CausalBanditAlg(ABC):
    """Abstract causal bandit algorithm."""

    def __init__(self, num_nodes: int, domain_size: int = 2, batch_size: int = 20, reward_node: int = -1) -> None:
        self.num_nodes = int(num_nodes)
        self.domain_size = int(domain_size)
        self.batch_size = int(batch_size)
        self.reward_node = reward_node if reward_node >= 0 else self.num_nodes + reward_node

    @property
    def name(self) -> str:
        """Return the algorithm name."""

        return self.__class__.__name__.lower()

    @abstractmethod
    def get_arms(self) -> np.ndarray:
        """Return the interventions to perform at the next interaction."""

    @abstractmethod
    def update_stats(self, arms: np.ndarray, rewards: np.ndarray) -> None:
        """Update internal statistics after receiving ``rewards``."""

    def start_run(self, horizon: Optional[int]) -> None:
        """Hook executed once before interacting with the bandit."""

        self.horizon = horizon


@dataclass
class UCBStats:
    """Statistics of a UCB-type algorithm."""

    means: np.ndarray
    conf_bounds: np.ndarray
    npulls: np.ndarray

    def __init__(self, narms: int, batch_size: int = 100) -> None:
        self.means = np.zeros([batch_size, narms], dtype=float)
        self.conf_bounds = np.zeros([batch_size, narms], dtype=float)
        self.npulls = np.zeros([batch_size, narms], dtype=int)

    def reset(self) -> None:
        """Reset the statistics."""

        self.means.fill(0)
        self.conf_bounds.fill(0)
        self.npulls.fill(0)


def index_to_values(index: np.ndarray, arm_variables: np.ndarray, domain_size: int = 2) -> np.ndarray:
    """Convert a flattened arm index to intervention values."""

    result = np.zeros((index.shape[0], len(arm_variables)), dtype=int)
    index = np.copy(index)
    for i in range(len(arm_variables) - 1, -1, -1):
        result[..., i] = index % domain_size
        index //= domain_size
    return result


def values_to_index(values: np.ndarray, arm_variables: np.ndarray, domain_size: int = 2) -> np.ndarray:
    """Convert intervention values to the corresponding arm index."""

    powers = domain_size ** np.arange(len(arm_variables) - 1, -1, -1)
    return np.sum(values[..., arm_variables] * powers, -1)


class UCB(CausalBanditAlg):
    """UCB with interventions of size K^n."""

    def __init__(self, arm_variables: Optional[np.ndarray], **kwargs: Union[int, float]) -> None:
        super().__init__(**kwargs)
        if arm_variables is None:
            mask = np.ones((self.num_nodes), dtype=bool)
            mask[self.reward_node] = False
            arm_variables = np.where(mask)[0]
        self.arm_variables = arm_variables
        self.stats = UCBStats(self.narms, self.batch_size)
        self.timestep = 0

    @classmethod
    def from_pcm(cls, pcm, reward_node: Optional[int] = None):
        """Create an instance from the reference PCM class."""

        return cls(
            arm_variables=None,
            num_nodes=len(pcm.adj),
            reward_node=reward_node,
            domain_size=pcm.domain_size,
            batch_size=pcm.batch_size,
        )

    @property
    def narms(self) -> int:
        """Return the number of arms of the bandit this algo interacts with."""

        return self.domain_size ** len(self.arm_variables)

    def get_arms(self) -> np.ndarray:
        arms = np.full((self.batch_size, self.num_nodes), self.domain_size)
        if self.timestep < self.narms:
            index = np.full(self.batch_size, self.timestep)
        else:
            index = np.argmax(self.stats.conf_bounds, -1)
        arms[:, self.arm_variables] = index_to_values(index, self.arm_variables, self.domain_size)
        return arms

    def update_stats(self, arms: np.ndarray, rewards: np.ndarray) -> None:
        arms_index = values_to_index(arms, self.arm_variables, self.domain_size)
        batch_index = np.arange(self.batch_size)
        npulls = self.stats.npulls[batch_index, arms_index]
        self.stats.means[batch_index, arms_index] = (
            npulls / (npulls + 1) * self.stats.means[batch_index, arms_index]
            + rewards[..., self.reward_node] / (npulls + 1)
        )
        self.stats.conf_bounds[batch_index, arms_index] = (
            self.stats.means[batch_index, arms_index]
            + np.sqrt(2 * np.log(self.timestep + 1) / (npulls + 1))
        )
        self.stats.npulls[batch_index, arms_index] += 1
        self.timestep += 1


# ============================================================================
# RAPS logic
# ============================================================================


@dataclass
class RAPSHistory:
    """Interaction history tracked by RAPS."""

    observations: Union[List[np.ndarray], np.ndarray] = field(default_factory=list)
    interventions: Union[List[np.ndarray], np.ndarray] = field(default_factory=list)


@dataclass
class LastAncestorsStats:
    """Statistics relative to the last ancestor discovered by RAPS."""

    ancestors: np.ndarray
    descendants: np.ndarray
    history: RAPSHistory


@dataclass
class RAPSStats:
    """Internal statistics tracked by RAPS."""

    candidates: np.ndarray
    banned_candidates: np.ndarray
    nodes: Union[None, np.ndarray]
    last_ancestors_stats: Union[None, LastAncestorsStats]
    parents: np.ndarray
    nodes_value: Union[None, int]
    parents_value_index: Union[None, int]
    history: RAPSHistory

    def __init__(self, batch_size: int, num_nodes: int, reward_node: int) -> None:
        self.candidates = np.ones((batch_size, num_nodes), dtype=bool)
        self.candidates[:, reward_node] = False
        self.banned_candidates = np.zeros((batch_size, num_nodes), dtype=bool)
        self.nodes = None
        self.last_ancestors_stats = None
        self.parents = np.zeros((batch_size, num_nodes), dtype=bool)
        self.nodes_value = None
        self.parents_value_index = None
        self.history = RAPSHistory()

    def reset_candidates(self, reward_node: int) -> None:
        """Reset the candidate nodes."""

        self.candidates.fill(True)
        self.candidates[:, reward_node] = False
        self.candidates[self.parents] = False
        self.candidates[self.banned_candidates] = False


class RAPSUCB(CausalBanditAlg):
    """Randomized parent search algorithm + UCB."""

    def __init__(
        self,
        eps: float,
        gap: float,
        delta: float,
        tau: float = 1.0,
        *,
        strict_tau: bool = False,
        **kwargs: Union[int, float],
    ) -> None:
        super().__init__(**kwargs)
        self.gap = float(gap)
        self.delta = float(delta)
        self.eps = float(eps)
        self.tau = max(0.0, min(1.0, float(tau)))
        self.strict_tau = bool(strict_tau)
        self.budget = self.compute_budget(self.eps, self.gap, self.delta, self.num_nodes, self.domain_size)
        self.stats = RAPSStats(self.batch_size, self.num_nodes, self.reward_node)
        self.ucb: Optional[UCB] = None
        self._ucb_parent_signature: Optional[Tuple[int, ...]] = None
        self.timestep = 0
        self.structure_steps = 0
        self.structure_budget_cap: Optional[int] = None
        self.total_rounds: Optional[int] = None
        self.discovery_active = True
        self._last_action = "observe"

    @staticmethod
    def compute_budget(eps: float, gap: float, delta: float, num_nodes: int, domain_size: int = 2) -> int:
        """Compute the batch size required by the finite-sample analysis."""

        return max(
            ceil(
                32
                * log(8 * num_nodes * domain_size * (domain_size + 1) ** num_nodes / delta)
                / gap**2
            ),
            ceil(
                8
                * log(
                    8 * num_nodes**2 * domain_size**2 * (domain_size + 1) ** num_nodes / delta
                )
                / eps**2
            ),
        )

    def start_run(self, horizon: Optional[int]) -> None:
        super().start_run(horizon)
        self.total_rounds = horizon
        if horizon is None:
            self.structure_budget_cap = None
        else:
            cap = floor(self.tau * horizon)
            if self.strict_tau:
                cap -= self.budget
            self.structure_budget_cap = max(0, cap)
        self.structure_steps = 0
        self.discovery_active = True
        self._last_action = "observe"
        self._ucb_parent_signature = None
        self.ucb = None

    @classmethod
    def from_pcm(cls, pcm, reward_node: Optional[int] = None, delta: float = 0.01, tau: float = 1.0, **kwargs):
        """Create an instance from the reference PCM class."""

        if reward_node is None:
            reward_node = len(pcm.adj) - 1
        eps, gap = pcm.min_eps_gap(reward_node)
        return cls(
            eps,
            gap,
            delta,
            tau=tau,
            num_nodes=len(pcm.adj),
            reward_node=reward_node,
            domain_size=pcm.domain_size,
            batch_size=pcm.batch_size,
            **kwargs,
        )

    def get_arms(self) -> np.ndarray:
        if self.timestep < self.budget:
            self._last_action = "observe"
            return np.full((self.batch_size, self.num_nodes), self.domain_size)

        if self._should_take_structure_step():
            self._last_action = "structure"
            self.structure_steps += 1
            return self._raps_arms()

        self._last_action = "exploit"
        self._ensure_ucb_policy()
        return self.ucb.get_arms()

    def _raps_arms(self) -> np.ndarray:
        arms = np.full((self.batch_size, self.num_nodes), self.domain_size)
        if self.stats.nodes is None:
            raise RuntimeError("Stats nodes must be initialized before requesting structure arms.")
        arms[np.arange(self.batch_size), self.stats.nodes] = self.stats.nodes_value
        if self.stats.parents_value_index is not None:
            parent_indices = np.where(self.stats.parents)[1]
            arms[self.stats.parents] = np.reshape(
                index_to_values(
                    self.stats.parents_value_index,
                    parent_indices,
                    self.domain_size,
                ),
                -1,
            )
        return arms

    def _current_parent_indices(self) -> np.ndarray:
        parent_indices = np.where(self.stats.parents)[1]
        if parent_indices.size == 0:
            return np.asarray([], dtype=int)
        return np.unique(parent_indices)

    def _ensure_ucb_policy(self) -> None:
        parent_indices = self._current_parent_indices()
        signature = tuple(parent_indices.tolist())
        if self.ucb is not None and signature == self._ucb_parent_signature:
            return
        self.ucb = UCB(
            parent_indices,
            num_nodes=self.num_nodes,
            domain_size=self.domain_size,
            batch_size=self.batch_size,
            reward_node=self.reward_node,
        )
        self._ucb_parent_signature = signature

    def _structure_budget_allows_step(self) -> bool:
        if self.structure_budget_cap is None:
            return True
        return self.structure_steps < self.structure_budget_cap

    def worth_struct_step(self) -> bool:
        """Heuristic deciding if a structure step is still valuable."""

        return bool(np.any(self.stats.candidates))

    def _should_take_structure_step(self) -> bool:
        if self.timestep < self.budget:
            return False
        if not self.discovery_active:
            return False
        if self.stats.nodes is None:
            return False
        if not self._structure_budget_allows_step():
            return False
        return self.worth_struct_step()

    def _mark_discovery_complete(self) -> None:
        self.discovery_active = False
        self.stats.nodes = None
        self.stats.last_ancestors_stats = None
        self._ensure_ucb_policy()

    def is_at_reward_ancestor(self) -> np.ndarray:
        """Return True if the current node is a reward ancestor."""

        assert self.should_update_nodes(), (
            len(self.stats.history.interventions),
            self.budget,
        )
        observations, interventions = map(
            np.asarray,
            (
                self.stats.history.observations,
                self.stats.history.interventions,
            ),
        )
        parents_domain = np.squeeze(self.domain_size ** np.sum(self.stats.parents, 1))
        observations = np.reshape(
            observations[..., self.reward_node],
            (1, parents_domain, self.budget, self.batch_size),
        )
        obs_reward = np.mean(observations, 2)
        interventions = np.reshape(
            interventions[..., self.reward_node],
            (self.domain_size, parents_domain, self.budget, self.batch_size),
        )
        interv_reward = np.mean(interventions, 2)
        return np.any(np.abs(obs_reward - interv_reward) > self.gap / 2, axis=(0, 1))

    def descendants(self) -> np.ndarray:
        """Return the estimated descendants of the current node."""

        assert self.should_update_nodes(), (
            len(self.stats.history.interventions),
            self.budget,
        )
        observations = np.asarray(self.stats.history.observations)
        parents_domain = np.squeeze(self.domain_size ** np.sum(self.stats.parents, 1))
        observations = np.reshape(
            observations,
            (1, parents_domain, self.budget, self.batch_size, self.num_nodes),
        )
        observations = np.mean(observations, 2)
        interventions = np.reshape(
            np.asarray(self.stats.history.interventions),
            (
                self.domain_size,
                parents_domain,
                self.budget,
                self.batch_size,
                self.num_nodes,
            ),
        )
        interventions = np.mean(interventions, 2)
        result = np.any(np.abs(observations - interventions) > self.eps / 2, axis=(0, 1))
        result[:, self.reward_node] = False
        result[self.stats.parents] = False
        return result

    def should_update_nodes(self) -> bool:
        """Return True if it is time to update the intervened node."""

        num_parents = np.squeeze(np.sum(self.stats.parents, 1))
        num_interventions = len(self.stats.history.interventions)
        return (
            num_interventions
            and (
                num_interventions
                % (self.domain_size ** (num_parents + 1) * self.budget)
                == 0
            )
        )

    def should_update_nodes_value(self) -> bool:
        """Return True if it is time to update the node value."""

        num_parents = np.sum(self.stats.parents, -1)
        num_interventions = len(self.stats.history.interventions)
        return (
            num_interventions > 0
            and (
                num_interventions
                % (self.domain_size ** num_parents * self.budget)
                == 0
            )
        )

    def should_update_parents_value_index(self) -> bool:
        """Return True if the parents value index should be updated."""

        num_interventions = len(self.stats.history.interventions)
        return num_interventions > 0 and num_interventions % self.budget == 0

    def sample_nodes(self) -> np.ndarray:
        """Sample candidate nodes."""

        return np.asarray(
            [np.random.choice(*np.where(candidates)) for candidates in self.stats.candidates]
        )

    def update_stats(self, arms: np.ndarray, rewards: np.ndarray) -> None:
        self.timestep += 1
        if self._last_action == "exploit":
            self._ensure_ucb_policy()
            if self.ucb is None:
                raise RuntimeError("UCB policy must be initialized before exploitation.")
            self.ucb.update_stats(arms, rewards)
            return
        intervened_mask = arms != self.domain_size
        if self.timestep <= self.budget:
            if np.any(intervened_mask):
                raise ValueError("Observation phase cannot contain interventions.")
            self.stats.history.observations.append(rewards)
            if self.timestep == self.budget:
                self.stats.nodes_value = 0
                self.stats.nodes = self.sample_nodes()
            return

        if np.any(np.sum(intervened_mask, 1) != np.sum(self.stats.parents, 1) + 1):
            raise ValueError("Mismatch between intervened nodes and tracked parents.")
        self.stats.history.interventions.append(rewards)
        if self.should_update_nodes_value():
            self.stats.nodes_value = (self.stats.nodes_value + 1) % self.domain_size
        if self.should_update_parents_value_index():
            parents_sum = np.sum(self.stats.parents, 1)
            self.stats.parents_value_index = (
                ((self.stats.parents_value_index or 0) + 1) % (self.domain_size ** parents_sum)
            )
        if not self.should_update_nodes():
            return
        descendants = self.descendants()
        if np.squeeze(self.is_at_reward_ancestor()):
            descendants[np.arange(self.batch_size), self.stats.nodes] = False
            self.stats.candidates &= descendants
            self.stats.last_ancestors_stats = LastAncestorsStats(
                np.copy(self.stats.nodes),
                descendants,
                RAPSHistory(interventions=np.array(self.stats.history.interventions)),
            )
        else:
            self.stats.candidates[descendants] = False
            self.stats.banned_candidates[descendants] = True

        if np.squeeze(np.any(self.stats.candidates, 1)):
            self.stats.history.interventions.clear()
            self.stats.nodes = self.sample_nodes()
            return
        if self.stats.last_ancestors_stats is not None:
            self.stats.banned_candidates[self.stats.last_ancestors_stats.descendants] = True
            if np.any(
                self.stats.parents[
                    np.arange(self.batch_size),
                    self.stats.last_ancestors_stats.ancestors,
                ]
            ):
                raise RuntimeError("Parents already discovered cannot be overwritten.")
            self.stats.parents[
                np.arange(self.batch_size),
                self.stats.last_ancestors_stats.ancestors,
            ] = True
            self.stats.history.observations = self.stats.last_ancestors_stats.history.interventions

        self.stats.reset_candidates(self.reward_node)
        if self.stats.last_ancestors_stats is None or not np.squeeze(np.any(self.stats.candidates, 1)):
            self._mark_discovery_complete()
        else:
            self.stats.nodes = self.sample_nodes()
            self.stats.last_ancestors_stats = None
            self.stats.history.interventions.clear()


# ============================================================================
# Adapter to the causal_bandits PCM interface
# ============================================================================

Assignment = MutableMapping[str, int]


@dataclass
class BudgetedRAPSStep:
    """Single interaction recorded while running Budgeted RAPS."""

    round: int
    action: str
    interventions: List[Assignment]
    rewards: np.ndarray
    parent_mask: np.ndarray


@dataclass
class BudgetedRAPSResult:
    """Aggregate information returned by :func:`run_budgeted_raps`."""

    parents: Tuple[str, ...]
    parent_mask: np.ndarray
    trace: List[BudgetedRAPSStep]
    reward_history: np.ndarray
    action_counts: Dict[str, int]
    algorithm: RAPSUCB


class _PCMAdapter:
    """Convert get_arms/update_stats calls into PCM sampling calls."""

    def __init__(self, pcm: PCM, batch_size: int) -> None:
        self.pcm = pcm
        self.batch_size = int(batch_size)
        self.nodes = list(pcm.V)
        if not self.nodes:
            raise ValueError("PCM must expose at least one node.")
        if pcm.reward_node not in self.nodes:
            raise ValueError("pcm.reward_node must be one of pcm.V")
        self.node_to_index = {name: idx for idx, name in enumerate(self.nodes)}
        self.domain_size = int(pcm.K)
        if self.domain_size <= 0:
            raise ValueError("pcm.K must be positive.")
        self.reward_index = self.node_to_index[pcm.reward_node]
        self.num_nodes = len(self.nodes)

    def _arms_to_assignments(self, arms: np.ndarray) -> List[Assignment]:
        assignments: List[Assignment] = []
        for row in arms:
            assignment: Dict[str, int] = {}
            for idx, value in enumerate(row):
                if value != self.domain_size:
                    assignment[self.nodes[idx]] = int(value)
            assignments.append(assignment)
        return assignments

    def _samples_to_array(self, samples: Sequence[MutableMapping[str, int]]) -> np.ndarray:
        arr = np.zeros((len(samples), self.num_nodes), dtype=int)
        for i, sample in enumerate(samples):
            for node in self.nodes:
                if node not in sample:
                    raise KeyError(f"Sample is missing node '{node}'.")
                arr[i, self.node_to_index[node]] = int(sample[node])
        return arr

    def pull(self, arms: np.ndarray) -> Tuple[np.ndarray, List[Assignment]]:
        if arms.shape != (self.batch_size, self.num_nodes):
            raise ValueError(
                f"Expected arms with shape {(self.batch_size, self.num_nodes)}, received {arms.shape}."
            )
        assignments = self._arms_to_assignments(arms)
        results = np.zeros((self.batch_size, self.num_nodes), dtype=int)

        obs_rows = [idx for idx, assignment in enumerate(assignments) if not assignment]
        if obs_rows:
            samples = self.pcm.observe(len(obs_rows))
            if len(samples) != len(obs_rows):
                raise ValueError("pcm.observe must return exactly B samples.")
            arr = self._samples_to_array(samples)
            for offset, row_idx in enumerate(obs_rows):
                results[row_idx] = arr[offset]

        for idx, assignment in enumerate(assignments):
            if not assignment:
                continue
            samples = self.pcm.intervene(assignment, B=1)
            if len(samples) != 1:
                raise ValueError("pcm.intervene must return exactly B samples.")
            results[idx] = self._samples_to_array(samples)[0]

        return results, assignments


def _parent_names(parent_mask: np.ndarray, nodes: Sequence[str]) -> Tuple[str, ...]:
    indices = np.where(parent_mask)[0]
    return tuple(nodes[idx] for idx in indices)


def run_budgeted_raps(
    pcm: PCM,
    params: RAPSParams,
    horizon: int,
    *,
    tau: float = 1.0,
    strict_tau: bool = False,
    batch_size: int = 1,
    delta: Optional[float] = None,
) -> BudgetedRAPSResult:
    """Execute the budgeted RAPS algorithm against ``pcm``.

    Parameters
    ----------
    pcm:
        Probabilistic causal model providing ``observe`` and ``intervene`` oracles.
    params:
        Finite-sample parameters used in the RAPS analysis.  ``params.eps`` and
        ``params.Delta`` map directly to the ``eps`` and ``gap`` constants used by
        RAPSUCB.
    horizon:
        Total number of rounds to interact with the PCM.
    tau:
        Fraction of the horizon allocated to structure discovery.  ``tau=1``
        recovers the original algorithm, ``tau=0`` disables structure steps after
        the initial observational budget is spent.
    strict_tau:
        If True, counts the initial observational budget ``B`` against the
        tau-constrained discovery budget, leaving at most ``floor(tau*horizon)-B``
        rounds for interventional structure steps.
    batch_size:
        Number of parallel samples drawn per round.  This mirrors the batch size
        used in the reference code and defaults to ``1`` for compatibility with
        the :mod:`causal_bandits` PCM interface.
    delta:
        Confidence parameter.  Defaults to ``params.delta`` when not provided.

    Returns
    -------
    BudgetedRAPSResult
        Summary containing the discovered parent set, the interaction trace and
        references to the underlying :class:`RAPSUCB` state for further analysis.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    adapter = _PCMAdapter(pcm, batch_size=batch_size)
    algorithm = RAPSUCB(
        eps=params.eps,
        gap=params.Delta,
        delta=params.delta if delta is None else delta,
        tau=tau,
        strict_tau=strict_tau,
        num_nodes=adapter.num_nodes,
        reward_node=adapter.reward_index,
        domain_size=adapter.domain_size,
        batch_size=adapter.batch_size,
    )
    algorithm.start_run(horizon)

    trace: List[BudgetedRAPSStep] = []
    reward_history = np.zeros(horizon, dtype=float)
    action_counts: Dict[str, int] = {"observe": 0, "structure": 0, "exploit": 0}

    for round_idx in range(1, horizon + 1):
        arms = algorithm.get_arms()
        samples, assignments = adapter.pull(arms)
        algorithm.update_stats(arms, samples)
        action_counts[algorithm._last_action] = action_counts.get(algorithm._last_action, 0) + 1
        rewards = samples[:, adapter.reward_index]
        reward_history[round_idx - 1] = float(np.mean(rewards))
        trace.append(
            BudgetedRAPSStep(
                round=round_idx,
                action=algorithm._last_action,
                interventions=[dict(assignment) for assignment in assignments],
                rewards=rewards.copy(),
                parent_mask=algorithm.stats.parents.copy(),
            )
        )

    parent_mask = np.any(algorithm.stats.parents, axis=0)
    parents = _parent_names(parent_mask, adapter.nodes)
    return BudgetedRAPSResult(
        parents=parents,
        parent_mask=parent_mask,
        trace=trace,
        reward_history=reward_history,
        action_counts=action_counts,
        algorithm=algorithm,
    )


__all__ = [
    "BudgetedRAPSResult",
    "BudgetedRAPSStep",
    "CausalBanditAlg",
    "RAPSUCB",
    "RAPSStats",
    "RAPSHistory",
    "UCB",
    "UCBStats",
    "index_to_values",
    "run_budgeted_raps",
]
