"""Structure learning utilities for budgeted causal bandit experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .causal_envs import CausalBanditInstance, InterventionArm


@dataclass(frozen=True)
class StructureConfig:
    """Hyper-parameters controlling structure-learning statistics."""

    effect_threshold: float = 0.05
    min_samples_per_value: int = 20
    max_steps: Optional[int] = None


@dataclass
class StructureStepResult:
    """Telemetry emitted after each structure-learning step."""

    arm: InterventionArm
    reward: float
    expected_mean: float
    candidate_index: int
    value: int
    parent_discovered: bool


class StructureLearner:
    """Abstract interface for sequential structure learning."""

    def needs_structure_step(self) -> bool:  # pragma: no cover - defined by subclasses
        raise NotImplementedError

    def step(self, rng: np.random.Generator) -> StructureStepResult:  # pragma: no cover
        raise NotImplementedError

    def parent_set(self) -> Tuple[int, ...]:  # pragma: no cover
        raise NotImplementedError

    def completed(self) -> bool:
        return not self.needs_structure_step()


class RAPSLearner(StructureLearner):
    """
    Simplified proxy for the RAPS parent-discovery routine.

    The learner independently intervenes on individual covariates and monitors
    the induced reward distribution. Once the difference between the largest
    and smallest conditional means exceeds the configured threshold, the node
    is declared a reward parent.
    """

    def __init__(
        self,
        instance: CausalBanditInstance,
        config: Optional[StructureConfig] = None,
    ) -> None:
        self.instance = instance
        self.config = config or StructureConfig()
        self._n = instance.config.n
        self._ell = instance.config.ell
        self._parents: List[int] = []
        self._tested_non_parents: set[int] = set()
        self._stats: Dict[int, Dict[int, List[float]]] = {
            idx: {val: [0.0, 0] for val in range(self._ell)} for idx in range(self._n)
        }
        self._steps_taken = 0

    def parent_set(self) -> Tuple[int, ...]:
        return tuple(sorted(self._parents))

    def needs_structure_step(self) -> bool:
        if self.config.max_steps is not None and self._steps_taken >= self.config.max_steps:
            return False
        if len(self._parents) >= self.instance.config.k:
            return False
        remaining = [
            idx
            for idx in range(self._n)
            if idx not in self._parents and idx not in self._tested_non_parents
        ]
        return bool(remaining)

    def _select_candidate(self) -> int:
        candidates = [
            idx
            for idx in range(self._n)
            if idx not in self._parents and idx not in self._tested_non_parents
        ]
        if not candidates:
            return 0
        # prefer nodes with the fewest collected samples
        priorities = []
        for idx in candidates:
            counts = [self._stats[idx][val][1] for val in range(self._ell)]
            priorities.append((min(counts), idx))
        priorities.sort()
        return priorities[0][1]

    def _select_value(self, idx: int) -> int:
        stats = self._stats[idx]
        counts = {val: stats[val][1] for val in stats}
        return min(counts.items(), key=lambda item: item[1])[0]

    def step(self, rng: np.random.Generator) -> StructureStepResult:
        idx = self._select_candidate()
        value = self._select_value(idx)
        arm = InterventionArm(variables=(idx,), values=(value,))
        reward = self.instance.sample_reward(arm, rng)
        expected_mean = self.instance.estimate_arm_mean(arm, rng, n_mc=512)
        total, count = self._stats[idx][value]
        total += reward
        count += 1
        self._stats[idx][value] = [total, count]
        self._steps_taken += 1
        parent_discovered = False
        if self._ready_to_decide(idx):
            parent_discovered = self._evaluate_candidate(idx)
        return StructureStepResult(
            arm=arm,
            reward=reward,
            expected_mean=expected_mean,
            candidate_index=idx,
            value=value,
            parent_discovered=parent_discovered,
        )

    def _ready_to_decide(self, idx: int) -> bool:
        return all(self._stats[idx][val][1] >= self.config.min_samples_per_value for val in range(self._ell))

    def _evaluate_candidate(self, idx: int) -> bool:
        means = [
            (self._stats[idx][val][0] / max(1, self._stats[idx][val][1]))
            for val in range(self._ell)
        ]
        effect = max(means) - min(means)
        if effect >= self.config.effect_threshold:
            self._parents.append(idx)
            return True
        self._tested_non_parents.add(idx)
        return False
