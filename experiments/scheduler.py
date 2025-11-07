"""Tau-scheduled orchestration of structure learning and exploitation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .causal_envs import CausalBanditInstance, InterventionArm
from .exploit import ArmBuilder, ArmID, ParentAwareUCB
from .structure import StructureLearner, StructureStepResult

SchedulerMode = Literal["interleaved", "etc", "two_phase"]


@dataclass
class RoundLog:
    t: int
    mode: Literal["structure", "exploit"]
    arm: InterventionArm
    reward: float
    expected_mean: float
    parent_set: Tuple[int, ...]
    arm_count: int


@dataclass
class RunSummary:
    logs: List[RoundLog]
    structure_steps: int
    exploit_steps: int
    final_parent_set: Tuple[int, ...]
    finished_discovery_round: Optional[int]


class TauSchedulerBase:
    """Coordinate structure learning and exploitation under a tau budget."""

    def __init__(
        self,
        *,
        instance: CausalBanditInstance,
        structure: StructureLearner,
        arm_builder: ArmBuilder,
        policy: ParentAwareUCB,
        tau: float,
        horizon: int,
    ) -> None:
        self.instance = instance
        self.structure = structure
        self.arm_builder = arm_builder
        self.policy = policy
        self.horizon = int(max(0, horizon))
        self.tau = max(0.0, min(1.0, tau))
        self.structure_cap = int(self.tau * self.horizon)

        self.logs: List[RoundLog] = []
        self.structure_steps = 0
        self.exploit_steps = 0
        self.t = 0
        self.finished_discovery_round: Optional[int] = None

        self._current_parent_set: Tuple[int, ...] = tuple(structure.parent_set())
        self._arm_count = 0
        self._arm_means: Dict[ArmID, float] = {}
        self._policy_initialized = False

    def run(self, rng: np.random.Generator) -> RunSummary:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _arm_id(self, arm: InterventionArm) -> ArmID:
        return (arm.variables, arm.values)

    def _refresh_arms(self, parent_set: Sequence[int], rng: np.random.Generator) -> None:
        arms, mean_lookup = self.arm_builder.build(parent_set, rng)
        warm_start = self.policy if self._policy_initialized else None
        self.policy.set_arms(arms, warm_start_from=warm_start)
        self._policy_initialized = True
        self._current_parent_set = tuple(sorted(parent_set))
        self._arm_means = dict(mean_lookup)
        self._arm_count = len(arms)

    def _log(
        self,
        *,
        mode: Literal["structure", "exploit"],
        arm: InterventionArm,
        reward: float,
        expected_mean: float,
    ) -> None:
        self.logs.append(
            RoundLog(
                t=self.t,
                mode=mode,
                arm=arm,
                reward=reward,
                expected_mean=expected_mean,
                parent_set=self._current_parent_set,
                arm_count=self._arm_count,
            )
        )

    def _structure_step(self, result: StructureStepResult) -> None:
        self.structure_steps += 1
        self.t += 1
        self._current_parent_set = tuple(self.structure.parent_set())
        if not self.structure.needs_structure_step() and self.finished_discovery_round is None:
            self.finished_discovery_round = self.t
        self._log(mode="structure", arm=result.arm, reward=result.reward, expected_mean=result.expected_mean)

    def _exploit_step(self, rng: np.random.Generator) -> None:
        if self._arm_count == 0:
            self._refresh_arms(self._current_parent_set, rng)
        arm_idx, arm = self.policy.select(rng)
        reward = self.instance.sample_reward(arm, rng)
        self.policy.observe(arm_idx, reward)
        self.exploit_steps += 1
        self.t += 1
        expected_mean = self._arm_means.get(self._arm_id(arm), float("nan"))
        self._log(mode="exploit", arm=arm, reward=reward, expected_mean=expected_mean)

    def _finalize(self) -> RunSummary:
        assert self.structure_steps <= self.structure_cap, "Structure budget exceeded"
        return RunSummary(
            logs=self.logs,
            structure_steps=self.structure_steps,
            exploit_steps=self.exploit_steps,
            final_parent_set=self._current_parent_set,
            finished_discovery_round=self.finished_discovery_round,
        )


class TauSchedulerETC(TauSchedulerBase):
    """Explore-then-commit scheduler (structure first, then exploitation)."""

    def __init__(self, *args, use_full_budget: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_full_budget = use_full_budget

    def run(self, rng: np.random.Generator) -> RunSummary:
        while self.t < self.horizon and self.structure_steps < self.structure_cap:
            needs_step = self.structure.needs_structure_step()
            if not needs_step:
                if self.finished_discovery_round is None:
                    self.finished_discovery_round = self.t
                if not self.use_full_budget:
                    break
            result = self.structure.step(rng)
            self._structure_step(result)

        # Final parent estimate informs exploitation phase
        self._refresh_arms(self._current_parent_set, rng)
        while self.t < self.horizon:
            self._exploit_step(rng)

        return self._finalize()



class TauSchedulerInterleaved(TauSchedulerBase):
    """Per-round scheduler that interleaves structure and exploitation."""

    def run(self, rng: np.random.Generator) -> RunSummary:
        self._refresh_arms(self._current_parent_set, rng)

        while self.t < self.horizon:
            take_structure = (
                self.structure_steps < self.structure_cap and self.structure.needs_structure_step()
            )
            if take_structure:
                result = self.structure.step(rng)
                # refresh arms if parent set changed after the step
                previous_parents = self._current_parent_set
                self._structure_step(result)
                if self._current_parent_set != previous_parents:
                    self._refresh_arms(self._current_parent_set, rng)
                continue

            if not self.structure.needs_structure_step() and self.finished_discovery_round is None:
                self.finished_discovery_round = self.t
            self._exploit_step(rng)

        return self._finalize()


def build_scheduler(
    *,
    mode: SchedulerMode,
    instance: CausalBanditInstance,
    structure: StructureLearner,
    arm_builder: ArmBuilder,
    policy: ParentAwareUCB,
    tau: float,
    horizon: int,
    use_full_budget: bool = True,
) -> TauSchedulerBase:
    normalized_mode = "etc" if mode == "two_phase" else mode
    if normalized_mode == "etc":
        return TauSchedulerETC(
            instance=instance,
            structure=structure,
            arm_builder=arm_builder,
            policy=policy,
            tau=tau,
            horizon=horizon,
            use_full_budget=use_full_budget,
        )
    if normalized_mode == "interleaved":
        return TauSchedulerInterleaved(
            instance=instance,
            structure=structure,
            arm_builder=arm_builder,
            policy=policy,
            tau=tau,
            horizon=horizon,
        )
    raise ValueError(f"Unknown scheduler mode: {mode}")
