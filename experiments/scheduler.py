"""Tau-scheduled orchestration of structure learning and exploitation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence

import numpy as np

from .causal_envs import CausalBanditInstance, InterventionArm
from .exploit import ArmBuilder, ArmInfo, ParentAwareUCB
from .structure import StructureLearner

SchedulerMode = Literal["interleaved", "two_phase"]


@dataclass
class RoundLog:
    t: int
    mode: Literal["structure", "exploit"]
    arm: InterventionArm
    reward: float
    expected_mean: float
    parent_set: Sequence[int]


@dataclass
class RunSummary:
    logs: List[RoundLog]
    structure_steps: int
    exploitation_steps: int
    final_parent_set: Sequence[int]


class TauScheduler:
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
        mode: SchedulerMode = "interleaved",
        optimal_mean: float,
    ) -> None:
        self.instance = instance
        self.structure = structure
        self.arm_builder = arm_builder
        self.policy = policy
        self.tau = max(0.0, min(1.0, tau))
        self.horizon = int(horizon)
        self.mode = mode
        self.optimal_mean = optimal_mean
        self.structure_budget = int(self.tau * self.horizon)

    def _refresh_arms(self, parent_set: Sequence[int], rng: np.random.Generator) -> None:
        arms = self.arm_builder.build(parent_set, rng)
        self.policy.set_arms(arms)

    def run(self, rng: np.random.Generator) -> RunSummary:
        parent_set: Sequence[int] = tuple(self.structure.parent_set())
        self._refresh_arms(parent_set, rng)
        logs: List[RoundLog] = []
        structure_steps = 0
        exploitation_steps = 0

        phase = "structure" if self.mode == "two_phase" else "mixed"

        for t in range(1, self.horizon + 1):
            take_structure = False
            if phase == "structure":
                take_structure = structure_steps < self.structure_budget and self.structure.needs_structure_step()
                if not take_structure:
                    phase = "done"
            if phase in {"mixed", "done"}:
                take_structure = (
                    self.structure.needs_structure_step()
                    and structure_steps < self.structure_budget
                    and self.mode == "interleaved"
                )

            if take_structure:
                result = self.structure.step(rng)
                structure_steps += 1
                if result.parent_discovered:
                    parent_set = tuple(self.structure.parent_set())
                    self._refresh_arms(parent_set, rng)
                logs.append(
                    RoundLog(
                        t=t,
                        mode="structure",
                        arm=result.arm,
                        reward=result.reward,
                        expected_mean=result.expected_mean,
                        parent_set=parent_set,
                    )
                )
                continue

            arm_idx, arm = self.policy.select(rng)
            reward = self.instance.sample_reward(arm, rng)
            self.policy.observe(arm_idx, reward)
            exploitation_steps += 1
            expected_mean = self.policy.arms()[arm_idx].true_mean
            logs.append(
                RoundLog(
                    t=t,
                    mode="exploit",
                    arm=arm,
                    reward=reward,
                    expected_mean=expected_mean,
                    parent_set=parent_set,
                )
            )

        return RunSummary(
            logs=logs,
            structure_steps=structure_steps,
            exploitation_steps=exploitation_steps,
            final_parent_set=parent_set,
        )
