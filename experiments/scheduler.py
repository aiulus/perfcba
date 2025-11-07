"""Tau-scheduled orchestration of structure learning and exploitation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .causal_envs import CausalBanditInstance, InterventionArm
from .exploit import ArmBuilder, ArmID, ParentAwareUCB
from .structure import StructureLearner, StructureStepResult

SchedulerMode = Literal["interleaved", "etc", "two_phase", "adaptive_burst"]


@dataclass
class RoundLog:
    t: int
    mode: Literal["structure", "exploit"]
    arm: InterventionArm
    reward: float
    expected_mean: float
    parent_set: Tuple[int, ...]
    arm_count: int
    burst_remaining: Optional[int] = None
    stall_flag: Optional[bool] = None
    improvement_stat: Optional[float] = None


@dataclass
class RunSummary:
    logs: List[RoundLog]
    structure_steps: int
    exploit_steps: int
    final_parent_set: Tuple[int, ...]
    finished_discovery_round: Optional[int]


@dataclass
class AdaptiveBurstConfig:
    """Tunable parameters for the adaptive-burst scheduler."""

    start_mode: Literal["exploit_first", "explore_first"] = "exploit_first"
    initial_burst: int = 1
    growth_factor: float = 2.0
    window: Optional[int] = None
    stall_min_exploit: Optional[int] = None
    metric: Literal["reward", "regret", "opt_rate"] = "reward"
    eta_down: float = -0.15
    eta_up: float = 0.1
    reset_mode: Literal["one", "x0"] = "one"
    cooldown: Optional[int] = None
    tail_fraction: float = 0.25
    opt_rate_tolerance: float = 0.01
    ewma_lambda: float = 0.2
    enable_page_hinkley: bool = False
    ph_delta: float = 1e-3
    ph_lambda: float = 0.05
    ph_alpha: float = 0.1
    scale_floor: float = 1e-6


class EWMA:
    """Lightweight exponentially weighted moving average."""

    def __init__(self, lam: float) -> None:
        self.lam = float(max(0.0, min(1.0, lam)))
        self.value: Optional[float] = None

    def update(self, sample: float) -> float:
        if self.value is None or not math.isfinite(self.value):
            self.value = sample
        else:
            self.value = self.lam * sample + (1.0 - self.lam) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class PageHinkley:
    """Simple Page–Hinkley drift detector."""

    def __init__(self, delta: float, lam: float, alpha: float) -> None:
        self.delta = delta
        self.lam = lam
        self.alpha = alpha
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0
        self.initialized = False

    def reset(self) -> None:
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0
        self.initialized = False

    def update(self, sample: float) -> bool:
        if not self.initialized:
            self.mean = sample
            self.initialized = True
        else:
            self.mean = (1.0 - self.lam) * self.mean + self.lam * sample
        self.cum += sample - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        return (self.cum - self.min_cum) < -self.alpha


class WelfordTracker:
    """Online mean/variance tracker for exploitation spans."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    def std(self) -> float:
        if self.n < 2:
            return 0.0
        variance = self.m2 / max(1, self.n - 1)
        return math.sqrt(max(0.0, variance))


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

    def _exploit_step_with_stats(
        self, rng: np.random.Generator
    ) -> Tuple[int, InterventionArm, float, float]:
        if self._arm_count == 0:
            self._refresh_arms(self._current_parent_set, rng)
        arm_idx, arm = self.policy.select(rng)
        reward = self.instance.sample_reward(arm, rng)
        self.policy.observe(arm_idx, reward)
        self.exploit_steps += 1
        self.t += 1
        expected_mean = self._arm_means.get(self._arm_id(arm), float("nan"))
        self._log(mode="exploit", arm=arm, reward=reward, expected_mean=expected_mean)
        return arm_idx, arm, reward, expected_mean

    def _exploit_step(self, rng: np.random.Generator) -> None:
        self._exploit_step_with_stats(rng)

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


class TauSchedulerAdaptiveBurst(TauSchedulerBase):
    """Adaptive scheduler that alternates exploitation with reactive exploration bursts."""

    def __init__(
        self,
        *args,
        adaptive_config: Optional[AdaptiveBurstConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        cfg = adaptive_config or AdaptiveBurstConfig()
        self.cfg = replace(cfg)
        self.cfg.initial_burst = max(1, int(self.cfg.initial_burst))
        self.cfg.growth_factor = max(1.0, float(self.cfg.growth_factor))
        self.cfg.scale_floor = max(1e-9, float(self.cfg.scale_floor))
        self.window = self.cfg.window or max(5, int(0.05 * max(1, self.horizon)))
        self.stall_min = max(1, self.cfg.stall_min_exploit or self.window)
        self.cooldown_default = (
            max(0, int(self.cfg.cooldown))
            if self.cfg.cooldown is not None
            else max(1, self.window // 2)
        )
        self.tail_fraction = min(1.0, max(0.0, self.cfg.tail_fraction))
        self.base_burst = 1 if self.cfg.reset_mode == "one" else self.cfg.initial_burst
        self.mode: Literal["explore", "exploit"] = "exploit"
        self.burst_remaining = 0
        self.last_burst = self.cfg.initial_burst
        self.metric_ewma = EWMA(self.cfg.ewma_lambda)
        self.span_tracker = WelfordTracker()
        self.metric_anchor: Optional[float] = None
        self.exploit_span = 0
        self.cooldown_remaining = 0
        self.page_hinkley = (
            PageHinkley(
                delta=self.cfg.ph_delta,
                lam=self.cfg.ph_lambda,
                alpha=self.cfg.ph_alpha,
            )
            if self.cfg.enable_page_hinkley
            else None
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_exploit_span(self, *, reset_anchor: bool, full_reset: bool = False) -> None:
        self.exploit_span = 0
        self.span_tracker.reset()
        if full_reset:
            self.metric_ewma.reset()
            self.metric_anchor = None
        elif reset_anchor:
            self.metric_anchor = self.metric_ewma.value
        if self.page_hinkley is not None:
            self.page_hinkley.reset()

    def _metric_sample(self, reward: float, expected_mean: float) -> float:
        estimate = expected_mean if math.isfinite(expected_mean) else reward
        if self.cfg.metric == "regret":
            best = max(self._arm_means.values()) if self._arm_means else estimate
            return estimate - best
        if self.cfg.metric == "opt_rate":
            best = max(self._arm_means.values()) if self._arm_means else estimate
            return 1.0 if estimate >= best - self.cfg.opt_rate_tolerance else 0.0
        return reward

    def _normalized_gain(self) -> Optional[float]:
        anchor = self.metric_anchor
        current = self.metric_ewma.value
        if anchor is None or current is None:
            return None
        scale = max(self.cfg.scale_floor, self.span_tracker.std())
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        return (current - anchor) / scale

    def _update_log(self, stalled: Optional[bool], gain: Optional[float]) -> None:
        if not self.logs:
            return
        entry = self.logs[-1]
        entry.burst_remaining = self.burst_remaining if self.mode == "explore" else 0
        entry.stall_flag = stalled
        entry.improvement_stat = gain

    def _schedule_burst(self) -> bool:
        remaining_budget = self.structure_cap - self.structure_steps
        if remaining_budget <= 0 or not self.structure.needs_structure_step():
            return False
        tail_limit = remaining_budget
        if self.tail_fraction > 0.0:
            tail_limit = min(
                tail_limit,
                max(1, int(self.tail_fraction * max(0, self.horizon - self.t))),
            )
        burst_seed = max(self.cfg.initial_burst, self.last_burst)
        raw = max(1, int(math.ceil(self.cfg.growth_factor * burst_seed)))
        burst = min(raw, tail_limit)
        if burst <= 0:
            return False
        self.mode = "explore"
        self.burst_remaining = burst
        self.last_burst = burst
        self.cooldown_remaining = self.cooldown_default
        self._reset_exploit_span(reset_anchor=True, full_reset=False)
        return True

    def _handle_improvement(self) -> None:
        self.last_burst = self.base_burst
        self.cooldown_remaining = self.cooldown_default
        self._reset_exploit_span(reset_anchor=True, full_reset=False)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, rng: np.random.Generator) -> RunSummary:
        self._refresh_arms(self._current_parent_set, rng)
        if self.cfg.start_mode == "explore_first" and self.structure_cap > 0:
            self.mode = "explore"
            self.burst_remaining = min(self.cfg.initial_burst, self.structure_cap)
            self.last_burst = self.burst_remaining
        else:
            self.mode = "exploit"
            self.burst_remaining = 0
        self._reset_exploit_span(reset_anchor=True, full_reset=True)

        while self.t < self.horizon:
            can_structure = (
                self.structure_steps < self.structure_cap and self.structure.needs_structure_step()
            )
            if self.mode == "explore":
                if not can_structure or self.burst_remaining <= 0:
                    self.mode = "exploit"
                    self.burst_remaining = 0
                    self._reset_exploit_span(reset_anchor=True, full_reset=False)
                    continue
                previous_parents = self._current_parent_set
                result = self.structure.step(rng)
                self._structure_step(result)
                log_entry = self.logs[-1]
                log_entry.burst_remaining = self.burst_remaining
                if self._current_parent_set != previous_parents:
                    self._refresh_arms(self._current_parent_set, rng)
                    self._reset_exploit_span(reset_anchor=True, full_reset=True)
                self.burst_remaining -= 1
                if self.burst_remaining <= 0:
                    self.mode = "exploit"
                    self._reset_exploit_span(reset_anchor=True, full_reset=False)
                continue

            # default: exploitation
            if not self.structure.needs_structure_step() and self.finished_discovery_round is None:
                self.finished_discovery_round = self.t
            _, _, reward, expected_mean = self._exploit_step_with_stats(rng)
            sample = self._metric_sample(reward, expected_mean)
            current = self.metric_ewma.update(sample)
            self.span_tracker.update(sample)
            if self.metric_anchor is None:
                self.metric_anchor = current
            self.exploit_span += 1
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
            gain = self._normalized_gain()
            ph_alarm = False
            if self.page_hinkley is not None and current is not None:
                ph_alarm = self.page_hinkley.update(current)

            stalled: Optional[bool] = None
            improvement = False
            if (
                self.exploit_span >= self.stall_min
                and self.cooldown_remaining == 0
                and self.structure_steps < self.structure_cap
                and gain is not None
            ):
                improvement = gain >= self.cfg.eta_up and not ph_alarm
                stalled = (gain <= self.cfg.eta_down) or ph_alarm
            self._update_log(stalled, gain)

            if stalled:
                triggered = self._schedule_burst()
                if triggered:
                    self.exploit_span = 0
                continue

            if improvement:
                self._handle_improvement()

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
    adaptive_config: Optional[AdaptiveBurstConfig] = None,
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
    if normalized_mode == "adaptive_burst":
        return TauSchedulerAdaptiveBurst(
            instance=instance,
            structure=structure,
            arm_builder=arm_builder,
            policy=policy,
            tau=tau,
            horizon=horizon,
            adaptive_config=adaptive_config,
        )
    raise ValueError(f"Unknown scheduler mode: {mode}")
