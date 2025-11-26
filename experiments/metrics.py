"""Metric computations for tau-scheduled experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .scheduler import RoundLog


@dataclass
class AggregateMetrics:
    cumulative_regret: float
    time_to_optimality: int
    optimal_action_rate: float
    simple_regret: float


def cumulative_regret(logs: Sequence[RoundLog], optimal_mean: float) -> float:
    return sum(optimal_mean - entry.expected_mean for entry in logs)


def optimal_action_rate(
    logs: Sequence[RoundLog],
    optimal_mean: float,
    *,
    epsilon: float | None = None,
    n_mc: int = 1024,
    adaptive: bool = True,
) -> float:
    # Consider only true interventions/exploitation rounds (non-empty arms).
    intervention_logs = [entry for entry in logs if entry.arm.variables]
    if not intervention_logs:
        return 0.0

    def _resolve_epsilon() -> float:
        if epsilon is not None:
            return float(epsilon)
        if adaptive:
            # Approximate 3-sigma band for the difference of two independent Bernoulli means.
            se = (0.25 / max(1, n_mc)) ** 0.5
            return 3.0 * (2.0**0.5) * se
        return 1e-3

    eps = _resolve_epsilon()
    hits = sum(1 for entry in intervention_logs if abs(entry.expected_mean - optimal_mean) <= eps)
    return hits / len(intervention_logs)


def time_to_optimality(
    logs: Sequence[RoundLog],
    optimal_mean: float,
    *,
    epsilon: float | None = None,
    adaptive: bool = True,
    n_mc: int = 1024,
    window_fraction: float = 0.1,
    threshold: float = 0.9,
) -> int:
    intervention_logs = [entry for entry in logs if entry.arm.variables]
    if not intervention_logs:
        return 0

    def _resolve_epsilon() -> float:
        if epsilon is not None:
            return float(epsilon)
        if adaptive:
            se = (0.25 / max(1, n_mc)) ** 0.5
            return 3.0 * (2.0**0.5) * se
        return 1e-3

    eps = _resolve_epsilon()
    window = max(1, int(len(intervention_logs) * window_fraction))
    for start in range(0, len(intervention_logs) - window + 1):
        window_entries = intervention_logs[start : start + window]
        hits = sum(1 for entry in window_entries if abs(entry.expected_mean - optimal_mean) <= eps)
        if hits / window >= threshold:
            return start + 1
    return len(intervention_logs)


def simple_regret(logs: Sequence[RoundLog], optimal_mean: float) -> float:
    intervention_logs = [entry for entry in logs if entry.arm.variables and math.isfinite(entry.expected_mean)]
    if not intervention_logs:
        return float(optimal_mean)
    best_played = max((entry.expected_mean for entry in intervention_logs), default=float("-inf"))
    if best_played == float("-inf"):
        return float(optimal_mean)
    return max(0.0, optimal_mean - best_played)


def summarize(
    logs: Sequence[RoundLog],
    optimal_mean: float,
    *,
    opt_rate_epsilon: float | None = None,
    opt_rate_adaptive: bool = True,
    opt_rate_n_mc: int = 1024,
    tto_window_fraction: float = 0.1,
    tto_threshold: float = 0.9,
) -> AggregateMetrics:
    return AggregateMetrics(
        cumulative_regret=cumulative_regret(logs, optimal_mean),
        time_to_optimality=time_to_optimality(
            logs,
            optimal_mean,
            epsilon=opt_rate_epsilon,
            adaptive=opt_rate_adaptive,
            n_mc=opt_rate_n_mc,
            window_fraction=tto_window_fraction,
            threshold=tto_threshold,
        ),
        optimal_action_rate=optimal_action_rate(
            logs,
            optimal_mean,
            epsilon=opt_rate_epsilon,
            adaptive=opt_rate_adaptive,
            n_mc=opt_rate_n_mc,
        ),
        simple_regret=simple_regret(logs, optimal_mean),
    )
