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
    epsilon: float = 1e-3,
) -> float:
    if not logs:
        return 0.0
    hits = sum(1 for entry in logs if abs(entry.expected_mean - optimal_mean) <= epsilon)
    return hits / len(logs)


def time_to_optimality(
    logs: Sequence[RoundLog],
    optimal_mean: float,
    *,
    epsilon: float = 0.01,
    window_fraction: float = 0.1,
    threshold: float = 0.9,
) -> int:
    if not logs:
        return 0
    window = max(1, int(len(logs) * window_fraction))
    for start in range(0, len(logs) - window + 1):
        window_entries = logs[start : start + window]
        hits = sum(1 for entry in window_entries if abs(entry.expected_mean - optimal_mean) <= epsilon)
        if hits / window >= threshold:
            return start + 1
    return len(logs)


def simple_regret(logs: Sequence[RoundLog], optimal_mean: float) -> float:
    if not logs:
        return float(optimal_mean)
    best_played = max(
        (entry.expected_mean for entry in logs if math.isfinite(entry.expected_mean)),
        default=float("-inf"),
    )
    if best_played == float("-inf"):
        return float(optimal_mean)
    return max(0.0, optimal_mean - best_played)


def summarize(logs: Sequence[RoundLog], optimal_mean: float) -> AggregateMetrics:
    return AggregateMetrics(
        cumulative_regret=cumulative_regret(logs, optimal_mean),
        time_to_optimality=time_to_optimality(logs, optimal_mean),
        optimal_action_rate=optimal_action_rate(logs, optimal_mean),
        simple_regret=simple_regret(logs, optimal_mean),
    )
