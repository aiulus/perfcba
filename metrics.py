metrics.py
+68
-0

from __future__ import annotations

"""Evaluation metrics for bandit experiments."""

from typing import Iterable, Optional

import numpy as np

from Algorithm import History
from Bandit import AbstractBandit


def cumulative_pseudo_regret(history: History, bandit: AbstractBandit) -> np.ndarray:
    """Return the cumulative pseudo-regret curve for ``history``."""

    means = np.array([bandit.mean(a) for a in range(bandit.n_arms)], dtype=float)
    optimal = float(np.max(means))
    played_means = means[history.actions]
    gaps = optimal - played_means
    return np.cumsum(gaps)


def simple_regret(history: History, bandit: AbstractBandit) -> float:
    means = np.array([bandit.mean(a) for a in range(bandit.n_arms)], dtype=float)
    best = float(np.max(means))
    most_played = int(np.argmax(history.pulls))
    return float(best - means[most_played])


def area_under_regret_curve(regret: np.ndarray) -> float:
    if regret.size == 0:
        return 0.0
    return float(np.trapz(regret, dx=1.0))


def time_to_epsilon_optimal(regret: np.ndarray, epsilon: float) -> Optional[int]:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    indices = np.flatnonzero(regret <= epsilon)
    if indices.size == 0:
        return None
    return int(indices[0] + 1)  # convert to 1-indexed horizon


def mean_squared_error(estimates: Iterable[float], truth: Iterable[float]) -> float:
    est = np.asarray(list(estimates), dtype=float)
    tru = np.asarray(list(truth), dtype=float)
    if est.shape != tru.shape:
        raise ValueError("estimates and truth must have matching shapes")
    if est.size == 0:
        return 0.0
    diff = est - tru
    return float(np.mean(diff ** 2))


def pulls_per_arm(history: History) -> np.ndarray:
    return history.pulls.copy()


def summary(history: History, bandit: AbstractBandit, *, epsilon: float = 1.0) -> dict:
    regret_curve = cumulative_pseudo_regret(history, bandit)
    return {
        "total_regret": float(regret_curve[-1]) if regret_curve.size else 0.0,
        "simple_regret": simple_regret(history, bandit),
        "aurc": area_under_regret_curve(regret_curve),
        "time_to_epsilon": time_to_epsilon_optimal(regret_curve, epsilon),
        "pulls": history.pulls.tolist(),
    }