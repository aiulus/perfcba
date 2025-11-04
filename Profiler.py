# Profiler.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any

from .Algorithm import History
from .Bandit import AbstractBandit


class Profiler:
    """
    Offline evaluator: uses History + true bandit means to compute metrics/curves.
    """

    def __init__(self, history: History, bandit: AbstractBandit):
        self.h = history
        self.bandit = bandit
        self._means_fn = getattr(bandit, "means_at", None)
        if self._means_fn is None:
            self._means_fn = lambda t: np.array([bandit.mean(a) for a in range(bandit.n_arms)], dtype=float)
        self._stationary = getattr(bandit, "is_stationary", True)
        self._initial_means = np.asarray(self._means_fn(1), dtype=float)
        self._means = self._initial_means
        self._mu_star = float(np.max(self._initial_means))

    # --- Standard cumulative pseudo-regret curve ---
    def regret_curve(self) -> np.ndarray:
        """R_t = Σ_{s=1..t} (mu* - mu_{A_s})"""
        regrets = np.zeros(self.h.T, dtype=float)
        for idx, a in enumerate(self.h.actions, start=1):
            means_t = np.asarray(self._means_fn(idx), dtype=float)
            if means_t.size != self.bandit.n_arms:
                raise ValueError("means_at must return an array with one entry per arm")
            mu_star = float(np.max(means_t))
            regrets[idx - 1] = mu_star - float(means_t[a])
        return np.cumsum(regrets)

    def total_regret(self) -> float:
        return float(self.regret_curve()[-1]) if self.h.T else 0.0

    # --- Simple regret (fixed-budget) ---
    def simple_regret(self) -> float:
        """mu* - mu_{a_hat}, where a_hat is the most-played (or highest-mean) arm."""
        # Most played arm (ties -> smallest index)
        hat_a = int(np.argmax(self.h.pulls))
        return float(self._mu_star - self._means[hat_a])

    # --- Convenience summaries ---
    def summary(self) -> Dict[str, Any]:
        return {
            "T": self.h.T,
            "cum_reward": self.h.cum_reward,
            "total_regret": self.total_regret(),
            "simple_regret": self.simple_regret(),
            "pulls": self.h.pulls.tolist(),
        }
