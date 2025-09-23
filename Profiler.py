# Profiler.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any

from Algorithm import History
from Bandit import AbstractBandit


class Profiler:
    """
    Offline evaluator: uses History + true bandit means to compute metrics/curves.
    """

    def __init__(self, history: History, bandit: AbstractBandit):
        self.h = history
        self.bandit = bandit
        self._means = np.array([bandit.mean(a) for a in range(bandit.n_arms)], dtype=float)
        self._mu_star = float(self._means.max())

    # --- Standard cumulative pseudo-regret curve ---
    def regret_curve(self) -> np.ndarray:
        """R_t = Σ_{s=1..t} (mu* - mu_{A_s})"""
        mu_by_arm = self._means[self.h.actions]
        gaps = self._mu_star - mu_by_arm
        return np.cumsum(gaps)

    def total_regret(self) -> float:
        return float(self.regret_curve()[-1]) if self.h.T else 0.0

    # --- Simple regret (fixed-budget) ---
    def simple_regret(self) -> float:
        """mu* - mu_{\hat a}, where \hat a is the most-played (or highest-mean) arm."""
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
