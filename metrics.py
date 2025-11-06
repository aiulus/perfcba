from __future__ import annotations
from typing import Optional
import numpy as np
from .Algorithm import History
from .Bandit import AbstractBandit

def cumulative_pseudo_regret(history: History, bandit: AbstractBandit) -> np.ndarray:
    """R_t = sum_{s<=t} (mu* - mu_{a_s})."""
    if history.actions.size == 0:
        return np.zeros(0, dtype=float)
    mu_star = bandit.best_mean()
    means = np.array([bandit.mean(a) for a in history.actions], dtype=float)
    inst = mu_star - means
    return np.cumsum(inst)

def simple_regret(history: History, bandit: AbstractBandit) -> float:
    """mu* - mu_{hat a}, hat a = most played arm (ties -> smallest index)."""
    if history.actions.size == 0:
        return 0.0
    pulls = np.bincount(history.actions, minlength=bandit.n_arms)
    hat_a = int(np.argmax(pulls))
    return float(bandit.best_mean() - bandit.mean(hat_a))

def area_under_regret_curve(regret_curve: np.ndarray) -> float:
    """Discrete AUC of the cumulative regret curve (lower = better)."""
    if regret_curve.size == 0:
        return 0.0
    # Equivalent to trapezoid with unit spacing
    return float(np.trapz(regret_curve, dx=1.0))

def time_to_epsilon_optimal(regret_curve: np.ndarray, epsilon: float = 1.0) -> Optional[int]:
    """
    First t such that average regret <= epsilon, i.e., R_t / t <= epsilon.
    Returns None if never achieved.
    """
    if regret_curve.size == 0:
        return None
    t = np.arange(1, regret_curve.size + 1)
    avg = regret_curve / t
    idx = np.nonzero(avg <= epsilon)[0]
    return int(idx[0] + 1) if idx.size else None

def pulls_per_arm(history: History) -> np.ndarray:
    return history.pulls.copy()

def summary(history: History, bandit: AbstractBandit, *, epsilon: float = 1.0) -> dict:
    rc = cumulative_pseudo_regret(history, bandit)
    return {
        "total_regret": float(rc[-1]) if rc.size else 0.0,
        "simple_regret": simple_regret(history, bandit),
        "aurc": area_under_regret_curve(rc),
        "time_to_epsilon": time_to_epsilon_optimal(rc, epsilon),
        "pulls": history.pulls.tolist(),
    }
