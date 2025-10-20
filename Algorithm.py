from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np



class BasePolicy(ABC):
    """
    Base class for bandit policies.
    Policies see *only* the history and their own internal state.
    """

    name: str = "base-policy"

    @abstractmethod
    def reset(self, n_arms: int, horizon: Optional[int] = None, **params: Any) -> None:
        """Initialize internal state for a fresh run."""
        ...

    @abstractmethod
    def choose(self, t: int, history: "History") -> int:
        """Pick an arm at round t given the (t-1)-round history."""
        ...

    @abstractmethod
    def update(
        self,
        t: int,
        a: int,
        x: float,
        *,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Process the observed reward from round ``t``.

        Parameters
        ----------
        t:
            Current round (1-indexed).
        a:
            Arm pulled at round ``t``.
        x:
            Observed reward.
        info:
            Optional side-information emitted by the environment together with
            the reward (e.g., observed covariates for structured/causal
            policies).  Policies that do not make use of side-information can
            safely ignore this argument.
        """
        ...

    def get_params(self) -> Dict[str, Any]:
        return {}


@dataclass
class History:
    T: int
    actions: np.ndarray
    rewards: np.ndarray
    means_star: np.ndarray
    n_arms: int
    pulls: np.ndarray
    cum_reward: float
    observations: Optional[List[Dict[str, Any]]] = None


def running_means_from_history(history: History, n_arms: int) -> np.ndarray:
    """
    Utility for policies that compute empirical means from the history.
    Returns shape (n_arms,) with empirical means after the *current* history.
    """
    means = np.zeros(n_arms, dtype=float)
    counts = np.zeros(n_arms, dtype=int)
    for a, x in zip(history.actions, history.rewards):
        counts[a] += 1
        means[a] += x
    np.divide(means, np.maximum(1, counts), out=means)
    return means
