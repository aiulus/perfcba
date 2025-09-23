# classical_bandits.py
from __future__ import annotations
import math
import numpy as np
from typing import Optional

from Algorithm import BasePolicy, History


class ExploreThenCommit(BasePolicy):
    """
    ETC with exploration fraction tau in (0,1].
    - Explore round-robin for floor(tau*T) pulls.
    - Commit to argmax empirical mean for the rest.
    """
    name = "etc"

    def __init__(self, tau: float = 0.2) -> None:
        if not (0.0 < tau <= 1.0):
            raise ValueError("tau must be in (0,1].")
        self.tau = float(tau)

        # Internal state (set in reset)
        self.n_arms: int = 0
        self.T: Optional[int] = None
        self.n_explore: int = 0
        self.counts: np.ndarray
        self.sums: np.ndarray
        self._committed_arm: Optional[int] = None

    def reset(self, n_arms: int, horizon: Optional[int] = None, **_: object) -> None:
        self.n_arms = int(n_arms)
        self.T = None if horizon is None else int(horizon)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.sums = np.zeros(self.n_arms, dtype=float)
        self._committed_arm = None
        self.n_explore = 0 if self.T is None else int(math.floor(self.tau * self.T))

    def choose(self, t: int, history: History) -> int:
        # Phase 1: exploration (round-robin)
        if self.T is not None and t <= self.n_explore:
            return (t - 1) % self.n_arms  # 0-based

        # Phase 2: commit
        if self._committed_arm is None:
            # Compute empirical means from our tracked state
            means = np.divide(self.sums, np.maximum(1, self.counts), out=np.zeros_like(self.sums), where=True)
            self._committed_arm = int(np.argmax(means))
        return self._committed_arm

    def update(self, t: int, a: int, x: float) -> None:
        self.counts[a] += 1
        self.sums[a] += x

    def get_params(self):
        return {"tau": self.tau}
