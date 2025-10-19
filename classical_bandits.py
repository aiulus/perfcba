# classical_bandits.py
from __future__ import annotations
import math
import random
from typing import List, Optional, Sequence, Tuple
import numpy as np

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


class UCB(BasePolicy):
    r"""Upper Confidence Bound policy for bounded/sub-Gaussian rewards.

    This implements the standard "mean + width" index based on Hoeffding-style
    confidence radii. Two exploration schedules are available:

    * ``schedule="ucb1_alpha"`` (default ``alpha=2``): classic UCB1 with
      exploration bonus :math:`\sqrt{(\alpha \log t)/(2 n_i)}`.
    * ``schedule="asymptotic"``: asymptotically optimal schedule with
      :math:`f(t) = 1 + t (\log t)^2` leading to index
      :math:`\hat\mu_i + \sqrt{(2 \log f(t))/n_i}`.

    Parameters
    ----------
    schedule:
        Exploration schedule name. Either ``"ucb1_alpha"`` or ``"asymptotic"``.
    alpha:
        Exploration constant :math:`\alpha > 1` used by the ``ucb1_alpha``
        schedule.
    tie_break:
        Strategy for resolving ties between equal indices. Either ``"first"``
        (deterministic) or ``"random"``.
    """

    name = "ucb"

    def __init__(
        self,
        schedule: str = "asymptotic",
        *,
        alpha: float = 2.0,
        tie_break: str = "random",
    ) -> None:
        self.schedule = schedule
        self.alpha = float(alpha)
        if self.alpha <= 1.0 and self.schedule == "ucb1_alpha":
            raise ValueError("alpha must be > 1 for the ucb1_alpha schedule")
        if tie_break not in {"first", "random"}:
            raise ValueError("tie_break must be 'first' or 'random'")
        self.tie_break = tie_break

        # Internal state initialised in reset()
        self.n_arms: int = 0
        self.counts: np.ndarray
        self.sums: np.ndarray
        self.total_pulls: int = 0
        self._warmup: List[int] = []

    def reset(self, n_arms: int, horizon: Optional[int] = None, **_: object) -> None:
        del horizon  # unused but kept for API compatibility
        self.n_arms = int(n_arms)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.sums = np.zeros(self.n_arms, dtype=float)
        self.total_pulls = 0
        # Pull each arm once before using confidence bounds.
        self._warmup = list(range(self.n_arms))

    def _f(self, t: int) -> float:
        """Asymptotically optimal exploration schedule f(t)."""

        if t <= 1:
            return 1.0
        return 1.0 + t * (math.log(t) ** 2)

    def _width(self, pulls: int) -> float:
        """Confidence radius for an arm with ``pulls`` samples."""

        if pulls == 0:
            # Should not be used beyond warm-up but guard regardless.
            return float("inf")

        total = max(2, self.total_pulls)
        if self.schedule == "ucb1_alpha":
            value = (self.alpha * math.log(total)) / (2.0 * pulls)
            return math.sqrt(max(0.0, value))
        if self.schedule == "asymptotic":
            value = (2.0 * math.log(self._f(total))) / pulls
            return math.sqrt(max(0.0, value))
        raise ValueError(f"Unknown schedule '{self.schedule}'")

    def _resolve_ties(self, candidates: Sequence[int]) -> int:
        if self.tie_break == "first":
            return candidates[0]
        return random.choice(list(candidates))

    def choose(self, t: int, history: History) -> int:
        del t, history  # policy keeps its own sufficient statistics
        if self._warmup:
            return self._warmup.pop(0)

        estimates = np.divide(
            self.sums,
            np.maximum(1, self.counts),
            out=np.zeros_like(self.sums),
        )

        indices = estimates + np.fromiter(
            (self._width(int(pulls)) for pulls in self.counts),
            dtype=float,
            count=self.n_arms,
        )

        max_index = float(np.max(indices))
        best = np.flatnonzero(np.isclose(indices, max_index)).tolist()
        return self._resolve_ties(best)

    def update(self, t: int, a: int, x: float) -> None:
        del t
        self.total_pulls += 1
        self.counts[a] += 1
        self.sums[a] += float(x)

    def ucb_index(self, arm: int) -> float:
        pulls = int(self.counts[arm])
        if pulls == 0:
            return float("inf")
        mean = self.sums[arm] / pulls
        return mean + self._width(pulls)

    def get_params(self) -> dict:
        return {
            "schedule": self.schedule,
            "alpha": self.alpha,
            "tie_break": self.tie_break,
        }


def kl_bernoulli(p: float, q: float) -> float:
    """Bernoulli Kullback–Leibler divergence with continuity at the edges."""

    eps = 1e-12
    p = min(max(p, eps), 1.0 - eps)
    q = min(max(q, eps), 1.0 - eps)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def kl_ucb_solve_upper(
    mu_hat: float,
    pulls: int,
    log_ft: float,
    *,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> float:
    """Solve for the KL-UCB upper confidence bound via bisection."""

    if pulls == 0:
        return 1.0

    target = log_ft / max(1, pulls)
    lo = float(mu_hat)
    hi = 1.0

    if kl_bernoulli(lo, hi) <= target:
        return hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(lo, mid) <= target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo


class KLUCB(BasePolicy):
    """KL-UCB policy for Bernoulli rewards."""

    name = "kl-ucb"

    def __init__(self, *, tie_break: str = "random") -> None:
        if tie_break not in {"first", "random"}:
            raise ValueError("tie_break must be 'first' or 'random'")
        self.tie_break = tie_break

        self.n_arms: int = 0
        self.counts: np.ndarray
        self.sums: np.ndarray
        self.total_pulls: int = 0
        self._warmup: List[int] = []

    def reset(self, n_arms: int, horizon: Optional[int] = None, **_: object) -> None:
        del horizon
        self.n_arms = int(n_arms)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.sums = np.zeros(self.n_arms, dtype=float)
        self.total_pulls = 0
        self._warmup = list(range(self.n_arms))

    def _f(self, t: int) -> float:
        if t <= 1:
            return 1.0
        return 1.0 + t * (math.log(t) ** 2)

    def _resolve_ties(self, candidates: Sequence[int]) -> int:
        if self.tie_break == "first":
            return candidates[0]
        return random.choice(list(candidates))

    def choose(self, t: int, history: History) -> int:
        del t, history
        if self._warmup:
            return self._warmup.pop(0)

        log_ft = math.log(self._f(max(2, self.total_pulls)))
        estimates = np.divide(
            self.sums,
            np.maximum(1, self.counts),
            out=np.zeros_like(self.sums),
        )
        indices = np.array(
            [
                kl_ucb_solve_upper(
                    float(estimates[i]), int(self.counts[i]), log_ft
                )
                for i in range(self.n_arms)
            ],
            dtype=float,
        )
        max_index = float(np.max(indices))
        best = np.flatnonzero(np.isclose(indices, max_index)).tolist()
        return self._resolve_ties(best)

    def update(self, t: int, a: int, x: float) -> None:
        del t
        if x < 0.0 or x > 1.0:
            raise ValueError("KL-UCB expects Bernoulli rewards in [0, 1]")
        self.total_pulls += 1
        self.counts[a] += 1
        self.sums[a] += float(x)

    def kl_index(self, arm: int) -> float:
        pulls = int(self.counts[arm])
        if pulls == 0:
            return 1.0
        mean = self.sums[arm] / pulls
        return kl_ucb_solve_upper(mean, pulls, math.log(self._f(max(2, self.total_pulls))))

    def get_params(self) -> dict:
        return {"tie_break": self.tie_break}


def run_bandit(env, policy: BasePolicy, horizon: int) -> Tuple[List[int], List[float]]:
    """Utility to roll out a policy for ``horizon`` steps on ``env``.

    Parameters
    ----------
    env:
        Either a callable ``env(arm)`` returning the observed reward or an
        object exposing a ``pull(arm)`` method.
    policy:
        Bandit policy implementing the :class:`BasePolicy` interface. It is
        expected that ``policy.reset`` has been called beforehand.
    horizon:
        Number of interaction rounds.
    """

    actions: List[int] = []
    rewards: List[float] = []

    for t in range(1, horizon + 1):
        arm = policy.choose(t, history=None)  # type: ignore[arg-type]
        if hasattr(env, "pull"):
            reward = env.pull(arm)
        else:
            reward = env(arm)  # type: ignore[call-arg]
        policy.update(t, arm, reward)
        actions.append(arm)
        rewards.append(reward)

    return actions, rewards