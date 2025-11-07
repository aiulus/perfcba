# classical_bandits.py
from __future__ import annotations
import math
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

from .Algorithm import BasePolicy, History


class RandomPolicy(BasePolicy):
    """Policy that selects an action uniformly at random each round."""

    name = "random"

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self.n_arms: int = 0

    def reset(self, n_arms: int, horizon: Optional[int] = None, **_: object) -> None:
        del horizon
        self.n_arms = int(n_arms)
        if self.n_arms <= 0:
            raise ValueError("Number of arms must be positive")
        # Recreate RNG to provide deterministic behaviour across resets when seeded.
        self._rng = random.Random(self._seed)

    def choose(self, t: int, history: History) -> int:
        del t, history
        return self._rng.randrange(self.n_arms)

    def update(self, t: int, a: int, x: float, *, info: Optional[Dict[str, Any]] = None) -> None:
        del t, a, x, info  # Random policy is memoryless.

    def get_params(self) -> dict:
        return {"seed": self._seed}


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

    def update(self, t: int, a: int, x: float, *, info: Optional[Dict[str, Any]] = None) -> None:
        self.counts[a] += 1
        self.sums[a] += x

        if info is not None:
            del info

    def get_params(self):
        return {"tau": self.tau}

class EpsilonGreedy(BasePolicy):
    """Classical :math:`\varepsilon`-greedy exploration strategy."""

    name = "epsilon-greedy"

    def __init__(
        self,
        epsilon: Union[float, Callable[[int], float]] = 0.1,
        *,
        tie_break: str = "random",
    ) -> None:
        if isinstance(epsilon, (int, float)):
            if not (0.0 <= float(epsilon) <= 1.0):
                raise ValueError("epsilon must lie in [0, 1]")
        elif not callable(epsilon):
            raise TypeError("epsilon must be a float or a callable schedule")

        if tie_break not in {"first", "random"}:
            raise ValueError("tie_break must be 'first' or 'random'")

        self._epsilon = epsilon
        self.tie_break = tie_break

        self.n_arms: int = 0
        self.counts: np.ndarray
        self.sums: np.ndarray
        self._warmup: List[int] = []
        self._rng = random.Random()

    def reset(self, n_arms: int, horizon: Optional[int] = None, **_: object) -> None:
        del horizon
        self.n_arms = int(n_arms)
        if self.n_arms <= 0:
            raise ValueError("Number of arms must be positive")
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.sums = np.zeros(self.n_arms, dtype=float)
        self._warmup = list(range(self.n_arms))
        self._rng = random.Random()

    def _epsilon_value(self, t: int) -> float:
        if callable(self._epsilon):
            value = float(self._epsilon(t))
        else:
            value = float(self._epsilon)
        return float(min(1.0, max(0.0, value)))

    def _resolve_ties(self, candidates: Sequence[int]) -> int:
        if self.tie_break == "first":
            return candidates[0]
        return self._rng.choice(list(candidates))

    def choose(self, t: int, history: History) -> int:
        del history
        if self._warmup:
            return self._warmup.pop(0)

        eps = self._epsilon_value(t)
        if self._rng.random() < eps:
            return self._rng.randrange(self.n_arms)

        means = np.divide(
            self.sums,
            np.maximum(1, self.counts),
            out=np.zeros_like(self.sums),
        )
        best_value = float(np.max(means))
        candidates = np.flatnonzero(np.isclose(means, best_value)).tolist()
        return self._resolve_ties(candidates)

    def update(
        self,
        t: int,
        a: int,
        x: float,
        *,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        del t, info
        self.counts[a] += 1
        self.sums[a] += float(x)

    def get_params(self) -> dict:
        return {"epsilon": self._epsilon, "tie_break": self.tie_break}



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
        self._global_count: int = 0
        self._global_mean: float = 0.0
        self._global_M2: float = 0.0
        self._explore_multiplier: float = 2.0
        self._explore_cutoff: int = 300
        self._gap_scale: float = 1.0
        self._gap_ready: bool = False

    def reset(self, n_arms: int, horizon: Optional[int] = None, **_: object) -> None:
        del horizon  # unused but kept for API compatibility
        self.n_arms = int(n_arms)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.sums = np.zeros(self.n_arms, dtype=float)
        self.total_pulls = 0
        # Pull each arm once before using confidence bounds.
        self._warmup = list(range(self.n_arms))
        self._global_count = 0
        self._global_mean = 0.0
        self._global_M2 = 0.0
        self._gap_scale = 1.0
        self._gap_ready = False

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

        means = np.divide(
            self.sums,
            np.maximum(1, self.counts),
            out=np.zeros_like(self.sums),
        )
        if not self._gap_ready and means.size >= 2:
            sorted_means = np.sort(means)
            gap_estimate = float(sorted_means[-1] - sorted_means[-2])
            gap_estimate = max(gap_estimate, 1e-6)
            self._gap_scale = float(np.clip(0.05 / gap_estimate, 0.5, 4.0))
            self._gap_ready = True

        if self.total_pulls < self._explore_cutoff:
            target_multiplier = self._explore_multiplier * self._gap_scale
            target = math.ceil(target_multiplier * math.log(max(2, self.total_pulls + 1)))
            under_sampled = np.nonzero(self.counts < target)[0]
            if under_sampled.size:
                return int(under_sampled[0])
        if self._global_count > 1 and self._global_M2 > 0.0:
            scale = math.sqrt(self._global_M2 / (self._global_count - 1))
        else:
            scale = 1.0
        if not math.isfinite(scale) or scale < 1e-9:
            scale = 1.0
        mean_shift = self._global_mean if math.isfinite(self._global_mean) else 0.0
        normalized_means = (means - mean_shift) / scale
        if self._global_count > 1 and self._global_M2 > 0.0:
            scale = math.sqrt(self._global_M2 / (self._global_count - 1))
        else:
            scale = 1.0
        if not math.isfinite(scale) or scale < 1e-9:
            scale = 1.0
        mean_shift = self._global_mean if math.isfinite(self._global_mean) else 0.0
        normalized_means = (means - mean_shift) / scale

        widths = np.fromiter(
            (self._width(int(pulls)) for pulls in self.counts),
            dtype=float,
            count=self.n_arms,
        )
        indices = normalized_means + widths

        if self.tie_break == "first":
            # np.argmax is deterministic and avoids near-ties triggered by isclose tolerances.
            return int(np.argmax(indices))

        max_index = float(np.max(indices))
        candidates = np.flatnonzero(
            np.isclose(indices, max_index, rtol=1e-12, atol=1e-12)
        ).tolist()
        return self._resolve_ties(candidates or [int(np.argmax(indices))])

    def update(
        self,
        t: int,
        a: int,
        x: float,
        *,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        del t, info
        self.total_pulls += 1
        self.counts[a] += 1
        x = float(x)
        self.sums[a] += x
        self._global_count += 1
        delta = x - self._global_mean
        self._global_mean += delta / self._global_count
        delta2 = x - self._global_mean
        self._global_M2 += delta * delta2

    def ucb_index(self, arm: int) -> float:
        pulls = int(self.counts[arm])
        if pulls == 0:
            return float("inf")
        mean = self.sums[arm] / pulls
        if self._global_count > 1 and self._global_M2 > 0.0:
            scale = math.sqrt(self._global_M2 / (self._global_count - 1))
        else:
            scale = 1.0
        if not math.isfinite(scale) or scale < 1e-9:
            scale = 1.0
        mean_shift = self._global_mean if math.isfinite(self._global_mean) else 0.0
        normalized_index = (mean - mean_shift) / scale + self._width(pulls)
        return mean_shift + scale * normalized_index

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
    mu_hat = float(min(max(mu_hat, 0.0), 1.0))
    lo = mu_hat
    hi = 1.0

    if kl_bernoulli(mu_hat, hi) <= target:
        return hi

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(mu_hat, mid) <= target:
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

    def update(
        self,
        t: int,
        a: int,
        x: float,
        *,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        del t, info
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




class ThompsonSamplingPolicy(BasePolicy):
    """Generic Thompson Sampling policy (Algorithm 4).

    The policy delegates posterior sampling and updates to ``prior`` and uses a
    ``planner`` to pick the arm that maximises the sampled value. The
    ``action_space`` argument can either be a static sequence of feasible arms or
    a callable returning the action set for the current round. Sub-classes can
    override :meth:`update` to pass richer information (e.g., contextual
    features) to the underlying prior.
    """

    name = "thompson"

    def __init__(
        self,
        prior,
        planner: Callable[[Iterable[float], Sequence[object], Callable[[float], float], Optional[object]], int],
        *,
        action_space: Optional[Union[Sequence[object], Callable[[int, Optional[History]], Sequence[object]]]] = None,
        reward_function: Optional[Callable[[float], float]] = None,
        likelihood: Optional[object] = None,
    ) -> None:
        self.prior = prior
        self.planner = planner
        self._action_provider = action_space
        self.reward_function = reward_function if reward_function is not None else (lambda y: y)
        self.likelihood = likelihood

        self.n_arms: int = 0
        self.horizon: Optional[int] = None
        self._last_actions: Sequence[object] = ()

    def reset(self, n_arms: int, horizon: Optional[int] = None, **params: object) -> None:
        del params
        self.n_arms = int(n_arms)
        self.horizon = None if horizon is None else int(horizon)
        if self._action_provider is None:
            self._action_provider = lambda _t, _history: list(range(self.n_arms))
        if hasattr(self.prior, "reset"):
            self.prior.reset(n_arms=self.n_arms, horizon=self.horizon)

    def _current_action_set(self, t: int, history: Optional[History]) -> Sequence[object]:
        if callable(self._action_provider):
            actions = self._action_provider(t, history)
        else:
            actions = self._action_provider
        if not isinstance(actions, Sequence):
            actions = tuple(actions)
        return actions

    def choose(self, t: int, history: History) -> int:
        theta_hat = self.prior.sample()
        actions = self._current_action_set(t, history)
        if len(actions) == 0:
            raise ValueError("Action space must contain at least one action")
        index = self.planner(theta_hat, actions, self.reward_function, self.likelihood)
        chosen_idx = int(index)
        if not (0 <= chosen_idx < len(actions)):
            raise IndexError("Planner returned an invalid action index")
        self._last_actions = tuple(actions)
        chosen = actions[chosen_idx]
        if isinstance(chosen, (np.integer, int)):
            return int(chosen)
        if isinstance(chosen, np.ndarray) and chosen.shape == ():
            return int(chosen.item())
        raise TypeError("Action provider must yield integer-typed arm identifiers")

    def update(
        self,
        t: int,
        a: int,
        x: float,
        *,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        del t, info
        self.prior.update(a, x)


class BetaBernoulliPrior:
    """Independent Beta priors for Bernoulli arms."""

    def __init__(
        self,
        alpha: Optional[Sequence[float]] = None,
        beta: Optional[Sequence[float]] = None,
    ) -> None:
        if (alpha is None) != (beta is None):
            raise ValueError("alpha and beta must be provided together")
        if alpha is not None and len(alpha) != len(beta or []):
            raise ValueError("alpha and beta must have the same length")
        self._alpha0 = None if alpha is None else list(alpha)
        self._beta0 = None if beta is None else list(beta)
        self.alpha: List[float] = []
        self.beta: List[float] = []

    def reset(self, *, n_arms: int, **_: object) -> None:
        if self._alpha0 is None or self._beta0 is None:
            self._alpha0 = [1.0] * n_arms
            self._beta0 = [1.0] * n_arms
        elif n_arms != len(self._alpha0):
            raise ValueError("Number of arms does not match prior dimension")
        self.alpha = list(self._alpha0)
        self.beta = list(self._beta0)

    def sample(self) -> List[float]:
        return [random.betavariate(self.alpha[k], self.beta[k]) for k in range(len(self.alpha))]

    def update(self, arm: int, reward: float) -> None:
        if not (math.isclose(reward, 0.0, abs_tol=1e-9) or math.isclose(reward, 1.0, abs_tol=1e-9)):
            raise ValueError("Bernoulli rewards must lie in {0, 1}")
        reward_int = int(round(reward))
        self.alpha[arm] += reward_int
        self.beta[arm] += 1 - reward_int


class BernoulliBanditPlanner:
    """Planner that selects the arm with highest sampled mean."""

    def __init__(self, *, tie_break: str = "random") -> None:
        if tie_break not in {"first", "random"}:
            raise ValueError("tie_break must be 'first' or 'random'")
        self.tie_break = tie_break

    def __call__(
        self,
        theta_hat: Iterable[float],
        action_space: Sequence[object],
        reward_fn: Callable[[float], float],
        likelihood: Optional[object],
    ) -> int:
        del reward_fn, likelihood
        samples = list(theta_hat)
        if len(samples) != len(action_space):
            raise ValueError("Theta dimension and action space size mismatch")
        best_value = max(samples)
        candidates = [i for i, value in enumerate(samples) if math.isclose(value, best_value, rel_tol=1e-12, abs_tol=1e-12)]
        if self.tie_break == "first":
            return candidates[0]
        return random.choice(candidates)


class BernoulliThompsonPolicy(ThompsonSamplingPolicy):
    """Classical Beta-Bernoulli Thompson Sampling (Algorithm 2)."""

    name = "thompson-bernoulli"

    def __init__(
        self,
        alpha: Optional[Sequence[float]] = None,
        beta: Optional[Sequence[float]] = None,
        *,
        tie_break: str = "random",
    ) -> None:
        prior = BetaBernoulliPrior(alpha, beta)
        super().__init__(
            prior=prior,
            planner=BernoulliBanditPlanner(tie_break=tie_break),
            action_space=None,
            reward_function=lambda y: y,
        )


class NormalNormalArm:
    """Sufficient statistics for a Normal-Normal conjugate arm."""

    def __init__(self, mean0: float, var0: float) -> None:
        if var0 <= 0.0:
            raise ValueError("Prior variance must be positive")
        self.mean0 = float(mean0)
        self.var0 = float(var0)
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.sum_rewards = 0.0

    def posterior_params(self, sigma2: float) -> Tuple[float, float]:
        precision0 = 1.0 / self.var0
        precision_likelihood = self.n / sigma2
        precision_post = precision0 + precision_likelihood
        var_post = 1.0 / precision_post
        mean_post = var_post * (precision0 * self.mean0 + (self.sum_rewards / sigma2))
        return mean_post, var_post


class GaussianArmsPrior:
    """Independent Normal priors for Gaussian arms with known variance."""

    def __init__(
        self,
        means0: Sequence[float],
        variances0: Sequence[float],
        sigma2: float,
    ) -> None:
        if len(means0) != len(variances0):
            raise ValueError("Prior means and variances must have the same length")
        if sigma2 <= 0.0:
            raise ValueError("Known reward variance must be positive")
        self._means0 = list(map(float, means0))
        self._vars0 = list(map(float, variances0))
        self.sigma2 = float(sigma2)
        self.arms: List[NormalNormalArm] = [NormalNormalArm(m, v) for m, v in zip(self._means0, self._vars0)]

    def reset(self, *, n_arms: int, **_: object) -> None:
        if n_arms != len(self._means0):
            raise ValueError("Number of arms does not match prior dimension")
        for arm in self.arms:
            arm.reset()

    def sample(self) -> List[float]:
        samples: List[float] = []
        for arm in self.arms:
            mean_post, var_post = arm.posterior_params(self.sigma2)
            samples.append(random.gauss(mean_post, math.sqrt(var_post)))
        return samples

    def update(self, arm: int, reward: float) -> None:
        self.arms[arm].n += 1
        self.arms[arm].sum_rewards += float(reward)


class GaussianThompsonPolicy(ThompsonSamplingPolicy):
    """Thompson Sampling for Gaussian rewards with known variance."""

    name = "thompson-gaussian"

    def __init__(
        self,
        means0: Sequence[float],
        variances0: Sequence[float],
        sigma2: float,
        *,
        tie_break: str = "random",
    ) -> None:
        prior = GaussianArmsPrior(means0, variances0, sigma2)
        super().__init__(
            prior=prior,
            planner=BernoulliBanditPlanner(tie_break=tie_break),
            action_space=None,
            reward_function=lambda y: y,
        )


class BayesianLinearPrior:
    """Bayesian linear regression posterior for contextual Thompson Sampling."""

    def __init__(
        self,
        d: int,
        *,
        sigma2: float = 1.0,
        mu0: Optional[np.ndarray] = None,
        Sigma0: Optional[np.ndarray] = None,
    ) -> None:
        if d <= 0:
            raise ValueError("Dimension must be positive")
        if sigma2 <= 0.0:
            raise ValueError("Observation noise variance must be positive")
        self.d = int(d)
        self.sigma2 = float(sigma2)
        self._mu0 = np.zeros(self.d) if mu0 is None else np.asarray(mu0, dtype=float)
        if self._mu0.shape != (self.d,):
            raise ValueError("mu0 must have shape (d,)")
        self._Sigma0 = np.eye(self.d) if Sigma0 is None else np.asarray(Sigma0, dtype=float)
        if self._Sigma0.shape != (self.d, self.d):
            raise ValueError("Sigma0 must have shape (d, d)")
        self.reset()

    def reset(self, **_: object) -> None:
        self.precision = np.linalg.inv(self._Sigma0)
        self.information = self.precision @ self._mu0
        self.mu = self._mu0.copy()

    def sample(self) -> np.ndarray:
        cov = np.linalg.inv(self.precision)
        return np.random.multivariate_normal(self.mu, cov)

    def update(self, context: np.ndarray, reward: float) -> None:
        a = np.asarray(context, dtype=float).reshape(self.d)
        self.precision += np.outer(a, a) / self.sigma2
        self.information += (a * (reward / self.sigma2))
        self.mu = np.linalg.solve(self.precision, self.information)


class LinearPlanner:
    """Greedy planner that maximises the sampled linear payoff."""

    def __call__(
        self,
        theta_hat: Iterable[float],
        action_space: Sequence[object],
        reward_fn: Callable[[float], float],
        likelihood: Optional[object],
    ) -> int:
        del reward_fn, likelihood
        theta = np.asarray(theta_hat, dtype=float)
        values = [float(np.dot(theta, np.asarray(a, dtype=float))) for a in action_space]
        best_value = max(values)
        for idx, value in enumerate(values):
            if math.isclose(value, best_value, rel_tol=1e-12, abs_tol=1e-12):
                return idx
        return int(np.argmax(values))


class LinearThompsonPolicy(ThompsonSamplingPolicy):
    """Contextual Thompson Sampling using Bayesian linear regression."""

    name = "thompson-linear"

    def __init__(
        self,
        d: int,
        *,
        sigma2: float = 1.0,
        mu0: Optional[np.ndarray] = None,
        Sigma0: Optional[np.ndarray] = None,
        action_space: Optional[Callable[[int, Optional[History]], Sequence[object]]] = None,
    ) -> None:
        prior = BayesianLinearPrior(d, sigma2=sigma2, mu0=mu0, Sigma0=Sigma0)
        super().__init__(
            prior=prior,
            planner=LinearPlanner(),
            action_space=action_space,
            reward_function=lambda y: y,
        )
        self._last_context: Optional[np.ndarray] = None

    def choose(self, t: int, history: History) -> int:
        theta_hat = self.prior.sample()
        actions = self._current_action_set(t, history)
        contexts = [np.asarray(a, dtype=float) for a in actions]
        index = self.planner(theta_hat, contexts, self.reward_function, self.likelihood)
        self._last_context = contexts[index]
        return int(index)

    def update(
        self,
        t: int,
        a: int,
        x: float,
        *,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        del a, info
        if self._last_context is None:
            raise RuntimeError("Context must be available before updating the linear prior")
        self.prior.update(self._last_context, x)
        self._last_context = None


def laplace_ts_sample(
    theta0: np.ndarray,
    grad_logpost: Callable[[np.ndarray], np.ndarray],
    hess_logpost: Callable[[np.ndarray], np.ndarray],
    *,
    newton_steps: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """Approximate Thompson sample via Laplace's method.

    Parameters
    ----------
    theta0:
        Initial iterate for Newton's method.
    grad_logpost / hess_logpost:
        Callables returning the gradient and Hessian of the log-posterior.
    newton_steps:
        Maximum number of Newton iterations for the MAP search.
    tol:
        Stopping tolerance on the parameter update.
    """

    theta = np.asarray(theta0, dtype=float)
    for _ in range(newton_steps):
        grad = np.asarray(grad_logpost(theta), dtype=float)
        hess = np.asarray(hess_logpost(theta), dtype=float)
        step = np.linalg.solve(-hess, grad)
        step_norm = np.linalg.norm(step)
        theta_next = theta + step
        if step_norm < tol:
            theta = theta_next
            break
        theta = theta_next
    hessian = np.asarray(hess_logpost(theta), dtype=float)
    cov = np.linalg.inv(-hessian)
    return np.random.multivariate_normal(theta, cov)


def sgld_ts_sample(
    theta_init: np.ndarray,
    grad_logpost: Callable[[np.ndarray], np.ndarray],
    *,
    steps: int = 200,
    step_size: float = 1e-3,
    minibatch_grad: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    preconditioner: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate an approximate posterior sample using (preconditioned) SGLD."""

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    theta = np.asarray(theta_init, dtype=float).copy()
    dim = theta.shape[0]
    if preconditioner is None:
        preconditioner = np.eye(dim)
    chol = np.linalg.cholesky(preconditioner)
    for _ in range(steps):
        grad = minibatch_grad(theta) if minibatch_grad is not None else grad_logpost(theta)
        noise = chol @ np.random.normal(size=dim)
        theta = theta + step_size * (preconditioner @ grad) + math.sqrt(2.0 * step_size) * noise
    return theta


def bootstrap_map_sample(
    resample_history: Callable[[], Sequence[object]],
    log_likelihood: Callable[[np.ndarray, Sequence[object]], float],
    sample_from_prior: Callable[[], np.ndarray],
    prior_cov: np.ndarray,
    *,
    max_iter: int = 200,
    step_size: float = 1e-2,
) -> np.ndarray:
    """Approximate posterior draw via bootstrap-anchored MAP optimisation."""

    history_sample = resample_history()
    theta_anchor = np.asarray(sample_from_prior(), dtype=float)
    theta = theta_anchor.copy()
    cov_inv = np.linalg.inv(np.asarray(prior_cov, dtype=float))

    def log_posterior(theta_vec: np.ndarray) -> float:
        diff = theta_vec - theta_anchor
        prior_term = -0.5 * float(diff.T @ cov_inv @ diff)
        return prior_term + float(log_likelihood(theta_vec, history_sample))

    def approx_grad(theta_vec: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        grad = np.zeros_like(theta_vec)
        for i in range(theta_vec.size):
            e = np.zeros_like(theta_vec)
            e[i] = eps
            grad[i] = (log_posterior(theta_vec + e) - log_posterior(theta_vec - e)) / (2.0 * eps)
        return grad

    for _ in range(max_iter):
        gradient = approx_grad(theta)
        theta = theta + step_size * gradient

    return theta


def ensemble_ts_pick(
    parameter_samples: Sequence[np.ndarray],
    action_space: Sequence[object],
    value_fn: Callable[[np.ndarray, object], float],
) -> int:
    """Select an action using an ensemble of approximate Thompson samples."""

    if not parameter_samples:
        raise ValueError("At least one parameter sample is required")
    idx = random.randrange(len(parameter_samples))
    theta = parameter_samples[idx]
    values = [float(value_fn(theta, action)) for action in action_space]
    return int(np.argmax(values))



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
        policy.update(t, arm, reward, info=None)
        actions.append(arm)
        rewards.append(reward)

    return actions, rewards
