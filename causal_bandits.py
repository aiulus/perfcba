"""Causal bandit algorithms from Lattimore, Lattimore, and Reid (2016).

This module provides reference implementations of the two main algorithms in
"Parallel Bandits: Understanding and Exploiting Causal Structures".  The goal is
simple regret minimisation when interacting with a structural causal model with a
known graph.

The algorithms are provided in a lightweight, simulator-agnostic style so they
can be plugged into different experimental harnesses.  Users supply callables
for drawing samples under the required interventions as well as the structural
information (for Algorithm 2).
"""

from __future__ import annotations

import collections
import math
import random
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Sequence, Tuple


__all__ = [
    "m_of_q",
    "parallel_bandit_algorithm",
    "compute_Q",
    "compute_m_eta",
    "choose_truncation_B",
    "general_causal_bandit_algorithm",
    "approx_minimize_m_eta",
]


# ---------------------------------------------------------------------------
# Algorithm 1 utilities
# ---------------------------------------------------------------------------

def m_of_q(q: Sequence[float]) -> int:
    """Return the difficulty index :math:`m(q)` for a Bernoulli vector ``q``.

    The index is defined in Lattimore et al. (2016, Definition 3) as::

        m(q) = min { tau in {2, …, N} : |I_tau| <= tau },
        I_tau = { i : min(q_i, 1 - q_i) < 1 / tau }.

    Parameters
    ----------
    q:
        Empirical means :math:`q_i = P(X_i = 1)` for the observational regime.

    Returns
    -------
    int
        The smallest admissible :math:`tau`, bounded above by ``len(q)``.
    """

    n = len(q)
    if n == 0:
        raise ValueError("q must contain at least one probability")

    for tau in range(2, n + 1):
        i_tau = [i for i, p in enumerate(q) if min(p, 1.0 - p) < 1.0 / tau]
        if len(i_tau) <= tau:
            return tau
    return n


def parallel_bandit_algorithm(
    T: int,
    N: int,
    env_pull_do_empty: Callable[[], Tuple[Sequence[int], int]],
    env_pull_do_single: Callable[[int, int], int],
) -> Tuple[Tuple[int, int], Dict[Tuple[int, int], float]]:
    """Implementation of Algorithm 1 (Parallel Bandit) from LLR16.

    Parameters
    ----------
    T:
        Total interaction budget.
    N:
        Number of parents of the reward node (binary variables).
    env_pull_do_empty:
        Callable returning ``(x, y)`` where ``x`` is the full :math:`{0,1}^N`
        vector sampled under the empty intervention and ``y`` is the reward.
    env_pull_do_single:
        Callable returning a reward when the intervention ``do(X_i = j)`` is
        applied.

    Returns
    -------
    best_arm:
        Tuple ``(i, j)`` describing the recommended intervention.
    mu_hat:
        Dictionary of empirical means for each single-variable intervention.

    Notes
    -----
    The algorithm splits the budget into an observational phase followed by a
    targeted phase focused on "rare" arms.  The definition of rare arms follows
    the paper exactly and is governed by ``m_of_q`` above.
    """

    if T <= 0:
        raise ValueError("Budget T must be positive")
    if N <= 0:
        raise ValueError("Number of parents N must be positive")

    # Phase split (Algorithm 1, lines 2-3)
    T_obs = T // 2
    T_tar = T - T_obs

    count_a: Dict[Tuple[int, int], int] = {(i, j): 0 for i in range(N) for j in (0, 1)}
    sumY_a: Dict[Tuple[int, int], float] = {(i, j): 0.0 for i in range(N) for j in (0, 1)}

    # Phase 1: draw T_obs samples under do() and reuse occurrences of X_i=j
    for _ in range(T_obs):
        x, y = env_pull_do_empty()
        if len(x) != N:
            raise ValueError("env_pull_do_empty returned a vector of unexpected length")
        for i in range(N):
            j = x[i]
            if j not in (0, 1):
                raise ValueError("env_pull_do_empty must return binary assignments")
            count_a[(i, j)] += 1
            sumY_a[(i, j)] += y

    # Empirical frequencies (Algorithm 1, line 8)
    p_hat: Dict[Tuple[int, int], float] = {}
    for i in range(N):
        for j in (0, 1):
            p_hat[(i, j)] = (2.0 * count_a[(i, j)] / T) if T > 0 else 0.0

    q_hat = [p_hat[(i, 1)] for i in range(N)]
    m_hat = m_of_q(q_hat) if q_hat else 2

    # Identify rare arms (Algorithm 1, line 9)
    denominator = max(2, m_hat)
    rare_arms = [
        (i, j) for (i, j), probability in p_hat.items() if probability <= 1.0 / denominator
    ]
    rare_arms.sort()

    mu_hat: Dict[Tuple[int, int], float] = {}
    for key, total in sumY_a.items():
        observations = count_a[key]
        mu_hat[key] = total / observations if observations > 0 else math.nan

    # Phase 2: targeted sampling of rare arms (Algorithm 1, lines 10-14)
    if rare_arms:
        allocations = max(1, len(rare_arms))
        T_per_arm = T_tar // allocations
    else:
        T_per_arm = 0

    for i, j in rare_arms:
        if T_per_arm == 0:
            break
        total = 0.0
        for _ in range(T_per_arm):
            y = env_pull_do_single(i, j)
            total += y
        mu_hat[(i, j)] = total / T_per_arm if T_per_arm > 0 else mu_hat[(i, j)]

    # Recommend the best single-variable arm according to empirical mean
    def value(item: Tuple[Tuple[int, int], float]) -> float:
        _, mean = item
        return mean if not math.isnan(mean) else float("-inf")

    best_arm = max(mu_hat.items(), key=value)[0]
    return best_arm, mu_hat


# ---------------------------------------------------------------------------
# Algorithm 2 utilities
# ---------------------------------------------------------------------------

def compute_Q(
    eta: MutableMapping[Any, float],
    P_Pa_given_a: MutableMapping[Any, Callable[[Tuple[Any, ...]], float]],
    support_PaY: Iterable[Tuple[Any, ...]],
) -> Dict[Tuple[Any, ...], float]:
    """Return the mixture distribution ``Q`` over parent assignments.

    The mixture is defined as ``Q(x) = sum_a eta[a] * P(Pa_Y = x | a)``.  The
    support needs to cover every parent assignment with positive probability for
    any considered arm, otherwise the importance weights would be undefined.
    """

    Q: Dict[Tuple[Any, ...], float] = collections.defaultdict(float)
    for x in support_PaY:
        Q[x] = sum(eta[a] * P_Pa_given_a[a](x) for a in eta)
    return Q


def compute_m_eta(
    eta: MutableMapping[Any, float],
    P_Pa_given_a: MutableMapping[Any, Callable[[Tuple[Any, ...]], float]],
    support_PaY: Iterable[Tuple[Any, ...]],
) -> float:
    """Difficulty index ``m(eta)`` for the general causal bandit algorithm."""

    Q = compute_Q(eta, P_Pa_given_a, support_PaY)

    def ratio_sum(arm: Any) -> float:
        total = 0.0
        for x in support_PaY:
            p_ax = P_Pa_given_a[arm](x)
            q_x = Q[x]
            if p_ax == 0:
                continue
            if q_x <= 0:
                raise ValueError(
                    "Support mismatch: Q(x) is zero while P(Pa_Y=x|a) is positive."
                )
            total += p_ax * (p_ax / q_x)
        return total

    return max(ratio_sum(a) for a in eta)


def choose_truncation_B(T: int, A_size: int, m_eta: float) -> float:
    """Return the truncation level suggested by Theorem 3 of LLR16."""

    if T <= 0:
        raise ValueError("Budget T must be positive")
    if A_size <= 0:
        raise ValueError("Number of arms must be positive")
    if m_eta <= 0:
        raise ValueError("m_eta must be positive")

    denom = math.log(2.0 * T * A_size)
    denom = max(denom, 1.0)
    return math.sqrt(m_eta * T / denom)


def general_causal_bandit_algorithm(
    T: int,
    A: Sequence[Any],
    eta: MutableMapping[Any, float],
    P_Pa_given_a: MutableMapping[Any, Callable[[Tuple[Any, ...]], float]],
    support_PaY: Iterable[Tuple[Any, ...]],
    extract_PaY: Callable[[Any], Tuple[Any, ...]],
    env_pull: Callable[[Any], Tuple[Any, int]],
    adaptive_B: bool = False,
    rng: random.Random | None = None,
) -> Tuple[Any, Dict[Any, float]]:
    """Implementation of Algorithm 2 (general causal bandit) from LLR16.

    The algorithm samples interventions i.i.d. from the design distribution
    ``eta`` and uses truncated importance weighting to estimate the mean reward
    of every arm simultaneously.
    """

    if T <= 0:
        raise ValueError("Budget T must be positive")
    if not A:
        raise ValueError("Arm list must not be empty")
    if abs(sum(eta.get(a, 0.0) for a in A) - 1.0) > 1e-8:
        raise ValueError("eta must define a probability distribution over A")

    rng = rng or random

    Q = compute_Q(eta, P_Pa_given_a, support_PaY)
    m_eta = compute_m_eta(eta, P_Pa_given_a, support_PaY)
    base_B = choose_truncation_B(T, len(A), m_eta)
    truncation = math.inf if adaptive_B else base_B

    observations: List[Tuple[Any, ...]] = []
    rewards: List[int] = []

    # Fixed design sampling (Algorithm 2, lines 4-6)
    for _ in range(T):
        r = rng.random()
        cumulative = 0.0
        chosen_arm = A[-1]
        for arm in A:
            cumulative += eta[arm]
            if r <= cumulative:
                chosen_arm = arm
                break
        x_full, y = env_pull(chosen_arm)
        observations.append(extract_PaY(x_full))
        rewards.append(y)

    # Post-processing: compute truncated importance-weighted estimates
    mu_hat: Dict[Any, float] = {}
    for arm in A:
        total = 0.0
        for x_pa, y in zip(observations, rewards):
            p_ax = P_Pa_given_a[arm](x_pa)
            q_x = Q[x_pa]
            if q_x <= 0:
                raise ValueError(
                    "Support mismatch: encountered Pa_Y value with zero mixture probability."
                )
            ratio = p_ax / q_x if q_x > 0 else 0.0
            if ratio <= truncation:
                total += y * ratio
        mu_hat[arm] = total / T

    best_arm = max(mu_hat.items(), key=lambda kv: kv[1])[0]
    return best_arm, mu_hat


# ---------------------------------------------------------------------------
# Helper routines
# ---------------------------------------------------------------------------

def _project_to_simplex(v: Dict[Any, float]) -> Dict[Any, float]:
    """Project a dictionary of weights onto the probability simplex."""

    items = list(v.items())
    values = [max(0.0, weight) for _, weight in items]
    total = sum(values)
    if total == 0.0:
        # revert to uniform distribution
        uniform = 1.0 / len(values)
        return {key: uniform for key, _ in items}

    sorted_values = sorted(values, reverse=True)
    cumulative = 0.0
    rho = -1
    theta = 0.0
    for idx, value in enumerate(sorted_values, start=1):
        cumulative += value
        t = (cumulative - 1.0) / idx
        if value - t > 0:
            rho = idx
            theta = t
    if rho == -1:
        theta = (cumulative - 1.0) / len(values)
    projected: Dict[Any, float] = {}
    for (key, _), value in zip(items, values):
        projected[key] = max(0.0, value - theta)

    total_projected = sum(projected.values())
    if total_projected <= 0:
        uniform = 1.0 / len(projected)
        return {key: uniform for key in projected}

    for key in projected:
        projected[key] /= total_projected
    return projected


def approx_minimize_m_eta(
    A: Sequence[Any],
    P_Pa_given_a: MutableMapping[Any, Callable[[Tuple[Any, ...]], float]],
    support_PaY: Iterable[Tuple[Any, ...]],
    steps: int = 500,
    lr: float = 0.5,
) -> Dict[Any, float]:
    """Heuristic subgradient descent for minimising ``m(eta)`` over the simplex.

    This helper is not part of the original paper but is provided to ease the
    choice of design distribution when an analytic solution is not available.
    The routine performs a projected subgradient descent with a constant step
    size.  The iterate is projected back to the simplex after every update.
    """

    if not A:
        raise ValueError("Arm list must not be empty")
    eta = {arm: 1.0 / len(A) for arm in A}

    for _ in range(max(1, steps)):
        Q = compute_Q(eta, P_Pa_given_a, support_PaY)

        def value(arm: Any) -> float:
            total = 0.0
            for x in support_PaY:
                p_ax = P_Pa_given_a[arm](x)
                q_x = Q[x]
                if p_ax == 0:
                    continue
                if q_x <= 0:
                    raise ValueError(
                        "Support mismatch: Q(x) is zero while P(Pa_Y=x|a) is positive."
                    )
                total += p_ax * (p_ax / q_x)
            return total

        worst_arm = max(A, key=value)

        gradient = {arm: 0.0 for arm in A}
        for x in support_PaY:
            p_worst = P_Pa_given_a[worst_arm](x)
            denom = Q[x]
            if denom <= 0:
                continue
            for arm in A:
                p_ax = P_Pa_given_a[arm](x)
                gradient[arm] -= p_worst * (p_ax / (denom * denom))

        for arm in A:
            eta[arm] = eta[arm] - lr * gradient[arm]

        eta = _project_to_simplex(eta)

    return eta