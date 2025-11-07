"""Reference implementations of causal bandit algorithms.

This module collects faithful Python translations of the algorithms presented in
*Learning Good Interventions via Causal Inference* (Lattimore, Lattimore and
Reid, 2016) and *Structural Causal Bandits: Where to Intervene?* (Lee and
Bareinboim, 2018).  The goal is to expose the exact decision rules used in the
original works so that they can be re-used in different experimental harnesses.

The implementation is intentionally close to the pseudo-code and public
reference implementations released by the authors.  Only light engineering
wrapping is applied (type annotations, argument validation and numerical
safeguards) to make the routines easier to compose with the rest of the code
base while staying behaviourally equivalent to the source material.
"""

from __future__ import annotations

import collections
import itertools
import math
import numbers
import random
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np


# =============================================================================
# Algorithms from Lattimore, Lattimore & Reid (2016)
# =============================================================================

def m_of_q(q: Sequence[float]) -> int:
    """Return the difficulty index :math:`m(q)` for a Bernoulli vector ``q``."""

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
    env_pull_do_empty: Callable[[], Tuple[Sequence[int], float]],
    env_pull_do_single: Callable[[int, int], float],
) -> Tuple[Union[str, Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """Implementation of Algorithm 1 (Parallel Bandit) from LLR16."""

    if T <= 0:
        raise ValueError("Budget T must be positive")
    if N <= 0:
        raise ValueError("Number of parents N must be positive")

    T_obs = T // 2
    T_tar = T - T_obs

    count_a: Dict[Tuple[int, int], int] = {(i, j): 0 for i in range(N) for j in (0, 1)}
    sumY_a: Dict[Tuple[int, int], float] = {(i, j): 0.0 for i in range(N) for j in (0, 1)}
    baseline_sum = 0.0

    for _ in range(T_obs):
        x, y = env_pull_do_empty()
        if len(x) != N:
            raise ValueError("env_pull_do_empty returned a vector of unexpected length")
        baseline_sum += y
        for i in range(N):
            j = x[i]
            if j not in (0, 1):
                raise ValueError("env_pull_do_empty must return binary assignments")
            count_a[(i, j)] += 1
            sumY_a[(i, j)] += y

    p_hat: Dict[Tuple[int, int], float] = {}
    for i in range(N):
        for j in (0, 1):
            p_hat[(i, j)] = (2.0 * count_a[(i, j)] / T) if T > 0 else 0.0

    q_hat = [p_hat[(i, 1)] for i in range(N)]
    m_hat = m_of_q(q_hat) if q_hat else 2

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
        base_alloc = T_tar // len(rare_arms)
        remainder = T_tar % len(rare_arms)
    else:
        base_alloc = 0
        remainder = 0

    for index, (i, j) in enumerate(rare_arms):
        pulls = base_alloc + (1 if index < remainder else 0)
        if pulls <= 0:
            continue
        total = 0.0
        for _ in range(pulls):
            y = env_pull_do_single(i, j)
            total += y
        mu_hat[(i, j)] = total / pulls

    mu_hat_do = baseline_sum / T_obs if T_obs > 0 else math.nan

    best_arm: Union[str, Tuple[int, int]] = "do()"
    best_val = mu_hat_do if not math.isnan(mu_hat_do) else float("-inf")

    for arm, mean in mu_hat.items():
        value = mean if not math.isnan(mean) else float("-inf")
        if value > best_val:
            best_val = value
            best_arm = arm

    return best_arm, mu_hat


def compute_Q(
    eta: MutableMapping[Any, float],
    P_Pa_given_a: MutableMapping[Any, Callable[[Tuple[Any, ...]], float]],
    support_PaY: Iterable[Tuple[Any, ...]],
) -> Dict[Tuple[Any, ...], float]:
    """Return the mixture distribution ``Q`` over parent assignments.
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
            if p_ax == 0.0:
                continue
            if q_x <= 0.0:
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
    env_pull: Callable[[Any], Tuple[Any, float]],
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
    rewards: List[float] = []

    # Fixed design sampling (Algorithm 2, lines 4-6)
    probabilities = np.fromiter((eta[arm] for arm in A), dtype=float, count=len(A))
    if np.any(probabilities < 0.0):
        raise ValueError("eta must not assign negative mass to any arm")
    total_mass = float(probabilities.sum())
    if not math.isclose(total_mass, 1.0, rel_tol=1e-8, abs_tol=1e-12):
        if total_mass <= 0.0:
            raise ValueError("eta must assign positive total mass")
        probabilities = probabilities / total_mass

    def _draw_arm() -> Any:
        if hasattr(rng, "choice"):
            try:
                # Prefer numpy-style generators when available.
                index = rng.choice(len(A), p=probabilities)  # type: ignore[arg-type]
                return A[int(index)]
            except (TypeError, AttributeError):
                # Fall back to manual sampling below.
                pass
        random_fn = getattr(rng, "random", None)
        if random_fn is None:
            random_fn = random.random
        # Use 1 - U to retain uniformity while matching expected seeded draws.
        u = 1.0 - float(random_fn())
        cumulative = 0.0
        chosen = A[-1]
        for arm, prob in zip(A, probabilities):
            cumulative += prob
            if u <= cumulative:
                chosen = arm
                break
        return chosen

    for _ in range(T):
        chosen_arm = _draw_arm()
        x_full, y = env_pull(chosen_arm)
        observations.append(extract_PaY(x_full))
        rewards.append(float(y))

    # Post-processing: compute truncated importance-weighted estimates
    mu_hat: Dict[Any, float] = {}
    for arm in A:
        total = 0.0
        for x_pa, y in zip(observations, rewards):
            p_ax = P_Pa_given_a[arm](x_pa)
            q_x = Q[x_pa]
            if q_x <= 0.0:
                raise ValueError(
                    "Support mismatch: encountered Pa_Y value with zero mixture probability."
                )
            ratio = p_ax / q_x if q_x > 0 else 0.0
            if math.isfinite(truncation):
                ratio = min(ratio, truncation)
            total += y * ratio
        mu_hat[arm] = total / T

    best_arm = max(mu_hat.items(), key=lambda kv: kv[1])[0]
    return best_arm, mu_hat


# ---------------------------------------------------------------------------
# Helper routines used by approx_minimize_m_eta
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
                if p_ax == 0.0:
                    continue
                if q_x <= 0.0:
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
            if denom <= 0.0:
                continue
            for arm in A:
                p_ax = P_Pa_given_a[arm](x)
                gradient[arm] -= p_worst * (p_ax / (denom * denom))

        for arm in A:
            eta[arm] = eta[arm] - lr * gradient[arm]

        eta = _project_to_simplex(eta)

    return eta


# =============================================================================
# Algorithms from Lee & Bareinboim (2018)
# =============================================================================


def _node_set(graph) -> Set[str]:
    """Return the set of node names for *graph*.

    The helper is defensive: depending on the underlying graph class the node
    container might live under ``V``, ``nodes`` or ``vertices`` (either as an
    attribute or a method).  The causal bandit routines only rely on the node
    *names*, so converting to a ``set`` keeps downstream code simple.
    """

    for attr in ("V", "nodes", "vertices"):
        if hasattr(graph, attr):
            value = getattr(graph, attr)
            value = value() if callable(value) else value
            if value is not None:
                return set(value)
    raise AttributeError("Graph object does not expose a node container (V/nodes/vertices).")


def _filter_sequence(seq: Sequence[str], allowed: Iterable[str]) -> List[str]:
    """Filter ``seq`` to values that appear in ``allowed`` while preserving order."""

    allowed_set = set(allowed)
    return [x for x in seq if x in allowed_set]


def _ensure_frozenset(items: Iterable[str]) -> FrozenSet[str]:
    return frozenset(set(items))


def rand_argmax(values: Sequence[float]) -> int:
    """Return the index of a maximal entry chosen uniformly at random."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Cannot take argmax of an empty sequence.")
    max_val = np.nanmax(array)
    max_mask = np.isclose(array, max_val, rtol=1e-12, atol=1e-12)
    idxs = np.flatnonzero(max_mask)
    if idxs.size == 1:
        return int(idxs[0])
    return int(np.random.choice(idxs))


# ---------------------------------------------------------------------------
# Minimal intervention sets (MISs)
# ---------------------------------------------------------------------------

def MISs(graph, outcome: str) -> FrozenSet[FrozenSet[str]]:
    """Enumerate all minimal intervention sets (MISs) for ``outcome``.

    Parameters
    ----------
    graph:
        A causal diagram exposing the API used throughout the pseudocode
        (``An``, ``do``, ``causal_order`` and slicing via ``__getitem__``).
    outcome:
        Name of the reward node ``Y`` in the paper.
    """

    ancestors = graph.An(outcome)
    induced = graph[ancestors]
    nodes = _node_set(induced)
    candidate_vars = nodes - {outcome}
    order = induced.causal_order(backward=True)
    order = _filter_sequence(order, candidate_vars)
    results = _subMISs(induced, outcome, frozenset(), order)
    return frozenset(results)


def _subMISs(
    graph,
    outcome: str,
    chosen: FrozenSet[str],
    remaining: Sequence[str],
) -> Set[FrozenSet[str]]:   
    """Recursive helper implementing the MIS enumeration logic."""

    out: Set[FrozenSet[str]] = {chosen}
    for idx, var in enumerate(remaining):
        next_graph = graph.do({var})
        next_graph = next_graph[next_graph.An(outcome)]
        next_nodes = _node_set(next_graph)
        new_chosen = _ensure_frozenset(set(chosen) | {var})
        next_remaining = _filter_sequence(remaining[idx + 1 :], next_nodes)
        out |= _subMISs(next_graph, outcome, new_chosen, next_remaining)
    return out


# ---------------------------------------------------------------------------
# Minimal unobserved-confounder territory (MUCT) and interventional border (IB)
# ---------------------------------------------------------------------------

def MUCT(graph, outcome: str) -> FrozenSet[str]:
    """Compute the minimal unobserved-confounder territory for ``outcome``."""

    induced = graph[graph.An(outcome)]
    territory: Set[str] = {outcome}
    frontier: Set[str] = {outcome}
    while frontier:
        node = frontier.pop()
        component = set(induced.c_component(node))
        new_nodes = component - territory
        territory |= new_nodes
        descendants = set(induced.de(set(component)))
        frontier |= (descendants - territory)
    return _ensure_frozenset(territory)


def IB(graph, outcome: str) -> FrozenSet[str]:
    """Interventional border for ``outcome`` (parents of its MUCT)."""

    territory = MUCT(graph, outcome)
    parents = set(graph.pa(set(territory)))
    return _ensure_frozenset(parents - set(territory))


def MUCT_IB(graph, outcome: str) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """Return both the MUCT and its border (helper for POMIS enumeration)."""

    territory = MUCT(graph, outcome)
    border = _ensure_frozenset(set(graph.pa(set(territory))) - set(territory))
    return territory, border


# ---------------------------------------------------------------------------
# Possibly-optimal MISs (POMISs)
# ---------------------------------------------------------------------------

def POMISs(graph, outcome: str) -> Set[FrozenSet[str]]:
    """Enumerate all possibly-optimal minimal intervention sets (POMISs)."""

    induced = graph[graph.An(outcome)]
    territory, border = MUCT_IB(induced, outcome)
    result: Set[FrozenSet[str]] = {border}
    if border:
        work_graph = induced.do(set(border))
    else:
        work_graph = induced
    slice_nodes = set(territory) | set(border)
    work_graph = work_graph[slice_nodes]
    order = work_graph.causal_order(backward=True)
    order = _filter_sequence(order, set(territory) - {outcome})
    result |= _subPOMISs(work_graph, outcome, order, obs=set())
    return { _ensure_frozenset(x) for x in result }


def _subPOMISs(
    graph,
    outcome: str,
    ordered_vars: Sequence[str],
    obs: Set[str],
) -> Set[FrozenSet[str]]:   
    """Depth-first generation helper mirroring Algorithm 1 from the paper."""

    if not ordered_vars:
        return set()
    out: Set[FrozenSet[str]] = set()
    for idx, var in enumerate(ordered_vars):
        territory, border = MUCT_IB(graph.do({var}), outcome)
        new_obs = obs | set(ordered_vars[:idx])
        if set(border) & new_obs:
            continue
        out.add(border)
        next_vars = _filter_sequence(ordered_vars[idx + 1 :], territory)
        if next_vars:
            next_graph = graph.do(set(border))
            slice_nodes = set(territory) | set(border)
            next_graph = next_graph[slice_nodes]
            out |= _subPOMISs(next_graph, outcome, next_vars, new_obs)
    return out


# ---------------------------------------------------------------------------
# Converting SCMs into bandit machines
# ---------------------------------------------------------------------------

def SCM_to_bandit_machine(
    model,
    outcome: str = "Y",
) -> Tuple[Tuple[float, ...], Dict[int, Dict[str, numbers.Number]]]:
    """Enumerate all interventional arms and their expected rewards."""

    graph = model.G
    nodes = sorted(_node_set(graph) - {outcome})
    mu_values: List[float] = []
    arm_settings: Dict[int, Dict[str, numbers.Number]] = {}
    arm_id = 0

    for r in range(len(nodes) + 1):
        for subset in itertools.combinations(nodes, r):
            domains = [model.D[var] for var in subset]
            for values in itertools.product(*domains) if domains else [()]:
                intervention = dict(zip(subset, values))
                dist = model.query((outcome,), intervention=intervention)
                expectation = 0.0
                for y_val in model.D[outcome]:
                    expectation += y_val * float(dist[(y_val,)])
                mu_values.append(expectation)
                arm_settings[arm_id] = intervention
                arm_id += 1

    return tuple(mu_values), arm_settings


# ---------------------------------------------------------------------------
# Arm filters
# ---------------------------------------------------------------------------

def arms_of(
    kind: str,
    arm_settings: Mapping[int, Mapping[str, numbers.Number]],
    graph,
    outcome: str,
) -> Tuple[int, ...]:
    """Return the arm indices matching ``kind``."""

    kind = kind.lower()
    if kind == "pomis":
        pomis_sets = POMISs(graph, outcome)
        return tuple(
            i for i, setting in arm_settings.items() if frozenset(setting.keys()) in pomis_sets
        )
    if kind == "mis":
        mis_sets = MISs(graph, outcome)
        return tuple(
            i for i, setting in arm_settings.items() if frozenset(setting.keys()) in mis_sets
        )
    if kind == "all-at-once":
        full = _node_set(graph) - {outcome}
        return tuple(i for i, setting in arm_settings.items() if set(setting.keys()) == full)
    if kind == "brute-force":
        return tuple(sorted(arm_settings.keys()))
    raise ValueError(f"Unknown arm kind: {kind}")


# ---------------------------------------------------------------------------
# KL-UCB utilities
# ---------------------------------------------------------------------------

def default_klUCB_exploration(t: int) -> float:
    """Exploration schedule used by KL-UCB."""

    if t < 3:
        return 1.0
    return math.log(t) + 3.0 * math.log(math.log(t))


def KL(p: float, q: float, eps: float = 1e-12) -> float:
    """Bernoulli KL divergence with clamping for numerical stability."""

    p = float(np.clip(p, eps, 1.0 - eps))
    q = float(np.clip(q, eps, 1.0 - eps))
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def sup_KL(mu_ref: float, divergence: float, lower: Optional[float] = None) -> float:
    """Compute ``sup_{mu >= mu_ref} { KL(mu_ref, mu) <= divergence }`` via bisection."""

    mu_ref = float(mu_ref)
    if divergence <= 0.0:
        return mu_ref
    if KL(mu_ref, 1.0) <= divergence:
        return 1.0
    low = mu_ref if lower is None else float(lower)
    high = 1.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if KL(mu_ref, mid) <= divergence:
            low = mid
        else:
            high = mid
    return low


def KL_UCB_run(
    horizon: int,
    mu_true: Sequence[float],
    allowed_arms: Sequence[int],
    exploration_schedule: Callable[[int], float] = default_klUCB_exploration,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate KL-UCB restricted to ``allowed_arms`` with Bernoulli rewards."""

    mu_true = np.asarray(mu_true, dtype=float)
    num_arms = mu_true.size
    allowed = np.asarray(tuple(allowed_arms), dtype=int)
    if allowed.size == 0:
        raise ValueError("KL-UCB requires at least one allowed arm.")

    pulls = np.zeros(horizon, dtype=int)
    rewards = np.zeros(horizon, dtype=float)
    counts = np.zeros(num_arms, dtype=float)
    estimates = np.zeros(num_arms, dtype=float)

    warm = min(horizon, allowed.size)
    for t in range(warm):
        arm = allowed[t]
        reward = float(np.random.random() <= mu_true[arm])
        pulls[t] = arm
        rewards[t] = reward
        counts[arm] += 1.0
        estimates[arm] += (reward - estimates[arm]) / counts[arm]

    if warm == horizon:
        return pulls, rewards

    upper_bounds = np.zeros(num_arms, dtype=float)
    for arm in range(num_arms):
        if counts[arm] > 0.0:
            upper_bounds[arm] = sup_KL(
                estimates[arm], exploration_schedule(warm) / counts[arm]
        )
        else:
            upper_bounds[arm] = 1.0

    for t in range(warm, horizon):
        arm = allowed[rand_argmax(upper_bounds[allowed])]
        reward = float(np.random.random() <= mu_true[arm])
        pulls[t] = arm
        rewards[t] = reward
        counts[arm] += 1.0
        estimates[arm] += (reward - estimates[arm]) / counts[arm]
        upper_bounds[arm] = sup_KL(
            estimates[arm], exploration_schedule(t + 1) / counts[arm]
        )
    return pulls, rewards


# ---------------------------------------------------------------------------
# Thompson sampling
# ---------------------------------------------------------------------------

def Thompson_run(
    horizon: int,
    mu_true: Sequence[float],
    allowed_arms: Sequence[int],
    prior_succ_fail: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bernoulli Thompson sampling restricted to ``allowed_arms``."""

    mu_true = np.asarray(mu_true, dtype=float)
    num_arms = mu_true.size
    allowed = tuple(int(a) for a in allowed_arms)
    if not allowed:
        raise ValueError("Thompson sampling requires at least one allowed arm.")

    successes = np.zeros(num_arms, dtype=float)
    failures = np.zeros(num_arms, dtype=float)
    if prior_succ_fail is not None:
        s, f = prior_succ_fail
        successes[: len(s)] = np.asarray(s, dtype=float)
        failures[: len(f)] = np.asarray(f, dtype=float)

    pulls = np.zeros(horizon, dtype=int)
    rewards = np.zeros(horizon, dtype=float)
    for t in range(horizon):
        theta = np.array(
            [np.random.beta(successes[i] + 1.0, failures[i] + 1.0) for i in range(num_arms)]
        )
        arm = allowed[rand_argmax(theta[list(allowed)])]
        reward = float(np.random.random() <= mu_true[arm])
        pulls[t] = arm
        rewards[t] = reward
        if reward > 0.0:
            successes[arm] += 1.0
        else:
            failures[arm] += 1.0

    return pulls, rewards

# =============================================================================
# RAPS primitives (used for follow-up experiments)
# =============================================================================

class PCM(Protocol):
    """Protocol describing the probabilistic causal model interface.

    A PCM exposes access to the nodes (``V``), the discrete domain size ``K``
    shared by all nodes, the name of the reward node, and two sampling oracles:
    ``observe`` for observational data and ``intervene`` for interventional
    data.  The definitions match the model assumed in the paper.
    """

    V: Sequence[str]
    K: int
    reward_node: str

    def observe(self, B: int) -> List[MutableMapping[str, int]]:
        """Draw ``B`` observational samples from ``P``."""

    def intervene(self, do: MutableMapping[str, int], B: int) -> List[MutableMapping[str, int]]:
        """Draw ``B`` samples from ``P(. | do(do))``."""


def empirical_mean_R(samples: Sequence[MutableMapping[str, int]], reward_node: str) -> float:
    if not samples:
        raise ValueError("At least one sample is required to compute the empirical mean.")

    total = 0.0
    for sample in samples:
        total += float(sample[reward_node])
    return total / len(samples)


def empirical_marginal(samples: Sequence[MutableMapping[str, int]], node: str, K: int) -> List[float]:
    if K <= 0:
        raise ValueError("Domain size K must be positive.")
    if not samples:
        raise ValueError("At least one sample is required to compute marginals.")

    counts = [0.0 for _ in range(K)]
    for sample in samples:
        value = sample[node]
        if not 0 <= value < K:
            raise ValueError(
                f"Sampled value {value} for node {node} is outside of the expected domain."
            )
        counts[value] += 1.0

    total = float(len(samples))
    return [count / total for count in counts]


def l1_dist(p: Sequence[float], q: Sequence[float]) -> float:
    """Compute the ℓ₁ distance between two discrete distributions."""

    if len(p) != len(q):
        raise ValueError("Distributions must have the same support.")
    return sum(abs(pi - qi) for pi, qi in zip(p, q))


def _sorted_nodes(nodes: Iterable[str]) -> List[str]:
    """Return a deterministic ordering of nodes."""

    return sorted(nodes)


def all_assignments(nodes: Iterable[str], K: int) -> Iterable[Dict[str, int]]:
    """Yield all assignments for ``nodes`` with domain size ``K``.

    Nodes are traversed in sorted order to guarantee determinism.
    """

    node_list = _sorted_nodes(nodes)
    if not node_list:
        yield {}
        return

    for values in itertools.product(range(K), repeat=len(node_list)):
        assignment = {node: value for node, value in zip(node_list, values)}
        yield assignment


@dataclass
class RAPSParams:
    """Finite-sample parameters for the RAPS algorithm."""

    eps: float
    Delta: float
    delta: float

    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise ValueError("Parameter eps must be strictly positive.")
        if self.Delta <= 0:
            raise ValueError("Parameter Delta must be strictly positive.")
        if not (0 < self.delta < 1):
            raise ValueError("Parameter delta must lie in (0, 1).")


def compute_budget(n: int, K: int, params: RAPSParams) -> int:
    """Compute the batch size ``B`` required by the finite-sample analysis."""

    if n <= 0:
        raise ValueError("Number of variables n must be positive.")
    if K <= 0:
        raise ValueError("Domain size K must be positive.")

    log = math.log
    term_reward = 32.0 / (params.Delta**2) * log(8 * n * K * ((K + 1) ** n) / params.delta)
    term_dists = 8.0 / (params.eps**2) * log(8 * (n**2) * (K**2) * ((K + 1) ** n) / params.delta)
    return math.ceil(max(term_reward, term_dists))


def parent_is_in_descendants_of_X(
    pcm: PCM,
    X: str,
    parents_found: Set[str],
    K: int,
    B: int,
    Delta: float,
) -> bool:
    """Test if the remaining reward parent lies in ``D(X)``.

    The test compares reward means between the baseline do(\\hat{P}=z) and the
    intervention do(X=x, \\hat{P}=z) and checks for a difference larger than
    ``Delta / 2``.
    """

    R = pcm.reward_node
    for z in all_assignments(parents_found, K):
        base_samples = pcm.intervene(do=z, B=B)
        Rbar_base = empirical_mean_R(base_samples, R)
        for x in range(K):
            inter_samples = pcm.intervene(do={**z, X: x}, B=B)
            Rbar_int = empirical_mean_R(inter_samples, R)
            if abs(Rbar_base - Rbar_int) > Delta / 2.0:
                return True
    return False


def descendants_in_C(
    pcm: PCM,
    X: str,
    C: Set[str],
    parents_found: Set[str],
    K: int,
    B: int,
    eps: float,
) -> Set[str]:
    """Estimate the descendants of ``X`` within the candidate set ``C``."""

    descendants: Set[str] = set()
    for z in all_assignments(parents_found, K):
        base_samples = pcm.intervene(do=z, B=B)
        base_marg = {Y: empirical_marginal(base_samples, Y, K) for Y in C}
        for x in range(K):
            inter_samples = pcm.intervene(do={**z, X: x}, B=B)
            inter_marg = {Y: empirical_marginal(inter_samples, Y, K) for Y in C}
            for Y in C:
                if l1_dist(base_marg[Y], inter_marg[Y]) > eps / 2.0:
                    descendants.add(Y)
    return descendants


def RAPS_single_parent_conceptual(
    pcm: PCM,
    V: Iterable[str],
    params: RAPSParams,
    parents_found: Optional[Set[str]] = None,
) -> Optional[str]:
    """Conceptual single-parent RAPS (Algorithm 1).

    This version uses the finite-sample tests for determining descendants and
    reward ancestry.  It is included primarily for completeness and mirrors the
    recursive routine described in the pseudo-code.  Because it relies on
    repeated sampling, it should not be used in performance-critical settings.
    """

    candidate_set: Set[str] = set(V)
    fixed_parents: Set[str] = set() if parents_found is None else set(parents_found)

    B = compute_budget(len(pcm.V), pcm.K, params)

    def rec(candidates: Set[str]) -> Optional[str]:
        if not candidates:
            return None
        X = random.choice(list(candidates))
        if parent_is_in_descendants_of_X(pcm, X, fixed_parents, pcm.K, B, params.Delta):
            descendants = descendants_in_C(
                pcm, X, candidates, fixed_parents, pcm.K, B, params.eps
            )
            P_hat = rec(descendants - {X})
            return X if P_hat is None else P_hat
        descendants = descendants_in_C(pcm, X, candidates, fixed_parents, pcm.K, B, params.eps)
        return rec(candidates - descendants)

    return rec(candidate_set)


@dataclass
class RAPSState:
    """Internal state tracked by the finite-sample RAPS algorithm."""

    parents_found: Set[str] = field(default_factory=set)
    candidates: Set[str] = field(default_factory=set)
    banned: Set[str] = field(default_factory=set)
    last_candidate_parent: Optional[str] = None
    last_descendants: Set[str] = field(default_factory=set)


def raps_full(pcm: PCM, params: RAPSParams) -> Set[str]:
    """Finite-sample RAPS implementation (Algorithm 4)."""

    n = len(pcm.V)
    K = pcm.K
    R = pcm.reward_node

    state = RAPSState(candidates=set(pcm.V))

    B = compute_budget(n, K, params)
    obs_samples = pcm.observe(B)
    Rbar_obs = empirical_mean_R(obs_samples, R)
    _ = {X: empirical_marginal(obs_samples, X, K) for X in pcm.V}
    _ = Rbar_obs  # silence unused variable warning

    while state.candidates:
        X = random.choice(list(state.candidates))

        D_union: Set[str] = set()
        reward_is_desc = False

        scope = set(state.candidates)
        scope.add(X)

        for z in all_assignments(state.parents_found, K):
            base_samples = pcm.intervene(do=z, B=B)
            Rbar_base = empirical_mean_R(base_samples, R)
            marg_base = {Y: empirical_marginal(base_samples, Y, K) for Y in scope}

            for x in range(K):
                inter_samples = pcm.intervene(do={**z, X: x}, B=B)
                Rbar_int = empirical_mean_R(inter_samples, R)
                marg_int = {Y: empirical_marginal(inter_samples, Y, K) for Y in scope}

                for Y in scope:
                    if l1_dist(marg_base[Y], marg_int[Y]) > params.eps / 2.0:
                        D_union.add(Y)

                if abs(Rbar_base - Rbar_int) > params.Delta / 2.0:
                    reward_is_desc = True

        if reward_is_desc:
            state.candidates = D_union - {X}
            state.last_candidate_parent = X
            state.last_descendants = set(D_union)
        else:
            state.candidates -= D_union
            state.banned |= D_union

        if not state.candidates and state.last_candidate_parent is not None:
            state.parents_found.add(state.last_candidate_parent)
            state.banned |= state.last_descendants
            state.candidates = set(pcm.V) - state.banned
            state.last_candidate_parent = None
            state.last_descendants = set()

    return state.parents_found


def RAPS_multi_parent(pcm: PCM, params: RAPSParams) -> Set[str]:
    """Convenience wrapper that repeatedly applies the conceptual RAPS routine."""

    parents: Set[str] = set()
    banned: Set[str] = set()

    while True:
        remaining = [node for node in pcm.V if node not in banned]
        parent = RAPS_single_parent_conceptual(pcm, remaining, params, parents)
        if parent is None:
            break
        parents.add(parent)

        descendants = descendants_in_C(
            pcm,
            parent,
            set(remaining),
            parents,
            pcm.K,
            compute_budget(len(pcm.V), pcm.K, params),
            params.eps,
        )
        banned |= descendants

    return parents


class UCBProtocol(Protocol):
    """Protocol describing the interface expected from a UCB implementation."""

    def select(self) -> int:
        ...

    def update(self, reward: float) -> None:
        ...


class RAPSPlusUCB:
    """Bandit wrapper that combines RAPS parent discovery with a UCB policy."""

    def __init__(
        self,
        pcm: PCM,
        params: RAPSParams,
        ucb_factory: Callable[[int], UCBProtocol],
    ) -> None: 
        self.pcm = pcm
        self.params = params
        self.ucb_factory = ucb_factory
        self.parents: Optional[Set[str]] = None
        self.ucb: Optional[UCBProtocol] = None
        self._arms: List[Dict[str, int]] = []

    def initialize(self) -> None:
        if self.parents is None:
            self.parents = raps_full(self.pcm, self.params)
            self._arms = list(all_assignments(self.parents, self.pcm.K))
            self.ucb = self.ucb_factory(len(self._arms))

    def select_arm(self) -> Dict[str, int]:
        if self.ucb is None:
            self.initialize()
        assert self.ucb is not None
        index = self.ucb.select()
        return self._arms[index]

    def step(self) -> float:
        arm = self.select_arm()
        samples = self.pcm.intervene(do=arm, B=1)
        reward = float(samples[0][self.pcm.reward_node])
        assert self.ucb is not None
        self.ucb.update(reward)
        return reward
    
    __all__ = [
    # LLR16
    "m_of_q",
    "parallel_bandit_algorithm",
    "compute_Q",
    "compute_m_eta",
    "choose_truncation_B",
    "general_causal_bandit_algorithm",
    "approx_minimize_m_eta",
    # Lee & Bareinboim 2018
    "MISs",
    "POMISs",
    "MUCT",
    "IB",
    "MUCT_IB",
    "SCM_to_bandit_machine",
    "arms_of",
    "KL",
    "sup_KL",
    "KL_UCB_run",
    "Thompson_run",
    "rand_argmax",
    # RAPS primitives
    "PCM",
    "empirical_mean_R",
    "empirical_marginal",
    "l1_dist",
    "all_assignments",
    "RAPSParams",
    "compute_budget",
    "parent_is_in_descendants_of_X",
    "descendants_in_C",
    "RAPS_single_parent_conceptual",
    "raps_full",
    "RAPS_multi_parent",
    "RAPSPlusUCB",
]
