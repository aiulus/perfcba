from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .. import metrics
from ..Experiment import Experiment, RunConfig
from ..Profiler import Profiler
from ..Algorithm import History
from ..Bandit import AbstractBandit


PolFactory = Callable[[], "BasePolicy"]
BanditFactory = Callable[[], AbstractBandit]


def ensure_dir(path: str) -> None:
    """Create *path* if it is missing."""

    os.makedirs(path, exist_ok=True)


def mean_confidence_interval(values: Sequence[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Return (mean, half-width) for a normal-based CI (default 95%)."""

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0
    if not math.isclose(confidence, 0.95):
        raise ValueError("Only 95% confidence intervals are supported.")
    z = 1.96
    se = np.std(arr, ddof=1) / math.sqrt(arr.size)
    return mean, float(z * se)


def _arm_estimates(history: History, bandit: AbstractBandit) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (counts, empirical_means, errors, true_means) for each arm."""

    n_arms = bandit.n_arms
    counts = history.pulls.astype(float)
    sums = np.zeros(n_arms, dtype=float)
    true_sums = np.zeros(n_arms, dtype=float)
    horizon = int(history.T)
    if horizon > 0:
        means_path = np.stack(
            [np.asarray(bandit.means_at(t), dtype=float) for t in range(1, horizon + 1)],
            axis=0,
        )
        time_avg_means = np.mean(means_path, axis=0)
    else:
        time_avg_means = np.asarray(bandit.means_at(1), dtype=float)

    for idx, (a, r) in enumerate(zip(history.actions, history.rewards), start=1):
        sums[a] += float(r)
        means_t = np.asarray(bandit.means_at(idx), dtype=float)
        if means_t.shape[0] != n_arms:
            raise ValueError("means_at must return an array with one entry per arm")
        true_sums[a] += float(means_t[a])
    empirical = np.full(n_arms, np.nan, dtype=float)
    positive = counts > 0
    empirical[positive] = sums[positive] / counts[positive]
    true_means = np.full(n_arms, np.nan, dtype=float)
    if np.any(positive):
        true_means[positive] = true_sums[positive] / counts[positive]
    if np.any(~positive):
        true_means[~positive] = time_avg_means[~positive]
    errors = empirical - true_means
    return counts, empirical, errors, true_means


def run_policy_trials(
    *,
    bandit_factory: BanditFactory,
    policy_factory: Callable[[], "BasePolicy"],
    horizons: Sequence[int],
    seeds: Sequence[int],
    progress: bool = False,
    epsilon: float = 1.0,
) -> Dict[int, Dict[str, np.ndarray | float]]:
    """
    Execute ``policy_factory`` against ``bandit_factory`` across ``horizons`` and ``seeds``.

    Returns
    -------
    dict
        Mapping horizon -> aggregated metrics including regret curves, pulls, and
        empirical arm mean errors.  All statistics are numpy arrays keyed as:

        - ``mean_regret``: scalar
        - ``ci_regret``: scalar
        - ``mean_curve``: shape (horizon,)
        - ``ci_curve``: shape (horizon,)
        - ``mean_pulls``: shape (n_arms,)
        - ``ci_pulls``: shape (n_arms,)
        - ``mean_abs_error``: shape (n_arms,)
        - ``ci_abs_error``: shape (n_arms,)
        - ``true_means``: shape (n_arms,)
        - ``mean_simple_regret``: scalar
        - ``ci_simple_regret``: scalar
        - ``mean_aurc``: scalar (area under the cumulative regret curve)
        - ``ci_aurc``: scalar
        - ``mean_time_to_epsilon``: scalar (first t with average regret <= ``epsilon``; NaN if never achieved)
        - ``ci_time_to_epsilon``: scalar
    """

    horizons = list(horizons)
    seeds = list(seeds)
    results: Dict[int, Dict[str, np.ndarray | float]] = {}

    for horizon in horizons:
        n_runs = len(seeds)
        regret_curves: Optional[np.ndarray] = None
        pulls: Optional[np.ndarray] = None
        abs_errors: Optional[np.ndarray] = None
        total_regrets: List[float] = []
        aurc_vals = np.zeros(n_runs, dtype=float)
        time_to_eps_vals = np.full(n_runs, np.nan, dtype=float)
        simple_regrets = np.zeros(n_runs, dtype=float)
        true_means_runs: List[np.ndarray] = []

        for idx, seed in enumerate(seeds):
            random.seed(seed)
            np.random.seed(seed)
            bandit = bandit_factory()
            policy = policy_factory()
            run = Experiment(bandit, policy, RunConfig(T=horizon, seed=seed)).run()
            profiler = Profiler(run, bandit)
            curve = profiler.regret_curve()
            aurc_vals[idx] = metrics.area_under_regret_curve(curve)
            t_eps = metrics.time_to_epsilon_optimal(curve, epsilon=epsilon)
            time_to_eps_vals[idx] = float(t_eps) if t_eps is not None else math.nan
            simple_regrets[idx] = profiler.simple_regret()

            if regret_curves is None:
                regret_curves = np.zeros((n_runs, horizon), dtype=float)
            regret_curves[idx, :] = curve
            total_regrets.append(float(curve[-1]) if curve.size else 0.0)

            counts, _, errors, arm_true_means = _arm_estimates(run, bandit)
            if pulls is None:
                pulls = np.zeros((n_runs, bandit.n_arms), dtype=float)
                abs_errors = np.zeros((n_runs, bandit.n_arms), dtype=float)
            pulls[idx, :] = counts
            abs_errors[idx, :] = np.abs(errors)
            true_means_runs.append(arm_true_means)

        if regret_curves is None or pulls is None or abs_errors is None:
            raise RuntimeError("No runs executed. Check seeds/horizons inputs.")

        mean_curve = regret_curves.mean(axis=0)
        ci_curve = 1.96 * regret_curves.std(axis=0, ddof=1) / math.sqrt(n_runs) if n_runs > 1 else np.zeros_like(mean_curve)
        mean_pulls = pulls.mean(axis=0)
        ci_pulls = 1.96 * pulls.std(axis=0, ddof=1) / math.sqrt(n_runs) if n_runs > 1 else np.zeros_like(mean_pulls)

        mean_abs_err = np.nanmean(abs_errors, axis=0)
        ci_abs_err = np.zeros_like(mean_abs_err)
        if n_runs > 1:
            # Compute CI ignoring NaNs by operating column-wise
            for j in range(abs_errors.shape[1]):
                col = abs_errors[:, j]
                mask = ~np.isnan(col)
                if np.count_nonzero(mask) <= 1:
                    ci_abs_err[j] = 0.0
                else:
                    se = np.std(col[mask], ddof=1) / math.sqrt(np.count_nonzero(mask))
                    ci_abs_err[j] = 1.96 * se

        mean_regret, ci_regret = mean_confidence_interval(total_regrets)
        mean_simple_regret, ci_simple_regret = mean_confidence_interval(simple_regrets.tolist())
        mean_aurc, ci_aurc = mean_confidence_interval(aurc_vals.tolist())
        if true_means_runs:
            stacked = np.stack(true_means_runs, axis=0)
            avg_true_means = np.nanmean(stacked, axis=0)
        else:
            avg_true_means = np.array([], dtype=float)
        mask_time = ~np.isnan(time_to_eps_vals)
        if np.count_nonzero(mask_time):
            mean_time_to_eps, ci_time_to_eps = mean_confidence_interval(time_to_eps_vals[mask_time].tolist())
        else:
            mean_time_to_eps, ci_time_to_eps = math.nan, 0.0

        results[horizon] = {
            "mean_regret": mean_regret,
            "ci_regret": ci_regret,
            "mean_curve": mean_curve,
            "ci_curve": ci_curve,
            "mean_pulls": mean_pulls,
            "ci_pulls": ci_pulls,
            "mean_abs_error": mean_abs_err,
            "ci_abs_error": ci_abs_err,
            "true_means": avg_true_means,
            "mean_simple_regret": mean_simple_regret,
            "ci_simple_regret": ci_simple_regret,
            "mean_aurc": mean_aurc,
            "ci_aurc": ci_aurc,
            "mean_time_to_epsilon": mean_time_to_eps,
            "ci_time_to_epsilon": ci_time_to_eps,
        }

    return results


def sweep_1d(
    *,
    property_name: str,
    values: Sequence[Any],
    bandit_factory_for: Callable[[Any], BanditFactory],
    policy_builders: Mapping[str, Callable[[Any], PolFactory]],
    horizon: int,
    seeds: Sequence[int],
    label_fn: Optional[Callable[[Any], str]] = None,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Any]]:
    """Run a 1-D sweep returning metrics per policy and property value."""

    if label_fn is None:
        label_fn = lambda v: str(v)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {
        name: {} for name in policy_builders.keys()
    }
    value_meta: Dict[str, Any] = {"property": property_name, "values": {}}

    for value in values:
        label = label_fn(value)
        value_meta["values"][label] = value
        bandit_factory = bandit_factory_for(value)
        for name, builder in policy_builders.items():
            policy_factory = builder(value)
            metrics = run_policy_trials(
                bandit_factory=bandit_factory,
                policy_factory=policy_factory,
                horizons=[horizon],
                seeds=seeds,
            )[horizon]
            summary[name][label] = {
                "mean_regret": float(metrics["mean_regret"]),
                "ci_regret": float(metrics["ci_regret"]),
                "mean_simple_regret": float(metrics["mean_simple_regret"]),
                "ci_simple_regret": float(metrics["ci_simple_regret"]),
                "mean_aurc": float(metrics["mean_aurc"]),
                "ci_aurc": float(metrics["ci_aurc"]),
                "mean_time_to_epsilon": float(metrics["mean_time_to_epsilon"]),
                "ci_time_to_epsilon": float(metrics["ci_time_to_epsilon"]),
            }
    return summary, value_meta


def sweep_2d(
    *,
    property_x: str,
    values_x: Sequence[Any],
    property_y: str,
    values_y: Sequence[Any],
    bandit_factory_for: Callable[[Any, Any], BanditFactory],
    policy_builders: Mapping[str, Callable[[Any, Any], PolFactory]],
    horizon: int,
    seeds: Sequence[int],
    label_x: Optional[Callable[[Any], str]] = None,
    label_y: Optional[Callable[[Any], str]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Any]]]:
    """Run a 2-D grid sweep returning mean regret tensors per policy."""

    if label_x is None:
        label_x = lambda v: str(v)
    if label_y is None:
        label_y = lambda v: str(v)

    tensors: Dict[str, np.ndarray] = {
        name: np.zeros((len(values_x), len(values_y)), dtype=float)
        for name in policy_builders.keys()
    }

    for ix, vx in enumerate(values_x):
        for iy, vy in enumerate(values_y):
            bandit_factory = bandit_factory_for(vx, vy)
            for name, builder in policy_builders.items():
                policy_factory = builder(vx, vy)
                metrics = run_policy_trials(
                    bandit_factory=bandit_factory,
                    policy_factory=policy_factory,
                    horizons=[horizon],
                    seeds=seeds,
                )[horizon]
                tensors[name][ix, iy] = float(metrics["mean_regret"])

    metadata = {
        "property_x": property_x,
        "values_x": [label_x(v) for v in values_x],
        "raw_x": list(values_x),
        "property_y": property_y,
        "values_y": [label_y(v) for v in values_y],
        "raw_y": list(values_y),
    }
    return tensors, metadata


def save_json(path: str, payload: Mapping[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_json_default)


def _json_default(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
