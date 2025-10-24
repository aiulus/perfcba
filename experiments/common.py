from __future__ import annotations

import json
import math
import os
import random
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

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


def _arm_estimates(history: History, bandit: AbstractBandit) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (counts, empirical_means, errors) for each arm, using NaN for never-pulled arms."""

    n_arms = bandit.n_arms
    counts = history.pulls.astype(float)
    sums = np.zeros(n_arms, dtype=float)
    for a, r in zip(history.actions, history.rewards):
        sums[a] += float(r)
    empirical = np.full(n_arms, np.nan, dtype=float)
    positive = counts > 0
    empirical[positive] = sums[positive] / counts[positive]
    true_means = np.array([bandit.mean(a) for a in range(n_arms)], dtype=float)
    errors = empirical - true_means
    return counts, empirical, errors


def run_policy_trials(
    *,
    bandit_factory: BanditFactory,
    policy_factory: Callable[[], "BasePolicy"],
    horizons: Sequence[int],
    seeds: Sequence[int],
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
        true_means: Optional[np.ndarray] = None

        for idx, seed in enumerate(seeds):
            random.seed(seed)
            np.random.seed(seed)
            bandit = bandit_factory()
            policy = policy_factory()
            run = Experiment(bandit, policy, RunConfig(T=horizon, seed=seed)).run()
            profiler = Profiler(run, bandit)
            curve = profiler.regret_curve()

            if regret_curves is None:
                regret_curves = np.zeros((n_runs, horizon), dtype=float)
            regret_curves[idx, :] = curve
            total_regrets.append(float(curve[-1]) if curve.size else 0.0)

            counts, _, errors = _arm_estimates(run, bandit)
            if pulls is None:
                pulls = np.zeros((n_runs, bandit.n_arms), dtype=float)
                abs_errors = np.zeros((n_runs, bandit.n_arms), dtype=float)
                true_means = np.array([bandit.mean(a) for a in range(bandit.n_arms)], dtype=float)
            pulls[idx, :] = counts
            abs_errors[idx, :] = np.abs(errors)

        if regret_curves is None or pulls is None or abs_errors is None or true_means is None:
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

        results[horizon] = {
            "mean_regret": mean_regret,
            "ci_regret": ci_regret,
            "mean_curve": mean_curve,
            "ci_curve": ci_curve,
            "mean_pulls": mean_pulls,
            "ci_pulls": ci_pulls,
            "mean_abs_error": mean_abs_err,
            "ci_abs_error": ci_abs_err,
            "true_means": true_means,
        }

    return results


def save_json(path: str, payload: Mapping[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_json_default)


def _json_default(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
