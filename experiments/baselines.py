"""Baseline estimators for tau-study visualizations."""

from __future__ import annotations

import math
from itertools import combinations, product
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .causal_envs import CausalBanditConfig, build_random_scm_with_gaps


def _config_from_record(record: Mapping[str, Any]) -> Optional[CausalBanditConfig]:
    """Reconstruct a configuration object from a stored result row."""

    try:
        n = int(record["node_count"])
        ell = int(record["alphabet"])
        k = int(record["parent_count"])
        m = int(record["intervention_size"])
    except (KeyError, TypeError, ValueError):
        return None

    edge_prob = float(record.get("graph_density", record.get("knob_value", 0.0)))
    edge_prob_covariates = float(record.get("edge_prob_covariates", edge_prob))
    edge_prob_to_reward = float(record.get("edge_prob_to_reward", edge_prob))

    return CausalBanditConfig(
        n=n,
        ell=ell,
        k=k,
        m=m,
        edge_prob=edge_prob,
        edge_prob_covariates=edge_prob_covariates,
        edge_prob_to_reward=edge_prob_to_reward,
        hard_margin=float(record.get("hard_margin", 0.0)),
        reward_logit_scale=float(record.get("arm_variance", 1.0)),
        arm_heterogeneity_mode=str(record.get("arm_heterogeneity_mode", "uniform")),
        sparse_fraction=float(record.get("sparse_fraction", 0.1)),
        sparse_separation=float(record.get("sparse_separation", 0.3)),
        cluster_count=int(record.get("cluster_count", 3)),
        gap_enforcement_mode=str(record.get("gap_enforcement_mode", "soft")),
    )


def _compute_optimal_mean(instance, rng: np.random.Generator, mc_samples: int) -> float:
    """Copy of run_tau_study.compute_optimal_mean to avoid circular imports."""

    ell = instance.config.ell
    parent_indices = instance.parent_indices()
    best = 0.0
    max_size = min(instance.config.m, len(parent_indices))
    for size in range(0, max_size + 1):
        for subset in combinations(parent_indices, size):
            for assignment in product(range(ell), repeat=size):
                mean = instance.estimate_subset_mean(subset, assignment, rng, n_mc=mc_samples)
                best = max(best, mean)
    return best


def _baseline_for_metric(
    metric: str, optimal_mean: float, observational_mean: float, horizon: float
) -> Optional[float]:
    """Convert an observational reward mean into the requested metric's units."""

    gap = optimal_mean - observational_mean
    if metric == "cumulative_regret":
        return gap * horizon
    if metric == "simple_regret":
        return max(0.0, gap)
    return None


def estimate_observational_baseline(
    records: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    mc_samples: int = 2048,
) -> Optional[float]:
    """
    Estimate the metric value when the bandit never intervenes (purely observational).

    The routine rebuilds SCMs from the stored configs (one per seed/knob pair),
    computes the observational reward mean, converts it to the requested metric,
    and averages across the unique environments.
    """

    cache: Dict[float, Tuple[float, float]] = {}
    baselines = []
    for record in records:
        try:
            seed = int(record["seed"])
            knob_value = float(record.get("knob_value", record.get("graph_density")))
            horizon = float(record["horizon"])
        except (KeyError, TypeError, ValueError):
            continue

        key = knob_value
        if key not in cache:
            cfg = _config_from_record(record)
            if cfg is None:
                continue
            rng = np.random.default_rng(seed)
            instance, _ = build_random_scm_with_gaps(cfg, rng=rng, enforce_targets=False)
            obs_mean = instance.estimate_subset_mean([], [], np.random.default_rng(seed + 1), n_mc=mc_samples)
            opt_mean = _compute_optimal_mean(instance, np.random.default_rng(seed + 2), mc_samples=mc_samples)
            cache[key] = (opt_mean, obs_mean)

        opt_mean, obs_mean = cache[key]
        baseline = _baseline_for_metric(metric, opt_mean, obs_mean, horizon)
        if baseline is not None and math.isfinite(baseline):
            baselines.append(float(baseline))

    if not baselines:
        return None
    return float(np.mean(baselines))
