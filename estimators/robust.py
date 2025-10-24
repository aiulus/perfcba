from __future__ import annotations

"""Doubly robust and importance weighting utilities for causal bandits."""

from typing import Callable, Tuple

import numpy as np


def ess(weights: np.ndarray) -> float:
    """Effective sample size of a weight vector."""

    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return 0.0
    numerator = np.sum(w) ** 2
    denominator = np.sum(w ** 2)
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def ipw_estimate(
    y: np.ndarray,
    a: np.ndarray,
    x: np.ndarray,
    arm: int,
    propensity: Callable[[np.ndarray], np.ndarray],
    *,
    clip: float | None = 10.0,
) -> Tuple[float, np.ndarray]:
    """Compute an inverse-propensity weighted mean for ``arm``."""

    del x
    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=int)
    probs = np.clip(propensity(arm), 1e-8, 1.0)
    weights = (a == arm).astype(float) / probs
    if clip is not None:
        weights = np.minimum(weights, float(clip))
    normaliser = np.sum(weights)
    if normaliser <= 0:
        return 0.0, weights
    return float(np.sum(weights * y) / normaliser), weights


def dr_crossfit(
    y: np.ndarray,
    a: np.ndarray,
    x: np.ndarray,
    arm: int,
    outcome_model,
    propensity_model,
    *,
    K: int = 2,
    clip: float | None = 10.0,
    random_state: int = 0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Cross-fitted doubly-robust estimator for ``E[Y | do(A = arm)]``."""

    y = np.asarray(y, dtype=float)
    a = np.asarray(a, dtype=int)
    X = np.asarray(x, dtype=float)

    if y.shape[0] != a.shape[0] or y.shape[0] != X.shape[0]:
        raise ValueError("Inputs must have matching number of samples")
    n = y.shape[0]
    if n == 0:
        return 0.0, np.zeros(0), np.zeros(0)

    n_arms = int(np.max(a)) + 1
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    folds = np.array_split(perm, max(1, int(K)))

    e_hat = np.zeros((n, n_arms), dtype=float)
    m_hat = np.zeros(n, dtype=float)

    for fold in folds:
        test_idx = np.asarray(fold, dtype=int)
        if test_idx.size == 0:
            continue
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        train_idx = np.nonzero(mask)[0]
        if train_idx.size == 0:
            continue

        prop = propensity_model.clone()
        prop.fit(X[train_idx], a[train_idx])
        e_hat[test_idx] = prop.predict_proba(X[test_idx])

        outcome = outcome_model.clone()
        features = np.column_stack([a[train_idx], X[train_idx]])
        outcome.fit(features, y[train_idx])
        counterfactual = np.column_stack([np.full(test_idx.size, arm), X[test_idx]])
        m_hat[test_idx] = outcome.predict(counterfactual)

    denom = np.clip(e_hat[:, arm], 1e-8, 1.0)
    weights = (a == arm).astype(float) / denom
    if clip is not None:
        weights = np.minimum(weights, float(clip))
    dr_values = m_hat + weights * (y - m_hat)
    return float(np.mean(dr_values)), weights, dr_values