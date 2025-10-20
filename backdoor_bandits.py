from __future__ import annotations

"""Causal bandit policies using back-door adjustment."""

from typing import Dict, List, Optional, Any

import numpy as np

from Algorithm import BasePolicy, History
from estimators import (
    MultinomialLogisticRegression,
    RidgeOutcomeRegressor,
    dr_crossfit,
    ess,
)


def _info_to_array(info: Optional[Dict[str, Any]]) -> np.ndarray:
    if info is None:
        raise ValueError("Causal policies require observed covariates in info")
    if "x" in info:
        arr = np.asarray(info["x"], dtype=float)
    elif "covariates" in info:
        arr = np.asarray(info["covariates"], dtype=float)
    else:
        keys = sorted(info.keys())
        values = [info[k] for k in keys]
        arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.astype(float)


class CausalUCBBackdoor(BasePolicy):
    """UCB-style policy using doubly-robust estimates under back-door adjustment."""

    name = "causal-ucb-dr"

    def __init__(
        self,
        n_arms: int,
        *,
        outcome_model: RidgeOutcomeRegressor,
        propensity_model: MultinomialLogisticRegression,
        refit_every: int = 50,
        min_samples: int = 20,
        clip: float = 10.0,
        alpha: float = 1.0,
    ) -> None:
        if n_arms <= 1:
            raise ValueError("n_arms must be at least 2")
        self.n_arms = int(n_arms)
        self._outcome_proto = outcome_model
        self._propensity_proto = propensity_model
        self.refit_every = int(refit_every)
        self.min_samples = int(min_samples)
        self.clip = float(clip)
        self.alpha = float(alpha)

        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self.X: List[np.ndarray] = []
        self.a: List[int] = []
        self.y: List[float] = []
        self.mu_hat = np.zeros(self.n_arms, dtype=float)
        self.var_hat = np.ones(self.n_arms, dtype=float)
        self.ess = np.zeros(self.n_arms, dtype=float)
        self._last_refit = 0
        self._t = 0

    def reset(self, n_arms: int, horizon: Optional[int] = None, **_: Any) -> None:
        del horizon
        if int(n_arms) != self.n_arms:
            raise ValueError("n_arms mismatch for causal policy")
        self._reset_buffers()

    def _refit(self) -> None:
        if len(self.y) < self.min_samples:
            return
        X = np.vstack(self.X)
        a = np.asarray(self.a, dtype=int)
        y = np.asarray(self.y, dtype=float)
        for arm in range(self.n_arms):
            mu, weights, dr_vals = dr_crossfit(
                y,
                a,
                X,
                arm,
                self._outcome_proto.clone(),
                self._propensity_proto.clone(),
                clip=self.clip,
            )
            self.mu_hat[arm] = mu
            if dr_vals.size > 1:
                self.var_hat[arm] = float(np.var(dr_vals, ddof=1))
            else:
                self.var_hat[arm] = 1.0
            self.ess[arm] = ess(weights)
        self._last_refit = self._t

    def choose(self, t: int, history: History) -> int:
        del history
        self._t = t
        if t <= self.n_arms:
            return t - 1
        if (t - self._last_refit) >= self.refit_every:
            self._refit()
        if len(self.y) < self.min_samples:
            return np.random.randint(self.n_arms)
        scale = np.sqrt(np.log(max(3, t))) / max(1, len(self.y))
        bonuses = self.alpha * np.sqrt(np.maximum(self.var_hat, 1e-9)) * scale
        indices = self.mu_hat + bonuses
        return int(np.argmax(indices))

    def update(
        self,
        t: int,
        a: int,
        x: float,
        *,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        del t
        covariates = _info_to_array(info)
        self.X.append(covariates)
        self.a.append(int(a))
        self.y.append(float(x))

    def diagnostics(self) -> Dict[str, Any]:
        return {"mu_hat": self.mu_hat.copy(), "var_hat": self.var_hat.copy(), "ess": self.ess.copy()}


class CausalThompsonBackdoor(CausalUCBBackdoor):
    """Thompson sampling variant using doubly-robust mean estimates."""

    name = "causal-ts-dr"

    def __init__(
        self,
        n_arms: int,
        *,
        outcome_model: RidgeOutcomeRegressor,
        propensity_model: MultinomialLogisticRegression,
        refit_every: int = 50,
        min_samples: int = 20,
        clip: float = 10.0,
        variance_scale: float = 1.0,
    ) -> None:
        super().__init__(
            n_arms,
            outcome_model=outcome_model,
            propensity_model=propensity_model,
            refit_every=refit_every,
            min_samples=min_samples,
            clip=clip,
            alpha=1.0,
        )
        self.variance_scale = float(variance_scale)

    def choose(self, t: int, history: History) -> int:
        del history
        self._t = t
        if t <= self.n_arms:
            return t - 1
        if (t - self._last_refit) >= self.refit_every:
            self._refit()
        if len(self.y) < self.min_samples:
            return np.random.randint(self.n_arms)
        scale = self.variance_scale * np.sqrt(np.maximum(self.var_hat, 1e-9)) / max(1, len(self.y))
        samples = np.random.normal(self.mu_hat, scale)
        return int(np.argmax(samples))