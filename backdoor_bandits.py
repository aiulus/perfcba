from __future__ import annotations
from typing import Dict, List, Optional, Any
import numpy as np
from .Algorithm import BasePolicy, History
from .estimators.models import MultinomialLogisticRegression, RidgeOutcomeRegressor
from .estimators.robust import dr_crossfit, ess

def _info_to_array(info: Optional[Dict[str, Any]]) -> np.ndarray:
    """Flatten info dict (e.g., {'Z': zvec, 'X': x}) into 1D covariate vector.

    Drops obvious post-treatment entries (e.g., the logged action ``T``) so the
    covariate vector only contains pre-treatment information.
    """
    if not info:
        return np.zeros(0, dtype=float)
    parts: List[np.ndarray] = []
    for k in sorted(info.keys()):
        if isinstance(k, str):
            key = k.strip().upper()
            # Skip treatment/action keys to avoid post-treatment conditioning.
            if key in {"T", "A", "ARM", "ACTION", "TREATMENT"} or key.startswith("DO("):
                continue
        v = info[k]
        arr = np.atleast_1d(np.asarray(v, dtype=float))
        parts.append(arr.ravel())
    return np.concatenate(parts) if parts else np.zeros(0, dtype=float)

class BackdoorUCB(BasePolicy):
    """
    UCB-style policy using back-door adjustment via DR estimates per arm.
    Re-fits propensity (multinomial logistic) and outcome (ridge) models on a schedule.
    """
    name = "backdoor-ucb"

    def __init__(self, n_arms: int, refit_every: int = 25, clip: Optional[float] = 20.0,
                 alpha: float = 1.0) -> None:
        self.n_arms = int(n_arms)
        self.refit_every = int(refit_every)
        self.clip = clip
        self.alpha = float(alpha)

    def reset(self, n_arms: int, horizon: int) -> None:
        del horizon
        if n_arms != self.n_arms:
            self.n_arms = int(n_arms)
        self.X: List[np.ndarray] = []
        self.a: List[int] = []
        self.y: List[float] = []
        self._last_refit = 0
        self.mu_hat = np.zeros(self.n_arms, dtype=float)
        self.var_hat = np.ones(self.n_arms, dtype=float)
        self._ess = np.zeros(self.n_arms, dtype=float)
        self._min_prop = np.full(self.n_arms, np.nan, dtype=float)

    def _refit(self) -> None:
        if len(self.y) < self.n_arms:
            self._ess.fill(0.0)
            self._min_prop.fill(np.nan)
            return
        X = np.vstack(self.X) if self.X else np.zeros((0, 0), dtype=float)
        a = np.asarray(self.a, dtype=int)
        y = np.asarray(self.y, dtype=float)
        prop = MultinomialLogisticRegression(n_classes=self.n_arms)
        out = RidgeOutcomeRegressor(l2=1e-2)
        for arm in range(self.n_arms):
            mean_arm, weights, dr_vals = dr_crossfit(
                y,
                a,
                X,
                arm,
                out,
                prop,
                K=2,
                clip=self.clip,
            )
            self.mu_hat[arm] = mean_arm
            # conservative variance proxy
            self.var_hat[arm] = float(np.var(np.asarray(dr_vals), ddof=1)) if len(dr_vals) > 1 else 1.0
            ess_val = ess(weights)
            if ess_val < 0:
                ess_val = 0.0
            self._ess[arm] = float(ess_val)
            positive = weights[weights > 0]
            if positive.size:
                props = 1.0 / positive
                self._min_prop[arm] = float(np.min(props))
            else:
                self._min_prop[arm] = np.nan
        self._last_refit = self._t

    def choose(self, t: int, history: History) -> int:
        del history
        self._t = t
        if t <= self.n_arms:
            return t - 1
        if (t - self._last_refit) >= self.refit_every:
            self._refit()
        n = max(1, len(self.y))
        ucb = self.mu_hat + self.alpha * np.sqrt(self.var_hat / n)
        return int(np.argmax(ucb))

    def update(self, t: int, a: int, x: float, *, info: Optional[Dict[str, Any]] = None) -> None:
        del t
        self.X.append(_info_to_array(info))
        self.a.append(int(a))
        self.y.append(float(x))

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "mu_hat": self.mu_hat.tolist(),
            "var_hat": self.var_hat.tolist(),
            "n": len(self.y),
            "ess": self._ess.tolist(),
            "min_prop": self._min_prop.tolist(),
        }

class BackdoorTS(BasePolicy):
    """Thompson sampling over DR means with variance scaling."""
    name = "backdoor-ts"

    def __init__(self, n_arms: int, refit_every: int = 25, clip: Optional[float] = 20.0,
                 variance_scale: float = 1.0, min_samples: int = 10) -> None:
        self.n_arms = int(n_arms)
        self.refit_every = int(refit_every)
        self.clip = clip
        self.variance_scale = float(variance_scale)
        self.min_samples = int(min_samples)

    def reset(self, n_arms: int, horizon: int) -> None:
        del horizon
        if n_arms != self.n_arms:
            self.n_arms = int(n_arms)
        self.X: List[np.ndarray] = []
        self.a: List[int] = []
        self.y: List[float] = []
        self._last_refit = 0
        self.mu_hat = np.zeros(self.n_arms, dtype=float)
        self.var_hat = np.ones(self.n_arms, dtype=float)
        self._ess = np.zeros(self.n_arms, dtype=float)
        self._min_prop = np.full(self.n_arms, np.nan, dtype=float)

    def _refit(self) -> None:
        if len(self.y) < self.n_arms:
            self._ess.fill(0.0)
            self._min_prop.fill(np.nan)
            return
        X = np.vstack(self.X) if self.X else np.zeros((0, 0), dtype=float)
        a = np.asarray(self.a, dtype=int)
        y = np.asarray(self.y, dtype=float)
        prop = MultinomialLogisticRegression(n_classes=self.n_arms)
        out = RidgeOutcomeRegressor(l2=1e-2)
        for arm in range(self.n_arms):
            mean_arm, weights, dr_vals = dr_crossfit(
                y,
                a,
                X,
                arm,
                out,
                prop,
                K=2,
                clip=self.clip,
            )
            self.mu_hat[arm] = mean_arm
            self.var_hat[arm] = float(np.var(np.asarray(dr_vals), ddof=1)) if len(dr_vals) > 1 else 1.0
            ess_val = ess(weights)
            if ess_val < 0:
                ess_val = 0.0
            self._ess[arm] = float(ess_val)
            positive = weights[weights > 0]
            if positive.size:
                props = 1.0 / positive
                self._min_prop[arm] = float(np.min(props))
            else:
                self._min_prop[arm] = np.nan
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
        n = max(1, len(self.y))
        scale = self.variance_scale * np.sqrt(np.maximum(self.var_hat, 1e-9)) / n
        samples = np.random.normal(self.mu_hat, scale)
        return int(np.argmax(samples))

    def update(self, t: int, a: int, x: float, *, info: Optional[Dict[str, Any]] = None) -> None:
        del t
        self.X.append(_info_to_array(info))
        self.a.append(int(a))
        self.y.append(float(x))

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "mu_hat": self.mu_hat.tolist(),
            "var_hat": self.var_hat.tolist(),
            "n": len(self.y),
            "ess": self._ess.tolist(),
            "min_prop": self._min_prop.tolist(),
        }
