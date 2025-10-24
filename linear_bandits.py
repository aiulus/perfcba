from __future__ import annotations

"""Structured (feature-based) bandit policies."""

from typing import Dict, Optional, Any

import numpy as np

from Algorithm import BasePolicy, History


class LinUCB(BasePolicy):
    """OFUL-style linear UCB with known feature matrix."""

    name = "lin-ucb"

    def __init__(self, features: np.ndarray, *, alpha: float = 1.0, lam: float = 1.0) -> None:
        X = np.asarray(features, dtype=float)
        if X.ndim != 2:
            raise ValueError("features must be a 2-D array")
        if lam <= 0:
            raise ValueError("lam must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.X = X
        self.alpha = float(alpha)
        self.lam = float(lam)

        self.n_arms = int(X.shape[0])
        self.d = int(X.shape[1])
        self.A = self.lam * np.eye(self.d)
        self.b = np.zeros(self.d)

    def reset(self, n_arms: int, horizon: Optional[int] = None, **_: Any) -> None:
        del horizon
        if int(n_arms) != self.n_arms:
            raise ValueError("Number of arms does not match feature matrix")
        self.A = self.lam * np.eye(self.d)
        self.b = np.zeros(self.d)

    def choose(self, t: int, history: History) -> int:
        del t, history
        A_inv = np.linalg.inv(self.A)
        theta_hat = A_inv @ self.b
        pred = self.X @ theta_hat
        rad = np.sqrt(np.sum((self.X @ A_inv) * self.X, axis=1))
        indices = pred + self.alpha * rad
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
        if info is None or "x" not in info:
            raise ValueError("LinUCB requires feature vector 'x' in info")
        context = np.asarray(info["x"], dtype=float)
        if context.shape != (self.d,):
            context = context.reshape(self.d)
        self.A += np.outer(context, context)
        self.b += context * float(x)

    def get_params(self) -> dict:
        return {"alpha": self.alpha, "lambda": self.lam}


class LinThompsonSampling(BasePolicy):
    """Thompson Sampling with a Bayesian linear model."""

    name = "lin-ts"

    def __init__(
        self,
        features: np.ndarray,
        *,
        sigma2: float = 1.0,
        lam: float = 1.0,
    ) -> None:
        X = np.asarray(features, dtype=float)
        if X.ndim != 2:
            raise ValueError("features must be a 2-D array")
        if sigma2 <= 0 or lam <= 0:
            raise ValueError("sigma2 and lam must be positive")
        self.X = X
        self.n_arms = int(X.shape[0])
        self.d = int(X.shape[1])
        self.sigma2 = float(sigma2)
        self.lam = float(lam)

        self.A = self.lam * np.eye(self.d)
        self.b = np.zeros(self.d)

    def reset(self, n_arms: int, horizon: Optional[int] = None, **_: Any) -> None:
        del horizon
        if int(n_arms) != self.n_arms:
            raise ValueError("Number of arms does not match feature matrix")
        self.A = self.lam * np.eye(self.d)
        self.b = np.zeros(self.d)

    def choose(self, t: int, history: History) -> int:
        del t, history
        A_inv = np.linalg.inv(self.A)
        cov = self.sigma2 * A_inv
        mean = A_inv @ self.b
        theta = np.random.multivariate_normal(mean, cov)
        values = self.X @ theta
        return int(np.argmax(values))

    def update(
        self,
        t: int,
        a: int,
        x: float,
        *,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        del t
        if info is None or "x" not in info:
            raise ValueError("LinTS requires feature vector 'x' in info")
        context = np.asarray(info["x"], dtype=float)
        if context.shape != (self.d,):
            context = context.reshape(self.d)
        self.A += np.outer(context, context)
        self.b += context * float(x)

    def get_params(self) -> dict:
        return {"sigma2": self.sigma2, "lambda": self.lam}