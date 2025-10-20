from __future__ import annotations

"""Simple models for propensity score and outcome estimation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MultinomialLogisticRegression:
    n_classes: int
    learning_rate: float = 0.1
    l2: float = 1e-3
    max_iter: int = 500
    tol: float = 1e-6
    random_state: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_classes <= 1:
            raise ValueError("n_classes must be at least 2")
        if self.learning_rate <= 0 or self.max_iter <= 0:
            raise ValueError("learning_rate and max_iter must be positive")
        if self.l2 < 0:
            raise ValueError("l2 must be non-negative")
        self._coef: Optional[np.ndarray] = None

    def clone(self) -> "MultinomialLogisticRegression":
        return MultinomialLogisticRegression(
            n_classes=self.n_classes,
            learning_rate=self.learning_rate,
            l2=self.l2,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatched number of samples")
        n, d = X.shape
        Xb = np.column_stack([np.ones(n), X])
        rng = np.random.default_rng(self.random_state)
        W = rng.normal(scale=1e-2, size=(d + 1, self.n_classes))

        eye = np.eye(d + 1)
        eye[0, 0] = 0.0  # no penalty on intercept

        for _ in range(self.max_iter):
            logits = Xb @ W
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            Y = np.zeros_like(probs)
            Y[np.arange(n), y] = 1.0
            grad = Xb.T @ (Y - probs) / n - self.l2 * (eye @ W)

            step_norm = np.linalg.norm(grad)
            W += self.learning_rate * grad
            if step_norm < self.tol:
                break

        self._coef = W

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("Model must be fitted before calling predict_proba")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = X.shape[0]
        Xb = np.column_stack([np.ones(n), X])
        logits = Xb @ self._coef
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs


@dataclass
class RidgeOutcomeRegressor:
    l2: float = 1e-3

    def clone(self) -> "RidgeOutcomeRegressor":
        return RidgeOutcomeRegressor(l2=self.l2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatched number of samples")
        n = X.shape[0]
        Xb = np.column_stack([np.ones(n), X])
        d = Xb.shape[1]
        reg = self.l2 * np.eye(d)
        reg[0, 0] = 0.0
        gram = Xb.T @ Xb + reg
        rhs = Xb.T @ y
        try:
            self._coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            self._coef = np.linalg.pinv(gram) @ rhs

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_coef"):
            raise RuntimeError("Model must be fitted before calling predict")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = X.shape[0]
        Xb = np.column_stack([np.ones(n), X])
        return Xb @ self._coef