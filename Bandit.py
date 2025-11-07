# Bandit.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Callable, Optional, Dict, Any,  Iterable, List, Tuple
from .SCM import SCM, LinearGaussianSCM, Intervention
from .experiments.causal_envs import (
    CausalBanditInstance,
    InterventionArm,
    InterventionSpace,
    arm_to_intervention,
)

import numpy as np


class AbstractBandit(ABC):
    """Environment: maps chosen arm to reward sample; exposes true means for evaluation"""
    n_arms: int

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        ...

    @abstractmethod
    def sample(self, a: int, rng: np.random.Generator) -> float:
        ...

    @abstractmethod
    def mean(self, a: int) -> float:
        ...

    def means_at(self, t: int) -> np.ndarray:
        """Return expected rewards at round ``t`` (default: stationary)."""

        del t
        return np.array([self.mean(a) for a in range(self.n_arms)], dtype=float)

    def best_arm(self) -> int:
        return int(np.argmax(self.means_at(1)))
    
    def best_mean(self) -> float:
        return float(np.max(self.means_at(1)))

    def info(self) -> Dict[str, Any]:
        return {}

class AntiCorrelatedGaussianBandit(AbstractBandit):
    """Two-arm Gaussian bandit with perfectly anti-correlated means."""

    def __init__(self, mean_arm0: float, sigma: float = 1.0) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu0 = float(mean_arm0)
        self.mu1 = float(1.0 - mean_arm0)
        self.sigma = float(sigma)
        self.n_arms = 2

    def reset(self, rng: np.random.Generator) -> None:
        del rng

    def sample(self, a: int, rng: np.random.Generator) -> float:
        if a not in (0, 1):
            raise IndexError("arm must be 0 or 1")
        mean = self.mu0 if a == 0 else self.mu1
        return float(rng.normal(loc=mean, scale=self.sigma))

    def mean(self, a: int) -> float:
        if a not in (0, 1):
            raise IndexError("arm must be 0 or 1")
        return float(self.mu0 if a == 0 else self.mu1)

    def info(self) -> Dict[str, Any]:
        return {"type": "AntiCorrelatedGaussian", "mu0": self.mu0, "mu1": self.mu1, "sigma": self.sigma}


class LinearBandit(AbstractBandit):
    """Structured bandit with known arm features and linear rewards."""

    def __init__(
        self,
        features: np.ndarray,
        theta: np.ndarray,
        *,
        sigma: float = 1.0,
        provide_features: bool = True,
    ) -> None:
        X = np.asarray(features, dtype=float)
        if X.ndim != 2:
            raise ValueError("features must be a 2-D array")
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.shape[0] != X.shape[1]:
            raise ValueError("theta dimension must match feature dimension")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.X = X
        self.theta = theta
        self.sigma = float(sigma)
        self.n_arms = int(X.shape[0])
        self._means = X @ theta
        self.provide_features = bool(provide_features)

    def reset(self, rng: np.random.Generator) -> None:
        del rng

    def sample(self, a: int, rng: np.random.Generator) -> float | tuple[float, Dict[str, Any]]:
        if not (0 <= a < self.n_arms):
            raise IndexError("arm out of range")
        mean = float(self._means[a])
        reward = float(rng.normal(loc=mean, scale=self.sigma))
        if self.provide_features:
            return reward, {"x": self.X[a].copy()}
        return reward

    def mean(self, a: int) -> float:
        if not (0 <= a < self.n_arms):
            raise IndexError("arm out of range")
        return float(self._means[a])

    def info(self) -> Dict[str, Any]:
        return {
            "type": "LinearBandit",
            "n_arms": self.n_arms,
            "dimension": int(self.X.shape[1]),
            "sigma": self.sigma,
        }


# -------- Unstructured examples --------

class BernoulliBandit(AbstractBandit):
    def __init__(self, probs: Sequence[float]) -> None:
        self._p = np.asarray(probs, dtype=float)
        if np.any((self._p < 0) | (self._p > 1)):
            raise ValueError("Bernoulli probabilities must be in [0,1].")
        self.n_arms = int(self._p.size)

    def reset(self, rng: np.random.Generator) -> None:
        pass  

    def sample(self, a: int, rng: np.random.Generator) -> float:
        return float(rng.random() < self._p[a])

    def mean(self, a: int) -> float:
        return float(self._p[a])

    def info(self):
        return {"type": "Bernoulli", "probs": self._p.tolist()}


class GaussianBandit(AbstractBandit):
    def __init__(self, means: Sequence[float], sigma: float = 1.0) -> None:
        self._mu = np.asarray(means, dtype=float)
        self._sigma = float(sigma)
        if self._sigma <= 0:
            raise ValueError("sigma must be > 0")
        self.n_arms = int(self._mu.size)
        self._arm_generators: List[np.random.Generator] = []

    def _arm_seed_sequence(self, rng: np.random.Generator) -> np.random.SeedSequence:
        state = rng.bit_generator.state
        seed_value: int
        if isinstance(state, dict):
            inner_state = state.get("state")
            if isinstance(inner_state, dict) and "state" in inner_state:
                seed_value = int(inner_state["state"])
            else:
                seed_value = int(inner_state)
        else:
            seed_value = int(state)
        return np.random.SeedSequence(seed_value)

    def reset(self, rng: np.random.Generator) -> None:
        seed_sequence = self._arm_seed_sequence(rng)
        counts: Dict[float, int] = {}
        keys: List[Tuple[Tuple[float, int], int]] = []
        for idx, mean in enumerate(self._mu.tolist()):
            occurrence = counts.get(mean, 0)
            counts[mean] = occurrence + 1
            keys.append(((mean, occurrence), idx))
        keys.sort(key=lambda item: item[0])
        child_sequences = seed_sequence.spawn(self.n_arms)
        generators: List[np.random.Generator] = [None] * self.n_arms  # type: ignore[list-item]
        for child_seq, (_, original_idx) in zip(child_sequences, keys):
            generators[original_idx] = np.random.default_rng(child_seq)
        self._arm_generators = generators

    def sample(self, a: int, rng: np.random.Generator) -> float:
        if not (0 <= a < self.n_arms):
            raise IndexError("arm out of range")
        if not self._arm_generators:
            # Fallback to provided rng if reset has not been called.
            return float(rng.normal(loc=self._mu[a], scale=self._sigma))
        noise = self._arm_generators[a].standard_normal()
        return float(self._mu[a] + self._sigma * noise)

    def mean(self, a: int) -> float:
        return float(self._mu[a])

    def info(self):
        return {"type": "Gaussian", "means": self._mu.tolist(), "sigma": self._sigma}


# -------- Structured / causal examples --------

class TwoArmComplementBernoulli(AbstractBandit):
    """
    Simple structured bandit: two arms share a single parameter theta.
    Arm 0 ~ Ber(theta), Arm 1 ~ Ber(1-theta).
    """
    def __init__(self, theta: float) -> None:
        if not (0.0 <= theta <= 1.0):
            raise ValueError("theta must be in [0,1].")
        self.theta = float(theta)
        self.n_arms = 2

    def reset(self, rng: np.random.Generator) -> None:
        pass

    def sample(self, a: int, rng: np.random.Generator) -> float:
        p = self.theta if a == 0 else (1.0 - self.theta)
        return float(rng.random() < p)

    def mean(self, a: int) -> float:
        return float(self.theta if a == 0 else (1.0 - self.theta))

    def info(self):
        return {"type": "TwoArmComplementBernoulli", "theta": self.theta}


class CausalInterventionalBandit(AbstractBandit):
    """
    Bandit whose arms correspond to interventions in a causal SCM.

    The class receives a fully specified :class:`CausalBanditInstance` and a
    collection of intervention arms (typically generated via
    :class:`InterventionSpace`). Each pull applies the corresponding hard
    intervention to the SCM and returns the sampled reward along with the
    observed values of non-intervened covariates.
    """

    def __init__(
        self,
        instance: CausalBanditInstance,
        interventions: Sequence[InterventionArm],
        *,
        mean_mc_samples: int = 2048,
    ) -> None:
        if not interventions:
            raise ValueError("At least one intervention arm is required.")
        self.instance = instance
        self._covariate_names = tuple(instance.node_names[:-1])
        if len(self._covariate_names) != instance.config.n:
            raise ValueError("Mismatch between config.n and covariate names.")
        self._arms: Tuple[InterventionArm, ...] = tuple(interventions)
        self._interventions: Tuple[Intervention, ...] = tuple(
            arm_to_intervention(arm, self._covariate_names, name=f"arm_{idx}")
            for idx, arm in enumerate(self._arms)
        )
        self.n_arms = len(self._arms)
        self._mean_cache: Dict[int, float] = {}
        self._mean_mc_samples = max(1, int(mean_mc_samples))

    @classmethod
    def from_space(
        cls,
        instance: CausalBanditInstance,
        *,
        space: Optional[InterventionSpace] = None,
        subset_size: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        mean_mc_samples: int = 2048,
    ) -> "CausalInterventionalBandit":
        rng = rng or instance.config.rng()
        space = space or InterventionSpace(instance.config.n, instance.config.ell, instance.config.m)
        if subset_size is None:
            interventions = list(space.arms())
        else:
            interventions = space.random_subset(subset_size, rng, replace=False)
        return cls(instance, interventions, mean_mc_samples=mean_mc_samples)

    def reset(self, rng: np.random.Generator) -> None:
        del rng  # nothing to reset, SCM is stateless between pulls

    def _non_intervened_values(self, sample: Dict[str, Any], arm: InterventionArm) -> Dict[str, Any]:
        intervened = set(arm.variables)
        result: Dict[str, Any] = {}
        for idx, name in enumerate(self._covariate_names):
            if idx in intervened:
                continue
            result[name] = sample[name]
        return result

    def sample(self, a: int, rng: np.random.Generator) -> Tuple[float, Dict[str, Any]]:
        if not (0 <= a < self.n_arms):
            raise IndexError("arm out of range")
        intervention = self._interventions[a]
        sample = self.instance.scm.sample(rng, intervention=intervention)
        reward = float(sample[self.instance.reward_node])
        info = {
            "assignment": sample,
            "intervention": intervention,
            "non_intervened": self._non_intervened_values(sample, self._arms[a]),
        }
        return reward, info

    def mean(self, a: int) -> float:
        if not (0 <= a < self.n_arms):
            raise IndexError("arm out of range")
        if a not in self._mean_cache:
            self._mean_cache[a] = float(
                self.instance.scm.mean(
                    self.instance.reward_node,
                    intervention=self._interventions[a],
                    n_mc=self._mean_mc_samples,
                )
            )
        return self._mean_cache[a]

    def info(self) -> Dict[str, Any]:
        data = super().info()
        data.update(
            {
                "type": "CausalInterventionalBandit",
                "n_covariates": self.instance.config.n,
                "ell": self.instance.config.ell,
                "k": self.instance.config.k,
                "m": self.instance.config.m,
                "n_arms": self.n_arms,
            }
        )
        return data


class SCMBandit(AbstractBandit):
    """
    Arms = interventions on an SCM; reward = value of `reward_node` under do(a).
    If feedback == "causal", also returns a dict of observed nodes (e.g., parents of Y or 'all').
    """
    def __init__(
        self,
        scm: SCM,
        interventions: List[Intervention],
        reward_node: str,
        observe: Optional[Iterable[str] | str] = None,  # None | "parents" | "all" | list of nodes
        feedback: str = "reward",
        mc_mean: int = 10_000,   # MC samples if analytic mean is not available
    ):
        self.scm = scm
        self.interventions = list(interventions)
        self.reward_node = reward_node
        self.n_arms = len(self.interventions)
        self.feedback = feedback
        self.observe = observe
        self.mc_mean = int(mc_mean)

        if reward_node not in scm.nodes:
            raise ValueError(f"reward_node '{reward_node}' not in SCM nodes")

    def reset(self, rng: np.random.Generator) -> None:
        pass

    def _obs_nodes(self) -> List[str]:
        if self.observe is None or self.feedback == "reward":
            return []
        if isinstance(self.observe, str):
            if self.observe == "all":
                return list(self.scm.nodes)
            if self.observe == "parents":
                return self.scm.parents_of(self.reward_node)
            raise ValueError("observe must be None|'parents'|'all'|list[str]")
        return list(self.observe)

    def sample(self, a: int, rng: np.random.Generator) -> float | Tuple[float, Dict[str, Any]]:
        interv = self.interventions[a]
        nodes_to_return = set([self.reward_node] + self._obs_nodes())
        vals = self.scm.sample(rng, intervention=interv, return_nodes=nodes_to_return)
        y = float(vals[self.reward_node])
        if self.feedback == "causal":
            obs = {k: v for k, v in vals.items() if k != self.reward_node}
            treated_nodes = set(interv.hard.keys()) | set(interv.soft.keys())
            if treated_nodes:
                obs = {k: v for k, v in obs.items() if k not in treated_nodes}
            return y, obs
        return y

    def mean(self, a: int) -> float:
        interv = self.interventions[a]
        # Prefer analytic mean if SCM supports it (LinearGaussianSCM)
        if isinstance(self.scm, LinearGaussianSCM) and not interv.soft:
            return float(self.scm.mean(self.reward_node, intervention=interv))
        # Fallback to Monte Carlo mean for general SCM or soft interventions
        return float(self.scm.mean(self.reward_node, intervention=interv, n_mc=self.mc_mean))

    def info(self) -> Dict[str, Any]:
        return {
            "type": "SCMBandit",
            "reward_node": self.reward_node,
            "arms": [iv.name for iv in self.interventions],
            "feedback": self.feedback,
            "observe": self.observe if not isinstance(self.observe, list) else list(self.observe),
        }
    
# -------- Variants for extended experiments --------


class StudentTBandit(GaussianBandit):
    """GaussianBandit with heavy-tailed Student-t observation noise."""

    def __init__(self, means: Sequence[float], sigma: float = 1.0, nu: int = 5) -> None:
        super().__init__(means, sigma=sigma)
        if nu <= 2:
            raise ValueError("nu must be greater than 2 for finite variance")
        self.nu = int(nu)

    def sample(self, a: int, rng: np.random.Generator) -> float:
        if not (0 <= a < self.n_arms):
            raise IndexError("arm out of range")
        if self._arm_generators:
            noise = self._arm_generators[a].standard_t(self.nu) * self._sigma
        else:
            noise = rng.standard_t(self.nu) * self._sigma
        return float(self._mu[a] + noise)

    def info(self) -> Dict[str, Any]:
        data = super().info()
        data["nu"] = self.nu
        data["type"] = "StudentTGaussian"
        return data


class DriftingGaussianBandit(GaussianBandit):
    """Gaussian bandit whose means drift linearly over time."""

    def __init__(
        self,
        means: Sequence[float],
        *,
        drift: Sequence[float],
        sigma: float = 1.0,
        horizon: Optional[int] = None,
    ) -> None:
        super().__init__(means, sigma=sigma)
        base = np.asarray(means, dtype=float)
        drift_vec = np.asarray(drift, dtype=float)
        if drift_vec.shape != base.shape:
            raise ValueError("drift vector must match number of arms")
        self._base_means = base
        self._drift = drift_vec
        self._horizon = int(horizon) if horizon is not None and horizon > 0 else None
        self._t = 0
        self.is_stationary = False

    def set_horizon(self, horizon: int) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self._horizon = int(horizon)

    def reset(self, rng: np.random.Generator) -> None:
        super().reset(rng)
        self._t = 0

    def means_at(self, t: int) -> np.ndarray:
        if t <= 0:
            raise ValueError("t must be >= 1")
        if self._horizon is not None and self._horizon > 1:
            frac = min(max(t - 1, 0) / (self._horizon - 1), 1.0)
        else:
            frac = float(max(t - 1, 0))
        return self._base_means + frac * self._drift

    def sample(self, a: int, rng: np.random.Generator) -> float:
        if not (0 <= a < self.n_arms):
            raise IndexError("arm out of range")
        t = self._t + 1
        means = self.means_at(t)
        if self._arm_generators:
            noise = self._arm_generators[a].standard_normal()
        else:
            noise = rng.standard_normal()
        reward = means[a] + self._sigma * noise
        self._t = t
        return float(reward)

    def mean(self, a: int) -> float:
        if not (0 <= a < self.n_arms):
            raise IndexError("arm out of range")
        return float(self._base_means[a])

    def info(self) -> Dict[str, Any]:
        data = super().info()
        data.update({"type": "DriftingGaussian", "drift": self._drift.tolist(), "horizon": self._horizon})
        return data


class NonlinearBandit(LinearBandit):
    """LinearBandit variant with quadratic feature misspecification."""

    def __init__(
        self,
        features: np.ndarray,
        theta_linear: np.ndarray,
        theta_quadratic: np.ndarray,
        *,
        sigma: float = 1.0,
        provide_features: bool = True,
    ) -> None:
        self._theta_linear = np.asarray(theta_linear, dtype=float)
        self._theta_quadratic = np.asarray(theta_quadratic, dtype=float)
        if self._theta_quadratic.ndim != 1:
            self._theta_quadratic = self._theta_quadratic.reshape(-1)
        super().__init__(features=features, theta=self._theta_linear, sigma=sigma, provide_features=provide_features)
        expected = [self._compute_mean(self.X[a]) for a in range(self.n_arms)]
        self._means = np.asarray(expected, dtype=float)

    def _feature_map(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([x, x * x])

    def _compute_mean(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        linear = float(np.dot(self._theta_linear, x))
        quad = float(np.dot(self._theta_quadratic, self._feature_map(x)))
        return linear + quad

    def sample(self, a: int, rng: np.random.Generator) -> float | tuple[float, Dict[str, Any]]:
        if not (0 <= a < self.n_arms):
            raise IndexError("arm out of range")
        mean = self._compute_mean(self.X[a])
        reward = float(rng.normal(loc=mean, scale=self.sigma))
        if self.provide_features:
            return reward, {"x": self.X[a].copy()}
        return reward

    def mean(self, a: int) -> float:
        if not (0 <= a < self.n_arms):
            raise IndexError("arm out of range")
        return float(self._means[a])

    def info(self) -> Dict[str, Any]:
        data = super().info()
        data.update({"type": "NonlinearBandit"})
        return data


# -------- Structured / causal examples --------

class TwoArmComplementBernoulli(AbstractBandit):
    """
    Simple structured bandit: two arms share a single parameter theta.
    Arm 0 ~ Ber(theta), Arm 1 ~ Ber(1-theta).
    """
    def __init__(self, theta: float) -> None:
        if not (0.0 <= theta <= 1.0):
            raise ValueError("theta must be in [0,1].")
        self.theta = float(theta)
        self.n_arms = 2

    def reset(self, rng: np.random.Generator) -> None:
        pass

    def sample(self, a: int, rng: np.random.Generator) -> float:
        p = self.theta if a == 0 else (1.0 - self.theta)
        return float(rng.random() < p)

    def mean(self, a: int) -> float:
        return float(self.theta if a == 0 else (1.0 - self.theta))

    def info(self):
        return {"type": "TwoArmComplementBernoulli", "theta": self.theta}
