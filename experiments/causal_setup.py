"""
High-level helpers for instantiating and running causal bandit experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from ..Algorithm import BasePolicy
from ..Experiment import Experiment, RunConfig
from ..Bandit import CausalInterventionalBandit
from .causal_envs import (
    CausalBanditConfig,
    CausalBanditInstance,
    InterventionSpace,
    build_random_scm,
)


@dataclass
class GeneratedCausalBandit:
    """Container for a generated causal bandit environment."""

    instance: CausalBanditInstance
    space: InterventionSpace
    bandit: CausalInterventionalBandit


def make_causal_bandit(
    config: CausalBanditConfig,
    *,
    subset_size: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    mean_mc_samples: int = 2048,
) -> GeneratedCausalBandit:
    """
    Sample an SCM from ``config`` and wrap it into a
    :class:`CausalInterventionalBandit`.
    """

    rng = rng or config.rng()
    instance = build_random_scm(config, rng=rng)
    space = InterventionSpace(config.n, config.ell, config.m)
    if subset_size is None:
        arms = list(space.arms())
    else:
        arms = space.random_subset(subset_size, rng, replace=False)
    bandit = CausalInterventionalBandit(instance, arms, mean_mc_samples=mean_mc_samples)
    return GeneratedCausalBandit(instance=instance, space=space, bandit=bandit)


def run_causal_experiment(
    *,
    config: CausalBanditConfig,
    policy_factory: Callable[[], BasePolicy],
    horizon: int,
    subset_size: Optional[int] = None,
    master_seed: Optional[int] = None,
    mean_mc_samples: int = 2048,
) -> tuple[GeneratedCausalBandit, Sequence[float]]:
    """
    Convenience wrapper that generates a bandit, instantiates the policy and
    runs a single :class:`Experiment`.

    Returns the generated bandit metadata together with the observed rewards.
    """

    rng = np.random.default_rng(master_seed)
    generated = make_causal_bandit(
        config,
        subset_size=subset_size,
        rng=rng,
        mean_mc_samples=mean_mc_samples,
    )
    policy = policy_factory()
    history = Experiment(
        generated.bandit,
        policy,
        RunConfig(T=horizon, seed=int(rng.integers(2**32 - 1))),
    ).run()
    return generated, history.rewards
