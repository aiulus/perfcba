from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np

from .Algorithm import History, BasePolicy
from .Bandit import AbstractBandit


@dataclass
class RunConfig:
    T: int
    seed: int = 0
    extra: Dict[str, Any] = None


class Experiment:
    """
    Orchestrator for a single policy-vs-bandit run.
    Responsible for RNG, time loop, and logging; metrics here.
    """
    def __init__(self, bandit: AbstractBandit, policy: BasePolicy, config: RunConfig):
        self.bandit = bandit
        self.policy = policy
        self.config = config

    def run(self) -> History:
        T = int(self.config.T)
        rng = np.random.default_rng(self.config.seed)

        self.bandit.reset(rng)
        self.policy.reset(self.bandit.n_arms, horizon=T)

        actions = np.zeros(T, dtype=int)
        rewards = np.zeros(T, dtype=float)
        mu_star = np.full(T, self.bandit.best_mean(), dtype=float)
        obs_log: list[dict] = []

        for t in range(1, T + 1):
            hist_for_policy = History(
                T=t - 1,
                actions=actions[: t - 1].copy(),
                rewards=rewards[: t - 1].copy(),
                means_star=mu_star[: t - 1].copy(),
                n_arms=self.bandit.n_arms,
                pulls=np.bincount(actions[: t - 1], minlength=self.bandit.n_arms),
                cum_reward=float(rewards[: t - 1].sum()),
                observations=obs_log[: t - 1].copy(),
            )
            a = int(self.policy.choose(t, hist_for_policy))
            out = self.bandit.sample(a, rng)
            if isinstance(out, tuple):
                x, obs = out
                obs_log.append(obs)
            else:
                x = out
                obs_log.append({})

            self.policy.update(t, a, float(x), info=obs_log[-1])
            actions[t - 1] = a
            rewards[t - 1] = float(x)

        final_pulls = np.bincount(actions, minlength=self.bandit.n_arms)
        return History(
            T=T,
            actions=actions,
            rewards=rewards,
            means_star=mu_star,
            n_arms=self.bandit.n_arms,
            pulls=final_pulls,
            cum_reward=float(rewards.sum()),
            observations=obs_log,
        )