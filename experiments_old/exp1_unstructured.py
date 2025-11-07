from __future__ import annotations

import argparse
import os
from typing import Callable, Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..Algorithm import BasePolicy
from ..Bandit import GaussianBandit
from ..classical_bandits import (
    EpsilonGreedy,
    ExploreThenCommit,
    GaussianThompsonPolicy,
    RandomPolicy,
    UCB,
)
from .common import ensure_dir, run_policy_trials, save_json


RESULTS_DIR = os.path.join("results", "exp1_unstructured")
ensure_dir(RESULTS_DIR)


MEANS = np.array([0.0, 0.25, 0.15, -0.05, 0.1], dtype=float)
SIGMA = 0.3  # known variance setting


def make_bandit() -> GaussianBandit:
    return GaussianBandit(MEANS, sigma=SIGMA)


def policy_factories(sigma: float) -> Dict[str, Callable[[], BasePolicy]]:
    prior_means = np.zeros_like(MEANS)
    prior_vars = np.ones_like(MEANS)
    sigma2 = float(sigma ** 2)
    return {
        "random": lambda: RandomPolicy(seed=12345),
        "etc": lambda: ExploreThenCommit(tau=0.2),
        "epsilon-greedy": lambda: EpsilonGreedy(epsilon=0.1),
        "ucb": lambda: UCB(schedule="ucb1_alpha", alpha=2.0),
        "thompson-gaussian": lambda: GaussianThompsonPolicy(prior_means, prior_vars, sigma2),
    }


def plot_regret_curves(
    *,
    horizon: int,
    curves: Dict[str, np.ndarray],
    ci: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    t = np.arange(1, horizon + 1)
    plt.figure(figsize=(8, 5))
    for name, curve in curves.items():
        band = ci[name]
        lower = np.maximum(curve - band, 0.0)
        upper = curve + band
        plt.plot(t, curve, label=name)
        plt.fill_between(t, lower, upper, alpha=0.2)
    plt.xlabel("round t")
    plt.ylabel("cumulative pseudo-regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pulls(
    *,
    means: Dict[str, np.ndarray],
    ci: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    policies = list(means.keys())
    n_arms = len(MEANS)
    x = np.arange(n_arms)
    width = 0.12
    plt.figure(figsize=(9, 5))
    for idx, name in enumerate(policies):
        offset = (idx - (len(policies) - 1) / 2) * width
        plt.bar(x + offset, means[name], width=width, label=name, yerr=ci[name], capsize=3)
    plt.xticks(x, [f"arm {i}" for i in range(n_arms)])
    plt.ylabel("mean pulls")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 1: unstructured bandits baseline.")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1_000, 10_000], help="Interaction horizons.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(50)), help="Random seeds to run.")
    parser.add_argument("--sigma", type=float, default=SIGMA, help="Known Gaussian reward std deviation.")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR, help="Directory to store JSON/plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seeds = args.seeds
    horizons = args.horizons
    dirname = args.results_dir
    ensure_dir(dirname)

    factories = policy_factories(args.sigma)

    all_results: Dict[str, Dict[int, Dict[str, np.ndarray | float]]] = {}
    for name, factory in factories.items():
        metrics = run_policy_trials(
            bandit_factory=lambda: GaussianBandit(MEANS, sigma=args.sigma),
            policy_factory=factory,
            horizons=horizons,
            seeds=seeds,
        )
        all_results[name] = metrics
        for horizon, payload in metrics.items():
            save_json(
                os.path.join(dirname, f"{name}_H{horizon}.json"),
                {k: v for k, v in payload.items()},
            )
            mean_regret = float(payload["mean_regret"])
            ci_regret = float(payload["ci_regret"])
            print(f"[exp1] policy={name} horizon={horizon} mean regret = {mean_regret:.2f} ± {ci_regret:.2f}")

    plot_horizon = max(horizons)
    curves = {name: np.asarray(results[plot_horizon]["mean_curve"]) for name, results in all_results.items()}
    curve_ci = {name: np.asarray(results[plot_horizon]["ci_curve"]) for name, results in all_results.items()}
    plot_regret_curves(
        horizon=plot_horizon,
        curves=curves,
        ci=curve_ci,
        output_path=os.path.join(dirname, f"regret_H{plot_horizon}.png"),
    )

    pull_means = {name: np.asarray(results[plot_horizon]["mean_pulls"]) for name, results in all_results.items()}
    pull_ci = {name: np.asarray(results[plot_horizon]["ci_pulls"]) for name, results in all_results.items()}
    plot_pulls(
        means=pull_means,
        ci=pull_ci,
        output_path=os.path.join(dirname, f"pulls_H{plot_horizon}.png"),
    )


if __name__ == "__main__":
    main()
