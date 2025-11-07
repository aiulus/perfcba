from __future__ import annotations

import argparse
import os
from typing import Callable, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..Algorithm import BasePolicy
from ..Bandit import LinearBandit
from ..classical_bandits import ExploreThenCommit, UCB
from .common import ensure_dir, run_policy_trials, save_json
from ..linear_bandits import LinThompsonSampling, LinUCB


DEFAULT_RESULTS_DIR = os.path.join("results", "exp2_linear")
ensure_dir(DEFAULT_RESULTS_DIR)


FEATURES = np.array(
    [
        [1.0, 0.0],
        [0.7, 0.3],
        [0.4, 0.6],
        [0.0, 1.0],
        [0.5, 0.5],
    ],
    dtype=float,
)
THETA = np.array([0.4, 0.8], dtype=float)
SIGMA = 0.2


def make_bandit() -> LinearBandit:
    return LinearBandit(features=FEATURES, theta=THETA, sigma=SIGMA, provide_features=True)


def policy_factories(sigma: float) -> Dict[str, Callable[[], BasePolicy]]:
    n_arms = FEATURES.shape[0]
    return {
        "etc": lambda: ExploreThenCommit(tau=0.2),
        "ucb": lambda: UCB(schedule="ucb1_alpha", alpha=2.0),
        "lin-ucb": lambda: LinUCB(FEATURES, alpha=1.5, lam=1.0),
        "lin-thompson": lambda: LinThompsonSampling(FEATURES, sigma2=sigma ** 2, lam=1.0),
    }


def plot_regret(
    *,
    horizon: int,
    mean_curves: Dict[str, np.ndarray],
    ci_curves: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    t = np.arange(1, horizon + 1)
    plt.figure(figsize=(8, 5))
    for name, curve in mean_curves.items():
        ci = ci_curves[name]
        plt.plot(t, curve, label=name)
        plt.fill_between(t, np.maximum(curve - ci, 0.0), curve + ci, alpha=0.2)
    plt.xlabel("round t")
    plt.ylabel("cumulative pseudo-regret")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_estimation_error(
    *,
    true_means: np.ndarray,
    mean_errors: Dict[str, np.ndarray],
    ci_errors: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    arms = np.arange(len(true_means))
    width = 0.18
    plt.figure(figsize=(9, 5))
    for idx, (name, errs) in enumerate(mean_errors.items()):
        offset = (idx - (len(mean_errors) - 1) / 2) * width
        plt.bar(arms + offset, errs, width=width, label=name, yerr=ci_errors[name], capsize=3)
    plt.xticks(arms, [f"arm {i}" for i in arms])
    plt.ylabel("mean absolute error |μ̂_i - μ_i|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 2: linear bandits with shared structure.")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1_000, 10_000], help="Interaction horizons.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(50)), help="Random seeds to run.")
    parser.add_argument("--sigma", type=float, default=SIGMA, help="Reward noise standard deviation.")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Directory for outputs.")
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
            bandit_factory=lambda: LinearBandit(FEATURES, THETA, sigma=args.sigma, provide_features=True),
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
            print(
                f"[exp2] policy={name} horizon={horizon} mean regret = "
                f"{payload['mean_regret']:.2f} ± {payload['ci_regret']:.2f}"
            )

    plot_horizon = max(horizons)
    mean_curves = {name: np.asarray(results[plot_horizon]["mean_curve"]) for name, results in all_results.items()}
    ci_curves = {name: np.asarray(results[plot_horizon]["ci_curve"]) for name, results in all_results.items()}
    plot_regret(
        horizon=plot_horizon,
        mean_curves=mean_curves,
        ci_curves=ci_curves,
        output_path=os.path.join(dirname, f"regret_H{plot_horizon}.png"),
    )

    # Estimation error bars (use true means from any policy since identical)
    true_means = next(iter(all_results.values()))[plot_horizon]["true_means"]
    mean_errors = {name: np.asarray(results[plot_horizon]["mean_abs_error"]) for name, results in all_results.items()}
    ci_errors = {name: np.asarray(results[plot_horizon]["ci_abs_error"]) for name, results in all_results.items()}
    plot_estimation_error(
        true_means=np.asarray(true_means),
        mean_errors=mean_errors,
        ci_errors=ci_errors,
        output_path=os.path.join(dirname, f"mean_error_H{plot_horizon}.png"),
    )


if __name__ == "__main__":
    main()
