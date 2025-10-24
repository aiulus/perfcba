from __future__ import annotations

import os
from typing import Callable, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Algorithm import BasePolicy
from Bandit import LinearBandit
from classical_bandits import UCB
from experiments.common import ensure_dir, run_policy_trials, save_json
from linear_bandits import LinThompsonSampling, LinUCB


RESULTS_DIR = os.path.join("results", "exp4_ablation")
ensure_dir(RESULTS_DIR)

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

SIGMA_LEVELS = [0.1, 0.3, 0.6]
HORIZON_LEVELS = [500, 2_000, 5_000, 10_000]
NOISE_EVAL_HORIZON = 5_000
SEEDS = list(range(50))


def bandit_factory_for_sigma(sigma: float) -> Callable[[], LinearBandit]:
    def _factory() -> LinearBandit:
        return LinearBandit(features=FEATURES, theta=THETA, sigma=sigma, provide_features=True)

    return _factory


def policy_builders() -> Dict[str, Callable[[float], Callable[[], BasePolicy]]]:
    return {
        "ucb": lambda _sigma: (lambda: UCB(schedule="ucb1_alpha", alpha=2.0)),
        "lin-ucb": lambda _sigma: (lambda: LinUCB(FEATURES, alpha=1.5, lam=1.0)),
        "lin-thompson": lambda sigma: (lambda: LinThompsonSampling(FEATURES, sigma2=sigma ** 2, lam=1.0)),
    }


def plot_curve(stats: Dict[str, Dict[float, Dict[str, float]]], xlabel: str, title: str, path: str) -> None:
    plt.figure(figsize=(7, 4))
    for name, values in stats.items():
        xs = list(sorted(values.keys()))
        means = [values[val]["mean_regret"] for val in xs]
        cis = [values[val]["ci_regret"] for val in xs]
        plt.plot(xs, means, label=name)
        plt.fill_between(xs, np.array(means) - np.array(cis), np.array(means) + np.array(cis), alpha=0.15)
    plt.xlabel(xlabel)
    plt.ylabel("mean total regret")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    builders = policy_builders()

    # Noise ablation
    noise_results: Dict[str, Dict[float, Dict[str, float]]] = {name: {} for name in builders}
    for sigma in SIGMA_LEVELS:
        bandit_factory = bandit_factory_for_sigma(sigma)
        for name, builder in builders.items():
            metrics = run_policy_trials(
                bandit_factory=bandit_factory,
                policy_factory=builder(sigma),
                horizons=[NOISE_EVAL_HORIZON],
                seeds=SEEDS,
            )[NOISE_EVAL_HORIZON]
            noise_results[name][sigma] = {
                "mean_regret": float(metrics["mean_regret"]),
                "ci_regret": float(metrics["ci_regret"]),
            }
            print(
                f"[exp4-noise] policy={name} sigma={sigma} mean regret = "
                f"{metrics['mean_regret']:.2f} ± {metrics['ci_regret']:.2f}"
            )

    save_json(os.path.join(RESULTS_DIR, "noise_ablation.json"), noise_results)
    plot_curve(
        stats=noise_results,
        xlabel="reward noise σ",
        title=f"Noise sensitivity (T={NOISE_EVAL_HORIZON})",
        path=os.path.join(RESULTS_DIR, "noise_ablation.png"),
    )

    # Horizon ablation (fix sigma to medium level)
    sigma_ref = SIGMA_LEVELS[1]
    horizon_results: Dict[str, Dict[float, Dict[str, float]]] = {name: {} for name in builders}
    bandit_factory = bandit_factory_for_sigma(sigma_ref)
    for name, builder in builders.items():
        metrics_per_h = run_policy_trials(
            bandit_factory=bandit_factory,
            policy_factory=builder(sigma_ref),
            horizons=HORIZON_LEVELS,
            seeds=SEEDS,
        )
        for horizon, payload in metrics_per_h.items():
            horizon_results[name][horizon] = {
                "mean_regret": float(payload["mean_regret"]),
                "ci_regret": float(payload["ci_regret"]),
            }
            print(
                f"[exp4-horizon] policy={name} horizon={horizon} mean regret = "
                f"{payload['mean_regret']:.2f} ± {payload['ci_regret']:.2f}"
            )

    save_json(os.path.join(RESULTS_DIR, "horizon_ablation.json"), horizon_results)
    plot_curve(
        stats=horizon_results,
        xlabel="horizon T",
        title=f"Horizon scaling (σ={sigma_ref})",
        path=os.path.join(RESULTS_DIR, "horizon_ablation.png"),
    )


if __name__ == "__main__":
    main()
