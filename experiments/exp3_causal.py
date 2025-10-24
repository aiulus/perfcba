from __future__ import annotations

import os
from typing import Callable, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Algorithm import BasePolicy
from Bandit import SCMBandit
from SCM import Intervention, SCM
from backdoor_bandits import BackdoorTS, BackdoorUCB
from classical_bandits import GaussianThompsonPolicy, UCB
from estimators.models import MultinomialLogisticRegression, RidgeOutcomeRegressor
from estimators.robust import dr_crossfit
from experiments.common import ensure_dir, run_policy_trials, save_json


RESULTS_DIR = os.path.join("results", "exp3_causal")
ensure_dir(RESULTS_DIR)


def build_scm() -> SCM:
    nodes = ["Z", "T", "Y"]
    parents = {"Z": [], "T": ["Z"], "Y": ["T", "Z"]}

    def f_z(_parents, rng: np.random.Generator) -> float:
        return float(rng.normal(loc=0.0, scale=1.0))

    def f_t(par, rng: np.random.Generator) -> float:
        z = par["Z"]
        noise = rng.normal(loc=0.0, scale=0.5)
        return float(1.0 if z + noise > 0.0 else 0.0)

    def f_y(par, rng: np.random.Generator) -> float:
        t = par["T"]
        z = par["Z"]
        eps = rng.normal(loc=0.0, scale=0.3)
        return float(2.0 * t + 0.7 * z + eps)

    mechanisms = {"Z": f_z, "T": f_t, "Y": f_y}
    return SCM(nodes=nodes, parents=parents, f=mechanisms)


INTERVENTIONS = [
    Intervention("do(T=0)", hard={"T": 0.0}),
    Intervention("do(T=1)", hard={"T": 1.0}),
]


def make_bandit() -> SCMBandit:
    scm = build_scm()
    return SCMBandit(
        scm=scm,
        interventions=INTERVENTIONS,
        reward_node="Y",
        observe="parents",
        feedback="causal",
    )


def policy_factories() -> Dict[str, Callable[[], BasePolicy]]:
    prior_means = [0.0, 0.0]
    prior_vars = [1.0, 1.0]
    sigma2 = 0.3 ** 2
    return {
        "ucb": lambda: UCB(schedule="ucb1_alpha", alpha=2.0),
        "thompson-gaussian": lambda: GaussianThompsonPolicy(prior_means, prior_vars, sigma2),
        "backdoor-ucb": lambda: BackdoorUCB(n_arms=2, refit_every=25, clip=10.0, alpha=1.0),
        "backdoor-ts": lambda: BackdoorTS(n_arms=2, refit_every=25, clip=10.0, variance_scale=1.0, min_samples=10),
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


def sample_observational_dataset(n_samples: int = 20_000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SCM]:
    scm = build_scm()
    rng = np.random.default_rng(seed)
    y = np.zeros(n_samples, dtype=float)
    a = np.zeros(n_samples, dtype=int)
    z = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        sample = scm.sample(rng)
        z[i] = float(sample["Z"])
        a[i] = int(sample["T"])
        y[i] = float(sample["Y"])
    x = z.reshape(-1, 1)
    return y, a, x, scm


def compute_observational_bias() -> Dict[str, Dict[str, float]]:
    y, a, x, scm = sample_observational_dataset()
    results: Dict[str, Dict[str, float]] = {}
    naive: Dict[int, float] = {}
    dr_est: Dict[int, float] = {}
    true_means: Dict[int, float] = {}

    for arm in range(2):
        mask = a == arm
        naive[arm] = float(np.mean(y[mask])) if np.any(mask) else float("nan")
        prop = MultinomialLogisticRegression(n_classes=2)
        outcome = RidgeOutcomeRegressor(l2=1e-2)
        dr_value, _, _ = dr_crossfit(y, a, x, arm, outcome, prop, K=2, clip=10.0, random_state=42)
        dr_est[arm] = float(dr_value)
        true_means[arm] = float(scm.mean("Y", intervention=INTERVENTIONS[arm], n_mc=10000, seed=1234))

    results["true"] = {f"arm_{k}": v for k, v in true_means.items()}
    results["naive"] = {f"arm_{k}": naive[k] - true_means[k] for k in true_means}
    results["doubly_robust"] = {f"arm_{k}": dr_est[k] - true_means[k] for k in true_means}
    return results


def plot_bias(bias: Dict[str, Dict[str, float]], output_path: str) -> None:
    estimators = [k for k in bias.keys() if k != "true"]
    arms = sorted(next(iter(bias.values())).keys())
    x = np.arange(len(arms))
    width = 0.3
    plt.figure(figsize=(7, 4))
    for idx, est in enumerate(estimators):
        offsets = (idx - (len(estimators) - 1) / 2) * width
        values = [bias[est][arm] for arm in arms]
        plt.bar(x + offsets, values, width=width, label=est.replace("_", " "))
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xticks(x, arms)
    plt.ylabel("bias (estimate - true mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    seeds = list(range(50))
    horizons = [1_000, 10_000]
    factories = policy_factories()

    all_results: Dict[str, Dict[int, Dict[str, np.ndarray | float]]] = {}
    for name, factory in factories.items():
        metrics = run_policy_trials(
            bandit_factory=make_bandit,
            policy_factory=factory,
            horizons=horizons,
            seeds=seeds,
        )
        all_results[name] = metrics
        for horizon, payload in metrics.items():
            save_json(
                os.path.join(RESULTS_DIR, f"{name}_H{horizon}.json"),
                {k: v for k, v in payload.items()},
            )
            print(
                f"[exp3] policy={name} horizon={horizon} mean regret = "
                f"{payload['mean_regret']:.2f} ± {payload['ci_regret']:.2f}"
            )

    plot_horizon = max(horizons)
    mean_curves = {name: np.asarray(results[plot_horizon]["mean_curve"]) for name, results in all_results.items()}
    ci_curves = {name: np.asarray(results[plot_horizon]["ci_curve"]) for name, results in all_results.items()}
    plot_regret(
        horizon=plot_horizon,
        mean_curves=mean_curves,
        ci_curves=ci_curves,
        output_path=os.path.join(RESULTS_DIR, f"regret_H{plot_horizon}.png"),
    )

    bias = compute_observational_bias()
    save_json(os.path.join(RESULTS_DIR, "observational_bias.json"), bias)
    plot_bias(bias, os.path.join(RESULTS_DIR, "observational_bias.png"))


if __name__ == "__main__":
    main()
