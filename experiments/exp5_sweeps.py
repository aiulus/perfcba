from __future__ import annotations

import argparse
import math
import os
from typing import Any, Callable, Dict, List, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..Algorithm import BasePolicy
from ..Bandit import (
    DriftingGaussianBandit,
    GaussianBandit,
    LinearBandit,
    NonlinearBandit,
    SCMBandit,
    StudentTBandit,
)
from ..SCM import Intervention, SCM
from ..backdoor_bandits import BackdoorTS, BackdoorUCB
from ..classical_bandits import (
    EpsilonGreedy,
    ExploreThenCommit,
    GaussianThompsonPolicy,
    RandomPolicy,
    UCB,
)
from ..estimators.models import MultinomialLogisticRegression, RidgeOutcomeRegressor
from ..estimators.robust import dr_crossfit, ess, ipw_estimate
from ..linear_bandits import LinThompsonSampling, LinUCB
from .common import ensure_dir, save_json, sweep_1d


DEFAULT_RESULTS_DIR = os.path.join("results", "exp5_sweeps")
ensure_dir(DEFAULT_RESULTS_DIR)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _ordered_labels(values: Mapping[str, Any]) -> List[str]:
    items = list(values.items())
    try:
        items.sort(key=lambda kv: kv[1])
    except TypeError:
        pass
    return [label for label, _ in items]


def _plot_sweep_metric(
    *,
    summary: Mapping[str, Mapping[str, Mapping[str, float]]],
    meta: Mapping[str, Any],
    metric_key: str,
    ci_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    path: str,
) -> None:
    if not summary:
        return
    values = meta.get("values", {})
    labels = _ordered_labels(values)
    if not labels:
        return
    xs = np.asarray([values[label] for label in labels], dtype=float)

    plt.figure(figsize=(7, 4))
    plotted = False
    for name, records in summary.items():
        series = []
        cis = []
        missing = False
        for label in labels:
            metrics = records.get(label)
            if metrics is None or metric_key not in metrics or ci_key not in metrics:
                missing = True
                break
            series.append(metrics[metric_key])
            cis.append(metrics[ci_key])
        if missing:
            continue
        series_arr = np.asarray(series, dtype=float)
        if np.all(np.isnan(series_arr)):
            continue
        cis_arr = np.asarray(cis, dtype=float)
        plt.plot(xs, series_arr, label=name)
        plt.fill_between(xs, series_arr - cis_arr, series_arr + cis_arr, alpha=0.18)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_sweep_curves(
    *,
    summary: Mapping[str, Mapping[str, Mapping[str, float]]],
    meta: Mapping[str, Any],
    xlabel: str,
    title: str,
    path: str,
) -> None:
    _plot_sweep_metric(
        summary=summary,
        meta=meta,
        metric_key="mean_regret",
        ci_key="ci_regret",
        xlabel=xlabel,
        ylabel="mean total regret",
        title=title,
        path=path,
    )
    base, ext = os.path.splitext(path)
    ext = ext if ext else ".png"
    _plot_sweep_metric(
        summary=summary,
        meta=meta,
        metric_key="mean_simple_regret",
        ci_key="ci_simple_regret",
        xlabel=xlabel,
        ylabel="mean simple regret",
        title=f"{title} (simple regret)",
        path=f"{base}_simple_regret{ext}",
    )
    _plot_sweep_metric(
        summary=summary,
        meta=meta,
        metric_key="mean_time_to_epsilon",
        ci_key="ci_time_to_epsilon",
        xlabel=xlabel,
        ylabel="mean time to epsilon",
        title=f"{title} (time to epsilon)",
        path=f"{base}_time_to_epsilon{ext}",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        value = np.asarray(value).item()
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def meta_to_json(meta: Mapping[str, Any]) -> Dict[str, Any]:
    values = meta.get("values", {})
    clean = {label: _json_safe(val) for label, val in values.items()}
    return {"property": meta.get("property"), "values": clean}


def log_sweep_results(tag: str, summary: Mapping[str, Mapping[str, Mapping[str, float]]], meta: Mapping[str, Any]) -> None:
    prop = meta.get("property", "value")
    values = meta.get("values", {})
    for name, records in summary.items():
        for label, metrics in records.items():
            value = values.get(label, label)
            mean = metrics["mean_regret"]
            ci = metrics["ci_regret"]
            print(f"[{tag}] policy={name} {prop}={value} mean regret = {mean:.2f} ± {ci:.2f}")


# ---------------------------------------------------------------------------
# Unstructured bandit sweeps
# ---------------------------------------------------------------------------


UNSTRUCTURED_BASE_MEANS = np.array([0.0, 0.2, 0.1, -0.1, 0.05], dtype=float)
UNSTRUCTURED_BASE_SIGMA = 0.3


def _hardness_from_means(means: np.ndarray) -> float:
    best = float(np.max(means))
    gaps = best - means
    mask = gaps > 1e-12
    return float(np.sum(1.0 / np.square(gaps[mask]))) if np.any(mask) else 0.0


def _means_for_delta(delta: float, k: int) -> np.ndarray:
    means = np.full(k, -float(delta), dtype=float)
    means[0] = 0.0
    if k > 1:
        means[1] = -2.0 * float(delta)
    return means


def _means_for_k(k: int, delta: float) -> np.ndarray:
    return _means_for_delta(delta, int(k))


def _drift_direction(k: int) -> np.ndarray:
    if k <= 1:
        return np.zeros(k, dtype=float)
    direction = np.linspace(-1.0, 1.0, k)
    norm = np.linalg.norm(direction, ord=np.inf)
    if norm <= 0:
        return direction
    return direction / norm


def _unstructured_policy_builders(
    get_n_arms: Callable[[Any], int],
    sigma_fn: Callable[[Any], float],
) -> Dict[str, Callable[[Any], Callable[[], BasePolicy]]]:
    def make_thompson(value: Any) -> Callable[[], BasePolicy]:
        def _factory() -> BasePolicy:
            n = int(get_n_arms(value))
            sigma = float(sigma_fn(value))
            prior_means = np.zeros(n, dtype=float)
            prior_vars = np.ones(n, dtype=float)
            return GaussianThompsonPolicy(prior_means, prior_vars, sigma ** 2)

        return _factory

    return {
        "random": lambda value: (lambda: RandomPolicy(seed=12345)),
        "etc": lambda value: (lambda: ExploreThenCommit(tau=0.2)),
        "epsilon-greedy": lambda value: (lambda: EpsilonGreedy(epsilon=0.1)),
        "ucb": lambda value: (lambda: UCB(schedule="ucb1_alpha", alpha=2.0)),
        "thompson-gaussian": make_thompson,
    }


def run_unstructured(args: argparse.Namespace) -> None:
    ensure_dir(args.results_dir)
    seeds = args.seeds
    horizon = args.horizon

    if "noise" in args.sweeps:
        builders = _unstructured_policy_builders(lambda _: UNSTRUCTURED_BASE_MEANS.size, lambda sigma: float(sigma))

        def bandit_factory_for_sigma(val: float) -> Callable[[], GaussianBandit]:
            means = UNSTRUCTURED_BASE_MEANS
            return lambda: GaussianBandit(means, sigma=float(val))

        summary, meta = sweep_1d(
            property_name="sigma",
            values=args.noise_levels,
            bandit_factory_for=bandit_factory_for_sigma,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
        )
        derived = {
            label: {"sigma": float(value)}
            for label, value in meta["values"].items()
        }
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "unstructured_noise.json"), payload)
        log_sweep_results("unstructured-noise", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="reward noise σ",
            title=f"Noise sensitivity (T={horizon})",
            path=os.path.join(args.results_dir, "unstructured_noise.png"),
        )

    if "delta" in args.sweeps:
        k = UNSTRUCTURED_BASE_MEANS.size
        builders = _unstructured_policy_builders(lambda _: k, lambda _: UNSTRUCTURED_BASE_SIGMA)

        def bandit_factory_for_delta(val: float) -> Callable[[], GaussianBandit]:
            means = _means_for_delta(val, k)
            return lambda: GaussianBandit(means, sigma=UNSTRUCTURED_BASE_SIGMA)

        summary, meta = sweep_1d(
            property_name="delta_min",
            values=args.delta_levels,
            bandit_factory_for=bandit_factory_for_delta,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
        )
        derived = {}
        for label, value in meta["values"].items():
            means = _means_for_delta(float(value), k)
            derived[label] = {
                "delta_min": float(value),
                "hardness": _hardness_from_means(means),
            }
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "unstructured_delta.json"), payload)
        log_sweep_results("unstructured-delta", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="Δ_min",
            title=f"Gap profile sensitivity (T={horizon})",
            path=os.path.join(args.results_dir, "unstructured_delta.png"),
        )

    if "arms" in args.sweeps:
        builders = _unstructured_policy_builders(lambda k: int(k), lambda _: UNSTRUCTURED_BASE_SIGMA)

        def bandit_factory_for_k(val: int) -> Callable[[], GaussianBandit]:
            means = _means_for_k(int(val), args.k_delta)
            return lambda: GaussianBandit(means, sigma=UNSTRUCTURED_BASE_SIGMA)

        summary, meta = sweep_1d(
            property_name="n_arms",
            values=args.k_levels,
            bandit_factory_for=bandit_factory_for_k,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
        )
        derived = {}
        for label, value in meta["values"].items():
            means = _means_for_k(int(value), args.k_delta)
            derived[label] = {
                "n_arms": int(value),
                "hardness": _hardness_from_means(means),
            }
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "unstructured_arms.json"), payload)
        log_sweep_results("unstructured-arms", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="number of arms",
            title=f"Scaling with arms (Δ_min={args.k_delta})",
            path=os.path.join(args.results_dir, "unstructured_arms.png"),
        )

    if "student_t" in args.sweeps:
        builders = _unstructured_policy_builders(lambda _: UNSTRUCTURED_BASE_MEANS.size, lambda _: UNSTRUCTURED_BASE_SIGMA)

        def bandit_factory_for_nu(val: float) -> Callable[[], GaussianBandit]:
            if np.isinf(val):
                return lambda: GaussianBandit(UNSTRUCTURED_BASE_MEANS, sigma=UNSTRUCTURED_BASE_SIGMA)
            return lambda: StudentTBandit(UNSTRUCTURED_BASE_MEANS, sigma=UNSTRUCTURED_BASE_SIGMA, nu=int(val))

        summary, meta = sweep_1d(
            property_name="nu",
            values=args.nu_levels,
            bandit_factory_for=bandit_factory_for_nu,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
        )
        derived = {
            label: {"nu": ("inf" if np.isinf(value) else float(value))}
            for label, value in meta["values"].items()
        }
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "unstructured_studentt.json"), payload)
        log_sweep_results("unstructured-studentt", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="degrees of freedom ν",
            title=f"Heavy-tailed noise (σ={UNSTRUCTURED_BASE_SIGMA})",
            path=os.path.join(args.results_dir, "unstructured_studentt.png"),
        )

    if "drift" in args.sweeps:
        k = UNSTRUCTURED_BASE_MEANS.size
        direction = _drift_direction(k)
        builders = _unstructured_policy_builders(lambda _: k, lambda _: UNSTRUCTURED_BASE_SIGMA)

        def bandit_factory_for_drift(val: float) -> Callable[[], DriftingGaussianBandit]:
            drift = direction * float(val)
            return lambda: DriftingGaussianBandit(
                UNSTRUCTURED_BASE_MEANS,
                drift=drift,
                sigma=UNSTRUCTURED_BASE_SIGMA,
            )

        summary, meta = sweep_1d(
            property_name="drift",
            values=args.drift_levels,
            bandit_factory_for=bandit_factory_for_drift,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
        )
        derived = {
            label: {
                "drift_linf": float(np.max(np.abs(direction * float(value)))),
                "drift_l2": float(np.linalg.norm(direction * float(value))),
            }
            for label, value in meta["values"].items()
        }
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "unstructured_drift.json"), payload)
        log_sweep_results("unstructured-drift", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="‖η‖∞",
            title=f"Nonstationarity sensitivity (σ={UNSTRUCTURED_BASE_SIGMA})",
            path=os.path.join(args.results_dir, "unstructured_drift.png"),
        )


# ---------------------------------------------------------------------------
# Linear bandit sweeps
# ---------------------------------------------------------------------------


LINEAR_BASE_K = 8
LINEAR_BASE_D = 4
LINEAR_BASE_SIGMA = 0.2


def _sample_features(k: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(k, d))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return X / norms


def _features_with_condition(k: int, d: int, kappa: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(k, d)))
    V, _ = np.linalg.qr(rng.normal(size=(d, d)))
    if kappa <= 1.0:
        s = np.ones(d, dtype=float)
    else:
        s = np.geomspace(1.0, 1.0 / float(kappa), num=d)
    X = U @ np.diag(s) @ V.T
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return X / norms


def _theta_direction(d: int) -> np.ndarray:
    vec = np.zeros(d, dtype=float)
    vec[0] = 1.0
    return vec


def _linear_policy_builders(
    feature_lookup: Mapping[Any, np.ndarray],
    sigma_lookup: Callable[[Any], float],
    expose_lookup: Callable[[Any], bool],
) -> Dict[str, Callable[[Any], Callable[[], BasePolicy]]]:
    def make_lin_ucb(value: Any) -> Callable[[], BasePolicy]:
        X = feature_lookup[value]
        if not expose_lookup(value):
            return lambda: UCB(schedule="ucb1_alpha", alpha=2.0)
        return lambda: LinUCB(X, alpha=1.5, lam=1.0)

    def make_lin_ts(value: Any) -> Callable[[], BasePolicy]:
        X = feature_lookup[value]
        sigma = sigma_lookup(value)
        if not expose_lookup(value):
            n = X.shape[0]
            return lambda: GaussianThompsonPolicy(np.zeros(n, dtype=float), np.ones(n, dtype=float), float(sigma ** 2))
        return lambda: LinThompsonSampling(X, sigma2=float(sigma ** 2), lam=1.0)

    def make_ucb(value: Any) -> Callable[[], BasePolicy]:
        expose = expose_lookup(value)
        if expose:
            return lambda: UCB(schedule="ucb1_alpha", alpha=2.0)

        # Without contexts, fall back to the same UCB policy.
        return lambda: UCB(schedule="ucb1_alpha", alpha=2.0)

    return {
        "ucb": make_ucb,
        "lin-ucb": make_lin_ucb,
        "lin-thompson": make_lin_ts,
    }


def run_linear(args: argparse.Namespace) -> None:
    ensure_dir(args.results_dir)
    seeds = args.seeds
    horizon = args.horizon

    base_features = _sample_features(LINEAR_BASE_K, LINEAR_BASE_D, args.feature_seed)
    theta_dir = _theta_direction(LINEAR_BASE_D)

    if "sigma" in args.sweeps:
        feature_lookup = {value: base_features for value in args.sigma_levels}
        sigma_lookup = lambda value: float(value)
        expose_lookup = lambda value: True

        def bandit_factory_for_sigma(val: float) -> Callable[[], LinearBandit]:
            theta = theta_dir * args.signal
            return lambda: LinearBandit(base_features, theta, sigma=float(val), provide_features=True)

        builders = _linear_policy_builders(feature_lookup, sigma_lookup, expose_lookup)
        summary, meta = sweep_1d(
            property_name="sigma",
            values=args.sigma_levels,
            bandit_factory_for=bandit_factory_for_sigma,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
        )
        derived = {label: {"sigma": float(value)} for label, value in meta["values"].items()}
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "linear_sigma.json"), payload)
        log_sweep_results("linear-sigma", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="noise σ",
            title=f"Linear bandit noise sensitivity (T={horizon})",
            path=os.path.join(args.results_dir, "linear_sigma.png"),
        )

    if "dimension" in args.sweeps:
        feature_lookup = {}
        for d in args.dim_levels:
            feature_lookup[d] = _sample_features(LINEAR_BASE_K, int(d), args.feature_seed + int(d))
        sigma_lookup = lambda _: LINEAR_BASE_SIGMA
        expose_lookup = lambda _: True

        def bandit_factory_for_dim(val: int) -> Callable[[], LinearBandit]:
            d = int(val)
            theta = _theta_direction(d) * args.signal
            X = feature_lookup[val]
            return lambda: LinearBandit(X, theta, sigma=LINEAR_BASE_SIGMA, provide_features=True)

        builders = _linear_policy_builders(feature_lookup, sigma_lookup, expose_lookup)
        summary, meta = sweep_1d(
            property_name="dimension",
            values=args.dim_levels,
            bandit_factory_for=bandit_factory_for_dim,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
        )
        derived = {}
        for label, value in meta["values"].items():
            d = int(value)
            X = feature_lookup[value]
            gram = X.T @ X
            cond = float(np.linalg.cond(gram)) if gram.size else 1.0
            derived[label] = {"dimension": d, "condition_number": cond}
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "linear_dimension.json"), payload)
        log_sweep_results("linear-dimension", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="dimension d",
            title=f"Dimensional scaling (σ={LINEAR_BASE_SIGMA})",
            path=os.path.join(args.results_dir, "linear_dimension.png"),
        )

    if "conditioning" in args.sweeps:
        feature_lookup = {}
        for level in args.condition_levels:
            feature_lookup[level] = _features_with_condition(LINEAR_BASE_K, LINEAR_BASE_D, float(level), args.feature_seed + int(level * 10))
        sigma_lookup = lambda _: LINEAR_BASE_SIGMA
        expose_lookup = lambda _: True
        theta = theta_dir * args.signal

        def bandit_factory_for_condition(val: float) -> Callable[[], LinearBandit]:
            X = feature_lookup[val]
            return lambda: LinearBandit(X, theta, sigma=LINEAR_BASE_SIGMA, provide_features=True)

        builders = _linear_policy_builders(feature_lookup, sigma_lookup, expose_lookup)
        summary, meta = sweep_1d(
            property_name="kappa",
            values=args.condition_levels,
            bandit_factory_for=bandit_factory_for_condition,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
        )
        derived = {}
        for label, value in meta["values"].items():
            X = feature_lookup[value]
            gram = X.T @ X
            cond = float(np.linalg.cond(gram)) if gram.size else 1.0
            derived[label] = {"kappa": float(value), "condition_number": cond}
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "linear_conditioning.json"), payload)
        log_sweep_results("linear-conditioning", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="conditioning κ",
            title="Feature conditioning sweep",
            path=os.path.join(args.results_dir, "linear_conditioning.png"),
        )

    if "signal" in args.sweeps:
        feature_lookup = {value: base_features for value in args.signal_levels}
        sigma_lookup = lambda _: LINEAR_BASE_SIGMA
        expose_lookup = lambda _: True

        def bandit_factory_for_signal(val: float) -> Callable[[], LinearBandit]:
            theta = theta_dir * float(val)
            return lambda: LinearBandit(base_features, theta, sigma=LINEAR_BASE_SIGMA, provide_features=True)

        builders = _linear_policy_builders(feature_lookup, sigma_lookup, expose_lookup)
        summary, meta = sweep_1d(
            property_name="signal_norm",
            values=args.signal_levels,
            bandit_factory_for=bandit_factory_for_signal,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
        )
        derived = {label: {"signal_norm": float(value)} for label, value in meta["values"].items()}
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "linear_signal.json"), payload)
        log_sweep_results("linear-signal", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="‖θ*‖₂",
            title="Signal strength sweep",
            path=os.path.join(args.results_dir, "linear_signal.png"),
        )

    if "misspec" in args.sweeps:
        feature_lookup = {value: base_features for value in args.misspec_levels}
        sigma_lookup = lambda _: LINEAR_BASE_SIGMA
        expose_lookup = lambda _: True

        quad_dir = np.concatenate([np.zeros(base_features.shape[1]), np.ones(base_features.shape[1]) / base_features.shape[1]])

        def bandit_factory_for_misspec(val: float) -> Callable[[], NonlinearBandit]:
            theta_lin = theta_dir * args.signal
            theta_quad = quad_dir * float(val)
            return lambda: NonlinearBandit(
                base_features,
                theta_linear=theta_lin,
                theta_quadratic=theta_quad,
                sigma=LINEAR_BASE_SIGMA,
                provide_features=True,
            )

        builders = _linear_policy_builders(feature_lookup, sigma_lookup, expose_lookup)
        summary, meta = sweep_1d(
            property_name="beta",
            values=args.misspec_levels,
            bandit_factory_for=bandit_factory_for_misspec,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
        )
        derived = {label: {"beta": float(value)} for label, value in meta["values"].items()}
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "linear_misspec.json"), payload)
        log_sweep_results("linear-misspec", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="nonlinearity β",
            title="Misspecification sweep",
            path=os.path.join(args.results_dir, "linear_misspec.png"),
        )

    if "context" in args.sweeps:
        values = [True, False] if args.context_levels is None else args.context_levels
        feature_lookup = {value: base_features for value in values}
        sigma_lookup = lambda _: LINEAR_BASE_SIGMA
        expose_lookup = lambda value: bool(value)
        theta = theta_dir * args.signal

        def bandit_factory_for_context(val: bool) -> Callable[[], LinearBandit]:
            return lambda: LinearBandit(
                base_features,
                theta,
                sigma=LINEAR_BASE_SIGMA,
                provide_features=bool(val),
            )

        builders = _linear_policy_builders(feature_lookup, sigma_lookup, expose_lookup)
        summary, meta = sweep_1d(
            property_name="expose_features",
            values=values,
            bandit_factory_for=bandit_factory_for_context,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
            label_fn=lambda v: "true" if v else "false",
        )
        derived = {label: {"expose_features": bool(value)} for label, value in meta["values"].items()}
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "linear_context.json"), payload)
        log_sweep_results("linear-context", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="expose contexts?",
            title="Context ablation",
            path=os.path.join(args.results_dir, "linear_context.png"),
        )


# ---------------------------------------------------------------------------
# Causal bandit sweeps
# ---------------------------------------------------------------------------


def _build_parametric_scm(
    *,
    b: int,
    conf_t: float,
    conf_y: float,
    tau: float,
    sigma_z: float,
    sigma_t: float,
    sigma_y: float,
    nonlinear_eta: float,
    latent_t: float,
    latent_y: float,
    seed: int,
) -> SCM:
    rng = np.random.default_rng(seed)
    nodes = [f"Z{i}" for i in range(b)] + ["T", "Y"]
    parents = {name: [] for name in nodes}
    for name in nodes:
        if name.startswith("Z"):
            continue
        if name == "T":
            parents[name] = [f"Z{i}" for i in range(b)]
        elif name == "Y":
            parents[name] = ["T"] + [f"Z{i}" for i in range(b)]

    weight_vec = np.ones(b, dtype=float) / max(1, b)
    beta_t = weight_vec * conf_t * b
    beta_y = weight_vec * conf_y * b

    def f_z(_parents: Dict[str, Any], local_rng: np.random.Generator) -> float:
        return float(local_rng.normal(0.0, sigma_z))

    def f_t(par: Dict[str, Any], local_rng: np.random.Generator) -> float:
        z = np.array([par[f"Z{i}"] for i in range(b)], dtype=float)
        noise = local_rng.normal(0.0, sigma_t)
        latent = local_rng.normal(0.0, latent_t) if latent_t > 0 else 0.0
        logits = beta_t @ z + tau * float(np.sum(z)) + noise + latent
        prob = 1.0 / (1.0 + np.exp(-logits))
        return float(local_rng.random() < prob)

    def f_y(par: Dict[str, Any], local_rng: np.random.Generator) -> float:
        z = np.array([par[f"Z{i}"] for i in range(b)], dtype=float)
        t = float(par["T"])
        latent = local_rng.normal(0.0, latent_y) if latent_y > 0 else 0.0
        noise = local_rng.normal(0.0, sigma_y)
        interaction = nonlinear_eta * t * float(np.sum(z))
        base = 1.0 * t + beta_y @ z
        return float(base + interaction + latent + noise)

    mechanisms = {f"Z{i}": f_z for i in range(b)}
    mechanisms["T"] = f_t
    mechanisms["Y"] = f_y
    return SCM(nodes=nodes, parents=parents, f=mechanisms)


def _causal_policy_builders(refit_every: int, clip: float, sigma_y: float) -> Dict[str, Callable[[Any], Callable[[], BasePolicy]]]:
    def make_thompson(_: Any) -> Callable[[], BasePolicy]:
        prior_means = [0.0, 0.0]
        prior_vars = [1.0, 1.0]
        return lambda: GaussianThompsonPolicy(prior_means, prior_vars, sigma_y ** 2)

    return {
        "ucb": lambda _: (lambda: UCB(schedule="ucb1_alpha", alpha=2.0)),
        "thompson-gaussian": make_thompson,
        "backdoor-ucb": lambda _: (
            lambda: BackdoorUCB(n_arms=2, refit_every=refit_every, clip=clip, alpha=1.0)
        ),
        "backdoor-ts": lambda _: (
            lambda: BackdoorTS(
                n_arms=2,
                refit_every=refit_every,
                clip=clip,
                variance_scale=1.0,
                min_samples=10,
            )
        ),
    }


def _causal_bandit_factory(params: Dict[str, Any]) -> Callable[[], SCMBandit]:
    def _factory() -> SCMBandit:
        scm = _build_parametric_scm(**params)
        interventions = [
            Intervention("do(T=0)", hard={"T": 0.0}),
            Intervention("do(T=1)", hard={"T": 1.0}),
        ]
        return SCMBandit(
            scm=scm,
            interventions=interventions,
            reward_node="Y",
            observe="parents",
            feedback="causal",
        )

    return _factory


def _observational_summary(scm: SCM, n_samples: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    b = sum(1 for name in scm.nodes if name.startswith("Z"))
    Z = np.zeros((n_samples, b), dtype=float)
    T = np.zeros(n_samples, dtype=int)
    Y = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        sample = scm.sample(rng)
        Z[i] = np.array([sample[f"Z{j}"] for j in range(b)], dtype=float)
        T[i] = int(sample["T"])
        Y[i] = float(sample["Y"])

    prop_model = MultinomialLogisticRegression(n_classes=2, random_state=0)
    prop_model.fit(Z, T)
    probs = prop_model.predict_proba(Z)
    kappa = float(np.min(probs))

    def propensity_fn(arm: int) -> np.ndarray:
        return probs[:, arm]

    weights = {}
    ess_per_arm = {}
    for arm in (0, 1):
        _, w = ipw_estimate(Y, T, Z, arm, propensity_fn, clip=None)
        weights[arm] = w
        ess_per_arm[arm] = ess(w)

    outcome = RidgeOutcomeRegressor(l2=1e-2)
    dr_mean, _, dr_vals = dr_crossfit(Y, T, Z, 1, outcome, prop_model, K=2, clip=10.0, random_state=0)
    return {
        "kappa": kappa,
        "ess": ess_per_arm,
        "dr_mean_arm1": float(dr_mean),
        "dr_var": float(np.var(dr_vals)) if dr_vals.size else 0.0,
    }


def run_causal(args: argparse.Namespace) -> None:
    ensure_dir(args.results_dir)
    seeds = args.seeds
    horizon = args.horizon
    base_params = dict(
        b=args.backdoor_size,
        conf_t=args.conf_t,
        conf_y=args.conf_y,
        tau=args.tau,
        sigma_z=args.sigma_z,
        sigma_t=args.sigma_t,
        sigma_y=args.sigma_y,
        nonlinear_eta=args.nonlinear_eta,
        latent_t=args.latent_t,
        latent_y=args.latent_y,
        seed=args.scm_seed,
    )

    builders = _causal_policy_builders(args.refit_every, args.clip, args.sigma_y)

    if "confounding" in args.sweeps:
        def bandit_factory_for_conf(val: float) -> Callable[[], SCMBandit]:
            params = dict(base_params)
            params["conf_t"] = float(val)
            params["conf_y"] = float(val)
            return _causal_bandit_factory(params)

        summary, meta = sweep_1d(
            property_name="confounding",
            values=args.conf_levels,
            bandit_factory_for=bandit_factory_for_conf,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
        )
        derived = {}
        for label, value in meta["values"].items():
            params = dict(base_params)
            params["conf_t"] = float(value)
            params["conf_y"] = float(value)
            scm = _build_parametric_scm(**params)
            derived[label] = {
                "conf_t": params["conf_t"],
                "conf_y": params["conf_y"],
                **_observational_summary(scm, args.obs_samples, args.obs_seed),
            }
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "causal_confounding.json"), payload)
        log_sweep_results("causal-confounding", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="confounding strength",
            title="Causal confounding sweep",
            path=os.path.join(args.results_dir, "causal_confounding.png"),
        )

    if "overlap" in args.sweeps:
        def bandit_factory_for_tau(val: float) -> Callable[[], SCMBandit]:
            params = dict(base_params)
            params["tau"] = float(val)
            return _causal_bandit_factory(params)

        summary, meta = sweep_1d(
            property_name="tau",
            values=args.tau_levels,
            bandit_factory_for=bandit_factory_for_tau,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
        )
        derived = {}
        for label, value in meta["values"].items():
            params = dict(base_params)
            params["tau"] = float(value)
            scm = _build_parametric_scm(**params)
            derived[label] = {"tau": params["tau"], **_observational_summary(scm, args.obs_samples, args.obs_seed)}
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "causal_overlap.json"), payload)
        log_sweep_results("causal-overlap", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="propensity slope τ",
            title="Positivity / overlap sweep",
            path=os.path.join(args.results_dir, "causal_overlap.png"),
        )

    if "backdoor" in args.sweeps:
        def bandit_factory_for_b(val: int) -> Callable[[], SCMBandit]:
            params = dict(base_params)
            params["b"] = int(val)
            return _causal_bandit_factory(params)

        summary, meta = sweep_1d(
            property_name="backdoor_size",
            values=args.backdoor_levels,
            bandit_factory_for=bandit_factory_for_b,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
        )
        derived = {}
        for label, value in meta["values"].items():
            params = dict(base_params)
            params["b"] = int(value)
            scm = _build_parametric_scm(**params)
            derived[label] = {"backdoor_size": params["b"], **_observational_summary(scm, args.obs_samples, args.obs_seed)}
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "causal_backdoor.json"), payload)
        log_sweep_results("causal-backdoor", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="|Z|",
            title="Back-door set size sweep",
            path=os.path.join(args.results_dir, "causal_backdoor.png"),
        )

    if "noise" in args.sweeps:
        def bandit_factory_for_sigma_y(val: float) -> Callable[[], SCMBandit]:
            params = dict(base_params)
            params["sigma_y"] = float(val)
            return _causal_bandit_factory(params)

        summary, meta = sweep_1d(
            property_name="sigma_y",
            values=args.outcome_noise_levels,
            bandit_factory_for=bandit_factory_for_sigma_y,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
        )
        derived = {label: {"sigma_y": float(value)} for label, value in meta["values"].items()}
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "causal_noise.json"), payload)
        log_sweep_results("causal-noise", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="σ_Y",
            title="Outcome noise sweep",
            path=os.path.join(args.results_dir, "causal_noise.png"),
        )

    if "misspec" in args.sweeps:
        def bandit_factory_for_eta(val: float) -> Callable[[], SCMBandit]:
            params = dict(base_params)
            params["nonlinear_eta"] = float(val)
            return _causal_bandit_factory(params)

        summary, meta = sweep_1d(
            property_name="eta",
            values=args.eta_levels,
            bandit_factory_for=bandit_factory_for_eta,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
        )
        derived = {label: {"eta": float(value)} for label, value in meta["values"].items()}
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "causal_misspec.json"), payload)
        log_sweep_results("causal-misspec", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="interaction η",
            title="Mechanism misspecification sweep",
            path=os.path.join(args.results_dir, "causal_misspec.png"),
        )

    if "latent" in args.sweeps:
        def bandit_factory_for_latent(val: float) -> Callable[[], SCMBandit]:
            params = dict(base_params)
            params["latent_t"] = float(val)
            params["latent_y"] = float(val)
            return _causal_bandit_factory(params)

        summary, meta = sweep_1d(
            property_name="latent_strength",
            values=args.latent_levels,
            bandit_factory_for=bandit_factory_for_latent,
            policy_builders=builders,
            horizon=horizon,
            seeds=seeds,
        )
        derived = {label: {"latent_strength": float(value)} for label, value in meta["values"].items()}
        payload = {"summary": summary, "meta": meta_to_json(meta), "derived": derived}
        save_json(os.path.join(args.results_dir, "causal_latent.json"), payload)
        log_sweep_results("causal-latent", summary, meta)
        plot_sweep_curves(
            summary=summary,
            meta=meta,
            xlabel="latent strength",
            title="Latent confounder sweep",
            path=os.path.join(args.results_dir, "causal_latent.png"),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 5: structured sweeps across bandit families")
    subparsers = parser.add_subparsers(dest="family", required=True)

    # Unstructured bandits
    u = subparsers.add_parser("unstructured", help="Sweeps for unstructured Gaussian bandits")
    u.add_argument("--sweeps", nargs="+", default=["noise", "delta", "arms", "student_t", "drift"], help="Which sweeps to run")
    u.add_argument("--noise-levels", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4, 0.8])
    u.add_argument("--delta-levels", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.4])
    u.add_argument("--k-levels", type=int, nargs="+", default=[5, 10, 20, 40])
    u.add_argument("--k-delta", type=float, default=0.1, help="Δ_min used when varying K")
    u.add_argument("--nu-levels", type=float, nargs="+", default=[3, 5, 8, np.inf])
    u.add_argument("--drift-levels", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    u.add_argument("--horizon", type=int, default=10_000)
    u.add_argument("--seeds", type=int, nargs="+", default=list(range(32)))
    u.add_argument("--results-dir", type=str, default=os.path.join(DEFAULT_RESULTS_DIR, "unstructured"))

    # Linear bandits
    l = subparsers.add_parser("linear", help="Sweeps for linear bandits")
    l.add_argument("--sweeps", nargs="+", default=["sigma", "dimension", "conditioning", "signal", "misspec", "context"], help="Which sweeps to run")
    l.add_argument("--sigma-levels", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4, 0.8])
    l.add_argument("--dim-levels", type=int, nargs="+", default=[2, 4, 8, 16])
    l.add_argument("--condition-levels", type=float, nargs="+", default=[1, 3, 10, 30, 100])
    l.add_argument("--signal-levels", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    l.add_argument("--misspec-levels", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    l.add_argument("--context-levels", type=lambda v: v.lower() == "true", nargs="*", default=None)
    l.add_argument("--signal", type=float, default=1.0, help="Baseline signal norm for linear sweeps")
    l.add_argument("--feature-seed", type=int, default=0)
    l.add_argument("--horizon", type=int, default=10_000)
    l.add_argument("--seeds", type=int, nargs="+", default=list(range(32)))
    l.add_argument("--results-dir", type=str, default=os.path.join(DEFAULT_RESULTS_DIR, "linear"))

    # Causal bandits
    c = subparsers.add_parser("causal", help="Sweeps for causal bandits")
    c.add_argument("--sweeps", nargs="+", default=["confounding", "overlap", "backdoor", "noise", "misspec", "latent"], help="Which sweeps to run")
    c.add_argument("--conf-levels", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8])
    c.add_argument("--tau-levels", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0, 4.0])
    c.add_argument("--backdoor-levels", type=int, nargs="+", default=[1, 2, 4, 8])
    c.add_argument("--outcome-noise-levels", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.8])
    c.add_argument("--eta-levels", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    c.add_argument("--latent-levels", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    c.add_argument("--backdoor-size", type=int, default=2)
    c.add_argument("--conf-t", type=float, default=0.4)
    c.add_argument("--conf-y", type=float, default=0.4)
    c.add_argument("--tau", type=float, default=1.0)
    c.add_argument("--sigma-z", type=float, default=1.0)
    c.add_argument("--sigma-t", type=float, default=0.2)
    c.add_argument("--sigma-y", type=float, default=0.3)
    c.add_argument("--nonlinear-eta", type=float, default=0.0)
    c.add_argument("--latent-t", type=float, default=0.0)
    c.add_argument("--latent-y", type=float, default=0.0)
    c.add_argument("--refit-every", type=int, default=25)
    c.add_argument("--clip", type=float, default=10.0)
    c.add_argument("--horizon", type=int, default=10_000)
    c.add_argument("--seeds", type=int, nargs="+", default=list(range(32)))
    c.add_argument("--obs-samples", type=int, default=5000)
    c.add_argument("--obs-seed", type=int, default=0)
    c.add_argument("--scm-seed", type=int, default=0)
    c.add_argument("--results-dir", type=str, default=os.path.join(DEFAULT_RESULTS_DIR, "causal"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.family == "unstructured":
        run_unstructured(args)
    elif args.family == "linear":
        run_linear(args)
    elif args.family == "causal":
        run_causal(args)
    else:
        raise ValueError(f"Unknown family {args.family}")


if __name__ == "__main__":
    main()
