"""Analysis utilities for tau-study experiments.

This module consumes the JSONL outputs written by ``run_tau_study.py`` and
generates:

* heat maps annotated with best-:math:`\\tau` markers and completion rates,
* gradient flow overlays that highlight the direction of steepest decrease in
  the investigated metric, and
* statistical tests that examine the interaction between graph density and
  :math:`\\tau`.

The implementation intentionally keeps ``run_tau_study.py`` untouched – all
post-processing happens here.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import logging

import matplotlib.pyplot as plt
import numpy as np

try:  # Optional – gradient smoothing and some stats use SciPy when available.
    from scipy import stats as scipy_stats
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - SciPy may not be installed in CI.
    scipy_stats = None
    gaussian_filter = None


LOGGER = logging.getLogger(__name__)

KNOWN_METRICS = ("cumulative_regret", "tto", "optimal_rate", "simple_regret")
METRIC_LABELS = {
    "cumulative_regret": "Cumulative Regret",
    "tto": "Time to Optimality",
    "simple_regret": "Simple Regret",
    "optimal_rate": "Optimal Action Rate",
}

@dataclass
class LoadedResults:
    records: List[Dict[str, Any]]
    tau_values: List[float]
    knob_values: List[float]
    metric_keys: List[str]
    seeds: List[int]


@dataclass
class RegressionOutput:
    coefficients: Dict[str, float]
    std_errors: Dict[str, float]
    t_stats: Dict[str, float]
    p_values: Dict[str, float]
    residual_df: float
    r_squared: float
    cov_type: str
    hypothesis_p_value: Optional[float]


def load_results(path: Path) -> LoadedResults:
    records: List[Dict[str, Any]] = []
    tau_values: set[float] = set()
    knob_values: set[float] = set()
    metric_keys: set[str] = set()
    seeds: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record: Dict[str, Any] = json.loads(line)
            records.append(record)
            tau_values.add(float(record["tau"]))
            knob_values.add(float(record["knob_value"]))
            seeds.add(int(record["seed"]))
            metric_keys.update(
                key
                for key in record
                if key
                not in {
                    "tau",
                    "knob_value",
                    "seed",
                    "instance_id",
                    "density",
                    "horizon",
                    "scheduler",
                    "finished_discovery",
                    "finished_discovery_round",
                }
                and isinstance(record[key], (int, float))
            )
    return LoadedResults(
        records=records,
        tau_values=sorted(tau_values),
        knob_values=sorted(knob_values),
        metric_keys=sorted(metric_keys),
        seeds=sorted(seeds),
    )


def compute_log_density(values: Sequence[float], n: Optional[int]) -> Optional[List[float]]:
    if n is None:
        return None
    log_vals: List[float] = []
    for val in values:
        scaled = max(n * val, 1e-12)
        log_vals.append(math.log(scaled))
    return log_vals


def aggregate_matrix(
    records: Sequence[Mapping[str, Any]],
    tau_values: Sequence[float],
    knob_values: Sequence[float],
    metric: str,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], List[float]], List[float]]:
    tau_index = {tau: idx for idx, tau in enumerate(tau_values)}
    knob_index = {knob: idx for idx, knob in enumerate(knob_values)}
    matrix = np.full((len(tau_values), len(knob_values)), np.nan, dtype=float)
    counts = np.zeros_like(matrix, dtype=int)
    sums = np.zeros_like(matrix, dtype=float)
    per_cell_samples: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    finished_counts = np.zeros(len(knob_values), dtype=int)
    finished_denoms = np.zeros(len(knob_values), dtype=int)

    for record in records:
        if metric not in record:
            continue
        tau_idx = tau_index.get(float(record["tau"]))
        knob_idx = knob_index.get(float(record["knob_value"]))
        if tau_idx is None or knob_idx is None:
            continue
        value = float(record[metric])
        counts[tau_idx, knob_idx] += 1
        sums[tau_idx, knob_idx] += value
        per_cell_samples[(tau_idx, knob_idx)].append(value)
        finished_flag = record.get("finished_discovery")
        if finished_flag is not None:
            finished_counts[knob_idx] += int(bool(finished_flag))
            finished_denoms[knob_idx] += 1

    nonzero = counts > 0
    matrix[nonzero] = sums[nonzero] / counts[nonzero]
    finished_rates: List[float] = []
    for count, denom in zip(finished_counts, finished_denoms):
        finished_rates.append(count / denom if denom else math.nan)
    return matrix, per_cell_samples, finished_rates


def aggregate_metric_series(
    records: Sequence[Mapping[str, Any]],
    knob_values: Sequence[float],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_knob: Dict[float, List[float]] = defaultdict(list)
    for record in records:
        if metric not in record:
            continue
        knob = float(record["knob_value"])
        per_knob[knob].append(float(record[metric]))

    means: List[float] = []
    stds: List[float] = []
    counts: List[int] = []
    for knob in knob_values:
        samples = per_knob.get(float(knob))
        if not samples:
            means.append(math.nan)
            stds.append(math.nan)
            counts.append(0)
            continue
        arr = np.asarray(samples, dtype=float)
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr)))
        counts.append(int(arr.size))
    return np.asarray(means, dtype=float), np.asarray(stds, dtype=float), np.asarray(counts, dtype=int)


def plot_metric_lines(
    knob_values: Sequence[float],
    series: Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    *,
    x_label: str,
    output_path: Path,
) -> None:
    xs = np.asarray(knob_values, dtype=float)
    if xs.size == 0:
        raise ValueError("No knob values available for line plot.")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for metric, means, stds, _counts in series:
        if means.size != xs.size:
            continue
        mask = np.isfinite(means)
        if not mask.any():
            continue
        label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
        ax.plot(xs[mask], means[mask], marker="o", linewidth=2.0, label=label)
        if stds.size == means.size:
            std_vals = stds[mask]
            if std_vals.size and np.isfinite(std_vals).any():
                std_vals = np.where(np.isfinite(std_vals), std_vals, 0.0)
                ax.fill_between(
                    xs[mask],
                    means[mask] - std_vals,
                    means[mask] + std_vals,
                    alpha=0.15,
                )
        plotted = True

    if not plotted:
        plt.close(fig)
        raise ValueError("No valid metric series available for plotting.")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Metric value")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def aggregate_metric_series_by_knob(
    records: Sequence[Mapping[str, Any]],
    tau_values: Sequence[float],
    metric: str,
) -> Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Group per knob value and aggregate mean/std/count across seeds for each tau."""
    tau_index = {tau: idx for idx, tau in enumerate(tau_values)}
    per_knob: Dict[float, List[List[float]]] = {}
    for record in records:
        if metric not in record:
            continue
        tau_idx = tau_index.get(float(record["tau"]))
        if tau_idx is None:
            continue
        knob = float(record["knob_value"])
        if knob not in per_knob:
            per_knob[knob] = [[] for _ in tau_values]
        per_knob[knob][tau_idx].append(float(record[metric]))

    aggregated: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for knob, buckets in per_knob.items():
        means: List[float] = []
        stds: List[float] = []
        counts: List[int] = []
        for bucket in buckets:
            if not bucket:
                means.append(math.nan)
                stds.append(math.nan)
                counts.append(0)
                continue
            arr = np.asarray(bucket, dtype=float)
            means.append(float(np.mean(arr)))
            stds.append(float(np.std(arr)))
            counts.append(int(arr.size))
        aggregated[knob] = (np.asarray(means), np.asarray(stds), np.asarray(counts))
    return aggregated


def plot_lines_by_knob(
    tau_values: Sequence[float],
    knob_series: Mapping[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    vary: str,
    output_path: Path,
    colormap: Optional[str] = None,
) -> None:
    xs = np.asarray(tau_values, dtype=float)
    if xs.size == 0:
        raise ValueError("No tau values available for line plot.")
    knobs_sorted = sorted(knob_series.keys())
    if not knobs_sorted:
        raise ValueError("No knob series available for plotting.")

    cmap_name = colormap
    default_cmaps = {
        "graph_density": "Greens",
        "parent_count": "Blues",
        "arm_variance": "magma",
        "intervention_size": "Oranges",
        "node_count": "cividis",
    }
    if not cmap_name:
        cmap_name = default_cmaps.get(vary, "viridis")
    cmap = plt.get_cmap(cmap_name)

    vmin, vmax = min(knobs_sorted), max(knobs_sorted)
    span = vmax - vmin if vmax != vmin else 1.0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for knob in knobs_sorted:
        means, stds, _counts = knob_series[knob]
        if means.size != xs.size:
            continue
        mask = np.isfinite(means)
        if not mask.any():
            continue
        normed = (knob - vmin) / span
        color = cmap(normed)
        label = f"{vary.replace('_', ' ')}={knob:g}"
        ax.plot(xs[mask], means[mask], marker="o", linewidth=1.8, color=color, label=label)
        if stds.size == means.size:
            std_vals = stds[mask]
            if std_vals.size and np.isfinite(std_vals).any():
                std_vals = np.where(np.isfinite(std_vals), std_vals, 0.0)
                ax.fill_between(
                    xs[mask],
                    means[mask] - std_vals,
                    means[mask] + std_vals,
                    alpha=0.12,
                    color=color,
                )

    ax.set_xlabel("Tau")
    ax.set_ylabel("Metric value")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(ncol=2, fontsize="small", title=vary.replace("_", " ").title())
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def aggregate_env_lines(
    records: Sequence[Mapping[str, Any]],
    knob_values: Sequence[float],
    metric: str,
    *,
    env_key: str,
    tau_filter: Optional[float],
) -> Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Aggregate mean/std/count for metric vs swept knob, grouped by an environment key."""
    per_env: Dict[float, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if metric not in record or env_key not in record:
            continue
        if tau_filter is not None and not math.isclose(float(record["tau"]), float(tau_filter)):
            continue
        env_val = float(record[env_key])
        knob = float(record["knob_value"])
        per_env[env_val][knob].append(float(record[metric]))

    aggregated: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for env_val, buckets in per_env.items():
        means: List[float] = []
        stds: List[float] = []
        counts: List[int] = []
        for knob in knob_values:
            samples = buckets.get(float(knob), [])
            if not samples:
                means.append(math.nan)
                stds.append(math.nan)
                counts.append(0)
                continue
            arr = np.asarray(samples, dtype=float)
            means.append(float(np.mean(arr)))
            stds.append(float(np.std(arr)))
            counts.append(int(arr.size))
        aggregated[env_val] = (np.asarray(means), np.asarray(stds), np.asarray(counts))
    return aggregated


def plot_lines_by_env_knob(
    knob_values: Sequence[float],
    env_series: Mapping[float, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    vary: str,
    env_key: str,
    output_path: Path,
    colormap: Optional[str] = None,
) -> None:
    xs = np.asarray(knob_values, dtype=float)
    if xs.size == 0:
        raise ValueError("No knob values available for line plot.")
    env_sorted = sorted(env_series.keys())
    if not env_sorted:
        raise ValueError("No environment series available for plotting.")

    default_cmaps = {
        "graph_density": "Greens",
        "parent_count": "Blues",
        "arm_variance": "magma",
        "intervention_size": "Oranges",
        "node_count": "cividis",
        "alphabet": "plasma",
    }
    cmap_name = colormap or default_cmaps.get(env_key, "viridis")
    cmap = plt.get_cmap(cmap_name)
    vmin, vmax = min(env_sorted), max(env_sorted)
    span = vmax - vmin if vmax != vmin else 1.0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for env_val in env_sorted:
        means, stds, _counts = env_series[env_val]
        if means.size != xs.size:
            continue
        mask = np.isfinite(means)
        if not mask.any():
            continue
        normed = (env_val - vmin) / span
        color = cmap(normed)
        label = f"{env_key.replace('_', ' ')}={env_val:g}"
        ax.plot(xs[mask], means[mask], marker="o", linewidth=1.8, color=color, label=label)
        if stds.size == means.size:
            std_vals = stds[mask]
            if std_vals.size and np.isfinite(std_vals).any():
                std_vals = np.where(np.isfinite(std_vals), std_vals, 0.0)
                ax.fill_between(
                    xs[mask],
                    means[mask] - std_vals,
                    means[mask] + std_vals,
                    alpha=0.12,
                    color=color,
                )

    ax.set_xlabel(vary.replace("_", " ").title())
    ax.set_ylabel("Metric value")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(ncol=2, fontsize="small", title=env_key.replace("_", " ").title())
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _fill_nan_with_column_means(matrix: np.ndarray) -> np.ndarray:
    filled = np.array(matrix, copy=True)
    if not np.isnan(filled).any():
        return filled
    col_means = np.nanmean(filled, axis=0)
    for col in range(filled.shape[1]):
        value = col_means[col]
        if math.isnan(value):
            value = np.nanmean(filled)
        filled[np.isnan(filled[:, col]), col] = value if not math.isnan(value) else 0.0
    return filled


def _finite_difference(values: np.ndarray, axis_coords: Sequence[float], axis: int) -> np.ndarray:
    grad = np.zeros_like(values)
    coord = np.asarray(axis_coords, dtype=float)
    axis_len = values.shape[axis]
    for idx in range(axis_len):
        if axis == 0:
            slice_front = values[idx + 1, ...] if idx + 1 < axis_len else values[idx, ...]
            slice_back = values[idx - 1, ...] if idx - 1 >= 0 else values[idx, ...]
        else:
            slice_front = values[:, idx + 1] if idx + 1 < axis_len else values[:, idx]
            slice_back = values[:, idx - 1] if idx - 1 >= 0 else values[:, idx]
        if idx == 0:
            delta = coord[1] - coord[0] if axis_len > 1 else 1.0
            diff = slice_front - values[idx, ...] if axis == 0 else slice_front - values[:, idx]
        elif idx == axis_len - 1:
            delta = coord[idx] - coord[idx - 1]
            diff = values[idx, ...] - slice_back if axis == 0 else values[:, idx] - slice_back
        else:
            delta = coord[idx + 1] - coord[idx - 1]
            diff = slice_front - slice_back if axis == 0 else slice_front - slice_back
            delta *= 0.5
        delta = delta if delta != 0 else 1.0
        if axis == 0:
            grad[idx, ...] = diff / delta
        else:
            grad[:, idx] = diff / delta
    return grad


def compute_gradients(
    matrix: np.ndarray,
    tau_values: Sequence[float],
    knob_coords: Sequence[float],
    *,
    sigma: float = 0.5,
    smooth_for_plot: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.size == 0:
        empty = np.zeros_like(matrix)
        return empty, empty, empty
    working = _fill_nan_with_column_means(matrix)
    if smooth_for_plot and sigma > 0 and gaussian_filter is not None:
        working = gaussian_filter(working, sigma=sigma, mode="nearest")
    grad_tau = _finite_difference(working, tau_values, axis=0)
    grad_knob = _finite_difference(working, knob_coords, axis=1)
    grad_tau[np.isnan(matrix)] = np.nan
    grad_knob[np.isnan(matrix)] = np.nan
    return grad_tau, grad_knob, working


def bootstrap_gradients(
    samples: Mapping[Tuple[int, int], Sequence[float]],
    tau_values: Sequence[float],
    knob_coords: Sequence[float],
    *,
    sigma: float,
    iterations: int,
    seed: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    shape = (len(tau_values), len(knob_coords))
    grad_tau_stack: List[np.ndarray] = []
    grad_knob_stack: List[np.ndarray] = []
    if iterations <= 0:
        return {}

    for _ in range(iterations):
        bootstrap_matrix = np.full(shape, np.nan, dtype=float)
        for (i, j), values in samples.items():
            if not values:
                continue
            resampled = rng.choice(values, size=len(values), replace=True)
            bootstrap_matrix[i, j] = float(np.mean(resampled))
        grad_tau, grad_knob, _ = compute_gradients(
            bootstrap_matrix,
            tau_values,
            knob_coords,
            sigma=sigma,
            smooth_for_plot=True,
        )
        grad_tau_stack.append(grad_tau)
        grad_knob_stack.append(grad_knob)

    grad_tau_arr = np.stack(grad_tau_stack, axis=0)
    grad_knob_arr = np.stack(grad_knob_stack, axis=0)

    def _quantiles(arr: np.ndarray, q: float) -> np.ndarray:
        try:
            return np.nanquantile(arr, q, axis=0)
        except ValueError:
            return np.full(arr.shape[1:], np.nan)

    lower = 0.025
    upper = 0.975
    return {
        "grad_tau": (
            _quantiles(grad_tau_arr, lower),
            _quantiles(grad_tau_arr, upper),
        ),
        "grad_knob": (
            _quantiles(grad_knob_arr, lower),
            _quantiles(grad_knob_arr, upper),
        ),
    }


def plot_heatmap_with_annotations(
    matrix: np.ndarray,
    tau_values: Sequence[float],
    knob_values: Sequence[float],
    *,
    finished_rates: Sequence[float],
    title: str,
    metric_label: str,
    x_label: str,
    output_path: Path,
    cmap: str = "viridis",
) -> Tuple[plt.Figure, plt.Axes]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(4, len(knob_values)), 6))
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap=cmap,
    )
    ax.set_xticks(range(len(knob_values)))
    ax.set_xticklabels([f"{v:.3f}" for v in knob_values], rotation=45, ha="right")
    ax.set_yticks(range(len(tau_values)))
    ax.set_yticklabels([f"{tau:.2f}" for tau in tau_values])
    ax.set_ylim(-1, len(tau_values) - 0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\tau$")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_label)

    for j in range(matrix.shape[1]):
        column = matrix[:, j]
        if np.isnan(column).all():
            continue
        best_idx = int(np.nanargmin(column))
        ax.scatter(j, best_idx, color="white", edgecolors="black", s=40, linewidth=0.5)
        rate = finished_rates[j] if j < len(finished_rates) else math.nan
        if not math.isnan(rate):
            ax.text(
                j,
                -0.6,
                f"{100 * rate:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="lightgray",
            )

    fig.tight_layout()
    fig.savefig(output_path)
    return fig, ax


def plot_gradient_flow(
    ax: plt.Axes,
    grad_tau: np.ndarray,
    grad_knob: np.ndarray,
    *,
    ci: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> None:
    if grad_tau.size == 0 or grad_knob.size == 0:
        return
    rows, cols = grad_tau.shape
    x, y = np.meshgrid(range(cols), range(rows))
    dx = -grad_knob
    dy = -grad_tau
    magnitude = np.sqrt(dx**2 + dy**2)
    with np.errstate(invalid="ignore"):
        norm_dx = np.divide(dx, magnitude, out=np.zeros_like(dx), where=magnitude > 0)
        norm_dy = np.divide(dy, magnitude, out=np.zeros_like(dy), where=magnitude > 0)
    try:
        max_mag = float(np.nanmax(magnitude))
    except ValueError:
        max_mag = 0.0
    alpha_base = magnitude / max_mag if max_mag > 0 else magnitude
    alpha = np.clip(np.nan_to_num(alpha_base, nan=0.0), 0.2, 1.0)

    if ci:
        ci_tau = ci.get("grad_tau")
        ci_knob = ci.get("grad_knob")
        mask = np.ones_like(magnitude, dtype=bool)
        if ci_tau:
            lower_tau, upper_tau = ci_tau
            mask &= (lower_tau <= 0) & (upper_tau >= 0)
        if ci_knob:
            lower_knob, upper_knob = ci_knob
            mask &= (lower_knob <= 0) & (upper_knob >= 0)
        alpha = np.where(mask, 0.1, alpha)

    ax.quiver(
        x,
        y,
        norm_dx,
        norm_dy,
        angles="xy",
        scale_units="xy",
        scale=1.5,
        alpha=alpha,
        color="white",
        width=0.003,
    )


def _student_t_cdf(value: float, df: float) -> float:
    if scipy_stats is not None:
        return float(scipy_stats.t.cdf(value, df))
    # Fallback to normal approximation for large df.
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def _chi2_sf(value: float, df: float) -> float:
    if scipy_stats is not None:
        return float(scipy_stats.chi2.sf(value, df))
    # Wilson-Hilferty approximation for chi-square survival.
    if df <= 0:
        return math.nan
    z = ((value / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def _ols_with_covariance(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    n, p = X.shape
    df = max(n - p, 1)
    XtX_inv = np.linalg.pinv(X.T @ X)
    cov_type = "HC3"
    if groups is not None and len(groups):
        unique_groups = np.unique(groups)
        meat = np.zeros((p, p))
        for group in unique_groups:
            idx = groups == group
            Xg = X[idx]
            residual_vec = residuals[idx][:, None]
            meat += Xg.T @ (residual_vec @ residual_vec.T) @ Xg
        cov = XtX_inv @ meat @ XtX_inv
        cov_type = "cluster"
    else:
        leverage = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
        leverage = np.clip(1 - leverage, 1e-8, None)
        scale = (residuals**2) / (leverage**2)
        weighted = X * scale[:, None]
        cov = XtX_inv @ (X.T @ weighted) @ XtX_inv
    residual_var = float(residuals.T @ residuals) / df
    return beta, cov, residuals, df, cov_type


def test_density_tau_interaction(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    g_key: str = "g_density",
    cluster_key: str = "instance_id",
) -> Optional[RegressionOutput]:
    y: List[float] = []
    tau_values: List[float] = []
    g_values: List[float] = []
    groups: List[int] = []
    for record in records:
        if metric not in record or g_key not in record:
            continue
        value = float(record[metric])
        tau = float(record["tau"])
        g_val = float(record[g_key])
        y.append(value)
        tau_values.append(tau)
        g_values.append(g_val)
        if cluster_key in record:
            groups.append(int(record[cluster_key]))
    if not y:
        return None
    y_arr = np.asarray(y, dtype=float)
    tau_arr = np.asarray(tau_values, dtype=float)
    g_arr = np.asarray(g_values, dtype=float)
    g_arr = g_arr - np.mean(g_arr)
    X = np.column_stack(
        [
            np.ones_like(tau_arr),
            tau_arr,
            g_arr,
            tau_arr * g_arr,
        ]
    )
    group_array = np.asarray(groups) if groups else None
    beta, cov, residuals, df, cov_type = _ols_with_covariance(X, y_arr, groups=group_array if groups else None)
    diag = np.diag(cov)
    std_errors = np.sqrt(np.maximum(diag, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / std_errors
    t_stats = np.where(np.isfinite(t_stats), t_stats, np.nan)
    p_values = np.array([2 * (1 - _student_t_cdf(abs(t), df)) for t in t_stats])
    hypothesis_idx = 3  # tau*g interaction coefficient.
    t_value = t_stats[hypothesis_idx]
    p_one_sided = 1 - _student_t_cdf(-t_value, df) if np.isfinite(t_value) else math.nan
    predictions = X @ beta
    ss_total = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    ss_res = float(np.sum((y_arr - predictions) ** 2))
    r_squared = 1 - ss_res / ss_total if ss_total > 0 else math.nan
    labels = ["intercept", "tau", "g_density", "interaction"]
    return RegressionOutput(
        coefficients=dict(zip(labels, beta)),
        std_errors=dict(zip(labels, std_errors)),
        t_stats=dict(zip(labels, t_stats)),
        p_values=dict(zip(labels, p_values)),
        residual_df=float(df),
        r_squared=r_squared,
        cov_type=cov_type,
        hypothesis_p_value=p_one_sided,
    )


def _spearman_rho(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    if len(x) != len(y) or len(x) < 3:
        return math.nan, math.nan
    order_x = np.argsort(x)
    ranks_x = np.zeros(len(x), dtype=float)
    ranks_y = np.zeros(len(y), dtype=float)
    ranks_x[order_x] = np.arange(1, len(x) + 1)
    order_y = np.argsort(y)
    ranks_y[order_y] = np.arange(1, len(y) + 1)
    rho = np.corrcoef(ranks_x, ranks_y)[0, 1]
    df = len(x) - 2
    if not np.isfinite(rho) or abs(rho) >= 1:
        return rho, math.nan
    t_stat = rho * math.sqrt(df / (1 - rho**2))
    p_value = 2 * (1 - _student_t_cdf(abs(t_stat), df))
    return rho, p_value


def column_spearman(matrix: np.ndarray, tau_values: Sequence[float]) -> Dict[str, List[Dict[str, float]]]:
    per_column: List[Dict[str, float]] = []
    p_values: List[float] = []
    tau_axis = list(tau_values)
    for col_idx in range(matrix.shape[1]):
        column = matrix[:, col_idx]
        mask = ~np.isnan(column)
        if mask.sum() < 3:
            per_column.append({"density_index": col_idx, "rho": math.nan, "p_value": math.nan})
            continue
        rho, p_val = _spearman_rho(np.array(tau_axis)[mask], column[mask])
        per_column.append({"density_index": col_idx, "rho": rho, "p_value": p_val})
        if not math.isnan(p_val):
            p_values.append(p_val)
    fisher_stat = -2 * sum(math.log(p) for p in p_values if p > 0)
    fisher_df = 2 * len(p_values)
    fisher_p = _chi2_sf(fisher_stat, fisher_df) if fisher_df else math.nan
    return {"per_column": per_column, "fisher_p_value": fisher_p}


def slope_vs_density(
    records: Sequence[Mapping[str, Any]],
    knob_values: Sequence[float],
    g_values: Optional[Sequence[float]],
    metric: str,
) -> Dict[str, float]:
    slopes: List[float] = []
    slope_ses: List[float] = []
    g_list: List[float] = []
    paired_g = g_values if g_values is not None else knob_values
    for knob, g_val in zip(knob_values, paired_g):
        subset = [r for r in records if float(r["knob_value"]) == knob and metric in r]
        if len(subset) < 3:
            continue
        x = np.array([float(r["tau"]) for r in subset], dtype=float)
        y = np.array([float(r[metric]) for r in subset], dtype=float)
        X = np.column_stack([np.ones_like(x), x])
        beta, cov, *_ = _ols_with_covariance(X, y)
        slope = beta[1]
        se = math.sqrt(max(cov[1, 1], 0))
        slopes.append(slope)
        slope_ses.append(se if se > 0 else 1.0)
        g_list.append(g_val)
    if not slopes:
        return {}
    weights = [1 / (se**2) if se > 0 else 1.0 for se in slope_ses]
    X_meta = np.column_stack([np.ones(len(g_list)), g_list])
    W = np.diag(weights)
    beta_meta = np.linalg.pinv(X_meta.T @ W @ X_meta) @ (X_meta.T @ W @ np.array(slopes))
    cov_meta = np.linalg.pinv(X_meta.T @ W @ X_meta)
    slope_coeff = beta_meta[1]
    slope_se = math.sqrt(max(cov_meta[1, 1], 0))
    t_stat = slope_coeff / slope_se if slope_se > 0 else math.nan
    df = max(len(slopes) - 2, 1)
    p_value = 2 * (1 - _student_t_cdf(abs(t_stat), df)) if math.isfinite(t_stat) else math.nan
    return {
        "slope_vs_density": slope_coeff,
        "slope_se": slope_se,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def build_report_markdown(
    metric: str,
    interaction: Optional[RegressionOutput],
    spearman: Dict[str, List[Dict[str, float]]],
    slope_meta: Dict[str, float],
) -> str:
    lines = [f"# Analysis Report – {metric}"]
    if interaction:
        beta = interaction.coefficients["interaction"]
        se = interaction.std_errors["interaction"]
        p_val = interaction.hypothesis_p_value
        lines.append(
            f"*Interaction (tau × g):* β={beta:.4f}, SE={se:.4f}, one-sided p={p_val:.4g}, R²={interaction.r_squared:.3f}, cov={interaction.cov_type}"
        )
    else:
        lines.append("Interaction test could not be computed (insufficient data).")
    fisher_p = spearman.get("fisher_p_value", math.nan)
    lines.append(f"*Spearman combined p-value:* {fisher_p:.4g}" if not math.isnan(fisher_p) else "Spearman tests unavailable.")
    if slope_meta:
        lines.append(
            "*Slope meta-regression:* coeff={slope_vs_density:.4f}, SE={slope_se:.4f}, t={t_stat:.2f}, p={p_value:.4g}".format(
                **slope_meta
            )
        )
    else:
        lines.append("Slope meta-regression unavailable.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process tau-study JSONL outputs.")
    parser.add_argument("--results", type=Path, required=True, help="Path to results.jsonl produced by run_tau_study.")
    parser.add_argument("--vary", type=str, default="graph_density", help="Knob that was varied.")
    parser.add_argument("--n", type=int, default=None, help="Number of variables; required for log-density transforms.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["cumulative_regret"],
        help="Metrics to analyze (must match keys in results, or use 'all').",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to store plots and reports.")
    parser.add_argument(
        "--plot-mode",
        choices=("heatmap", "line"),
        default="heatmap",
        help="Select 'line' to collapse across tau and draw multi-metric line plots versus the swept knob.",
    )
    parser.add_argument(
        "--lines-by-knob",
        action="store_true",
        help=(
            "Draw a family of lines (one per knob value) over the tau grid for the first requested metric, "
            "with color shading by knob magnitude. Uses --vary to determine the label."
        ),
    )
    parser.add_argument(
        "--line-colormap",
        type=str,
        default=None,
        help="Matplotlib colormap name for --lines-by-knob; defaults to a preset based on --vary.",
    )
    parser.add_argument(
        "--lines-by-env-knob",
        type=str,
        default=None,
        help=(
            "Draw a family of lines over the swept knob (x-axis = --vary knob) grouped by an environment key "
            "(e.g., graph_density) for the first requested metric. Requires a single tau or --lines-env-tau."
        ),
    )
    parser.add_argument(
        "--lines-env-tau",
        type=float,
        default=None,
        help="Tau value to filter when using --lines-by-env-knob; if omitted, requires the results to have a single tau.",
    )
    parser.add_argument(
        "--lines-env-colormap",
        type=str,
        default=None,
        help="Matplotlib colormap name for --lines-by-env-knob; defaults to a preset based on the chosen env key.",
    )
    parser.add_argument("--sigma", type=float, default=0.5, help="Gaussian smoothing sigma used for gradient visualization.")
    parser.add_argument("--no-smooth", action="store_true", help="Disable smoothing before computing gradient arrows.")
    parser.add_argument("--bootstrap", type=int, default=256, help="Number of bootstrap resamples for gradient CIs.")
    parser.add_argument("--bootstrap-seed", type=int, default=0, help="Random seed for bootstrap resampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    requested_metrics = args.metrics
    if any(metric.lower() == "all" for metric in requested_metrics):
        if any(metric.lower() != "all" for metric in requested_metrics):
            LOGGER.warning("--metrics 'all' overrides the other entries; analyzing all metrics.")
        metrics_to_analyze: Sequence[str] = list(KNOWN_METRICS)
    else:
        metrics_to_analyze = requested_metrics

    loaded = load_results(args.results)
    g_values = compute_log_density(loaded.knob_values, args.n) if args.vary == "graph_density" else None
    if args.vary == "graph_density" and g_values is None:
        LOGGER.warning("Graph density analysis requested but --n was not provided; interaction tests will be skipped.")
    for record in loaded.records:
        knob = float(record["knob_value"])
        if g_values is not None:
            try:
                idx = loaded.knob_values.index(knob)
            except ValueError:
                continue
            record["g_density"] = g_values[idx]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    line_series: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for metric in metrics_to_analyze:
        matrix, per_cell_samples, finished_rates = aggregate_matrix(
            loaded.records,
            loaded.tau_values,
            loaded.knob_values,
            metric,
        )

        if args.plot_mode == "heatmap":
            g_coords = g_values if g_values is not None else loaded.knob_values
            grad_tau, grad_knob, _ = compute_gradients(
                matrix,
                loaded.tau_values,
                g_coords,
                sigma=args.sigma,
                smooth_for_plot=not args.no_smooth,
            )
            ci = bootstrap_gradients(
                per_cell_samples,
                loaded.tau_values,
                g_coords,
                sigma=args.sigma,
                iterations=args.bootstrap,
                seed=args.bootstrap_seed,
            )

            heatmap_path = args.out_dir / f"heatmap_{metric}.png"
            fig, ax = plot_heatmap_with_annotations(
                matrix,
                loaded.tau_values,
                loaded.knob_values,
                finished_rates=finished_rates,
                title=f"{metric.replace('_', ' ').title()}",
                metric_label=metric.replace("_", " ").title(),
                x_label=args.vary.replace("_", " ").title(),
                output_path=heatmap_path,
            )
            plot_gradient_flow(ax, grad_tau, grad_knob, ci=ci)
            fig.savefig(args.out_dir / f"flow_{metric}.png")
            plt.close(fig)
        else:
            means, stds, counts = aggregate_metric_series(
                loaded.records,
                loaded.knob_values,
                metric,
            )
            line_series.append((metric, means, stds, counts))

        interaction = test_density_tau_interaction(loaded.records, metric=metric)
        spearman_stats = column_spearman(matrix, loaded.tau_values)
        slope_stats = slope_vs_density(loaded.records, loaded.knob_values, g_values, metric)

        report = {
            "metric": metric,
            "interaction": interaction.__dict__ if interaction else None,
            "spearman": spearman_stats,
            "slope_vs_density": slope_stats,
            "settings": {
                "sigma": args.sigma,
                "bootstrap": args.bootstrap,
                "n": args.n,
                "vary": args.vary,
                "tau_values": loaded.tau_values,
                "knob_values": loaded.knob_values,
            },
        }
        json_path = args.out_dir / f"tests_{metric}.json"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        md_content = build_report_markdown(metric, interaction, spearman_stats, slope_stats)
        (args.out_dir / f"tests_{metric}.md").write_text(md_content, encoding="utf-8")

    if args.plot_mode == "line" and line_series:
        try:
            plot_metric_lines(
                loaded.knob_values,
                line_series,
                x_label=args.vary.replace("_", " ").title(),
                output_path=args.out_dir / f"line_{args.vary}.png",
            )
        except ValueError as err:
            LOGGER.warning("Line plot skipped: %s", err)

    if args.lines_by_knob and metrics_to_analyze:
        metric = metrics_to_analyze[0]
        try:
            knob_series = aggregate_metric_series_by_knob(loaded.records, loaded.tau_values, metric)
            plot_lines_by_knob(
                loaded.tau_values,
                knob_series,
                vary=args.vary,
                output_path=args.out_dir / f"lines_by_knob_{metric}.png",
                colormap=args.line_colormap,
            )
        except ValueError as err:
            LOGGER.warning("lines-by-knob plot skipped: %s", err)

    if args.lines_by_env_knob and metrics_to_analyze:
        metric = metrics_to_analyze[0]
        tau_filter = args.lines_env_tau
        if tau_filter is None and len(loaded.tau_values) != 1:
            LOGGER.warning(
                "lines-by-env-knob requested but multiple tau values found and --lines-env-tau not set; skipping."
            )
        else:
            tau_target = tau_filter if tau_filter is not None else loaded.tau_values[0]
            try:
                env_series = aggregate_env_lines(
                    loaded.records,
                    loaded.knob_values,
                    metric,
                    env_key=args.lines_by_env_knob,
                    tau_filter=tau_target,
                )
                plot_lines_by_env_knob(
                    loaded.knob_values,
                    env_series,
                    vary=args.vary,
                    env_key=args.lines_by_env_knob,
                    output_path=args.out_dir / f"line_{args.vary}_by_{args.lines_by_env_knob}.png",
                    colormap=args.lines_env_colormap,
                )
            except ValueError as err:
                LOGGER.warning("lines-by-env-knob plot skipped: %s", err)


if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    main()
