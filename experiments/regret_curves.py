"""Utilities for building classical regret curves from tau-study artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .artifacts import TrialArtifact


def cumulative_regret_trajectory(artifact: TrialArtifact) -> np.ndarray:
    logs = artifact.summary.logs
    if not logs:
        return np.zeros(0, dtype=np.float64)
    deltas = np.empty(len(logs), dtype=np.float64)
    optimal = float(artifact.optimal_mean)
    for idx, entry in enumerate(logs):
        expected = entry.expected_mean
        if not math.isfinite(expected):
            expected = optimal
        deltas[idx] = optimal - expected
    return np.cumsum(deltas)


def aggregate_regret_curves(
    artifacts: Sequence[TrialArtifact],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not artifacts:
        raise ValueError("No artifacts provided.")
    curves = [cumulative_regret_trajectory(artifact) for artifact in artifacts]
    max_len = max(curve.size for curve in curves) if curves else 0
    if max_len == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros((len(curves), 0), dtype=np.float64),
        )
    matrix = np.zeros((len(curves), max_len), dtype=np.float64)
    for idx, curve in enumerate(curves):
        if curve.size == 0:
            continue
        matrix[idx, : curve.size] = curve
        if curve.size < max_len:
            matrix[idx, curve.size:] = curve[-1]
    xs = np.arange(1, max_len + 1)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    return xs, mean, std, matrix


def plot_regret_band(
    xs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    output_path: Path,
    label: Optional[str] = None,
    color: str = "#1f77b4",
    individual_curves: Optional[np.ndarray] = None,
) -> None:
    if xs.size == 0:
        raise ValueError("Regret curve is empty; nothing to plot.")
    fig, ax = plt.subplots(figsize=(7, 4))
    if individual_curves is not None and individual_curves.size:
        for row in individual_curves:
            ax.plot(xs, row, color=color, alpha=0.2, linewidth=0.8)
    ax.plot(xs, mean, color=color, linewidth=2.2, label=label or "Mean cumulative regret")
    ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.2, label="±1 std")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative regret")
    ax.grid(True, linestyle="--", alpha=0.3)
    if label:
        ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


__all__ = [
    "aggregate_regret_curves",
    "cumulative_regret_trajectory",
    "plot_regret_band",
]
