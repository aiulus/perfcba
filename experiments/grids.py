"""Predefined grids for tau-scheduled causal bandit experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np


TAU_GRID = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]

DEFAULTS = {
    "n": 50,
    "ell": 2,
    "k": 2,
    "m": 2,
    "p": None,  # filled as 2/n by helper
    "T": 10_000,
}


def _graph_density_grid(n: int) -> List[float]:
    values = [1 / n, 2 / n, 4 / n, np.log(n) / n, (np.log(n) ** 2) / n]
    return [min(1.0, float(v)) for v in values]


def _node_count_grid(n: int) -> List[int]:
    """Default node-count grid centered on the baseline n."""

    lower = max(3, n // 2)
    upper = max(lower + 1, min(2 * n, max(4, n + 1)))
    return sorted({lower, int(n), upper})


GRIDS = {
    "graph_density": _graph_density_grid,
    "node_count": _node_count_grid,
    "parent_count": [1, 2, 4, 8],
    "intervention_size": lambda k, n: [max(1, k // 2), k, min(k + 2, n)],
    "alphabet": [2, 4],
    "horizon": [100],
    #"horizon": [2_000, 10_000, 50_000],
    # Scales < 1 push Bernoulli means toward 0.5 (higher variance); > 1 make them sharper.
    "arm_variance": [0.5, 0.75, 1.0, 1.5, 2.0],
    "algo_eps": [0.01, 0.02, 0.05, 0.1],
    "algo_delta": [0.01, 0.03, 0.05, 0.1],
    "hard_margin": [0.0, 0.05, 0.1, 0.2],
    "tau": TAU_GRID,
}


def list_knobs() -> Sequence[str]:
    return tuple(GRIDS.keys())


def grid_values(name: str, *, n: int, k: int) -> Sequence[float]:
    grid = GRIDS[name]
    if callable(grid):
        if name == "graph_density":
            return grid(n)
        if name == "node_count":
            return grid(n)
        if name == "intervention_size":
            return grid(k, n)
    return grid
