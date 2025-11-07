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

GRIDS = {
    "graph_density": lambda n: [1 / n, 2 / n, 4 / n, np.log(n) / n, (np.log(n) ** 2) / n],
    "parent_count": [1, 2, 4, 8],
    "intervention_size": lambda k, n: [max(1, k // 2), k, min(k + 2, n)],
    "alphabet": [2, 4],
    "horizon": [2_000, 10_000, 50_000],
}


def list_knobs() -> Sequence[str]:
    return tuple(GRIDS.keys())


def grid_values(name: str, *, n: int, k: int) -> Sequence[float]:
    grid = GRIDS[name]
    if callable(grid):
        if name == "graph_density":
            return grid(n)
        if name == "intervention_size":
            return grid(k, n)
    return grid
