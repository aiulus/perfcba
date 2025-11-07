"""Utilities for plotting structure/exploitation timelines."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .scheduler import RoundLog


def encode_schedule(logs: Sequence[RoundLog]) -> np.ndarray:
    """Convert a sequence of RoundLog entries into a binary schedule array.

    Returns
    -------
    np.ndarray
        Array of shape (len(logs),) with 1 for structure rounds and 0 for exploitation.
    """
    arr = np.zeros(len(logs), dtype=np.float32)
    for idx, entry in enumerate(logs):
        arr[idx] = 1.0 if entry.mode == "structure" else 0.0
    return arr


def _downsample(matrix: np.ndarray, max_columns: int) -> np.ndarray:
    if max_columns <= 0 or matrix.shape[1] <= max_columns:
        return matrix
    chunk = int(math.ceil(matrix.shape[1] / max_columns))
    if chunk <= 1:
        return matrix
    cols = int(math.ceil(matrix.shape[1] / chunk))
    down = np.zeros((matrix.shape[0], cols), dtype=np.float32)
    for col in range(cols):
        start = col * chunk
        end = min(start + chunk, matrix.shape[1])
        down[:, col] = matrix[:, start:end].mean(axis=1)
    return down


def plot_time_allocation(
    matrix: np.ndarray,
    output_path: Path,
    *,
    title: Optional[str] = None,
    yticklabels: Optional[Sequence[str]] = None,
    max_columns: int = 2000,
) -> None:
    """Render a time-allocation diagram to disk.

    Parameters
    ----------
    matrix:
        2D array where rows correspond to trials/seeds and columns to rounds.
        Entries represent the fraction of structure rounds in that slot (0=exploit/white, 1=structure/red).
    output_path:
        Destination image path (PNG recommended).
    title:
        Optional figure title.
    yticklabels:
        Labels for each row.
    max_columns:
        Max number of columns to display; longer horizons are average-downsampled.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    if matrix.size == 0:
        return
    matrix = _downsample(matrix, max_columns)
    rows, cols = matrix.shape
    width = max(6.0, min(18.0, cols / 400.0 + 2.0))
    height = max(1.5, rows * 0.6)

    cmap = LinearSegmentedColormap.from_list("explore_exploit", ["#ffffff", "#c62828"])
    fig, ax = plt.subplots(figsize=(width, height))
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xlabel("Round")
    if title:
        ax.set_title(title)
    if yticklabels:
        ax.set_yticks(np.arange(rows))
        ax.set_yticklabels(list(yticklabels))
    else:
        ax.set_yticks([])
    ax.set_xticks([])
    ax.spines[:].set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


__all__ = ["encode_schedule", "plot_time_allocation"]
