"""Plotting utilities for tau-study heat maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(
    values: np.ndarray,
    tau_values: Sequence[float],
    knob_values: Sequence[float],
    *,
    title: str,
    cbar_label: str,
    x_label: str,
    output_path: Path,
    overlay_mask: Optional[np.ndarray] = None,
    overlay_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(4, len(knob_values)), 6))
    im = ax.imshow(
        values,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    ax.set_xticks(range(len(knob_values)))
    ax.set_xticklabels([f"{v:.2f}" for v in knob_values], rotation=45, ha="right")
    ax.set_yticks(range(len(tau_values)))
    ax.set_yticklabels([f"{tau:.2f}" for tau in tau_values])
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\tau$")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    if overlay_mask is not None:
        if overlay_mask.shape != values.shape:
            raise ValueError("overlay_mask must have the same shape as values")
        marker_config = {
            "marker": "+",
            "s": 50,
            "linewidths": 1.5,
            "color": "white",
        }
        if overlay_kwargs:
            marker_config.update(overlay_kwargs)
        rows, cols = np.where(overlay_mask)
        if rows.size:
            ax.scatter(cols, rows, **marker_config)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
