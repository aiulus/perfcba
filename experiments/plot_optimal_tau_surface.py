"""Generate 3D surface plot of optimal tau across (epsilon, delta) space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot optimal tau surface over (epsilon, delta).")
    parser.add_argument("--results-jsonl", type=Path, required=True, help="Path to results.jsonl from joint sweep.")
    parser.add_argument("--output-path", type=Path, required=True, help="Path to save the surface plot.")
    args = parser.parse_args()

    with args.results_jsonl.open() as f:
        results: List[Dict] = [json.loads(line) for line in f]

    eps_vals = sorted({r.get("algo_eps", r.get("measured_epsilon", np.nan)) for r in results if not np.isnan(r.get("algo_eps", r.get("measured_epsilon", np.nan)))})
    delta_vals = sorted({r.get("algo_delta", r.get("measured_delta", np.nan)) for r in results if not np.isnan(r.get("algo_delta", r.get("measured_delta", np.nan)))})

    if len(eps_vals) < 2 or len(delta_vals) < 2:
        raise ValueError("Need at least two epsilon and delta values to plot a surface.")

    optimal_taus = np.full((len(eps_vals), len(delta_vals)), np.nan, dtype=float)

    for i, eps in enumerate(eps_vals):
        for j, delta in enumerate(delta_vals):
            subset = [
                r
                for r in results
                if np.isclose(r.get("algo_eps", r.get("measured_epsilon", -1)), eps, atol=5e-3)
                and np.isclose(r.get("algo_delta", r.get("measured_delta", -1)), delta, atol=5e-3)
            ]
            if not subset:
                continue
            tau_regrets: Dict[float, List[float]] = {}
            for r in subset:
                tau_regrets.setdefault(float(r["tau"]), []).append(float(r["cumulative_regret"]))
            tau_means = {tau: float(np.mean(vals)) for tau, vals in tau_regrets.items()}
            optimal_tau = min(tau_means, key=tau_means.get)
            optimal_taus[i, j] = optimal_tau

    EPS, DELTA = np.meshgrid(eps_vals, delta_vals, indexing="ij")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(EPS, DELTA, optimal_taus, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("ε (Ancestral Gap)")
    ax.set_ylabel("Δ (Reward Gap)")
    ax.set_zlabel("Optimal τ")
    ax.set_title("Optimal τ over (ε, Δ)")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=200)
    plt.close(fig)
    print(f"Saved surface plot to {args.output_path}")
    print(f"Optimal tau stats: min={np.nanmin(optimal_taus):.3f}, max={np.nanmax(optimal_taus):.3f}, mean={np.nanmean(optimal_taus):.3f}, std={np.nanstd(optimal_taus):.3f}")


if __name__ == "__main__":
    main()
