"""Analyze correlation between measured gaps and regret."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlation between measured gaps and regret.")
    parser.add_argument("--results-jsonl", type=Path, required=True, help="Path to results.jsonl from tau study.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for plots.")
    args = parser.parse_args()

    with args.results_jsonl.open() as f:
        results: List[Dict] = [json.loads(line) for line in f]

    tau_values = sorted({r["tau"] for r in results})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for tau in tau_values:
        subset = [r for r in results if r["tau"] == tau]
        if not subset or "measured_epsilon" not in subset[0]:
            print(f"Warning: No gap measurements for tau={tau}")
            continue
        epsilon_vals = np.array([r.get("measured_epsilon", np.nan) for r in subset], dtype=float)
        delta_vals = np.array([r.get("measured_delta", np.nan) for r in subset], dtype=float)
        regrets = np.array([r.get("cumulative_regret", np.nan) for r in subset], dtype=float)
        mask = np.isfinite(epsilon_vals) & np.isfinite(delta_vals) & np.isfinite(regrets)
        epsilon_vals, delta_vals, regrets = epsilon_vals[mask], delta_vals[mask], regrets[mask]
        if regrets.size < 3:
            continue

        corr_eps, p_eps = stats.spearmanr(epsilon_vals, regrets)
        corr_delta, p_delta = stats.spearmanr(delta_vals, regrets)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].scatter(epsilon_vals, regrets, alpha=0.6, s=40)
        axes[0].set_xlabel("Measured ε")
        axes[0].set_ylabel("Cumulative Regret")
        axes[0].set_title(f"tau={tau}: ρ={corr_eps:.3f}, p={p_eps:.3f}")
        z = np.polyfit(epsilon_vals, regrets, 1)
        axes[0].plot(epsilon_vals, np.poly1d(z)(epsilon_vals), "r--", alpha=0.7)
        axes[0].grid(True, alpha=0.3)

        axes[1].scatter(delta_vals, regrets, alpha=0.6, s=40, color="orange")
        axes[1].set_xlabel("Measured Δ")
        axes[1].set_ylabel("Cumulative Regret")
        axes[1].set_title(f"tau={tau}: ρ={corr_delta:.3f}, p={p_delta:.3f}")
        z2 = np.polyfit(delta_vals, regrets, 1)
        axes[1].plot(delta_vals, np.poly1d(z2)(delta_vals), "r--", alpha=0.7)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(args.output_dir / f"gap_regret_correlation_tau{tau}.png", dpi=200)
        plt.close(fig)

        print(f"tau={tau}: eps-rho={corr_eps:.3f} (p={p_eps:.4f}), delta-rho={corr_delta:.3f} (p={p_delta:.4f})")


if __name__ == "__main__":
    main()
