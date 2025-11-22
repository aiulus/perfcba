"""Compare regret curves across scheduler types."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .artifacts import load_all_artifacts
from .regret_curves import cumulative_regret_trajectory


def _aggregate_curves(artifact_dir: Path) -> np.ndarray:
    artifacts = load_all_artifacts(artifact_dir)
    if not artifacts:
        return np.array([])
    curves = [cumulative_regret_trajectory(a) for a in artifacts]
    max_len = max(c.size for c in curves)
    matrix = np.zeros((len(curves), max_len))
    for i, curve in enumerate(curves):
        matrix[i, : curve.size] = curve
        if curve.size < max_len:
            matrix[i, curve.size :] = curve[-1]
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare regret curves across schedulers.")
    parser.add_argument("--base-dir", type=Path, required=True, help="Base directory containing results by difficulty.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write comparison plots.")
    parser.add_argument(
        "--difficulties",
        type=str,
        nargs="+",
        default=["easy", "medium", "hard"],
        help="Difficulty subdirectories to compare.",
    )
    parser.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6],
        help="Tau values to plot.",
    )
    parser.add_argument(
        "--schedulers",
        type=str,
        nargs="+",
        default=["etc", "adaptive"],
        help="Scheduler labels in subdirectories (e.g., etc_tau0.2, adaptive_tau0.2).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for difficulty in args.difficulties:
        fig, ax = plt.subplots(figsize=(10, 6))
        for scheduler in args.schedulers:
            for tau in args.taus:
                dir_name = f"{scheduler}_tau{tau}"
                artifact_dir = args.base_dir / difficulty / dir_name
                if not artifact_dir.exists():
                    continue
                matrix = _aggregate_curves(artifact_dir)
                if matrix.size == 0:
                    continue
                mean = matrix.mean(axis=0)
                std = matrix.std(axis=0)
                xs = np.arange(1, mean.size + 1)
                linestyle = "-" if scheduler == "etc" else "--"
                ax.plot(xs, mean, label=f"{scheduler.upper()} τ={tau}", linestyle=linestyle, linewidth=2)
                ax.fill_between(xs, mean - std, mean + std, alpha=0.15)
        ax.set_xlabel("Round")
        ax.set_ylabel("Cumulative Regret")
        ax.set_title(f"Scheduler Comparison - {difficulty.title()}")
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.3)
        fig.savefig(args.output_dir / f"comparison_{difficulty}.png", dpi=200)
        plt.close(fig)
        print(f"Generated comparison plot for {difficulty}")


if __name__ == "__main__":
    main()
