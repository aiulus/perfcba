"""Validate that generated SCMs satisfy requested gap constraints."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .causal_envs import CausalBanditConfig, build_random_scm_with_gaps


def validate_gaps(config: CausalBanditConfig, n_samples: int = 50) -> Dict:
    """Generate n_samples SCMs and measure gap statistics."""

    results = {
        "epsilon_values": [],
        "delta_values": [],
        "attempts": [],
        "satisfied": [],
    }

    for seed in range(n_samples):
        cfg = dataclasses.replace(config, seed=seed)
        instance, diagnostics = build_random_scm_with_gaps(cfg)
        results["epsilon_values"].append(diagnostics.get("measured_epsilon", np.nan))
        results["delta_values"].append(diagnostics.get("measured_delta", np.nan))
        results["attempts"].append(diagnostics.get("attempts", 0))
        results["satisfied"].append(bool(diagnostics.get("gap_satisfied", False)))

    return results


def plot_gap_distributions(results: Dict, output_path: Path) -> None:
    """Plot distributions of measured gaps."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    eps_vals = [v for v in results["epsilon_values"] if np.isfinite(v)]
    delta_vals = [v for v in results["delta_values"] if np.isfinite(v)]
    attempts = [v for v in results["attempts"] if v]

    axes[0].hist(eps_vals, bins=20, alpha=0.7, edgecolor="black")
    if eps_vals:
        axes[0].axvline(np.mean(eps_vals), color="red", linestyle="--", label=f"Mean: {np.mean(eps_vals):.3f}")
        axes[0].legend()
    axes[0].set_xlabel("Measured ε")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Ancestral Gap Distribution")

    axes[1].hist(delta_vals, bins=20, alpha=0.7, edgecolor="black")
    if delta_vals:
        axes[1].axvline(np.mean(delta_vals), color="red", linestyle="--", label=f"Mean: {np.mean(delta_vals):.3f}")
        axes[1].legend()
    axes[1].set_xlabel("Measured Δ")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Reward Gap Distribution")

    if attempts:
        axes[2].hist(attempts, bins=range(1, max(attempts) + 2), alpha=0.7, edgecolor="black")
    axes[2].set_xlabel("Rejection Attempts")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title(f"Acceptance Rate: {np.mean(results['satisfied']):.1%}")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate gap satisfaction for generated SCMs.")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--target-epsilon", type=float, default=0.05)
    parser.add_argument("--target-delta", type=float, default=0.05)
    parser.add_argument("--edge-prob", type=float, default=None)
    parser.add_argument("--edge-prob-covariates", type=float, default=None)
    parser.add_argument("--edge-prob-to-reward", type=float, default=None)
    parser.add_argument("--arm-heterogeneity-mode", choices=["uniform", "sparse", "clustered"], default="uniform")
    parser.add_argument("--sparse-fraction", type=float, default=0.1)
    parser.add_argument("--sparse-separation", type=float, default=0.3)
    parser.add_argument("--cluster-count", type=int, default=3)
    parser.add_argument("--max-rejection-attempts", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base_edge_prob = args.edge_prob if args.edge_prob is not None else 2.0 / max(1, args.n)
    config = CausalBanditConfig(
        n=args.n,
        k=args.k,
        m=args.k,
        ell=args.ell,
        edge_prob=base_edge_prob,
        edge_prob_covariates=args.edge_prob_covariates,
        edge_prob_to_reward=args.edge_prob_to_reward,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        gap_enforcement_mode="reject",
        max_rejection_attempts=args.max_rejection_attempts,
        arm_heterogeneity_mode=args.arm_heterogeneity_mode,
        sparse_fraction=args.sparse_fraction,
        sparse_separation=args.sparse_separation,
        cluster_count=args.cluster_count,
    )

    print(
        f"Validating gaps for n={args.n}, k={args.k}, target ε={args.target_epsilon}, Δ={args.target_delta} "
        f"(mode={config.gap_enforcement_mode}, attempts={config.max_rejection_attempts})"
    )
    results = validate_gaps(config, n_samples=args.n_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "gap_validation.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_gap_distributions(results, args.output_dir / "gap_distributions.png")

    print(f"\nResults (n={len(results['epsilon_values'])} samples):")
    print(f"  ε: {np.nanmean(results['epsilon_values']):.4f} ± {np.nanstd(results['epsilon_values']):.4f}")
    print(f"  Δ: {np.nanmean(results['delta_values']):.4f} ± {np.nanstd(results['delta_values']):.4f}")
    print(f"  Acceptance rate: {np.mean(results['satisfied']):.1%}")
    print(f"  Avg rejection attempts: {np.mean(results['attempts']):.1f}")


if __name__ == "__main__":
    main()
