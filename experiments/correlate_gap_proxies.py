"""Quick experiments to test whether proxy knobs track true gap parameters.

We sweep over:
1) hard_margin (l1/TV spacing between CPDs) vs. measured epsilon (ancestral gap)
2) reward heterogeneity vs. measured Delta (reward gap)

For each setting we sample several SCMs, measure epsilon/Delta via Monte Carlo,
compute simple heterogeneity statistics on the reward CPT, and run a Spearman
correlation test between the proxy knob and the measured gap.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

try:
    from scipy import stats
except Exception:  # pragma: no cover - optional dependency for quick experiments
    stats = None

from .causal_envs import CausalBanditConfig, CausalBanditInstance, build_random_scm


def reward_heterogeneity(instance: CausalBanditInstance) -> Dict[str, float]:
    """Compute simple spread statistics over reward means."""

    means = np.array(list(instance.reward_means.values()), dtype=float)
    if means.size == 0:
        return {
            "std": 0.0,
            "range": 0.0,
            "pairwise_mean": 0.0,
            "pairwise_min": 0.0,
            "pairwise_max": 0.0,
        }

    pairwise = np.abs(means[:, None] - means[None, :])
    triu = pairwise[np.triu_indices_from(pairwise, k=1)]
    return {
        "std": float(np.std(means)),
        "range": float(np.max(means) - np.min(means)),
        "pairwise_mean": float(np.mean(triu)) if triu.size else 0.0,
        "pairwise_min": float(np.min(triu)) if triu.size else 0.0,
        "pairwise_max": float(np.max(triu)) if triu.size else 0.0,
    }


def spearman_corr(x: Iterable[float], y: Iterable[float]) -> Dict[str, float] | None:
    """Compute Spearman correlation and p-value if possible."""

    if stats is None:
        return None
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 3:
        return None
    rho, p = stats.spearmanr(x_arr[mask], y_arr[mask])
    return {"rho": float(rho), "p_value": float(p), "n": int(mask.sum())}


def build_base_config(args: argparse.Namespace, *, hard_margin: float) -> CausalBanditConfig:
    """Create a CausalBanditConfig with shared defaults."""

    base_edge_prob = args.edge_prob if args.edge_prob is not None else 2.0 / max(1, args.n)
    return CausalBanditConfig(
        n=args.n,
        k=args.k,
        m=args.m,
        ell=args.ell,
        edge_prob=base_edge_prob,
        edge_prob_covariates=args.edge_prob_covariates,
        edge_prob_to_reward=args.edge_prob_to_reward,
        hard_margin=hard_margin,
        arm_heterogeneity_mode=args.arm_heterogeneity_mode,
        sparse_fraction=args.sparse_fraction,
        sparse_separation=args.sparse_separation,
        cluster_count=args.cluster_count,
        reward_logit_scale=args.reward_logit_scale,
        gap_strict=not args.gap_non_strict,
        gap_use_shrinkage=args.gap_shrink,
        gap_max_hops=args.gap_max_hops,
    )


def measure_instance(
    cfg: CausalBanditConfig,
    *,
    seed: int,
    gap_mc_samples: int,
    gap_alpha: float,
    gap_exact_parent_scope: int,
) -> Dict:
    """Sample an SCM, measure epsilon/Delta, and collect heterogeneity stats."""

    seeded_cfg = dataclasses.replace(cfg, seed=seed)
    instance = build_random_scm(seeded_cfg)
    gaps = instance.estimate_min_eps_delta(
        max_parent_scope_exact=gap_exact_parent_scope,
        n_mc=gap_mc_samples,
        alpha=gap_alpha,
        max_hops=seeded_cfg.gap_max_hops,
        strict=seeded_cfg.gap_strict,
        shrink=seeded_cfg.gap_use_shrinkage,
        rng=np.random.default_rng(seed),
    )
    hetero = reward_heterogeneity(instance)
    return {
        "measured_epsilon": float(gaps.eps),
        "measured_delta": float(gaps.Delta),
        "heterogeneity": hetero,
        "config": {
            "n": seeded_cfg.n,
            "k": seeded_cfg.k,
            "ell": seeded_cfg.ell,
            "m": seeded_cfg.m,
            "edge_prob": seeded_cfg.edge_prob,
            "edge_prob_covariates": seeded_cfg.edge_prob_covariates,
            "edge_prob_to_reward": seeded_cfg.edge_prob_to_reward,
        },
    }


def run_hard_margin_sweep(args: argparse.Namespace) -> List[Dict]:
    """Sweep over hard_margin values and measure epsilon."""

    records: List[Dict] = []
    for idx, margin in enumerate(args.hard_margin_grid):
        cfg = build_base_config(args, hard_margin=float(margin))
        for rep in range(args.samples_per_setting):
            seed = args.seed_offset + idx * args.samples_per_setting + rep
            meas = measure_instance(
                cfg,
                seed=seed,
                gap_mc_samples=args.gap_mc_samples,
                gap_alpha=args.gap_alpha,
                gap_exact_parent_scope=args.gap_exact_parent_scope,
            )
            record = {
                "experiment": "hard_margin",
                "seed": seed,
                "hard_margin": float(margin),
                "measured_epsilon": meas["measured_epsilon"],
                "measured_delta": meas["measured_delta"],
                "heterogeneity": meas["heterogeneity"],
                "config": meas["config"],
            }
            records.append(record)
    return records


def run_heterogeneity_sweep(args: argparse.Namespace) -> List[Dict]:
    """Sweep over reward heterogeneity knobs and measure Delta."""

    records: List[Dict] = []
    for idx, knob_value in enumerate(args.heterogeneity_grid):
        cfg = build_base_config(args, hard_margin=args.heterogeneity_hard_margin)
        if cfg.arm_heterogeneity_mode == "sparse":
            cfg = dataclasses.replace(cfg, sparse_separation=float(knob_value))
        elif cfg.arm_heterogeneity_mode == "clustered":
            cfg = dataclasses.replace(cfg, cluster_count=max(1, int(round(knob_value))))
        else:
            cfg = dataclasses.replace(cfg, reward_logit_scale=float(knob_value))

        for rep in range(args.samples_per_setting):
            seed = args.seed_offset + len(args.hard_margin_grid) * args.samples_per_setting + idx * args.samples_per_setting + rep
            meas = measure_instance(
                cfg,
                seed=seed,
                gap_mc_samples=args.gap_mc_samples,
                gap_alpha=args.gap_alpha,
                gap_exact_parent_scope=args.gap_exact_parent_scope,
            )
            hetero_metric = meas["heterogeneity"]["pairwise_mean"]
            record = {
                "experiment": "heterogeneity",
                "seed": seed,
                "heterogeneity_knob": float(knob_value),
                "heterogeneity_metric": hetero_metric,
                "measured_epsilon": meas["measured_epsilon"],
                "measured_delta": meas["measured_delta"],
                "hard_margin": cfg.hard_margin,
                "config": meas["config"],
                "heterogeneity_stats": meas["heterogeneity"],
            }
            records.append(record)
    return records


def summarize(records: Sequence[Dict]) -> Dict:
    """Compute correlation tests and per-setting summaries."""

    hard = [r for r in records if r["experiment"] == "hard_margin"]
    hetero = [r for r in records if r["experiment"] == "heterogeneity"]

    hard_corr = spearman_corr(
        [r["hard_margin"] for r in hard],
        [r["measured_epsilon"] for r in hard],
    )
    hetero_corr = spearman_corr(
        [r["heterogeneity_metric"] for r in hetero],
        [r["measured_delta"] for r in hetero],
    )
    knob_corr = spearman_corr(
        [r["heterogeneity_knob"] for r in hetero],
        [r["measured_delta"] for r in hetero],
    )

    def aggregate_by(key: str, subset: Sequence[Dict], target: str) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        grouped: Dict[float, List[float]] = {}
        for rec in subset:
            grouped.setdefault(float(rec[key]), []).append(float(rec[target]))
        for val, vals in grouped.items():
            arr = np.asarray(vals, dtype=float)
            out[str(val)] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "n": int(arr.size),
            }
        return out

    return {
        "hard_margin_vs_epsilon": hard_corr,
        "heterogeneity_metric_vs_delta": hetero_corr,
        "heterogeneity_knob_vs_delta": knob_corr,
        "per_hard_margin_epsilon": aggregate_by("hard_margin", hard, "measured_epsilon"),
        "per_heterogeneity_knob_delta": aggregate_by("heterogeneity_knob", hetero, "measured_delta"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether hard_margin and arm heterogeneity track true gaps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=20, help="Number of covariates.")
    parser.add_argument("--k", type=int, default=3, help="Reward parent count.")
    parser.add_argument("--m", type=int, default=2, help="Intervention budget.")
    parser.add_argument("--ell", type=int, default=2, help="Alphabet size.")
    parser.add_argument("--edge-prob", type=float, default=None, help="Baseline edge probability (defaults to 2/n).")
    parser.add_argument("--edge-prob-covariates", type=float, default=None, help="Optional override for covariate edges.")
    parser.add_argument("--edge-prob-to-reward", type=float, default=None, help="Optional override for reward edges.")

    parser.add_argument(
        "--hard-margin-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2],
        help="Grid of hard_margin values to test against measured epsilon.",
    )
    parser.add_argument(
        "--heterogeneity-grid",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.25, 1.5],
        help="Values for the reward heterogeneity knob (logit scale, separation, or clusters).",
    )
    parser.add_argument(
        "--arm-heterogeneity-mode",
        choices=["uniform", "sparse", "clustered"],
        default="uniform",
        help="How to inject heterogeneity into the reward CPD.",
    )
    parser.add_argument("--sparse-fraction", type=float, default=0.1, help="Fraction of assignments made distinct in sparse mode.")
    parser.add_argument("--sparse-separation", type=float, default=0.3, help="Base separation for sparse mode (overridden by grid).")
    parser.add_argument("--cluster-count", type=int, default=3, help="Base cluster count for clustered mode (overridden by grid).")
    parser.add_argument("--reward-logit-scale", type=float, default=1.0, help="Base logit scale (used when heterogeneity grid is absent).")
    parser.add_argument(
        "--heterogeneity-hard-margin",
        type=float,
        default=0.0,
        help="Fixed hard_margin to use while sweeping heterogeneity.",
    )

    parser.add_argument("--samples-per-setting", type=int, default=30, help="Number of SCMs per grid value.")
    parser.add_argument("--seed-offset", type=int, default=0, help="Offset applied to generated seeds.")

    parser.add_argument("--gap-mc-samples", type=int, default=4000, help="Monte Carlo draws for gap estimation.")
    parser.add_argument("--gap-alpha", type=float, default=0.05, help="Alpha for Hoeffding shrinkage (only if --gap-shrink).")
    parser.add_argument("--gap-exact-parent-scope", type=int, default=2, help="Exact enumeration cutoff for parent scopes.")
    parser.add_argument("--gap-max-hops", type=int, default=None, help="Optional cap on ancestor subset size when estimating gaps.")
    parser.add_argument("--gap-non-strict", action="store_true", help="Use non-strict gap definition (local parents).")
    parser.add_argument("--gap-shrink", action="store_true", help="Apply Hoeffding shrinkage to gap estimates.")

    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write JSONL records and summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if stats is None:
        raise ImportError("scipy is required for correlation tests; install with `pip install scipy`.")

    records: List[Dict] = []
    records.extend(run_hard_margin_sweep(args))
    records.extend(run_heterogeneity_sweep(args))

    summary = summarize(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "proxy_gap_records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with (args.output_dir / "proxy_gap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    hard = summary.get("hard_margin_vs_epsilon")
    hetero = summary.get("heterogeneity_metric_vs_delta")
    knob = summary.get("heterogeneity_knob_vs_delta")
    if hard:
        print(f"hard_margin vs epsilon: rho={hard['rho']:.3f}, p={hard['p_value']:.4g}, n={hard['n']}")
    if hetero:
        print(f"heterogeneity metric vs Delta: rho={hetero['rho']:.3f}, p={hetero['p_value']:.4g}, n={hetero['n']}")
    if knob:
        print(f"heterogeneity knob vs Delta: rho={knob['rho']:.3f}, p={knob['p_value']:.4g}, n={knob['n']}")
    print(f"Wrote records to {records_path}")


if __name__ == "__main__":
    main()
