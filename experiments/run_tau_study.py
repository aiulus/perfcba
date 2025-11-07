"""CLI entry point for the tau-scheduled causal bandit study."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .causal_envs import CausalBanditConfig, InterventionSpace, build_random_scm
from .exploit import ArmBuilder, ParentAwareUCB
from .grids import TAU_GRID, grid_values
from .heatmap import plot_heatmap
from .metrics import summarize
from .scheduler import TauScheduler
from .structure import RAPSLearner, StructureConfig


def subset_size_for_known_k(cfg: CausalBanditConfig, horizon: int) -> int:
    ell = cfg.ell
    n = cfg.n
    k = cfg.k
    m = cfg.m
    if m < k:
        return min(math.comb(n, m) * (ell**m), 10_000)
    term = (ell**k) * math.comb(n, k) / max(1, math.comb(m, k))
    cap = (ell**m) * math.comb(n, m)
    n0 = int(min(term * max(1.0, math.log(math.sqrt(horizon))), cap))
    return max(ell, min(n0, cap))


def compute_optimal_mean(
    instance,
    rng: np.random.Generator,
    *,
    mc_samples: int = 2048,
) -> float:
    ell = instance.config.ell
    parent_indices = instance.parent_indices()
    best = 0.0
    max_size = min(instance.config.m, len(parent_indices))
    for size in range(0, max_size + 1):
        for subset in combinations(parent_indices, size):
            for assignment in product(range(ell), repeat=size):
                mean = instance.estimate_subset_mean(subset, assignment, rng, mc_samples)
                best = max(best, mean)
    return best


def run_trial(
    *,
    base_cfg: CausalBanditConfig,
    horizon: int,
    tau: float,
    seed: int,
    knob_value: float,
    subset_size: int,
    scheduler_mode: str,
    effect_threshold: float,
    min_samples: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    instance = build_random_scm(base_cfg, rng=rng)
    optimal_mean = compute_optimal_mean(instance, rng)
    space = InterventionSpace(
        instance.config.n,
        instance.config.ell,
        instance.config.m,
        include_lower=False,
    )
    structure = RAPSLearner(
        instance,
        StructureConfig(effect_threshold=effect_threshold, min_samples_per_value=min_samples),
    )
    policy = ParentAwareUCB()
    arm_builder = ArmBuilder(
        instance,
        space,
        subset_size=subset_size,
        mc_samples=1024,
    )
    scheduler = TauScheduler(
        instance=instance,
        structure=structure,
        arm_builder=arm_builder,
        policy=policy,
        tau=tau,
        horizon=horizon,
        mode=scheduler_mode,  # type: ignore[arg-type]
        optimal_mean=optimal_mean,
    )
    summary = scheduler.run(rng)
    metrics = summarize(summary.logs, optimal_mean)
    return {
        "tau": tau,
        "knob_value": knob_value,
        "seed": seed,
        "cumulative_regret": metrics.cumulative_regret,
        "tto": metrics.time_to_optimality,
        "optimal_rate": metrics.optimal_action_rate,
        "structure_steps": summary.structure_steps,
        "parents_found": len(summary.final_parent_set),
    }


def aggregate_heatmap(
    results: Sequence[Dict[str, float]],
    tau_values: Sequence[float],
    knob_values: Sequence[float],
    metric_key: str,
) -> np.ndarray:
    matrix = np.zeros((len(tau_values), len(knob_values)))
    counts = np.zeros_like(matrix)
    tau_index = {tau: idx for idx, tau in enumerate(tau_values)}
    knob_index = {kv: idx for idx, kv in enumerate(knob_values)}
    for record in results:
        t_idx = tau_index[record["tau"]]
        k_idx = knob_index[record["knob_value"]]
        matrix[t_idx, k_idx] += record[metric_key]
        counts[t_idx, k_idx] += 1
    counts[counts == 0] = 1
    return matrix / counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tau-scheduled causal bandit study.")
    parser.add_argument(
        "--vary",
        choices=["graph_density", "parent_count", "intervention_size", "alphabet", "horizon"],
        required=True,
        help="Environment knob to sweep.",
    )
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--T", type=int, default=10_000)
    parser.add_argument("--tau-grid", type=float, nargs="*", default=TAU_GRID)
    parser.add_argument("--seeds", type=str, default="0:9", help="Seed range start:end.")
    parser.add_argument("--scheduler", choices=["interleaved", "two_phase"], default="interleaved")
    parser.add_argument("--output-dir", type=Path, default=Path("results/tau_study"))
    parser.add_argument("--effect-threshold", type=float, default=0.05)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--metric", choices=["cumulative_regret", "tto"], default="cumulative_regret")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_start, seed_end = map(int, args.seeds.split(":"))
    seeds = list(range(seed_start, seed_end + 1))
    m_value = args.m if args.m is not None else args.k
    base_cfg = CausalBanditConfig(
        n=args.n,
        ell=args.ell,
        k=args.k,
        m=m_value,
        edge_prob=2.0 / max(1, args.n),
    )
    knob_values = grid_values(args.vary, n=args.n, k=args.k)
    results: List[Dict[str, float]] = []

    for knob_value in knob_values:
        cfg = base_cfg
        if args.vary == "graph_density":
            cfg = CausalBanditConfig(
                n=cfg.n,
                ell=cfg.ell,
                k=cfg.k,
                m=cfg.m,
                edge_prob=float(knob_value),
            )
        elif args.vary == "parent_count":
            cfg = CausalBanditConfig(
                n=cfg.n,
                ell=cfg.ell,
                k=int(knob_value),
                m=max(int(knob_value), cfg.m),
                edge_prob=cfg.edge_prob,
            )
        elif args.vary == "intervention_size":
            cfg = CausalBanditConfig(
                n=cfg.n,
                ell=cfg.ell,
                k=cfg.k,
                m=int(knob_value),
                edge_prob=cfg.edge_prob,
            )
        elif args.vary == "alphabet":
            cfg = CausalBanditConfig(
                n=cfg.n,
                ell=int(knob_value),
                k=cfg.k,
                m=cfg.m,
                edge_prob=cfg.edge_prob,
            )
        elif args.vary == "horizon":
            pass  # handled via args.T when running trials

        current_horizon = args.T if args.vary != "horizon" else int(knob_value)
        subset_size = subset_size_for_known_k(cfg, current_horizon)
        for tau in args.tau_grid:
            for seed in seeds:
                record = run_trial(
                    base_cfg=cfg,
                    horizon=current_horizon,
                    tau=tau,
                    seed=seed,
                    knob_value=float(knob_value),
                    subset_size=subset_size,
                    scheduler_mode=args.scheduler,
                    effect_threshold=args.effect_threshold,
                    min_samples=args.min_samples,
                )
                results.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    tau_values = args.tau_grid
    matrix = aggregate_heatmap(results, tau_values, knob_values, args.metric)
    plot_heatmap(
        matrix,
        tau_values=tau_values,
        knob_values=knob_values,
        title=f"{args.metric} heatmap ({args.vary})",
        cbar_label=args.metric,
        output_path=args.output_dir / f"heatmap_{args.metric}.png",
    )


if __name__ == "__main__":
    main()
