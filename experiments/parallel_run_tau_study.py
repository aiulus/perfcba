"""Parallel wrapper around run_tau_study that keeps per-trial semantics intact."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

from . import grids
from .causal_envs import CausalBanditConfig
from .baselines import estimate_observational_baseline
from functools import partial

from .parallel_utils import run_jobs_in_pool, run_trial_worker
from ..causal_bandits import RAPSParams
from .run_tau_study import (
    KNOB_LABELS,
    METRIC_LABELS,
    PreparedInstance,
    SamplingSettings,
    adaptive_config_from_args,
    aggregate_heatmap_with_std,
    build_random_scm_with_gaps,
    compute_effect_threshold,
    enrich_record_with_metadata,
    plot_heatmap,
    report_heatmap_std,
    subset_size_for_known_k,
)


def parse_args() -> argparse.Namespace:
    # Parse num-workers first, forward the rest to the existing parser.
    parser = argparse.ArgumentParser(description="Parallel tau study")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers (default: 1=serial).")
    parser.add_argument(
        "--executor",
        choices=["process", "thread"],
        default="process",
        help="Executor type for parallelism (default: process). Use 'thread' if pickling causes issues on Windows.",
    )
    # We reuse the original parser for all other args.
    from .run_tau_study import parse_args as base_parse

    known, remaining = parser.parse_known_args()
    # Reconstruct argv for base parser.
    import sys

    argv = [sys.argv[0]] + remaining
    sys.argv = argv
    base_args = base_parse()
    base_args.num_workers = max(1, int(known.num_workers))
    return base_args


def _parse_grid_tokens(tokens: Optional[Sequence[str]], caster: Any) -> Optional[List[Any]]:
    if tokens is None:
        return None
    expanded: List[Any] = []
    for token in tokens:
        if ":" in token:
            parts = token.split(":")
            if len(parts) == 2:
                start, stop = map(float, parts)
                step = 1.0 if stop >= start else -1.0
            else:
                start, step, stop = map(float, parts)
            cur = start
            for _ in range(100000):
                if (step > 0 and cur > stop + 1e-9) or (step < 0 and cur < stop - 1e-9):
                    break
                expanded.append(caster(cur))
                cur += step
        else:
            expanded.append(caster(token))
    return expanded


def apply_knob(cfg: CausalBanditConfig, vary: str, value: float, *, edge_prob_user: bool, args) -> Tuple[CausalBanditConfig, Optional[int], Optional[List[float]]]:
    horizon_override: Optional[int] = None
    tau_override: Optional[List[float]] = None
    if vary == "graph_density":
        cfg = dataclasses.replace(cfg, edge_prob=float(value))
    elif vary == "reward_edge_density":
        cfg = dataclasses.replace(cfg, edge_prob_to_reward=float(value))
    elif vary == "covariate_edge_density":
        cfg = dataclasses.replace(cfg, edge_prob_covariates=float(value))
    elif vary == "node_count":
        new_n = int(value)
        new_m = min(cfg.m, new_n)
        if edge_prob_user:
            cfg = dataclasses.replace(cfg, n=new_n, m=new_m)
        else:
            new_edge_prob = 2.0 / max(1, new_n)
            cfg = dataclasses.replace(
                cfg,
                n=new_n,
                m=new_m,
                edge_prob=new_edge_prob,
                edge_prob_covariates=args.edge_prob_covariates or new_edge_prob,
                edge_prob_to_reward=args.edge_prob_to_reward or new_edge_prob,
            )
    elif vary == "parent_count":
        new_k = int(value)
        cfg = dataclasses.replace(cfg, k=new_k, m=max(new_k, cfg.m))
    elif vary == "intervention_size":
        cfg = dataclasses.replace(cfg, m=int(value))
    elif vary == "alphabet":
        cfg = dataclasses.replace(cfg, ell=int(value))
    elif vary == "arm_variance":
        if cfg.arm_heterogeneity_mode == "sparse":
            cfg = dataclasses.replace(cfg, sparse_fraction=float(value))
        elif cfg.arm_heterogeneity_mode == "clustered":
            cfg = dataclasses.replace(cfg, cluster_count=int(value))
        else:
            cfg = dataclasses.replace(cfg, reward_logit_scale=float(value))
    elif vary == "hard_margin":
        cfg = dataclasses.replace(cfg, hard_margin=float(value))
    elif vary == "horizon":
        horizon_override = int(value)
    elif vary == "tau":
        tau_override = [float(value)]
    return cfg, horizon_override, tau_override


def main() -> None:
    args = parse_args()
    if args.effect_threshold is not None and args.effect_threshold_mode != "fixed":
        raise ValueError("--effect-threshold requires --effect-threshold-mode=fixed.")

    # Preprocess grids.
    args.parent_grid = _parse_grid_tokens(args.parent_grid, int)
    args.graph_grid = _parse_grid_tokens(args.graph_grid, float)
    args.node_grid = _parse_grid_tokens(args.node_grid, int)
    args.intervention_grid = _parse_grid_tokens(args.intervention_grid, int)
    args.algo_eps_grid = _parse_grid_tokens(args.algo_eps_grid, float)
    args.algo_delta_grid = _parse_grid_tokens(args.algo_delta_grid, float)
    args.sparse_fraction_grid = _parse_grid_tokens(args.sparse_fraction_grid, float)
    args.cluster_count_grid = _parse_grid_tokens(args.cluster_count_grid, int)
    if args.env_vary is not None:
        if args.env_grid is None:
            raise ValueError("--env-vary requires --env-grid.")
        parser_type = int if args.env_vary in {"node_count", "parent_count", "intervention_size", "alphabet"} else float
        args.env_grid = _parse_grid_tokens(args.env_grid, parser_type)

    seed_start, seed_end = map(int, args.seeds.split(":"))
    seeds = list(range(seed_start, seed_end + 1))
    m_value = args.m if args.m is not None else args.k
    edge_prob_user = args.edge_prob is not None
    base_edge_prob = args.edge_prob if args.edge_prob is not None else 2.0 / max(1, args.n)
    # Mirror hard_margin handling from run_tau_study.
    if args.hard_margin is None:
        base_hard_margin = 0.0
    elif len(args.hard_margin) == 0:
        base_hard_margin = 0.1
    else:
        if len(args.hard_margin) != 1:
            raise ValueError("Specify a single --hard-margin value when not varying hard_margin.")
        base_hard_margin = float(args.hard_margin[0])
    base_cfg = CausalBanditConfig(
        n=args.n,
        ell=args.ell,
        k=args.k,
        m=m_value,
        edge_prob=base_edge_prob,
        edge_prob_covariates=args.edge_prob_covariates,
        edge_prob_to_reward=args.edge_prob_to_reward,
        scm_mode=args.scm_mode,
        parent_effect=args.parent_effect,
        reward_logit_scale=args.reward_logit_scale,
        hard_margin=base_hard_margin,
        scm_epsilon=args.scm_epsilon,
        scm_delta=args.scm_delta,
        arm_heterogeneity_mode=args.arm_heterogeneity_mode,
        sparse_fraction=args.sparse_fraction,
        sparse_separation=args.sparse_separation,
        cluster_count=args.cluster_count,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        gap_enforcement_mode=args.gap_enforcement_mode,
        max_rejection_attempts=args.max_rejection_attempts,
    )

    # Knob values.
    if args.vary == "parent_count" and args.parent_grid:
        knob_values = [int(value) for value in args.parent_grid]
    elif args.vary == "graph_density" and args.graph_grid:
        knob_values = [float(value) for value in args.graph_grid]
    elif args.vary == "reward_edge_density" and args.graph_grid:
        knob_values = [float(value) for value in args.graph_grid]
    elif args.vary == "covariate_edge_density" and args.graph_grid:
        knob_values = [float(value) for value in args.graph_grid]
    elif args.vary == "node_count" and args.node_grid:
        knob_values = [int(value) for value in args.node_grid]
    elif args.vary == "intervention_size" and args.intervention_grid:
        knob_values = [int(value) for value in args.intervention_grid]
    elif args.vary == "algo_eps" and args.algo_eps_grid:
        knob_values = [float(value) for value in args.algo_eps_grid]
    elif args.vary == "algo_delta" and args.algo_delta_grid:
        knob_values = [float(value) for value in args.algo_delta_grid]
    elif args.vary == "arm_variance":
        if args.arm_heterogeneity_mode == "sparse" and args.sparse_fraction_grid:
            knob_values = [float(value) for value in args.sparse_fraction_grid]
        elif args.arm_heterogeneity_mode == "clustered" and args.cluster_count_grid:
            knob_values = [int(value) for value in args.cluster_count_grid]
        else:
            knob_values = grids.grid_values(args.vary, n=args.n, k=args.k)
    elif args.vary == "hard_margin":
        knob_values = [float(value) for value in (args.hard_margin or [0.0])]
    elif args.vary == "tau":
        knob_values = [float(value) for value in args.tau_grid]
    else:
        knob_values = grids.grid_values(args.vary, n=args.n, k=args.k)

    if args.env_vary is not None:
        env_values = [float(val) for val in args.env_grid]
    else:
        env_values = [None]

    # Sampling scale
    flag_count = sum(bool(flag) for flag in (args.small, args.very_small, args.tiny))
    if flag_count > 1:
        raise ValueError("Only one of --small/--very-small/--tiny may be specified.")
    if args.tiny:
        sample_scale = 1.0 / 16.0
    elif args.very_small:
        sample_scale = 1.0 / 8.0
    elif args.small:
        sample_scale = 0.25
    else:
        sample_scale = 1.0

    def _scaled(value: int) -> int:
        return max(1, int(math.ceil(value * sample_scale)))

    sampling = SamplingSettings(
        min_samples=max(1, int(math.ceil(args.min_samples * sample_scale))),
        structure_mc_samples=_scaled(args.structure_mc_samples),
        arm_mc_samples=_scaled(args.arm_mc_samples),
        optimal_mean_mc_samples=_scaled(args.optimal_mean_mc_samples),
    )
    effect_threshold_value = compute_effect_threshold(
        min_samples_per_value=sampling.min_samples,
        mode=args.effect_threshold_mode,
        fixed_value=args.effect_threshold,
        scale=args.effect_threshold_scale,
        hoeffding_alpha=args.effect_threshold_alpha,
    )

    base_algo_eps = args.algo_eps
    base_algo_delta = args.algo_delta

    # Build job list.
    jobs: List[Tuple[str, Any]] = []
    ordered_records: List[Tuple[float, float, int, Dict[str, Any]]] = []
    progress = tqdm(total=len(env_values) * len(knob_values) * len(args.tau_grid) * len(seeds), desc="Tau study", unit="trial")

    for env_value in env_values:
        cfg_env, env_horizon_override, env_tau_override = apply_knob(base_cfg, args.env_vary, env_value, edge_prob_user=edge_prob_user, args=args) if args.env_vary else (base_cfg, None, None)
        for knob_value in knob_values:
            cfg, knob_horizon_override, knob_tau_override = apply_knob(cfg_env, args.vary, knob_value, edge_prob_user=edge_prob_user, args=args)
            current_horizon = env_horizon_override if env_horizon_override is not None else args.T
            if knob_horizon_override is not None:
                current_horizon = knob_horizon_override
            if knob_tau_override is not None:
                tau_iter = knob_tau_override
            elif env_tau_override is not None:
                tau_iter = env_tau_override
            else:
                tau_iter = list(args.tau_grid)

            subset_size = subset_size_for_known_k(cfg, current_horizon)
            adaptive_cfg = adaptive_config_from_args(args, current_horizon)
            adaptive_cfg_dict = dataclasses.asdict(adaptive_cfg) if adaptive_cfg is not None else None
            arm_builder_cfg = None  # parallel path does not use hybrid arms currently
            current_eps = float(base_algo_eps)
            current_reward_delta = float(base_algo_delta)
            if args.vary == "algo_eps":
                current_eps = float(knob_value)
            elif args.vary == "algo_delta":
                current_reward_delta = float(knob_value)
            raps_params_for_knob: Optional[RAPSParams] = None
            if args.structure_backend == "budgeted_raps":
                raps_params_for_knob = RAPSParams(
                    eps=current_eps,
                    Delta=current_reward_delta,
                    delta=args.raps_delta,
                )

            for tau in tau_iter:
                for seed in seeds:
                    job_id = f"{env_value}_{knob_value}_{tau}_{seed}"
                    fn = partial(
                        run_trial_worker,
                        cfg=cfg,
                        horizon=current_horizon,
                        tau=tau,
                        strict_tau=args.strict_tau,
                        seed=seed,
                        knob_value=float(knob_value),
                        subset_size=subset_size,
                        scheduler_mode=args.scheduler,
                        use_full_budget=args.etc_use_full_budget,
                        effect_threshold=effect_threshold_value,
                        sampling=sampling,
                        adaptive_config=adaptive_cfg,
                        structure_backend=args.structure_backend,  # type: ignore[arg-type]
                        raps_params=raps_params_for_knob,
                        arm_builder_cfg=arm_builder_cfg,
                    )
                    jobs.append((job_id, fn))
                progress.update(len(seeds))
    progress.close()

    # Execute in pool.
    results_map = run_jobs_in_pool(
        jobs,
        num_workers=args.num_workers,
        show_progress=True,
        executor=getattr(args, "executor", "process"),
    )

    # Aggregate records.
    records: List[Dict[str, Any]] = []
    for env_value in env_values:
        for knob_value in knob_values:
            for tau in (env_tau_override or list(args.tau_grid)):
                for seed in seeds:
                    job_id = f"{env_value}_{knob_value}_{tau}_{seed}"
                    record, summary, optimal_mean = results_map[job_id]
                    identity = None  # identity digest not needed for aggregation here
                    record = enrich_record_with_metadata(
                        record,
                        summary=summary,
                        identity=None,  # type: ignore[arg-type]
                        horizon=current_horizon,
                        scheduler=args.scheduler if args.structure_backend == "proxy" else "budgeted_raps",
                    )
                    records.append(record)

    # Write JSONL
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Heatmaps
    tau_values = list(map(float, args.tau_grid))
    knob_label, knob_label_plural = KNOB_LABELS.get(
        args.vary, (args.vary.replace("_", " ").title(), f"{args.vary.replace('_', ' ')}s")
    )
    metric_label = METRIC_LABELS.get(args.metric, args.metric.replace("_", " ").capitalize())
    matrix, std_matrix, counts = aggregate_heatmap_with_std(records, tau_values, knob_values, args.metric)
    graph_success_matrix = np.zeros_like(matrix)
    overlay_mask = graph_success_matrix >= 0.5
    report_heatmap_std(
        matrix,
        std_matrix,
        counts,
        metric_label=metric_label,
        tau_values=tau_values,
        knob_values=knob_values,
    )
    obs_baseline = estimate_observational_baseline(
        records, args.metric, mc_samples=args.optimal_mean_mc_samples
    )
    plot_heatmap(
        matrix,
        tau_values=tau_values,
        knob_values=knob_values,
        title=f"{metric_label} for varying {knob_label_plural}",
        cbar_label=metric_label,
        x_label=knob_label,
        output_path=args.output_dir / f"heatmap_{args.metric}.png",
        colorbar_marker=obs_baseline,
    )
    plot_heatmap(
        matrix,
        tau_values=tau_values,
        knob_values=knob_values,
        title=f"{metric_label} for varying {knob_label_plural} (structure overlay)",
        cbar_label=metric_label,
        x_label=knob_label,
        output_path=args.output_dir / f"overlayed_heatmap_{args.metric}.png",
        overlay_mask=overlay_mask,
        colorbar_marker=obs_baseline,
    )


if __name__ == "__main__":
    main()
