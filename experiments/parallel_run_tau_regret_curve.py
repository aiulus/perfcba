"""Parallel wrapper around run_tau_regret_curve to distribute seeds across workers."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from .artifacts import (
    TrialArtifact,
    build_metadata,
    load_trial_artifact,
    make_trial_identity,
    write_trial_artifact,
)
from .causal_envs import CausalBanditConfig
from functools import partial

from .parallel_utils import run_jobs_in_pool, run_trial_worker
from .regret_curves import aggregate_regret_curves, plot_regret_band
from .run_tau_study import (
    SamplingSettings,
    adaptive_config_from_args,
    enrich_record_with_metadata,
    run_trial,
    subset_size_for_known_k,
)
from .structure import compute_effect_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel regret-curve runner.")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers (default: 1).")
    parser.add_argument(
        "--executor",
        choices=["process", "thread"],
        default="process",
        help="Executor type for parallelism (default: process). Use 'thread' if pickling causes issues on Windows.",
    )
    # reuse most args from run_tau_regret_curve
    base_parser = argparse.ArgumentParser(add_help=False)
    from .run_tau_regret_curve import parse_args as base_parse

    # Parse known num-workers, then delegate.
    known, remaining = parser.parse_known_args()
    import sys

    sys.argv = [sys.argv[0]] + remaining
    args = base_parse()
    args.num_workers = max(1, int(known.num_workers))
    return args


def main() -> None:
    args = parse_args()
    if args.effect_threshold is not None and args.effect_threshold_mode != "fixed":
        raise ValueError("--effect-threshold requires --effect-threshold-mode=fixed.")
    seed_start, seed_end = map(int, args.seeds.split(":"))
    seeds = list(range(seed_start, seed_end + 1))
    if not seeds:
        raise ValueError("No seeds specified.")
    m_value = args.m if args.m is not None else args.k
    base_edge_prob = args.edge_prob if args.edge_prob is not None else 2.0 / max(1, args.n)
    cfg = CausalBanditConfig(
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
        hard_margin=args.hard_margin,
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
    horizon = args.T
    subset_size = subset_size_for_known_k(cfg, horizon)
    adaptive_cfg = adaptive_config_from_args(args, horizon)
    adaptive_cfg_dict = dataclasses.asdict(adaptive_cfg) if adaptive_cfg is not None else None
    cli_args_snapshot = vars(args).copy()

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
        structure_mc_samples=_scaled(512),
        arm_mc_samples=_scaled(1024),
        optimal_mean_mc_samples=_scaled(2048),
    )
    effect_threshold_value = compute_effect_threshold(
        min_samples_per_value=sampling.min_samples,
        mode=args.effect_threshold_mode,
        fixed_value=args.effect_threshold,
        scale=args.effect_threshold_scale,
        hoeffding_alpha=args.effect_threshold_alpha,
    )

    artifact_dir: Path = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for seed in seeds:
        fn = partial(
            run_trial_worker,
            cfg=cfg,
            horizon=horizon,
            tau=args.tau,
            seed=seed,
            knob_value=float(getattr(args, "knob_value", 0.0)),
            subset_size=subset_size,
            scheduler_mode=args.scheduler,
            use_full_budget=args.etc_use_full_budget,
            effect_threshold=effect_threshold_value,
            sampling=sampling,
            adaptive_config=adaptive_cfg,
            structure_backend="budgeted_raps" if args.scheduler == "budgeted_raps" else args.scheduler,
            raps_params=None,
            arm_builder_cfg=None,
        )
        jobs.append((str(seed), fn))
    # Execute and collect records directly; no artifact reuse to simplify.
    results_map = run_jobs_in_pool(
        jobs,
        num_workers=args.num_workers,
        show_progress=True,
        executor=getattr(args, "executor", "process"),
    )

    artifacts = []
    for seed in seeds:
        record, summary, optimal_mean = results_map[str(seed)]
        identity = make_trial_identity(
            cfg,
            horizon=horizon,
            tau=float(args.tau),
            seed=seed,
            knob_value=float(getattr(args, "knob_value", 0.0)),
            scheduler=args.scheduler,
            subset_size=subset_size,
            use_full_budget=args.etc_use_full_budget,
            effect_threshold=effect_threshold_value,
            min_samples=sampling.min_samples,
            adaptive_config=adaptive_cfg_dict,
            structure_mc_samples=sampling.structure_mc_samples,
            arm_mc_samples=sampling.arm_mc_samples,
            optimal_mean_mc_samples=sampling.optimal_mean_mc_samples,
        )
        record = enrich_record_with_metadata(
            record,
            summary=summary,
            identity=identity,
            horizon=horizon,
            scheduler=args.scheduler,
        )
        metadata = build_metadata(cli_args={**cli_args_snapshot, "seed": seed})
        artifact = TrialArtifact(
            identity=identity,
            record=record,
            summary=summary,
            optimal_mean=optimal_mean,
            metadata=metadata,
        )
        write_trial_artifact(artifact_dir, artifact)
        artifacts.append(artifact)

    xs, mean, std, matrix = aggregate_regret_curves(artifacts)
    plot_path = args.plot_path or (artifact_dir / "regret_curve.png")
    plot_regret_band(
        xs,
        mean,
        std,
        output_path=plot_path,
        label=args.label or f"tau={args.tau}",
        individual_curves=matrix if args.individual_curves else None,
    )
    stats = {
        "tau": args.tau,
        "seeds": seeds,
        "horizon": horizon,
        "mean_final_regret": float(mean[-1]) if mean.size else 0.0,
        "std_final_regret": float(std[-1]) if std.size else 0.0,
        "num_trials": len(artifacts),
        "subset_size": subset_size,
    }
    stats_path = artifact_dir / "regret_curve_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
