"""CLI entry point for the tau-scheduled causal bandit study."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

from .artifacts import (
    TrialArtifact,
    TrialIdentity,
    build_metadata,
    load_trial_artifact,
    make_trial_identity,
    trial_identity_digest,
    write_trial_artifact,
)
from .causal_envs import CausalBanditConfig, InterventionSpace, build_random_scm
from .exploit import ArmBuilder, ParentAwareUCB
from .grids import TAU_GRID, grid_values
from .heatmap import plot_heatmap
from .metrics import summarize
from .scheduler import AdaptiveBurstConfig, RunSummary, build_scheduler
from .timeline import encode_schedule, plot_time_allocation
from .structure import RAPSLearner, StructureConfig


KNOB_LABELS = {
    "graph_density": ("Graph Density", "graph densities"),
    "parent_count": ("Parent Count", "parent counts"),
    "intervention_size": ("Intervention Size", "intervention sizes"),
    "alphabet": ("Alphabet Size", "alphabet sizes"),
    "horizon": ("Horizon", "horizons"),
    "arm_variance": ("Arm Variance Scale", "arm variance scales"),
}

METRIC_LABELS = {
    "cumulative_regret": "Cumulative regret",
    "tto": "Time to optimality",
}


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


def _format_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4g}"


def adaptive_config_from_args(
    args: argparse.Namespace,
    horizon: int,
) -> Optional[AdaptiveBurstConfig]:
    if args.scheduler != "adaptive_burst":
        return None
    window = args.ab_window if args.ab_window is not None else max(5, int(0.05 * max(1, horizon)))
    stall_min = args.ab_stall_min if args.ab_stall_min is not None else window
    cooldown = args.ab_cooldown
    return AdaptiveBurstConfig(
        start_mode=args.ab_start_mode,
        initial_burst=max(1, args.ab_x0),
        growth_factor=max(1.0, args.ab_gamma),
        window=window,
        stall_min_exploit=stall_min,
        metric=args.ab_metric,
        eta_down=args.ab_eta_down,
        eta_up=args.ab_eta_up,
        reset_mode=args.ab_x_reset,
        cooldown=cooldown,
        tail_fraction=max(0.0, args.ab_tail_frac),
        opt_rate_tolerance=args.ab_opt_gap,
        ewma_lambda=args.ab_ewma_lam,
        enable_page_hinkley=args.ab_enable_ph,
        ph_delta=args.ab_ph_delta,
        ph_lambda=args.ab_ph_lambda,
        ph_alpha=args.ab_ph_alpha,
    )


def run_trial(
    *,
    base_cfg: CausalBanditConfig,
    horizon: int,
    tau: float,
    seed: int,
    knob_value: float,
    subset_size: int,
    scheduler_mode: str,
    use_full_budget: bool,
    effect_threshold: float,
    min_samples: int,
    adaptive_config: Optional[AdaptiveBurstConfig],
) -> Tuple[Dict[str, Any], RunSummary, float]:
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
    scheduler = build_scheduler(
        mode=scheduler_mode,  # type: ignore[arg-type]
        instance=instance,
        structure=structure,
        arm_builder=arm_builder,
        policy=policy,
        tau=tau,
        horizon=horizon,
        use_full_budget=use_full_budget,
        adaptive_config=adaptive_config,
    )
    summary = scheduler.run(rng)
    metrics = summarize(summary.logs, optimal_mean)
    finished_round = summary.finished_discovery_round
    record: Dict[str, Any] = {
        "tau": tau,
        "knob_value": knob_value,
        "seed": seed,
        "cumulative_regret": metrics.cumulative_regret,
        "tto": metrics.time_to_optimality,
        "optimal_rate": metrics.optimal_action_rate,
        "structure_steps": summary.structure_steps,
        "parents_found": len(summary.final_parent_set),
        "finished_discovery": finished_round is not None,
        "finished_discovery_round": finished_round,
        "horizon": horizon,
        "scheduler": scheduler_mode,
    }
    return record, summary, optimal_mean


def enrich_record_with_metadata(
    record: Dict[str, Any],
    *,
    summary: RunSummary,
    identity: TrialIdentity,
    horizon: int,
    scheduler: str,
) -> Dict[str, Any]:
    """Ensure each JSONL row carries the shared identifiers used by analysis."""
    enriched = dict(record)
    finished_round = summary.finished_discovery_round
    enriched["finished_discovery"] = finished_round is not None
    enriched["finished_discovery_round"] = int(finished_round) if finished_round is not None else None
    enriched["instance_id"] = trial_identity_digest(identity)
    enriched["horizon"] = int(horizon)
    enriched["scheduler"] = str(scheduler)
    enriched["tau"] = float(identity.tau)
    enriched["knob_value"] = float(identity.knob_value)
    enriched["seed"] = int(identity.seed)
    return enriched


def aggregate_heatmap(
    results: Sequence[Dict[str, Any]],
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
        choices=["graph_density", "parent_count", "intervention_size", "alphabet", "horizon", "arm_variance"],
        required=True,
        help="Environment knob to sweep.",
    )
    parser.add_argument(
        "--parent-grid",
        type=int,
        nargs="+",
        default=None,
        help="Override the default parent counts when --vary parent_count is used.",
    )
    parser.add_argument(
        "--graph-grid",
        type=float,
        nargs="+",
        default=None,
        help="Override the default edge probabilities when --vary graph_density is used.",
    )
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--ell", type=int, default=2)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument(
        "--scm-mode",
        choices=["beta_dirichlet", "reference"],
        default="beta_dirichlet",
        help="SCM generation scheme for the sampled environments.",
    )
    parser.add_argument(
        "--parent-effect",
        type=float,
        default=1.0,
        help="Reference-mode mixing coefficient between base and parent-specific CPDs.",
    )
    parser.add_argument(
        "--reward-logit-scale",
        type=float,
        default=1.0,
        help="Scale applied to Bernoulli logits to control per-arm variance.",
    )
    parser.add_argument("--T", type=int, default=10_000)
    parser.add_argument("--tau-grid", type=float, nargs="*", default=TAU_GRID)
    parser.add_argument("--seeds", type=str, default="0:9", help="Seed range start:end.")
    parser.add_argument(
        "--scheduler",
        choices=["interleaved", "two_phase", "etc", "adaptive_burst"],
        default="two_phase",
    )
    parser.add_argument(
        "--etc-use-full-budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If false, the ETC scheduler commits early once structure learning is done.",
    )
    parser.add_argument(
        "--ab-start-mode",
        choices=["exploit_first", "explore_first"],
        default="exploit_first",
        help="(adaptive_burst) Start in exploitation or exploration.",
    )
    parser.add_argument(
        "--ab-x0",
        type=int,
        default=1,
        help="(adaptive_burst) Initial exploration burst size.",
    )
    parser.add_argument(
        "--ab-gamma",
        type=float,
        default=2.0,
        help="(adaptive_burst) Burst growth factor.",
    )
    parser.add_argument(
        "--ab-window",
        type=int,
        default=None,
        help="(adaptive_burst) Exploitation window for stall detection.",
    )
    parser.add_argument(
        "--ab-stall-min",
        type=int,
        default=None,
        help="(adaptive_burst) Minimum exploitation rounds before stall checks.",
    )
    parser.add_argument(
        "--ab-cooldown",
        type=int,
        default=None,
        help="(adaptive_burst) Cooldown (in exploit rounds) after a burst.",
    )
    parser.add_argument(
        "--ab-metric",
        choices=["reward", "regret", "opt_rate"],
        default="reward",
        help="(adaptive_burst) Statistic used for stall detection.",
    )
    parser.add_argument(
        "--ab-eta-down",
        type=float,
        default=-0.15,
        help="(adaptive_burst) Trigger threshold (normalized gain).",
    )
    parser.add_argument(
        "--ab-eta-up",
        type=float,
        default=0.1,
        help="(adaptive_burst) Release threshold (normalized gain).",
    )
    parser.add_argument(
        "--ab-x-reset",
        choices=["one", "x0"],
        default="one",
        help="(adaptive_burst) Burst size after improvement resumes.",
    )
    parser.add_argument(
        "--ab-tail-frac",
        type=float,
        default=0.25,
        help="(adaptive_burst) Max fraction of remaining rounds spent in one burst.",
    )
    parser.add_argument(
        "--ab-opt-gap",
        type=float,
        default=0.01,
        help="(adaptive_burst) Tolerance for opt_rate metric.",
    )
    parser.add_argument(
        "--ab-ewma-lam",
        type=float,
        default=0.2,
        help="(adaptive_burst) EWMA smoothing parameter.",
    )
    parser.add_argument(
        "--ab-enable-ph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="(adaptive_burst) Enable Page–Hinkley guard.",
    )
    parser.add_argument("--ab-ph-delta", type=float, default=1e-3, help="(adaptive_burst) Page–Hinkley δ.")
    parser.add_argument("--ab-ph-lambda", type=float, default=0.05, help="(adaptive_burst) Page–Hinkley λ.")
    parser.add_argument("--ab-ph-alpha", type=float, default=0.1, help="(adaptive_burst) Page–Hinkley α.")
    parser.add_argument(
        "--timeline-dir",
        type=Path,
        default=None,
        help="If set, write per-run and per-sweep time allocation diagrams to this directory.",
    )
    parser.add_argument(
        "--timeline-max-columns",
        type=int,
        default=2000,
        help="Max columns to plot in time allocation diagrams (longer horizons are downsampled).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/tau_study"))
    parser.add_argument("--effect-threshold", type=float, default=0.05)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--metric", choices=["cumulative_regret", "tto"], default="cumulative_regret")
    parser.add_argument(
        "--persist-trials",
        type=Path,
        default=None,
        help="Optional directory for storing per-trial artifacts (disabled by default).",
    )
    parser.add_argument(
        "--reuse-trials",
        type=Path,
        default=None,
        help="Optional directory containing cached trial artifacts to reuse.",
    )
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
        scm_mode=args.scm_mode,
        parent_effect=args.parent_effect,
        reward_logit_scale=args.reward_logit_scale,
    )
    if args.vary == "parent_count" and args.parent_grid:
        knob_values = [int(value) for value in args.parent_grid]
    elif args.vary == "graph_density" and args.graph_grid:
        knob_values = [float(value) for value in args.graph_grid]
    else:
        knob_values = grid_values(args.vary, n=args.n, k=args.k)
    results: List[Dict[str, Any]] = []
    total_trials = len(knob_values) * len(args.tau_grid) * len(seeds)
    progress = tqdm(total=total_trials, desc="Tau study", unit="trial")
    timeline_dir: Optional[Path] = args.timeline_dir
    timeline_store = defaultdict(list) if timeline_dir is not None else None
    if timeline_dir is not None:
        timeline_dir.mkdir(parents=True, exist_ok=True)

    persist_dir: Optional[Path] = args.persist_trials
    reuse_dir: Optional[Path] = args.reuse_trials
    cli_args_snapshot = vars(args).copy()

    try:
        for knob_value in knob_values:
            cfg = base_cfg
            if args.vary == "graph_density":
                cfg = dataclasses.replace(cfg, edge_prob=float(knob_value))
            elif args.vary == "parent_count":
                new_k = int(knob_value)
                cfg = dataclasses.replace(cfg, k=new_k, m=max(new_k, cfg.m))
            elif args.vary == "intervention_size":
                cfg = dataclasses.replace(cfg, m=int(knob_value))
            elif args.vary == "alphabet":
                cfg = dataclasses.replace(cfg, ell=int(knob_value))
            elif args.vary == "horizon":
                pass  # handled via args.T when running trials
            elif args.vary == "arm_variance":
                cfg = dataclasses.replace(cfg, reward_logit_scale=float(knob_value))

            current_horizon = args.T if args.vary != "horizon" else int(knob_value)
            subset_size = subset_size_for_known_k(cfg, current_horizon)
            adaptive_cfg = adaptive_config_from_args(args, current_horizon)
            adaptive_cfg_dict = dataclasses.asdict(adaptive_cfg) if adaptive_cfg is not None else None

            for tau in args.tau_grid:
                for seed in seeds:
                    identity = make_trial_identity(
                        cfg,
                        horizon=current_horizon,
                        tau=float(tau),
                        seed=seed,
                        knob_value=float(knob_value),
                        scheduler=args.scheduler,
                        subset_size=subset_size,
                        use_full_budget=args.etc_use_full_budget,
                        effect_threshold=args.effect_threshold,
                        min_samples=args.min_samples,
                        adaptive_config=adaptive_cfg_dict,
                    )

                    artifact = load_trial_artifact(reuse_dir, identity) if reuse_dir is not None else None
                    ran_trial = artifact is None

                    if artifact is not None:
                        record = artifact.record
                        summary = artifact.summary
                        optimal_mean = artifact.optimal_mean
                    else:
                        record, summary, optimal_mean = run_trial(
                            base_cfg=cfg,
                            horizon=current_horizon,
                            tau=tau,
                            seed=seed,
                            knob_value=float(knob_value),
                            subset_size=subset_size,
                            scheduler_mode=args.scheduler,
                            use_full_budget=args.etc_use_full_budget,
                            effect_threshold=args.effect_threshold,
                            min_samples=args.min_samples,
                            adaptive_config=adaptive_cfg,
                        )

                    record = enrich_record_with_metadata(
                        record,
                        summary=summary,
                        identity=identity,
                        horizon=current_horizon,
                        scheduler=args.scheduler,
                    )

                    if persist_dir is not None and ran_trial:
                        metadata = build_metadata(
                            cli_args={
                                **cli_args_snapshot,
                                "tau": float(tau),
                                "seed": int(seed),
                                "knob_value": float(knob_value),
                            }
                        )
                        trial_artifact = TrialArtifact(
                            identity=identity,
                            record=record,
                            summary=summary,
                            optimal_mean=optimal_mean,
                            metadata=metadata,
                        )
                        write_trial_artifact(persist_dir, trial_artifact)

                    results.append(record)
                    if timeline_dir is not None:
                        schedule = encode_schedule(summary.logs)
                        key = (float(knob_value), float(tau))
                        timeline_store[key].append((seed, schedule))
                        per_seed_path = timeline_dir / f"timeline_knob-{_format_value(float(knob_value))}_tau-{_format_value(float(tau))}_seed-{seed}.png"
                        plot_time_allocation(
                            schedule,
                            per_seed_path,
                            title=f"knob={_format_value(float(knob_value))}, tau={_format_value(float(tau))}, seed={seed}",
                            yticklabels=[f"seed {seed}"],
                            max_columns=args.timeline_max_columns,
                        )
                    progress.set_postfix(knob=knob_value, tau=tau, seed=seed)
                    progress.update(1)
    finally:
        progress.close()

    if timeline_dir is not None and timeline_store:
        for (knob_value, tau), rows in timeline_store.items():
            rows.sort(key=lambda item: item[0])
            max_len = max(arr.shape[0] for _, arr in rows)
            matrix = np.zeros((len(rows), max_len), dtype=np.float32)
            labels: List[str] = []
            for idx, (seed, arr) in enumerate(rows):
                fill_value = float(arr[-1]) if arr.size else 0.0
                matrix[idx, : arr.shape[0]] = arr
                if arr.shape[0] < max_len:
                    matrix[idx, arr.shape[0] :] = fill_value
                labels.append(f"seed {seed}")
            agg_path = timeline_dir / f"timeline_knob-{_format_value(knob_value)}_tau-{_format_value(tau)}_grid.png"
            plot_time_allocation(
                matrix,
                agg_path,
                title=f"knob={_format_value(knob_value)}, tau={_format_value(tau)}",
                yticklabels=labels,
                max_columns=args.timeline_max_columns,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    tau_values = args.tau_grid
    matrix = aggregate_heatmap(results, tau_values, knob_values, args.metric)
    knob_label, knob_label_plural = KNOB_LABELS.get(
        args.vary, (args.vary.replace("_", " ").title(), f"{args.vary.replace('_', ' ')}s")
    )
    metric_label = METRIC_LABELS.get(args.metric, args.metric.replace("_", " ").capitalize())
    plot_heatmap(
        matrix,
        tau_values=tau_values,
        knob_values=knob_values,
        title=f"{metric_label} for varying {knob_label_plural}",
        cbar_label=metric_label,
        x_label=knob_label,
        output_path=args.output_dir / f"heatmap_{args.metric}.png",
    )


if __name__ == "__main__":
    main()
