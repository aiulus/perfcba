"""Reduced tau-study driver producing classical cumulative regret curves."""

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
from .regret_curves import aggregate_regret_curves, plot_regret_band
from .run_tau_study import (
    adaptive_config_from_args,
    enrich_record_with_metadata,
    run_trial,
    SamplingSettings,
    subset_size_for_known_k,
)
from .structure import compute_effect_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a single tau configuration and plot classical regret curves."
    )
    parser.add_argument("--n", type=int, default=10)
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
    parser.add_argument("--T", type=int, default=10_000, help="Horizon length.")
    parser.add_argument("--tau", type=float, required=True, help="Tau budget for structure learning.")
    parser.add_argument("--knob-value", type=float, default=0.0, help="Metadata label recorded in artifacts.")
    parser.add_argument("--seeds", type=str, default="0:9", help="Seed range start:end (inclusive).")
    parser.add_argument(
        "--scheduler",
        choices=["interleaved", "two_phase", "etc", "adaptive_burst"],
        default="interleaved",
    )
    parser.add_argument(
        "--etc-use-full-budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If false, the ETC scheduler commits early once structure learning is done.",
    )
    parser.add_argument(
        "--effect-threshold",
        type=float,
        default=None,
        help="Fixed effect threshold (requires --effect-threshold-mode=fixed).",
    )
    parser.add_argument(
        "--effect-threshold-mode",
        choices=["scale", "hoeffding", "fixed"],
        default="scale",
        help="How to set the structure effect threshold (default: scale-based).",
    )
    parser.add_argument(
        "--effect-threshold-scale",
        type=float,
        default=0.75,
        help="c in c * sqrt(1/min_samples) when mode=scale.",
    )
    parser.add_argument(
        "--effect-threshold-alpha",
        type=float,
        default=0.05,
        help="α in sqrt(0.5 * ln(2/α) / min_samples) when mode=hoeffding.",
    )
    parser.add_argument("--min-samples", type=int, default=20, help="Structure learner samples per value.")
    parser.add_argument(
        "--small",
        action="store_true",
        help="Reduce min_samples and Monte Carlo sample counts by 4x for faster (but noisier) runs.",
    )
    parser.add_argument(
        "--very-small",
        action="store_true",
        help="Reduce min_samples/MC counts by 8x (even faster, more noise).",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Reduce min_samples/MC counts by 16x (fastest, highest noise).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory where trial artifacts are stored/loaded.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Destination path for the regret curve figure (PNG). Defaults to artifact-dir/regret_curve.png",
    )
    parser.add_argument("--label", type=str, default=None, help="Legend label for the plotted curve.")
    parser.add_argument(
        "--plot-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip simulations and only generate plots from existing artifacts.",
    )
    parser.add_argument(
        "--individual-curves",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overlay individual seed trajectories on top of the mean curve.",
    )
    # Adaptive-burst knobs
    parser.add_argument("--ab-start-mode", choices=["exploit_first", "explore_first"], default="exploit_first")
    parser.add_argument("--ab-x0", type=int, default=1)
    parser.add_argument("--ab-gamma", type=float, default=2.0)
    parser.add_argument("--ab-window", type=int, default=None)
    parser.add_argument("--ab-stall-min", type=int, default=None)
    parser.add_argument("--ab-cooldown", type=int, default=None)
    parser.add_argument("--ab-metric", choices=["reward", "regret", "opt_rate"], default="reward")
    parser.add_argument("--ab-eta-down", type=float, default=-0.15)
    parser.add_argument("--ab-eta-up", type=float, default=0.1)
    parser.add_argument("--ab-x-reset", choices=["one", "x0"], default="one")
    parser.add_argument("--ab-tail-frac", type=float, default=0.25)
    parser.add_argument("--ab-opt-gap", type=float, default=0.01)
    parser.add_argument("--ab-ewma-lam", type=float, default=0.2)
    parser.add_argument("--ab-enable-ph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ab-ph-delta", type=float, default=1e-3)
    parser.add_argument("--ab-ph-lambda", type=float, default=0.05)
    parser.add_argument("--ab-ph-alpha", type=float, default=0.1)
    return parser.parse_args()


def ensure_artifacts(
    *,
    seeds: List[int],
    cfg: CausalBanditConfig,
    horizon: int,
    tau: float,
    knob_value: float,
    subset_size: int,
    scheduler: str,
    use_full_budget: bool,
    effect_threshold: float,
    sampling: SamplingSettings,
    adaptive_cfg,
    adaptive_cfg_dict: Optional[Dict[str, object]],
    artifact_dir: Path,
    cli_args_snapshot: Dict[str, object],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        identity = make_trial_identity(
            cfg,
            horizon=horizon,
            tau=float(tau),
            seed=seed,
            knob_value=float(knob_value),
            scheduler=scheduler,
            subset_size=subset_size,
            use_full_budget=use_full_budget,
            effect_threshold=effect_threshold,
            min_samples=sampling.min_samples,
            adaptive_config=adaptive_cfg_dict,
            structure_mc_samples=sampling.structure_mc_samples,
            arm_mc_samples=sampling.arm_mc_samples,
            optimal_mean_mc_samples=sampling.optimal_mean_mc_samples,
        )
        artifact = load_trial_artifact(artifact_dir, identity)
        if artifact is not None:
            continue
        record, summary, optimal_mean = run_trial(
            base_cfg=cfg,
            horizon=horizon,
            tau=tau,
            seed=seed,
            knob_value=float(knob_value),
            subset_size=subset_size,
            scheduler_mode=scheduler,
            use_full_budget=use_full_budget,
            effect_threshold=effect_threshold,
            sampling=sampling,
            adaptive_config=adaptive_cfg,
        )
        record = enrich_record_with_metadata(
            record,
            summary=summary,
            identity=identity,
            horizon=horizon,
            scheduler=scheduler,
        )
        metadata = build_metadata(
            cli_args={
                **cli_args_snapshot,
                "seed": seed,
            }
        )
        artifact = TrialArtifact(
            identity=identity,
            record=record,
            summary=summary,
            optimal_mean=optimal_mean,
            metadata=metadata,
        )
        write_trial_artifact(artifact_dir, artifact)


def load_artifacts_for_seeds(
    *,
    seeds: List[int],
    cfg: CausalBanditConfig,
    horizon: int,
    tau: float,
    knob_value: float,
    subset_size: int,
    scheduler: str,
    use_full_budget: bool,
    effect_threshold: float,
    sampling: SamplingSettings,
    adaptive_cfg_dict: Optional[Dict[str, object]],
    artifact_dir: Path,
) -> List[TrialArtifact]:
    artifacts: List[TrialArtifact] = []
    for seed in seeds:
        identity = make_trial_identity(
            cfg,
            horizon=horizon,
            tau=float(tau),
            seed=seed,
            knob_value=float(knob_value),
            scheduler=scheduler,
            subset_size=subset_size,
            use_full_budget=use_full_budget,
            effect_threshold=effect_threshold,
            min_samples=sampling.min_samples,
            adaptive_config=adaptive_cfg_dict,
            structure_mc_samples=sampling.structure_mc_samples,
            arm_mc_samples=sampling.arm_mc_samples,
            optimal_mean_mc_samples=sampling.optimal_mean_mc_samples,
        )
        artifact = load_trial_artifact(artifact_dir, identity)
        if artifact is None:
            raise FileNotFoundError(
                f"Missing artifact for seed {seed} in {artifact_dir}. Run without --plot-only to generate it."
            )
        artifacts.append(artifact)
    return artifacts


def main() -> None:
    args = parse_args()
    if args.effect_threshold is not None and args.effect_threshold_mode != "fixed":
        raise ValueError("--effect-threshold requires --effect-threshold-mode=fixed.")
    seed_start, seed_end = map(int, args.seeds.split(":"))
    seeds = list(range(seed_start, seed_end + 1))
    if not seeds:
        raise ValueError("No seeds specified.")
    m_value = args.m if args.m is not None else args.k
    cfg = CausalBanditConfig(
        n=args.n,
        ell=args.ell,
        k=args.k,
        m=m_value,
        edge_prob=2.0 / max(1, args.n),
        scm_mode=args.scm_mode,
        parent_effect=args.parent_effect,
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

    if not args.plot_only:
        ensure_artifacts(
            seeds=seeds,
            cfg=cfg,
            horizon=horizon,
            tau=args.tau,
            knob_value=args.knob_value,
            subset_size=subset_size,
            scheduler=args.scheduler,
            use_full_budget=args.etc_use_full_budget,
            effect_threshold=effect_threshold_value,
            sampling=sampling,
            adaptive_cfg=adaptive_cfg,
            adaptive_cfg_dict=adaptive_cfg_dict,
            artifact_dir=args.artifact_dir,
            cli_args_snapshot=cli_args_snapshot,
        )

    artifacts = load_artifacts_for_seeds(
        seeds=seeds,
        cfg=cfg,
        horizon=horizon,
        tau=args.tau,
        knob_value=args.knob_value,
        subset_size=subset_size,
        scheduler=args.scheduler,
        use_full_budget=args.etc_use_full_budget,
        effect_threshold=effect_threshold_value,
        sampling=sampling,
        adaptive_cfg_dict=adaptive_cfg_dict,
        artifact_dir=args.artifact_dir,
    )
    xs, mean, std, matrix = aggregate_regret_curves(artifacts)
    plot_path = args.plot_path or (args.artifact_dir / "regret_curve.png")
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
        "knob_value": args.knob_value,
        "seeds": seeds,
        "horizon": horizon,
        "mean_final_regret": float(mean[-1]) if mean.size else 0.0,
        "std_final_regret": float(std[-1]) if std.size else 0.0,
        "num_trials": len(artifacts),
        "subset_size": subset_size,
    }
    stats_path = args.artifact_dir / "regret_curve_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
