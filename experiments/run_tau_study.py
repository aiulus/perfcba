"""CLI entry point for the tau-scheduled causal bandit study."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence, Tuple, Literal

import numpy as np
from tqdm.auto import tqdm

from ..SCM import Intervention
from ..budgeted_raps import BudgetedRAPSResult, run_budgeted_raps
from ..causal_bandits import RAPSParams
from .artifacts import (
    TrialArtifact,
    TrialIdentity,
    build_metadata,
    load_trial_artifact,
    make_trial_identity,
    trial_identity_digest,
    write_trial_artifact,
)
from .causal_envs import (
    CausalBanditConfig,
    CausalBanditInstance,
    InterventionArm,
    InterventionSpace,
    build_random_scm,
)
from .exploit import ArmBuilder, HybridArmConfig, ParentAwareUCB
from .grids import TAU_GRID, grid_values
from .heatmap import plot_heatmap
from .metrics import summarize
from .scheduler import AdaptiveBurstConfig, RunSummary, RoundLog, build_scheduler
from .timeline import encode_schedule, plot_time_allocation
from .structure import RAPSLearner, StructureConfig, compute_effect_threshold
from .sampler_cache import SamplerCache


KNOB_LABELS = {
    "graph_density": ("Graph Density", "graph densities"),
    "node_count": ("Node Count", "node counts"),
    "parent_count": ("Parent Count", "parent counts"),
    "intervention_size": ("Intervention Size", "intervention sizes"),
    "alphabet": ("Alphabet Size", "alphabet sizes"),
    "horizon": ("Horizon", "horizons"),
    "arm_variance": ("Reward Logit Scale", "reward logit scales"),
    "algo_eps": ("Ancestral Gap ε", "ancestral gaps"),
    "algo_delta": ("Reward Gap Δ", "reward gaps"),
    "hard_margin": ("Hard Margin", "hard margins"),
    "tau": ("Tau Budget", "tau values"),
}

METRIC_LABELS = {
    "cumulative_regret": "Cumulative regret",
    "tto": "Time to optimality",
    "simple_regret": "Simple regret",
    "optimal_rate": "Optimal action rate",
}

OVERLAY_SUCCESS_THRESHOLD = 0.5
DEFAULT_HARD_MARGIN = 0.1


@dataclasses.dataclass(frozen=True)
class SamplingSettings:
    min_samples: int
    structure_mc_samples: int
    arm_mc_samples: int
    optimal_mean_mc_samples: int


StructureBackend = Literal["proxy", "budgeted_raps"]


class InstancePCM:
    """Adapter exposing a :class:`CausalBanditInstance` through the PCM interface."""

    def __init__(self, instance: CausalBanditInstance, rng: np.random.Generator) -> None:
        self.instance = instance
        self._rng = rng
        self.V = list(instance.node_names)
        self.K = int(instance.config.ell)
        self.reward_node = instance.reward_node

    def _spawn_rng(self) -> np.random.Generator:
        return np.random.default_rng(int(self._rng.integers(2**32 - 1)))

    def _sample(
        self,
        B: int,
        *,
        intervention: Optional[Intervention] = None,
    ) -> List[MutableMapping[str, int]]:
        if B <= 0:
            raise ValueError("PCM sampling requires B > 0.")
        rng = self._spawn_rng()
        samples: List[MutableMapping[str, int]] = []
        for _ in range(B):
            draw = self.instance.scm.sample(rng, intervention=intervention)
            samples.append({name: int(draw[name]) for name in self.instance.node_names})
        return samples

    def observe(self, B: int) -> List[MutableMapping[str, int]]:
        return self._sample(B)

    def intervene(self, do: MutableMapping[str, int], B: int) -> List[MutableMapping[str, int]]:
        intervention = Intervention(name="do", hard=dict(do))
        return self._sample(B, intervention=intervention)


ArmKey = Tuple[Tuple[int, ...], Tuple[int, ...]]


def _arm_from_assignment(
    assignment: MutableMapping[str, int],
    node_to_index: Dict[str, int],
    reward_index: int,
) -> InterventionArm:
    if not assignment:
        return InterventionArm(tuple(), tuple())
    ordered = sorted(assignment.items(), key=lambda item: node_to_index[item[0]])
    variables: List[int] = []
    values: List[int] = []
    for name, value in ordered:
        idx = node_to_index[name]
        if idx == reward_index:
            continue
        variables.append(idx)
        values.append(int(value))
    return InterventionArm(tuple(variables), tuple(values))


def _budgeted_summary_from_trace(
    result: BudgetedRAPSResult,
    instance: CausalBanditInstance,
    sampling: SamplingSettings,
    rng: np.random.Generator,
) -> RunSummary:
    node_to_index = {name: idx for idx, name in enumerate(instance.node_names)}
    reward_index = node_to_index[instance.reward_node]
    rng_means = np.random.default_rng(int(rng.integers(2**32 - 1)))
    mean_cache: Dict[ArmKey, float] = {}

    def mean_for_arm(arm: InterventionArm) -> float:
        key: ArmKey = (arm.variables, arm.values)
        if key not in mean_cache:
            mean_cache[key] = instance.estimate_arm_mean(arm, rng_means, n_mc=sampling.arm_mc_samples)
        return mean_cache[key]

    logs: List[RoundLog] = []
    structure_steps = 0
    exploit_steps = 0
    finished_round: Optional[int] = None
    parent_set_snapshot: Tuple[int, ...] = tuple()
    true_parent_set = tuple(sorted(instance.parent_indices()))

    for idx, step in enumerate(result.trace):
        assignment = step.interventions[0] if step.interventions else {}
        arm = _arm_from_assignment(assignment, node_to_index, reward_index)
        reward = float(np.mean(step.rewards)) if len(step.rewards) else 0.0
        expected = mean_for_arm(arm)
        parent_mask = step.parent_mask
        if parent_mask.ndim == 2:
            parent_vector = parent_mask[0]
        else:
            parent_vector = parent_mask
        parents = tuple(
            sorted(i for i, flag in enumerate(parent_vector) if flag and i != reward_index)
        )
        parent_set_snapshot = parents
        action = step.action
        mode = "exploit" if action == "exploit" else "structure"
        if action == "exploit":
            exploit_steps += 1
        else:
            structure_steps += 1
        if finished_round is None and parents == true_parent_set:
            finished_round = idx + 1
        logs.append(
            RoundLog(
                t=idx + 1,
                mode=mode,
                arm=arm,
                reward=reward,
                expected_mean=expected,
                parent_set=parents,
                arm_count=0,
            )
        )

    return RunSummary(
        logs=logs,
        structure_steps=structure_steps,
        exploit_steps=exploit_steps,
        final_parent_set=parent_set_snapshot,
        finished_discovery_round=finished_round,
    )


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


def _cast_grid_value(value: float, caster: Callable[[str], Any]) -> Any:
    if caster is int:
        return int(round(value))
    if caster is float:
        return float(value)
    return caster(str(value))


def _expand_range(token: str, caster: Callable[[str], Any]) -> List[Any]:
    parts = token.split(":")
    if len(parts) == 2:
        start = float(parts[0])
        stop = float(parts[1])
        step = 1.0 if stop >= start else -1.0
    elif len(parts) == 3:
        start = float(parts[0])
        step = float(parts[1])
        stop = float(parts[2])
        if step == 0:
            raise ValueError("Range step cannot be zero.")
    else:
        raise ValueError(f"Invalid grid range specifier: {token!r}")
    values: List[Any] = []
    max_iters = 100_000
    current = start
    epsilon = 1e-9
    for _ in range(max_iters):
        if step > 0 and current > stop + epsilon:
            break
        if step < 0 and current < stop - epsilon:
            break
        values.append(_cast_grid_value(current, caster))
        current += step
    else:
        raise ValueError(f"Range specifier {token!r} produced too many values; check the step size.")
    return values


def _parse_grid_tokens(tokens: Optional[Sequence[str]], caster: Callable[[str], Any]) -> Optional[List[Any]]:
    if tokens is None:
        return None
    expanded: List[Any] = []
    for token in tokens:
        if ":" in token:
            expanded.extend(_expand_range(token, caster))
        else:
            expanded.append(caster(token))
    return expanded


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
    sampling: SamplingSettings,
    adaptive_config: Optional[AdaptiveBurstConfig],
    structure_backend: StructureBackend,
    raps_params: Optional[RAPSParams],
    arm_builder_cfg: Optional[HybridArmConfig] = None,
    prepared: Optional[PreparedInstance] = None,
) -> Tuple[Dict[str, Any], RunSummary, float]:
    if prepared is not None:
        instance = prepared.instance
        optimal_mean = prepared.optimal_mean
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(prepared.rng_state)
    else:
        rng = np.random.default_rng(seed)
        instance = build_random_scm(base_cfg, rng=rng)
        optimal_mean = compute_optimal_mean(instance, rng, mc_samples=sampling.optimal_mean_mc_samples)
    if structure_backend == "budgeted_raps":
        if raps_params is None:
            raise ValueError("RAPS parameters must be provided when using the budgeted backend.")
        pcm = InstancePCM(instance, rng)
        np.random.seed(int(rng.integers(2**32 - 1)))
        result = run_budgeted_raps(
            pcm,
            params=raps_params,
            horizon=horizon,
            tau=tau,
            batch_size=1,
        )
        summary = _budgeted_summary_from_trace(result, instance, sampling, rng)
    else:
        space = InterventionSpace(
            instance.config.n,
            instance.config.ell,
            instance.config.m,
            include_lower=False,
        )
        structure = RAPSLearner(
            instance,
            StructureConfig(
                effect_threshold=effect_threshold,
                min_samples_per_value=sampling.min_samples,
                mean_mc_samples=sampling.structure_mc_samples,
            ),
        )
        policy = ParentAwareUCB()
        arm_builder = ArmBuilder(
            instance,
            space,
            subset_size=subset_size,
            mc_samples=sampling.arm_mc_samples,
            hybrid_config=arm_builder_cfg,
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
    true_parent_set = tuple(sorted(instance.parent_indices()))
    found_parent_set = tuple(sorted(summary.final_parent_set))
    intersection = set(true_parent_set) & set(found_parent_set)
    parent_precision = len(intersection) / max(1, len(found_parent_set))
    parent_recall = len(intersection) / max(1, len(true_parent_set))
    graph_success = found_parent_set == true_parent_set
    finished_round = summary.finished_discovery_round
    record: Dict[str, Any] = {
        "tau": tau,
        "knob_value": knob_value,
        "seed": seed,
        "cumulative_regret": metrics.cumulative_regret,
        "tto": metrics.time_to_optimality,
        "optimal_rate": metrics.optimal_action_rate,
        "simple_regret": metrics.simple_regret,
        "structure_steps": summary.structure_steps,
        "parents_found": len(summary.final_parent_set),
        "finished_discovery": finished_round is not None,
        "finished_discovery_round": finished_round,
        "horizon": horizon,
        "scheduler": "budgeted_raps" if structure_backend == "budgeted_raps" else scheduler_mode,
        "structure_backend": structure_backend,
        "graph_success": graph_success,
        "parent_precision": parent_precision,
        "parent_recall": parent_recall,
        "hard_margin": float(instance.config.hard_margin),
        "graph_density": float(instance.config.edge_prob),
        "node_count": int(instance.config.n),
        "parent_count": int(instance.config.k),
        "intervention_size": int(instance.config.m),
        "alphabet": int(instance.config.ell),
        "arm_variance": float(instance.config.reward_logit_scale),
    }
    if structure_backend == "budgeted_raps" and raps_params is not None:
        record["algo_eps"] = raps_params.eps
        record["algo_delta"] = raps_params.Delta
        record["raps_delta"] = raps_params.delta
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
    enriched["structure_backend"] = identity.structure_backend
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
    sums, _, counts = _accumulate_heatmap_stats(results, tau_values, knob_values, metric_key)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return means


def aggregate_heatmap_with_std(
    results: Sequence[Dict[str, Any]],
    tau_values: Sequence[float],
    knob_values: Sequence[float],
    metric_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sums, sumsq, counts = _accumulate_heatmap_stats(results, tau_values, knob_values, metric_key)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        mean_sq = np.divide(sumsq, counts, out=np.zeros_like(sumsq), where=counts > 0)
    variance = np.maximum(mean_sq - means**2, 0.0)
    std = np.sqrt(variance)
    return means, std, counts


def _accumulate_heatmap_stats(
    results: Sequence[Dict[str, Any]],
    tau_values: Sequence[float],
    knob_values: Sequence[float],
    metric_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sums = np.zeros((len(tau_values), len(knob_values)), dtype=np.float64)
    sumsq = np.zeros_like(sums)
    counts = np.zeros_like(sums)
    tau_index = {tau: idx for idx, tau in enumerate(tau_values)}
    knob_index = {kv: idx for idx, kv in enumerate(knob_values)}
    for record in results:
        t_idx = tau_index[record["tau"]]
        k_idx = knob_index[record["knob_value"]]
        value = float(record[metric_key])
        sums[t_idx, k_idx] += value
        sumsq[t_idx, k_idx] += value * value
        counts[t_idx, k_idx] += 1
    return sums, sumsq, counts


def report_heatmap_std(
    means: np.ndarray,
    std: np.ndarray,
    counts: np.ndarray,
    *,
    metric_label: str,
) -> None:
    mask = counts > 0
    if not np.any(mask):
        return
    avg_std = float(np.mean(std[mask]))
    max_std = float(np.max(std[mask]))
    reference = float(np.mean(np.abs(means[mask])))
    reference = max(reference, 1e-8)
    avg_ratio = avg_std / reference
    max_ratio = max_std / reference
    print(
        f"[tau-study] {metric_label}: mean std={avg_std:.4g} "
        f"(avg ratio {avg_ratio:.1%}), max std={max_std:.4g} (ratio {max_ratio:.1%})"
    )
    if avg_ratio > 0.10:
        print(
            f"[tau-study][warning] Average std for {metric_label} exceeds 10% of the mean "
            f"(avg std {avg_std:.4g}, reference {reference:.4g})."
        )
    if max_ratio > 0.20:
        print(
            f"[tau-study][warning] At least one cell has std above 20% of the mean "
            f"(max std {max_std:.4g}, reference {reference:.4g})."
        )


@dataclasses.dataclass(frozen=True)
class PreparedInstance:
    """Cache of expensive per-(config, seed) artifacts."""

    config: CausalBanditConfig
    seed: int
    instance: CausalBanditInstance
    optimal_mean: float
    rng_state: Dict[str, Any]


def prepare_instance(
    cfg: CausalBanditConfig,
    seed: int,
    *,
    enable_cache: bool,
    sampling: SamplingSettings,
) -> PreparedInstance:
    """Build the SCM, compute the optimal mean once, and record the RNG state."""

    rng = np.random.default_rng(seed)
    instance = build_random_scm(cfg, rng=rng)
    sampler_cache = SamplerCache() if enable_cache else None
    if sampler_cache is not None:
        instance = dataclasses.replace(instance, sampler_cache=sampler_cache)
    optimal_mean = compute_optimal_mean(instance, rng, mc_samples=sampling.optimal_mean_mc_samples)
    rng_state = copy.deepcopy(rng.bit_generator.state)
    return PreparedInstance(
        config=cfg,
        seed=seed,
        instance=instance,
        optimal_mean=optimal_mean,
        rng_state=rng_state,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tau-scheduled causal bandit study.")
    parser.add_argument(
        "--vary",
        choices=[
            "graph_density",
            "node_count",
            "parent_count",
            "intervention_size",
            "alphabet",
            "horizon",
            "arm_variance",
            "algo_eps",
            "algo_delta",
            "hard_margin",
            "tau",
        ],
        required=True,
        help="Environment knob to sweep.",
    )
    parser.add_argument(
        "--env-vary",
        choices=[
            "graph_density",
            "node_count",
            "parent_count",
            "intervention_size",
            "alphabet",
            "arm_variance",
            "hard_margin",
        ],
        default=None,
        help="Optional secondary environment knob to sweep alongside the primary --vary.",
    )
    parser.add_argument(
        "--env-grid",
        type=str,
        nargs="+",
        default=None,
        help="Grid for --env-vary (accepts start[:step]:stop ranges).",
    )
    parser.add_argument(
        "--parent-grid",
        type=str,
        nargs="+",
        default=None,
        help="Override parent counts when --vary parent_count is used (accepts start[:step]:stop ranges).",
    )
    parser.add_argument(
        "--graph-grid",
        type=str,
        nargs="+",
        default=None,
        help="Override edge probabilities when --vary graph_density is used (accepts start[:step]:stop ranges).",
    )
    parser.add_argument(
        "--node-grid",
        type=str,
        nargs="+",
        default=None,
        help="Override node counts when --vary node_count is used (accepts start[:step]:stop ranges).",
    )
    parser.add_argument(
        "--intervention-grid",
        type=str,
        nargs="+",
        default=None,
        help="Override intervention sizes when --vary intervention_size is used (accepts start[:step]:stop ranges).",
    )
    parser.add_argument(
        "--algo-eps-grid",
        type=str,
        nargs="+",
        default=None,
        help="Override epsilon (ancestral gap) values when --vary algo_eps is used.",
    )
    parser.add_argument(
        "--algo-delta-grid",
        type=str,
        nargs="+",
        default=None,
        help="Override reward-gap Δ values when --vary algo_delta is used.",
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
    parser.add_argument(
        "--hard-margin",
        nargs="*",
        type=float,
        default=None,
        metavar="M",
        help="Enable the hard parent-effect margin (TV distance). Provide zero/one/many values; no values use the default 0.1.",
    )
    parser.add_argument(
        "--scm-epsilon",
        type=float,
        default=0.0,
        help="Lower bound applied to every CPT entry in generated SCMs.",
    )
    parser.add_argument(
        "--scm-delta",
        type=float,
        default=0.0,
        help="Lower bound applied to reward-conditionals only (after --scm-epsilon).",
    )
    parser.add_argument("--seeds", type=str, default="0:9", help="Seed range start:end.")
    parser.add_argument(
        "--scheduler",
        choices=["interleaved", "two_phase", "etc", "adaptive_burst"],
        default="two_phase",
    )
    parser.add_argument(
        "--structure-backend",
        choices=["proxy", "budgeted_raps"],
        default="budgeted_raps",
        help="Selects the structure learner: 'proxy' uses RAPSLearner, 'budgeted_raps' reuses the official implementation.",
    )
    parser.add_argument(
        "--algo-eps",
        type=float,
        default=0.05,
        help="Epsilon parameter for the budgeted RAPS backend.",
    )
    parser.add_argument(
        "--algo-delta",
        type=float,
        default=0.05,
        help="Reward-gap parameter (Δ) for the budgeted RAPS backend.",
    )
    parser.add_argument(
        "--raps-delta",
        type=float,
        default=0.05,
        help="Confidence parameter δ for the budgeted RAPS backend.",
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
    parser.add_argument(
        "--effect-threshold",
        type=float,
        default=None,
        help="Fixed threshold (requires --effect-threshold-mode=fixed).",
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
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument(
        "--structure-mc-samples",
        type=int,
        default=512,
        help="Monte Carlo samples per structure-arm mean estimate.",
    )
    parser.add_argument(
        "--arm-mc-samples",
        type=int,
        default=1024,
        help="Monte Carlo samples per exploitation arm mean estimate.",
    )
    parser.add_argument(
        "--optimal-mean-mc-samples",
        type=int,
        default=2048,
        help="Monte Carlo samples used when computing the optimal mean.",
    )
    parser.add_argument(
        "--metric",
        choices=["cumulative_regret", "tto", "simple_regret", "optimal_rate"],
        default="cumulative_regret",
    )
    parser.add_argument(
        "--hybrid-arms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Augment partial parent sets with hybrid interventions that use remaining arity.",
    )
    parser.add_argument(
        "--hybrid-max-fillers",
        type=int,
        default=None,
        help="Optional cap on filler-variable combinations considered for hybrid arms.",
    )
    parser.add_argument(
        "--hybrid-max-hybrid-arms",
        type=int,
        default=None,
        help="Optional cap on the total number of hybrid intervention assignments per parent set.",
    )
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
    parser.add_argument(
        "--sampler-cache",
        choices=["auto", "on", "off"],
        default="auto",
        help="Control whether sampler caches are attached to each SCM (default: auto=on).",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.effect_threshold is not None and args.effect_threshold_mode != "fixed":
        raise ValueError("--effect-threshold requires --effect-threshold-mode=fixed.")
    args.parent_grid = _parse_grid_tokens(args.parent_grid, int)
    args.graph_grid = _parse_grid_tokens(args.graph_grid, float)
    args.node_grid = _parse_grid_tokens(args.node_grid, int)
    args.intervention_grid = _parse_grid_tokens(args.intervention_grid, int)
    args.algo_eps_grid = _parse_grid_tokens(args.algo_eps_grid, float)
    args.algo_delta_grid = _parse_grid_tokens(args.algo_delta_grid, float)
    if args.env_vary is not None:
        if args.env_grid is None:
            raise ValueError("--env-vary requires --env-grid.")
        parser_type = int if args.env_vary in {"node_count", "parent_count", "intervention_size", "alphabet"} else float
        args.env_grid = _parse_grid_tokens(args.env_grid, parser_type)
    if args.hard_margin is None:
        hard_margin_values: Optional[List[float]] = None
    else:
        if len(args.hard_margin) == 0:
            hard_margin_values = [DEFAULT_HARD_MARGIN]
        else:
            hard_margin_values = [float(value) for value in args.hard_margin]
    if args.vary == "hard_margin":
        if hard_margin_values is None:
            hard_margin_values = [float(value) for value in grid_values("hard_margin", n=args.n, k=args.k)]
        base_hard_margin = float(hard_margin_values[0]) if hard_margin_values else 0.0
    else:
        if hard_margin_values is None:
            base_hard_margin = 0.0
        else:
            if len(hard_margin_values) != 1:
                raise ValueError("Specify a single --hard-margin value unless --vary hard_margin is active.")
            base_hard_margin = float(hard_margin_values[0])
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
        hard_margin=base_hard_margin,
        scm_epsilon=args.scm_epsilon,
        scm_delta=args.scm_delta,
    )
    base_algo_eps = args.algo_eps
    base_algo_delta = args.algo_delta
    if args.vary == "parent_count" and args.parent_grid:
        knob_values = [int(value) for value in args.parent_grid]
    elif args.vary == "graph_density" and args.graph_grid:
        knob_values = [float(value) for value in args.graph_grid]
    elif args.vary == "node_count" and args.node_grid:
        knob_values = [int(value) for value in args.node_grid]
    elif args.vary == "intervention_size" and args.intervention_grid:
        knob_values = [int(value) for value in args.intervention_grid]
    elif args.vary == "algo_eps" and args.algo_eps_grid:
        knob_values = [float(value) for value in args.algo_eps_grid]
    elif args.vary == "algo_delta" and args.algo_delta_grid:
        knob_values = [float(value) for value in args.algo_delta_grid]
    elif args.vary == "hard_margin":
        if hard_margin_values is None:
            raise ValueError("No hard-margin values provided.")
        knob_values = [float(value) for value in hard_margin_values]
    elif args.vary == "tau":
        knob_values = [float(value) for value in args.tau_grid]
    else:
        knob_values = grid_values(args.vary, n=args.n, k=args.k)
    if args.env_vary is not None:
        if not args.env_grid:
            raise ValueError("No values provided for --env-grid.")
        env_values = [float(val) for val in args.env_grid]
    else:
        env_values = [None]
    results: List[Dict[str, Any]] = []
    tau_loop_len = len(args.tau_grid) if args.vary != "tau" else 1
    total_trials = len(env_values) * len(knob_values) * tau_loop_len * len(seeds)
    progress = tqdm(total=total_trials, desc="Tau study", unit="trial")
    timeline_dir: Optional[Path] = args.timeline_dir
    timeline_store = defaultdict(list) if timeline_dir is not None else None
    if timeline_dir is not None:
        timeline_dir.mkdir(parents=True, exist_ok=True)

    persist_dir: Optional[Path] = args.persist_trials
    reuse_dir: Optional[Path] = args.reuse_trials
    cli_args_snapshot = vars(args).copy()
    prepared_cache: Dict[
        Tuple[CausalBanditConfig, int, str, int, int, int, int],
        PreparedInstance,
    ] = {}
    cache_mode = args.sampler_cache
    cache_enabled = cache_mode != "off"

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

    def apply_knob(cfg: CausalBanditConfig, vary: str, value: float) -> Tuple[CausalBanditConfig, Optional[int], Optional[List[float]]]:
        """Return updated config plus optional horizon override and tau grid override."""
        horizon_override: Optional[int] = None
        tau_override: Optional[List[float]] = None
        if vary == "graph_density":
            cfg = dataclasses.replace(cfg, edge_prob=float(value))
        elif vary == "node_count":
            new_n = int(value)
            if new_n < cfg.k:
                raise ValueError(f"node_count grid value {new_n} must be >= k={cfg.k}")
            new_m = min(cfg.m, new_n)
            cfg = dataclasses.replace(
                cfg,
                n=new_n,
                m=new_m,
                edge_prob=2.0 / max(1, new_n),
            )
        elif vary == "parent_count":
            new_k = int(value)
            cfg = dataclasses.replace(cfg, k=new_k, m=max(new_k, cfg.m))
        elif vary == "intervention_size":
            cfg = dataclasses.replace(cfg, m=int(value))
        elif vary == "alphabet":
            cfg = dataclasses.replace(cfg, ell=int(value))
        elif vary == "arm_variance":
            cfg = dataclasses.replace(cfg, reward_logit_scale=float(value))
        elif vary == "hard_margin":
            cfg = dataclasses.replace(cfg, hard_margin=float(value))
        elif vary == "horizon":
            horizon_override = int(value)
        elif vary == "tau":
            tau_override = [float(value)]
        return cfg, horizon_override, tau_override

    try:
        for env_value in env_values:
            cfg_env, env_horizon_override, env_tau_override = apply_knob(base_cfg, args.env_vary, env_value) if args.env_vary else (base_cfg, None, None)
            for knob_value in knob_values:
                cfg, knob_horizon_override, knob_tau_override = apply_knob(cfg_env, args.vary, knob_value)
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
                arm_builder_cfg = HybridArmConfig(
                    enabled=bool(args.hybrid_arms),
                    max_fillers=args.hybrid_max_fillers,
                    max_hybrid_arms=args.hybrid_max_hybrid_arms,
                )
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
                        cache_key = (
                            cfg,
                            int(seed),
                            cache_mode,
                            sampling.min_samples,
                            sampling.structure_mc_samples,
                            sampling.arm_mc_samples,
                            sampling.optimal_mean_mc_samples,
                        )
                        identity = make_trial_identity(
                            cfg,
                            horizon=current_horizon,
                            tau=float(tau),
                            seed=seed,
                            knob_value=float(knob_value),
                            scheduler=args.scheduler,
                            structure_backend=args.structure_backend,
                            subset_size=subset_size,
                            use_full_budget=args.etc_use_full_budget,
                            effect_threshold=effect_threshold_value,
                            min_samples=sampling.min_samples,
                            adaptive_config=adaptive_cfg_dict,
                            hybrid_config=dataclasses.asdict(arm_builder_cfg),
                            raps_params=dataclasses.asdict(raps_params_for_knob)
                            if raps_params_for_knob is not None
                            else None,
                            structure_mc_samples=sampling.structure_mc_samples,
                            arm_mc_samples=sampling.arm_mc_samples,
                            optimal_mean_mc_samples=sampling.optimal_mean_mc_samples,
                        )

                        artifact = load_trial_artifact(reuse_dir, identity) if reuse_dir is not None else None
                        ran_trial = artifact is None

                        if artifact is not None:
                            record = artifact.record
                            summary = artifact.summary
                            optimal_mean = artifact.optimal_mean
                        else:
                            prepared_instance = prepared_cache.get(cache_key)
                            if prepared_instance is None:
                                prepared_instance = prepare_instance(
                                    cfg,
                                    seed,
                                    enable_cache=cache_enabled,
                                    sampling=sampling,
                                )
                                prepared_cache[cache_key] = prepared_instance
                            record, summary, optimal_mean = run_trial(
                                base_cfg=cfg,
                                horizon=current_horizon,
                                tau=tau,
                                seed=seed,
                                knob_value=float(knob_value),
                                subset_size=subset_size,
                                scheduler_mode=args.scheduler,
                                use_full_budget=args.etc_use_full_budget,
                                effect_threshold=effect_threshold_value,
                                sampling=sampling,
                                adaptive_config=adaptive_cfg,
                                structure_backend=args.structure_backend,
                                raps_params=raps_params_for_knob,
                                arm_builder_cfg=arm_builder_cfg,
                                prepared=prepared_instance,
                            )

                        scheduler_label = args.scheduler if args.structure_backend == "proxy" else "budgeted_raps"
                        record = enrich_record_with_metadata(
                            record,
                            summary=summary,
                            identity=identity,
                            horizon=current_horizon,
                            scheduler=scheduler_label,
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

    tau_values = list(map(float, args.tau_grid))
    matrix, std_matrix, counts = aggregate_heatmap_with_std(results, tau_values, knob_values, args.metric)
    graph_success_matrix = aggregate_heatmap(results, tau_values, knob_values, "graph_success")
    overlay_mask = graph_success_matrix >= OVERLAY_SUCCESS_THRESHOLD
    knob_label, knob_label_plural = KNOB_LABELS.get(
        args.vary, (args.vary.replace("_", " ").title(), f"{args.vary.replace('_', ' ')}s")
    )
    metric_label = METRIC_LABELS.get(args.metric, args.metric.replace("_", " ").capitalize())
    report_heatmap_std(matrix, std_matrix, counts, metric_label=metric_label)
    std_report_path = args.output_dir / f"heatmap_{args.metric}_std.json"
    with std_report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tau_values": list(map(float, tau_values)),
                "knob_values": list(map(float, knob_values)),
                "std": std_matrix.tolist(),
                "mean": matrix.tolist(),
                "counts": counts.astype(int).tolist(),
            },
            f,
        )
    plot_heatmap(
        matrix,
        tau_values=tau_values,
        knob_values=knob_values,
        title=f"{metric_label} for varying {knob_label_plural}",
        cbar_label=metric_label,
        x_label=knob_label,
        output_path=args.output_dir / f"heatmap_{args.metric}.png",
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
    )


if __name__ == "__main__":
    main()
