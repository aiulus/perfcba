"""CLI entry point for the binary graph-discovery study."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from .causal_envs import (
    CausalBanditConfig,
    CausalBanditInstance,
    InterventionSpace,
    build_random_scm,
)
from .exploit import ArmBuilder, ParentAwareUCB
from .grids import grid_values
from .metrics import summarize
from .scheduler import RoundLog, RunSummary
from .structure import RAPSLearner, StructureConfig
from .timeline import encode_schedule, plot_time_allocation
from .run_tau_study import (
    SamplingSettings,
    PreparedInstance,
    compute_optimal_mean,
    prepare_instance,
    subset_size_for_known_k,
)


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


def _format_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4g}"


@dataclasses.dataclass
class DecisionOutcome:
    attempt_structure: bool
    metadata: Dict[str, float] = dataclasses.field(default_factory=dict)


class DecisionRule:
    """Interface for deciding whether to run graph discovery."""

    name: str

    def decide(
        self,
        *,
        cfg: CausalBanditConfig,
        horizon: int,
        sampling: SamplingSettings,
        knob_value: float,
    ) -> DecisionOutcome:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError


@dataclasses.dataclass
class AlwaysGraphRule(DecisionRule):
    name: str = "graph_first"

    def decide(
        self,
        *,
        cfg: CausalBanditConfig,
        horizon: int,
        sampling: SamplingSettings,
        knob_value: float,
    ) -> DecisionOutcome:
        del cfg, horizon, sampling, knob_value
        return DecisionOutcome(True, {})


@dataclasses.dataclass
class AlwaysBanditRule(DecisionRule):
    name: str = "bandit_only"

    def decide(
        self,
        *,
        cfg: CausalBanditConfig,
        horizon: int,
        sampling: SamplingSettings,
        knob_value: float,
    ) -> DecisionOutcome:
        del cfg, horizon, sampling, knob_value
        return DecisionOutcome(False, {})


@dataclasses.dataclass
class CostBoundRule(DecisionRule):
    budget_fraction: float
    margin: float
    safety_factor: float
    name: str = "cost_bound"

    def decide(
        self,
        *,
        cfg: CausalBanditConfig,
        horizon: int,
        sampling: SamplingSettings,
        knob_value: float,
    ) -> DecisionOutcome:
        del knob_value
        margin = max(0.0, self.margin)
        nodes_to_probe = max(cfg.k, int(math.ceil(cfg.k * (1.0 + margin))))
        nodes_to_probe = min(cfg.n, nodes_to_probe)
        unit_cost = cfg.ell * sampling.min_samples
        estimate = self.safety_factor * nodes_to_probe * unit_cost
        budget_cap = max(0.0, min(1.0, self.budget_fraction)) * horizon
        metadata = {
            "cost_estimate": float(estimate),
            "budget_cap": float(budget_cap),
            "unit_cost": float(unit_cost),
            "nodes_considered": float(nodes_to_probe),
        }
        return DecisionOutcome(attempt_structure=estimate <= budget_cap, metadata=metadata)


def make_decision_rule(name: str, args: argparse.Namespace) -> DecisionRule:
    if name == "graph_first":
        return AlwaysGraphRule()
    if name == "bandit_only":
        return AlwaysBanditRule()
    if name == "cost_bound":
        return CostBoundRule(
            budget_fraction=float(args.cost_budget_frac),
            margin=float(args.cost_margin),
            safety_factor=float(args.cost_safety_factor),
        )
    raise ValueError(f"Unsupported decision rule: {name}")


@dataclasses.dataclass
class BinaryRunSummary(RunSummary):
    decision_rule: str = ""
    attempted_structure: bool = False
    graph_success: bool = False
    graph_cost_estimate: float = math.nan
    decision_metadata: Dict[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _RunData:
    logs: List[RoundLog]
    structure_steps: int
    exploit_steps: int
    parent_set: Tuple[int, ...]
    finished_round: Optional[int]
    graph_success: bool


def _arm_id(arm) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    return (arm.variables, arm.values)


def _run_bandit_only(
    *,
    instance: CausalBanditInstance,
    arm_builder: ArmBuilder,
    rng: np.random.Generator,
    horizon: int,
) -> _RunData:
    policy = ParentAwareUCB()
    arms, mean_lookup = arm_builder.build((), rng)
    policy.set_arms(arms)
    arm_count = len(arms)
    logs: List[RoundLog] = []
    for t in range(1, horizon + 1):
        arm_idx, arm = policy.select(rng)
        reward = instance.sample_reward(arm, rng)
        policy.observe(arm_idx, reward)
        expected_mean = mean_lookup.get(_arm_id(arm), float("nan"))
        logs.append(
            RoundLog(
                t=t,
                mode="exploit",
                arm=arm,
                reward=reward,
                expected_mean=expected_mean,
                parent_set=tuple(),
                arm_count=arm_count,
            )
        )
    return _RunData(
        logs=logs,
        structure_steps=0,
        exploit_steps=len(logs),
        parent_set=tuple(),
        finished_round=None,
        graph_success=False,
    )


def _run_structure_then_exploit(
    *,
    instance: CausalBanditInstance,
    structure: RAPSLearner,
    arm_builder: ArmBuilder,
    rng: np.random.Generator,
    horizon: int,
    true_parents: Tuple[int, ...],
) -> _RunData:
    logs: List[RoundLog] = []
    t = 0
    structure_steps = 0
    finished_round: Optional[int] = None
    policy = ParentAwareUCB()
    arm_count = 0

    while t < horizon and structure.needs_structure_step():
        result = structure.step(rng)
        structure_steps += 1
        t += 1
        current_parents = tuple(structure.parent_set())
        if not structure.needs_structure_step() and finished_round is None:
            finished_round = t
        logs.append(
            RoundLog(
                t=t,
                mode="structure",
                arm=result.arm,
                reward=result.reward,
                expected_mean=result.expected_mean,
                parent_set=current_parents,
                arm_count=arm_count,
            )
        )

    parent_set = tuple(structure.parent_set())
    graph_success = finished_round is not None and set(parent_set) == set(true_parents)
    if t >= horizon:
        return _RunData(
            logs=logs,
            structure_steps=structure_steps,
            exploit_steps=0,
            parent_set=parent_set,
            finished_round=finished_round,
            graph_success=graph_success,
        )

    arms, mean_lookup = arm_builder.build(parent_set, rng)
    policy.set_arms(arms)
    arm_count = len(arms)
    exploit_steps = 0
    while t < horizon:
        arm_idx, arm = policy.select(rng)
        reward = instance.sample_reward(arm, rng)
        policy.observe(arm_idx, reward)
        t += 1
        exploit_steps += 1
        expected_mean = mean_lookup.get(_arm_id(arm), float("nan"))
        logs.append(
            RoundLog(
                t=t,
                mode="exploit",
                arm=arm,
                reward=reward,
                expected_mean=expected_mean,
                parent_set=parent_set,
                arm_count=arm_count,
            )
        )

    return _RunData(
        logs=logs,
        structure_steps=structure_steps,
        exploit_steps=exploit_steps,
        parent_set=parent_set,
        finished_round=finished_round,
        graph_success=graph_success,
    )


def run_binary_trial(
    *,
    base_cfg: CausalBanditConfig,
    horizon: int,
    seed: int,
    knob_value: float,
    subset_size: int,
    effect_threshold: float,
    sampling: SamplingSettings,
    decision_rule: DecisionRule,
    prepared: Optional[PreparedInstance] = None,
) -> Tuple[Dict[str, Any], BinaryRunSummary, float]:
    if prepared is not None:
        instance = prepared.instance
        optimal_mean = prepared.optimal_mean
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(prepared.rng_state)
    else:
        rng = np.random.default_rng(seed)
        instance = build_random_scm(base_cfg, rng=rng)
        optimal_mean = compute_optimal_mean(instance, rng)

    space = InterventionSpace(
        instance.config.n,
        instance.config.ell,
        instance.config.m,
        include_lower=False,
    )
    arm_builder = ArmBuilder(
        instance,
        space,
        subset_size=subset_size,
        mc_samples=sampling.arm_mc_samples,
    )
    structure = RAPSLearner(
        instance,
        StructureConfig(
            effect_threshold=effect_threshold,
            min_samples_per_value=sampling.min_samples,
            mean_mc_samples=sampling.structure_mc_samples,
        ),
    )
    outcome = decision_rule.decide(cfg=instance.config, horizon=horizon, sampling=sampling, knob_value=knob_value)
    true_parents = tuple(sorted(instance.parent_indices()))

    if outcome.attempt_structure:
        run_data = _run_structure_then_exploit(
            instance=instance,
            structure=structure,
            arm_builder=arm_builder,
            rng=rng,
            horizon=horizon,
            true_parents=true_parents,
        )
    else:
        run_data = _run_bandit_only(
            instance=instance,
            arm_builder=arm_builder,
            rng=rng,
            horizon=horizon,
        )

    summary = BinaryRunSummary(
        logs=run_data.logs,
        structure_steps=run_data.structure_steps,
        exploit_steps=run_data.exploit_steps,
        final_parent_set=run_data.parent_set,
        finished_discovery_round=run_data.finished_round,
        decision_rule=decision_rule.name,
        attempted_structure=outcome.attempt_structure,
        graph_success=run_data.graph_success,
        graph_cost_estimate=float(outcome.metadata.get("cost_estimate", math.nan)),
        decision_metadata=dict(outcome.metadata),
    )
    metrics = summarize(summary.logs, optimal_mean)
    parents_found = tuple(sorted(run_data.parent_set))
    true_parent_set = tuple(sorted(true_parents))
    intersection = set(parents_found) & set(true_parent_set)
    parent_precision = len(intersection) / max(1, len(parents_found))
    parent_recall = len(intersection) / max(1, len(true_parent_set))

    record: Dict[str, Any] = {
        "decision_rule": decision_rule.name,
        "attempted_structure": outcome.attempt_structure,
        "graph_success": run_data.graph_success,
        "structure_steps": run_data.structure_steps,
        "exploit_steps": run_data.exploit_steps,
        "finished_discovery_round": run_data.finished_round,
        "parents_found": len(parents_found),
        "true_parent_count": len(true_parent_set),
        "parent_precision": parent_precision,
        "parent_recall": parent_recall,
        "cumulative_regret": metrics.cumulative_regret,
        "tto": metrics.time_to_optimality,
        "optimal_rate": metrics.optimal_action_rate,
        "seed": seed,
        "knob_value": knob_value,
        "horizon": horizon,
        "cost_estimate": outcome.metadata.get("cost_estimate"),
        "cost_budget": outcome.metadata.get("budget_cap"),
        "decision_metadata": outcome.metadata,
        "graph_cost_fraction": run_data.structure_steps / max(1, horizon),
        "finished_discovery": run_data.finished_round is not None,
        "true_parents": list(true_parent_set),
        "recovered_parents": list(parents_found),
    }
    return record, summary, optimal_mean


def aggregate_heatmap(
    results: Sequence[Dict[str, Any]],
    decision_rules: Sequence[str],
    knob_values: Sequence[float],
    metric_key: str,
) -> np.ndarray:
    matrix = np.zeros((len(decision_rules), len(knob_values)))
    counts = np.zeros_like(matrix)
    rule_index = {name: idx for idx, name in enumerate(decision_rules)}
    knob_index = {float(val): idx for idx, val in enumerate(knob_values)}
    for record in results:
        r_idx = rule_index[record["decision_rule"]]
        k_idx = knob_index[float(record["knob_value"])]
        matrix[r_idx, k_idx] += record[metric_key]
        counts[r_idx, k_idx] += 1
    counts[counts == 0] = 1
    return matrix / counts


def plot_decision_heatmap(
    matrix: np.ndarray,
    decision_rules: Sequence[str],
    knob_values: Sequence[float],
    *,
    metric_label: str,
    knob_label: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(4, len(knob_values)), 6))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(knob_values)))
    ax.set_xticklabels([_format_value(v) for v in knob_values], rotation=45, ha="right")
    ax.set_yticks(range(len(decision_rules)))
    ax.set_yticklabels(decision_rules)
    ax.set_xlabel(knob_label)
    ax.set_ylabel("Decision rule")
    ax.set_title(f"{metric_label} across decision rules")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_label)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary discovery causal bandit study.")
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
        help="SCM generation scheme for sampled environments.",
    )
    parser.add_argument("--parent-effect", type=float, default=1.0)
    parser.add_argument("--reward-logit-scale", type=float, default=1.0)
    parser.add_argument("--T", type=int, default=10_000, help="Total horizon.")
    parser.add_argument("--seeds", type=str, default="0:9", help="Seed range start:end (inclusive).")
    parser.add_argument(
        "--decision-rules",
        nargs="+",
        choices=["cost_bound", "graph_first", "bandit_only"],
        default=["cost_bound", "graph_first", "bandit_only"],
        help="Decision policies controlling whether to run structure learning.",
    )
    parser.add_argument("--cost-budget-frac", type=float, default=0.3, help="Budget fraction tolerated by cost_bound.")
    parser.add_argument("--cost-margin", type=float, default=0.5, help="Extra multiple of k considered by cost_bound.")
    parser.add_argument(
        "--cost-safety-factor",
        type=float,
        default=1.25,
        help="Inflation factor on the theoretical discovery cost.",
    )
    parser.add_argument("--effect-threshold", type=float, default=0.05)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument(
        "--sampler-cache",
        choices=["auto", "on", "off"],
        default="auto",
        help="Attach sampler caches to SCM instances (default: auto=on).",
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
    parser.add_argument("--metric", choices=["cumulative_regret", "tto"], default="cumulative_regret")
    parser.add_argument("--output-dir", type=Path, default=Path("results/binary_discovery"))
    parser.add_argument(
        "--timeline-dir",
        type=Path,
        default=None,
        help="Optional directory for per-trial time allocation diagrams.",
    )
    parser.add_argument("--timeline-max-columns", type=int, default=2000)
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
        knob_values = [max(1, int(value)) for value in args.parent_grid]
    elif args.vary == "graph_density" and args.graph_grid:
        knob_values = [float(value) for value in args.graph_grid]
    else:
        knob_values = grid_values(args.vary, n=args.n, k=args.k)

    decision_rules: List[DecisionRule] = []
    seen_rules: set[str] = set()
    for name in args.decision_rules:
        if name in seen_rules:
            continue
        decision_rules.append(make_decision_rule(name, args))
        seen_rules.add(name)
    rule_names = [rule.name for rule in decision_rules]

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

    results: List[Dict[str, Any]] = []
    total_trials = len(knob_values) * len(rule_names) * len(seeds)
    progress = tqdm(total=total_trials, desc="Binary discovery", unit="trial")

    timeline_dir: Optional[Path] = args.timeline_dir
    timeline_store = defaultdict(list) if timeline_dir is not None else None
    if timeline_dir is not None:
        timeline_dir.mkdir(parents=True, exist_ok=True)

    prepared_cache: Dict[
        Tuple[CausalBanditConfig, int, str, int, int, int, int],
        PreparedInstance,
    ] = {}
    cache_mode = args.sampler_cache
    cache_enabled = cache_mode != "off"

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
            elif args.vary == "arm_variance":
                cfg = dataclasses.replace(cfg, reward_logit_scale=float(knob_value))

            current_horizon = args.T if args.vary != "horizon" else int(knob_value)
            subset_size = subset_size_for_known_k(cfg, current_horizon)

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
                prepared_instance = prepared_cache.get(cache_key)
                if prepared_instance is None:
                    prepared_instance = prepare_instance(
                        cfg,
                        seed,
                        enable_cache=cache_enabled,
                        sampling=sampling,
                    )
                    prepared_cache[cache_key] = prepared_instance

                for rule in decision_rules:
                    record, summary, optimal_mean = run_binary_trial(
                        base_cfg=cfg,
                        horizon=current_horizon,
                        seed=seed,
                        knob_value=float(knob_value),
                        subset_size=subset_size,
                        effect_threshold=args.effect_threshold,
                        sampling=sampling,
                        decision_rule=rule,
                        prepared=prepared_instance,
                    )

                    results.append(record)

                    if timeline_dir is not None and timeline_store is not None:
                        schedule = encode_schedule(summary.logs)
                        key = (rule.name, float(knob_value))
                        timeline_store[key].append((seed, schedule))
                        per_seed_path = (
                            timeline_dir
                            / f"timeline_rule-{rule.name}_knob-{_format_value(float(knob_value))}_seed-{seed}.png"
                        )
                        plot_time_allocation(
                            schedule,
                            per_seed_path,
                            title=f"rule={rule.name}, knob={_format_value(float(knob_value))}, seed={seed}",
                            yticklabels=[f"seed {seed}"],
                            max_columns=args.timeline_max_columns,
                        )

                    progress.set_postfix(
                        knob=_format_value(float(knob_value)),
                        rule=rule.name,
                        seed=seed,
                    )
                    progress.update(1)
    finally:
        progress.close()

    if timeline_dir is not None and timeline_store:
        for (rule_name, knob_value), rows in timeline_store.items():
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
            agg_path = timeline_dir / f"timeline_rule-{rule_name}_knob-{_format_value(knob_value)}_grid.png"
            plot_time_allocation(
                matrix,
                agg_path,
                title=f"rule={rule_name}, knob={_format_value(knob_value)}",
                yticklabels=labels,
                max_columns=args.timeline_max_columns,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    knob_label, knob_label_plural = KNOB_LABELS.get(
        args.vary, (args.vary.replace("_", " ").title(), f"{args.vary.replace('_', ' ')}s")
    )
    metric_label = METRIC_LABELS.get(args.metric, args.metric.replace("_", " ").capitalize())
    matrix = aggregate_heatmap(results, rule_names, knob_values, args.metric)
    plot_decision_heatmap(
        matrix,
        rule_names,
        knob_values,
        metric_label=metric_label,
        knob_label=knob_label,
        output_path=args.output_dir / f"heatmap_{args.metric}.png",
    )


if __name__ == "__main__":
    main()
