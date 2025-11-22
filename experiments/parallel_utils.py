"""Utilities for running existing trial logic in parallel without changing semantics."""

from __future__ import annotations

import dataclasses
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .causal_envs import CausalBanditConfig
from .run_tau_study import (
    PreparedInstance,
    SamplingSettings,
    adaptive_config_from_args,
    compute_optimal_mean,
    enrich_record_with_metadata,
    prepare_instance,
    run_trial,
    subset_size_for_known_k,
)


def run_jobs_in_pool(
    jobs: Sequence[Tuple[str, Callable[[], Any]]],
    num_workers: int,
) -> Dict[str, Any]:
    """Run callables in a process pool; returns a map from job id to result."""

    results: Dict[str, Any] = {}
    if num_workers <= 1:
        for job_id, fn in jobs:
            results[job_id] = fn()
        return results

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        future_map = {ex.submit(fn): job_id for job_id, fn in jobs}
        for fut in as_completed(future_map):
            job_id = future_map[fut]
            results[job_id] = fut.result()
    return results


def build_trial_callable(
    *,
    cfg: CausalBanditConfig,
    horizon: int,
    tau: float,
    seed: int,
    knob_value: float,
    subset_size: int,
    scheduler_mode: str,
    use_full_budget: bool,
    effect_threshold: float,
    sampling: SamplingSettings,
    adaptive_config,
    structure_backend: str,
    raps_params,
    arm_builder_cfg=None,
    cache_enabled: bool = True,
    cache_key: Optional[Tuple] = None,
) -> Callable[[], Tuple[Dict[str, Any], Any, float]]:
    """Create a pure callable that runs a single trial (used in parallel execution)."""

    # To keep semantics identical, we do not reuse PreparedInstance across processes.
    def _fn():
        prepared: Optional[PreparedInstance] = None
        if cache_enabled and cache_key is not None:
            # In parallel mode we skip shared cache; per-process run from scratch.
            prepared = None
        record, summary, optimal_mean = run_trial(
            base_cfg=cfg,
            horizon=horizon,
            tau=tau,
            seed=seed,
            knob_value=knob_value,
            subset_size=subset_size,
            scheduler_mode=scheduler_mode,
            use_full_budget=use_full_budget,
            effect_threshold=effect_threshold,
            sampling=sampling,
            adaptive_config=adaptive_config,
            structure_backend=structure_backend,  # type: ignore[arg-type]
            raps_params=raps_params,
            arm_builder_cfg=arm_builder_cfg,
            prepared=prepared,
            measure_gaps=True,
        )
        return record, summary, optimal_mean

    return _fn


def write_results_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(f"{rec}\n")

