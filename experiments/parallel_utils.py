"""Utilities for running existing trial logic in parallel without changing semantics."""

from __future__ import annotations

import dataclasses
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm.auto import tqdm

from .causal_envs import CausalBanditConfig
from .run_tau_study import (
    SamplingSettings,
    run_trial,
)


def run_jobs_in_pool(
    jobs: Sequence[Tuple[str, Callable[[], Any]]],
    num_workers: int,
    *,
    show_progress: bool = False,
    executor: str = "process",
) -> Dict[str, Any]:
    """Run callables in a process pool; returns a map from job id to result."""

    results: Dict[str, Any] = {}
    if num_workers <= 1:
        for job_id, fn in jobs:
            results[job_id] = fn()
        return results

    ExecutorCls = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
    progress = tqdm(total=len(jobs), desc="Parallel jobs", unit="job", disable=not show_progress)
    with ExecutorCls(max_workers=num_workers) as ex:
        future_map = {ex.submit(fn): job_id for job_id, fn in jobs}
        for fut in as_completed(future_map):
            job_id = future_map[fut]
            results[job_id] = fut.result()
            progress.update(1)
    progress.close()
    return results


def run_trial_worker(
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
) -> Tuple[Dict[str, Any], Any, float]:
    """Top-level worker to allow pickling under multiprocessing."""
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
        prepared=None,
        measure_gaps=True,
    )
    return record, summary, optimal_mean


def write_results_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(f"{rec}\n")
