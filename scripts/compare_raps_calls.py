#!/usr/bin/env python3
"""Deterministic comparison between our RAPSLearner proxy and reference logs.

The script emits a JSONL trace describing every atomic structure-learning call
made by ``RAPSLearner`` under a fixed random seed.  If a reference (official)
log is supplied, the two traces are diffed to highlight the first mismatch.

Typical usage (run from the repository root):

```
python scripts/compare_raps_calls.py \
    --seed 7 \
    --n 8 --k 2 --m 2 --ell 2 \
    --tau 0.6 --horizon 200 \
    --official-log /path/to/official_trace.jsonl
```

The script only writes to ``results/raps_validation`` and never mutates source
files, making it safe to run as evidence for scheduler-only changes.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from experiments.causal_envs import CausalBanditConfig, build_random_scm
from experiments.structure import RAPSLearner, StructureConfig


@dataclass
class StructureCall:
    """Serializable view of a structure-learning step."""

    t: int
    variables: Tuple[int, ...]
    values: Tuple[int, ...]
    reward: float
    expected_mean: float
    candidate_index: int
    sampled_value: int
    parent_discovered: bool
    parent_set: Tuple[int, ...]

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["variables"] = list(self.variables)
        payload["values"] = list(self.values)
        payload["parent_set"] = list(self.parent_set)
        return payload


def _timestamped_output_dir(base: Optional[Path]) -> Path:
    root = base or Path("results") / "raps_validation"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    path = root / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            entries.append(json.loads(text))
    return entries


def _extract_signature(entry: MutableMapping[str, Any]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Return (variables, values) tuple from a log entry."""

    def _maybe_sequence(candidate: Any) -> Optional[Tuple[int, ...]]:
        if candidate is None:
            return None
        if isinstance(candidate, dict):
            if "variables" in candidate and "values" in candidate:
                return tuple(int(v) for v in candidate["variables"]), tuple(int(v) for v in candidate["values"])
            return None
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            return tuple(int(v) for v in candidate)
        return None

    variables: Optional[Tuple[int, ...]] = None
    values: Optional[Tuple[int, ...]] = None

    if "variables" in entry:
        variables = tuple(int(v) for v in entry["variables"])
    if "values" in entry:
        values = tuple(int(v) for v in entry["values"])

    if variables is None or values is None:
        arm = entry.get("arm")
        if isinstance(arm, MutableMapping):
            arm_vars = arm.get("variables")
            arm_vals = arm.get("values")
            if variables is None and arm_vars is not None:
                variables = tuple(int(v) for v in arm_vars)
            if values is None and arm_vals is not None:
                values = tuple(int(v) for v in arm_vals)

    if variables is None:
        candidates = entry.get("nodes") or entry.get("candidate") or entry.get("candidate_index")
        variables = _maybe_sequence(candidates)
    if values is None:
        values = _maybe_sequence(entry.get("assignment") or entry.get("values"))

    if variables is None or values is None:
        raise ValueError(f"Unable to derive structure-call signature from entry: {entry}")

    return variables, values


def generate_structure_trace(
    *,
    seed: int,
    cfg: CausalBanditConfig,
    structure_cfg: StructureConfig,
    cap: Optional[int],
) -> Tuple[List[StructureCall], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    instance = build_random_scm(cfg, rng=rng)
    learner = RAPSLearner(instance, structure_cfg)
    calls: List[StructureCall] = []
    limit = cap if cap is not None else None

    while learner.needs_structure_step():
        if limit is not None and len(calls) >= limit:
            break
        result = learner.step(rng)
        calls.append(
            StructureCall(
                t=len(calls),
                variables=tuple(result.arm.variables),
                values=tuple(result.arm.values),
                reward=float(result.reward),
                expected_mean=float(result.expected_mean),
                candidate_index=int(result.candidate_index),
                sampled_value=int(result.value),
                parent_discovered=bool(result.parent_discovered),
                parent_set=tuple(learner.parent_set()),
            )
        )

    metadata = {
        "seed": seed,
        "config": {
            "n": cfg.n,
            "ell": cfg.ell,
            "k": cfg.k,
            "m": cfg.m,
            "edge_prob": cfg.edge_prob,
            "reward_alpha": cfg.reward_alpha,
            "reward_beta": cfg.reward_beta,
            "reward_logit_scale": cfg.reward_logit_scale,
            "scm_mode": cfg.scm_mode,
            "parent_effect": cfg.parent_effect,
        },
        "structure_config": {
            "effect_threshold": structure_cfg.effect_threshold,
            "min_samples_per_value": structure_cfg.min_samples_per_value,
            "max_steps": structure_cfg.max_steps,
        },
        "structure_cap": limit,
        "total_calls": len(calls),
        "parents_found": len(learner.parent_set()),
        "parent_set": list(learner.parent_set()),
    }
    return calls, metadata


def compare_traces(
    ours: Sequence[StructureCall],
    official: Sequence[Dict[str, Any]],
    *,
    max_report: int = 5,
) -> Dict[str, Any]:
    signature_official = [_extract_signature(dict(entry)) for entry in official]
    mismatches: List[Dict[str, Any]] = []
    for idx, ours_call in enumerate(ours):
        if idx >= len(signature_official):
            mismatches.append(
                {
                    "index": idx,
                    "ours": {"variables": list(ours_call.variables), "values": list(ours_call.values)},
                    "official": None,
                }
            )
            break
        off_vars, off_vals = signature_official[idx]
        ours_sig = (ours_call.variables, ours_call.values)
        if ours_sig != (off_vars, off_vals):
            mismatches.append(
                {
                    "index": idx,
                    "ours": {"variables": list(ours_call.variables), "values": list(ours_call.values)},
                    "official": {"variables": list(off_vars), "values": list(off_vals)},
                }
            )
            if len(mismatches) >= max_report:
                break

    if len(signature_official) > len(ours):
        mismatches.append(
            {
                "index": len(ours),
                "ours": None,
                "official": {
                    "variables": list(signature_official[len(ours)][0]),
                    "values": list(signature_official[len(ours)][1]),
                },
            }
        )

    summary = {
        "ours_steps": len(ours),
        "official_steps": len(signature_official),
        "length_match": len(ours) == len(signature_official),
        "mismatches": mismatches,
    }
    summary["sequences_match"] = summary["length_match"] and not mismatches
    return summary


def run_official_command(command: str, output_path: Path) -> None:
    """Invoke an external command expected to emit a JSONL trace on stdout."""
    env = os.environ.copy()
    proc = subprocess.run(
        command if isinstance(command, str) else shlex.join(command),  # type: ignore[arg-type]
        shell=isinstance(command, str),
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    output_path.write_text(proc.stdout, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Random seed forwarded to the SCM and learner.")
    parser.add_argument("--n", type=int, default=6, help="Number of observed covariates.")
    parser.add_argument("--ell", type=int, default=2, help="Alphabet size per covariate.")
    parser.add_argument("--k", type=int, default=2, help="Number of true reward parents.")
    parser.add_argument("--m", type=int, default=2, help="Maximum intervention size.")
    parser.add_argument("--edge-prob", type=float, default=0.3, help="Edge probability for random DAG sampling.")
    parser.add_argument("--reward-alpha", type=float, default=2.0, help="Beta prior alpha for reward CPD.")
    parser.add_argument("--reward-beta", type=float, default=2.0, help="Beta prior beta for reward CPD.")
    parser.add_argument("--reward-logit-scale", type=float, default=1.0, help="Temperature for reward logits.")
    parser.add_argument(
        "--scm-mode",
        type=str,
        default="beta_dirichlet",
        choices=("beta_dirichlet", "reference"),
        help="Sampling mode for SCM parameters.",
    )
    parser.add_argument("--parent-effect", type=float, default=1.0, help="Reference-mode mixing weight.")

    parser.add_argument("--tau", type=float, default=1.0, help="Fraction of the horizon dedicated to structure.")
    parser.add_argument("--horizon", type=int, default=200, help="Total scheduler horizon used for tau budgeting.")
    parser.add_argument(
        "--structure-steps",
        type=int,
        default=None,
        help="Hard cap on structure calls (overrides tau*horizon when provided).",
    )
    parser.add_argument("--effect-threshold", type=float, default=0.05, help="StructureConfig.effect_threshold.")
    parser.add_argument("--min-samples", type=int, default=20, help="StructureConfig.min_samples_per_value.")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Parent directory for artifacts (defaults to results/raps_validation/<timestamp>).",
    )
    parser.add_argument(
        "--official-log",
        type=Path,
        default=None,
        help="Path to a JSONL trace exported from the official RAPS repository.",
    )
    parser.add_argument(
        "--official-command",
        type=str,
        default=None,
        help="Optional shell command that emits official JSONL trace to stdout; saved next to our log.",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=5,
        help="Maximum mismatch entries to include in the comparison summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tau < 0 or args.tau > 1:
        raise ValueError("--tau must be within [0, 1]")
    if args.structure_steps is not None and args.structure_steps < 0:
        raise ValueError("--structure-steps must be non-negative when provided")

    cap = args.structure_steps
    if cap is None:
        cap = int(args.tau * max(0, args.horizon))

    cfg = CausalBanditConfig(
        n=args.n,
        ell=args.ell,
        k=args.k,
        m=args.m,
        edge_prob=args.edge_prob,
        reward_alpha=args.reward_alpha,
        reward_beta=args.reward_beta,
        reward_logit_scale=args.reward_logit_scale,
        scm_mode=args.scm_mode,
        parent_effect=args.parent_effect,
        seed=args.seed,
    )
    structure_cfg = StructureConfig(effect_threshold=args.effect_threshold, min_samples_per_value=args.min_samples)

    output_root = _timestamped_output_dir(args.output_dir)
    ours_path = output_root / "ours_calls.jsonl"
    metadata_path = output_root / "metadata.json"
    comparison_path = output_root / "comparison.json"
    official_path = output_root / "official_calls.jsonl"

    calls, metadata = generate_structure_trace(seed=args.seed, cfg=cfg, structure_cfg=structure_cfg, cap=cap)
    _write_jsonl(ours_path, [call.to_json() for call in calls])
    metadata["artifacts"] = {"ours_log": str(ours_path)}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    official_entries: Optional[List[Dict[str, Any]]] = None
    if args.official_command:
        run_official_command(args.official_command, official_path)
        args.official_log = official_path
    if args.official_log:
        official_entries = _load_jsonl(args.official_log)
        metadata["artifacts"]["official_log"] = str(args.official_log)

    if official_entries is not None:
        summary = compare_traces(calls, official_entries, max_report=args.max_mismatches)
        comparison_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[raps-compare] Comparison saved to {comparison_path}")
        if summary["sequences_match"]:
            print("[raps-compare] ✅ Structure call sequences match.")
        else:
            print("[raps-compare] ⚠️  Sequences diverge; inspect comparison.json for details.")
    else:
        print("[raps-compare] Official log not provided; only our trace was recorded.")
    print(f"[raps-compare] Artifacts written under {output_root}")


if __name__ == "__main__":
    main()
