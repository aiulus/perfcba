"""Persistence helpers for tau-study trials."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .causal_envs import CausalBanditConfig, InterventionArm
from .scheduler import RunSummary, RoundLog

ARTIFACT_VERSION = "1"


def _format_tag(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4g}"


def _serialize_config(cfg: CausalBanditConfig) -> Dict[str, Any]:
    data = asdict(cfg)
    # numpy scalars may sneak in; coerce to builtins
    return json.loads(json.dumps(data))


def _serialize_adaptive_cfg(cfg: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if cfg is None:
        return None
    return json.loads(json.dumps(cfg))


@dataclass(frozen=True)
class TrialIdentity:
    """Uniquely identifies a tau-study trial."""

    config: Dict[str, Any]
    horizon: int
    tau: float
    seed: int
    knob_value: float
    scheduler: str
    subset_size: int
    use_full_budget: bool
    effect_threshold: float
    min_samples: int
    adaptive_config: Optional[Dict[str, Any]]

    def payload(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "horizon": self.horizon,
            "tau": self.tau,
            "seed": self.seed,
            "knob_value": self.knob_value,
            "scheduler": self.scheduler,
            "subset_size": self.subset_size,
            "use_full_budget": self.use_full_budget,
            "effect_threshold": self.effect_threshold,
            "min_samples": self.min_samples,
            "adaptive_config": self.adaptive_config,
        }


def make_trial_identity(
    config: CausalBanditConfig,
    *,
    horizon: int,
    tau: float,
    seed: int,
    knob_value: float,
    scheduler: str,
    subset_size: int,
    use_full_budget: bool,
    effect_threshold: float,
    min_samples: int,
    adaptive_config: Optional[Dict[str, Any]],
) -> TrialIdentity:
    return TrialIdentity(
        config=_serialize_config(config),
        horizon=int(horizon),
        tau=float(tau),
        seed=int(seed),
        knob_value=float(knob_value),
        scheduler=str(scheduler),
        subset_size=int(subset_size),
        use_full_budget=bool(use_full_budget),
        effect_threshold=float(effect_threshold),
        min_samples=int(min_samples),
        adaptive_config=_serialize_adaptive_cfg(adaptive_config),
    )


def _identity_digest(identity: TrialIdentity) -> str:
    payload = json.dumps(identity.payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def artifact_filename(identity: TrialIdentity) -> str:
    knob_tag = _format_tag(identity.knob_value)
    tau_tag = _format_tag(identity.tau)
    digest = _identity_digest(identity)[:12]
    return f"trial_knob-{knob_tag}_tau-{tau_tag}_seed-{identity.seed}_{digest}.json"


def _serialize_round_log(log: RoundLog) -> Dict[str, Any]:
    return {
        "t": log.t,
        "mode": log.mode,
        "arm": {
            "variables": list(log.arm.variables),
            "values": list(log.arm.values),
        },
        "reward": log.reward,
        "expected_mean": log.expected_mean,
        "parent_set": list(log.parent_set),
        "arm_count": log.arm_count,
        "burst_remaining": log.burst_remaining,
        "stall_flag": log.stall_flag,
        "improvement_stat": log.improvement_stat,
    }


def _deserialize_round_log(payload: Dict[str, Any]) -> RoundLog:
    arm_payload = payload["arm"]
    arm = InterventionArm(tuple(arm_payload["variables"]), tuple(arm_payload["values"]))
    return RoundLog(
        t=int(payload["t"]),
        mode=str(payload["mode"]),
        arm=arm,
        reward=float(payload["reward"]),
        expected_mean=float(payload["expected_mean"]),
        parent_set=tuple(payload["parent_set"]),
        arm_count=int(payload["arm_count"]),
        burst_remaining=payload.get("burst_remaining"),
        stall_flag=payload.get("stall_flag"),
        improvement_stat=payload.get("improvement_stat"),
    )


def _serialize_summary(summary: RunSummary) -> Dict[str, Any]:
    return {
        "logs": [_serialize_round_log(log) for log in summary.logs],
        "structure_steps": summary.structure_steps,
        "exploit_steps": summary.exploit_steps,
        "final_parent_set": list(summary.final_parent_set),
        "finished_discovery_round": summary.finished_discovery_round,
    }


def _deserialize_summary(payload: Dict[str, Any]) -> RunSummary:
    logs = [_deserialize_round_log(entry) for entry in payload.get("logs", [])]
    return RunSummary(
        logs=logs,
        structure_steps=int(payload["structure_steps"]),
        exploit_steps=int(payload["exploit_steps"]),
        final_parent_set=tuple(payload["final_parent_set"]),
        finished_discovery_round=payload.get("finished_discovery_round"),
    )


@dataclass
class TrialArtifact:
    identity: TrialIdentity
    record: Dict[str, float]
    summary: RunSummary
    optimal_mean: float
    metadata: Dict[str, Any]
    version: str = ARTIFACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "identity": self.identity.payload(),
            "record": self.record,
            "summary": _serialize_summary(self.summary),
            "optimal_mean": self.optimal_mean,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "TrialArtifact":
        identity = TrialIdentity(**payload["identity"])
        summary = _deserialize_summary(payload["summary"])
        return TrialArtifact(
            identity=identity,
            record=payload["record"],
            summary=summary,
            optimal_mean=float(payload["optimal_mean"]),
            metadata=payload.get("metadata", {}),
            version=str(payload.get("version", ARTIFACT_VERSION)),
        )


def write_trial_artifact(directory: Path, artifact: TrialArtifact) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / artifact_filename(artifact.identity)
    blob = json.dumps(artifact.to_dict(), separators=(",", ":"))
    path.write_text(blob, encoding="utf-8")
    return path


def load_trial_artifact(directory: Path, identity: TrialIdentity) -> Optional[TrialArtifact]:
    path = directory / artifact_filename(identity)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact = TrialArtifact.from_dict(payload)
    # extra safety: confirm identity matches
    if artifact.identity.payload() != identity.payload():
        return None
    return artifact


def read_trial_artifact(path: Path) -> TrialArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TrialArtifact.from_dict(payload)


def load_all_artifacts(directory: Path) -> List[TrialArtifact]:
    if directory is None or not directory.exists():
        return []
    artifacts: List[TrialArtifact] = []
    for path in sorted(directory.glob("trial_*.json")):
        artifacts.append(read_trial_artifact(path))
    return artifacts


def build_metadata(*, cli_args: Dict[str, Any]) -> Dict[str, Any]:
    def _coerce(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [_coerce(item) for item in value]
        return value

    coerced = {key: _coerce(val) for key, val in cli_args.items()}
    coerced["created_at"] = datetime.now(timezone.utc).isoformat()
    return coerced


__all__ = [
    "TrialArtifact",
    "TrialIdentity",
    "build_metadata",
    "load_all_artifacts",
    "load_trial_artifact",
    "make_trial_identity",
    "read_trial_artifact",
    "write_trial_artifact",
]
