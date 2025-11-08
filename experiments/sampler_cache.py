"""Reusable sampling cache to avoid repeated SCM evaluations across runs."""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Tuple

import numpy as np

ArmKey = Tuple[Tuple[int, ...], Tuple[int, ...]]
SubsetKey = Tuple[Tuple[int, ...], Tuple[int, ...]]


@dataclass
class _CacheEntry:
    value: float
    next_state: Dict[str, object]


class SamplerCache:
    """Caches stochastic evaluations keyed by RNG state and call signature."""

    def __init__(self) -> None:
        self._reward_cache: Dict[Hashable, _CacheEntry] = {}
        self._mean_cache: Dict[Hashable, _CacheEntry] = {}
        self._subset_cache: Dict[Hashable, _CacheEntry] = {}

    def _snapshot(self, rng: np.random.Generator) -> bytes:
        state = rng.bit_generator.state
        return pickle.dumps(state, protocol=5)

    def _restore(self, rng: np.random.Generator, state: Dict[str, object]) -> None:
        rng.bit_generator.state = copy.deepcopy(state)

    def _lookup_or_compute(
        self,
        *,
        cache: Dict[Hashable, _CacheEntry],
        key: Hashable,
        rng: np.random.Generator,
        compute: Callable[[], float],
    ) -> float:
        fingerprint = (key, self._snapshot(rng))
        if fingerprint in cache:
            entry = cache[fingerprint]
            self._restore(rng, entry.next_state)
            return entry.value
        value = compute()
        cache[fingerprint] = _CacheEntry(value=value, next_state=copy.deepcopy(rng.bit_generator.state))
        return value

    def sample_reward(
        self,
        *,
        arm_key: ArmKey,
        rng: np.random.Generator,
        compute: Callable[[], float],
    ) -> float:
        key = ("sample_reward", arm_key)
        return self._lookup_or_compute(cache=self._reward_cache, key=key, rng=rng, compute=compute)

    def estimate_arm_mean(
        self,
        *,
        arm_key: ArmKey,
        n_mc: int,
        rng: np.random.Generator,
        compute: Callable[[], float],
    ) -> float:
        key = ("estimate_arm_mean", arm_key, int(n_mc))
        return self._lookup_or_compute(cache=self._mean_cache, key=key, rng=rng, compute=compute)

    def estimate_subset_mean(
        self,
        *,
        subset_key: SubsetKey,
        n_mc: int,
        rng: np.random.Generator,
        compute: Callable[[], float],
    ) -> float:
        key = ("estimate_subset_mean", subset_key, int(n_mc))
        return self._lookup_or_compute(cache=self._subset_cache, key=key, rng=rng, compute=compute)


__all__ = ["ArmKey", "SamplerCache", "SubsetKey"]
