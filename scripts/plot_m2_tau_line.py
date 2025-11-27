"""Plot cumulative regret vs tau at a fixed intervention size.

This reads the JSONL output produced by `perfcba.experiments.run_tau_study`
and draws a single line plot of mean cumulative regret as tau varies while
holding the intervention_size (m) fixed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_records(results_path: Path) -> List[dict]:
    records: List[dict] = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records loaded from {results_path}")
    return records


def aggregate_tau_slice(
    records: Iterable[dict],
    intervention_size: float,
    metric: str,
) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
    grouped: DefaultDict[float, List[float]] = defaultdict(list)
    for row in records:
        if np.isclose(row["intervention_size"], intervention_size):
            grouped[float(row["tau"])].append(float(row[metric]))
    if not grouped:
        raise ValueError(f"No rows matched intervention_size={intervention_size}")

    taus = sorted(grouped.keys())
    means = [float(np.mean(grouped[tau])) for tau in taus]
    stds = [float(np.std(grouped[tau])) for tau in taus]
    return taus, means, stds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cumulative regret vs tau at fixed intervention_size (m)."
    )
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=Path("results/intervention_length_small/results.jsonl"),
        help="Path to results.jsonl produced by run_tau_study.",
    )
    parser.add_argument(
        "--m",
        type=float,
        default=2.0,
        help="intervention_size (m) to hold fixed when sweeping tau.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cumulative_regret",
        help="Metric key to average over.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/intervention_length_small/plots/line_cumulative_regret_m2_tau.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    records = load_records(args.results_jsonl)
    taus, means, stds = aggregate_tau_slice(records, args.m, args.metric)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(taus, means, "-o", label=f"m={args.m}")
    ax.fill_between(taus, np.array(means) - stds, np.array(means) + stds, alpha=0.2)
    ax.set_xlabel("tau")
    ax.set_ylabel(f"mean {args.metric}")
    upper = float(np.max(np.array(means) + np.array(stds)))
    ax.set_ylim(0.0, upper)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.suptitle(f"{args.metric} vs tau at m={args.m}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
