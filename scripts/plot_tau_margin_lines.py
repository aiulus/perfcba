"""Plot tau/ hard-margin slices from tau study JSONL output.

This script is tailored to the tau study experiment output produced by
`perfcba.experiments.run_tau_study`. It draws two line plots:
  1) Fix tau at --tau and sweep over hard_margin values.
  2) Fix hard_margin at --hard-margin and sweep over tau values.
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


def aggregate(
    records: Iterable[dict],
    filter_key: str,
    filter_value: float,
    sweep_key: str,
    metric: str,
) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
    grouped: DefaultDict[float, List[float]] = defaultdict(list)
    for row in records:
        if np.isclose(row[filter_key], filter_value):
            grouped[float(row[sweep_key])].append(float(row[metric]))
    if not grouped:
        raise ValueError(f"No rows matched {filter_key}={filter_value}")

    xs = sorted(grouped.keys())
    means = [float(np.mean(grouped[x])) for x in xs]
    stds = [float(np.std(grouped[x])) for x in xs]
    return xs, means, stds


def draw_line(
    ax: plt.Axes,
    xs: Sequence[float],
    ys: Sequence[float],
    yerr: Sequence[float],
    *,
    label: str,
    xlabel: str,
) -> None:
    ax.plot(xs, ys, "-o", label=label)
    ax.fill_between(xs, np.array(ys) - yerr, np.array(ys) + yerr, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean cumulative regret")
    upper = float(np.max(np.array(ys) + np.array(yerr)))
    ax.set_ylim(0.0, upper)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot tau and hard_margin slices from tau study JSONL output."
    )
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=Path("results/eps_tau_heatmap/results.jsonl"),
        help="Path to results.jsonl produced by run_tau_study.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="Tau value to hold fixed when sweeping hard_margin.",
    )
    parser.add_argument(
        "--hard-margin",
        dest="hard_margin",
        type=float,
        default=0.35,
        help="Hard margin to hold fixed when sweeping tau.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cumulative_regret",
        help="Metric key to average over.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("results/eps_tau_heatmap/line_plots"),
        help="Path prefix for output PNGs; suffixes will be added.",
    )
    args = parser.parse_args()

    records = load_records(args.results_jsonl)

    xs_margin, means_margin, stds_margin = aggregate(
        records,
        filter_key="tau",
        filter_value=args.tau,
        sweep_key="hard_margin",
        metric=args.metric,
    )
    xs_tau, means_tau, stds_tau = aggregate(
        records,
        filter_key="hard_margin",
        filter_value=args.hard_margin,
        sweep_key="tau",
        metric=args.metric,
    )

    out_dir = args.out_prefix.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_margin, ax_margin = plt.subplots(figsize=(5, 4))
    draw_line(
        ax_margin,
        xs_margin,
        means_margin,
        stds_margin,
        label=f"tau={args.tau}",
        xlabel="hard_margin",
    )
    fig_margin.suptitle(f"{args.metric} at tau={args.tau}")
    fig_margin.tight_layout()
    out_margin = args.out_prefix.with_name(f"{args.out_prefix.name}_tau_fixed.png")
    fig_margin.savefig(out_margin, dpi=200)

    fig_tau, ax_tau = plt.subplots(figsize=(5, 4))
    draw_line(
        ax_tau,
        xs_tau,
        means_tau,
        stds_tau,
        label=f"hard_margin={args.hard_margin}",
        xlabel="tau",
    )
    fig_tau.suptitle(f"{args.metric} at hard_margin={args.hard_margin}")
    fig_tau.tight_layout()
    out_tau = args.out_prefix.with_name(f"{args.out_prefix.name}_hard_margin_fixed.png")
    fig_tau.savefig(out_tau, dpi=200)

    print(f"Wrote {out_margin}")
    print(f"Wrote {out_tau}")


if __name__ == "__main__":
    main()
