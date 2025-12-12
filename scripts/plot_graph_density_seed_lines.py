#!/usr/bin/env python3
"""Plot cumulative regret vs tau per seed at a fixed graph density."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_records(path: Path) -> List[Dict[str, object]]:
    """Read newline-delimited JSON records."""
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                records.append(json.loads(text))
    if not records:
        raise ValueError(f"No records loaded from {path}")
    return records


def series_by_seed(
    records: Iterable[Dict[str, object]], graph_density: float, metric: str
) -> Dict[int, List[Tuple[float, float]]]:
    """Return ordered (tau, metric) pairs for each seed at the given density."""
    grouped: DefaultDict[int, List[Tuple[float, float]]] = defaultdict(list)
    for row in records:
        if np.isclose(float(row.get("knob_value", 0.0)), graph_density):
            grouped[int(row["seed"])].append((float(row["tau"]), float(row[metric])))
    if not grouped:
        raise ValueError(f"No rows matched knob_value={graph_density}")

    # Sort tau grid so lines draw correctly.
    ordered: Dict[int, List[Tuple[float, float]]] = {}
    for seed, points in grouped.items():
        ordered[seed] = sorted(points, key=lambda pair: pair[0])
    return ordered


def _tau_grid(series: Dict[int, Sequence[Tuple[float, float]]]) -> List[float]:
    """Extract a representative tau grid for labelling."""
    first = next(iter(series.values()))
    return [tau for tau, _ in first]


def plot_lines(
    series: Dict[int, Sequence[Tuple[float, float]]],
    graph_density: float,
    metric: str,
    out_path: Path,
) -> None:
    taus = _tau_grid(series)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seed, points in sorted(series.items()):
        xs, ys = zip(*points)
        ax.plot(xs, ys, "-o", linewidth=1.2, markersize=4, alpha=0.75, label=f"seed {seed}")

    ax.set_xlabel("tau")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric} vs tau at graph density={graph_density}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xticks(taus)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-seed regret vs tau at a fixed graph density."
    )
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=Path("results/graph_density_underactuated/results.jsonl"),
        help="Path to results JSONL.",
    )
    parser.add_argument(
        "--graph-density",
        type=float,
        default=0.1,
        help="knob_value/graph density to slice.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cumulative_regret",
        help="Metric key to plot.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/graph_density_underactuated/plots/line_cumulative_regret_density0.1_seeds.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.results_jsonl)
    lines = series_by_seed(records, args.graph_density, args.metric)
    plot_lines(lines, args.graph_density, args.metric, args.out)


if __name__ == "__main__":
    main()
