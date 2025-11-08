#!/usr/bin/env python3
"""Compare two tau study JSONL outputs for exact equality.

Typical usage (run from repository root):

    python scripts/compare_tau_results.py \
        --lhs results/tau_study_baseline/results.jsonl \
        --rhs results/tau_study_new/results.jsonl

The tool canonicalizes each record using a configurable tuple of keys
(``--key``) and reports the first mismatch, making it easy to ensure that new
sampling modes preserve behaviour.  It exits with status 0 on success and 1 on
any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def _resolve_jsonl(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "results.jsonl"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"No JSONL file found at {path} (checked {candidate})")


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _key_for(record: Dict[str, object], keys: Sequence[str]) -> Tuple[object, ...]:
    return tuple(record.get(key) for key in keys)


def _build_index(records: Iterable[Dict[str, object]], keys: Sequence[str]) -> Dict[Tuple[object, ...], Dict[str, object]]:
    index: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for record in records:
        signature = _key_for(record, keys)
        if signature in index:
            raise ValueError(f"Duplicate record for key {signature}")
        index[signature] = record
    return index


def compare(lhs: Dict[Tuple[object, ...], Dict[str, object]], rhs: Dict[Tuple[object, ...], Dict[str, object]]) -> Tuple[bool, str]:
    if lhs.keys() != rhs.keys():
        missing_lhs = sorted(rhs.keys() - lhs.keys())
        missing_rhs = sorted(lhs.keys() - rhs.keys())
        if missing_lhs:
            return False, f"LHS missing {len(missing_lhs)} keys, first: {missing_lhs[0]}"
        return False, f"RHS missing {len(missing_rhs)} keys, first: {missing_rhs[0]}"
    for key in sorted(lhs.keys()):
        a = lhs[key]
        b = rhs[key]
        if a != b:
            return False, f"Mismatch for key {key}: lhs={a}, rhs={b}"
    return True, "Records match"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tau study JSONL outputs.")
    parser.add_argument("--lhs", type=Path, required=True, help="Path to baseline JSONL file or directory.")
    parser.add_argument("--rhs", type=Path, required=True, help="Path to candidate JSONL file or directory.")
    parser.add_argument(
        "--key",
        action="append",
        default=["instance_id", "tau", "knob_value", "seed"],
        help="Field used to align records (default: %(default)s). Specify multiple times for multiple keys.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lhs_path = _resolve_jsonl(args.lhs)
    rhs_path = _resolve_jsonl(args.rhs)
    lhs_records = _load_jsonl(lhs_path)
    rhs_records = _load_jsonl(rhs_path)
    keys: Sequence[str] = args.key
    lhs_index = _build_index(lhs_records, keys)
    rhs_index = _build_index(rhs_records, keys)
    ok, message = compare(lhs_index, rhs_index)
    print(message)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
