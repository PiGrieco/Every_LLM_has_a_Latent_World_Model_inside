#!/usr/bin/env python3
"""Print summary statistics for a built probe dataset.

Usage::

    python -m scripts.probe.inspect_dataset \\
        --manifest data/llm_probe/forward/manifest.json

The command auto-detects the dataset type from the manifest contents
(looks at the first item) and runs the matching validator:

  - forward/validation → validate_trajectory_statistics
  - branching           → validate_branching_divergence
  - reversed            → validate_reversed_differ
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.llm_probe import (
    TrajectoryShardReader,
    validate_branching_divergence,
    validate_reversed_differ,
    validate_trajectory_statistics,
)


def _infer_kind(reader: TrajectoryShardReader) -> str:
    """Look at the first item to decide which validator to run."""
    for item in reader.iter_items():
        if "trajectory_a" in item and "trajectory_b" in item:
            return "branching"
        if "forward_hidden" in item and "reversed_hidden" in item:
            return "reversed"
        if "hidden_states" in item:
            return "forward"
        break
    return "forward"


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect a probe dataset.")
    p.add_argument("--manifest", type=str, required=True,
                   help="Path to manifest.json (or the dataset dir).")
    p.add_argument("--kind", type=str, default=None,
                   choices=["forward", "branching", "reversed"],
                   help="Override autodetection of dataset kind.")
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    if manifest_path.is_dir():
        dataset_dir = manifest_path
    elif manifest_path.name == "manifest.json":
        dataset_dir = manifest_path.parent
    else:
        dataset_dir = manifest_path

    reader = TrajectoryShardReader(str(dataset_dir))

    print(f"\n=== manifest: {dataset_dir / 'manifest.json'} ===")
    m = reader.manifest
    header = {
        "dataset_dir": m.get("dataset_dir"),
        "n_shards": m.get("n_shards"),
        "n_items_total": m.get("n_items_total"),
        "save_dtype": m.get("save_dtype"),
        "created_at": m.get("created_at"),
        "unique_doc_ids": len(reader.get_doc_ids()),
    }
    print(json.dumps(header, indent=2))

    kind = args.kind or _infer_kind(reader)
    print(f"\n=== validator: {kind} ===")
    if kind == "branching":
        stats = validate_branching_divergence(reader)
    elif kind == "reversed":
        stats = validate_reversed_differ(reader)
    else:
        stats = validate_trajectory_statistics(reader)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
