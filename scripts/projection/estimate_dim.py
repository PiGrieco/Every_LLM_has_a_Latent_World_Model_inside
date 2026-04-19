#!/usr/bin/env python3
"""Estimate the intrinsic dimension of M1 hidden states before training.

Usage::

    python -m scripts.projection.estimate_dim \\
        --config configs/projection.yaml \\
        --n-sample 50000 \\
        --output ./outputs/projection/intrinsic_dim.json

The resulting JSON is read by ``scripts/projection/train.py`` when
``cfg.auto_adjust_dim=True``. If the raw intrinsic dim is clamped
(either below ``intrinsic_dim_min`` or above ``intrinsic_dim_max``),
the JSON records that fact so downstream tooling can flag it.
"""
from __future__ import annotations

import argparse
import dataclasses as _dc
import json
import logging
import os
import sys

import yaml

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.projection import (
    ProjectionConfig,
    estimate_intrinsic_dim,
    load_hidden_state_sample,
)


def _load_cfg(path: str) -> ProjectionConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    fields = {f.name for f in _dc.fields(ProjectionConfig)}
    return ProjectionConfig(**{k: v for k, v in raw.items() if k in fields})


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--n-sample", type=int, default=50_000)
    p.add_argument("--output", type=str,
                   default="./outputs/projection/intrinsic_dim.json")
    args = p.parse_args()

    cfg = _load_cfg(args.config)

    print(f"Loading {args.n_sample} hidden states from "
          f"{cfg.probe_data_dir}/{cfg.forward_dataset} ...")
    sample = load_hidden_state_sample(
        cfg.probe_data_dir, cfg.forward_dataset,
        n_sample=args.n_sample, seed=cfg.random_seed,
    )
    print(f"  Loaded {sample.shape[0]} rows × {sample.shape[1]} dims")

    result = estimate_intrinsic_dim(
        sample,
        target_variance=cfg.intrinsic_dim_target_variance,
        dim_min=cfg.intrinsic_dim_min,
        dim_max=cfg.intrinsic_dim_max,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== intrinsic dimension ===")
    print(f"  raw        = {result['raw_intrinsic_dim']}")
    print(f"  clamped to = {result['intrinsic_dim']}  "
          f"(range [{cfg.intrinsic_dim_min}, {cfg.intrinsic_dim_max}])")
    print(f"  variance at dim = {result['actual_variance_at_dim']:.4f}")
    if result["clamped"]:
        print("  NOTE: estimate was clamped. Consider widening the range "
              "if this surprises you.")
    print(f"\n  → wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
