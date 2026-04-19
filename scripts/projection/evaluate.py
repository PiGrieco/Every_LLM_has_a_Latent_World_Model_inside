#!/usr/bin/env python3
"""Evaluate the trained projection: retrieval + probe + drift + MSE.

Writes a JSON report consumed by ``scripts/projection/smoke_gate.py``.
"""
from __future__ import annotations

import argparse
import dataclasses as _dc
import json
import logging
import os
import sys
from typing import List

import torch
import yaml

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.llm_probe import TrajectoryShardReader
from src.projection import (
    ProjectionAutoencoder,
    ProjectionConfig,
    build_memory_and_queries,
    on_manifold_drift,
    retrieval_evaluation,
    train_identity_probe,
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
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="./outputs/projection/eval.json")
    args = p.parse_args()

    cfg = _load_cfg(args.config)
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load checkpoint to get d_model and state_dict.
    ckpt = torch.load(args.checkpoint, map_location=cfg.device, weights_only=False)
    d_model = int(ckpt.get("d_model", 4096))
    ae = ProjectionAutoencoder(cfg, d_model=d_model)
    ae.load_state_dict(ckpt["state_dict"])
    ae.to(cfg.device).eval()

    reader = TrajectoryShardReader(f"{cfg.probe_data_dir}/{cfg.forward_dataset}")

    # Retrieval via disjoint-doc-id memory/query split (revised contract).
    memory, queries_h, queries_doc_ids = build_memory_and_queries(
        reader,
        memory_size=cfg.retrieval_memory_size,
        n_queries=cfg.retrieval_queries,
        memory_fraction=cfg.memory_query_split_fraction,
        seed=cfg.random_seed,
    )
    retrieval_report = retrieval_evaluation(
        ae, memory, queries_h, queries_doc_ids, cfg.retrieval_topk,
    )

    # On-manifold drift on held-out query states.
    drift = on_manifold_drift(ae, queries_h[: min(1024, queries_h.shape[0])])

    # Identity probe on the held-out doc-ids.
    probe_report = train_identity_probe(ae, reader, cfg)

    # Reconstruction MSE / var(h).
    with torch.no_grad():
        h_eval = queries_h.to(cfg.device)
        _, h_hat = ae(h_eval)
        mse = ((h_eval - h_hat) ** 2).mean().item()
        var_h = h_eval.var().item()
        mse_ratio = mse / max(var_h, 1e-12)

    report = {
        "config_snapshot": _dc.asdict(cfg),
        "checkpoint": args.checkpoint,
        "reconstruction": {
            "mse": mse, "var_h": var_h, "mse_ratio": mse_ratio,
            "n_samples": int(queries_h.shape[0]),
        },
        "retrieval": retrieval_report,
        "on_manifold_drift": drift,
        "identity_probe": probe_report,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n=== EVALUATE SUMMARY ===")
    print(f"  reconstruction_mse_ratio   : {mse_ratio:.4f}")
    print(f"  retrieval top-5 ratio      : "
          f"{retrieval_report['projected_fraction_of_baseline_topk']['5']:.3f}")
    print(f"  on_manifold_drift          : {drift:.3f}")
    probe_acc = probe_report.get("accuracy", float('nan'))
    probe_chance = probe_report.get("random_baseline", 0.0)
    print(f"  identity_probe_accuracy    : {probe_acc:.3f}  (chance {probe_chance:.3f})")
    print(f"\n  → wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
