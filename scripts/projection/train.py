#!/usr/bin/env python3
"""Train the projection autoencoder (stages A, B, optional C).

Usage::

    python -m scripts.projection.train \\
        --config configs/projection.yaml \\
        [--resume-from ckpt.pt] \\
        [--skip-adversarial] \\
        [--stages abc]
"""
from __future__ import annotations

import argparse
import dataclasses as _dc
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.llm_probe import TrajectoryShardReader
from src.projection import (
    PairedHiddenStateDataset,
    ProjectionAutoencoder,
    ProjectionConfig,
    ProjectionTrainer,
)


def _load_cfg(path: str) -> ProjectionConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    fields = {f.name for f in _dc.fields(ProjectionConfig)}
    return ProjectionConfig(**{k: v for k, v in raw.items() if k in fields})


def _maybe_read_intrinsic_dim(cfg: ProjectionConfig) -> ProjectionConfig:
    if not cfg.auto_adjust_dim:
        return cfg
    if not os.path.exists(cfg.intrinsic_dim_json):
        raise SystemExit(
            f"auto_adjust_dim=True but {cfg.intrinsic_dim_json} not found.\n"
            "Run estimate_dim.py first:\n"
            "    python -m scripts.projection.estimate_dim "
            f"--config <yaml> --output {cfg.intrinsic_dim_json}"
        )
    with open(cfg.intrinsic_dim_json) as f:
        d = int(json.load(f).get("intrinsic_dim", cfg.latent_dim))
    if d != cfg.latent_dim:
        logging.getLogger(__name__).info(
            "auto_adjust_dim: latent_dim %d → %d (from %s)",
            cfg.latent_dim, d, cfg.intrinsic_dim_json,
        )
        cfg.latent_dim = d
    return cfg


def _split_doc_ids(all_ids: List[str], frac: float, seed: int):
    rng = random.Random(seed)
    shuffled = list(all_ids)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * frac)
    return set(shuffled[:n_train]), set(shuffled[n_train:])


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--skip-adversarial", action="store_true")
    p.add_argument("--stages", default="abc",
                   help="subset of 'abc' — e.g. 'ab' to skip stage C")
    args = p.parse_args()

    cfg = _load_cfg(args.config)
    if args.skip_adversarial:
        cfg.use_adversarial = False
    cfg = _maybe_read_intrinsic_dim(cfg)
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"

    reader = TrajectoryShardReader(f"{cfg.probe_data_dir}/{cfg.forward_dataset}")
    all_doc_ids = sorted(reader.get_doc_ids())
    train_ids, _val_ids = _split_doc_ids(
        all_doc_ids, cfg.train_split_fraction, cfg.random_seed,
    )
    if not train_ids:
        raise SystemExit("Empty training split — check the forward dataset.")

    # Build dataset (one shot; smoke configs keep it tiny).
    ds = PairedHiddenStateDataset(reader, allowed_doc_ids=train_ids)
    if len(ds) == 0:
        raise SystemExit(
            "PairedHiddenStateDataset is empty — forward trajectories may be "
            "too short for window pairs. Consider smaller window_stride."
        )
    # Probe one sample for d_model.
    h0, _ = ds[0]
    d_model = int(h0.shape[-1])

    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=min(cfg.num_workers, 4),
    )

    ae = ProjectionAutoencoder(cfg, d_model=d_model)
    if args.resume_from:
        torch.load_kwargs = {"map_location": cfg.device, "weights_only": False}
        ckpt = torch.load(args.resume_from, **torch.load_kwargs)
        ae.load_state_dict(ckpt["state_dict"])
        logging.getLogger(__name__).info("Resumed from %s", args.resume_from)

    trainer = ProjectionTrainer(cfg, ae, d_model=d_model)
    save_dir = cfg.checkpoint_dir
    history = trainer.train(loader, stages=args.stages, save_dir=save_dir)

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)
    print("\nTraining finished. Final checkpoint:",
          os.path.join(save_dir, "final.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
