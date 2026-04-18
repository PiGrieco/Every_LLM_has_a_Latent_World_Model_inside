#!/usr/bin/env python3
"""Build trajectory datasets (forward / branching / reversed / validation).

Usage:

    python -m scripts.probe.build_trajectories \\
        --config configs/probe.yaml \\
        --dataset {forward,branching,reversed,validation,all} \\
        [--dry-run] [--resume]

``--dry-run`` prints the plan (dataset sizes, rough timing/disk
estimates) and exits without loading the model. ``--resume`` reads any
existing shard manifests under ``output_dir/<dataset>/`` and skips
``doc_id``s already present.
"""
from __future__ import annotations

import argparse
import dataclasses as _dc
import json
import logging
import os
import sys
import traceback
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.llm_probe import (
    ProbeConfig,
    TrajectoryShardReader,
    TrajectoryShardWriter,
    assign_articles_to_datasets,
    build_article_pool,
    extract_reversed_pair,
    generate_branching_pairs,
    generate_forward_trajectories,
    install_activation_hook,
    load_model,
    validate_branching_divergence,
    validate_model_structure,
    validate_reversed_differ,
    validate_trajectory_statistics,
)

logger = logging.getLogger(__name__)


# ------------------------------ CLI / config ------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build LLM probing trajectories from Wikipedia.",
    )
    p.add_argument("--config", type=str, required=True,
                   help="Path to a probe YAML (see configs/probe.yaml).")
    p.add_argument(
        "--dataset", type=str, default="all",
        choices=["forward", "branching", "reversed", "validation", "all"],
        help="Which dataset(s) to build.",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and exit without loading the model.")
    p.add_argument("--resume", action="store_true",
                   help="Skip articles whose doc_id already appears in the "
                        "existing shards.")
    p.add_argument("--failure-threshold", type=float, default=0.10,
                   help="Abort if more than this fraction of articles fail "
                        "within a single dataset (default 0.10).")
    return p.parse_args()


def _load_cfg(path: str) -> ProbeConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    fields = {f.name for f in _dc.fields(ProbeConfig)}
    clean = {k: v for k, v in raw.items() if k in fields}
    missing = set(raw) - fields
    if missing:
        logger.warning("Ignoring unknown probe config keys: %s", sorted(missing))
    return ProbeConfig(**clean)


# ------------------------------ run planning ------------------------------

def _print_plan(cfg: ProbeConfig, datasets_to_run: list[str]) -> None:
    plan = {
        "model": cfg.model_name,
        "probe_layer": cfg.probe_layer,
        "window": (cfg.window_size, cfg.window_stride),
        "output_dir": cfg.output_dir,
        "save_dtype": cfg.save_dtype,
        "datasets": {},
    }
    bytes_per_item = 0
    for ds in datasets_to_run:
        if ds == "forward":
            n_items = cfg.n_articles_forward * cfg.k_trajectories
            # ~30 windows/trajectory × d_model (assume 4096) × 2 bytes (fp16)
            bytes_per_item = 30 * 4096 * 2
            plan["datasets"]["forward"] = {
                "n_articles": cfg.n_articles_forward,
                "n_items_est": n_items,
                "disk_gb_est": round(n_items * bytes_per_item / 1e9, 2),
            }
        elif ds == "branching":
            n_items = cfg.n_articles_branching * cfg.n_pairs_per_article
            bytes_per_item = 2 * 30 * 4096 * 2  # two trajectories per pair
            plan["datasets"]["branching"] = {
                "n_articles": cfg.n_articles_branching,
                "n_items_est": n_items,
                "disk_gb_est": round(n_items * bytes_per_item / 1e9, 2),
            }
        elif ds == "reversed":
            n_items = cfg.n_articles_reversed
            bytes_per_item = 2 * 30 * 4096 * 2
            plan["datasets"]["reversed"] = {
                "n_articles": cfg.n_articles_reversed,
                "n_items_est": n_items,
                "disk_gb_est": round(n_items * bytes_per_item / 1e9, 2),
            }
        elif ds == "validation":
            plan["datasets"]["validation"] = {
                "n_articles": cfg.n_articles_validation,
                "n_items_est": cfg.n_articles_validation * cfg.k_trajectories,
                "disk_gb_est": "negligible",
            }

    print("\n=== PROBE RUN PLAN ===")
    print(json.dumps(plan, indent=2))
    print(
        "\nTiming rule of thumb on one H100: ~5 s/article for forward, "
        "~15 s/article for branching, ~1 s/article for reversed. "
        "Adjust upwards if the prompt/continuation budget is large.\n"
    )


# --------------------------- per-dataset runner ---------------------------

def _run_dataset(
    name: str,
    articles: list[dict],
    cfg: ProbeConfig,
    model,
    tokenizer,
    hook_handle,
    captured_list,
    resume: bool,
    failure_threshold: float,
) -> None:
    """Run a single dataset (forward / branching / reversed / validation).

    validation reuses the forward generator with a small article list.
    """
    dataset_dir = os.path.join(cfg.output_dir, name)
    with TrajectoryShardWriter(
        dataset_dir=dataset_dir,
        shard_size=cfg.shard_size,
        save_dtype=cfg.save_dtype,
    ) as writer:

        existing = writer.existing_doc_ids() if resume else set()
        if resume and existing:
            logger.info("Resuming %s: skipping %d existing doc_ids", name, len(existing))

        n_processed = 0
        n_failed = 0

        for doc_idx, article in enumerate(tqdm(articles, desc=f"{name}")):
            if resume and article.get("doc_id") in existing:
                continue
            try:
                if name in ("forward", "validation"):
                    items = generate_forward_trajectories(
                        model, tokenizer, article, cfg, doc_idx,
                        hook_handle, captured_list,
                    )
                    if items is None:
                        n_failed += 1
                    else:
                        for it in items:
                            writer.add(it)
                elif name == "branching":
                    items = generate_branching_pairs(
                        model, tokenizer, article, cfg, doc_idx,
                        hook_handle, captured_list,
                    )
                    if items is None:
                        n_failed += 1
                    else:
                        for it in items:
                            writer.add(it)
                elif name == "reversed":
                    item = extract_reversed_pair(
                        model, tokenizer, article, cfg, doc_idx,
                        hook_handle, captured_list,
                    )
                    if item is None:
                        n_failed += 1
                    else:
                        writer.add(item)
                else:
                    raise ValueError(f"Unknown dataset name: {name}")
            except Exception:
                logger.warning(
                    "[%s] article %s (idx %d) FAILED:\n%s",
                    name, article.get("doc_id"), doc_idx, traceback.format_exc(),
                )
                n_failed += 1

            n_processed += 1
            if n_processed >= 10 and (n_failed / max(1, n_processed)) > failure_threshold:
                raise RuntimeError(
                    f"[{name}] failure rate {n_failed}/{n_processed} exceeded "
                    f"{failure_threshold:.0%}; aborting this dataset."
                )
            if n_processed % cfg.log_every_n_articles == 0:
                logger.info(
                    "[%s] processed %d / %d (failures: %d)",
                    name, n_processed, len(articles), n_failed,
                )

    # Validation pass.
    reader = TrajectoryShardReader(dataset_dir)
    if name in ("forward", "validation"):
        print(f"\n=== validate_trajectory_statistics ({name}) ===")
        print(json.dumps(validate_trajectory_statistics(reader), indent=2, default=str))
    elif name == "branching":
        print(f"\n=== validate_branching_divergence ({name}) ===")
        print(json.dumps(validate_branching_divergence(reader), indent=2, default=str))
    elif name == "reversed":
        print(f"\n=== validate_reversed_differ ({name}) ===")
        print(json.dumps(validate_reversed_differ(reader), indent=2, default=str))


# ----------------------------------- main -----------------------------------

def main() -> None:
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )
    args = _parse_args()
    cfg = _load_cfg(args.config)
    os.makedirs(cfg.output_dir, exist_ok=True)

    datasets_to_run = (
        ["forward", "branching", "reversed", "validation"]
        if args.dataset == "all" else [args.dataset]
    )

    if args.dry_run:
        _print_plan(cfg, datasets_to_run)
        return

    # Model + hook (installed ONCE, shared across datasets).
    model, tokenizer, info = load_model(cfg)
    validate_model_structure(model, cfg)
    hook_handle, captured_list = install_activation_hook(model, cfg.probe_layer)

    try:
        pool = build_article_pool(cfg, tokenizer)
        split = assign_articles_to_datasets(pool, cfg)

        for name in datasets_to_run:
            articles = split[name]
            if not articles:
                logger.warning("No articles assigned to dataset %s; skipping.", name)
                continue
            _run_dataset(
                name=name,
                articles=articles,
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                hook_handle=hook_handle,
                captured_list=captured_list,
                resume=args.resume,
                failure_threshold=args.failure_threshold,
            )
    finally:
        hook_handle.remove()


if __name__ == "__main__":
    main()
