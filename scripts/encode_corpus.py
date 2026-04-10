#!/usr/bin/env python3
"""
Pre-encode WikiText-103 for D2 experiments.

This script precomputes everything that doesn't depend on learnable
parameters, so the GPU-intensive encoding and LM scoring happen
once and are cached to disk.

Usage:
    python -m scripts.encode_corpus --config configs/d2_wikitext.yaml

    # Or with explicit params:
    python -m scripts.encode_corpus \
        --encoder sentence-transformers/all-MiniLM-L6-v2 \
        --lm gpt2-medium \
        --max_articles 5000 \
        --cache_dir ./cache

Output:
    ./cache/wikitext_embeddings.pt   — per-article paragraph embeddings
    ./cache/wikitext_lm_scores.pt    — per-transition LM log-probs
    ./cache/wikitext_preprocessed.pt — after centering + PC removal
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.wikitext import (
    load_wikitext_articles,
    encode_articles,
    compute_lm_log_probs,
)
from src.data.preprocessing import preprocess_trajectory_dataset


def main():
    parser = argparse.ArgumentParser(description="Pre-encode WikiText-103")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--lm", type=str, default="gpt2-medium")
    parser.add_argument("--max_articles", type=int, default=None,
                        help="Limit number of articles (None = all)")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--skip_lm", action="store_true",
                        help="Skip LM log-prob computation (corpus-only mode)")
    args = parser.parse_args()

    if args.config:
        cfg = Config.from_yaml(args.config)
    else:
        cfg = Config()
        cfg.encoder_name = args.encoder
        cfg.lm_name = args.lm
        cfg.cache_dir = args.cache_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.cache_dir, exist_ok=True)

    # ---- Step 1: Load articles ----
    print("Step 1/4: Loading WikiText-103 articles...")
    articles = load_wikitext_articles(
        split=cfg.wikitext_split,
        min_paragraphs=cfg.min_paragraphs,
        max_paragraphs=cfg.max_paragraphs,
        max_articles=args.max_articles,
    )
    print(f"  → {len(articles)} articles loaded")

    # Save article metadata
    meta = [{"title": t, "n_paragraphs": len(p)} for t, p in articles]
    torch.save(meta, os.path.join(cfg.cache_dir, "wikitext_metadata.pt"))

    # ---- Step 2: Encode paragraphs ----
    print("\nStep 2/4: Encoding paragraphs with sentence-transformer...")
    embeddings = encode_articles(
        articles,
        encoder_name=cfg.encoder_name,
        batch_size=cfg.encode_batch_size,
        device=device,
        cache_path=os.path.join(cfg.cache_dir, "wikitext_embeddings.pt"),
    )

    total_paragraphs = sum(len(e) for e in embeddings)
    print(f"  → {total_paragraphs} paragraph embeddings ({embeddings[0].shape[1]}D)")

    # ---- Step 3: Preprocess (center, remove top PCs, normalize) ----
    print("\nStep 3/4: Preprocessing embeddings...")
    processed, preprocessor = preprocess_trajectory_dataset(
        embeddings,
        n_pca_remove=cfg.n_pca_remove,
        normalize=cfg.normalize_embeddings,
    )
    torch.save(
        {"embeddings": processed, "preprocessor_mean": preprocessor.mean,
         "preprocessor_components": preprocessor.top_components},
        os.path.join(cfg.cache_dir, "wikitext_preprocessed.pt"),
    )
    print(f"  → Removed top {cfg.n_pca_remove} PCs, normalized")

    # ---- Step 4: LM log-probabilities ----
    if not args.skip_lm:
        print(f"\nStep 4/4: Computing LM log-probs with {cfg.lm_name}...")
        print("  (This is the slow step — grab a coffee)")
        log_probs = compute_lm_log_probs(
            articles,
            lm_name=cfg.lm_name,
            max_context=cfg.lm_max_context,
            batch_size=cfg.lm_batch_size,
            device=device,
            cache_path=os.path.join(cfg.cache_dir, "wikitext_lm_scores.pt"),
        )
        total_transitions = sum(len(lp) for lp in log_probs)
        print(f"  → {total_transitions} transition scores computed")
    else:
        print("\nStep 4/4: Skipped LM scoring (corpus-only mode)")

    # ---- Summary ----
    print("\n" + "=" * 50)
    print("ENCODING COMPLETE")
    print("=" * 50)
    print(f"  Articles:     {len(articles)}")
    print(f"  Paragraphs:   {total_paragraphs}")
    print(f"  Encoder dim:  {embeddings[0].shape[1]}")
    print(f"  Cache dir:    {cfg.cache_dir}/")
    print(f"\nCached files:")
    for f in sorted(os.listdir(cfg.cache_dir)):
        size = os.path.getsize(os.path.join(cfg.cache_dir, f))
        print(f"  {f:<40} {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
