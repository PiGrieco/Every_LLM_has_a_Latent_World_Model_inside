"""Build a Wikipedia article pool and partition it into the three datasets.

Loads the streaming Wikipedia dump, applies cheap pre-filters on word
count, tokenizes candidates to measure true token length, and samples
a pool large enough to cover forward / branching / reversed needs.
The pool is cached to disk as JSON so re-runs (and debugging) are
instantaneous.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, List

from tqdm import tqdm

from .config import ProbeConfig

logger = logging.getLogger(__name__)


def _doc_id_for(title: str) -> str:
    return "wiki_" + hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]


def build_article_pool(cfg: ProbeConfig, tokenizer) -> List[dict]:
    """Build (or reload from cache) a pool of usable Wikipedia articles.

    The pool is large enough to cover all three target datasets as
    *overlapping* subsets (not a partition); downstream
    :func:`assign_articles_to_datasets` draws from it.

    Args:
        cfg: Runtime configuration. Uses ``wiki_*`` fields for the
            source, ``corpus_*`` for pre-filtering and sampling,
            ``output_dir`` as the cache root.
        tokenizer: HF tokenizer (Llama-3 or a stand-in for tests).

    Returns:
        List of dicts ``{doc_id, title, text, token_count}``.

    Notes:
        Caching key: ``{cfg.output_dir}/article_pool.json``. The cache
        is rebuilt if it exists but is too small to cover the combined
        dataset requirements; otherwise reused verbatim.
    """
    cache_path = Path(cfg.output_dir) / "article_pool.json"
    total_needed = max(
        cfg.n_articles_forward
        + cfg.n_articles_branching
        + cfg.n_articles_reversed,
        # Overlap-friendly: the union may be smaller than the sum, but we
        # still want the cache to hold at least the largest dataset.
        cfg.n_articles_forward,
        cfg.n_articles_branching,
        cfg.n_articles_reversed,
    )

    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if isinstance(cached, list) and len(cached) >= total_needed:
                logger.info(
                    "Loaded cached article pool (%d articles) from %s",
                    len(cached), cache_path,
                )
                return cached
            logger.info(
                "Cached pool has only %d articles; need %d. Rebuilding.",
                len(cached), total_needed,
            )
        except Exception as exc:
            logger.warning("Could not parse %s (%s); rebuilding.", cache_path, exc)

    # ---- Build fresh ----
    from datasets import load_dataset

    logger.info("Loading Wikipedia (%s / %s) ...", cfg.wiki_dataset, cfg.wiki_config)
    ds = load_dataset(
        cfg.wiki_dataset, cfg.wiki_config, split="train", streaming=False,
    )

    # Cheap pre-filter on raw text length (word count) before tokenization.
    logger.info(
        "Pre-filtering on word count >= %d ...", cfg.corpus_prefilter_word_min,
    )
    prefiltered_indices = []
    for i, row in enumerate(ds):
        text = row["text"]
        if text and len(text.split()) >= cfg.corpus_prefilter_word_min:
            prefiltered_indices.append(i)
    logger.info(
        "After prefilter: %d / %d rows", len(prefiltered_indices), len(ds),
    )

    rng = random.Random(cfg.corpus_seed)
    rng.shuffle(prefiltered_indices)
    take = min(cfg.corpus_tokenize_candidates, len(prefiltered_indices))
    prefiltered_indices = prefiltered_indices[:take]

    # Tokenize in batches to measure the true token count; keep only those
    # inside [wiki_min_tokens, wiki_max_tokens].
    pool: List[dict] = []
    batch_size = 32
    logger.info(
        "Tokenizing %d candidates to filter into [%d, %d] tokens ...",
        take, cfg.wiki_min_tokens, cfg.wiki_max_tokens,
    )
    for b_start in tqdm(range(0, take, batch_size), desc="tokenize", unit="batch"):
        batch_rows = [ds[i] for i in prefiltered_indices[b_start:b_start + batch_size]]
        texts = [r["text"] for r in batch_rows]
        # Fast counting: no truncation so we see the true length; add_special_tokens=False
        # because Llama-3 adds BOS later during per-article generate().
        enc = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )
        for r, ids in zip(batch_rows, enc["input_ids"]):
            tok_count = len(ids)
            if cfg.wiki_min_tokens <= tok_count <= cfg.wiki_max_tokens:
                title = r.get("title") or ""
                text = r["text"]
                pool.append({
                    "doc_id": _doc_id_for(title or text[:64]),
                    "title": title,
                    "text": text,
                    "token_count": tok_count,
                })
                if len(pool) >= total_needed:
                    break
        if len(pool) >= total_needed:
            break

    if len(pool) < total_needed:
        logger.warning(
            "Only found %d articles matching the length window (needed %d). "
            "Consider widening [wiki_min_tokens, wiki_max_tokens] or raising "
            "corpus_tokenize_candidates.",
            len(pool), total_needed,
        )

    # Cache.
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(pool, f)
    logger.info("Cached pool of %d articles to %s", len(pool), cache_path)
    return pool


def assign_articles_to_datasets(
    pool: List[dict], cfg: ProbeConfig,
) -> dict:
    """Split the pool into overlapping subsets for the three datasets.

    The three subsets (forward / branching / reversed) are allowed to
    share articles — they probe the SAME model on DIFFERENT kinds of
    trajectories, so cross-use of articles is natural and, if anything,
    strengthens the comparison. The split is deterministic in
    ``cfg.corpus_seed``.

    Args:
        pool: Output of :func:`build_article_pool`.
        cfg: Runtime configuration.

    Returns:
        dict with keys ``"forward"``, ``"branching"``, ``"reversed"``,
        ``"validation"`` mapping to lists of article dicts.
    """
    rng = random.Random(cfg.corpus_seed)

    def _sample(n: int) -> List[dict]:
        if n >= len(pool):
            return list(pool)
        return rng.sample(pool, n)

    forward = _sample(cfg.n_articles_forward)
    branching = _sample(cfg.n_articles_branching)
    reversed_ = _sample(cfg.n_articles_reversed)
    validation = forward[: cfg.n_articles_validation]

    logger.info(
        "Dataset sizes: forward=%d, branching=%d, reversed=%d, validation=%d",
        len(forward), len(branching), len(reversed_), len(validation),
    )
    return {
        "forward": forward,
        "branching": branching,
        "reversed": reversed_,
        "validation": validation,
    }
