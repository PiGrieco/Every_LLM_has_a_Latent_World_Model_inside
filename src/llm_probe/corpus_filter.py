"""Build a Wikipedia article pool and partition it into the three datasets.

Loads the streaming Wikipedia dump, applies cheap word-count pre-filters,
tokenizes candidates to measure true token length, flags articles whose
lead sentence looks "narrative" (biographies / events / processes), and
caches the result. Downstream :func:`assign_articles_to_datasets`
draws overlapping subsets for the three target datasets; branching
uses a narrative-weighted sampling so we don't train the metric on
definition stubs that structurally don't branch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import random
from pathlib import Path
from typing import Any, List

from tqdm import tqdm

from .config import ProbeConfig

logger = logging.getLogger(__name__)


def _doc_id_for(title: str) -> str:
    return "wiki_" + hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]


def _compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p) for p in patterns]


def _is_narrative(text: str, compiled: List[re.Pattern]) -> bool:
    """Return True iff the first 200 chars match any narrative pattern."""
    lead = text[:200].strip()
    return any(p.search(lead) for p in compiled)


def build_article_pool(cfg: ProbeConfig, tokenizer) -> List[dict]:
    """Build (or reload from cache) a pool of usable Wikipedia articles.

    The pool holds enough articles to cover all three target datasets
    as *overlapping* subsets (not a partition).

    Args:
        cfg: Runtime configuration. Uses ``wiki_*`` fields for source,
            ``corpus_*`` for pre-filtering and sampling,
            ``narrative_patterns`` to flag narrative articles, and
            ``output_dir`` as the cache root.
        tokenizer: HF tokenizer (Llama-3 in production, any tokenizer
            in tests).

    Returns:
        List of dicts with keys ``doc_id``, ``title``, ``text``,
        ``token_count``, ``is_narrative``.
    """
    cache_path = Path(cfg.output_dir) / "article_pool.json"
    total_needed = max(
        cfg.n_articles_forward,
        cfg.n_articles_branching,
        cfg.n_articles_reversed,
    )
    # Combined target so one pool covers non-overlapping worst-case too.
    target_pool_size = max(
        total_needed,
        # Enough headroom for union sampling when the three dataset sizes
        # together exceed the individual max.
        cfg.n_articles_forward
        + cfg.n_articles_branching
        + cfg.n_articles_reversed,
    )

    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if isinstance(cached, list) and len(cached) >= total_needed:
                # If cache lacks is_narrative (pre-revision), recompute it.
                if cached and "is_narrative" not in cached[0]:
                    compiled = _compile_patterns(cfg.narrative_patterns)
                    for art in cached:
                        art["is_narrative"] = _is_narrative(art.get("text", ""), compiled)
                    with open(cache_path, "w") as f:
                        json.dump(cached, f)
                    logger.info(
                        "Backfilled is_narrative on %d cached articles", len(cached),
                    )
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

    compiled = _compile_patterns(cfg.narrative_patterns)

    logger.info("Loading Wikipedia (%s / %s) ...", cfg.wiki_dataset, cfg.wiki_config)
    ds = load_dataset(
        cfg.wiki_dataset, cfg.wiki_config, split="train", streaming=False,
    )

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

    pool: List[dict] = []
    batch_size = 32
    logger.info(
        "Tokenizing %d candidates to filter into [%d, %d] tokens ...",
        take, cfg.wiki_min_tokens, cfg.wiki_max_tokens,
    )
    for b_start in tqdm(range(0, take, batch_size), desc="tokenize", unit="batch"):
        batch_rows = [ds[i] for i in prefiltered_indices[b_start:b_start + batch_size]]
        texts = [r["text"] for r in batch_rows]
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
                    "is_narrative": _is_narrative(text, compiled),
                })
                if len(pool) >= target_pool_size:
                    break
        if len(pool) >= target_pool_size:
            break

    if len(pool) < total_needed:
        logger.warning(
            "Only found %d articles matching the length window (needed at least %d). "
            "Widen [wiki_min_tokens, wiki_max_tokens] or raise "
            "corpus_tokenize_candidates.", len(pool), total_needed,
        )

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(pool, f)
    logger.info(
        "Cached pool of %d articles to %s (%d narrative)",
        len(pool), cache_path, sum(1 for a in pool if a.get("is_narrative")),
    )
    return pool


def _weighted_sample_without_replacement(
    population: List[dict], weights: List[float], k: int, rng: random.Random,
) -> List[dict]:
    """Efraimidis–Spirakis weighted reservoir sampling without replacement."""
    if k >= len(population):
        return list(population)
    # Key_i = u_i ** (1 / w_i); take top-k by key.
    keyed = []
    for item, w in zip(population, weights):
        if w <= 0:
            w = 1e-12
        u = rng.random()
        # Use log for stability: log(u) / w
        import math
        key = math.log(u) / w
        keyed.append((key, item))
    keyed.sort(key=lambda t: t[0], reverse=True)
    return [item for _, item in keyed[:k]]


def assign_articles_to_datasets(
    pool: List[dict], cfg: ProbeConfig,
) -> dict:
    """Split the pool into overlapping subsets for the three datasets.

    forward / reversed get uniform samples. branching gets a weighted
    sample that favours ``is_narrative=True`` articles by
    ``cfg.branching_narrative_weight``. Any single article may appear in
    multiple datasets — the three measurements probe different
    properties of the SAME LM on the SAME text, and cross-use is
    natural.

    Returns:
        dict with keys ``"forward"``, ``"branching"``, ``"reversed"``,
        ``"validation"``.
    """
    rng = random.Random(cfg.corpus_seed)

    def _uniform(n: int) -> List[dict]:
        if n >= len(pool):
            return list(pool)
        return rng.sample(pool, n)

    def _narrative_weighted(n: int, weight: float) -> List[dict]:
        if n >= len(pool):
            return list(pool)
        weights = [
            weight if a.get("is_narrative") else 1.0 for a in pool
        ]
        return _weighted_sample_without_replacement(pool, weights, n, rng)

    forward = _uniform(cfg.n_articles_forward)
    branching = _narrative_weighted(
        cfg.n_articles_branching, cfg.branching_narrative_weight,
    )
    reversed_ = _uniform(cfg.n_articles_reversed)
    validation = sorted(
        forward[: cfg.n_articles_validation],
        key=lambda a: a.get("doc_id", ""),
    )

    n_narrative_branch = sum(1 for a in branching if a.get("is_narrative"))
    logger.info(
        "Dataset sizes: forward=%d, branching=%d (%d narrative), "
        "reversed=%d, validation=%d",
        len(forward), len(branching), n_narrative_branch,
        len(reversed_), len(validation),
    )
    return {
        "forward": forward,
        "branching": branching,
        "reversed": reversed_,
        "validation": validation,
    }
