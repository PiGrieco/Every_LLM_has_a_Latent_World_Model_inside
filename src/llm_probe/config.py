"""Configuration for LLM trajectory probing (v2 Milestone 1).

All knobs for corpus selection, generation, windowing, and storage live
in a single :class:`ProbeConfig` dataclass so that a run can be defined
by a single YAML file under ``configs/probe*.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProbeConfig:
    """Full configuration for the LLM probing pipeline.

    Attributes are grouped by responsibility: model loading, per-layer
    probing, window pooling, corpus assembly, trajectory generation,
    dataset sizes, storage, and runtime. Defaults target Llama-3-8B on a
    single H100 (80 GB); see ``configs/probe.yaml`` for the canonical
    run and ``configs/probe_smoke.yaml`` for a tiny smoke variant.
    """

    # -- Model --
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    hf_token: Optional[str] = None          # Fallback: env HF_TOKEN
    dtype: str = "float16"                  # "float16" | "bfloat16" | "float32"
    device: str = "cuda"

    # -- Layer probing --
    probe_layer: int = 20                   # 0-indexed; Llama-3-8B has 32 layers

    # -- Window pooling over per-token hidden states --
    window_size: int = 32
    window_stride: int = 16

    # -- Wikipedia corpus --
    wiki_dataset: str = "wikimedia/wikipedia"
    wiki_config: str = "20231101.en"
    wiki_min_tokens: int = 800              # Post-tokenizer length filter
    wiki_max_tokens: int = 4000
    corpus_seed: int = 42
    # Cheap pre-filter before (slow) tokenization; words as a proxy.
    corpus_prefilter_word_min: int = 150
    corpus_tokenize_candidates: int = 100_000

    # -- Generation --
    prompt_tokens: int = 128
    continuation_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    k_trajectories: int = 8
    gen_seed_base: int = 1000               # Per-article seed = base + doc_idx

    # -- Dataset A: forward --
    n_articles_forward: int = 5000

    # -- Dataset B: branching --
    n_articles_branching: int = 2000
    n_pairs_per_article: int = 3
    entropy_threshold: float = 3.0
    branching_window_start: int = 50        # Relative to end of prompt
    branching_window_end: int = 100

    # -- Dataset C: reversed --
    n_articles_reversed: int = 5000
    reversed_passage_tokens: int = 640

    # -- Validation subset (for smoke / sanity checks) --
    n_articles_validation: int = 20

    # -- Storage --
    output_dir: str = "./data/llm_probe"
    shard_size: int = 100
    save_dtype: str = "float16"             # On-disk dtype for hidden states

    # -- Runtime --
    gen_batch_size: int = 8                 # Trajectories per generate() call
    log_every_n_articles: int = 50
