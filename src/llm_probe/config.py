"""Configuration for LLM trajectory probing (v2 Milestone 1).

Fields are grouped by responsibility: model loading and revision
pinning, per-layer probing, window pooling, corpus assembly (with a
narrative preference), trajectory generation, dataset sizes, storage,
runtime, and the *smoke-gate thresholds* that the rigid
``scripts/probe/smoke_gate.py`` CLI consumes to decide whether M2 is
allowed to start.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# Default narrative patterns used to preferentially sample articles
# whose lead sentence looks like a biography / event / process. These
# articles are *more likely* to produce meaningful branching behaviour
# than Wikipedia definition stubs. Non-matching articles are still
# eligible — just with lower probability.
_DEFAULT_NARRATIVE_PATTERNS: List[str] = [
    r"^[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s+(was|is)\s+(a|an|the)\s+",  # biography
    r"^The\s+[A-Z][a-z]+\s+(of|Battle|War|Revolution)",            # event
    r"^[A-Z][a-z]+(\s+[A-Z][a-z]+)*\s+\([^)]+\)\s+(was|is)",       # person w/ parenthetical
]


@dataclass
class ProbeConfig:
    """Full configuration for the LLM probing pipeline.

    Defaults target Llama-3-8B on a single H100 (80 GB); see
    ``configs/probe.yaml`` for the canonical run and
    ``configs/probe_smoke.yaml`` for a tiny end-to-end variant used by
    the smoke gate.
    """

    # -- Model --
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    model_revision: str = "main"            # Or pin a commit SHA for full reproducibility
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
    wiki_min_tokens: int = 800
    wiki_max_tokens: int = 4000
    corpus_seed: int = 42
    # Cheap pre-filter before (slow) tokenization; words as a proxy.
    corpus_prefilter_word_min: int = 150
    corpus_tokenize_candidates: int = 100_000
    # Narrative preference: articles whose lead sentence matches one of the
    # patterns below are sampled with higher weight into the branching set.
    # This is NOT a strict filter; non-narrative articles still reach the
    # pool, they're just less likely to be picked for branching.
    narrative_preference_weight: float = 2.0
    narrative_patterns: List[str] = field(
        default_factory=lambda: list(_DEFAULT_NARRATIVE_PATTERNS)
    )

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
    # Sampling weight for narrative articles in the branching dataset.
    branching_narrative_weight: float = 3.0

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

    # -- Smoke-gate thresholds (HARD gates, not warnings) --
    # Consumed by src/llm_probe/validation.py::run_smoke_gate and by
    # scripts/probe/smoke_gate.py which exits non-zero on failure.
    gate_min_fraction_finite: float = 1.0
    gate_norm_min: float = 1.0
    gate_norm_max: float = 200.0
    gate_min_n_windows_median: float = 10.0
    gate_min_branching_divergence_fraction: float = 0.7
    gate_max_reversed_cosine_similarity: float = 0.95
