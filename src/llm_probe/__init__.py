"""LLM probing (v2 Milestone 1).

Extract per-layer hidden-state trajectories from a causal LM over a
Wikipedia corpus. Produces three dataset families — forward,
branching, reversed — cached as shard-based torch files with a JSON
manifest. See ``src/llm_probe/README.md`` for the design rationale and
expected timing on an H100.
"""

from .config import ProbeConfig
from .reproducibility import (
    capture_environment,
    capture_model_metadata,
    config_snapshot,
)
from .model_loader import (
    install_activation_hook,
    load_model,
    validate_model_structure,
)
from .activation_extractor import extract_trajectory_states, window_pool
from .corpus_filter import assign_articles_to_datasets, build_article_pool
from .trajectory_generator import (
    extract_reversed_pair,
    generate_branching_pairs,
    generate_forward_trajectories,
)
from .storage import TrajectoryShardReader, TrajectoryShardWriter
from .validation import (
    run_smoke_gate,
    validate_branching_divergence,
    validate_reversed_differ,
    validate_trajectory_statistics,
)

__all__ = [
    "ProbeConfig",
    "capture_environment",
    "capture_model_metadata",
    "config_snapshot",
    "load_model",
    "install_activation_hook",
    "validate_model_structure",
    "window_pool",
    "extract_trajectory_states",
    "build_article_pool",
    "assign_articles_to_datasets",
    "generate_forward_trajectories",
    "generate_branching_pairs",
    "extract_reversed_pair",
    "TrajectoryShardWriter",
    "TrajectoryShardReader",
    "validate_trajectory_statistics",
    "validate_branching_divergence",
    "validate_reversed_differ",
    "run_smoke_gate",
]
