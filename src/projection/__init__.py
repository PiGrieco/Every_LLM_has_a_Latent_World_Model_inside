"""Projection milestone (M2): Φ / Ψ with retrieval-based readout.

Public API re-exports for the M2 modules. See ``README.md`` for the
design notes and the "Design decisions revised" section for the
rationale behind the two-hard-gate policy.
"""

from .autoencoder import ProjectionAutoencoder
from .config import ProjectionConfig
from .decoder import Decoder
from .discriminator import Discriminator
from .encoder import Encoder
from .identity_probe import train_identity_probe
from .intrinsic_dim import estimate_intrinsic_dim, load_hidden_state_sample
from .retrieval import (
    HiddenStateMemory,
    build_memory_and_queries,
    on_manifold_drift,
    retrieval_evaluation,
)
from .trainer import PairedHiddenStateDataset, ProjectionTrainer

__all__ = [
    "ProjectionConfig",
    "Encoder",
    "Decoder",
    "Discriminator",
    "ProjectionAutoencoder",
    "ProjectionTrainer",
    "PairedHiddenStateDataset",
    "HiddenStateMemory",
    "build_memory_and_queries",
    "retrieval_evaluation",
    "on_manifold_drift",
    "train_identity_probe",
    "estimate_intrinsic_dim",
    "load_hidden_state_sample",
]
