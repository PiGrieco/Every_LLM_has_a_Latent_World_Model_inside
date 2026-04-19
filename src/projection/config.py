"""Configuration for the M2 projection autoencoder (Φ / Ψ + optional D).

Consumed by:

    scripts/projection/estimate_dim.py  # exploratory PCA
    scripts/projection/train.py         # 3-stage training
    scripts/projection/evaluate.py      # retrieval + probe + drift
    scripts/projection/smoke_gate.py    # HARD gate (rigid) + DIAGNOSTICS

The design reflects the post-review corrections: adversarial training is
opt-in (``use_adversarial=False`` by default), and only two metrics are
hard gates (retrieval top-5 ratio, reconstruction MSE ratio). Identity
probe accuracy and on-manifold drift are reported as *diagnostics*
with warning/error thresholds — they do not gate M3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProjectionConfig:
    """Full projection milestone configuration.

    Defaults target Llama-3-8B hidden states (d_model=4096) on one H100.
    Tests override the relevant fields to run on synthetic data.
    """

    # -- Source data (from M1) --
    probe_data_dir: str = "./data/llm_probe"
    forward_dataset: str = "forward"
    branching_dataset: str = "branching"

    # -- Latent dim (may be overridden via intrinsic_dim.json) --
    latent_dim: int = 64
    auto_adjust_dim: bool = True
    intrinsic_dim_target_variance: float = 0.99
    intrinsic_dim_min: int = 32
    intrinsic_dim_max: int = 256
    intrinsic_dim_json: str = "./outputs/projection/intrinsic_dim.json"

    # -- Encoder architecture --
    encoder_hidden: int = 512
    encoder_skip_proj: bool = True
    encoder_dropout: float = 0.0

    # -- Decoder architecture (symmetric) --
    decoder_hidden: int = 512
    decoder_dropout: float = 0.0

    # -- Discriminator (only used when use_adversarial=True) --
    use_adversarial: bool = False          # REVISION: opt-in, default off
    discriminator_hidden: int = 256
    discriminator_layers: int = 5
    discriminator_dropout: float = 0.1

    # -- Training: 3 stages --
    # Stage A: pure reconstruction
    stage_a_epochs: int = 20
    stage_a_lr: float = 3e-4

    # Stage B: + consistency (local isometry)
    stage_b_epochs: int = 15
    stage_b_lr: float = 1e-4
    lambda_consistency: float = 0.5

    # Stage C: + adversarial (on-manifold); skipped if use_adversarial=False
    stage_c_epochs: int = 10
    stage_c_lr_g: float = 5e-5
    stage_c_lr_d: float = 2e-4
    lambda_adversarial: float = 0.1
    discriminator_steps_per_g: int = 2
    discriminator_gp_weight: float = 10.0

    # -- Dataloader --
    batch_size: int = 512
    num_workers: int = 4
    train_split_fraction: float = 0.9      # of doc_ids (document-level split)
    random_seed: int = 42

    # -- Retrieval evaluation --
    retrieval_memory_size: int = 10_000
    retrieval_queries: int = 2_000
    retrieval_topk: List[int] = field(default_factory=lambda: [1, 5, 10, 50])
    # REVISION: memory/query doc_id split (disjoint, no leakage)
    memory_query_split_fraction: float = 0.6

    # -- Identity probe --
    identity_probe_n_classes: int = 1000
    identity_probe_samples_per_class: int = 8
    identity_probe_epochs: int = 20

    # -- HARD gates (M3 promotion rides on these two only) --
    gate_min_retrieval_top5_fraction_of_baseline: float = 0.80
    gate_max_reconstruction_mse_ratio: float = 0.05   # MSE / var(h)

    # -- DIAGNOSTICS (warn-only, do NOT gate M3) --
    identity_probe_warning_threshold: float = 0.20
    identity_probe_error_threshold: float = 0.05
    on_manifold_drift_warning_threshold: float = 2.0

    # -- Storage --
    output_dir: str = "./outputs/projection"
    checkpoint_dir: str = "./checkpoints/projection"
    device: str = "cuda"
    dtype: str = "float32"
