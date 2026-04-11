"""
Configuration for the Lorentzian World Model framework.

All hyperparameters live here as a single dataclass. We use YAML files
to override defaults per-dataset (see configs/).
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import yaml


@dataclass
class Config:
    # ---- Experiment identity ----
    experiment_name: str = "lorentzian_wm"
    seed: int = 42
    device: str = "cuda"

    # ---- Dataset ----
    dataset: Literal["d0_synthetic", "d1_branching", "d2_wikitext"] = "d0_synthetic"
    # D0/D1 synthetic params
    n_trajectories: int = 1000
    trajectory_length: int = 50
    drift_strength: float = 0.5
    noise_std: float = 0.1
    # D1 branching params
    prefix_length: int = 10
    n_branches: int = 4
    branch_length: int = 20
    # D2 params
    min_paragraphs: int = 5
    max_paragraphs: int = 100
    wikitext_split: str = "train"

    # ---- Encoder (D2 only) ----
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_dim: int = 384           # Output dim of the encoder
    encode_batch_size: int = 256

    # ---- LM for semantic scores (D2 Case A) ----
    lm_name: str = "gpt2-medium"
    lm_batch_size: int = 16
    lm_max_context: int = 512        # Max tokens of context for log-prob computation

    # ---- Preprocessing ----
    n_pca_remove: int = 3            # Number of top PCs to remove (anisotropy fix)
    normalize_embeddings: bool = True

    # ---- Architecture ----
    latent_dim: int = 16             # D: dimension of the narrative manifold
    adapter_hidden: int = 256        # Hidden dim for the geometry adapter MLP
    adapter_layers: int = 2
    metric_hidden: int = 128         # Hidden dim for the metric network
    metric_layers: int = 2
    wm_hidden: int = 256             # Hidden dim for world model q_θ
    wm_layers: int = 3
    wm_type: Literal["gaussian", "flow"] = "gaussian"

    # ---- Segmentation (Algorithm 1) ----
    use_segmentation: bool = False   # Start without segmentation for D0/D1
    seg_window: int = 3              # Smoothing window w
    seg_l_min: int = 3
    seg_l_max: int = 50
    # Thresholds are set adaptively as multiples of corpus median
    seg_tau_v_mult: float = 1.5
    seg_tau_kappa_mult: float = 1.5
    seg_tau_v_low_mult: float = 0.5
    seg_tau_kappa_low_mult: float = 0.5

    # ---- Candidate sets ----
    candidate_strategy: Literal["c1", "c1c2", "c1c2c3"] = "c1"
    candidate_set_size: int = 32     # Total candidates per state
    knn_rebuild_every: int = 10      # Epochs between kNN index rebuilds

    # ---- Stabilization ----
    interval_clamp_min: float = -10.0   # Min value for squared interval (prevents unbounded negative energy)
    gibbs_temperature: float = 1.0      # Temperature T in the Gibbs kernel (>1 softens, prevents saturation)
    auto_calibrate_weights: bool = True # Auto-calibrate λ at stage transitions via loss-magnitude matching

    # ---- Loss weights ----
    # Stage 1 (representation, only for VAE variant)
    beta_kl: float = 1.0
    beta_kl_anneal_epochs: int = 10
    # Stage 2 (+ causality)
    lambda_causal: float = 1.0
    causal_margin: float = -0.1      # ε in the time-likeness loss (negative margin)
    cone_margin_outside: float = 0.1  # Margin for space-like constraint on negatives
    cone_target_scale: float = 1.0    # Target for E[|Δσ²|] — prevents cone collapse/explosion
    cone_scale_weight: float = 0.5    # Weight of scale regularizer in cone loss
    cone_target_rate: float = 0.8     # Target fraction of time-like transitions
    cone_rate_weight: float = 1.0     # Weight of rate regularizer in cone loss
    lambda_future: float = 0.5        # Weight for time orientation loss L_future
    future_margin: float = 0.1        # δ: minimum required τ increment per step
    # Stage 3 (+ world model matching)
    lambda_match: float = 1.0
    lambda_smooth: float = 0.1
    lambda_triplet: float = 0.0      # Enable if semantic triplets are available
    lambda_wm: float = 1.0
    lambda_geo: float = 0.5          # Overall weight for Lgeo = Ltime + Lsmooth + Ltriplet
    lambda_g: float = 1.0            # Weight of geometric term in Lagrangian
    lambda_sem: float = 1.0          # Weight of semantic term in Lagrangian
    lambda_sem_reg: float = 0.1      # Weight for surrogate regression loss (D2)

    # ---- Training ----
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    # Staged training: how many epochs per stage
    stage1_epochs: int = 0           # 0 = skip stage 1 (no VAE for D0/D1)
    stage2_epochs: int = 50          # Metric + adapter + time-likeness
    stage3_epochs: int = 100         # + world model matching
    grad_clip: float = 1.0

    # ---- Evaluation ----
    eval_every: int = 10             # Evaluate M1-M6 every N epochs
    n_eval_trajectories: int = 200   # Trajectories for evaluation

    # ---- Geometry baselines ----
    geometry: Literal["lorentzian", "riemannian", "euclidean"] = "lorentzian"

    # ---- Paths ----
    cache_dir: str = "./cache"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file, overriding defaults."""
        with open(path, "r") as f:
            overrides = yaml.safe_load(f)
        return cls(**{k: v for k, v in overrides.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str):
        """Save current config to YAML."""
        import dataclasses
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)
