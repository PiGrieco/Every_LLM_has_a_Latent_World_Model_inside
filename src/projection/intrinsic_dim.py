"""Exploratory intrinsic-dimensionality estimate on M1 hidden states.

Runs before training so we can override ``latent_dim`` from the default
64 if the data asks for a larger (or smaller) bottleneck. Target is
the smallest number of principal components that captures
``target_variance`` of the sample variance, clamped to
``[dim_min, dim_max]``.
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def load_hidden_state_sample(
    data_dir: str,
    dataset: str,
    n_sample: int,
    seed: int = 0,
) -> torch.Tensor:
    """Sample ``n_sample`` hidden-state rows from an M1 dataset.

    Args:
        data_dir: Root dir (expects a ``{dataset}/manifest.json`` inside).
        dataset: One of ``"forward" | "branching" | "reversed"``.
        n_sample: Target sample size. If fewer rows exist, returns what
            is available.
        seed: Python ``random`` seed used for trajectory + window
            resampling.

    Returns:
        Float32 CPU tensor of shape ``(<=n_sample, d_model)``.
    """
    from src.llm_probe import TrajectoryShardReader

    reader = TrajectoryShardReader(f"{data_dir}/{dataset}")
    rng = random.Random(seed)
    collected: List[torch.Tensor] = []
    all_items: List[torch.Tensor] = []

    # Reservoir-style collection over windows — stream once.
    total_rows = 0
    for item in reader.iter_items():
        hs = item.get("hidden_states")
        # Branching items keep hs under trajectory_a/b; take trajectory_a.
        if hs is None and isinstance(item.get("trajectory_a"), dict):
            hs = item["trajectory_a"].get("hidden_states")
        if hs is None:
            continue
        # Reservoir sample per-row.
        for row in hs:
            if total_rows < n_sample:
                all_items.append(row.float())
            else:
                j = rng.randint(0, total_rows)
                if j < n_sample:
                    all_items[j] = row.float()
            total_rows += 1

    if not all_items:
        raise RuntimeError(f"No hidden states found under {data_dir}/{dataset}")

    return torch.stack(all_items, dim=0).contiguous()


def estimate_intrinsic_dim(
    hidden_states_sample: torch.Tensor,
    target_variance: float = 0.99,
    dim_min: int = 32,
    dim_max: int = 256,
) -> dict:
    """PCA-based intrinsic dimension estimate.

    Picks the smallest ``d`` such that the first ``d`` principal
    components cumulatively explain at least ``target_variance`` of
    the centred sample variance, then clamps to ``[dim_min, dim_max]``.

    Args:
        hidden_states_sample: ``(N, d_model)`` tensor. Must have N > 2.
        target_variance: fraction of variance to capture (e.g. 0.99).
        dim_min / dim_max: output clamp range.

    Returns:
        dict with:
          - ``intrinsic_dim``: int, recommended latent_dim
          - ``target_variance``: float (echoed back)
          - ``actual_variance_at_dim``: float, variance explained at the
            reported dim
          - ``cumulative_variance_curve``: list of the first 256 cumulative
            variance ratios (truncated to min(d_model, 256))
          - ``clamped``: bool, True iff the raw estimate was outside
            ``[dim_min, dim_max]``
    """
    if hidden_states_sample.ndim != 2:
        raise ValueError(
            f"Expected (N, d_model); got shape {tuple(hidden_states_sample.shape)}"
        )
    X = hidden_states_sample.float().cpu()
    N, D = X.shape
    if N <= 2:
        raise ValueError(f"Need N > 2 rows for PCA, got {N}")

    # Prefer sklearn for explained_variance_ratio_; fall back to torch.
    try:
        from sklearn.decomposition import PCA
        n_components = min(N, D, max(dim_max + 16, 256))
        pca = PCA(n_components=n_components, svd_solver="auto")
        pca.fit(X.numpy())
        ratios = pca.explained_variance_ratio_
    except Exception as exc:  # pragma: no cover — optional fallback
        logger.warning("sklearn PCA unavailable (%s); using torch fallback", exc)
        Xc = X - X.mean(dim=0, keepdim=True)
        _, S, _ = torch.linalg.svd(Xc, full_matrices=False)
        var = (S ** 2) / max(1, (N - 1))
        ratios = (var / var.sum()).numpy()

    import numpy as np
    cum = np.cumsum(ratios)
    # argmax returns the first index where the boolean is True.
    idx = int(np.argmax(cum >= target_variance))
    if cum[idx] < target_variance:
        # target_variance never reached — use the last index.
        idx = len(cum) - 1
    raw_dim = idx + 1

    clamped_dim = max(dim_min, min(dim_max, raw_dim))
    clamped = (clamped_dim != raw_dim)

    return {
        "intrinsic_dim": int(clamped_dim),
        "raw_intrinsic_dim": int(raw_dim),
        "target_variance": float(target_variance),
        "actual_variance_at_dim": float(cum[min(clamped_dim - 1, len(cum) - 1)]),
        "cumulative_variance_curve": [float(x) for x in cum[:256]],
        "clamped": bool(clamped),
        "n_samples": int(N),
        "d_model": int(D),
    }
