"""
Preprocessing P: centering, top-PC removal, normalization.

Follows the "all-but-the-top" strategy from Mu & Viswanath (2018)
to fix anisotropy in pretrained embeddings before the geometry adapter.

This is applied as a non-parametric transform BEFORE the learned adapter.
For synthetic data (D0/D1), preprocessing is identity (data is already clean).
"""

import torch
import numpy as np
from typing import Optional, Tuple


class EmbeddingPreprocessor:
    """
    Non-parametric preprocessing pipeline:
    1. Center (subtract mean)
    2. Remove top-k principal components (anisotropy fix)
    3. Optionally normalize to unit norm

    Fitted on training data, then applied to train/val/test.
    """

    def __init__(self, n_pca_remove: int = 3, normalize: bool = True):
        self.n_pca_remove = n_pca_remove
        self.normalize = normalize
        # Fitted parameters
        self.mean: Optional[torch.Tensor] = None
        self.top_components: Optional[torch.Tensor] = None
        self._fitted = False

    def fit(self, embeddings: torch.Tensor):
        """
        Fit the preprocessor on a matrix of embeddings.

        Args:
            embeddings: (N, D) tensor of raw encoder outputs
        """
        # 1. Compute mean
        self.mean = embeddings.mean(dim=0)  # (D,)

        # 2. Compute top-k PCs to remove
        centered = embeddings - self.mean
        if self.n_pca_remove > 0:
            # SVD on centered data: U @ diag(S) @ V^T = centered
            # Top PCs are the first k columns of V
            _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
            self.top_components = Vt[: self.n_pca_remove]  # (k, D)
        else:
            self.top_components = None

        self._fitted = True

    def transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply the fitted preprocessing to embeddings.

        Args:
            embeddings: (N, D) or (T, D) tensor

        Returns:
            preprocessed: same shape, with centering + PC removal + normalization
        """
        assert self._fitted, "Call fit() before transform()"

        # 1. Center
        z = embeddings - self.mean

        # 2. Remove top PCs: z -= sum_i (z @ v_i) * v_i
        if self.top_components is not None:
            # Project onto top components and subtract
            # projections: (N, k) = (N, D) @ (D, k)
            projections = z @ self.top_components.T  # (N, k)
            z = z - projections @ self.top_components  # (N, D)

        # 3. Normalize
        if self.normalize:
            norms = z.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            z = z / norms

        return z

    def fit_transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step."""
        self.fit(embeddings)
        return self.transform(embeddings)


def preprocess_trajectory_dataset(
    embeddings_list: list,
    preprocessor: Optional[EmbeddingPreprocessor] = None,
    n_pca_remove: int = 3,
    normalize: bool = True,
) -> Tuple[list, EmbeddingPreprocessor]:
    """
    Preprocess a list of per-trajectory embedding tensors.

    If preprocessor is None, fits a new one on the concatenated embeddings.

    Args:
        embeddings_list: list of (T_i, D) tensors
        preprocessor: optional pre-fitted preprocessor (for val/test)

    Returns:
        processed_list: list of (T_i, D) preprocessed tensors
        preprocessor: the fitted preprocessor
    """
    # Concatenate all embeddings for fitting
    if preprocessor is None:
        all_emb = torch.cat(embeddings_list, dim=0)
        preprocessor = EmbeddingPreprocessor(
            n_pca_remove=n_pca_remove, normalize=normalize
        )
        preprocessor.fit(all_emb)

    # Transform each trajectory
    processed = [preprocessor.transform(emb) for emb in embeddings_list]

    return processed, preprocessor
