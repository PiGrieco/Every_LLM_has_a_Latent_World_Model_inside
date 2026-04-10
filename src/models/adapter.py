"""
Geometry Adapter A_θ: maps preprocessed embeddings z_hat ∈ R^{D_0}
to latent narrative states s ∈ R^D on the manifold M.

This is a small MLP with optional spectral normalization to prevent
pathological warping of the latent space.

For synthetic data (D0/D1), the adapter is identity (data is already
in the target dimension). For D2, it compresses from encoder_dim (384)
to latent_dim (16).
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class GeometryAdapter(nn.Module):
    """
    MLP: R^{D_0} → R^D with optional spectral normalization.

    The adapter learns to compress high-dimensional encoder outputs
    into a low-dimensional latent space where the Lorentzian metric
    will be defined. Spectral norm constraints encourage the mapping
    to be Lipschitz-continuous, preventing the manifold from being
    stretched or folded pathologically.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        layers = []
        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]

        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            if use_spectral_norm:
                linear = spectral_norm(linear)
            layers.append(linear)

            # ReLU activation for all but the last layer
            if i < len(dims) - 2:
                layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_hat: (batch, D_0) preprocessed embeddings

        Returns:
            s: (batch, D) latent narrative states
        """
        return self.net(z_hat)


class IdentityAdapter(nn.Module):
    """
    Identity adapter for synthetic data where input_dim == output_dim.
    Useful for D0/D1 where we work directly in the target space.
    """

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        return z_hat
