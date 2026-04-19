"""Discriminator for optional adversarial on-manifold training.

Disabled by default (``ProjectionConfig.use_adversarial=False``). When
enabled, Stage C trains Ψ(Φ(h)) to land in the support of real ``h``
by adversarially scoring reconstructions against real hidden states,
using WGAN-GP with spectral normalisation on every linear layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Spectral-normalised MLP ``D: h → score``.

    The last layer is a single scalar with no activation: paired with
    WGAN-GP this is the recommended setup for stable adversarial
    training. If a probability interpretation is needed, pass through a
    sigmoid externally.
    """

    def __init__(
        self,
        d_model: int,
        hidden: int,
        n_layers: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError(f"Discriminator needs ≥ 2 layers, got {n_layers}")
        self.d_model = int(d_model)

        layers = [nn.utils.spectral_norm(nn.Linear(self.d_model, hidden))]
        for _ in range(n_layers - 2):
            layers.extend([
                nn.LeakyReLU(0.2, inplace=False),
                nn.Dropout(dropout),
                nn.utils.spectral_norm(nn.Linear(hidden, hidden)),
            ])
        layers.extend([
            nn.LeakyReLU(0.2, inplace=False),
            nn.utils.spectral_norm(nn.Linear(hidden, 1)),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)
