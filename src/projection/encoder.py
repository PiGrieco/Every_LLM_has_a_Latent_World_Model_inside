"""Φ encoder: hidden state → latent state."""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Residual gated MLP: ``h ∈ ℝ^{d_model} → s ∈ ℝ^D``.

    The skip path is a zero-bias ``Linear(d_model, D)`` projection added
    to the two-layer non-linear branch, followed by ``LayerNorm(D)``.
    The LayerNorm at the end matters: without it, the bottleneck is
    sensitive to the absolute scale of the LM hidden states and the
    downstream metric is numerically fragile.
    """

    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        hidden: int,
        use_skip: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)
        self.use_skip = bool(use_skip)

        self.fc1 = nn.Linear(self.d_model, hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden, self.latent_dim)
        self.skip_proj = (
            nn.Linear(self.d_model, self.latent_dim, bias=False)
            if self.use_skip else None
        )
        self.norm = nn.LayerNorm(self.latent_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.fc2(self.dropout(self.act(self.fc1(h))))
        if self.skip_proj is not None:
            z = z + self.skip_proj(h)
        return self.norm(z)
