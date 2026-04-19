"""Ψ decoder: latent state → hidden state reconstruction.

Structurally symmetric to the encoder. Note: this is NOT a generative
decoder over tokens — the M3 milestone introduces a suffix decoder
through the remaining LLM layers. In M2 the decoder's purpose is
reconstruction and retrieval readout only.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Residual gated MLP: ``s ∈ ℝ^D → ĥ ∈ ℝ^{d_model}``."""

    def __init__(
        self,
        latent_dim: int,
        d_model: int,
        hidden: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.d_model = int(d_model)

        self.fc1 = nn.Linear(self.latent_dim, hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden, self.d_model)
        self.skip_proj = nn.Linear(self.latent_dim, self.d_model, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.dropout(self.act(self.fc1(s))))
        return h + self.skip_proj(s)
