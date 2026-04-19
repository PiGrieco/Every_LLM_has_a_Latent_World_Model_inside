"""Orchestrator that wires Φ, Ψ, and (optionally) D into one module.

Exposes the two loss functions used by the trainer:

  - :meth:`reconstruction_loss` : ``|| h - Ψ(Φ(h)) ||^2``
  - :meth:`consistency_loss`    : local isometry ``||Φ(h_t) - Φ(h_{t+1})||
                                  ≈ α · ||h_t - h_{t+1}||`` for a learnable
                                  scalar α.

Discriminator training lives in the trainer, not here, so this module
can be used without adversarial machinery when ``use_adversarial=False``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import ProjectionConfig
from .decoder import Decoder
from .discriminator import Discriminator
from .encoder import Encoder


class ProjectionAutoencoder(nn.Module):
    """Bundle of (encoder, decoder, optional discriminator) + losses."""

    def __init__(self, cfg: ProjectionConfig, d_model: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = int(d_model)

        self.encoder = Encoder(
            d_model=self.d_model,
            latent_dim=cfg.latent_dim,
            hidden=cfg.encoder_hidden,
            use_skip=cfg.encoder_skip_proj,
            dropout=cfg.encoder_dropout,
        )
        self.decoder = Decoder(
            latent_dim=cfg.latent_dim,
            d_model=self.d_model,
            hidden=cfg.decoder_hidden,
            dropout=cfg.decoder_dropout,
        )
        if cfg.use_adversarial:
            self.discriminator: Optional[Discriminator] = Discriminator(
                d_model=self.d_model,
                hidden=cfg.discriminator_hidden,
                n_layers=cfg.discriminator_layers,
                dropout=cfg.discriminator_dropout,
            )
        else:
            self.discriminator = None

        # Learnable isometry scale α; small positive initialisation so the
        # consistency loss has a sensible start.
        self.log_alpha = nn.Parameter(
            torch.tensor(-2.3, dtype=torch.float32),  # exp(-2.3) ≈ 0.1
        )

    # ------------------------------------------------------------------
    # Core passes
    # ------------------------------------------------------------------
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return self.encoder(h)

    def decode(self, s: torch.Tensor) -> torch.Tensor:
        return self.decoder(s)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.encode(h)
        h_hat = self.decode(s)
        return s, h_hat

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def reconstruction_loss(self, h: torch.Tensor) -> torch.Tensor:
        _, h_hat = self.forward(h)
        return ((h - h_hat) ** 2).mean()

    @property
    def alpha(self) -> torch.Tensor:
        """Positive learnable scale α for the consistency loss."""
        return torch.exp(self.log_alpha)

    def consistency_loss(
        self, h_t: torch.Tensor, h_tp1: torch.Tensor,
    ) -> torch.Tensor:
        """Local isometry loss: ``(||Φ(h_t)-Φ(h_{t+1})|| - α·||h_t-h_{t+1}||)^2``.

        Interpretation: the latent map should preserve neighbouring
        distances up to a single scale α, which is learned jointly.
        Returns zero whenever ``h_t == h_tp1`` regardless of α.
        """
        z_t = self.encode(h_t)
        z_tp1 = self.encode(h_tp1)
        latent_dist = (z_t - z_tp1).norm(dim=-1)
        ambient_dist = (h_t - h_tp1).norm(dim=-1)
        diff = latent_dist - self.alpha * ambient_dist
        return (diff ** 2).mean()
