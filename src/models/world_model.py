"""
Parametric World Model q_θ(s_{t+1} | s_t).

This module implements the explicit, learnable transition model that
approximates the Gibbs kernel K_θ. For v1, we use a conditional
Gaussian: given s_t, the model predicts the mean and diagonal
covariance of s_{t+1}.

    q_θ(s_{t+1} | s_t) = N(s_{t+1}; μ_θ(s_t), diag(σ²_θ(s_t)))

The model is trained with two objectives:
1. Maximum likelihood: minimize -log q_θ(s_{t+1} | s_t) on real transitions
2. Candidate-set matching: minimize KL(q̃_θ || K̃_θ) on candidate sets

Under sufficient expressivity, q_θ ≈ K_θ, so the parametric world model
realizes the geometric–semantic Gibbs dynamics.
"""

import torch
import torch.nn as nn
import math


class ConditionalGaussianWorldModel(nn.Module):
    """
    Conditional Gaussian world model: q_θ(s' | s) = N(μ(s), σ²(s)).

    An MLP takes s and predicts the mean and log-variance of the next
    state distribution. The log-variance is clamped for stability.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        min_logvar: float = -10.0,
        max_logvar: float = 2.0,
    ):
        super().__init__()
        self.dim = dim
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        # Shared trunk
        trunk_layers = []
        trunk_dims = [dim] + [hidden_dim] * n_layers
        for i in range(len(trunk_dims) - 1):
            trunk_layers.append(nn.Linear(trunk_dims[i], trunk_dims[i + 1]))
            trunk_layers.append(nn.GELU())
        self.trunk = nn.Sequential(*trunk_layers)

        # Separate heads for mean and log-variance
        self.mean_head = nn.Linear(hidden_dim, dim)
        self.logvar_head = nn.Linear(hidden_dim, dim)

        # Initialize mean head to predict identity + small offset
        # (s_{t+1} ≈ s_t initially, which is a reasonable prior)
        with torch.no_grad():
            self.mean_head.weight.zero_()
            self.mean_head.bias.zero_()

    def forward(self, s: torch.Tensor):
        """
        Predict distribution parameters for next state.

        Args:
            s: (batch, D) current states

        Returns:
            mean: (batch, D) predicted mean of s_{t+1}
            logvar: (batch, D) predicted log-variance of s_{t+1}
        """
        h = self.trunk(s)
        mean = s + self.mean_head(h)  # Residual: predict offset from current state
        logvar = self.logvar_head(h)
        logvar = logvar.clamp(self.min_logvar, self.max_logvar)
        return mean, logvar

    def log_prob(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """
        Compute log q_θ(s_{t+1} | s_t) under the Gaussian model.

        Args:
            s: (batch, D) current states
            s_next: (batch, D) next states

        Returns:
            log_q: (batch,) log-probabilities
        """
        mean, logvar = self.forward(s)
        var = logvar.exp()

        # Gaussian log-probability (diagonal covariance)
        # log N(x; μ, σ²) = -½ [D log(2π) + Σ log(σ²_i) + Σ (x_i - μ_i)² / σ²_i]
        diff = s_next - mean
        log_q = -0.5 * (
            self.dim * math.log(2 * math.pi)
            + logvar.sum(dim=-1)
            + (diff ** 2 / var).sum(dim=-1)
        )
        return log_q

    def neg_log_prob(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """Convenience: negative log-probability (the NLL loss)."""
        return -self.log_prob(s, s_next)

    def sample(self, s: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Sample from q_θ(s' | s).

        Args:
            s: (batch, D) current states
            n_samples: number of samples per state

        Returns:
            samples: (batch, n_samples, D) sampled next states
        """
        mean, logvar = self.forward(s)
        std = (0.5 * logvar).exp()

        # Expand for multiple samples
        mean = mean.unsqueeze(1).expand(-1, n_samples, -1)
        std = std.unsqueeze(1).expand(-1, n_samples, -1)

        eps = torch.randn_like(mean)
        return mean + std * eps

    def log_prob_candidates(
        self, s: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log q_θ(s^(i) | s) for a set of candidates.

        This is used for candidate-set KL matching: we need the log-density
        of the world model evaluated at each candidate, not just the true
        next state.

        Args:
            s: (batch, D) current states
            candidates: (batch, C, D) candidate next states

        Returns:
            log_q: (batch, C) log-probabilities for each candidate
        """
        batch, n_cand, dim = candidates.shape
        mean, logvar = self.forward(s)  # (batch, D)
        var = logvar.exp()

        # Expand mean and var to match candidates
        mean = mean.unsqueeze(1).expand(-1, n_cand, -1)      # (batch, C, D)
        var = var.unsqueeze(1).expand(-1, n_cand, -1)         # (batch, C, D)
        logvar_exp = logvar.unsqueeze(1).expand(-1, n_cand, -1)

        diff = candidates - mean
        log_q = -0.5 * (
            dim * math.log(2 * math.pi)
            + logvar_exp.sum(dim=-1)
            + (diff ** 2 / var).sum(dim=-1)
        )
        return log_q
