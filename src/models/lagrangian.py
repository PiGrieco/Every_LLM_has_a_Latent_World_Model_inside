"""
Geometric–Semantic Lagrangian L_θ(s_t, s_{t+1}).

The Lagrangian is the core energy function of the framework:

    L_θ(s_t, s_{t+1}) = λ_g · L_g(s_t, s_{t+1}) + λ_sem · L_sem(s_t, s_{t+1})

where:
    L_g = ½ Δs^T g_θ(s_t) Δs     (geometric kinetic energy)
    L_sem = -log P_LM(d_{t+1}|d_{≤t})  (semantic surprise)

The Gibbs kernel is then:
    K_θ(s_{t+1} | s_t) ∝ exp(-L_θ(s_t, s_{t+1}))

Low Lagrangian = smooth geometry + plausible semantics = high probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .metric import MetricNetwork


class Lagrangian(nn.Module):
    """
    Full Lagrangian combining geometric and semantic terms.

    The geometric term comes from the learned Lorentzian metric.
    The semantic term is either:
    - Precomputed LM log-probs (Case A, real data)
    - A learned surrogate U_θ(s_t, s_{t+1}) (distilled or corpus-only)
    - Zero (for synthetic data where there's no semantic signal)
    """

    def __init__(
        self,
        metric: MetricNetwork,
        lambda_g: float = 1.0,
        lambda_sem: float = 1.0,
        use_semantic_surrogate: bool = False,
        latent_dim: int = 16,
        surrogate_hidden: int = 128,
        interval_clamp_min: float = -10.0,
    ):
        super().__init__()
        self.metric = metric
        self.lambda_g = lambda_g
        self.lambda_sem = lambda_sem
        self.use_semantic_surrogate = use_semantic_surrogate
        self.interval_clamp_min = interval_clamp_min
        self.temperature = 1.0
        self.time_fn = None
        self.lambda_future = 0.0

        # Optional: learned surrogate for semantic cost
        # Useful when LM queries are expensive or in corpus-only setting
        if use_semantic_surrogate:
            self.surrogate = nn.Sequential(
                nn.Linear(latent_dim * 2, surrogate_hidden),
                nn.GELU(),
                nn.Linear(surrogate_hidden, surrogate_hidden),
                nn.GELU(),
                nn.Linear(surrogate_hidden, 1),
                nn.Softplus(),  # Semantic cost is non-negative
            )

    def geometric_term(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """
        L_g(s_t, s_{t+1}) = ½ Δs^T g_θ(s_t) Δs

        Args:
            s: (batch, D) current states
            s_next: (batch, D) next states

        Returns:
            L_g: (batch,) geometric Lagrangian values
        """
        delta_s = s_next - s
        interval = self.metric.squared_interval(s, delta_s)
        # Clamp to prevent unbounded negative energy.
        # Without this, strongly time-like transitions produce arbitrarily
        # negative L_g, causing exp(-L_g) overflow in the Gibbs kernel and
        # incentivizing the metric to collapse all transitions to maximally
        # time-like. The clamp at -10 preserves the time-likeness training
        # signal (which only needs interval < 0) while bounding the reward.
        interval_clamped = interval.clamp(min=self.interval_clamp_min)
        return 0.5 * interval_clamped

    def semantic_term(
        self,
        s: torch.Tensor,
        s_next: torch.Tensor,
        precomputed_lsem: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        L_sem(s_t, s_{t+1}).

        Either uses precomputed values (from LM log-probs) or the learned
        surrogate network.

        Args:
            s, s_next: (batch, D) states
            precomputed_lsem: (batch,) optional precomputed -log P_LM values

        Returns:
            L_sem: (batch,) semantic Lagrangian values
        """
        if precomputed_lsem is not None:
            return precomputed_lsem

        if self.use_semantic_surrogate:
            # Surrogate takes concatenated (s, s_next) and predicts cost
            concat = torch.cat([s, s_next], dim=-1)
            return self.surrogate(concat).squeeze(-1)

        # No semantic term available (synthetic data case)
        return torch.zeros(s.shape[0], device=s.device)

    def forward(
        self,
        s: torch.Tensor,
        s_next: torch.Tensor,
        precomputed_lsem: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Full Lagrangian: L_θ = λ_g · L_g + λ_sem · L_sem

        Args:
            s: (batch, D) current states
            s_next: (batch, D) next states
            precomputed_lsem: (batch,) optional precomputed semantic costs

        Returns:
            L: (batch,) total Lagrangian values
        """
        L_g = self.geometric_term(s, s_next)
        L_sem = self.semantic_term(s, s_next, precomputed_lsem)

        total = self.lambda_g * L_g + self.lambda_sem * L_sem

        if self.time_fn is not None:
            d_tau = self.time_fn.delta_tau(s, s_next)
            L_future = F.softplus(0.1 - d_tau)
            total = total + self.lambda_future * L_future

        return total

    def action(
        self,
        trajectory: torch.Tensor,
        precomputed_lsem: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute the discrete action S_θ[γ] = Σ L_θ(s_t, s_{t+1}).

        Args:
            trajectory: (T, D) tensor of states along a worldline
            precomputed_lsem: (T-1,) optional semantic costs per transition

        Returns:
            S: scalar, total action along the trajectory
        """
        s = trajectory[:-1]      # (T-1, D)
        s_next = trajectory[1:]  # (T-1, D)
        L = self.forward(s, s_next, precomputed_lsem)
        return L.sum()

    def set_temperature(self, T: float):
        """Set Gibbs temperature for candidate-set matching."""
        self.temperature = max(T, 0.1)

    def set_time_orientation(self, time_fn, lambda_future: float = 0.5):
        """Attach a time orientation function to the Lagrangian."""
        self.time_fn = time_fn
        self.lambda_future = lambda_future
