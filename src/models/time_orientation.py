"""
Time Orientation τ_θ(s): a learned scalar function that assigns a
"narrative time" to each latent state, breaking time-reversal symmetry.

In Lorentzian geometry, the metric gives you cones (time-like vs space-like)
but not a preferred direction within the cone. You need a separate structure
— a time orientation — to distinguish "future" from "past". In physics this
is a global topological choice; in our framework, we learn it from data.

τ_θ: M → R is a small MLP that should satisfy τ_θ(s_{t+1}) > τ_θ(s_t)
for all observed forward transitions. This makes the Lagrangian asymmetric:

    L_future(s_t, s_{t+1}) = Softplus(δ - (τ(s_{t+1}) - τ(s_t)))

Forward transitions: Δτ > δ → L_future ≈ 0  (low cost)
Reversed transitions: Δτ < 0 → L_future >> 0  (high cost)

This term is:
  - Non-negative (no unbounded negative energy problem)
  - Smooth and differentiable everywhere
  - Explicitly asymmetric under time reversal

When added to the Lagrangian, the Gibbs kernel exp(-L_θ) genuinely prefers
forward over backward transitions, making M2(action gap) a meaningful probe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeOrientation(nn.Module):
    """
    Learned time function τ_θ: R^D → R.

    A small MLP that maps latent states to a scalar "narrative time".
    Trained so that τ increases along forward trajectories.

    The output is unconstrained (can be any real number), and the
    training signal comes from the future loss which penalizes
    non-increasing τ along observed transitions.
    """

    def __init__(self, dim: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()

        layers = []
        dims = [dim] + [hidden_dim] * n_layers + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

        # Initialize to produce small outputs near zero.
        # This means Δτ starts near zero and the loss is initially
        # non-zero but not huge, giving stable early gradients.
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute τ_θ(s) for a batch of states.

        Args:
            s: (batch, D) latent states

        Returns:
            tau: (batch,) scalar time values
        """
        return self.net(s).squeeze(-1)

    def delta_tau(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """
        Compute Δτ = τ(s_{t+1}) - τ(s_t).

        Positive Δτ = forward in narrative time.
        Negative Δτ = backward (time-reversed).

        Args:
            s: (batch, D) current states
            s_next: (batch, D) next states

        Returns:
            d_tau: (batch,) time increments
        """
        return self.forward(s_next) - self.forward(s)


def future_loss(
    time_fn: TimeOrientation,
    s: torch.Tensor,
    s_next: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    L_future = E[Softplus(δ - Δτ)]

    Penalizes transitions where τ doesn't increase by at least δ.
    Softplus is smooth and always non-negative, avoiding the
    unbounded-below energy problem.

    For forward transitions: Δτ >> δ → Softplus(δ - Δτ) ≈ 0
    For reversed transitions: Δτ << 0 → Softplus(δ - Δτ) >> 0

    Args:
        time_fn: TimeOrientation module
        s: (batch, D) current states
        s_next: (batch, D) next states
        margin: minimum required τ increment (δ)

    Returns:
        loss: scalar, average future loss
    """
    d_tau = time_fn.delta_tau(s, s_next)  # (batch,)
    # Softplus(δ - Δτ) = log(1 + exp(δ - Δτ))
    loss = F.softplus(margin - d_tau).mean()
    return loss
