"""
Time Orientation τ_θ: derives the arrow of time from the metric's
local Minkowski frame.

KEY DESIGN: τ is NOT an independent MLP. For Lorentzian geometry,
the time increment is the zeroth component of the displacement
in the local Minkowski frame defined by A_met(s):

    Δτ(s, s') = (A_met(s) · (s' - s))_0

This works because the metric is parameterized as g = A^T η A
where η = diag(-1, 1, ..., 1). The matrix A_met acts as a vielbein
(local frame), and the zeroth coordinate in that frame is the one
with the minus sign in η — i.e., the time direction.

Advantages over eigendecomposition:
  - No O(D³) eigenvalue computation per sample
  - Numerically stable (no eigenvalue crossings)
  - Differentiable through A_met (already in the computation graph)
  - Conceptually natural (uses the frame, not a derived quantity)

For Riemannian/Euclidean baselines: no time-like direction exists,
so we fall back to a learned MLP (harder to train, no geometric anchor).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeOrientation(nn.Module):
    """
    Geometry-coupled time orientation via A_met's local frame.

    For Lorentzian: Δτ = (A_met(s) · Δs)_0 (zeroth Minkowski component)
    For others: Δτ = MLP(s_{t+1}) - MLP(s_t) (fallback, no anchor)
    """

    def __init__(self, dim: int, geometry: str = "lorentzian",
                 hidden_dim: int = 64, n_layers: int = 2,
                 mode: str = "auto"):
        super().__init__()
        self.dim = dim
        self.geometry = geometry

        if mode == "auto":
            self.mode = "frame" if geometry == "lorentzian" else "mlp"
        else:
            self.mode = mode

        if self.mode == "mlp":
            # Fallback MLP for non-Lorentzian geometries.
            # Has NO geometric anchor — must learn directionality
            # purely from data, which is harder and noisier.
            layers = []
            dims = [dim] + [hidden_dim] * n_layers + [1]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.GELU())
            self.fallback_mlp = nn.Sequential(*layers)
            with torch.no_grad():
                self.fallback_mlp[-1].weight.mul_(0.01)
                self.fallback_mlp[-1].bias.zero_()

    def _get_time_component(self, metric, s: torch.Tensor,
                            delta_s: torch.Tensor) -> torch.Tensor:
        """
        Extract the time component of a displacement in the local
        Minkowski frame defined by A_met(s).

        Δτ = (A_met(s) · Δs)_0

        This is the natural "clock reading" for the displacement:
        how much it advances along the direction that carries the
        minus sign in η.

        Args:
            metric: MetricNetwork (must have _get_a_met method)
            s: (batch, D) current states
            delta_s: (batch, D) displacement vectors

        Returns:
            dt: (batch,) time components
        """
        A_met = metric._get_a_met(s)  # (batch, D, D)
        # A_met @ Δs gives the displacement in the local Minkowski frame
        # Shape: (batch, D, 1) -> squeeze to (batch, D)
        local_displacement = torch.bmm(A_met, delta_s.unsqueeze(-1)).squeeze(-1)
        # The zeroth component is the time direction (the one with -1 in η)
        return local_displacement[:, 0]

    def delta_tau(self, metric, s: torch.Tensor,
                  s_next: torch.Tensor) -> torch.Tensor:
        """
        Compute Δτ = time increment between consecutive states.

        For Lorentzian: projection of displacement onto the metric's
        time-like direction via A_met. This is FREE (no extra parameters,
        no eigendecomposition) — it reuses the metric's own structure.

        For others: difference of MLP outputs (must learn from scratch).

        Args:
            metric: MetricNetwork
            s: (batch, D) current states
            s_next: (batch, D) next states

        Returns:
            d_tau: (batch,) time increments
        """
        if self.mode == "frame":
            delta_s = s_next - s
            return self._get_time_component(metric, s, delta_s)
        else:
            tau_s = self.fallback_mlp(s).squeeze(-1)
            tau_sn = self.fallback_mlp(s_next).squeeze(-1)
            return tau_sn - tau_s

    def forward(self, metric, s: torch.Tensor) -> torch.Tensor:
        """
        Compute τ(s) — narrative time at state s.

        For frame mode: τ(s) = (A_met(s) · s)_0
        For MLP mode: τ(s) = MLP(s)
        """
        if self.mode == "frame":
            return self._get_time_component(metric, s, s)
        else:
            return self.fallback_mlp(s).squeeze(-1)


def future_loss(
    time_fn: TimeOrientation,
    metric,
    s: torch.Tensor,
    s_next: torch.Tensor,
    margin: float = 0.1,
    target_delta: float = 1.0,
    band_weight: float = 1.0,
) -> torch.Tensor:
    """
    Band-pass future loss with metric coupling.

    1. Forward penalty: Softplus(margin - Δτ) — must advance in time
    2. Band-pass: (Δτ - target_delta)² — normalize scale per-sample

    For Lorentzian, Δτ comes from A_met (geometric, no extra params).
    For baselines, Δτ comes from fallback MLP (must learn from scratch).
    """
    d_tau = time_fn.delta_tau(metric, s, s_next)

    loss_forward = F.softplus(margin - d_tau).mean()
    loss_band = ((d_tau - target_delta) ** 2).mean()

    return loss_forward + band_weight * loss_band
