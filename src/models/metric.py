"""
Lorentzian Metric g_θ on the narrative manifold M.

The metric is parameterized as:
    g_θ(s) = A^met_θ(s)^T  η  A^met_θ(s)

where η = diag(-1, +1, ..., +1) is the Minkowski signature.

This construction guarantees Lorentzian signature by design:
- The product A^T η A always has exactly one negative eigenvalue
  (corresponding to the time-like direction) and D-1 positive
  eigenvalues (spatial/semantic directions).

For geometry baselines:
- Riemannian: replace η with I (all positive eigenvalues)
- Euclidean: fix g(s) = I (no learned metric)

Key outputs:
- Squared interval: Δs^T g_θ(s) Δs  (negative = time-like, positive = space-like)
- Spatial metric: h_θ(s) = A^T Π A  where Π = diag(0, 1, ..., 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricNetwork(nn.Module):
    """
    Neural network that produces A^met_θ(s) ∈ R^{D×D} for each state s.

    The metric at state s is then:
        g_θ(s) = A^met_θ(s)^T  η  A^met_θ(s)

    A small MLP maps s ∈ R^D to a flattened D×D matrix, which is
    reshaped into A^met_θ(s). We add a residual connection to the
    identity to ensure the metric starts close to flat Minkowski space,
    which helps early training stability.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        geometry: str = "lorentzian",
    ):
        """
        Args:
            dim: latent dimension D
            hidden_dim: hidden layer width
            n_layers: number of hidden layers
            geometry: "lorentzian", "riemannian", or "euclidean"
        """
        super().__init__()
        self.dim = dim
        self.geometry = geometry

        if geometry == "euclidean":
            # No parameters needed - metric is fixed identity
            return

        # Build the MLP that produces A^met(s)
        layers = []
        dims = [dim] + [hidden_dim] * n_layers + [dim * dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)

        # Initialize to produce near-identity A^met, so initial metric ≈ η
        # The last layer should output near-zero (then we add identity)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

        # Signature matrix η
        # Lorentzian: diag(-1, 1, 1, ..., 1)
        # Riemannian: diag(1, 1, 1, ..., 1)
        if geometry == "lorentzian":
            eta = torch.ones(dim)
            eta[0] = -1.0
        else:  # riemannian
            eta = torch.ones(dim)

        self.register_buffer("eta", torch.diag(eta))

        # Spatial projector Π = diag(0, 1, 1, ..., 1)
        # Used for the spatial metric h_θ(s) = A^T Π A
        pi = torch.ones(dim)
        if geometry == "lorentzian":
            pi[0] = 0.0
        self.register_buffer("pi", torch.diag(pi))

    def _get_a_met(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute A^met_θ(s) for a batch of states.

        Args:
            s: (batch, D) latent states

        Returns:
            A_met: (batch, D, D) matrix-valued map
        """
        batch = s.shape[0]
        # MLP output → reshape to matrix
        flat = self.net(s)  # (batch, D*D)
        A_met = flat.view(batch, self.dim, self.dim)

        # Residual connection to identity for stability
        # This means the initial metric ≈ η (flat Minkowski or Euclidean)
        A_met = A_met + torch.eye(self.dim, device=s.device).unsqueeze(0)

        # Gauge fix: force A₀₀ > 0 so that the time direction has a
        # consistent sign convention across the batch. This is analogous
        # to choosing a time orientation on a Lorentzian manifold — not
        # cheating, just picking a convention.
        if self.geometry == "lorentzian":
            A_met = A_met.clone()
            A_met[:, 0, 0] = F.softplus(A_met[:, 0, 0]) + 1e-3

        return A_met

    def get_vielbein(self, s: torch.Tensor) -> torch.Tensor:
        """Return the vielbein A(s), shape (B, D, D).
        Public alias for _get_a_met — use this from outside the class."""
        if self.geometry == "euclidean":
            batch = s.shape[0]
            return torch.eye(self.dim, device=s.device).unsqueeze(0).expand(batch, -1, -1)
        return self._get_a_met(s)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute the full metric tensor g_θ(s) = A^T η A.

        Args:
            s: (batch, D) latent states

        Returns:
            g: (batch, D, D) metric tensors
        """
        if self.geometry == "euclidean":
            batch = s.shape[0]
            return torch.eye(self.dim, device=s.device).unsqueeze(0).expand(batch, -1, -1)

        A_met = self._get_a_met(s)  # (batch, D, D)
        batch = s.shape[0]
        # g = A^T η A
        eta_expanded = self.eta.unsqueeze(0).expand(batch, -1, -1)
        g = torch.bmm(A_met.transpose(1, 2), torch.bmm(eta_expanded, A_met))
        return g

    def squared_interval(self, s: torch.Tensor, delta_s: torch.Tensor) -> torch.Tensor:
        """
        Compute the squared interval Δs^T g_θ(s) Δs.

        This is the core geometric quantity:
        - Negative → time-like transition (narrative advances)
        - Positive → space-like transition (narrative branches)
        - Zero → null/light-like

        Args:
            s: (batch, D) current states
            delta_s: (batch, D) displacement vectors s_{t+1} - s_t

        Returns:
            interval: (batch,) squared intervals
        """
        if self.geometry == "euclidean":
            return (delta_s * delta_s).sum(dim=-1)

        batch = s.shape[0]
        A_met = self._get_a_met(s)  # (batch, D, D)
        # g Δs = A^T η A Δs
        A_delta = torch.bmm(A_met, delta_s.unsqueeze(-1))  # (batch, D, 1)
        # η (A Δs)
        eta_A_delta = A_delta * self.eta.diagonal().unsqueeze(0).unsqueeze(-1)
        # Δs^T g Δs = (A Δs)^T η (A Δs)
        interval = (A_delta * eta_A_delta).sum(dim=(1, 2))

        return interval

    def spatial_distance_sq(self, s: torch.Tensor, delta_s: torch.Tensor) -> torch.Tensor:
        """
        Compute the spatial (semantic) squared distance Δs^T h_θ(s) Δs.

        Uses the spatial metric h_θ = A^T Π A (time component projected out).
        This is always non-negative and measures semantic distance.

        Args:
            s: (batch, D) current states
            delta_s: (batch, D) displacements

        Returns:
            dist_sq: (batch,) spatial squared distances
        """
        if self.geometry == "euclidean":
            return (delta_s * delta_s).sum(dim=-1)

        batch = s.shape[0]
        A_met = self._get_a_met(s)  # (batch, D, D)
        A_delta = torch.bmm(A_met, delta_s.unsqueeze(-1))  # (batch, D, 1)
        # Π (A Δs) — zero out the time component
        pi_A_delta = A_delta * self.pi.diagonal().unsqueeze(0).unsqueeze(-1)
        dist_sq = (A_delta * pi_A_delta).sum(dim=(1, 2))

        return dist_sq

    def condition_number(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute the condition number of A^met for regularization.
        High condition number → metric is poorly conditioned → penalize.

        Returns:
            cond: (batch,) condition numbers
        """
        if self.geometry == "euclidean":
            return torch.ones(s.shape[0], device=s.device)

        A_met = self._get_a_met(s)
        # Singular values of A_met
        sv = torch.linalg.svdvals(A_met)
        cond = sv[:, 0] / (sv[:, -1] + 1e-8)
        return cond
