"""
Semantic Surrogate U_ψ(s, s'): learns to predict -log P_LM(d_{t+1}|d_{≤t})
from latent state pairs, avoiding expensive LM queries during training.

Trained with a combination of ranking loss (relative ordering of candidates)
and optional MSE (calibration). Once trained, it replaces the LM in the
Lagrangian for candidate-set evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSurrogate(nn.Module):
    """
    Learns to predict semantic transition cost from latent state pairs.
    U_ψ(s, s') ≈ -log P_LM(d' | d)
    """

    def __init__(self, dim: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = dim * 2
        dims = [in_dim] + [hidden_dim] * n_layers + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """
        Predict semantic cost for a batch of transitions.

        Args:
            s: (batch, D) current states
            s_next: (batch, D) next states

        Returns:
            cost: (batch,) predicted semantic costs (non-negative)
        """
        concat = torch.cat([s, s_next], dim=-1)
        return self.net(concat).squeeze(-1)

    def forward_candidates(
        self, s: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict semantic cost for all candidates at once.

        Args:
            s: (batch, D) current states
            candidates: (batch, C, D) candidate next states

        Returns:
            costs: (batch, C) predicted semantic costs
        """
        batch, n_cand, dim = candidates.shape
        s_exp = s.unsqueeze(1).expand(-1, n_cand, -1)
        concat = torch.cat([s_exp, candidates], dim=-1)
        return self.net(concat.reshape(-1, dim * 2)).squeeze(-1).reshape(batch, n_cand)


def train_semantic_surrogate(
    surrogate: SemanticSurrogate,
    s_train: torch.Tensor,
    s_next_train: torch.Tensor,
    lsem_train: torch.Tensor,
    n_epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    ranking_weight: float = 1.0,
    mse_weight: float = 0.1,
    device: str = "cuda",
) -> dict:
    """
    Train the semantic surrogate on cached LM scores.

    Uses ranking loss (U(s, s_neg) > U(s, s_true)) + optional MSE.
    Negatives are in-batch shuffled next states.
    """
    from torch.utils.data import TensorDataset, DataLoader

    surrogate = surrogate.to(device)
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=lr)

    dataset = TensorDataset(
        s_train.to(device),
        s_next_train.to(device),
        lsem_train.to(device),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    history = {"ranking_loss": [], "mse_loss": [], "total_loss": []}

    for epoch in range(n_epochs):
        epoch_losses = {"ranking": 0, "mse": 0, "total": 0}
        n_batches = 0

        for s, s_next, lsem_true in loader:
            pred_true = surrogate(s, s_next)
            mse = F.mse_loss(pred_true, lsem_true)

            perm = torch.randperm(len(s), device=device)
            s_neg = s_next[perm]
            pred_neg = surrogate(s, s_neg)

            margin = 0.5
            ranking = F.relu(pred_true - pred_neg + margin).mean()

            loss = ranking_weight * ranking + mse_weight * mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses["ranking"] += ranking.item()
            epoch_losses["mse"] += mse.item()
            epoch_losses["total"] += loss.item()
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
            history[f"{k}_loss"].append(epoch_losses[k])

        if (epoch + 1) % 10 == 0:
            print(f"  Surrogate epoch {epoch+1}/{n_epochs}: "
                  f"rank={epoch_losses['ranking']:.4f} "
                  f"mse={epoch_losses['mse']:.4f}")

    return history


def evaluate_surrogate_quality(
    surrogate: SemanticSurrogate,
    s: torch.Tensor,
    s_next: torch.Tensor,
    lsem_true: torch.Tensor,
    device: str = "cuda",
) -> dict:
    """
    Evaluate how well the surrogate predicts LM scores.

    Reports Pearson correlation, Spearman rank correlation, and MSE.
    """
    from scipy.stats import pearsonr, spearmanr

    surrogate.eval()
    with torch.no_grad():
        pred = surrogate(s.to(device), s_next.to(device)).cpu()

    true = lsem_true.cpu()
    pearson_r, _ = pearsonr(pred.numpy(), true.numpy())
    spearman_r, _ = spearmanr(pred.numpy(), true.numpy())
    mse = F.mse_loss(pred, true).item()

    return {
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "mse": mse,
    }
