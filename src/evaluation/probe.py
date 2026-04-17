"""
Discourse coherence probe — extrinsic test of the learned latent.

The only test in this repo that is extrinsic to the framework itself: it
asks whether the latent `s` still carries real discourse information
independently of the geometry we imposed. If it does, a LINEAR probe
on the concatenation [s_t ; s_{t+1}] should classify

    positive : (s_t, s_{t+1}) from consecutive paragraphs of the same article
    negative : (s_t, s'_{t+1}) with s' from a DIFFERENT article

at better-than-chance rates. We also run the same probe on the raw
preprocessed encoder embeddings so the latent result can be compared
against a known-good baseline: if the latent probe accuracy drops
sharply below the raw probe, the geometric projection is losing signal
— which is a result worth reporting honestly, not a bug of the probe.

The probe is a 1-layer linear classifier *by design*: we want to measure
linear separability of the latent, not the capacity of a non-linear
decoder.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------
# Data construction
# --------------------------------------------------------------------------

def build_coherence_pairs(
    trajectories: list,
    n_positive: int = 5000,
    n_negative: int = 5000,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a labelled coherence dataset from a list of per-article
    trajectories (each trajectory is a (T_i, D) tensor of states).

    Positive pair: (traj[i][t], traj[i][t+1]) — consecutive paragraphs
                   of article i.
    Negative pair: (traj[i][t], traj[j][t']) with j != i — paragraphs
                   from different articles (uniformly sampled, unordered).

    Returns:
        inputs: (N, 2D) tensor, N = n_positive + n_negative.
        labels: (N,) tensor in {0, 1}. 1 = coherent, 0 = incoherent.
    """
    g = torch.Generator().manual_seed(seed)

    # Keep only usable trajectories. We need at least one "transition"
    # (T ≥ 2) for positives, and at least 2 distinct articles for negatives.
    valid = [t for t in trajectories if len(t) >= 2]
    if len(valid) < 2:
        raise ValueError(
            f"build_coherence_pairs needs ≥ 2 trajectories with ≥ 2 states each, "
            f"got {len(valid)} usable out of {len(trajectories)}."
        )

    lens = torch.tensor([len(t) for t in valid], dtype=torch.long)
    n_articles = len(valid)

    # Flatten once so we can index with a single vectorized gather.
    all_states = torch.cat(valid, dim=0)   # (sum_T, D)
    offsets = torch.cat([
        torch.zeros(1, dtype=torch.long),
        torch.cumsum(lens, dim=0),
    ])                                     # (n_articles + 1,)

    # ---- Positive: same article, consecutive offsets ----
    art_pos = torch.randint(0, n_articles, (n_positive,), generator=g)
    # valid transitions per article = lens - 1 (≥ 1 by construction)
    max_t = (lens[art_pos] - 1).float()
    u = torch.rand(n_positive, generator=g)
    t_pos = (u * max_t).long().clamp_(max=(max_t.long() - 1))  # [0, lens-2]
    g_idx = offsets[art_pos] + t_pos
    pos_s = all_states[g_idx]
    pos_sn = all_states[g_idx + 1]

    # ---- Negative: different article, any offsets ----
    art_neg_i = torch.randint(0, n_articles, (n_negative,), generator=g)
    # Shift by a nonzero offset → guaranteed j != i, uniform over {0..N-1}\{i}
    shift = torch.randint(1, n_articles, (n_negative,), generator=g)
    art_neg_j = (art_neg_i + shift) % n_articles
    u_i = torch.rand(n_negative, generator=g)
    u_j = torch.rand(n_negative, generator=g)
    t_i = (u_i * lens[art_neg_i].float()).long().clamp_(max=lens[art_neg_i] - 1)
    t_j = (u_j * lens[art_neg_j].float()).long().clamp_(max=lens[art_neg_j] - 1)
    neg_s = all_states[offsets[art_neg_i] + t_i]
    neg_sn = all_states[offsets[art_neg_j] + t_j]

    pos_inputs = torch.cat([pos_s, pos_sn], dim=-1)   # (n_positive, 2D)
    neg_inputs = torch.cat([neg_s, neg_sn], dim=-1)   # (n_negative, 2D)

    inputs = torch.cat([pos_inputs, neg_inputs], dim=0)
    labels = torch.cat([
        torch.ones(n_positive),
        torch.zeros(n_negative),
    ])

    return inputs, labels


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """Single affine layer → scalar logit. One-layer BY DESIGN:
    we're measuring linear separability of the latent, not the capacity
    of a non-linear classifier."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def _accuracy(probe: LinearProbe, loader: DataLoader) -> float:
    probe.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = (probe(x) > 0).long()
            correct += (pred == y.long()).sum().item()
            total += len(y)
    return correct / max(total, 1)


def train_probe(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    val_frac: float = 0.2,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    batch_size: int = 256,
    patience: int = 5,
) -> dict:
    """Train LinearProbe on (inputs, labels) with a held-out validation
    split.

    Uses Adam + BCEWithLogitsLoss. Early-stops when validation accuracy
    fails to improve for `patience` consecutive epochs.

    Args:
        inputs: (N, 2D) concatenated pairs.
        labels: (N,) in {0, 1}.
        val_frac: fraction of samples held out for validation.
        seed: controls the train/val split and probe init.

    Returns:
        dict with keys:
          - train_acc  : final training accuracy
          - val_acc    : best validation accuracy seen during training
          - val_auroc  : AUROC of the probe at the best-val-acc epoch
          - epochs_run : number of epochs actually run before early stop
    """
    assert inputs.ndim == 2 and labels.ndim == 1 and len(inputs) == len(labels)

    # Deterministic split + probe initialisation
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)

    inputs = inputs.detach().float()
    labels = labels.detach().float()

    n = len(inputs)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(val_frac * n))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_loader = DataLoader(
        TensorDataset(inputs[train_idx], labels[train_idx]),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    val_loader = DataLoader(
        TensorDataset(inputs[val_idx], labels[val_idx]),
        batch_size=batch_size, shuffle=False,
    )

    in_dim = inputs.shape[1]
    probe = LinearProbe(in_dim)
    optimizer = torch.optim.Adam(
        probe.parameters(), lr=lr, weight_decay=weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_acc = -1.0
    best_state = None
    no_improve = 0
    epochs_run = 0

    for epoch in range(epochs):
        epochs_run = epoch + 1
        probe.train()
        for x, y in train_loader:
            logits = probe(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_acc = _accuracy(probe, val_loader)

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Restore the best checkpoint for the final metrics
    if best_state is not None:
        probe.load_state_dict(best_state)

    train_acc = _accuracy(probe, train_loader)

    # AUROC on validation set using the restored best probe.
    probe.eval()
    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for x, y in val_loader:
            val_logits_list.append(probe(x))
            val_labels_list.append(y)
    val_logits = torch.cat(val_logits_list).cpu().numpy()
    val_labels_arr = torch.cat(val_labels_list).cpu().numpy()

    try:
        from sklearn.metrics import roc_auc_score
        val_auroc = float(roc_auc_score(val_labels_arr, val_logits))
    except Exception:
        # Degenerate case (e.g. only one class in val split) or sklearn missing.
        val_auroc = float("nan")

    return {
        "train_acc": float(train_acc),
        "val_acc": float(best_val_acc),
        "val_auroc": val_auroc,
        "epochs_run": int(epochs_run),
    }
