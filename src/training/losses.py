"""
Loss functions for the Lorentzian World Model framework.
Includes a loss-magnitude matching utility for automatic λ calibration.

All losses from Section 4.3 of the paper:
- L_time: time-likeness penalty (transitions should be time-like)
- L_smooth: smoothness penalty (curvature regularization)
- L_match: candidate-set KL matching (align q_θ with K_θ)
- L_ML: maximum likelihood (q_θ should explain the data)
- L_cond: metric condition number regularization
"""

import torch
import torch.nn.functional as F
from typing import Optional

from ..models.time_orientation import TimeOrientation, future_loss


def contrastive_cone_loss(
    metric,
    s: torch.Tensor,
    s_next: torch.Tensor,
    margin_inside: float = -0.1,
    margin_outside: float = 0.1,
) -> torch.Tensor:
    """
    Contrastive cone loss: push real transitions INSIDE the cone (time-like)
    and push negatives OUTSIDE the cone (space-like).

    This prevents the "cone collapse" problem where the metric learns a
    degenerate cone that encompasses everything.

    Args:
        metric: MetricNetwork instance
        s: (batch, D) current states
        s_next: (batch, D) true next states
        margin_inside: negative margin for time-like constraint
        margin_outside: positive margin for space-like constraint

    Returns:
        loss: scalar, average contrastive cone loss
    """
    batch = s.shape[0]

    delta_real = s_next - s
    interval_real = metric.squared_interval(s, delta_real)
    loss_inside = F.relu(interval_real - margin_inside).mean()

    perm = torch.randperm(batch, device=s.device)
    s_neg = s_next[perm]
    same_mask = (perm == torch.arange(batch, device=s.device))
    if same_mask.any():
        s_neg[same_mask] = s_next[(perm[same_mask] + 1) % batch]

    delta_neg = s_neg - s
    interval_neg = metric.squared_interval(s, delta_neg)
    loss_outside = F.relu(margin_outside - interval_neg).mean()

    return loss_inside + loss_outside


def smoothness_loss(
    s_prev: torch.Tensor,
    s_curr: torch.Tensor,
    s_next: torch.Tensor,
) -> torch.Tensor:
    """
    L_smooth = E[||Δs_{t+1} - Δs_t||²]

    Penalizes sharp changes in velocity (high curvature). Encourages
    worldlines to be geometrically smooth, which is desirable for
    both interpretability and metric estimation quality.

    Args:
        s_prev, s_curr, s_next: (batch, D) three consecutive states

    Returns:
        loss: scalar, average squared curvature
    """
    delta_curr = s_curr - s_prev  # velocity at t
    delta_next = s_next - s_curr  # velocity at t+1
    curvature = (delta_next - delta_curr).pow(2).sum(dim=-1)
    return curvature.mean()


def candidate_set_matching_loss(
    lagrangian,
    world_model,
    s: torch.Tensor,
    candidates: torch.Tensor,
    true_next_idx: torch.Tensor = None,
    precomputed_lsem: torch.Tensor = None,
    temperature: float = 1.0,
    min_temperature: float = 0.1,
) -> torch.Tensor:
    """
    L_match = E_s[KL(q̃_θ(·|s) || K̃_θ(·|s))]

    The core world-model alignment loss. For each state s, we have
    a set of candidate next states {s^(i)}, and we want the world
    model's distribution over these candidates to match the Gibbs
    kernel's distribution.

    This avoids explicit partition function regression and implicitly
    realizes max-entropy dynamics.

    Args:
        lagrangian: Lagrangian module (computes energy for each candidate)
        world_model: WorldModel module (computes log-prob for each candidate)
        s: (batch, D) current states
        candidates: (batch, C, D) candidate next states
        true_next_idx: (batch,) index of true next state in candidate set (optional)
        precomputed_lsem: (batch, C) semantic costs per candidate (optional)
        temperature: Gibbs temperature (β in the paper, default 1.0)

    Returns:
        loss: scalar, average KL divergence
    """
    batch, n_cand, dim = candidates.shape

    # ---- Teacher: Gibbs kernel on candidate set ----
    # K̃(i|s) = exp(-L_θ(s, s^(i))) / Σ_j exp(-L_θ(s, s^(j)))
    s_expanded = s.unsqueeze(1).expand(-1, n_cand, -1)  # (batch, C, D)
    s_flat = s_expanded.reshape(-1, dim)
    c_flat = candidates.reshape(-1, dim)

    # Compute Lagrangian for all (s, candidate) pairs
    lsem_flat = None
    if precomputed_lsem is not None:
        lsem_flat = precomputed_lsem.reshape(-1)

    energy = lagrangian(s_flat, c_flat, lsem_flat).reshape(batch, n_cand)
    # Use lagrangian's temperature if caller didn't override
    if temperature == 1.0 and hasattr(lagrangian, 'temperature'):
        T = lagrangian.temperature
    else:
        T = temperature
    T = max(T, min_temperature)
    teacher_logits = -energy / T

    # ---- Student: world model log-probs on candidate set ----
    student_log_probs = world_model.log_prob_candidates(s, candidates)  # (batch, C)

    # Normalize both to distributions
    teacher_log_dist = F.log_softmax(teacher_logits, dim=-1)
    student_log_dist = F.log_softmax(student_log_probs, dim=-1)

    # KL(student || teacher) = Σ student · (log student - log teacher)
    kl = F.kl_div(teacher_log_dist, student_log_dist, log_target=True, reduction="batchmean")

    return kl


def maximum_likelihood_loss(
    world_model,
    s: torch.Tensor,
    s_next: torch.Tensor,
) -> torch.Tensor:
    """
    L_ML = E[-log q_θ(s_{t+1} | s_t)]

    Standard negative log-likelihood. The world model should assign
    high probability to actually observed transitions.

    Args:
        world_model: WorldModel module
        s: (batch, D) current states
        s_next: (batch, D) true next states

    Returns:
        loss: scalar, average NLL
    """
    return world_model.neg_log_prob(s, s_next).mean()


def condition_number_loss(
    metric,
    s: torch.Tensor,
    target_cond: float = 10.0,
) -> torch.Tensor:
    """
    Penalize poorly conditioned metrics.

    High condition number means the metric is nearly degenerate, which
    causes numerical instability and poor geometric properties.

    Args:
        metric: MetricNetwork instance
        s: (batch, D) states
        target_cond: maximum acceptable condition number

    Returns:
        loss: scalar penalty
    """
    cond = metric.condition_number(s)  # (batch,)
    return F.relu(cond - target_cond).mean()


def compute_total_loss(
    metric,
    lagrangian,
    world_model,
    s: torch.Tensor,
    s_next: torch.Tensor,
    candidates: torch.Tensor,
    s_prev: Optional[torch.Tensor] = None,
    precomputed_lsem: Optional[torch.Tensor] = None,
    precomputed_lsem_candidates: Optional[torch.Tensor] = None,
    cfg=None,
    stage: int = 2,
    dynamic_weights: Optional[dict] = None,
    time_fn=None,
) -> dict:
    """
    Compute the total loss for a training step, respecting staged training.

    Stage 2: L_cone_contrastive + L_future + L_cond
    Stage 3: L_cone_contrastive + L_future + L_match + L_ML + L_cond

    The contrastive cone loss replaces the old one-sided time-likeness loss,
    preventing cone collapse. The future loss (from τ_θ) breaks time-reversal
    symmetry in the energy, making M2(action gap) a meaningful probe.
    """
    losses = {}

    losses["cone"] = contrastive_cone_loss(
        metric, s, s_next,
        margin_inside=cfg.causal_margin if cfg else -0.1,
        margin_outside=cfg.cone_margin_outside if cfg else 0.1,
    )

    if time_fn is not None:
        losses["future"] = future_loss(
            time_fn, s, s_next,
            margin=cfg.future_margin if cfg else 0.1,
        )
    else:
        losses["future"] = torch.tensor(0.0, device=s.device)

    losses["cond"] = condition_number_loss(metric, s) * 0.1

    if stage >= 3:
        losses["ml"] = maximum_likelihood_loss(world_model, s, s_next)

        losses["match"] = candidate_set_matching_loss(
            lagrangian, world_model, s, candidates,
            precomputed_lsem=precomputed_lsem_candidates,
        )

        if s_prev is not None:
            losses["smooth"] = smoothness_loss(s_prev, s, s_next)
        else:
            losses["smooth"] = torch.tensor(0.0, device=s.device)

    lam = {
        "cone": cfg.lambda_geo if cfg else 0.5,
        "future": cfg.lambda_future if cfg else 0.5,
        "cond": 0.1,
        "ml": cfg.lambda_wm if cfg else 1.0,
        "match": cfg.lambda_match if cfg else 1.0,
        "smooth": cfg.lambda_smooth if cfg else 0.1,
    }
    if dynamic_weights is not None:
        lam.update(dynamic_weights)

    total = torch.tensor(0.0, device=s.device)
    for key, val in losses.items():
        total = total + lam.get(key, 1.0) * val

    losses["total"] = total
    return losses


@torch.no_grad()
def calibrate_loss_weights(
    metric,
    lagrangian,
    world_model,
    s: torch.Tensor,
    s_next: torch.Tensor,
    candidates: torch.Tensor,
    cfg=None,
) -> dict:
    """
    Loss-magnitude matching: compute each loss term on a sample batch
    and return λ values that make all terms roughly equal in magnitude.

    This eliminates the need for manual λ tuning. Call once at the start
    of each training stage and use the returned weights for that stage.

    Returns:
        weights: dict of loss_name -> recommended λ
    """
    l_cone = contrastive_cone_loss(metric, s, s_next)
    l_ml = maximum_likelihood_loss(world_model, s, s_next)
    l_match = candidate_set_matching_loss(lagrangian, world_model, s, candidates)

    magnitudes = {
        "cone": l_cone.item(),
        "ml": l_ml.item(),
        "match": l_match.item(),
    }

    # Only calibrate losses that are meaningfully above zero.
    # A near-zero loss (e.g. l_time after stage 2 convergence) should keep
    # its configured weight rather than receiving a blown-up coefficient.
    active = {k: v for k, v in magnitudes.items() if v > 1e-4}
    if len(active) < 2:
        return {"cone": 1.0, "ml": 1.0, "match": 1.0}

    ref = (torch.tensor(list(active.values())).prod() ** (1.0 / len(active))).item()
    weights = {}
    for k, v in magnitudes.items():
        if v > 1e-4:
            weights[k] = min(ref / v, 100.0)
        else:
            weights[k] = 1.0
    return weights
