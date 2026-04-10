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


def time_likeness_loss(
    metric,
    s: torch.Tensor,
    s_next: torch.Tensor,
    margin: float = -0.1,
) -> torch.Tensor:
    """
    L_time = E[max(0, Δs^T g_θ(s) Δs + ε)]

    Penalizes transitions that are NOT time-like. A time-like transition
    has Δσ² < 0 (negative squared interval). The margin ε < 0 enforces
    a safety gap — we want Δσ² < ε, not just Δσ² < 0.

    This is the key loss for H1: if it converges to near-zero, real
    transitions are predominantly time-like under the learned metric.

    Args:
        metric: MetricNetwork instance
        s: (batch, D) current states
        s_next: (batch, D) next states
        margin: negative margin ε (more negative = stricter)

    Returns:
        loss: scalar, average hinge loss
    """
    delta_s = s_next - s
    interval = metric.squared_interval(s, delta_s)  # (batch,)
    # Hinge: max(0, interval + |margin|)
    # If interval < -|margin|, loss = 0 (sufficiently time-like)
    loss = F.relu(interval - margin).mean()
    return loss


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
    # Temperature-scaled teacher logits. Clamp temperature from below
    # to prevent division-by-near-zero in early training.
    T = max(temperature, min_temperature)
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
) -> dict:
    """
    Compute the total loss for a training step, respecting staged training.

    Stage 2: L_time + L_cond
    Stage 3: L_time + L_match + L_ML + L_smooth + L_cond

    Returns a dict with individual loss terms and the total.
    """
    losses = {}

    # Always: time-likeness
    losses["time"] = time_likeness_loss(
        metric, s, s_next,
        margin=cfg.causal_margin if cfg else -0.1,
    )

    # Always: condition number regularization
    losses["cond"] = condition_number_loss(metric, s) * 0.01

    if stage >= 3:
        # World model maximum likelihood
        losses["ml"] = maximum_likelihood_loss(world_model, s, s_next)

        # Candidate-set matching
        losses["match"] = candidate_set_matching_loss(
            lagrangian, world_model, s, candidates,
            precomputed_lsem=precomputed_lsem_candidates,
        )

        # Smoothness (if triplets of consecutive states available)
        if s_prev is not None:
            losses["smooth"] = smoothness_loss(s_prev, s, s_next)
        else:
            losses["smooth"] = torch.tensor(0.0, device=s.device)

    # Weighted sum
    lam = {
        "time": cfg.lambda_geo if cfg else 0.5,
        "cond": 0.01,
        "ml": cfg.lambda_wm if cfg else 1.0,
        "match": cfg.lambda_match if cfg else 1.0,
        "smooth": cfg.lambda_smooth if cfg else 0.1,
    }

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
    l_time = time_likeness_loss(metric, s, s_next, margin=cfg.causal_margin if cfg else -0.1)
    l_ml = maximum_likelihood_loss(world_model, s, s_next)
    l_match = candidate_set_matching_loss(lagrangian, world_model, s, candidates)

    magnitudes = {
        "time": l_time.item(),
        "ml": l_ml.item(),
        "match": l_match.item(),
    }

    # Only calibrate losses that are meaningfully above zero.
    # A near-zero loss (e.g. l_time after stage 2 convergence) should keep
    # its configured weight rather than receiving a blown-up coefficient.
    active = {k: v for k, v in magnitudes.items() if v > 1e-4}
    if len(active) < 2:
        return {"time": 1.0, "ml": 1.0, "match": 1.0}

    ref = (torch.tensor(list(active.values())).prod() ** (1.0 / len(active))).item()
    weights = {}
    for k, v in magnitudes.items():
        if v > 1e-4:
            weights[k] = min(ref / v, 100.0)
        else:
            weights[k] = 1.0
    return weights
