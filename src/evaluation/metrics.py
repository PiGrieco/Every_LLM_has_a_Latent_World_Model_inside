"""
Evaluation metrics M1–M6 from Section 5.4 of the paper.

These metrics directly test the three primary hypotheses:
  H1 (Directedness) → M1: time-likeness rate
  H2 (Arrow of narrative) → M2: time-reversal gap
  H3 (Branching separation) → M3: space-like separation of branches

Plus diagnostic and quality metrics:
  M4: Cone alignment (Jaccard overlap of metric vs probabilistic cones)
  M5: Predictive quality (NLL under q_θ)
  M6: Branching/uncertainty signal (log Z correlation)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict


def m1_timelike_rate(
    metric,
    s: torch.Tensor,
    s_next: torch.Tensor,
) -> float:
    """
    M1: Time-likeness rate = Pr[Δs^T g_θ(s) Δs < 0] on real transitions.

    This is the most fundamental metric: if the learned Lorentzian metric
    is working, real transitions should be predominantly time-like
    (negative squared interval).

    A rate > 0.9 supports H1. A rate near 0.5 means the metric is not
    distinguishing time-like from space-like.

    Returns:
        rate: float in [0, 1]
    """
    delta_s = s_next - s
    intervals = metric.squared_interval(s, delta_s)  # (N,)
    return (intervals < 0).float().mean().item()


def m2_time_reversal_gap(
    metric,
    lagrangian,
    world_model,
    forward_trajectories: list,
    reversed_trajectories: list,
) -> Dict[str, float]:
    """
    M2: Time-reversal gap.

    Compare forward vs. reversed trajectories using:
    (a) Total action S_θ[γ] — reversed should have higher action
    (b) Average log-prob under q_θ — reversed should have lower prob

    A clear gap supports H2 (arrow of narrative). The Lorentzian metric
    should make this gap larger than Riemannian/Euclidean baselines.

    Args:
        forward_trajectories: list of (T, D) tensors
        reversed_trajectories: list of (T, D) tensors (same trajs, reversed)

    Returns:
        dict with:
            action_gap: mean(S_reversed) - mean(S_forward), should be > 0
            logprob_gap: mean(logp_forward) - mean(logp_reversed), should be > 0
            action_forward: mean forward action
            action_reversed: mean reversed action
    """
    forward_actions = []
    reversed_actions = []
    forward_logprobs = []
    reversed_logprobs = []

    for fwd, rev in zip(forward_trajectories, reversed_trajectories):
        # Action
        S_fwd = lagrangian.action(fwd).item()
        S_rev = lagrangian.action(rev).item()
        forward_actions.append(S_fwd)
        reversed_actions.append(S_rev)

        # Log-prob under world model
        s_fwd = fwd[:-1]
        s_next_fwd = fwd[1:]
        lp_fwd = world_model.log_prob(s_fwd, s_next_fwd).mean().item()

        s_rev = rev[:-1]
        s_next_rev = rev[1:]
        lp_rev = world_model.log_prob(s_rev, s_next_rev).mean().item()

        forward_logprobs.append(lp_fwd)
        reversed_logprobs.append(lp_rev)

    mean_S_fwd = sum(forward_actions) / len(forward_actions)
    mean_S_rev = sum(reversed_actions) / len(reversed_actions)
    mean_lp_fwd = sum(forward_logprobs) / len(forward_logprobs)
    mean_lp_rev = sum(reversed_logprobs) / len(reversed_logprobs)

    return {
        "action_gap": mean_S_rev - mean_S_fwd,
        "logprob_gap": mean_lp_fwd - mean_lp_rev,
        "action_forward": mean_S_fwd,
        "action_reversed": mean_S_rev,
    }


def m3_branching_separation(
    metric,
    branch_states: list,
    prefix_state: torch.Tensor,
) -> float:
    """
    M3: Branching separation.

    For a shared prefix state s, check whether different branch
    continuations are separated by space-like intervals:
        Pr[(s'^(a) - s'^(b))^T g_θ(s) (s'^(a) - s'^(b)) > 0]

    If branches are space-like separated, the Lorentzian cone structure
    is capturing the incompatibility of alternative continuations.

    Args:
        branch_states: list of (D,) tensors, one per branch continuation
        prefix_state: (D,) the shared prefix state

    Returns:
        rate: float in [0, 1], fraction of branch pairs that are space-like
    """
    n = len(branch_states)
    if n < 2:
        return 0.0

    spacelike_count = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            diff = branch_states[i] - branch_states[j]
            diff = diff.unsqueeze(0)  # (1, D)
            s = prefix_state.unsqueeze(0)  # (1, D)

            interval = metric.squared_interval(s, diff)
            if interval.item() > 0:
                spacelike_count += 1
            total_pairs += 1

    return spacelike_count / total_pairs


def m4_cone_alignment(
    metric,
    lagrangian,
    s: torch.Tensor,
    candidates: torch.Tensor,
    p: float = 0.8,
) -> dict:
    """
    M4: Cone alignment — overlap between the metric cone and the
    probabilistic cone on a candidate set.

    For each state s, we define:
      - Metric cone: candidates where Δs^T g_θ(s) Δs ≤ 0 (time-like)
      - Probabilistic cone: smallest set of candidates whose cumulative
        Gibbs probability ≥ p (the top-p mass set)

    Args:
        metric: MetricNetwork
        lagrangian: Lagrangian module
        s: (batch, D) current states
        candidates: (batch, C, D) candidate next states
        p: mass level for the probabilistic cone (default 0.8)

    Returns:
        dict with jaccard, precision, recall (averaged over batch)
    """
    batch, n_cand, dim = candidates.shape

    s_exp = s.unsqueeze(1).expand(-1, n_cand, -1)
    delta = candidates - s_exp

    s_flat = s_exp.reshape(-1, dim)
    delta_flat = delta.reshape(-1, dim)
    intervals = metric.squared_interval(s_flat, delta_flat).reshape(batch, n_cand)
    metric_cone = (intervals <= 0)

    c_flat = candidates.reshape(-1, dim)
    energies = lagrangian(s_flat, c_flat).reshape(batch, n_cand)

    gibbs_logits = -energies
    gibbs_probs = torch.softmax(gibbs_logits, dim=-1)

    sorted_probs, sorted_idx = gibbs_probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)

    prob_cone = torch.zeros_like(metric_cone)
    for i in range(batch):
        cutoff = (cumulative[i] >= p).float().argmax().item()
        top_indices = sorted_idx[i, : cutoff + 1]
        prob_cone[i, top_indices] = True

    intersection = (metric_cone & prob_cone).float().sum(dim=-1)
    union = (metric_cone | prob_cone).float().sum(dim=-1)
    metric_cone_size = metric_cone.float().sum(dim=-1)
    prob_cone_size = prob_cone.float().sum(dim=-1)

    jaccard = (intersection / union.clamp(min=1)).mean().item()
    precision = (intersection / prob_cone_size.clamp(min=1)).mean().item()
    recall = (intersection / metric_cone_size.clamp(min=1)).mean().item()

    return {"jaccard": jaccard, "precision": precision, "recall": recall}


def m5_predictive_nll(
    world_model,
    s: torch.Tensor,
    s_next: torch.Tensor,
) -> float:
    """
    M5: Predictive quality.

    Average negative log-likelihood under q_θ on held-out transitions.
    Lower is better — the world model assigns high probability to
    actually observed transitions.

    Returns:
        nll: float, average -log q_θ(s_{t+1} | s_t)
    """
    return world_model.neg_log_prob(s, s_next).mean().item()


def m6_branching_signal(
    lagrangian,
    s: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """
    M6: Branching/uncertainty signal.

    Compute log Ẑ_θ(s) = log Σ_i exp(-L_θ(s, s^(i))) for each state.
    High values = many plausible futures (branching point).
    Low values = determined continuation.

    Args:
        s: (batch, D) states
        candidates: (batch, C, D) candidate next states

    Returns:
        log_Z: (batch,) estimated log partition function
    """
    batch, n_cand, dim = candidates.shape

    # Compute Lagrangian for all candidates
    s_exp = s.unsqueeze(1).expand(-1, n_cand, -1).reshape(-1, dim)
    c_flat = candidates.reshape(-1, dim)
    energy = lagrangian(s_exp, c_flat).reshape(batch, n_cand)

    # Log-sum-exp
    log_Z = torch.logsumexp(-energy, dim=-1)  # (batch,)
    return log_Z


def compute_all_metrics(
    metric,
    world_model,
    lagrangian,
    s: torch.Tensor,
    s_next: torch.Tensor,
    forward_trajectories: Optional[list] = None,
    reversed_trajectories: Optional[list] = None,
    branch_data: Optional[dict] = None,
) -> Dict[str, float]:
    """
    Compute all applicable metrics given the available data.

    Returns a dict of metric_name -> value.
    """
    results = {}

    # M1: Time-likeness rate (always available)
    results["m1_timelike_rate"] = m1_timelike_rate(metric, s, s_next)

    # M5: Predictive NLL (always available)
    results["m5_nll"] = m5_predictive_nll(world_model, s, s_next)

    # M2: Time-reversal gap (needs forward + reversed trajectories)
    if forward_trajectories and reversed_trajectories:
        m2 = m2_time_reversal_gap(
            metric, lagrangian, world_model,
            forward_trajectories, reversed_trajectories,
        )
        results.update({f"m2_{k}": v for k, v in m2.items()})

    # M4: Cone alignment (needs candidate sets)
    if s.shape[0] >= 16:
        from ..training.candidates import build_candidate_set_c1
        cands_m4, _ = build_candidate_set_c1(s[:256], s_next[:256], candidate_size=32)
        m4 = m4_cone_alignment(metric, lagrangian, s[:256], cands_m4)
        results.update({f"m4_{k}": v for k, v in m4.items()})

    # M3: Branching separation (needs branch data)
    if branch_data:
        rates = []
        for prefix_s, branches in branch_data:
            rate = m3_branching_separation(metric, branches, prefix_s)
            rates.append(rate)
        results["m3_branching_sep"] = sum(rates) / len(rates)

    return results
