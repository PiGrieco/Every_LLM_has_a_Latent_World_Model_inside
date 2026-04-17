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
    time_fn=None,
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

    result_dict = {
        "action_gap": mean_S_rev - mean_S_fwd,
        "logprob_gap": mean_lp_fwd - mean_lp_rev,
        "action_forward": mean_S_fwd,
        "action_reversed": mean_S_rev,
    }

    if time_fn is not None:
        fwd_dtau, rev_dtau = [], []
        for fwd, rev in zip(forward_trajectories, reversed_trajectories):
            dt_fwd = time_fn.delta_tau(metric, fwd[:-1], fwd[1:]).mean().item()
            dt_rev = time_fn.delta_tau(metric, rev[:-1], rev[1:]).mean().item()
            fwd_dtau.append(dt_fwd)
            rev_dtau.append(dt_rev)
        result_dict["delta_tau_forward"] = sum(fwd_dtau) / len(fwd_dtau)
        result_dict["delta_tau_reversed"] = sum(rev_dtau) / len(rev_dtau)

    # --- M2 decomposition: geometry-only vs Δτ-only ---
    # Geometry-only: Δσ² gap without τ contribution
    fwd_intervals, rev_intervals = [], []
    for fwd, rev in zip(forward_trajectories, reversed_trajectories):
        si_fwd = metric.squared_interval(fwd[:-1], fwd[1:] - fwd[:-1]).mean().item()
        si_rev = metric.squared_interval(rev[:-1], rev[1:] - rev[:-1]).mean().item()
        fwd_intervals.append(si_fwd)
        rev_intervals.append(si_rev)
    mean_si_fwd = sum(fwd_intervals) / len(fwd_intervals)
    mean_si_rev = sum(rev_intervals) / len(rev_intervals)
    result_dict["geo_only_gap"] = mean_si_rev - mean_si_fwd

    if time_fn is not None:
        result_dict["future_only_gap"] = (
            result_dict["delta_tau_forward"] - result_dict["delta_tau_reversed"]
        )

    return result_dict


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


def m3_prime_joint_branching(
    metric,
    trajectories_by_branch: dict,
    prefix_length: int,
) -> dict:
    """
    M3': Joint branching metric requiring BOTH within-branch time-like
    AND cross-branch space-like. Only Lorentzian can achieve both.
    """
    within_intervals = []
    for branch_id, trajs in trajectories_by_branch.items():
        for traj in trajs:
            post = traj[prefix_length:]
            if len(post) < 2:
                continue
            s = post[:-1]
            s_next = post[1:]
            delta = s_next - s
            intervals = metric.squared_interval(s, delta)
            within_intervals.append(intervals)

    if within_intervals:
        all_within = torch.cat(within_intervals)
        within_timelike = (all_within < 0).float().mean().item()
    else:
        within_timelike = 0.0

    cross_intervals = []
    branch_ids = sorted(trajectories_by_branch.keys())
    for i, bid_a in enumerate(branch_ids):
        for bid_b in branch_ids[i+1:]:
            trajs_a = trajectories_by_branch[bid_a]
            trajs_b = trajectories_by_branch[bid_b]
            n_compare = min(len(trajs_a), len(trajs_b), 10)
            for k in range(n_compare):
                traj_a = trajs_a[k]
                traj_b = trajs_b[k]
                max_t = min(len(traj_a), len(traj_b))
                for t in range(prefix_length + 1, max_t, 2):
                    sa = traj_a[t].unsqueeze(0)
                    sb = traj_b[t].unsqueeze(0)
                    delta = sb - sa
                    interval = metric.squared_interval(sa, delta)
                    cross_intervals.append(interval)

    if cross_intervals:
        all_cross = torch.cat(cross_intervals)
        cross_spacelike = (all_cross > 0).float().mean().item()
    else:
        cross_spacelike = 0.0

    return {
        "within_timelike_rate": within_timelike,
        "cross_spacelike_rate": cross_spacelike,
        "joint_score": within_timelike * cross_spacelike,
    }


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


def m4_fair(
    metric,
    s: torch.Tensor,
    s_next: torch.Tensor,
    s_neg: torch.Tensor,
    geometry: str,
    base_rate: float = None,
) -> dict:
    """
    Geometry-agnostic M4: does the model's notion of "reachable from s"
    align with actual next states (vs random negatives)?

    For Lorentzian: reachable = Δσ² ≤ 0 (inside light cone)
    For Riemannian/Euclidean: reachable = distance² ≤ threshold,
        calibrated so reachable_rate matches base_rate (from Lorentzian)

    This makes the comparison fair: all geometries classify the same
    fraction of pairs as reachable, so differences in M4 reflect
    directional structure (cone vs ball), not threshold tuning.

    Args:
        metric: MetricNetwork
        s: (N, D) current states
        s_next: (N, D) true next states
        s_neg: (N, D) random negative states (shuffled next states)
        geometry: "lorentzian" | "riemannian" | "euclidean"
        base_rate: fraction of pairs to classify as reachable.
            If None and geometry is lorentzian, computed from data.
            If None and geometry is not lorentzian, uses 0.5.

    Returns:
        dict with m4f_jaccard, m4f_precision, m4f_recall, m4f_base_rate
    """
    delta_real = s_next - s
    delta_neg = s_neg - s
    intervals_real = metric.squared_interval(s, delta_real)
    intervals_neg = metric.squared_interval(s, delta_neg)

    if geometry == "lorentzian":
        reachable_real = intervals_real <= 0
        reachable_neg = intervals_neg <= 0
        if base_rate is None:
            all_intervals = torch.cat([intervals_real, intervals_neg])
            base_rate = (all_intervals <= 0).float().mean().item()
    else:
        all_intervals = torch.cat([intervals_real, intervals_neg])
        if base_rate is None:
            base_rate = 0.5
        threshold = torch.quantile(all_intervals.float(), base_rate)
        reachable_real = intervals_real <= threshold
        reachable_neg = intervals_neg <= threshold

    # "LM-plausible" = true next states; "not plausible" = random negatives
    # Among reachable pairs: what fraction are real?
    tp = reachable_real.float().sum()
    fp = reachable_neg.float().sum()
    fn = (~reachable_real).float().sum()

    precision = (tp / (tp + fp).clamp(min=1)).item()
    recall = (tp / (tp + fn).clamp(min=1)).item()
    jaccard = (tp / (tp + fp + fn).clamp(min=1)).item()

    return {
        "m4f_jaccard": jaccard,
        "m4f_precision": precision,
        "m4f_recall": recall,
        "m4f_base_rate": base_rate,
    }


def m4_fair_candidates(
    metric,
    lagrangian,
    s: torch.Tensor,
    candidates: torch.Tensor,
    geometry: str,
    p: float = 0.8,
    base_rate: float = None,
    semantic_costs: torch.Tensor = None,
) -> dict:
    """
    Fair M4 on candidate sets — geometry-agnostic, stack-aligned.

    Supersedes ``m4_fair`` (which uses shuffled negatives + a quantile
    threshold) in three ways:

    1. **Same candidate set the model sees at training time** (not random
       shuffles), so the evaluation matches the optimization objective.
    2. **Reachable-set semantics that make sense for non-Lorentzian
       baselines**: instead of thresholding on an interval quantile, we
       pick the top-k candidates by squared interval with
       ``k = round(base_rate · C)``. This guarantees all geometries
       classify the same count of candidates as reachable, and "closer"
       means the same thing in every geometry (small squared interval).
    3. **Plausibility set optionally defined by the semantic surrogate
       alone**, so the Lagrangian's geometric term does not bias the
       reference set toward the geometry being evaluated ("the Lagrangian
       judging itself" failure mode).

    Definitions:

      - Reachable:
          * lorentzian           : {c : Δσ²(s, c) ≤ 0}   (inside light cone)
          * riemannian/euclidean : top-k under Δσ² with k = round(base_rate·C)
      - Plausible:
          * if ``semantic_costs`` provided : top-p mass of softmax(-cost)
          * else                           : top-p mass of softmax(-L_θ)

    Args:
        metric: MetricNetwork.
        lagrangian: Lagrangian (used only when semantic_costs is None).
        s: (B, D) current states.
        candidates: (B, C, D) candidate next states.
        geometry: "lorentzian" | "riemannian" | "euclidean".
        p: mass cutoff for the probabilistic cone (default 0.8).
        base_rate: reachable fraction. None means auto-calibrate
            (Lorentzian only); baselines must receive the Lorentzian
            value for a fair comparison.
        semantic_costs: optional (B, C) per-candidate semantic costs
            (e.g. from the trained semantic surrogate). When provided,
            the plausibility set is defined without the geometric term.

    Returns:
        dict with m4f_jaccard, m4f_precision, m4f_recall, m4f_base_rate.
    """
    batch, C, D = candidates.shape

    s_exp = s.unsqueeze(1).expand(-1, C, -1)
    delta = candidates - s_exp
    s_flat = s_exp.reshape(-1, D)
    delta_flat = delta.reshape(-1, D)
    intervals = metric.squared_interval(s_flat, delta_flat).reshape(batch, C)

    # ---- Reachable set ----
    if geometry == "lorentzian":
        reachable = (intervals <= 0)
        if base_rate is None:
            base_rate = reachable.float().mean().item()
    else:
        if base_rate is None:
            raise ValueError(
                "m4_fair_candidates for non-Lorentzian needs base_rate "
                "(typically from the Lorentzian run on the same split)."
            )
        k = max(1, int(round(base_rate * C)))
        order = torch.argsort(intervals, dim=-1, descending=False)
        reachable = torch.zeros_like(intervals, dtype=torch.bool)
        reachable.scatter_(1, order[:, :k], True)

    # ---- Plausibility set ----
    if semantic_costs is not None:
        logits = -semantic_costs
    else:
        c_flat = candidates.reshape(-1, D)
        energies = lagrangian(s_flat, c_flat).reshape(batch, C)
        logits = -energies

    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)

    plausible = torch.zeros_like(reachable)
    for i in range(batch):
        cutoff = (cumulative[i] >= p).float().argmax().item()
        top_idx = sorted_idx[i, : cutoff + 1]
        plausible[i, top_idx] = True

    intersection = (reachable & plausible).float().sum(dim=-1)
    union = (reachable | plausible).float().sum(dim=-1)
    reachable_size = reachable.float().sum(dim=-1)
    plausible_size = plausible.float().sum(dim=-1)

    jaccard = (intersection / union.clamp(min=1)).mean().item()
    precision = (intersection / plausible_size.clamp(min=1)).mean().item()
    recall = (intersection / reachable_size.clamp(min=1)).mean().item()

    return {
        "m4f_jaccard": jaccard,
        "m4f_precision": precision,
        "m4f_recall": recall,
        "m4f_base_rate": base_rate,
    }


def m5_predictive_nll(
    world_model,
    s: torch.Tensor,
    s_next: torch.Tensor,
) -> float:
    """
    M5 continuous (DIAGNOSTIC ONLY — not the thesis metric).

    Average negative log-likelihood under the Gaussian world model q_θ
    on held-out transitions. Kept for backward compatibility and as a
    cheap sanity check.

    Why we do NOT rely on this value to compare geometries:
    `ConditionalGaussianWorldModel` clamps log-variance at
    `min_logvar = -10`. With latent dim D=16, a fully collapsed variance
    (σ² = e^{-10}) plus a perfect mean gives a theoretical NLL floor of

        floor = 0.5 · (D·log(2π) + D·min_logvar)  ≈  -65.297

    In practice, Riemannian/Euclidean baselines on D2 land at ≈ -65.29:
    that is *variance collapse*, not predictive skill. Use
    `m5_candidate_metrics` for a geometry-fair, clamp-invariant
    assessment based on candidate-set ranking — that's the metric
    aligned with the paper's Gibbsian formulation. Run
    `world_model_variance_diagnostic` alongside to flag when the
    continuous NLL is dominated by the clamp.

    Returns:
        nll: float, average -log q_θ(s_{t+1} | s_t)
    """
    return world_model.neg_log_prob(s, s_next).mean().item()


def world_model_variance_diagnostic(
    world_model,
    s: torch.Tensor,
) -> dict:
    """
    Flag Gaussian-world-model variance collapse.

    The Gaussian ``ConditionalGaussianWorldModel`` clamps per-dim
    log-variance at ``world_model.min_logvar`` (default -10). When
    training pushes log-variance to the floor, the continuous M5 NLL
    becomes dominated by the clamp (cf. the -65.30 floor for D=16) and
    is no longer a meaningful comparison signal. This helper reports

      - ``wm_mean_logvar``  : average log-variance on the sample
      - ``wm_frac_at_floor``: fraction of dims within 0.1 of the floor
      - ``wm_logvar_floor`` : the floor itself, for context
      - ``wm_collapsed``    : boolean verdict (mean < floor + 0.5)

    Args:
        world_model: module exposing a ``forward(s) -> (mean, logvar)``
            signature and a ``min_logvar`` attribute.
        s: (N, D) latent states to evaluate on.

    Returns:
        dict with the four keys above.
    """
    with torch.no_grad():
        _, logvar = world_model(s)
        mean_logvar = logvar.mean().item()
        frac_at_floor = (
            logvar <= world_model.min_logvar + 0.1
        ).float().mean().item()

    return {
        "wm_mean_logvar": mean_logvar,
        "wm_frac_at_floor": frac_at_floor,
        "wm_logvar_floor": float(world_model.min_logvar),
        "wm_collapsed": mean_logvar < world_model.min_logvar + 0.5,
    }


def m5_candidate_metrics(
    world_model,
    s: torch.Tensor,
    candidates: torch.Tensor,
    true_idx: torch.Tensor,
    topk: int = 5,
) -> dict:
    """
    Discrete predictive metrics on a pre-built candidate set.

    Uses ONLY the world model q_θ (not the Gibbs kernel K_θ), because:

      1. q_θ is the object the paper claims as the extracted world model.
      2. K_θ includes the Lagrangian's geometric term, so using it as a
         discrete scoring function would bias the comparison toward the
         Lorentzian geometry (it "judges itself").

    Reports four standard next-item metrics on the candidate set:

      - ``m5d_ce``   : cross-entropy (NLL of the true class under the
                      set-normalised softmax of log q)
      - ``m5d_top1`` : fraction of rows where argmax == true_idx
      - ``m5d_topk`` : fraction of rows where true_idx is in the top k
      - ``m5d_mrr``  : mean reciprocal rank of the true candidate
      - ``m5d_topk_k``: effective k used (min(topk, C))

    Ties in log q are resolved by ``argsort`` stably, so the reported
    rank is deterministic given the same inputs.

    Args:
        world_model: module exposing ``log_prob(s, s_next)``.
        s: (B, D) current states.
        candidates: (B, C, D) candidate next states.
        true_idx: (B,) long tensor; index of the true next state within
            each candidate set (0 for C1-built sets).
        topk: k for top-k recall (default 5).

    Returns:
        dict with the five keys listed above.
    """
    import torch.nn.functional as F

    batch, C, D = candidates.shape
    s_exp = s.unsqueeze(1).expand(-1, C, -1).reshape(-1, D)
    c_flat = candidates.reshape(-1, D)

    log_q = world_model.log_prob(s_exp, c_flat).reshape(batch, C)
    log_q_norm = F.log_softmax(log_q, dim=-1)

    # Cross-entropy on the true candidate
    ce = F.nll_loss(log_q_norm, true_idx.long()).item()

    # Ranks (1-based; ties broken by index position via argsort)
    order = torch.argsort(log_q, dim=-1, descending=True)
    true_idx_exp = true_idx.long().unsqueeze(1)
    rank = (order == true_idx_exp).float().argmax(dim=-1) + 1  # (B,)

    top1 = (rank == 1).float().mean().item()
    topk_eff = min(topk, C)
    topk_recall = (rank <= topk_eff).float().mean().item()
    mrr = (1.0 / rank.float()).mean().item()

    return {
        "m5d_ce": ce,
        "m5d_top1": top1,
        "m5d_topk": topk_recall,
        "m5d_mrr": mrr,
        "m5d_topk_k": topk_eff,
    }


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
    Light-weight training-time eval bundle.

    Invoked every ``cfg.eval_every`` epochs by ``WorldModelTrainer``, so
    this function intentionally excludes the expensive, protocol-heavy
    final-eval metrics (``m4_fair_candidates``, ``m5_candidate_metrics``,
    coherence probe). Those live in ``run_d2`` and are computed once on
    the held-out eval set at the end of training, where Lorentzian can
    calibrate a shared ``base_rate`` cleanly for the baselines.

    Included here:

      - M1 (time-likeness rate)
      - M5 continuous NLL (diagnostic; may be dominated by variance clamp)
      - world_model_variance_diagnostic (cheap and useful to track the
        collapse trajectory during training)
      - M4 raw cone alignment (tautological on non-Lorentzian; a
        consistency check at training time for the Lorentzian branch)
      - M2_* when forward/reversed trajectories are supplied (D0/D1)
      - M3 branching separation when branch data are supplied (D1)

    NOT included (deliberately): m4_fair_*, m5d_*, probe_*. Compute those
    at the end of training with the proper coupling protocol.
    """
    results = {}

    # M1: Time-likeness rate (always available)
    results["m1_timelike_rate"] = m1_timelike_rate(metric, s, s_next)

    # M5 continuous (diagnostic; unreliable under variance clamp)
    results["m5_nll"] = m5_predictive_nll(world_model, s, s_next)

    # Variance-collapse diagnostic — cheap; useful trajectory to plot
    n_diag = min(512, s.shape[0])
    results.update(world_model_variance_diagnostic(world_model, s[:n_diag]))

    # M2: Time-reversal gap (only when forward + reversed trajectories exist)
    if forward_trajectories and reversed_trajectories:
        m2 = m2_time_reversal_gap(
            metric, lagrangian, world_model,
            forward_trajectories, reversed_trajectories,
        )
        results.update({f"m2_{k}": v for k, v in m2.items()})

    # M4 raw (diagnostic only — m4fair_* / m5d_* belong to the final eval)
    if s.shape[0] >= 16:
        from ..training.candidates import build_candidate_set_c1
        n_ev = min(256, s.shape[0])
        cands_m4, _ = build_candidate_set_c1(
            s[:n_ev], s_next[:n_ev], candidate_size=32
        )
        m4 = m4_cone_alignment(metric, lagrangian, s[:n_ev], cands_m4)
        results.update({f"m4raw_{k}": v for k, v in m4.items()})

    # M3: Branching separation (D1)
    if branch_data:
        rates = []
        for prefix_s, branches in branch_data:
            rate = m3_branching_separation(metric, branches, prefix_s)
            rates.append(rate)
        results["m3_branching_sep"] = sum(rates) / len(rates)

    return results
