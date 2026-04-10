"""
Geometric segmentation of micro-worldlines into macro-events.

Implements Algorithm 1 from the paper: velocity + curvature thresholds
with hysteresis to detect narrative change-points in latent space.

For D0/D1 (synthetic), segmentation is typically skipped (each time
step is already a meaningful event). For D2 (real text), segmentation
groups nearby paragraphs into coherent narrative events.
"""

import torch
from typing import List, Tuple


def compute_velocities_curvatures(
    trajectory: torch.Tensor,
    metric_fn=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute velocity and curvature along a trajectory.

    If metric_fn is provided, velocity uses the spatial metric h_θ.
    Otherwise, Euclidean norm is used.

    Args:
        trajectory: (T, D) tensor of states
        metric_fn: optional function (s, delta_s) -> scalar spatial distance²

    Returns:
        velocities: (T-1,) speed at each transition
        curvatures: (T-2,) discrete curvature at each transition
    """
    T, D = trajectory.shape
    deltas = trajectory[1:] - trajectory[:-1]  # (T-1, D)

    # Velocities
    if metric_fn is not None:
        velocities = torch.sqrt(
            metric_fn(trajectory[:-1], deltas).clamp(min=1e-12)
        )
    else:
        velocities = deltas.norm(dim=-1)

    # Curvatures: |Δs_{t+1} - Δs_t| / (|Δs_t| + ε)
    if T >= 3:
        delta_diff = deltas[1:] - deltas[:-1]  # (T-2, D)
        curvatures = delta_diff.norm(dim=-1) / (deltas[:-1].norm(dim=-1) + 1e-8)
    else:
        curvatures = torch.zeros(0)

    return velocities, curvatures


def moving_average(x: torch.Tensor, window: int) -> torch.Tensor:
    """Simple moving average smoothing."""
    if window <= 1 or len(x) == 0:
        return x

    # Pad with edge values
    pad = window // 2
    padded = torch.cat([x[:1].expand(pad), x, x[-1:].expand(pad)])
    # Cumulative sum trick for efficient MA
    cumsum = padded.cumsum(dim=0)
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    return smoothed[: len(x)]


def segment_trajectory(
    trajectory: torch.Tensor,
    metric_fn=None,
    window: int = 3,
    l_min: int = 3,
    l_max: int = 50,
    tau_v_mult: float = 1.5,
    tau_kappa_mult: float = 1.5,
    tau_v_low_mult: float = 0.5,
    tau_kappa_low_mult: float = 0.5,
) -> Tuple[List[int], torch.Tensor]:
    """
    Algorithm 1: Geometric change-point segmentation.

    Detects segment boundaries based on velocity and curvature spikes
    with hysteresis thresholds. Thresholds are set as multiples of the
    trajectory-level median, making them adaptive to each trajectory.

    Args:
        trajectory: (T, D) tensor of micro-states
        metric_fn: optional spatial metric for velocity computation
        window: smoothing window for moving average
        l_min, l_max: min/max segment length constraints
        tau_*_mult: threshold multipliers (relative to median)

    Returns:
        breakpoints: list of boundary indices [0, τ_1, ..., T]
        events: (n_events, D) tensor of event-level states (averaged)
    """
    T = trajectory.shape[0]

    if T <= l_min:
        # Too short to segment — return as single event
        return [0, T], trajectory.mean(dim=0, keepdim=True)

    velocities, curvatures = compute_velocities_curvatures(trajectory, metric_fn)

    # Smooth
    v_smooth = moving_average(velocities, window)
    k_smooth = moving_average(curvatures, window) if len(curvatures) > 0 else curvatures

    # Adaptive thresholds based on trajectory-level statistics
    v_median = v_smooth.median().item() if len(v_smooth) > 0 else 1.0
    k_median = k_smooth.median().item() if len(k_smooth) > 0 else 1.0

    tau_v = v_median * tau_v_mult
    tau_k = k_median * tau_kappa_mult
    tau_v_low = v_median * tau_v_low_mult
    tau_k_low = k_median * tau_kappa_low_mult

    # Scan for breakpoints
    breakpoints = [0]
    last_bp = 0

    for t in range(1, T):
        gap = t - last_bp

        if gap >= l_max:
            # Force a break at max length
            breakpoints.append(t)
            last_bp = t
        elif gap >= l_min:
            # Check velocity/curvature spike with hysteresis
            v_t = v_smooth[t - 1].item() if t - 1 < len(v_smooth) else 0
            k_t = k_smooth[t - 2].item() if t - 2 < len(k_smooth) and t >= 2 else 0

            # Previous step should be "back to baseline" (hysteresis)
            v_prev = v_smooth[t - 2].item() if t - 2 >= 0 and t - 2 < len(v_smooth) else 0
            k_prev = k_smooth[t - 3].item() if t - 3 >= 0 and t - 3 < len(k_smooth) else 0

            spike = (v_t > tau_v) or (k_t > tau_k)
            baseline = (v_prev < tau_v_low) and (k_prev < tau_k_low)

            if spike and baseline:
                breakpoints.append(t)
                last_bp = t

    breakpoints.append(T)

    # Average states within each segment to get event-level states
    events = []
    for i in range(len(breakpoints) - 1):
        start, end = breakpoints[i], breakpoints[i + 1]
        event_state = trajectory[start:end].mean(dim=0)
        events.append(event_state)

    events = torch.stack(events)

    return breakpoints, events


def segment_dataset(
    trajectories: List[torch.Tensor],
    metric_fn=None,
    **kwargs,
) -> Tuple[List[torch.Tensor], List[List[int]]]:
    """
    Segment all trajectories in a dataset.

    Returns:
        event_trajectories: list of (T'_i, D) event-level trajectory tensors
        all_breakpoints: list of breakpoint lists (for analysis)
    """
    event_trajs = []
    all_bps = []

    for traj in trajectories:
        bps, events = segment_trajectory(traj, metric_fn=metric_fn, **kwargs)
        event_trajs.append(events)
        all_bps.append(bps)

    return event_trajs, all_bps
