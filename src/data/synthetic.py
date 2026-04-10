"""
Synthetic dataset generators for D0 and D1.

D0 (Time-reversal): Drift-diffusion trajectories in R^D with a built-in
    arrow of time. The first coordinate drifts monotonically, ensuring
    irreversibility. Used to validate that the Lorentzian metric can
    distinguish forward from reversed sequences.

D1 (Branching): Trajectories share a common prefix, then diverge into
    incompatible branches. Used to validate that alternative continuations
    are separated by space-like intervals (H3).

Both generators produce data directly in R^D, bypassing the encoder
pipeline entirely. This isolates the geometric machinery for debugging.
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional


class SyntheticTrajectoryDataset(Dataset):
    """
    Base dataset: each item is a trajectory (T, D) tensor and metadata.
    """

    def __init__(self, trajectories: List[torch.Tensor], labels: Optional[List[dict]] = None):
        self.trajectories = trajectories
        self.labels = labels or [{}] * len(trajectories)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return {
            "trajectory": self.trajectories[idx],  # (T, D)
            "labels": self.labels[idx],
        }


def generate_d0(
    n_trajectories: int = 1000,
    trajectory_length: int = 50,
    dim: int = 16,
    drift_strength: float = 0.5,
    noise_std: float = 0.1,
    seed: int = 42,
) -> Tuple[SyntheticTrajectoryDataset, SyntheticTrajectoryDataset]:
    """
    Generate D0: drift-diffusion trajectories with a clear arrow of time.

    The drift vector is concentrated on the first coordinate, so the first
    axis acts as "narrative time". The remaining coordinates receive only
    noise, acting as "semantic space". This mimics the structure we want
    the Lorentzian metric to discover: one time-like direction (the drift)
    and D-1 space-like directions (the noise).

    Returns:
        forward_dataset: trajectories in natural (forward) order
        reversed_dataset: the same trajectories in reversed order
    """
    torch.manual_seed(seed)

    # Drift vector: strong in first component, zero elsewhere
    # This ensures irreversibility — the first coordinate always increases
    drift = torch.zeros(dim)
    drift[0] = drift_strength

    forward_trajs = []
    reversed_trajs = []
    forward_labels = []
    reversed_labels = []

    for i in range(n_trajectories):
        # Start from a random point (but with first coordinate near 0)
        s0 = torch.randn(dim) * 0.3
        s0[0] = torch.randn(1).item() * 0.1  # Start near zero on time axis

        # Build trajectory: s_{t+1} = s_t + drift + noise
        traj = [s0]
        for t in range(trajectory_length - 1):
            noise = torch.randn(dim) * noise_std
            s_next = traj[-1] + drift + noise
            traj.append(s_next)

        forward = torch.stack(traj)   # (T, D)
        reversed_ = forward.flip(0)   # Time-reversed

        forward_trajs.append(forward)
        reversed_trajs.append(reversed_)
        forward_labels.append({"direction": "forward", "idx": i})
        reversed_labels.append({"direction": "reversed", "idx": i})

    return (
        SyntheticTrajectoryDataset(forward_trajs, forward_labels),
        SyntheticTrajectoryDataset(reversed_trajs, reversed_labels),
    )


def generate_d1(
    n_trajectories: int = 500,
    prefix_length: int = 10,
    n_branches: int = 4,
    branch_length: int = 20,
    dim: int = 16,
    drift_strength: float = 0.5,
    noise_std: float = 0.1,
    branch_spread: float = 1.0,
    seed: int = 42,
) -> Tuple[SyntheticTrajectoryDataset, dict]:
    """
    Generate D1: branching trajectories.

    All trajectories share a common prefix (same drift direction), then at
    the branch point, each trajectory follows one of n_branches divergent
    drift directions. The branch directions are chosen to be orthogonal
    in the spatial dimensions (components 1..D-1), so they're genuinely
    incompatible continuations.

    The "time" drift (component 0) continues forward in all branches, so
    all branches remain time-like. The difference between branches is
    purely in the spatial/semantic directions — exactly the structure
    that the Lorentzian metric should classify as "space-like separation."

    Returns:
        dataset: all trajectories (each includes prefix + one branch)
        branch_info: dict with branch_point_idx, branch_assignments, etc.
    """
    torch.manual_seed(seed)

    # Common drift for the prefix (time direction)
    time_drift = torch.zeros(dim)
    time_drift[0] = drift_strength

    # Branch-specific drift directions (in spatial dimensions only)
    # We create n_branches directions that are roughly orthogonal
    # by sampling random vectors in dimensions 1..D-1 and normalizing
    branch_drifts = []
    for b in range(n_branches):
        d = torch.zeros(dim)
        d[0] = drift_strength  # All branches still move forward in time
        # Each branch gets a strong push in a different spatial direction
        spatial_dir = 1 + (b % (dim - 1))
        d[spatial_dir] = branch_spread
        # Add some randomness to make it realistic
        d[1:] += torch.randn(dim - 1) * 0.1
        branch_drifts.append(d)

    trajectories = []
    labels = []

    # How many trajectories per branch (distribute evenly)
    trajs_per_branch = n_trajectories // n_branches

    for b in range(n_branches):
        for i in range(trajs_per_branch):
            # Generate shared prefix
            s0 = torch.randn(dim) * 0.1
            traj = [s0]
            for t in range(prefix_length - 1):
                noise = torch.randn(dim) * noise_std
                s_next = traj[-1] + time_drift + noise
                traj.append(s_next)

            # Branch point: continue with branch-specific drift
            for t in range(branch_length):
                noise = torch.randn(dim) * noise_std
                s_next = traj[-1] + branch_drifts[b] + noise
                traj.append(s_next)

            full_traj = torch.stack(traj)  # (prefix_length + branch_length, D)
            trajectories.append(full_traj)
            labels.append({
                "branch_id": b,
                "branch_point": prefix_length - 1,
                "traj_idx": i,
            })

    branch_info = {
        "n_branches": n_branches,
        "prefix_length": prefix_length,
        "branch_length": branch_length,
        "branch_drifts": torch.stack(branch_drifts),
    }

    return SyntheticTrajectoryDataset(trajectories, labels), branch_info


def extract_transitions(dataset: SyntheticTrajectoryDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract all consecutive (s_t, s_{t+1}) pairs from a trajectory dataset.

    Returns:
        states: (N_total, D) tensor of current states
        next_states: (N_total, D) tensor of next states
    """
    all_s = []
    all_s_next = []

    for item in dataset:
        traj = item["trajectory"]  # (T, D)
        all_s.append(traj[:-1])
        all_s_next.append(traj[1:])

    return torch.cat(all_s, dim=0), torch.cat(all_s_next, dim=0)
