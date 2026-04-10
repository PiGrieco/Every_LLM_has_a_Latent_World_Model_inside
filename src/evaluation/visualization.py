"""
Visualization utilities for the Lorentzian World Model.

Generates the plots recommended in Section 5.5:
  1. Histogram of Δσ² (real vs negatives) — shows metric discrimination
  2. Time-reversal gap curves — primary figure for H2
  3. Branching separation — primary figure for H3
  4. Training loss curves — diagnostic
  5. Latent space visualization — qualitative

All plots are saved to the output directory and can be displayed in
Colab notebooks.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab
from typing import Optional, Dict, List
import os


def plot_interval_histogram(
    metric,
    s: torch.Tensor,
    s_next: torch.Tensor,
    s_neg: torch.Tensor,
    save_path: str = "interval_histogram.png",
    title: str = "Squared Interval Distribution",
):
    """
    Histogram of Δσ² for real transitions vs random negatives.

    Real transitions should cluster in the negative region (time-like).
    Random negatives should be more spread out or positive (space-like).
    This is the visual signature of a working Lorentzian metric.
    """
    with torch.no_grad():
        delta_real = s_next - s
        intervals_real = metric.squared_interval(s, delta_real).cpu().numpy()

        delta_neg = s_neg - s
        intervals_neg = metric.squared_interval(s, delta_neg).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(intervals_real, bins=80, alpha=0.6, label="Real transitions",
            color="#2196F3", density=True)
    ax.hist(intervals_neg, bins=80, alpha=0.6, label="Random negatives",
            color="#FF5722", density=True)

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, label="Null boundary")
    ax.set_xlabel(r"Squared interval $\Delta\sigma^2$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)

    # Annotate regions
    ax.text(-0.5, ax.get_ylim()[1] * 0.85, "← Time-like", fontsize=10, ha="center",
            color="#2196F3", weight="bold")
    ax.text(0.5, ax.get_ylim()[1] * 0.85, "Space-like →", fontsize=10, ha="center",
            color="#FF5722", weight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_time_reversal_gap(
    forward_actions: List[float],
    reversed_actions: List[float],
    save_path: str = "time_reversal_gap.png",
    title: str = "Time-Reversal Gap (H2)",
):
    """
    Compare action distributions for forward vs reversed trajectories.

    This is the PRIMARY FIGURE for testing H2 (arrow of narrative).
    Reversed trajectories should have systematically higher action.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: paired scatter
    ax = axes[0]
    ax.scatter(forward_actions, reversed_actions, alpha=0.3, s=15, color="#1976D2")
    lims = [
        min(min(forward_actions), min(reversed_actions)),
        max(max(forward_actions), max(reversed_actions)),
    ]
    ax.plot(lims, lims, "k--", linewidth=1, label="S_fwd = S_rev")
    ax.set_xlabel("Forward action $S[\\gamma]$", fontsize=12)
    ax.set_ylabel("Reversed action $S[\\gamma^R]$", fontsize=12)
    ax.set_title("Paired comparison", fontsize=13)
    ax.legend()

    # Fraction above diagonal
    n_above = sum(1 for f, r in zip(forward_actions, reversed_actions) if r > f)
    frac = n_above / len(forward_actions)
    ax.text(0.05, 0.92, f"Rev > Fwd: {frac:.1%}",
            transform=ax.transAxes, fontsize=11, weight="bold",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Right: gap distribution
    ax = axes[1]
    gaps = [r - f for f, r in zip(forward_actions, reversed_actions)]
    ax.hist(gaps, bins=50, color="#43A047", alpha=0.7, edgecolor="white")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5)
    ax.axvline(x=np.mean(gaps), color="red", linestyle="-", linewidth=2,
               label=f"Mean gap = {np.mean(gaps):.3f}")
    ax.set_xlabel("Action gap (reversed - forward)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Gap distribution", fontsize=13)
    ax.legend(fontsize=11)

    plt.suptitle(title, fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_curves(
    history: Dict[str, list],
    save_path: str = "training_curves.png",
):
    """
    Plot loss curves over training epochs, with stage boundaries marked.
    """
    epochs = history.get("epoch", [])
    if not epochs:
        return

    # Determine which losses are available
    loss_keys = [k for k in history if k not in ("epoch", "stage") and not k.startswith("eval_")]
    eval_keys = [k for k in history if k.startswith("eval_")]

    n_plots = 1 + (1 if eval_keys else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Training losses
    ax = axes[0]
    for key in loss_keys:
        vals = history[key]
        if len(vals) == len(epochs):
            ax.plot(epochs, vals, label=key, linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Losses", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.grid(True, alpha=0.3)

    # Mark stage boundary
    stages = history.get("stage", [])
    if stages:
        for i in range(1, len(stages)):
            if stages[i] != stages[i - 1]:
                ax.axvline(x=epochs[i], color="gray", linestyle=":", linewidth=2)
                ax.text(epochs[i], ax.get_ylim()[1] * 0.9,
                        f"Stage {stages[i]}", fontsize=9, ha="center")

    # Evaluation metrics
    if eval_keys:
        ax = axes[1]
        # Eval metrics are logged less frequently; we need their epoch indices
        eval_epochs = list(range(0, len(epochs), max(1, len(epochs) // len(history.get(eval_keys[0], [1])))))
        for key in eval_keys:
            vals = history[key]
            ep = eval_epochs[: len(vals)]
            ax.plot(ep, vals, "o-", label=key.replace("eval_", ""), linewidth=1.5, markersize=4)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Metric", fontsize=12)
        ax.set_title("Evaluation Metrics", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_latent_space_2d(
    trajectories: List[torch.Tensor],
    labels: Optional[List[dict]] = None,
    method: str = "pca",
    save_path: str = "latent_space.png",
    title: str = "Latent Space Visualization",
    max_trajectories: int = 50,
):
    """
    2D projection of latent trajectories using PCA or UMAP.

    Useful for qualitative assessment: do trajectories flow in a
    consistent direction? Are branches visually separated?
    """
    # Collect all states
    all_states = []
    traj_indices = []
    for i, traj in enumerate(trajectories[:max_trajectories]):
        if isinstance(traj, dict):
            traj = traj["trajectory"]
        all_states.append(traj.detach().cpu())
        traj_indices.extend([i] * len(traj))

    all_states = torch.cat(all_states).numpy()

    # Project to 2D
    if method == "pca":
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2).fit_transform(all_states)
    elif method == "umap":
        import umap
        proj = umap.UMAP(n_components=2).fit_transform(all_states)
    else:
        proj = all_states[:, :2]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.tab20

    offset = 0
    for i, traj in enumerate(trajectories[:max_trajectories]):
        if isinstance(traj, dict):
            traj = traj["trajectory"]
        T = len(traj)
        pts = proj[offset : offset + T]

        # Color by trajectory or branch
        color_idx = i
        if labels and i < len(labels) and "branch_id" in labels[i]:
            color_idx = labels[i]["branch_id"]

        color = cmap(color_idx % 20)
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, alpha=0.4, linewidth=0.8)
        ax.scatter(pts[0, 0], pts[0, 1], color=color, marker="o", s=30, zorder=5)  # Start
        ax.scatter(pts[-1, 0], pts[-1, 1], color=color, marker="s", s=30, zorder=5) # End

        offset += T

    ax.set_xlabel(f"{method.upper()} 1", fontsize=12)
    ax.set_ylabel(f"{method.upper()} 2", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
