#!/usr/bin/env python3
"""
Main training script for the Lorentzian World Model.

Usage:
    # D0 with Lorentzian geometry (default)
    python -m scripts.train --dataset d0_synthetic --geometry lorentzian

    # D0 with Riemannian baseline
    python -m scripts.train --dataset d0_synthetic --geometry riemannian

    # D1 branching
    python -m scripts.train --dataset d1_branching --geometry lorentzian

    # D2 real text (requires pre-encoded data)
    python -m scripts.train --dataset d2_wikitext --geometry lorentzian

    # From config file
    python -m scripts.train --config configs/d0_synthetic.yaml
"""

import argparse
import os
import sys
import json
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.synthetic import generate_d0, generate_d1, extract_transitions
from src.training.trainer import WorldModelTrainer
from src.evaluation.metrics import m2_time_reversal_gap, m3_branching_separation
from src.evaluation.visualization import (
    plot_interval_histogram,
    plot_time_reversal_gap,
    plot_training_curves,
    plot_latent_space_2d,
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_d0(cfg: Config):
    """
    Run D0: synthetic time-reversal experiment.

    This is the debugging harness. If the Lorentzian metric can't
    distinguish forward from reversed drift-diffusion trajectories,
    there's a bug in the pipeline.
    """
    print("\n" + "=" * 60)
    print("D0: SYNTHETIC TIME-REVERSAL EXPERIMENT")
    print(f"Geometry: {cfg.geometry} | D={cfg.latent_dim}")
    print("=" * 60)

    # Generate data
    forward_ds, reversed_ds = generate_d0(
        n_trajectories=cfg.n_trajectories,
        trajectory_length=cfg.trajectory_length,
        dim=cfg.latent_dim,
        drift_strength=cfg.drift_strength,
        noise_std=cfg.noise_std,
        seed=cfg.seed,
    )

    # Extract transitions for training
    states, next_states = extract_transitions(forward_ds)
    print(f"Training transitions: {len(states)}")

    # Train/eval split (80/20)
    n = len(states)
    n_train = int(0.8 * n)
    train_s, eval_s = states[:n_train], states[n_train:]
    train_sn, eval_sn = next_states[:n_train], next_states[n_train:]

    # Train
    trainer = WorldModelTrainer(cfg)
    history = trainer.train(
        train_s, train_sn,
        eval_states=eval_s,
        eval_next_states=eval_sn,
    )

    # ---- Full evaluation ----
    device = trainer.device
    trainer.metric.eval()
    trainer.world_model.eval()
    trainer.lagrangian.eval()

    with torch.no_grad():
        # M1: Time-likeness on eval set
        from src.evaluation.metrics import m1_timelike_rate
        m1 = m1_timelike_rate(trainer.metric, eval_s.to(device), eval_sn.to(device))
        print(f"\nM1 (time-likeness rate): {m1:.4f}")

        # M2: Time-reversal gap
        # Use a subset of trajectories for evaluation
        n_eval = min(200, len(forward_ds))
        fwd_trajs = [forward_ds[i]["trajectory"].to(device) for i in range(n_eval)]
        rev_trajs = [reversed_ds[i]["trajectory"].to(device) for i in range(n_eval)]

        m2 = m2_time_reversal_gap(
            trainer.metric, trainer.lagrangian, trainer.world_model,
            fwd_trajs, rev_trajs,
        )
        print(f"M2 (action gap): {m2['action_gap']:.4f}")
        print(f"M2 (logprob gap): {m2['logprob_gap']:.4f}")

    # ---- Plots ----
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Interval histogram
    neg_idx = torch.randperm(len(eval_s))[:len(eval_s)]
    plot_interval_histogram(
        trainer.metric,
        eval_s[:500].to(device),
        eval_sn[:500].to(device),
        eval_sn[neg_idx[:500]].to(device),  # Shuffled negatives
        save_path=os.path.join(cfg.output_dir, f"d0_intervals_{cfg.geometry}.png"),
        title=f"D0 Interval Distribution ({cfg.geometry})",
    )

    # Time-reversal gap
    fwd_actions = [trainer.lagrangian.action(t).item() for t in fwd_trajs[:200]]
    rev_actions = [trainer.lagrangian.action(t).item() for t in rev_trajs[:200]]
    plot_time_reversal_gap(
        fwd_actions, rev_actions,
        save_path=os.path.join(cfg.output_dir, f"d0_reversal_{cfg.geometry}.png"),
        title=f"D0 Time-Reversal Gap ({cfg.geometry})",
    )

    # Training curves
    plot_training_curves(
        history,
        save_path=os.path.join(cfg.output_dir, f"d0_curves_{cfg.geometry}.png"),
    )

    # Latent space
    plot_latent_space_2d(
        [forward_ds[i]["trajectory"] for i in range(30)],
        method="pca",
        save_path=os.path.join(cfg.output_dir, f"d0_latent_{cfg.geometry}.png"),
        title=f"D0 Latent Trajectories ({cfg.geometry})",
    )

    # Save checkpoint
    trainer.save_checkpoint(
        os.path.join(cfg.checkpoint_dir, f"d0_{cfg.geometry}.pt")
    )

    # Save results
    results = {
        "m1_timelike_rate": m1,
        "m2_action_gap": m2["action_gap"],
        "m2_logprob_gap": m2["logprob_gap"],
        "geometry": cfg.geometry,
        "latent_dim": cfg.latent_dim,
    }
    with open(os.path.join(cfg.output_dir, f"d0_results_{cfg.geometry}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {cfg.output_dir}/")
    return results


def run_d1(cfg: Config):
    """
    Run D1: synthetic branching experiment.

    Tests H3: are alternative continuations from the same prefix
    separated by space-like intervals?
    """
    print("\n" + "=" * 60)
    print("D1: SYNTHETIC BRANCHING EXPERIMENT")
    print(f"Geometry: {cfg.geometry} | D={cfg.latent_dim} | Branches={cfg.n_branches}")
    print("=" * 60)

    # Generate data
    dataset, branch_info = generate_d1(
        n_trajectories=cfg.n_trajectories,
        prefix_length=cfg.prefix_length,
        n_branches=cfg.n_branches,
        branch_length=cfg.branch_length,
        dim=cfg.latent_dim,
        drift_strength=cfg.drift_strength,
        noise_std=cfg.noise_std,
        seed=cfg.seed,
    )

    # Extract transitions
    states, next_states = extract_transitions(dataset)
    print(f"Training transitions: {len(states)}")

    # Train/eval split
    n = len(states)
    n_train = int(0.8 * n)
    train_s, eval_s = states[:n_train], states[n_train:]
    train_sn, eval_sn = next_states[:n_train], next_states[n_train:]

    # Train
    trainer = WorldModelTrainer(cfg)
    history = trainer.train(
        train_s, train_sn,
        eval_states=eval_s,
        eval_next_states=eval_sn,
    )

    # ---- M3: Branching separation ----
    device = trainer.device
    trainer.metric.eval()

    with torch.no_grad():
        # M1
        from src.evaluation.metrics import m1_timelike_rate
        m1 = m1_timelike_rate(trainer.metric, eval_s.to(device), eval_sn.to(device))
        print(f"\nM1 (time-likeness rate): {m1:.4f}")

        # For M3: collect branch states right after the branch point
        branch_point = cfg.prefix_length  # Index of the branch point
        trajs_per_branch = cfg.n_trajectories // cfg.n_branches

        # Get the state at branch_point + 1 for each branch
        # (the first post-branch state)
        branch_eval_data = []
        n_samples = min(20, trajs_per_branch)
        for sample_idx in range(n_samples):
            branch_states = []
            for b in range(cfg.n_branches):
                traj_idx = b * trajs_per_branch + sample_idx
                traj = dataset[traj_idx]["trajectory"].to(device)
                # State right after branch
                branch_states.append(traj[branch_point + 1])

            # Prefix state (same for all branches, approximately)
            prefix_state = dataset[sample_idx]["trajectory"][branch_point - 1].to(device)
            branch_eval_data.append((prefix_state, branch_states))

        m3_rates = []
        for prefix_s, branches in branch_eval_data:
            rate = m3_branching_separation(trainer.metric, branches, prefix_s)
            m3_rates.append(rate)
        m3 = np.mean(m3_rates)
        print(f"M3 (branching separation): {m3:.4f}")

    # ---- Plots ----
    os.makedirs(cfg.output_dir, exist_ok=True)

    plot_training_curves(
        history,
        save_path=os.path.join(cfg.output_dir, f"d1_curves_{cfg.geometry}.png"),
    )

    # Latent space with branch coloring
    plot_latent_space_2d(
        [dataset[i]["trajectory"] for i in range(min(40, len(dataset)))],
        labels=[dataset[i]["labels"] for i in range(min(40, len(dataset)))],
        method="pca",
        save_path=os.path.join(cfg.output_dir, f"d1_latent_{cfg.geometry}.png"),
        title=f"D1 Branching Trajectories ({cfg.geometry})",
    )

    # Save
    trainer.save_checkpoint(
        os.path.join(cfg.checkpoint_dir, f"d1_{cfg.geometry}.pt")
    )

    results = {
        "m1_timelike_rate": m1,
        "m3_branching_separation": m3,
        "geometry": cfg.geometry,
        "n_branches": cfg.n_branches,
    }
    with open(os.path.join(cfg.output_dir, f"d1_results_{cfg.geometry}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {cfg.output_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Lorentzian World Model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--dataset", type=str, default="d0_synthetic",
                        choices=["d0_synthetic", "d1_branching", "d2_wikitext"])
    parser.add_argument("--geometry", type=str, default="lorentzian",
                        choices=["lorentzian", "riemannian", "euclidean"])
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--stage2_epochs", type=int, default=50)
    parser.add_argument("--stage3_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs")

    args = parser.parse_args()

    # Build config
    if args.config:
        cfg = Config.from_yaml(args.config)
    else:
        cfg = Config(
            dataset=args.dataset,
            geometry=args.geometry,
            latent_dim=args.latent_dim,
            stage2_epochs=args.stage2_epochs,
            stage3_epochs=args.stage3_epochs,
            seed=args.seed,
            output_dir=args.output_dir,
        )

    # Set device
    if torch.cuda.is_available():
        cfg.device = "cuda"
        gpu_name = torch.cuda.get_device_name()
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"Using GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")

        # Enable TF32 for Ampere+ GPUs (free ~2-3x speedup on matmuls)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Auto-scale batch size for large GPUs on synthetic datasets
        if cfg.dataset in ("d0_synthetic", "d1_branching") and gpu_mem_gb > 20:
            cfg.batch_size = max(cfg.batch_size, 2048)
            cfg.n_trajectories = max(cfg.n_trajectories, 5000)
            print(f"  Auto-scaled: batch_size={cfg.batch_size}, n_trajectories={cfg.n_trajectories}")
    else:
        cfg.device = "cpu"
        print("Using CPU (training will be slower)")

    set_seed(cfg.seed)

    # Dispatch to the right experiment
    if cfg.dataset == "d0_synthetic":
        run_d0(cfg)
    elif cfg.dataset == "d1_branching":
        run_d1(cfg)
    elif cfg.dataset == "d2_wikitext":
        print("D2 (WikiText) requires pre-encoding. Run encode_corpus.py first.")
        print("Full D2 pipeline coming in next iteration.")
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")


if __name__ == "__main__":
    main()
