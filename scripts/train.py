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
from src.data.preprocessing import preprocess_trajectory_dataset
from src.training.trainer import WorldModelTrainer
from src.models.adapter import IdentityAdapter
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
    states, next_states, t_indices, traj_indices = extract_transitions(forward_ds)
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
            time_fn=trainer.time_fn,
        )
        print(f"M2 (action gap): {m2['action_gap']:.4f}")
        print(f"M2 (logprob gap): {m2['logprob_gap']:.4f}")
        if "delta_tau_forward" in m2:
            print(f"M2 (Δτ forward):  {m2['delta_tau_forward']:.4f}")
            print(f"M2 (Δτ reversed): {m2['delta_tau_reversed']:.4f}")

        # Decomposition (now computed inside m2_time_reversal_gap)
        if "geo_only_gap" in m2:
            print(f"M2 (geo-only Δσ² gap):  {m2['geo_only_gap']:.4f}")
        if "future_only_gap" in m2:
            print(f"M2 (Δτ-only gap):       {m2['future_only_gap']:.4f}")

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
    if "delta_tau_forward" in m2:
        results["m2_delta_tau_forward"] = m2["delta_tau_forward"]
        results["m2_delta_tau_reversed"] = m2["delta_tau_reversed"]
    if "geo_only_gap" in m2:
        results["m2_geo_only_gap"] = m2["geo_only_gap"]
    if "future_only_gap" in m2:
        results["m2_future_only_gap"] = m2["future_only_gap"]
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
    states, next_states, t_indices, traj_indices = extract_transitions(dataset)
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

        # M3': Joint branching metric (distinguishes geometries)
        from src.evaluation.metrics import m3_prime_joint_branching

        trajs_per_branch = cfg.n_trajectories // cfg.n_branches
        trajs_by_branch = {}
        for b in range(cfg.n_branches):
            trajs_by_branch[b] = []
            for i in range(min(trajs_per_branch, 20)):
                traj_idx = b * trajs_per_branch + i
                trajs_by_branch[b].append(
                    dataset[traj_idx]["trajectory"].to(device)
                )

        m3p = m3_prime_joint_branching(
            trainer.metric, trajs_by_branch, cfg.prefix_length
        )
        print(f"M3' (within time-like):    {m3p['within_timelike_rate']:.4f}")
        print(f"M3' (cross space-like):    {m3p['cross_spacelike_rate']:.4f}")
        print(f"M3' (joint score):         {m3p['joint_score']:.4f}")

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
        "m3p_within_timelike": m3p["within_timelike_rate"],
        "m3p_cross_spacelike": m3p["cross_spacelike_rate"],
        "m3p_joint": m3p["joint_score"],
        "geometry": cfg.geometry,
        "n_branches": cfg.n_branches,
    }
    with open(os.path.join(cfg.output_dir, f"d1_results_{cfg.geometry}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {cfg.output_dir}/")
    return results


def run_d2(cfg):
    """
    D2: WikiText-103 real-text experiment.
    Tests the actual thesis: does Lorentzian geometry extract meaningful
    structure from a real LM's implicit world model?
    """
    import torch
    from pathlib import Path
    from src.data.preprocessing import preprocess_trajectory_dataset
    from src.training.trainer import WorldModelTrainer
    from src.evaluation.metrics import m1_timelike_rate, m5_predictive_nll

    print("\n" + "=" * 60)
    print("D2: WIKITEXT-103 REAL-TEXT EXPERIMENT")
    print(f"Geometry: {cfg.geometry} | D={cfg.latent_dim}")
    print("=" * 60)

    cache_dir = Path(cfg.cache_dir)

    # Load cached embeddings
    emb_path = cache_dir / "wikitext_embeddings.pt"
    if not emb_path.exists():
        print(f"ERROR: Run encode_corpus.py first")
        return None

    print("Loading cached embeddings...")
    emb_data = torch.load(emb_path, weights_only=False)
    embeddings = emb_data["embeddings"]
    print(f"  {len(embeddings)} articles, {sum(len(e) for e in embeddings)} paragraphs")

    # Load cached LM scores
    lm_path = cache_dir / "wikitext_lm_scores.pt"
    has_lm = lm_path.exists()
    if has_lm:
        print("Loading cached LM scores...")
        lm_data = torch.load(lm_path, weights_only=False)
        lm_scores = lm_data["log_probs"]
        print(f"  {sum(len(s) for s in lm_scores)} transition scores")
    elif cfg.lambda_sem > 0:
        raise FileNotFoundError(
            f"Config has lambda_sem={cfg.lambda_sem} but {lm_path} not found.\n"
            f"Run 'python -m scripts.encode_corpus --config configs/d2_wikitext.yaml' "
            f"first, or pass --no-semantic to train without the LM term."
        )
    else:
        print("No LM scores and lambda_sem=0 — running without semantic term")
        lm_scores = None

    # Article-level split for non-circular evaluation:
    # M4 is computed on articles never seen during training.
    split_path = cache_dir / "wikitext_split.pt"
    if split_path.exists():
        split_data = torch.load(split_path, weights_only=False)
        train_arts = set(split_data["train_indices"])
        eval_arts = set(split_data["eval_indices"])
        print(f"  Using article-level split: {len(train_arts)} train / {len(eval_arts)} eval articles")
    else:
        n_art = len(embeddings)
        perm_art = torch.randperm(n_art).tolist()
        n_train_art = int(0.8 * n_art)
        train_arts = set(perm_art[:n_train_art])
        eval_arts = set(perm_art[n_train_art:])
        print(f"  No split file — created random split: {len(train_arts)}/{len(eval_arts)}")

    # Preprocess AFTER split so the top PCs are fit only on train articles
    # (otherwise eval statistics leak into the centering / PC removal step).
    print("Preprocessing embeddings (fit on train articles only)...")
    processed, _ = preprocess_trajectory_dataset(
        embeddings,
        n_pca_remove=cfg.n_pca_remove,
        normalize=cfg.normalize_embeddings,
        fit_indices=sorted(train_arts),
    )

    def _extract_transitions(indices):
        ss, sn, ls = [], [], []
        for i in indices:
            if i >= len(processed):
                continue
            emb = processed[i]
            if len(emb) < 2:
                continue
            ss.append(emb[:-1])
            sn.append(emb[1:])
            if lm_scores is not None and i < len(lm_scores):
                ls.append(lm_scores[i])
            else:
                ls.append(torch.zeros(len(emb) - 1))
        if not ss:
            return None, None, None
        return torch.cat(ss), torch.cat(sn), torch.cat(ls)

    train_s, train_sn, train_lsem = _extract_transitions(sorted(train_arts))
    eval_s, eval_sn, _ = _extract_transitions(sorted(eval_arts))
    print(f"  Train transitions: {len(train_s)} | Eval transitions: {len(eval_s)}")

    cfg.encoder_dim = train_s.shape[1]

    # Train
    trainer = WorldModelTrainer(cfg)
    history = trainer.train(train_s, train_sn, lsem=train_lsem,
                            eval_states=eval_s, eval_next_states=eval_sn)

    # Evaluate on held-out ARTICLES (not just held-out transitions)
    # This makes M4 non-circular: the eval articles were never seen during training.
    device = trainer.device
    trainer.metric.eval()
    trainer.world_model.eval()
    trainer.adapter.eval()

    with torch.no_grad():
        es_lat = trainer._to_latent(eval_s)
        esn_lat = trainer._to_latent(eval_sn)

        m1 = m1_timelike_rate(trainer.metric, es_lat, esn_lat)
        m5 = m5_predictive_nll(trainer.world_model, es_lat, esn_lat)

        # ---- M4 raw cone alignment on held-out articles ----
        # Tautological for Euclidean/Riemannian where squared_interval ≥ 0,
        # but kept as a diagnostic for the Lorentzian case.
        from src.training.candidates import build_candidate_set_c1
        from src.evaluation.metrics import m4_cone_alignment, m4_fair
        n_ev = min(256, len(es_lat))
        cands, _ = build_candidate_set_c1(es_lat[:n_ev], esn_lat[:n_ev], 32)
        m4 = m4_cone_alignment(trainer.metric, trainer.lagrangian,
                               es_lat[:n_ev], cands)

        # ---- M4 fair with shared base_rate across geometries ----
        # Protocol: Lorentzian runs first and self-calibrates its base_rate,
        # which is persisted to disk; Riemannian/Euclidean load and reuse it
        # so all three classify the same FRACTION of pairs as "reachable",
        # making M4 differences reflect directional structure (cone vs ball)
        # rather than threshold tuning.
        base_rate_path = Path(cfg.output_dir) / "d2_base_rate.json"
        perm = torch.randperm(n_ev, device=es_lat.device)
        s_neg_eval = esn_lat[:n_ev][perm]

        if cfg.geometry == "lorentzian":
            m4f = m4_fair(
                trainer.metric, es_lat[:n_ev], esn_lat[:n_ev], s_neg_eval,
                geometry="lorentzian", base_rate=None,
            )
            os.makedirs(cfg.output_dir, exist_ok=True)
            with open(base_rate_path, "w") as f:
                json.dump({"base_rate": m4f["m4f_base_rate"]}, f, indent=2)
            print(f"  [M4-fair] Lorentzian auto-calibrated base_rate="
                  f"{m4f['m4f_base_rate']:.4f} → saved to {base_rate_path}")
        else:
            if base_rate_path.exists():
                with open(base_rate_path) as f:
                    shared_rate = json.load(f)["base_rate"]
                print(f"  [M4-fair] Loaded Lorentzian base_rate="
                      f"{shared_rate:.4f} for fair comparison")
            else:
                print(f"  [M4-fair] [WARN] {base_rate_path} not found. "
                      f"Run Lorentzian first for fair comparison. "
                      f"Falling back to base_rate=0.5.")
                shared_rate = 0.5
            m4f = m4_fair(
                trainer.metric, es_lat[:n_ev], esn_lat[:n_ev], s_neg_eval,
                geometry=cfg.geometry, base_rate=shared_rate,
            )

    print(f"\n--- D2 Results ({cfg.geometry}) ---")
    print(f"M1 (time-likeness rate):        {m1:.4f}")
    print(f"M4 raw (cone Jaccard):          {m4['jaccard']:.4f}  [diagnostic]")
    print(f"M4 raw (cone precision):        {m4['precision']:.4f}")
    print(f"M4 raw (cone recall):           {m4['recall']:.4f}")
    print(f"M4 fair (Jaccard):              {m4f['m4f_jaccard']:.4f}  "
          f"[base_rate={m4f['m4f_base_rate']:.4f}]")
    print(f"M4 fair (precision):            {m4f['m4f_precision']:.4f}")
    print(f"M4 fair (recall):               {m4f['m4f_recall']:.4f}")
    print(f"M5 (predictive NLL):            {m5:.4f}")

    # Save
    os.makedirs(cfg.output_dir, exist_ok=True)
    results = {
        "m1": m1,
        "m4raw_jaccard": m4["jaccard"],
        "m4raw_precision": m4["precision"],
        "m4raw_recall": m4["recall"],
        "m4fair_jaccard": m4f["m4f_jaccard"],
        "m4fair_precision": m4f["m4f_precision"],
        "m4fair_recall": m4f["m4f_recall"],
        "m4fair_base_rate": m4f["m4f_base_rate"],
        "m5": m5,
        "geometry": cfg.geometry,
        "has_semantic": has_lm,
        "eval_on_held_out_articles": True,
    }
    with open(os.path.join(cfg.output_dir, f"d2_results_{cfg.geometry}.json"), "w") as f:
        json.dump(results, f, indent=2)

    from src.evaluation.visualization import plot_training_curves, plot_interval_histogram
    plot_training_curves(history,
        save_path=os.path.join(cfg.output_dir, f"d2_curves_{cfg.geometry}.png"))
    neg_perm = torch.randperm(len(esn_lat))
    plot_interval_histogram(trainer.metric,
        es_lat[:500], esn_lat[:500], esn_lat[neg_perm[:500]],
        save_path=os.path.join(cfg.output_dir, f"d2_intervals_{cfg.geometry}.png"),
        title=f"D2 WikiText Intervals ({cfg.geometry})")

    trainer.save_checkpoint(os.path.join(cfg.checkpoint_dir, f"d2_{cfg.geometry}.pt"))
    print(f"Results saved to {cfg.output_dir}/")
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
    parser.add_argument("--no-semantic", action="store_true",
                        help="Disable semantic term (set lambda_sem=0) for D2")

    args = parser.parse_args()

    # Build config — CLI args override YAML when explicitly provided
    if args.config:
        cfg = Config.from_yaml(args.config)
        if '--geometry' in sys.argv:
            cfg.geometry = args.geometry
        if '--dataset' in sys.argv:
            cfg.dataset = args.dataset
        if '--latent_dim' in sys.argv:
            cfg.latent_dim = args.latent_dim
        if '--stage2_epochs' in sys.argv:
            cfg.stage2_epochs = args.stage2_epochs
        if '--stage3_epochs' in sys.argv:
            cfg.stage3_epochs = args.stage3_epochs
        if '--seed' in sys.argv:
            cfg.seed = args.seed
        if '--output_dir' in sys.argv:
            cfg.output_dir = args.output_dir
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
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
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

    if getattr(args, 'no_semantic', False):
        cfg.lambda_sem = 0.0
        print("  --no-semantic: lambda_sem forced to 0.0")

    set_seed(cfg.seed)

    # Dispatch to the right experiment
    if cfg.dataset == "d0_synthetic":
        run_d0(cfg)
    elif cfg.dataset == "d1_branching":
        run_d1(cfg)
    elif cfg.dataset == "d2_wikitext":
        run_d2(cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")


if __name__ == "__main__":
    main()
