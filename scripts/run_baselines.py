#!/usr/bin/env python3
"""
Run all geometry baselines on D0 and D1.

This script automates the systematic comparison between Lorentzian,
Riemannian, and Euclidean geometries — producing the results needed
to evaluate H1, H2, and H3.

Usage:
    python -m scripts.run_baselines

Output:
    ./outputs/baseline_comparison.json  — all results in one file
    ./outputs/d0/  — per-geometry plots for D0
    ./outputs/d1/  — per-geometry plots for D1
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import run_d0, run_d1, set_seed
from src.config import Config


def main():
    all_results = {}

    geometries = ["lorentzian", "riemannian", "euclidean"]

    # ---- D0: Time-reversal across geometries ----
    print("\n" + "#" * 60)
    print("# RUNNING D0 ACROSS ALL GEOMETRY BASELINES")
    print("#" * 60)

    d0_results = {}
    for geo in geometries:
        print(f"\n>>> D0 with {geo} geometry <<<")
        cfg = Config(
            dataset="d0_synthetic",
            geometry=geo,
            latent_dim=16,
            n_trajectories=1000,
            trajectory_length=50,
            stage2_epochs=50,
            stage3_epochs=100,
            output_dir=f"./outputs/d0",
            checkpoint_dir=f"./checkpoints/d0",
            seed=42,
        )
        set_seed(cfg.seed)
        try:
            results = run_d0(cfg)
        except Exception as e:
            print(f"  [ERROR] D0/{geo} failed: {e}")
            results = {"error": str(e)}
        d0_results[geo] = results

    all_results["d0"] = d0_results

    # ---- D1: Branching across geometries ----
    print("\n" + "#" * 60)
    print("# RUNNING D1 ACROSS ALL GEOMETRY BASELINES")
    print("#" * 60)

    d1_results = {}
    for geo in geometries:
        print(f"\n>>> D1 with {geo} geometry <<<")
        cfg = Config(
            dataset="d1_branching",
            geometry=geo,
            latent_dim=16,
            n_trajectories=500,
            prefix_length=10,
            n_branches=4,
            branch_length=20,
            stage2_epochs=50,
            stage3_epochs=100,
            output_dir=f"./outputs/d1",
            checkpoint_dir=f"./checkpoints/d1",
            seed=42,
        )
        set_seed(cfg.seed)
        try:
            results = run_d1(cfg)
        except Exception as e:
            print(f"  [ERROR] D1/{geo} failed: {e}")
            results = {"error": str(e)}
        d1_results[geo] = results

    all_results["d1"] = d1_results

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 60)

    print("\nD0 — Time-Reversal (H1 + H2):")
    print(f"{'Geometry':<15} {'M1 (timelike)':<18} {'M2 (action gap)':<18} {'M2 (logprob gap)':<18}")
    print("-" * 69)
    for geo in geometries:
        r = d0_results.get(geo, {})
        if "error" in r:
            print(f"{geo:<15} {'ERROR':<18} {r['error'][:36]}")
            continue
        m1 = r.get("m1_timelike_rate", "N/A")
        m2a = r.get("m2_action_gap", "N/A")
        m2l = r.get("m2_logprob_gap", "N/A")
        m1_s = f"{m1:.4f}" if isinstance(m1, float) else m1
        m2a_s = f"{m2a:.4f}" if isinstance(m2a, float) else m2a
        m2l_s = f"{m2l:.4f}" if isinstance(m2l, float) else m2l
        print(f"{geo:<15} {m1_s:<18} {m2a_s:<18} {m2l_s:<18}")

    print("\nD1 — Branching (H3):")
    print(f"{'Geometry':<15} {'M1 (timelike)':<18} {'M3 (branch sep)':<18}")
    print("-" * 51)
    for geo in geometries:
        r = d1_results.get(geo, {})
        if "error" in r:
            print(f"{geo:<15} {'ERROR':<18} {r['error'][:36]}")
            continue
        m1 = r.get("m1_timelike_rate", "N/A")
        m3 = r.get("m3_branching_separation", "N/A")
        m1_s = f"{m1:.4f}" if isinstance(m1, float) else m1
        m3_s = f"{m3:.4f}" if isinstance(m3, float) else m3
        print(f"{geo:<15} {m1_s:<18} {m3_s:<18}")

    # Save all results
    os.makedirs("./outputs", exist_ok=True)
    with open("./outputs/baseline_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to ./outputs/baseline_comparison.json")


if __name__ == "__main__":
    main()
