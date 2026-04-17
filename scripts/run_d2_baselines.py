#!/usr/bin/env python3
"""
Run all three geometries on D2 with shared split and base_rate coupling.

Prerequisite: run scripts/encode_corpus.py first to populate
``./cache/wikitext_embeddings.pt`` (and, if lambda_sem > 0,
``./cache/wikitext_lm_scores.pt``).

Execution order matters: Lorentzian MUST run first so it can
self-calibrate the ``base_rate`` and persist
``./outputs/d2/d2_base_rate.json``. Riemannian and Euclidean then load
that value for a fair M4 comparison. We also wipe any stale
``d2_base_rate.json`` before starting to avoid using a coupling from a
previous, potentially inconsistent run.
"""
import os
import sys
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import run_d2, set_seed
from src.config import Config


def _build_cfg(geometry: str, device: str) -> Config:
    """D2 hyper-parameters shared across the three geometries.

    Prefer the YAML file so any later tweak propagates to all three runs;
    fall back to an explicit dataclass construction if for any reason the
    YAML cannot be parsed.
    """
    yaml_path = "configs/d2_wikitext.yaml"
    if os.path.exists(yaml_path):
        cfg = Config.from_yaml(yaml_path)
    else:
        cfg = Config(
            dataset="d2_wikitext",
            latent_dim=16,
            batch_size=64,
            lr=5e-4,
            weight_decay=1e-5,
            stage2_epochs=80,
            stage3_epochs=150,
            candidate_strategy="c1c2",
            candidate_set_size=64,
            lambda_g=1.0,
            lambda_sem=0.5,
            lambda_sem_reg=0.1,
            lambda_match=1.0,
            lambda_smooth=0.1,
            lambda_wm=1.0,
            lambda_geo=0.5,
            lambda_future=0.0,
            n_pca_remove=3,
            normalize_embeddings=True,
            output_dir="./outputs/d2",
            checkpoint_dir="./checkpoints/d2",
            cache_dir="./cache",
            seed=42,
        )
    # Geometry + device are always runner-controlled
    cfg.geometry = geometry
    cfg.device = device
    cfg.dataset = "d2_wikitext"
    return cfg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    geometries = ["lorentzian", "riemannian", "euclidean"]

    # Wipe any stale base_rate file to prevent a split_hash mismatch or an
    # accidental coupling to a previous Lorentzian run.
    base_rate_path = "./outputs/d2/d2_base_rate.json"
    if os.path.exists(base_rate_path):
        print(f"Removing stale {base_rate_path}")
        os.remove(base_rate_path)

    all_results = {}

    for geo in geometries:
        print("\n" + "#" * 60)
        print(f"# D2 — {geo.upper()}")
        print("#" * 60)

        cfg = _build_cfg(geo, device)
        set_seed(cfg.seed)
        try:
            results = run_d2(cfg)
        except Exception as e:
            print(f"  [ERROR] D2/{geo} failed: {e}")
            import traceback
            traceback.print_exc()
            results = {"error": str(e)}
            if geo == "lorentzian":
                print("  Lorentzian failed — aborting subsequent runs "
                      "(no base_rate to propagate).")
                all_results[geo] = results
                break
        all_results[geo] = results

    os.makedirs("./outputs/d2", exist_ok=True)
    with open("./outputs/d2/d2_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Summary ===")
    for geo, r in all_results.items():
        if not isinstance(r, dict) or "error" in r:
            err = r.get("error", "") if isinstance(r, dict) else r
            print(f"  {geo}: ERROR ({err})")
            continue

        def _fmt(v, prec=3):
            return f"{v:.{prec}f}" if isinstance(v, (int, float)) else str(v)

        print(
            f"  {geo:<12} m4f_jaccard={_fmt(r.get('m4f_jaccard', 'NA'))} "
            f"m5d_top1={_fmt(r.get('m5d_top1', 'NA'))} "
            f"m5d_mrr={_fmt(r.get('m5d_mrr', 'NA'))} "
            f"wm_collapsed={r.get('wm_collapsed', 'NA')}"
        )


if __name__ == "__main__":
    main()
