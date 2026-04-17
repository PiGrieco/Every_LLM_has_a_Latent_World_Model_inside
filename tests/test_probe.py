"""
Tests for the discourse coherence probe.

Run with:
    python -m tests.test_probe
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.probe import (
    build_coherence_pairs,
    train_probe,
    LinearProbe,
)


def _make_cluster_articles(n_art=20, T=10, D=8, cluster_std=0.1, seed=42):
    """Tight Gaussian clusters per article. Used by the no-signal test
    and by the invariant checks (shapes/labels) — *not* by the probe
    accuracy test: the probe cannot linearly separate concat pairs by
    same-cluster-vs-different-cluster alone (detecting correlation
    between the two halves is a quadratic property)."""
    torch.manual_seed(seed)
    trajectories = []
    for _ in range(n_art):
        base = torch.randn(D) * 3.0
        traj = base.unsqueeze(0) + cluster_std * torch.randn(T, D)
        trajectories.append(traj)
    return trajectories


def _make_drift_articles(n_art=80, T=12, D=8, seed=42):
    """Articles with a *strictly positive* per-step drift in dim 0.

    For two paragraphs from the same article (positive pair), we always
    have `s_{t+1}[0] - s_t[0] ≈ 3` (by construction). For a cross-article
    pair sampled at random times, that difference is a zero-mean random
    variable with large spread (std ≈ 3·√2·T/2 ≈ 12 at T=12). A one-layer
    linear probe on the concat can therefore learn the rule "predict
    positive iff the second half advances in dim 0 by ~3" — a linear
    signal with clean margin.

    This is the construction used to validate ``train_probe`` because it
    lies in the (small) class of coherence-style tasks that are linearly
    separable on [s_t ; s_{t+1}]. We deliberately avoid the same-cluster
    vs different-cluster construction: detecting correlation between the
    two halves of a concat is a quadratic property and NO one-layer
    linear probe can achieve > chance on it."""
    torch.manual_seed(seed)
    drift = 3.0
    trajectories = []
    for _ in range(n_art):
        # All articles start at zero in dim 0 — no base variation to dilute
        # the signal. Other dims are still random.
        base = torch.zeros(D)
        base[1:] = torch.randn(D - 1)
        traj = [base]
        for _ in range(T - 1):
            s_next = traj[-1].clone()
            s_next[0] = s_next[0] + drift + abs(torch.randn(1).item()) * 0.05
            s_next[1:] = s_next[1:] + 0.3 * torch.randn(D - 1)
            traj.append(s_next)
        trajectories.append(torch.stack(traj))
    return trajectories


def test_build_coherence_pairs_shape_and_labels():
    trajs = _make_cluster_articles(n_art=8, T=6, D=5)
    inputs, labels = build_coherence_pairs(trajs, n_positive=100, n_negative=100, seed=0)

    assert inputs.shape == (200, 10), f"got {inputs.shape}"
    assert labels.shape == (200,)
    assert (labels[:100] == 1.0).all(), "first n_positive labels must be 1"
    assert (labels[100:] == 0.0).all(), "next n_negative labels must be 0"
    # No NaNs
    assert torch.isfinite(inputs).all()
    print("  [OK] build_coherence_pairs: correct shapes and label layout")


def test_positive_pairs_come_from_same_article():
    """Positive pairs must be consecutive paragraphs from the SAME article.
    We test this by making each article's states uniquely identifiable."""
    # One-hot article signature in first 4 dims, then 2 more random dims.
    n_art, T, D = 5, 4, 6
    torch.manual_seed(0)
    trajs = []
    for a in range(n_art):
        sig = torch.zeros(D)
        sig[a % 4] = 1.0  # ambiguity between 0 and 4 only happens for a=4, which is fine
        traj = sig.unsqueeze(0).expand(T, D).clone()
        traj[:, 4:] = torch.randn(T, 2)
        trajs.append(traj)

    inputs, labels = build_coherence_pairs(trajs, n_positive=200, n_negative=0, seed=0)
    # Split s_t and s_{t+1}
    s_t = inputs[:, :D]
    s_tp1 = inputs[:, D:]
    # For coherent pairs, the first D-2 dims (the "signature") must be identical.
    sig_eq = (s_t[:, :4] == s_tp1[:, :4]).all(dim=-1)
    assert sig_eq.all(), (
        f"{(~sig_eq).sum().item()}/{len(sig_eq)} positive pairs crossed "
        "articles — same-article invariant broken"
    )
    print("  [OK] Positive pairs: guaranteed same-article")


def test_negative_pairs_come_from_different_articles():
    n_art, T, D = 6, 4, 6
    torch.manual_seed(0)
    trajs = []
    for a in range(n_art):
        sig = torch.zeros(D)
        sig[a] = 1.0
        traj = sig.unsqueeze(0).expand(T, D).clone()
        traj[:, n_art:] = torch.randn(T, D - n_art)
        trajs.append(traj)

    inputs, labels = build_coherence_pairs(trajs, n_positive=0, n_negative=300, seed=0)
    s_t = inputs[:, :D]
    s_tp1 = inputs[:, D:]
    # For incoherent pairs, the signatures MUST differ in exactly one position.
    sig_eq = (s_t[:, :n_art] == s_tp1[:, :n_art]).all(dim=-1)
    assert not sig_eq.any(), (
        f"{sig_eq.sum().item()}/{len(sig_eq)} negative pairs share an "
        "article — cross-article invariant broken"
    )
    print("  [OK] Negative pairs: guaranteed cross-article")


def test_seed_reproducibility():
    trajs = _make_cluster_articles(n_art=8, T=5, D=4)
    a_in, a_lbl = build_coherence_pairs(trajs, 100, 100, seed=17)
    b_in, b_lbl = build_coherence_pairs(trajs, 100, 100, seed=17)
    assert torch.equal(a_in, b_in) and torch.equal(a_lbl, b_lbl), "seed not honoured"
    # Drive the global RNG; seed-controlled output must be unchanged.
    _ = torch.randn(1000)
    c_in, c_lbl = build_coherence_pairs(trajs, 100, 100, seed=17)
    assert torch.equal(a_in, c_in), "global RNG leaked into build_coherence_pairs"
    print("  [OK] Deterministic and isolated from global RNG")


def test_needs_minimum_articles():
    # Only one valid article → cannot build negatives
    try:
        build_coherence_pairs([torch.randn(3, 4)], 10, 10)
        raise AssertionError("should have raised ValueError")
    except ValueError:
        pass
    # Trajectories too short to produce transitions
    try:
        build_coherence_pairs([torch.randn(1, 4), torch.randn(1, 4)], 10, 10)
        raise AssertionError("should have raised ValueError")
    except ValueError:
        pass
    print("  [OK] Input validation: raises on insufficient data")


def test_probe_learns_synthetic_signal():
    """With a linearly-separable coherence signal (articles have a strictly
    positive drift in dim 0), the one-layer probe must reach > 0.6 on
    held-out validation. This is the sanity check required by the spec.

    We deliberately do NOT use same-cluster/different-cluster synthetic
    data here: detecting cluster identity from concat pairs requires a
    quadratic feature (dot product / distance) and is mathematically out
    of reach for a linear probe — which would yield val_acc ≈ 0.5 and
    misleadingly "fail" an otherwise-correct probe implementation."""
    trajs = _make_drift_articles(n_art=20, T=10, D=8)
    inputs, labels = build_coherence_pairs(trajs, 2000, 2000, seed=0)

    r = train_probe(inputs, labels, val_frac=0.2, epochs=30, seed=0)

    assert r["val_acc"] > 0.6, (
        f"val_acc={r['val_acc']:.3f} ≤ 0.6 on the strictly-positive-drift "
        f"synthetic signal — probe or data construction is broken"
    )
    assert 0.0 <= r["val_auroc"] <= 1.0
    assert 0.0 <= r["train_acc"] <= 1.0
    print(f"  [OK] Probe learns linear signal: "
          f"val_acc={r['val_acc']:.3f}, AUROC={r['val_auroc']:.3f}, "
          f"epochs_run={r['epochs_run']}")


def test_probe_chance_level_on_unlearnable_signal():
    """Sanity in the opposite direction: cluster-identity coherence is
    non-linearly-separable from concat pairs, so a linear probe must
    stay close to chance on it. This guards against the probe
    accidentally learning something spurious via memorisation of the
    small training set."""
    trajs = _make_cluster_articles(
        n_art=20, T=10, D=8, cluster_std=0.1,
    )
    inputs, labels = build_coherence_pairs(trajs, 1000, 1000, seed=0)
    r = train_probe(inputs, labels, val_frac=0.2, epochs=10, seed=0)
    # With no linearly-extractable signal, val_acc should stay well below
    # the 0.6 target that the "drift" test passes. Allow a wide margin for
    # stochastic variation.
    assert r["val_acc"] < 0.7, (
        f"val_acc={r['val_acc']:.3f} suspiciously high on a task that is "
        "not linearly separable — possible overfit or probe misuse"
    )
    print(f"  [OK] Probe stays near chance on a non-linearly-separable task: "
          f"val_acc={r['val_acc']:.3f}")


def test_linear_probe_is_actually_linear():
    """One affine layer, no hidden layer — guards against accidental MLP."""
    p = LinearProbe(in_dim=32)
    n_params = sum(x.numel() for x in p.parameters())
    assert n_params == 32 + 1, (
        f"LinearProbe has {n_params} params — expected 32 + 1 = 33. "
        "Has someone snuck a hidden layer in there?"
    )
    # Structural check: exactly one Linear module
    linears = [m for m in p.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 1
    print("  [OK] LinearProbe is one affine layer (33 params for in_dim=32)")


def main():
    print("Running coherence probe tests...\n")
    test_build_coherence_pairs_shape_and_labels()
    test_positive_pairs_come_from_same_article()
    test_negative_pairs_come_from_different_articles()
    test_seed_reproducibility()
    test_needs_minimum_articles()
    test_linear_probe_is_actually_linear()
    test_probe_learns_synthetic_signal()
    test_probe_chance_level_on_unlearnable_signal()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
