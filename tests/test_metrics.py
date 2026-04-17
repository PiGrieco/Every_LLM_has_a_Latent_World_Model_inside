"""
Tests for the discrete M5 metric (m5_discrete_rank).

Run with:
    python -m tests.test_metrics
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import (
    m5_discrete_rank,
    m5_predictive_nll,
    compute_all_metrics,
)
from src.models.metric import MetricNetwork
from src.models.lagrangian import Lagrangian
from src.models.world_model import ConditionalGaussianWorldModel


def _mk_stack(dim: int = 8, geometry: str = "lorentzian"):
    torch.manual_seed(0)
    metric = MetricNetwork(dim=dim, hidden_dim=16, n_layers=1, geometry=geometry)
    lagrangian = Lagrangian(metric=metric, lambda_g=1.0, lambda_sem=0.0,
                            latent_dim=dim)
    world_model = ConditionalGaussianWorldModel(
        dim=dim, hidden_dim=32, n_layers=2
    )
    return metric, lagrangian, world_model


def _mk_data(dim: int = 8, n: int = 64):
    torch.manual_seed(1)
    s = torch.randn(n, dim)
    s_next = s + 0.3 * torch.randn(n, dim)
    return s, s_next


def test_basic_ranges():
    """Every reported quantity must lie in its mathematically valid range."""
    D, B, C = 8, 64, 32
    metric, lag, wm = _mk_stack(dim=D)
    s, s_next = _mk_data(dim=D, n=B)

    r = m5_discrete_rank(wm, lag, s, s_next, candidate_size=C, seed=0)

    for prefix in ("q_", "K_"):
        t1 = r[prefix + "top1_acc"]
        t5 = r[prefix + "top5_acc"]
        mr = r[prefix + "mean_rank"]
        nll = r[prefix + "nll_on_set"]

        assert 0.0 <= t1 <= 1.0, f"{prefix}top1_acc={t1} out of [0,1]"
        assert 0.0 <= t5 <= 1.0, f"{prefix}top5_acc={t5} out of [0,1]"
        # top1 ⊆ top5 ⇒ top1_acc ≤ top5_acc (within fp noise)
        assert t1 <= t5 + 1e-6, f"{prefix}top1 > top5 ({t1} > {t5})"
        assert 1.0 <= mr <= float(C), f"{prefix}mean_rank={mr} not in [1,{C}]"
        assert nll >= 0.0, f"{prefix}nll_on_set={nll} must be ≥ 0"

    print("  [OK] Ranges: top-k in [0,1], top1 ≤ top5, mean_rank in [1,C], nll ≥ 0")


def test_deterministic_given_seed():
    """Same seed → same candidates → same metrics. Different seed → may differ."""
    D, B, C = 8, 32, 16
    metric, lag, wm = _mk_stack(dim=D)
    s, s_next = _mk_data(dim=D, n=B)

    r_a = m5_discrete_rank(wm, lag, s, s_next, candidate_size=C, seed=7)
    r_b = m5_discrete_rank(wm, lag, s, s_next, candidate_size=C, seed=7)
    for k in r_a:
        assert r_a[k] == r_b[k], f"seed=7 not deterministic on {k}"

    # Drive the global RNG to make sure fork_rng actually isolates us
    _ = torch.randn(1000)
    r_c = m5_discrete_rank(wm, lag, s, s_next, candidate_size=C, seed=7)
    for k in r_a:
        assert r_a[k] == r_c[k], (
            f"m5_discrete_rank polluted by global RNG (differs on {k})"
        )

    print("  [OK] Deterministic + isolated from global RNG")


def test_variance_collapse_still_ranks():
    """The thesis of the metric: with σ² pinned at the clamp, the
    continuous NLL becomes extreme (sign depends on whether the mean
    happens to land near s_next or not), while the discrete M5 stays
    inside its natural ranges — the clamp cannot dominate a softmax
    over a fixed-size candidate set."""
    D, B, C = 8, 64, 32
    metric, lag, wm = _mk_stack(dim=D)
    s, s_next = _mk_data(dim=D, n=B)

    # Force variance to the floor (-10) to simulate a collapsed baseline.
    with torch.no_grad():
        wm.logvar_head.weight.zero_()
        wm.logvar_head.bias.fill_(-12.0)  # Pre-clamp → clamped to min_logvar.

    continuous_nll = m5_predictive_nll(wm, s, s_next)
    # With σ² at floor the continuous NLL is dominated by either the
    # log-|Σ| term (very negative) or the quadratic term (very positive),
    # depending on how close the untrained mean_head is to s_next. Either
    # way it is extreme — and therefore an unreliable comparison metric.
    assert abs(continuous_nll) > 50.0, (
        f"variance-collapsed NLL expected to be extreme in magnitude, "
        f"got {continuous_nll}"
    )

    # The discrete metric must remain in its valid ranges regardless.
    r = m5_discrete_rank(wm, lag, s, s_next, candidate_size=C, seed=0)
    for prefix in ("q_", "K_"):
        assert 0.0 <= r[prefix + "top1_acc"] <= 1.0
        assert 1.0 <= r[prefix + "mean_rank"] <= float(C)
        assert r[prefix + "nll_on_set"] >= 0.0

    print(f"  [OK] Variance collapse: |continuous NLL|={abs(continuous_nll):.1f} "
          f"(extreme); discrete metrics stay in bounds")


def test_trivial_world_model_beats_random():
    """Sanity: an identity-ish world model (predicts mean = s_t) should
    rank the true next state better than 1/C on average when s_next is
    closer to s_t than to a random other s_next in the batch."""
    D, B, C = 8, 128, 32
    metric, lag, wm = _mk_stack(dim=D)
    # Make s_next tightly centered on s (s_next = s + tiny noise)
    torch.manual_seed(42)
    s = torch.randn(B, D)
    s_next = s + 0.05 * torch.randn(B, D)

    r = m5_discrete_rank(wm, lag, s, s_next, candidate_size=C, seed=0)
    # The Gaussian WM is initialized so mean(s) ≈ s; with near-identity
    # next states, top-1 should comfortably beat chance (1/C = 3.1%).
    assert r["q_top1_acc"] > 1.0 / C + 0.05, (
        f"q_top1_acc={r['q_top1_acc']} barely above chance {1.0/C}"
    )
    print(f"  [OK] Near-identity transitions: q_top1_acc={r['q_top1_acc']:.3f} "
          f"> chance {1.0/C:.3f}")


def test_integration_into_compute_all_metrics():
    """compute_all_metrics must carry m5disc_* keys after the change."""
    D, B = 8, 64
    metric, lag, wm = _mk_stack(dim=D)
    s, s_next = _mk_data(dim=D, n=B)
    r = compute_all_metrics(metric, wm, lag, s, s_next)

    expected = {
        "m5disc_q_top1_acc", "m5disc_q_top5_acc",
        "m5disc_q_mean_rank", "m5disc_q_nll_on_set",
        "m5disc_K_top1_acc", "m5disc_K_top5_acc",
        "m5disc_K_mean_rank", "m5disc_K_nll_on_set",
    }
    missing = expected - set(r.keys())
    assert not missing, f"missing keys in compute_all_metrics: {missing}"

    # Continuous m5_nll must still be there (diagnostic)
    assert "m5_nll" in r, "m5_nll diagnostic disappeared"

    print("  [OK] compute_all_metrics exposes m5disc_* alongside m5_nll")


def main():
    print("Running M5 discrete-rank tests...\n")
    test_basic_ranges()
    test_deterministic_given_seed()
    test_variance_collapse_still_ranks()
    test_trivial_world_model_beats_random()
    test_integration_into_compute_all_metrics()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
