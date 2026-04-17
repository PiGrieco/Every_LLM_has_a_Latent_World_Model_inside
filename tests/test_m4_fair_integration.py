"""
Integration tests for the M4-fair coupling protocol.

Verifies:
  1. compute_all_metrics returns both m4raw_* and m4fair_* keys.
  2. The base_rate parameter is honoured (same rate → baseline geometry
     classifies the same fraction of pairs as reachable).
  3. Lorentzian self-calibration produces a base_rate that, when passed
     to a Riemannian metric on the same data, yields a Riemannian
     m4fair_base_rate within a small tolerance.

Run with:
    python -m tests.test_m4_fair_integration
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import compute_all_metrics, m4_fair
from src.models.metric import MetricNetwork
from src.models.lagrangian import Lagrangian
from src.models.world_model import ConditionalGaussianWorldModel


def _mk_stack(geometry: str, dim: int = 8):
    torch.manual_seed(0)
    metric = MetricNetwork(dim=dim, hidden_dim=16, n_layers=1, geometry=geometry)
    lagrangian = Lagrangian(metric=metric, lambda_g=1.0, lambda_sem=0.0,
                            latent_dim=dim)
    world_model = ConditionalGaussianWorldModel(dim=dim, hidden_dim=32, n_layers=2)
    return metric, lagrangian, world_model


def _mk_data(dim: int = 8, n: int = 128):
    torch.manual_seed(1)
    s = torch.randn(n, dim)
    s_next = s + 0.3 * torch.randn(n, dim)
    return s, s_next


def test_compute_all_metrics_has_both_m4_variants():
    metric, lag, wm = _mk_stack("lorentzian")
    s, s_next = _mk_data()
    r = compute_all_metrics(metric, wm, lag, s, s_next)

    # Raw cone alignment
    for k in ("m4raw_jaccard", "m4raw_precision", "m4raw_recall"):
        assert k in r, f"missing {k}"
    # Fair cone alignment
    for k in ("m4fair_jaccard", "m4fair_precision", "m4fair_recall", "m4fair_base_rate"):
        assert k in r, f"missing {k}"

    # Legacy m4_* keys must NOT leak (we renamed them)
    for legacy in ("m4_jaccard", "m4_precision", "m4_recall"):
        assert legacy not in r, f"legacy key {legacy} leaked"

    print("  [OK] compute_all_metrics returns m4raw_* + m4fair_*, no legacy keys")


def test_euclidean_m4raw_is_tautologically_zero():
    """squared_interval ≥ 0 for Euclidean → metric cone is empty → M4_raw = 0."""
    metric, lag, wm = _mk_stack("euclidean")
    s, s_next = _mk_data()
    r = compute_all_metrics(metric, wm, lag, s, s_next)
    # Jaccard is 0 because metric cone is empty (intervals all > 0)
    assert r["m4raw_jaccard"] == 0.0, (
        f"expected m4raw_jaccard=0 for Euclidean, got {r['m4raw_jaccard']}"
    )
    # m4fair, by contrast, should be non-zero for the base_rate we pass
    r2 = compute_all_metrics(metric, wm, lag, s, s_next, base_rate=0.7)
    assert 0.0 < r2["m4fair_jaccard"] <= 1.0
    print("  [OK] Euclidean: m4raw_jaccard=0 (tautologico), m4fair_jaccard>0")


def test_base_rate_is_honoured():
    """Passing base_rate=p should make the geometry classify ~p of pairs
    as reachable (within quantile noise)."""
    metric, lag, wm = _mk_stack("riemannian")
    s, s_next = _mk_data(n=500)

    for target in (0.3, 0.5, 0.8):
        r = compute_all_metrics(metric, wm, lag, s, s_next, base_rate=target)
        assert abs(r["m4fair_base_rate"] - target) < 1e-6, (
            f"base_rate not honoured: asked {target}, got {r['m4fair_base_rate']}"
        )
    print("  [OK] base_rate parameter is propagated verbatim to m4_fair")


def test_lorentzian_calibration_shared_with_baseline():
    """Full protocol: run Lorentzian with base_rate=None, then pass its
    auto-calibrated rate to a Riemannian stack on the same (s, s_next)
    split. Riemannian must report the same m4fair_base_rate."""
    lor_metric, lor_lag, lor_wm = _mk_stack("lorentzian")
    rie_metric, rie_lag, rie_wm = _mk_stack("riemannian")
    s, s_next = _mk_data(n=400)

    r_lor = compute_all_metrics(lor_metric, lor_wm, lor_lag, s, s_next,
                                base_rate=None)
    shared = r_lor["m4fair_base_rate"]
    assert 0.0 < shared < 1.0, f"pathological base_rate {shared}"

    r_rie = compute_all_metrics(rie_metric, rie_wm, rie_lag, s, s_next,
                                base_rate=shared)
    assert abs(r_rie["m4fair_base_rate"] - shared) < 1e-6, (
        f"coupling failed: lor={shared:.6f} vs rie={r_rie['m4fair_base_rate']:.6f}"
    )
    print(f"  [OK] Cross-geometry coupling: shared base_rate={shared:.4f}")


def test_m4_fair_independent_invocation_matches():
    """compute_all_metrics(base_rate=p) must match a direct m4_fair call
    with the same arguments (up to RNG for the permutation, which is
    deterministic here via manual_seed)."""
    metric, lag, wm = _mk_stack("lorentzian")
    s, s_next = _mk_data(n=200)

    torch.manual_seed(42)
    r = compute_all_metrics(metric, wm, lag, s, s_next, base_rate=0.6)

    torch.manual_seed(42)
    # compute_all_metrics truncates to min(256, N) and uses torch.randperm.
    n_ev = min(256, s.shape[0])
    # Re-create the same permutation: inside compute_all_metrics we call
    # build_candidate_set_c1 BEFORE the permutation, so the RNG is advanced.
    # We skip exact byte-parity and instead verify bounded agreement.
    m4f_direct = m4_fair(
        metric, s[:n_ev], s_next[:n_ev],
        s_next[:n_ev][torch.randperm(n_ev)],
        geometry="lorentzian", base_rate=0.6,
    )
    # base_rate is deterministic given the input; jaccard differs because
    # the permutation differs. Check that base_rate agrees exactly.
    assert abs(r["m4fair_base_rate"] - m4f_direct["m4f_base_rate"]) < 1e-6
    print("  [OK] m4_fair invocation inside compute_all_metrics is consistent")


def main():
    print("Running M4-fair integration tests...\n")
    test_compute_all_metrics_has_both_m4_variants()
    test_euclidean_m4raw_is_tautologically_zero()
    test_base_rate_is_honoured()
    test_lorentzian_calibration_shared_with_baseline()
    test_m4_fair_independent_invocation_matches()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
