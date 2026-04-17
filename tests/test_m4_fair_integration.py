"""
Integration tests for the M4-fair coupling protocol.

Verifies:
  1. compute_all_metrics returns both m4raw_* and m4fair_* keys.
  2. The base_rate parameter is honoured (same rate → baseline geometry
     classifies the same fraction of pairs as reachable).
  3. Lorentzian self-calibration produces a base_rate that, when passed
     to a Riemannian metric on the same data, yields a Riemannian
     m4fair_base_rate within a small tolerance.
  4. The new m4_fair_candidates contract: non-Lorentzian geometries MUST
     receive an explicit base_rate (no silent default); the function
     raises ValueError when called without one.

Run with:
    python -m tests.test_m4_fair_integration
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import (
    compute_all_metrics,
    m4_fair,
    m4_fair_candidates,
)
from src.training.candidates import build_candidate_set_c1
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
    """squared_interval ≥ 0 for Euclidean → metric cone is empty → M4_raw = 0.
    m4_fair_candidates, by contrast, yields a meaningful non-zero Jaccard
    when a base_rate is provided."""
    metric, lag, wm = _mk_stack("euclidean")
    s, s_next = _mk_data()
    # The new m4_fair_candidates requires an explicit base_rate for
    # non-Lorentzian geometries (no silent default). We pass 0.7 here.
    r = compute_all_metrics(metric, wm, lag, s, s_next, base_rate=0.7)
    assert r["m4raw_jaccard"] == 0.0, (
        f"expected m4raw_jaccard=0 for Euclidean, got {r['m4raw_jaccard']}"
    )
    assert 0.0 < r["m4fair_jaccard"] <= 1.0, (
        f"m4fair_jaccard={r['m4fair_jaccard']} not in (0, 1]"
    )
    print("  [OK] Euclidean: m4raw_jaccard=0 (tautologico), m4fair_jaccard>0")


def test_non_lorentzian_requires_explicit_base_rate():
    """m4_fair_candidates must refuse to run on non-Lorentzian geometries
    without an explicit base_rate (to prevent silent uncoupled comparisons
    that would be meaningless)."""
    metric, lag, _ = _mk_stack("riemannian")
    s, s_next = _mk_data(n=64)
    cands, _ = build_candidate_set_c1(s, s_next, candidate_size=32)
    try:
        m4_fair_candidates(
            metric, lag, s, cands,
            geometry="riemannian", base_rate=None,
        )
    except ValueError as e:
        assert "base_rate" in str(e), f"wrong error message: {e}"
        print("  [OK] Non-Lorentzian without base_rate raises ValueError")
        return
    raise AssertionError(
        "m4_fair_candidates should have raised on Riemannian + base_rate=None"
    )


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


def test_legacy_m4_fair_is_still_callable():
    """The legacy shuffled-negatives m4_fair must remain importable and
    functional (kept as a diagnostic; compute_all_metrics now uses
    m4_fair_candidates internally)."""
    metric, _, _ = _mk_stack("lorentzian")
    s, s_next = _mk_data(n=64)
    r = m4_fair(
        metric, s, s_next,
        s_next[torch.randperm(len(s_next))],
        geometry="lorentzian", base_rate=None,
    )
    assert "m4f_base_rate" in r
    assert 0.0 <= r["m4f_base_rate"] <= 1.0
    print(f"  [OK] Legacy m4_fair still callable: "
          f"m4f_base_rate={r['m4f_base_rate']:.4f}")


def main():
    print("Running M4-fair integration tests...\n")
    test_compute_all_metrics_has_both_m4_variants()
    test_euclidean_m4raw_is_tautologically_zero()
    test_non_lorentzian_requires_explicit_base_rate()
    test_base_rate_is_honoured()
    test_lorentzian_calibration_shared_with_baseline()
    test_legacy_m4_fair_is_still_callable()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
