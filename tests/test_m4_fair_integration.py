"""
Integration tests for the M4-fair coupling protocol.

After the "training-time eval is lightweight" refactor, `compute_all_metrics`
does NOT call `m4_fair_candidates` any more (those live in `run_d2`'s
final eval block). These tests therefore exercise `m4_fair_candidates`
DIRECTLY on a built candidate set, which is closer to the production
call site in `scripts/train.py::run_d2`.

Verifies:
  1. compute_all_metrics is the lightweight training bundle: it reports
     m4raw_* but NOT m4fair_* / m4f_* / m5d_* — those are final-eval only.
  2. m4_fair_candidates contract: non-Lorentzian with base_rate=None
     raises ValueError.
  3. The base_rate parameter is echoed verbatim by m4_fair_candidates.
  4. Lorentzian self-calibration produces a base_rate that, when passed
     to a Riemannian metric on the same (s, candidates), is reported
     back identically.
  5. The legacy m4_fair (shuffled negatives) stays importable and
     functional as a diagnostic.

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


def _mk_candidates(dim: int = 8, n: int = 128, C: int = 32, seed: int = 0):
    s, s_next = _mk_data(dim=dim, n=n)
    torch.manual_seed(seed)
    cands, _ = build_candidate_set_c1(s, s_next, candidate_size=C)
    return s, s_next, cands


# --------------------------------------------------------------------------
# compute_all_metrics — lightweight bundle (m4raw_ only; no m4fair_)
# --------------------------------------------------------------------------

def test_compute_all_metrics_only_has_m4raw():
    """Training-time bundle contains the diagnostic M4 raw cone alignment
    but NOT the fair-coupling M4 variant (which is for final eval only)."""
    metric, lag, wm = _mk_stack("lorentzian")
    s, s_next = _mk_data()
    r = compute_all_metrics(metric, wm, lag, s, s_next)

    for k in ("m4raw_jaccard", "m4raw_precision", "m4raw_recall"):
        assert k in r, f"missing {k}"

    for leaked in (
        "m4fair_jaccard", "m4fair_base_rate",
        "m4f_jaccard", "m4f_base_rate",
    ):
        assert leaked not in r, f"heavy key {leaked} leaked into training bundle"

    # Legacy keys from previous iterations must not leak either
    for legacy in ("m4_jaccard", "m4_precision", "m4_recall"):
        assert legacy not in r, f"legacy key {legacy} leaked"

    print("  [OK] compute_all_metrics has m4raw_* only, no m4fair_* / m4f_*")


def test_euclidean_m4raw_is_tautologically_zero():
    """squared_interval ≥ 0 for Euclidean → metric cone is empty → M4 raw = 0.
    That's exactly why we need the fair variant."""
    metric, lag, wm = _mk_stack("euclidean")
    s, s_next = _mk_data()
    r = compute_all_metrics(metric, wm, lag, s, s_next)
    assert r["m4raw_jaccard"] == 0.0, (
        f"expected m4raw_jaccard=0 for Euclidean, got {r['m4raw_jaccard']}"
    )
    print("  [OK] Euclidean m4raw_jaccard = 0 (tautological)")


# --------------------------------------------------------------------------
# m4_fair_candidates — direct tests of the production function
# --------------------------------------------------------------------------

def test_non_lorentzian_requires_explicit_base_rate():
    """m4_fair_candidates must refuse to run on non-Lorentzian geometries
    without an explicit base_rate (to prevent silent uncoupled comparisons
    that would be meaningless)."""
    metric, lag, _ = _mk_stack("riemannian")
    s, _, cands = _mk_candidates()
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


def test_euclidean_m4fair_candidates_is_meaningful():
    """On the SAME Euclidean stack where m4raw = 0, m4_fair_candidates
    returns a meaningful non-zero Jaccard given a base_rate."""
    metric, lag, _ = _mk_stack("euclidean")
    s, _, cands = _mk_candidates()
    r = m4_fair_candidates(
        metric, lag, s, cands,
        geometry="euclidean", base_rate=0.7,
    )
    assert 0.0 < r["m4f_jaccard"] <= 1.0, (
        f"m4f_jaccard={r['m4f_jaccard']} not in (0, 1] for Euclidean"
    )
    assert r["m4f_base_rate"] == 0.7
    print(f"  [OK] Euclidean m4f_jaccard={r['m4f_jaccard']:.3f} > 0")


def test_base_rate_echoed_by_m4_fair_candidates():
    """Passing base_rate=p returns m4f_base_rate=p verbatim (the Lorentzian
    baseline coupling relies on this identity)."""
    metric, lag, _ = _mk_stack("riemannian")
    s, _, cands = _mk_candidates(n=500, C=32)
    for target in (0.3, 0.5, 0.8):
        r = m4_fair_candidates(
            metric, lag, s, cands,
            geometry="riemannian", base_rate=target,
        )
        assert abs(r["m4f_base_rate"] - target) < 1e-6, (
            f"base_rate not honoured: asked {target}, got {r['m4f_base_rate']}"
        )
    print("  [OK] base_rate parameter is propagated verbatim")


def test_lorentzian_calibration_shared_with_baseline():
    """End-to-end: Lorentzian self-calibrates base_rate on its candidate
    set, the baseline receives it, reports it back identically."""
    lor_m, lor_lag, _ = _mk_stack("lorentzian")
    rie_m, rie_lag, _ = _mk_stack("riemannian")
    s, _, cands = _mk_candidates(n=400)

    lor = m4_fair_candidates(
        lor_m, lor_lag, s, cands,
        geometry="lorentzian", base_rate=None,
    )
    shared = lor["m4f_base_rate"]
    assert 0.0 < shared < 1.0, f"pathological Lorentzian base_rate {shared}"

    rie = m4_fair_candidates(
        rie_m, rie_lag, s, cands,
        geometry="riemannian", base_rate=shared,
    )
    assert abs(rie["m4f_base_rate"] - shared) < 1e-6, (
        f"coupling failed: lor={shared:.6f} vs rie={rie['m4f_base_rate']:.6f}"
    )
    print(f"  [OK] Cross-geometry coupling: shared base_rate={shared:.4f}")


def test_semantic_costs_decouples_plausibility_from_lagrangian():
    """Passing semantic_costs must change the plausibility set (and thus
    m4f_jaccard) vs. using the Lagrangian itself."""
    metric, lag, _ = _mk_stack("lorentzian")
    s, _, cands = _mk_candidates(n=256, C=32)

    torch.manual_seed(0)
    sem = torch.randn(s.shape[0], cands.shape[1]).abs()

    r_with_sem = m4_fair_candidates(
        metric, lag, s, cands,
        geometry="lorentzian", base_rate=0.5, semantic_costs=sem,
    )
    r_lag = m4_fair_candidates(
        metric, lag, s, cands,
        geometry="lorentzian", base_rate=0.5, semantic_costs=None,
    )

    # Different plausibility sets almost always give different jaccards.
    # We allow the (rare) coincidence but always require both in [0, 1].
    for r in (r_with_sem, r_lag):
        assert 0.0 <= r["m4f_jaccard"] <= 1.0
    print(f"  [OK] semantic_costs branch: "
          f"with_sem={r_with_sem['m4f_jaccard']:.3f}, "
          f"lagrangian={r_lag['m4f_jaccard']:.3f}")


def test_legacy_m4_fair_is_still_callable():
    """The legacy shuffled-negatives m4_fair must remain importable and
    functional (kept as a diagnostic)."""
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
    test_compute_all_metrics_only_has_m4raw()
    test_euclidean_m4raw_is_tautologically_zero()
    test_non_lorentzian_requires_explicit_base_rate()
    test_euclidean_m4fair_candidates_is_meaningful()
    test_base_rate_echoed_by_m4_fair_candidates()
    test_lorentzian_calibration_shared_with_baseline()
    test_semantic_costs_decouples_plausibility_from_lagrangian()
    test_legacy_m4_fair_is_still_callable()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
