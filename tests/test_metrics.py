"""
Tests for m5_candidate_metrics and world_model_variance_diagnostic.

Run with:
    python -m tests.test_metrics
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import (
    m5_candidate_metrics,
    m5_predictive_nll,
    world_model_variance_diagnostic,
    compute_all_metrics,
)
from src.training.candidates import build_candidate_set_c1
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


def _mk_candidates(dim: int = 8, n: int = 64, C: int = 32, seed: int = 0):
    s, s_next = _mk_data(dim=dim, n=n)
    torch.manual_seed(seed)
    cands, true_idx = build_candidate_set_c1(s, s_next, candidate_size=C)
    return s, cands, true_idx


# --------------------------------------------------------------------------
# m5_candidate_metrics
# --------------------------------------------------------------------------

def test_m5_candidate_metrics_ranges():
    """All returned quantities must lie in their valid ranges."""
    D, B, C = 8, 64, 32
    _, _, wm = _mk_stack(dim=D)
    s, cands, true_idx = _mk_candidates(dim=D, n=B, C=C)

    r = m5_candidate_metrics(wm, s, cands, true_idx, topk=5)

    assert 0.0 <= r["m5d_top1"] <= 1.0, r["m5d_top1"]
    assert 0.0 <= r["m5d_topk"] <= 1.0, r["m5d_topk"]
    # top1 ⊆ topk ⇒ top1 ≤ topk (within fp noise)
    assert r["m5d_top1"] <= r["m5d_topk"] + 1e-6
    # MRR ∈ [1/C, 1]
    assert 1.0 / C - 1e-6 <= r["m5d_mrr"] <= 1.0 + 1e-6, r["m5d_mrr"]
    # CE ≥ 0 (log_softmax outputs are ≤ 0, NLL is ≥ 0)
    assert r["m5d_ce"] >= 0.0, r["m5d_ce"]
    # topk_k is min(topk, C)
    assert r["m5d_topk_k"] == min(5, C)
    print("  [OK] Ranges: top-k ∈ [0,1], MRR ∈ [1/C, 1], CE ≥ 0, topk_k = min(5,C)")


def test_m5_candidate_metrics_perfect_model():
    """A world model whose mean exactly hits the true next and whose
    log-density for the wrong candidates is lower → MRR = 1, top-1 = 1,
    CE ≈ 0."""
    D, B, C = 8, 32, 16
    _, _, wm = _mk_stack(dim=D)
    s, cands, true_idx = _mk_candidates(dim=D, n=B, C=C)

    # Force mean_head to predict a perfect shift: set mean = true_next.
    # `forward` computes mean = s + self.mean_head(h); we want mean = s_next.
    # So mean_head(h) should output (s_next - s). We make it a lookup: overwrite
    # forward with a no-op stub that returns (true_next_for_this_s, tiny_logvar).
    true_next = cands[torch.arange(B), true_idx]  # (B, D)

    class PerfectWM(torch.nn.Module):
        def __init__(self, inner, s_in, sn_out):
            super().__init__()
            self.inner = inner
            self.min_logvar = inner.min_logvar
            self.dim = inner.dim
            # Memorise (s -> sn) for the rows we know about.
            self._map = {tuple(x.tolist()): y.detach().clone()
                         for x, y in zip(s_in, sn_out)}

        def forward(self, s_):
            mean = torch.stack([
                self._map[tuple(x.tolist())].to(x.dtype)
                if tuple(x.tolist()) in self._map else x
                for x in s_
            ])
            logvar = torch.full_like(mean, -2.0)
            return mean, logvar

        def log_prob(self, s_, s_next_):
            mean, logvar = self.forward(s_)
            import math
            var = logvar.exp()
            diff = s_next_ - mean
            return -0.5 * (
                self.dim * math.log(2 * math.pi)
                + logvar.sum(dim=-1)
                + (diff ** 2 / var).sum(dim=-1)
            )

        def log_prob_candidates(self, s_, cands_):
            return self.inner.log_prob_candidates(s_, cands_)

    perfect = PerfectWM(wm, s, true_next)
    r = m5_candidate_metrics(perfect, s, cands, true_idx, topk=5)

    assert r["m5d_top1"] > 0.99, f"perfect model top1 = {r['m5d_top1']}"
    assert r["m5d_mrr"] > 0.99, f"perfect model mrr = {r['m5d_mrr']}"
    assert r["m5d_ce"] < 1.0, f"perfect model CE suspiciously high: {r['m5d_ce']}"
    print(f"  [OK] Perfect model: top1={r['m5d_top1']:.3f}, "
          f"MRR={r['m5d_mrr']:.3f}, CE={r['m5d_ce']:.3f}")


def test_m5_candidate_metrics_random_model():
    """A random-init world model on random data → MRR ≈ 1/C + chance noise,
    top-1 ≈ 1/C. We don't test exact values (small sample, Gaussian tail
    behaviour), just that top-1 ≤ topk ≤ 1 and CE is finite."""
    D, B, C = 8, 64, 32
    _, _, wm = _mk_stack(dim=D)
    s, cands, true_idx = _mk_candidates(dim=D, n=B, C=C, seed=123)

    r = m5_candidate_metrics(wm, s, cands, true_idx, topk=5)

    # Sanity: finite, in bounds.
    for k in ("m5d_top1", "m5d_topk", "m5d_mrr", "m5d_ce"):
        v = r[k]
        assert torch.isfinite(torch.tensor(v)), f"{k}={v} not finite"
    # topk_recall must dominate top1.
    assert r["m5d_top1"] <= r["m5d_topk"] + 1e-6
    print(f"  [OK] Random model: top1={r['m5d_top1']:.3f}, "
          f"topk={r['m5d_topk']:.3f}, MRR={r['m5d_mrr']:.3f}, CE={r['m5d_ce']:.3f}")


def test_m5_candidate_metrics_topk_capped_to_C():
    """If the user asks for topk larger than C, we should cap, not error."""
    D, B, C = 8, 16, 4
    _, _, wm = _mk_stack(dim=D)
    s, cands, true_idx = _mk_candidates(dim=D, n=B, C=C)
    r = m5_candidate_metrics(wm, s, cands, true_idx, topk=20)
    assert r["m5d_topk_k"] == C
    # topk with k=C means every true is in it ⇒ topk_recall = 1.0
    assert r["m5d_topk"] > 0.999
    print(f"  [OK] topk capped to C={C}; topk_recall = {r['m5d_topk']:.3f}")


# --------------------------------------------------------------------------
# world_model_variance_diagnostic
# --------------------------------------------------------------------------

def test_variance_diagnostic_not_collapsed():
    """Freshly-initialised world model → logvar ≈ 0 → not collapsed."""
    D = 8
    _, _, wm = _mk_stack(dim=D)
    s, _ = _mk_data(dim=D, n=64)
    r = world_model_variance_diagnostic(wm, s)
    assert r["wm_logvar_floor"] == -10.0
    assert r["wm_frac_at_floor"] < 0.3
    assert r["wm_collapsed"] is False
    print(f"  [OK] Fresh world model not flagged: "
          f"mean_logvar={r['wm_mean_logvar']:.2f}, "
          f"frac_at_floor={r['wm_frac_at_floor']:.2f}")


def test_variance_diagnostic_detects_collapse():
    """Force the logvar bias to the floor → must report collapsed = True
    and the continuous NLL must indeed be extreme."""
    D = 8
    _, _, wm = _mk_stack(dim=D)
    s, s_next = _mk_data(dim=D, n=64)
    with torch.no_grad():
        wm.logvar_head.weight.zero_()
        wm.logvar_head.bias.fill_(-12.0)  # pre-clamp → clamped to -10

    r = world_model_variance_diagnostic(wm, s)
    assert r["wm_collapsed"] is True, (
        f"should be flagged: mean_logvar={r['wm_mean_logvar']:.3f}"
    )
    assert r["wm_frac_at_floor"] > 0.99
    # The continuous NLL on this collapsed model should be extreme.
    ce = m5_predictive_nll(wm, s, s_next)
    assert abs(ce) > 50.0, f"collapsed NLL expected extreme, got {ce}"
    print(f"  [OK] Collapsed world model flagged: "
          f"mean_logvar={r['wm_mean_logvar']:.2f}, |continuous NLL|={abs(ce):.1f}")


# --------------------------------------------------------------------------
# compute_all_metrics integration
# --------------------------------------------------------------------------

def test_compute_all_metrics_is_lightweight():
    """compute_all_metrics is the training-time eval bundle: it must
    surface cheap diagnostics (M1, M5 NLL, variance diagnostic, M4 raw)
    but NOT the heavy final-eval metrics (m4f_*, m5d_*, probe_*). The
    heavy ones belong to run_d2's end-of-training block, where we can
    properly couple base_rate across geometries."""
    D, B = 8, 64
    metric, lag, wm = _mk_stack(dim=D)
    s, s_next = _mk_data(dim=D, n=B)
    r = compute_all_metrics(metric, wm, lag, s, s_next)

    # Must contain: lightweight keys
    for k in (
        "m1_timelike_rate", "m5_nll",
        "wm_mean_logvar", "wm_frac_at_floor", "wm_logvar_floor", "wm_collapsed",
        "m4raw_jaccard", "m4raw_precision", "m4raw_recall",
    ):
        assert k in r, f"missing lightweight key {k}"

    # Must NOT contain: heavy final-eval keys. Those live in run_d2.
    for heavy in (
        "m4fair_jaccard", "m4fair_precision", "m4fair_recall", "m4fair_base_rate",
        "m4f_jaccard", "m4f_precision", "m4f_recall", "m4f_base_rate",
        "m5d_ce", "m5d_top1", "m5d_topk", "m5d_mrr", "m5d_topk_k",
        "probe_latent_acc", "probe_raw_acc",
    ):
        assert heavy not in r, (
            f"heavy key {heavy} leaked into the training-time eval bundle; "
            f"move it to run_d2's final eval"
        )
    print("  [OK] compute_all_metrics lightweight bundle: "
          "m1/m5_nll/wm_*/m4raw_* only, no m4f_*/m5d_*/probe_*")


def main():
    print("Running metrics tests (m5_candidate_metrics + variance diag)...\n")
    test_m5_candidate_metrics_ranges()
    test_m5_candidate_metrics_topk_capped_to_C()
    test_m5_candidate_metrics_random_model()
    test_m5_candidate_metrics_perfect_model()
    test_variance_diagnostic_not_collapsed()
    test_variance_diagnostic_detects_collapse()
    test_compute_all_metrics_is_lightweight()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
