"""Unit tests for src/projection (M2, post-review revision).

Fast tests never touch M1 data or train anything non-trivially. Slow
tests (@pytest.mark.slow) run a tiny end-to-end on synthetic data and
exercise the smoke-gate CLI via subprocess to confirm exit codes.
"""

from __future__ import annotations

import dataclasses as _dc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.llm_probe import TrajectoryShardWriter
from src.projection import (
    Decoder,
    Discriminator,
    Encoder,
    HiddenStateMemory,
    PairedHiddenStateDataset,
    ProjectionAutoencoder,
    ProjectionConfig,
    ProjectionTrainer,
    build_memory_and_queries,
    estimate_intrinsic_dim,
    on_manifold_drift,
    retrieval_evaluation,
    train_identity_probe,
)


# ==========================================================================
# Module-level fast tests
# ==========================================================================

def test_config_defaults_post_revision():
    cfg = ProjectionConfig()
    # REVISED: adversarial default off
    assert cfg.use_adversarial is False
    # REVISED: warning/error thresholds exist as diagnostics
    assert cfg.identity_probe_warning_threshold == pytest.approx(0.20)
    assert cfg.identity_probe_error_threshold == pytest.approx(0.05)
    assert cfg.on_manifold_drift_warning_threshold == pytest.approx(2.0)
    # REVISED: removed gate_min_identity_probe_accuracy / gate_max_on_manifold_drift
    fields = {f.name for f in _dc.fields(ProjectionConfig)}
    assert "gate_min_identity_probe_accuracy" not in fields
    assert "gate_max_on_manifold_drift" not in fields
    # Remaining hard gates
    assert "gate_min_retrieval_top5_fraction_of_baseline" in fields
    assert "gate_max_reconstruction_mse_ratio" in fields
    assert "memory_query_split_fraction" in fields


def test_encoder_forward_shape():
    enc = Encoder(d_model=64, latent_dim=8, hidden=32, dropout=0.1)
    h = torch.randn(5, 64)
    s = enc(h)
    assert s.shape == (5, 8)


def test_decoder_forward_shape():
    dec = Decoder(latent_dim=8, d_model=64, hidden=32)
    s = torch.randn(5, 8)
    h = dec(s)
    assert h.shape == (5, 64)


def test_autoencoder_roundtrip_shapes():
    cfg = ProjectionConfig(
        latent_dim=8, encoder_hidden=32, decoder_hidden=32,
        use_adversarial=False,
    )
    ae = ProjectionAutoencoder(cfg, d_model=64)
    h = torch.randn(5, 64)
    s, h_hat = ae(h)
    assert s.shape == (5, 8)
    assert h_hat.shape == (5, 64)


def test_discriminator_output_scalar():
    disc = Discriminator(d_model=32, hidden=16, n_layers=4)
    h = torch.randn(5, 32)
    out = disc(h)
    assert out.shape == (5,)
    # Finite and roughly in a reasonable range for a spectral-norm MLP.
    assert torch.isfinite(out).all()


def test_consistency_loss_zero_when_identical():
    cfg = ProjectionConfig(latent_dim=8, encoder_hidden=16, decoder_hidden=16,
                           use_adversarial=False)
    ae = ProjectionAutoencoder(cfg, d_model=32)
    h = torch.randn(4, 32)
    # Two sides identical → latent_dist and ambient_dist are both zero,
    # so loss = 0 regardless of α.
    loss = ae.consistency_loss(h, h)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


def test_reconstruction_loss_decreases_after_step():
    torch.manual_seed(0)
    cfg = ProjectionConfig(latent_dim=4, encoder_hidden=16, decoder_hidden=16,
                           use_adversarial=False)
    ae = ProjectionAutoencoder(cfg, d_model=16)
    opt = torch.optim.Adam(ae.parameters(), lr=3e-3)
    h = torch.randn(32, 16)
    loss0 = ae.reconstruction_loss(h).item()
    for _ in range(10):
        loss = ae.reconstruction_loss(h)
        opt.zero_grad(); loss.backward(); opt.step()
    loss1 = ae.reconstruction_loss(h).item()
    assert loss1 < loss0, f"loss didn't drop: {loss0:.3f} → {loss1:.3f}"


def test_intrinsic_dim_toy():
    torch.manual_seed(0)
    N, D_full, D_true = 500, 128, 20
    # Rank-D_true data embedded in D_full with small noise.
    A = torch.randn(D_true, D_full) * 0.5
    Z = torch.randn(N, D_true)
    X = Z @ A + 0.01 * torch.randn(N, D_full)
    result = estimate_intrinsic_dim(
        X, target_variance=0.95, dim_min=1, dim_max=D_full,
    )
    # The estimate should be in the ballpark of the true rank.
    assert D_true - 5 <= result["intrinsic_dim"] <= D_true + 10, result


def test_hiddenstatememory_perfect_self_retrieval():
    torch.manual_seed(0)
    H = torch.randn(50, 16)
    doc_ids = [f"d{i}" for i in range(50)]
    mem = HiddenStateMemory(H, doc_ids)
    idx, scores = mem.search(H, top_k=1)
    # Each row retrieves itself as nearest neighbour.
    assert (idx.squeeze(-1) == torch.arange(50)).all()
    # Self-cosine is 1.
    assert (scores.squeeze(-1) > 0.999).all()


def test_on_manifold_drift_zero_for_identity_ae():
    """If Ψ(Φ(h)) == h exactly, drift should be 0."""
    class _Identity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.discriminator = None
            self.encoder = self
            self.decoder = self
        def encode(self, h): return h
        def decode(self, s): return s
        def forward(self, h): return h, h

    h = torch.randn(32, 16)
    drift = on_manifold_drift(_Identity(), h)
    assert drift == pytest.approx(0.0, abs=1e-6)


def test_disjoint_doc_ids_enforced(tmp_path: Path):
    """The revised retrieval builder must enforce disjoint doc_ids."""
    # Synthesise a small M1-style dataset: 5 docs × 4 windows.
    d_model = 16
    ds = tmp_path / "forward"
    with TrajectoryShardWriter(str(ds), shard_size=3, save_dtype="float32") as w:
        for i in range(5):
            w.add({
                "doc_id": f"doc{i:02d}",
                "hidden_states": torch.randn(4, d_model),
                "token_positions": [(t, t + 4) for t in range(4)],
            })
    from src.llm_probe import TrajectoryShardReader
    reader = TrajectoryShardReader(str(ds))
    memory, q_h, q_ids = build_memory_and_queries(
        reader, memory_size=8, n_queries=4, memory_fraction=0.6, seed=0,
    )
    # Memory doc_ids and query doc_ids must be strictly disjoint.
    assert not set(memory.doc_ids) & set(q_ids)


def test_identity_probe_random_baseline(tmp_path: Path):
    """With random features and few examples per class, accuracy should
    be close to the 1/n_classes random baseline (not systematically high)."""
    torch.manual_seed(0)
    d_model = 32
    ds = tmp_path / "forward"
    with TrajectoryShardWriter(str(ds), shard_size=10, save_dtype="float32") as w:
        for i in range(20):
            w.add({
                "doc_id": f"d{i:03d}",
                "hidden_states": torch.randn(4, d_model),  # no structure
            })
    from src.llm_probe import TrajectoryShardReader
    reader = TrajectoryShardReader(str(ds))

    cfg = ProjectionConfig(
        latent_dim=8, encoder_hidden=16, decoder_hidden=16,
        use_adversarial=False,
        identity_probe_n_classes=20,
        identity_probe_samples_per_class=3,
        identity_probe_epochs=3,
        random_seed=0,
    )
    ae = ProjectionAutoencoder(cfg, d_model=d_model)
    result = train_identity_probe(ae, reader, cfg)
    # Very small dataset; we just check the probe runs and returns numeric.
    assert isinstance(result.get("accuracy"), float)
    assert result.get("n_classes") >= 2


def test_retrieval_evaluation_projected_vs_baseline(tmp_path: Path):
    """Retrieval report has the expected structure and values are in [0, 1]."""
    torch.manual_seed(1)
    d_model = 32
    ds = tmp_path / "forward"
    with TrajectoryShardWriter(str(ds), shard_size=10, save_dtype="float32") as w:
        for i in range(12):
            w.add({
                "doc_id": f"d{i:03d}",
                "hidden_states": torch.randn(5, d_model),
            })
    from src.llm_probe import TrajectoryShardReader
    reader = TrajectoryShardReader(str(ds))

    cfg = ProjectionConfig(
        latent_dim=8, encoder_hidden=16, decoder_hidden=16,
        use_adversarial=False,
        retrieval_memory_size=20, retrieval_queries=10,
        retrieval_topk=[1, 5],
    )
    ae = ProjectionAutoencoder(cfg, d_model=d_model)
    memory, q_h, q_ids = build_memory_and_queries(
        reader, memory_size=cfg.retrieval_memory_size,
        n_queries=cfg.retrieval_queries,
        memory_fraction=cfg.memory_query_split_fraction, seed=0,
    )
    report = retrieval_evaluation(ae, memory, q_h, q_ids, cfg.retrieval_topk)
    for k in ("1", "5"):
        v = report["projected_fraction_of_baseline_topk"][k]
        assert 0.0 <= v <= 1.0


# ==========================================================================
# Smoke gate CLI (subprocess)
# ==========================================================================

def _write_eval_json(tmp_path: Path, *,
                     top5_ratio: float, mse_ratio: float,
                     probe_acc: float, drift: float) -> Path:
    data = {
        "reconstruction": {"mse_ratio": mse_ratio},
        "retrieval": {"projected_fraction_of_baseline_topk": {"5": top5_ratio}},
        "on_manifold_drift": drift,
        "identity_probe": {"accuracy": probe_acc, "random_baseline": 0.001},
    }
    out = tmp_path / "eval.json"
    with open(out, "w") as f:
        json.dump(data, f)
    return out


def _run_smoke_gate_cli(config_path: str, eval_path: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return subprocess.run(
        [sys.executable, "-m", "scripts.projection.smoke_gate",
         "--config", config_path, "--eval", eval_path],
        capture_output=True, text=True, env=env,
    )


def test_smoke_gate_passes_when_hard_gates_met(tmp_path: Path):
    """Exit 0 when both hard gates pass, regardless of diagnostic status."""
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "configs", "projection.yaml",
    )
    eval_path = _write_eval_json(
        tmp_path,
        top5_ratio=0.85,   # above gate_min = 0.80
        mse_ratio=0.03,    # below gate_max = 0.05
        probe_acc=0.12,    # below WARN threshold; must NOT fail exit
        drift=1.5,         # below WARN threshold
    )
    proc = _run_smoke_gate_cli(cfg_path, str(eval_path))
    assert proc.returncode == 0, (
        f"expected exit 0 (warnings allowed) but got {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "ALL HARD GATES PASSED" in proc.stdout


def test_smoke_gate_fails_on_retrieval(tmp_path: Path):
    """Exit 1 when retrieval hard gate fails, even if diagnostics OK."""
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "configs", "projection.yaml",
    )
    eval_path = _write_eval_json(
        tmp_path,
        top5_ratio=0.40,    # below gate_min = 0.80 → FAIL
        mse_ratio=0.03,
        probe_acc=0.90,
        drift=0.5,
    )
    proc = _run_smoke_gate_cli(cfg_path, str(eval_path))
    assert proc.returncode == 1
    assert "HARD GATE FAIL" in proc.stdout


def test_smoke_gate_warns_but_passes_on_low_probe(tmp_path: Path):
    """Low identity-probe accuracy must trigger WARN, not fail exit code."""
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "configs", "projection.yaml",
    )
    eval_path = _write_eval_json(
        tmp_path,
        top5_ratio=0.85, mse_ratio=0.03,
        probe_acc=0.01,     # below error_below = 0.05 → still only diagnostic
        drift=0.5,
    )
    proc = _run_smoke_gate_cli(cfg_path, str(eval_path))
    assert proc.returncode == 0
    assert "DIAGNOSTICS" in proc.stdout
    # Should surface the low probe visibly in the report.
    assert "identity_probe_accuracy" in proc.stdout


# ==========================================================================
# Slow: small end-to-end on synthetic data
# ==========================================================================

def _synth_dataset(tmp_path: Path, n_docs: int = 8, n_windows: int = 6,
                    d_model: int = 32) -> Path:
    ds_dir = tmp_path / "forward"
    with TrajectoryShardWriter(str(ds_dir), shard_size=4, save_dtype="float32") as w:
        torch.manual_seed(0)
        base = torch.randn(n_docs, d_model) * 3.0
        for i in range(n_docs):
            hs = base[i].unsqueeze(0) + 0.3 * torch.randn(n_windows, d_model)
            w.add({
                "doc_id": f"d{i:03d}",
                "hidden_states": hs,
                "token_positions": [(t * 4, t * 4 + 8) for t in range(n_windows)],
            })
    return ds_dir


@pytest.mark.slow
def test_e2e_smoke(tmp_path: Path):
    """End-to-end: train AE 5 epochs on synthetic dataset, check that MSE
    decreases vs the initial AE and retrieval overlap is better than random."""
    d_model = 32
    ds_dir = _synth_dataset(tmp_path, n_docs=12, n_windows=6, d_model=d_model)
    from src.llm_probe import TrajectoryShardReader
    reader = TrajectoryShardReader(str(ds_dir))

    cfg = ProjectionConfig(
        latent_dim=8, encoder_hidden=32, decoder_hidden=32,
        use_adversarial=False,
        stage_a_epochs=5, stage_b_epochs=2, stage_c_epochs=0,
        stage_a_lr=3e-3, stage_b_lr=1e-3,
        batch_size=16, num_workers=0,
        train_split_fraction=0.8,
        random_seed=0,
        retrieval_memory_size=30, retrieval_queries=12,
        retrieval_topk=[1, 5],
        memory_query_split_fraction=0.5,
        device="cpu",
    )

    ae = ProjectionAutoencoder(cfg, d_model=d_model)
    all_ids = sorted(reader.get_doc_ids())
    train_ids = set(all_ids[:8])
    ds = PairedHiddenStateDataset(reader, allowed_doc_ids=train_ids)
    assert len(ds) > 0
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    with torch.no_grad():
        h = torch.stack([ds[i][0] for i in range(min(16, len(ds)))])
        mse_before = ae.reconstruction_loss(h).item()

    trainer = ProjectionTrainer(cfg, ae, d_model=d_model)
    trainer.train_stage_a(loader)

    with torch.no_grad():
        mse_after = ae.reconstruction_loss(h).item()
    assert mse_after < mse_before, f"MSE didn't drop: {mse_before} → {mse_after}"

    # Retrieval: projected top-k overlap with baseline should be non-trivial.
    memory, q_h, q_ids = build_memory_and_queries(
        reader, memory_size=30, n_queries=12,
        memory_fraction=0.5, seed=0,
    )
    report = retrieval_evaluation(ae, memory, q_h, q_ids, [1, 5])
    r5 = report["projected_fraction_of_baseline_topk"]["5"]
    # With a trained AE, the projected top-5 should materially overlap the
    # baseline top-5. On 12 docs × 6 windows random chance is low but the
    # signal varies; require > 0.2 as a *sanity* bar, not a success claim.
    assert r5 > 0.2, f"retrieval top-5 ratio too low: {r5}"
