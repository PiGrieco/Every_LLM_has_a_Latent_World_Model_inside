"""Unit tests for src/llm_probe (v2 Milestone 1).

Fast tests (default) use only synthetic tensors and never download a
model. Slow tests (``-m slow``) use ``sshleifer/tiny-gpt2`` — a ~1 MB
model that covers the generate + hook path end-to-end.

Fast tests are expected to complete in under 15 s; slow tests under
8 min (most of that is the one-off tiny-gpt2 download).
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path

import pytest
import torch

# Make the repo importable both when invoked from the repo root and from tests/
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.llm_probe import (
    ProbeConfig,
    TrajectoryShardReader,
    TrajectoryShardWriter,
    assign_articles_to_datasets,
    extract_trajectory_states,
    install_activation_hook,
    validate_branching_divergence,
    validate_reversed_differ,
    validate_trajectory_statistics,
    window_pool,
)


# --------------------------------------------------------------------------
# Fast tests (no model)
# --------------------------------------------------------------------------

def test_window_pool_shapes():
    """Standard case: seq_len > window, non-trivial number of windows."""
    seq_len, d, window, stride = 100, 16, 32, 16
    hs = torch.randn(seq_len, d)
    pooled, positions = window_pool(hs, window, stride)
    expected_n = (seq_len - window) // stride + 1
    assert pooled.shape == (expected_n, d)
    assert len(positions) == expected_n
    for start, end in positions:
        assert end - start == window
        assert 0 <= start < seq_len


def test_window_pool_empty():
    """seq_len < window → (0, d) tensor + empty list, not an exception."""
    hs = torch.randn(5, 8)
    pooled, positions = window_pool(hs, window=10, stride=2)
    assert pooled.shape == (0, 8)
    assert positions == []


def test_window_pool_stride_one():
    """stride=1 gives seq_len - window + 1 windows."""
    seq_len, d, window = 64, 12, 10
    hs = torch.randn(seq_len, d)
    pooled, positions = window_pool(hs, window=window, stride=1)
    assert pooled.shape == (seq_len - window + 1, d)
    # Means are actually computed (not just copies of the first window)
    assert torch.allclose(pooled[0], hs[:window].mean(dim=0))
    assert torch.allclose(pooled[-1], hs[-window:].mean(dim=0))


def test_config_defaults():
    """ProbeConfig() must be instantiable with no arguments."""
    cfg = ProbeConfig()
    assert cfg.probe_layer == 20
    assert cfg.window_size == 32
    assert cfg.window_stride == 16
    assert cfg.save_dtype in ("float16", "fp16")


def test_storage_roundtrip(tmp_path: Path):
    """Write 20 tiny items, reload them, compare tensors and doc_ids."""
    dataset_dir = tmp_path / "forward"
    d_model = 8
    items = []
    for i in range(20):
        items.append({
            "doc_id": f"wiki_{i:03d}",
            "trajectory_idx": 0,
            "hidden_states": torch.randn(5, d_model),
            "seq_len": 160,
        })

    with TrajectoryShardWriter(str(dataset_dir), shard_size=7, save_dtype="float16") as w:
        for it in items:
            w.add(it)

    reader = TrajectoryShardReader(str(dataset_dir))
    assert len(reader) == 20
    round_tripped = list(reader.iter_items())
    assert len(round_tripped) == 20
    for orig, got in zip(items, round_tripped):
        assert got["doc_id"] == orig["doc_id"]
        # On-disk dtype is fp16, so we compare in fp32 with tolerance.
        assert torch.allclose(
            got["hidden_states"].float(), orig["hidden_states"].float(), atol=1e-2,
        )


def test_storage_manifest_integrity(tmp_path: Path):
    """Manifest must track n_shards, n_items_total, and doc_ids exactly."""
    dataset_dir = tmp_path / "ds"
    with TrajectoryShardWriter(str(dataset_dir), shard_size=3) as w:
        for i in range(7):
            w.add({"doc_id": f"d{i}", "hidden_states": torch.randn(2, 4)})
    reader = TrajectoryShardReader(str(dataset_dir))
    m = reader.manifest
    assert m["n_items_total"] == 7
    # 7 items, shard_size=3 → 3 shards of sizes 3, 3, 1
    assert m["n_shards"] == 3
    sizes = [s["n_items"] for s in m["shards"]]
    assert sizes == [3, 3, 1]
    assert reader.get_doc_ids() == {f"d{i}" for i in range(7)}


def test_assign_articles_overlap():
    """Assignments may share articles across datasets (overlap, not partition)."""
    cfg = ProbeConfig(
        n_articles_forward=10,
        n_articles_branching=10,
        n_articles_reversed=10,
        n_articles_validation=3,
        corpus_seed=7,
    )
    pool = [
        {"doc_id": f"a{i:03d}", "title": f"t{i}", "text": "x", "token_count": 1000}
        for i in range(15)
    ]
    split = assign_articles_to_datasets(pool, cfg)
    forward_ids = {a["doc_id"] for a in split["forward"]}
    branching_ids = {a["doc_id"] for a in split["branching"]}
    reversed_ids = {a["doc_id"] for a in split["reversed"]}
    # Each dataset gets 10 of the 15; overlap must exist.
    assert len(forward_ids) == 10
    assert len(branching_ids) == 10
    assert len(reversed_ids) == 10
    assert forward_ids & branching_ids, "No overlap — sampling should be independent"
    # validation is a sub-prefix of forward
    assert {a["doc_id"] for a in split["validation"]} <= forward_ids
    assert len(split["validation"]) == 3


# --------------------------------------------------------------------------
# Slow tests — require a downloaded model (tiny-gpt2)
# --------------------------------------------------------------------------

_TINY = "sshleifer/tiny-gpt2"


def _tiny_stack(tmp_path: Path, **overrides) -> tuple:
    """Load tiny-gpt2 + tokenizer + a ProbeConfig sized to its geometry."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_TINY)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(_TINY).eval()
    # tiny-gpt2: 2 layers, d_model=2; irrelevant dims but enough for the test path.
    # We still monkey-patch .model.layers below so validate_model_structure
    # and our hook behave as if we had a real decoder stack.
    cfg = ProbeConfig(
        model_name=_TINY,
        device="cpu",
        dtype="float32",
        save_dtype="float32",
        probe_layer=0,
        window_size=4,
        window_stride=2,
        prompt_tokens=8,
        continuation_tokens=8,
        k_trajectories=2,
        n_articles_forward=2,
        n_articles_branching=1,
        n_articles_reversed=1,
        n_articles_validation=2,
        n_pairs_per_article=1,
        branching_window_start=0,
        branching_window_end=4,
        reversed_passage_tokens=16,
        shard_size=4,
        output_dir=str(tmp_path),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)

    # tiny-gpt2 exposes model.transformer.h instead of model.model.layers.
    # We re-expose it in the shape the probe expects — read-only.
    import torch.nn as nn
    if not hasattr(model, "model"):
        class _Adapter(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.layers = inner.transformer.h
                # Ensure each layer has .self_attn and .mlp attributes.
                for layer in self.layers:
                    if not hasattr(layer, "self_attn"):
                        layer.self_attn = layer.attn
                    if not hasattr(layer, "mlp"):
                        layer.mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        model.model = _Adapter(model)
    return model, tok, cfg


@pytest.mark.slow
def test_forward_smoke(tmp_path: Path):
    """2 articles, K=2 trajectories each: shapes sane, no NaN."""
    from src.llm_probe.trajectory_generator import generate_forward_trajectories

    model, tok, cfg = _tiny_stack(tmp_path)
    handle, captured = install_activation_hook(model, cfg.probe_layer)
    try:
        articles = [{"doc_id": "wiki_aaa", "title": "t", "text": "Wikipedia is a free online encyclopedia."},
                    {"doc_id": "wiki_bbb", "title": "u", "text": "Trajectories are captured per layer."}]
        total = 0
        for i, art in enumerate(articles):
            trajs = generate_forward_trajectories(
                model, tok, art, cfg, doc_idx=i,
                hook_handle=handle, captured_list=captured,
            )
            # Not all articles may produce valid trajectories for such a small
            # prompt; at minimum we need some items across the two articles.
            if trajs:
                total += len(trajs)
                for t in trajs:
                    assert torch.isfinite(t["hidden_states"]).all()
                    assert t["hidden_states"].shape[-1] == model.config.hidden_size
        assert total > 0, "no forward trajectories were produced"
    finally:
        handle.remove()


@pytest.mark.slow
def test_branching_smoke(tmp_path: Path):
    """Branching pair diverges after the branching point (|a-b| > 0)."""
    from src.llm_probe.trajectory_generator import generate_branching_pairs

    model, tok, cfg = _tiny_stack(
        tmp_path, n_pairs_per_article=1, entropy_threshold=0.1,
    )
    handle, captured = install_activation_hook(model, cfg.probe_layer)
    try:
        article = {
            "doc_id": "wiki_branch",
            "title": "t",
            "text": "A single article used for branching generation at tiny scale.",
        }
        pairs = generate_branching_pairs(
            model, tok, article, cfg, doc_idx=0,
            hook_handle=handle, captured_list=captured,
        )
        if pairs is None:
            pytest.skip("tiny-gpt2 could not produce a branching pair on this toy input")
        assert pairs, "empty pair list"
        p = pairs[0]
        a = p["trajectory_a"]["hidden_states"].float()
        b = p["trajectory_b"]["hidden_states"].float()
        T = min(a.shape[0], b.shape[0])
        assert T > 0
        divergence = (a[:T] - b[:T]).norm(dim=-1).mean().item()
        assert divergence > 0.0
    finally:
        handle.remove()


@pytest.mark.slow
def test_reversed_smoke(tmp_path: Path):
    """Forward and reversed hidden states differ (cosine sim < 0.99)."""
    from src.llm_probe.trajectory_generator import extract_reversed_pair

    model, tok, cfg = _tiny_stack(tmp_path, reversed_passage_tokens=24)
    handle, captured = install_activation_hook(model, cfg.probe_layer)
    try:
        article = {
            "doc_id": "wiki_rev",
            "title": "t",
            "text": (
                "Reversing a token sequence should produce distinctly "
                "different hidden states because attention is causal."
            ) * 4,
        }
        item = extract_reversed_pair(
            model, tok, article, cfg, doc_idx=0,
            hook_handle=handle, captured_list=captured,
        )
        if item is None:
            pytest.skip("article too short for reversed passage after tokenization")
        fwd = item["forward_hidden"].float()
        rev = item["reversed_hidden"].float()
        T = min(fwd.shape[0], rev.shape[0])
        assert T > 0
        cos = torch.nn.functional.cosine_similarity(
            fwd[:T], rev.flip(0)[-T:], dim=-1,
        ).mean().item()
        assert cos < 0.99, f"forward/reversed too similar (cos={cos:.3f})"
    finally:
        handle.remove()


@pytest.mark.slow
def test_validation_functions(tmp_path: Path):
    """Run the three validator functions on a synthetic mini-dataset."""
    d_model = 8
    # forward
    fwd_dir = tmp_path / "forward"
    with TrajectoryShardWriter(str(fwd_dir), shard_size=4) as w:
        for i in range(6):
            w.add({
                "doc_id": f"f{i}",
                "hidden_states": torch.randn(5, d_model) * 3.0,
            })
    # branching
    br_dir = tmp_path / "branching"
    with TrajectoryShardWriter(str(br_dir), shard_size=3) as w:
        for i in range(4):
            a = torch.randn(6, d_model)
            b = a + 0.5 * torch.randn_like(a)  # ensure divergence > 0
            w.add({
                "doc_id": f"b{i}",
                "trajectory_a": {"hidden_states": a},
                "trajectory_b": {"hidden_states": b},
            })
    # reversed
    rv_dir = tmp_path / "reversed"
    with TrajectoryShardWriter(str(rv_dir), shard_size=3) as w:
        for i in range(4):
            fwd = torch.randn(6, d_model)
            rev = torch.randn(6, d_model)  # random, definitely not cosine-similar
            w.add({
                "doc_id": f"r{i}",
                "forward_hidden": fwd,
                "reversed_hidden": rev,
            })

    stats_fwd = validate_trajectory_statistics(TrajectoryShardReader(str(fwd_dir)))
    stats_br = validate_branching_divergence(TrajectoryShardReader(str(br_dir)))
    stats_rv = validate_reversed_differ(TrajectoryShardReader(str(rv_dir)))

    assert stats_fwd["fraction_finite"] == 1.0
    assert stats_br["mean_divergence"] > 0.0
    assert stats_rv["mean_cosine_similarity"] < 0.95
