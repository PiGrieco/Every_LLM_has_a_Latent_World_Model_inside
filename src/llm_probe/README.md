# `src/llm_probe/` — v2 Milestone 1

LLM-native trajectory probing: extract per-layer hidden-state
trajectories from Llama-3-8B over Wikipedia and package them as three
dataset families (**forward**, **branching**, **reversed**). Shard
files are accompanied by a rich JSON manifest that pins library
versions, model commit sha, and a hash of the probe configuration so
the data can always be retraced to the exact run that produced it.

## Purpose

The v1 pipeline trained a small Lorentzian world model on
sentence-transformer *embeddings*. That worked as a geometric
proof-of-concept but is one step removed from the paper's central
claim: LLMs already carry a latent world model; we need only surface
it. In v2 the trajectory is the hidden state at a single middle layer
of the LM itself — no surrogate encoder — so every downstream metric
(geometry fit, branching, time reversal, interventional causality) is
about the LM directly.

## The three dataset families

| Dataset       | Purpose | Size default |
|---------------|---------|--------------|
| **forward**   | K continuations from the same prompt. Tests whether the latent captures multiple plausible futures. | 5000 articles × 8 trajectories |
| **branching** | Pairs `(τ_a, τ_b)` sharing a prefix, diverging at a high-entropy step via swapping one top-k token. Tests whether alternative continuations separate in the latent. | 2000 articles × 3 pairs |
| **reversed**  | The SAME passage fed forward and reversed. Tests direction-sensitivity of the latent ("arrow of narrative"). | 5000 articles |

Article selection for *branching* uses a **narrative-weighted sample**
(`cfg.branching_narrative_weight = 3.0`). Wikipedia articles whose
lead sentence matches a biography / event / process pattern are
preferred; definition stubs, which structurally don't branch, remain
eligible but with baseline weight. The narrative-corpus question will
be revisited in M6 (likely TinyStories or selected BookCorpus for
intervention experiments).

## Design decisions

- **Why layer 20?** Mid-late residual stream of Llama-3-8B (32-layer
  model). Empirically, layers around the ¾ mark carry the richest
  discourse-level representations while still being before the final
  two layers that specialise for next-token prediction.
- **Why window-pool?** The per-token hidden state is too fine a
  trajectory to match a paragraph-level geometric picture; mean-pool
  over 32-token windows with stride 16 produces a clean, fixed-size
  state that tracks discourse rather than tokens.
- **Why separate forward passes?** `generate()` does expose hidden
  states via `output_hidden_states=True`, but only at the end and for
  all layers simultaneously (expensive memory, messy hook capture).
  We run `generate()` to get token sequences, then a separate
  `forward()` per sequence with the hook active. This is the clean
  path; it costs an extra forward per trajectory (factored into the
  revised timing estimate below).
- **Why fp16 on disk?** Typical probe hidden-state magnitudes live
  in the tens; fp16 retains ~3 decimal digits of precision and
  halves storage.

## Not included in M1 (by design)

- **No decoder Ψ.** The state is extracted but not inverted. The
  decoding strategy for M2+ starts with **retrieval**: nearest-neighbor
  in a hidden-state memory (proves the state is *informative*). Only
  in M3+ do we build a generative decoder, and it will NOT be
  `LM_head(Ψ(s))` directly — the layer-20 residual stream is not in
  the unembedding space, so applying `LM_head` to it skips 12 layers
  of downstream processing. The proper form is a **suffix decoder**:
  pass `Ψ(s)` through remaining layers ℓ+1…L of the LLM, then
  `final_norm`, then `LM_head`.
- **No metric or dynamics.** Geometry learning starts at M3. M2
  builds the projection Φ / Ψ plus the retrieval-based decoder.
- **No null-cone objective.** The earlier sketch of using
  "deterministic sampling ⇒ null-likeness" as a training target is
  retracted; determinism of token sampling does not imply null-like
  geometry and that extra structure would be an unverified
  assumption. The null cone, if anything, will be a probe-only
  diagnostic in M3+ (measure where temperature-0 transitions fall
  relative to the learned cone structure).

## Quick start

```bash
# 0. Pin deps (once)
pip install -r requirements.txt

# 1. Plan check — no GPU, no downloads
python -m scripts.probe.build_trajectories --config configs/probe.yaml \
    --dataset all --dry-run

# 2. Smoke run on H100 (~20-30 min including generate+hook double pass)
export HF_TOKEN=hf_...                 # Llama-3 is gated
python -m scripts.probe.build_trajectories --config configs/probe_smoke.yaml \
    --dataset all

# 3. RIGID SMOKE GATE — exit code 0 means "go to full run"
python -m scripts.probe.smoke_gate \
    --output-dir ./data/llm_probe \
    --config configs/probe_smoke.yaml
echo "exit code: $?"                   # must be 0

# 4. Inspect a dataset (with --show-metadata to dump pinning info)
python -m scripts.probe.inspect_dataset \
    --manifest data/llm_probe/forward/manifest.json --show-metadata

# 5. Full run (3 commands; 1.5-3 days total on H100)
python -m scripts.probe.build_trajectories --config configs/probe.yaml --dataset forward
python -m scripts.probe.build_trajectories --config configs/probe.yaml --dataset branching
python -m scripts.probe.build_trajectories --config configs/probe.yaml --dataset reversed

# 6. Re-run the rigid gate against the full datasets
python -m scripts.probe.smoke_gate --output-dir ./data/llm_probe \
    --config configs/probe.yaml
```

`--resume` on `build_trajectories` is idempotent: on restart, any
`doc_id` already present in the shard manifest is skipped.

## The rigid smoke gate

`scripts/probe/smoke_gate.py` is the single go/no-go checkpoint for
letting M2 start. It walks every dataset under `--output-dir`, runs
the applicable validator, and exits with:

- **0** iff every check on every dataset passed.
- **1** otherwise (with a per-check breakdown printed).

Thresholds live in the probe config as `gate_*` fields so they travel
with the data (they're also recorded inside each manifest's
`probe_config_snapshot`). A failing gate is not a warning; the
convention is "do not start M2 with exit ≠ 0".

## Storage layout

```
data/llm_probe/
├── article_pool.json
├── forward/
│   ├── manifest.json              # schema below
│   ├── shard_00000.pt             # {"items": [...], "shard_idx": 0}
│   └── shard_*.pt
├── branching/
├── reversed/
└── validation/
```

### `manifest.json`

```json
{
  "dataset_dir": "data/llm_probe/forward",
  "dataset_name": "forward",
  "n_shards": 50,
  "n_items_total": 40000,
  "shards": [
    {"idx": 0, "path": "shard_00000.pt", "n_items": 100, "doc_ids": ["wiki_...", ...]}
  ],
  "save_dtype": "float16",
  "created_at": "2026-04-18T...Z",
  "environment": {
    "python_version": "3.11.8",
    "torch_version": "2.4.1",
    "transformers_version": "4.44.2",
    "cuda_version": "12.1",
    "gpu_name": "NVIDIA H100 80GB HBM3"
  },
  "model_metadata": {
    "model_name": "meta-llama/Meta-Llama-3-8B",
    "model_revision": "main",
    "model_commit_sha": "abc123...",
    "probe_layer_idx": 20,
    "d_model": 4096,
    "config_hash": "f01e..."
  },
  "probe_config_snapshot": { /* the full ProbeConfig, minus hf_token */ }
}
```

## Compute & storage on H100

Revised from the initial estimate: the generate→re-forward strategy
plus variance on long articles makes the one-pass-per-trajectory
budget non-trivial. Target hardware: 1×H100 80 GB, Llama-3-8B fp16.

| Phase        | Revised H100 time |
|--------------|-------------------|
| forward      | ~30-50 h          |
| branching    | ~10-15 h (two generate + two re-forward per pair) |
| reversed     | ~5-8 h            |
| **total**    | **1.5–3 days** (not 24-30 h) |

Disk budget is unchanged (~15 GB for the three datasets).

## Resuming / rollback

- Resume: `build_trajectories --resume` skips items already in the
  manifest.
- Rollback v1: `git checkout v1-final` recovers the pre-M1 state,
  including the old Colab notebook, for reproducing v1 experiments.
