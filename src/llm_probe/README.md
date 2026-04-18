# `src/llm_probe/` — v2 Milestone 1

LLM-native probing: extract per-layer hidden-state trajectories from
Llama-3-8B running over Wikipedia, package them as three dataset
families — **forward**, **branching**, **reversed** — and persist them
as shard-based torch files with a JSON manifest. These trajectories
are the raw material for the v2 geometric stack (M2–M6).

## Rationale

The v1 pipeline trained a small Lorentzian world model on
sentence-transformer *embeddings*. That worked as a geometric
proof-of-concept but is **one step removed** from the actual claim of
the paper: LLMs *already* carry a latent world model; we only need to
surface it. In v2 the trajectory is the hidden state at a single
middle layer of the LM itself — no surrogate encoder — so every
downstream metric (geometry fit, branching, time reversal,
interventional causality) speaks about the LM directly.

## The three dataset families

| Dataset       | Purpose                                  | Size default      |
|---------------|-------------------------------------------|-------------------|
| **forward**   | Trajectories of *different continuations* of the same prompt. Tests that the latent captures multiple plausible futures. | 5000 articles × 8 trajectories |
| **branching** | Pairs ``(τ_a, τ_b)`` that share a prefix and diverge at a high-entropy step by swapping one top-k token. Tests whether alternative continuations are separated in the latent. | 2000 articles × 3 pairs |
| **reversed**  | The SAME passage fed forward and in reversed-token order. Tests the direction-sensitivity of the latent ("arrow of narrative"). | 5000 articles |

The three sets are allowed to **overlap at the article level**: the
same Wikipedia article may appear in more than one dataset because
each probes a different property of the LM's representation.

## Quick start

```bash
# 0. One-off: pin dependencies
pip install -r requirements.txt

# 1. Plan check — no GPU, no downloads
python -m scripts.probe.build_trajectories --config configs/probe.yaml \
    --dataset all --dry-run

# 2. End-to-end smoke on H100 (~15 min)
export HF_TOKEN=hf_...                 # Llama-3 is gated
python -m scripts.probe.build_trajectories --config configs/probe_smoke.yaml \
    --dataset all

# 3. Inspect one dataset afterwards
python -m scripts.probe.inspect_dataset --manifest data/llm_probe/forward/manifest.json

# 4. Full run (3 commands, ~24-30h total on H100)
python -m scripts.probe.build_trajectories --config configs/probe.yaml --dataset forward    # ~16h
python -m scripts.probe.build_trajectories --config configs/probe.yaml --dataset branching  # ~6h
python -m scripts.probe.build_trajectories --config configs/probe.yaml --dataset reversed   # ~4h
```

`--resume` is idempotent: on restart, any ``doc_id`` already present
in the shard manifest is skipped, so an interrupted overnight run
completes without re-doing work.

## Storage layout

```
data/llm_probe/
├── article_pool.json              # shared cache (all datasets)
├── forward/
│   ├── manifest.json              # schema below
│   ├── shard_00000.pt             # torch.save({"items": [...], "shard_idx": 0})
│   ├── shard_00001.pt
│   └── ...
├── branching/
│   ├── manifest.json
│   └── shard_*.pt
├── reversed/
│   ├── manifest.json
│   └── shard_*.pt
└── validation/
    ├── manifest.json
    └── shard_*.pt
```

### `manifest.json`

```json
{
  "dataset_dir": "data/llm_probe/forward",
  "n_shards": 50,
  "n_items_total": 40000,
  "shards": [
    {"idx": 0, "path": "shard_00000.pt", "n_items": 100, "doc_ids": ["wiki_...", ...]},
    ...
  ],
  "save_dtype": "float16",
  "created_at": "2026-04-18T12:00:00Z"
}
```

### Shard item schema

**forward / validation** — one item per `(article, trajectory_idx)`:

```python
{
    "doc_id": str,
    "trajectory_idx": int,
    "prompt_tokens": Tensor,        # (prompt_tokens,)
    "full_tokens": Tensor,          # (prompt_tokens + continuation,)
    "hidden_states": Tensor,        # (n_windows, d_model) fp16
    "token_positions": list[(int, int)],
    "seq_len": int,
}
```

**branching** — one item per `(article, pair_idx)`:

```python
{
    "doc_id": str,
    "pair_idx": int,
    "branching_point": int,         # token position where τ_a and τ_b diverge
    "original_token": int,
    "intervention_token": int,
    "trajectory_a": {"hidden_states", "full_tokens", "token_positions", "seq_len"},
    "trajectory_b": {"hidden_states", "full_tokens", "token_positions", "seq_len"},
}
```

**reversed** — one item per article:

```python
{
    "doc_id": str,
    "forward_tokens": Tensor,
    "reversed_tokens": Tensor,      # == forward_tokens.flip(0)
    "forward_hidden": Tensor,       # (n_windows, d_model) fp16
    "reversed_hidden": Tensor,
    "forward_positions": list, "reversed_positions": list,
    "forward_seq_len": int, "reversed_seq_len": int,
}
```

## Disk & timing estimates (H100 80 GB, Llama-3-8B, fp16)

Assumptions: `window_size=32`, `window_stride=16`, ≈30 windows per
trajectory, `d_model=4096`, on-disk `float16` (2 bytes per value).

| Dataset    | N items      | Bytes/item | Total disk |
|------------|--------------|------------|------------|
| forward    | 5000 × 8 = 40k   | 30 × 4096 × 2 = 246 KB | **10 GB** |
| branching  | 2000 × 3 = 6k pairs × 2 trajs = 12k | 246 KB | **3 GB** |
| reversed   | 5000 × 2 = 10k | 246 KB | **2.5 GB** |
| **Total**  | ~62k items   | —          | **~15 GB** |

Wall-clock guidance:

| Phase        | H100 time (approx.) |
|--------------|---------------------|
| forward      | 16 h                |
| branching    | 6 h (extra forward per pair for entropy + alt-token regeneration) |
| reversed     | 4 h                 |
| **total**    | **24–30 h**         |

## Resuming an interrupted run

Every shard flush atomically updates `manifest.json`. The writer
reloads the manifest on `__init__`, so relaunching with the same
`--dataset` and `--resume` picks up from the last persisted item:

```bash
python -m scripts.probe.build_trajectories --config configs/probe.yaml \
    --dataset forward --resume
```

The failure threshold (default 10% per dataset) still applies across
the resumed counter.

## Relationship to v1

v1 lives at the `v1-final` git tag. That code — MiniLM embeddings →
PCA preprocessing → Lorentzian metric on 16-D latent → candidate-set
matching → M1/M4/M5/coherence probe evaluation — is untouched by v2
and can be reproduced with:

```bash
git checkout v1-final
pip install -r requirements.txt
# …follow the v1 README
```

v2 milestones are additive through M1. Starting with M2, the v1
`src/{models,training,evaluation}` code and its tests begin to be
retired and replaced.
