# `src/projection/` — v2 Milestone 2

Projection autoencoder **Φ / Ψ** with retrieval-based readout. The
goal of M2 is NOT a generative token decoder (that is M3). The goal
is to show that the latent state is *informative*: given a latent
`s_t`, we should be able to find the original hidden state `h_t` (and
thus the underlying text) inside a memory of other hidden states with
accuracy that substantially preserves what raw-`h` retrieval would
achieve. If that holds, M3 can safely build a generative decoder on
top; if not, geometry on the latent would be a castle on sand.

## What M2 produces

- `Encoder` Φ: `h ∈ ℝ^{4096} → s ∈ ℝ^{D}`.
- `Decoder` Ψ: `s → ĥ ∈ ℝ^{4096}`.
- (Optional) `Discriminator` D for adversarial on-manifold training.
- A retrieval baseline, a probe baseline, an on-manifold drift
  measurement.
- A rigid go/no-go **smoke gate** that exits 0 when the two hard
  gates pass and 1 otherwise (diagnostics are reported as WARN/OK).

## Architecture

```
h (4096)
   │
   ▼
  fc1 (4096 → 512)  — GELU — dropout — fc2 (512 → D)
   │                                     +
   │                                     skip = Linear(4096 → D, bias=False)
   │                                     ▼
   └──────────────────────────── LayerNorm(D) = s

s (D)
   │
   ▼
  fc1 (D → 512)     — GELU — dropout — fc2 (512 → 4096)
   │                                     +
   │                                     skip = Linear(D → 4096, bias=False)
   │                                     ▼
   └──────────────────────────────────────────── ĥ
```

`D` defaults to 64; if the intrinsic dim estimate lands outside
`[intrinsic_dim_min, intrinsic_dim_max] = [32, 256]`, we clamp with a
recorded ``"clamped": true`` flag and continue. The whole run is
idempotent in this field through ``configs/projection.yaml ::
latent_dim`` and the ``outputs/projection/intrinsic_dim.json`` cache.

## Three-stage training

1. **Stage A — pure reconstruction.** `L_A = mean ||h - Ψ(Φ(h))||^2`.
   Establishes the bottom-line reconstruction path. Adam, lr 3e-4,
   early stop on 3-epoch plateau.
2. **Stage B — +local isometry.** Consistency loss
   `(||Φ(h_t) - Φ(h_{t+1})|| - α · ||h_t - h_{t+1}||)^2` with `α` a
   learnable positive scalar (initialised to 0.1). Encourages the
   latent to preserve small-scale neighbourhood distances. Adam, lr
   1e-4, `λ_consistency = 0.5`.
3. **Stage C — optional adversarial on-manifold (WGAN-GP).** Only if
   `use_adversarial=True`. Self-aborts if the discriminator
   saturates for two consecutive epochs (`D_acc > 0.95`), falling
   back gracefully to A+B. Default is OFF — adversarial training is
   notoriously fragile, and M3 does not depend on Ψ landing perfectly
   in-distribution because the M3 suffix decoder acts as a denoiser.

## Decoder strategy — why NOT `LM_head(Ψ(s))`

**We deliberately do not build a generative decoder in M2.** At layer
`ℓ=20` the residual stream is *not* in the unembedding space:
applying `LM_head` directly would skip 12 layers of downstream
processing. In M2 we verify the latent via *retrieval* (cosine
nearest-neighbour in a hidden-state memory). The generative decoder
comes in M3+ as a **suffix decoder** that passes `Ψ(s)` through the
remaining layers `ℓ+1..L`, the model's final norm, and only then
`LM_head`. The residual stream becomes the unembedding-space vector
*as a by-product of the model's own denoising*, not a bolted-on
linear projection.

## Gate policy (post-review)

### HARD gates — these block M3 promotion

| Gate | Meaning | Default threshold |
|---|---|---|
| `retrieval_top5_ratio` | Mean fraction of *baseline top-5 neighbours* that the projected retrieval also recovers. | `≥ 0.80` |
| `reconstruction_mse_ratio` | `MSE(h, Ψ(Φ(h))) / var(h)` on held-out queries. | `≤ 0.05` |

Both test the stated purpose of M2: a non-destructive compression with
a verifiable readout. Fail either → smoke gate exits 1 → M3 waits.

### DIAGNOSTICS — warn only, never gate

| Diagnostic | Warn / Error thresholds |
|---|---|
| `identity_probe_accuracy` | warning < 0.20; error < 0.05 |
| `on_manifold_drift` | warning > 2.0 |

### Design decisions revised

Why only two hard gates:

- Retrieval top-5 and reconstruction MSE *directly* test what M2 is
  for: non-destructive compression with a verifiable readout. If they
  pass, M3 has a solid floor to stand on.
- Identity probe with 1 000 classes and ~8 examples per class has
  high variance; a hard threshold at 0.50 was over-aggressive and
  would reject perfectly usable latents on noise alone.
- On-manifold drift depends on adversarial training, which is
  fragile and optional. Making it a hard gate would couple M2
  progress to a notoriously unstable training loop we deliberately
  left off by default.

When the diagnostics flag something, the smoke-gate report prints a
WARN line and the paper records the fact transparently. It does **not**
stop M3.

## Memory / query disjointness

`build_memory_and_queries` partitions the M1 forward dataset's
`doc_id`s with a deterministic permutation: 60 % go to the memory,
40 % to queries. Memory and query `doc_id`s are *strictly disjoint*
both pre-sample (ValueError on overlap) and post-sample (runtime
assert). This eliminates the trivial "retrieve yourself" failure mode
and means a top-k hit reflects genuine structure-preserving
compression, not memorisation.

## Quick start

```bash
# 0. Prerequisite: M1 must be complete (data/llm_probe/*/manifest.json exist).

# 1. Estimate intrinsic dimension (cheap; writes intrinsic_dim.json).
python -m scripts.projection.estimate_dim \
    --config configs/projection.yaml \
    --n-sample 50000

# Review outputs/projection/intrinsic_dim.json. If "clamped": true, consider
# adjusting intrinsic_dim_min/max before spending hours on training.

# 2. Train (Stage A+B by default; pass --skip-adversarial explicitly if the
#    config enables it).
python -m scripts.projection.train --config configs/projection.yaml

# 3. Evaluate on held-out data.
python -m scripts.projection.evaluate \
    --config configs/projection.yaml \
    --checkpoint ./checkpoints/projection/final.pt \
    --output ./outputs/projection/eval.json

# 4. Rigid gate — exit 0 means M3 can proceed.
python -m scripts.projection.smoke_gate \
    --config configs/projection.yaml \
    --eval ./outputs/projection/eval.json
echo "gate exit code: $?"
```

## Known risks

- **Intrinsic dim >> 64.** Layer-20 Llama-3 may use 200+ effective
  dimensions for discourse. Mitigation: `estimate_dim.py` runs first;
  if the estimate is > 128, raise `latent_dim` in the config before
  training.
- **Discriminator diverges.** Adversarial training is opt-in and
  Stage C self-aborts on saturation.
- **Retrieval fails even with low MSE.** Φ might compress in a
  non-discriminative way (all projections land in a narrow band).
  Mitigation: Stage B's consistency loss penalises distance collapse.
  If retrieval still fails, M2 is genuinely not converged and M3
  doesn't get to start.

## Expected compute

- Stage A + Stage B on full forward dataset (~1.2 M states): **~8 h**
  on one H100.
- Stage C (optional): adds **~2-3 h**.
- Evaluation (retrieval + probe + drift + MSE): **< 10 min**.
- Total full-run budget: **10-13 h** on H100.

Smoke run on the M1 smoke dataset: **< 20 min** end to end.
