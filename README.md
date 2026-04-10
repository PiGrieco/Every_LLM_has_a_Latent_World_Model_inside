# Every LLM Has an Implicit World Model Inside It

### A Lorentzian Geometric Framework for Making It Explicit

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://github.com/PiGrieco/Every_LLM_has_a_Latent_World_Model_inside)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Piermatteo Grieco** (April 2026)
>
> Large language models behave as powerful *implicit* world models, yet this world model remains distributed in the network weights with no explicit notion of state, dynamics or spacetime structure. We propose a framework to transform any LLM — or merely its training corpus — into an **explicit geometric world model**: a latent manifold endowed with a learned **Lorentzian metric** that separates *time-like* (narrative-advancing) from *space-like* (narrative-branching) directions, a **geometric–semantic Lagrangian**, and a **Gibbs measure** over worldlines inducing transition probabilities.

---

## Table of Contents

- [Key Idea](#key-idea)
- [Quick Start (Google Colab)](#quick-start-google-colab)
- [Installation (Local)](#installation-local)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Experimental Protocol](#experimental-protocol)
- [Execution Guide](#execution-guide)
- [Hardware Requirements](#hardware-requirements)
- [Design Decisions](#design-decisions)
- [Downstream Applications](#downstream-applications)
- [Citation](#citation)
- [References](#references)

---

## Key Idea

Standard Riemannian or Euclidean geometries treat all directions equally — they have no notion of *forward vs. sideways*. By introducing a **Lorentzian signature** `(−, +, …, +)` on the latent manifold, we get **causal cones** for free:

| Direction | Signature | Interpretation |
|-----------|-----------|----------------|
| **Time-like** | Δs<sup>T</sup> g<sub>θ</sub>(s) Δs < 0 | Narrative advances forward |
| **Space-like** | Δs<sup>T</sup> g<sub>θ</sub>(s) Δs > 0 | Alternative branches / counterfactuals |
| **Null** | Δs<sup>T</sup> g<sub>θ</sub>(s) Δs = 0 | Boundary of the light cone |

The framework builds five components on top of any base encoder:

1. **Latent narrative manifold** M ⊂ ℝ<sup>D</sup>
2. **Learned Lorentzian metric** g<sub>θ</sub>(s) = A<sup>met</sup><sub>θ</sub>(s)<sup>T</sup> η A<sup>met</sup><sub>θ</sub>(s), with η = diag(−1, 1, …, 1)
3. **Geometric–semantic Lagrangian** L<sub>θ</sub> = λ<sub>g</sub> L<sub>g</sub> + λ<sub>sem</sub> L<sub>sem</sub>
4. **Gibbs transition kernel** K<sub>θ</sub>(s' | s) ∝ exp(−L<sub>θ</sub>(s, s'))
5. **Parametric world model** q<sub>θ</sub>(s' | s) ≈ K<sub>θ</sub>(s' | s)

---

## Quick Start (Google Colab)

Open a Colab notebook with **A100 GPU** runtime and run:

```python
# Cell 1 — Clone and install
!git clone https://github.com/PiGrieco/Every_LLM_has_a_Latent_World_Model_inside.git
%cd Every_LLM_has_a_Latent_World_Model_inside
!pip install -q torch transformers sentence-transformers datasets \
    faiss-cpu matplotlib seaborn scikit-learn pyyaml tqdm umap-learn

# Cell 2 — D0 sanity check (< 5 min on A100)
!python -m scripts.train --dataset d0_synthetic --geometry lorentzian \
    --stage2_epochs 30 --stage3_epochs 70

# Cell 3 — Run all baselines on D0 + D1 (~30 min)
!python -m scripts.run_baselines

# Cell 4 — View results
import json, pprint
with open("outputs/baseline_comparison.json") as f:
    results = json.load(f)
pprint.pprint(results)
```

---

## Installation (Local)

```bash
git clone https://github.com/PiGrieco/Every_LLM_has_a_Latent_World_Model_inside.git
cd Every_LLM_has_a_Latent_World_Model_inside
pip install -r requirements.txt
```

> **Note:** Use `faiss-cpu` instead of `faiss-gpu` if you don't have a CUDA GPU.

---

## Architecture Overview

The pipeline follows the structure described in Section 1.3 of the paper:

```
Text segments  d_0, d_1, ..., d_N
        │
        ▼
┌───────────────┐
│  Base Encoder  │  E_0 : T → ℝ^{D_0}  (e.g. MiniLM-L6-v2, 384-d)
│   (frozen)     │
└───────┬───────┘
        ▼
┌───────────────┐
│ Preprocessing  │  P(e): centering, top-k PC removal, ℓ₂ normalization
└───────┬───────┘
        ▼
┌───────────────┐
│   Geometry     │  A_θ : ℝ^{D_0} → ℝ^D  (MLP, D=16 default)
│   Adapter      │
└───────┬───────┘
        ▼
   s_t ∈ M ⊂ ℝ^D   ← Latent narrative manifold
        │
        ├──→  Lorentzian Metric  g_θ(s) = A^T η A
        ├──→  Lagrangian         L_θ = λ_g·L_g + λ_sem·L_sem
        ├──→  Gibbs Kernel       K_θ(s'|s) ∝ exp(−L_θ)
        └──→  World Model        q_θ(s'|s) ≈ K_θ(s'|s)
```

### Two Instantiation Paths

| Path | Description | Decoder? |
|------|-------------|----------|
| **(A) Encoder + Adapter** (default) | Fixed E₀ + learned adapter A<sub>θ</sub> | No |
| **(B) Geometry-aware VAE** | Trainable (E<sub>ϕ</sub>, D<sub>ϕ</sub>) with Lorentzian regularization | Yes — enables counterfactual decoding |

---

## Project Structure

```
├── configs/
│   ├── d0_synthetic.yaml        # Time-reversal experiment
│   ├── d1_branching.yaml        # Branching experiment
│   └── d2_wikitext.yaml         # Real-text (WikiText-103) experiment
├── src/
│   ├── config.py                # All hyperparameters (single dataclass + YAML override)
│   ├── __init__.py
│   ├── data/
│   │   ├── synthetic.py         # D0 (drift-diffusion) & D1 (branching) generators
│   │   ├── wikitext.py          # D2: WikiText-103 loader + encoding + LM scoring
│   │   └── preprocessing.py     # Centering, PC removal, ℓ₂ normalization
│   ├── models/
│   │   ├── adapter.py           # Geometry adapter A_θ : ℝ^{D₀} → ℝ^D
│   │   ├── metric.py            # Lorentzian metric g_θ (+ Riemannian/Euclidean baselines)
│   │   ├── lagrangian.py        # L_θ = λ_g·L_g + λ_sem·L_sem  (with energy stabilization)
│   │   ├── world_model.py       # Conditional Gaussian q_θ(s'|s)
│   │   └── segmentation.py      # Algorithm 1: geometric change-point detection
│   ├── training/
│   │   ├── losses.py            # L_time, L_smooth, L_match, L_ML + auto-calibration
│   │   ├── candidates.py        # C1 (in-batch) / C2 (kNN retrieval) / C3 (LM-generated)
│   │   └── trainer.py           # Staged training loop (Stage 2 → Stage 3)
│   └── evaluation/
│       ├── metrics.py           # M1–M6 quantitative evaluation protocol
│       └── visualization.py     # Paper-ready plots (histograms, cones, action curves)
├── scripts/
│   ├── train.py                 # Main entry point
│   ├── run_baselines.py         # Run all geometry baselines on D0 + D1
│   └── encode_corpus.py         # Pre-compute embeddings + LM scores for D2
└── requirements.txt
```

---

## Experimental Protocol

### Three Datasets

| Dataset | Type | Purpose | Encoder needed? |
|---------|------|---------|:---:|
| **D0** | Synthetic drift-diffusion | Sanity check — time-reversal detection (H1, H2) | No |
| **D1** | Synthetic branching | Test H3 — space-like separation of branches | No |
| **D2** | WikiText-103 paragraphs | Ecological validity on real text | Yes |

### Three Hypotheses

| ID | Hypothesis | Metric | Pass criterion |
|----|-----------|--------|----------------|
| **H1** | Directedness | M1 (time-likeness rate) | > 0.9 |
| **H2** | Arrow of narrative | M2 (forward vs. reversed action gap) | > 0 |
| **H3** | Branching = space-like | M3 (space-like separation rate) | > 0.7 |

### Three Geometry Baselines

| Geometry | Signature | Cone structure? |
|----------|-----------|:---:|
| **Lorentzian** (ours) | (−, +, …, +) | Yes |
| **Riemannian** | (+, +, …, +) | No |
| **Euclidean** (fixed I) | Identity | No |

### Evaluation Metrics (M1–M6)

| Metric | Name | What it measures |
|--------|------|------------------|
| **M1** | Time-likeness rate | Pr[Δσ² < 0] on real transitions |
| **M2** | Time-reversal gap | Action(forward) vs Action(reversed) |
| **M3** | Branching separation | Pr[inter-branch interval > 0] (space-like) |
| **M4** | Cone alignment | Jaccard overlap: metric cone ∩ probabilistic cone |
| **M5** | Predictive quality | NLL of q<sub>θ</sub> on held-out transitions |
| **M6** | Branching signal | Correlation of log Z<sub>θ</sub>(s) with LM entropy |

---

## Execution Guide

### Phase 1 — D0 Validation (~5 min/geometry, ~15 min total)

```bash
python -m scripts.train --dataset d0_synthetic --geometry lorentzian
python -m scripts.train --dataset d0_synthetic --geometry riemannian
python -m scripts.train --dataset d0_synthetic --geometry euclidean
```

**Expected:** Lorentzian M1 > 0.95, M2 gap ≫ 0. If M1 < 0.8 → debug before proceeding.

### Phase 2 — D1 Branching (~5 min/geometry, ~15 min total)

```bash
python -m scripts.train --dataset d1_branching --geometry lorentzian
python -m scripts.train --dataset d1_branching --geometry riemannian
python -m scripts.train --dataset d1_branching --geometry euclidean
```

**Expected:** Lorentzian M3 > 0.7. Riemannian/Euclidean M3 ≈ random.

### Phase 3 — D2 WikiText-103 (~2–4 hours on A100)

```bash
# Step 1: Encode corpus (one-time, ~30 min)
python -m scripts.encode_corpus --config configs/d2_wikitext.yaml --max_articles 5000

# Step 2: Train
python -m scripts.train --config configs/d2_wikitext.yaml --geometry lorentzian
python -m scripts.train --config configs/d2_wikitext.yaml --geometry riemannian
```

### Phase 4 — Ablations

| Variable | Values |
|----------|--------|
| Latent dim D | {8, 16, 32} |
| Candidate set size \|C\| | {16, 32, 64} |
| Segmentation | none / Algorithm 1 / differentiable |

---

## Hardware Requirements

| | D0 / D1 (synthetic) | D2 (WikiText-103) |
|---|---|---|
| **GPU** | Any (CPU OK) | A100 recommended |
| **VRAM** | < 1 GB | ~10 GB (encoder + LM) |
| **Disk** | < 100 MB | ~2 GB (cached embeddings) |
| **Time** | ~15 min per geometry | ~2 hours per geometry |

---

## Design Decisions

**Why Lorentzian, not Riemannian?** A Riemannian metric treats all directions as geometrically equivalent — it cannot distinguish "narrative goes forward" from "narrative branches sideways." The single negative eigenvalue in the Lorentzian signature `(−, +, …, +)` is the minimal modification that creates causal cones, separating time-like from space-like directions (see Remark 1 in the paper).

**Why a conditional Gaussian for q<sub>θ</sub>?** Closed-form log-probabilities and easy sampling. The goal of v1 is validating the *geometric framework*, not the expressivity of the world model. Normalizing flows or neural samplers are a natural upgrade for v2.

**Why MiniLM-L6-v2 for E₀?** 384-dimensional embeddings that are already semantically meaningful. The geometry adapter then compresses to D=16. Proposition 2 (encoder invariance) guarantees that different encoders lead to equivalent physics up to diffeomorphism.

**Why GPT-2-medium for L<sub>sem</sub>?** We need log-probabilities, not generation quality. GPT-2-medium is fast, fits on any GPU, and provides a reasonable semantic surprise signal.

**Energy stabilization.** Two mechanisms prevent training collapse: (1) interval clamping `Δσ²_clamped = max(Δσ², −10)` bounds the geometric reward; (2) temperature scaling in the Gibbs kernel prevents softmax saturation. See Section 2.3 of the paper.

**Auto-calibration of λ weights.** At the transition to Stage 3, loss-magnitude matching automatically sets λ values so that all loss terms contribute equally, eliminating manual tuning.

---

## Downstream Applications

The extracted world model W<sub>θ</sub> supports three concrete applications (Section 6 of the paper):

### 1. Counterfactual Generation
Displace a state s<sub>t</sub> along a **space-like** eigenvector of g<sub>θ</sub>(s<sub>t</sub>) and roll forward with q<sub>θ</sub>. The Lorentzian structure provides a principled criterion for "sideways" perturbations — impossible with a Riemannian metric.

### 2. Narrative Anomaly Detection
A transition whose Lagrangian spikes relative to the trajectory median signals an anomaly — a passage that doesn't follow naturally from context. Useful for quality control, plagiarism detection, and topic drift.

### 3. Uncertainty Quantification
The local partition function log Ẑ<sub>θ</sub>(s) measures how many plausible futures exist at state s. High values = narrative branching point; low values = continuation is nearly determined. Can be correlated with LM continuation entropy.

---

## Citation

```bibtex
@article{grieco2026lorentzian,
  title   = {Every {LLM} Has an Implicit World Model Inside It:
             A Lorentzian Geometric Framework for Making It Explicit},
  author  = {Grieco, Piermatteo},
  year    = {2026}
}
```

---

## References

- Clough & Evans (2017). *Embedding graphs in Lorentzian spacetime.* PLOS ONE.
- Ha & Schmidhuber (2018). *Recurrent world models facilitate policy evolution.* NeurIPS.
- Hafner et al. (2020). *Dream to control: Learning behaviors by latent imagination.* ICLR.
- Hao et al. (2023). *Reasoning with language model is planning with world model.* EMNLP.
- Law & Lucas (2023). *Spacetime representation learning.* ICLR.
- LeCun et al. (2006). *A tutorial on energy-based learning.* MIT Press.
- Noroozizadeh et al. (2025). *Deep sequence models tend to memorize geometrically.* arXiv:2510.26745.
- Ziebart et al. (2008). *Maximum entropy inverse reinforcement learning.* AAAI.
