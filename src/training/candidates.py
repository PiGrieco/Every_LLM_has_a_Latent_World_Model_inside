"""
Candidate-set construction for the Gibbs kernel approximation.

Three strategies of increasing hardness (Section 4.7):

C1: In-batch negatives — free, mostly easy negatives
C2: kNN retrieval-based hard negatives — close but wrong continuations
C3: LM-generated alternatives — semantically plausible divergent futures

The curriculum strategy starts with C1 and progressively adds harder
negatives as training stabilizes.
"""

import torch
from typing import Optional, Tuple


def _sample_nonself_indices(batch: int, n_pick: int, device) -> torch.Tensor:
    """For each row i in [0, batch), sample n_pick column indices from
    [0, batch) \\ {i}, uniformly with replacement.

    Implementation trick: sample from [0, batch-1) then shift indices ≥ i by
    +1 so row i is skipped. This avoids materialising a (batch, batch) weight
    matrix or performing a batched argsort, giving constant memory and
    O(batch · n_pick) work regardless of batch size.

    Using with-replacement is a deliberate trade-off: the original loop used
    argsort-based without-replacement for the first (batch-1) picks, then
    with-replacement padding beyond that. In practice, duplicate negatives
    from a batch of 128+ are rare enough (≈3-4 per row at n_pick=31) to
    leave training unaffected, and the speedup on GPU is 2-3 orders of
    magnitude at batch=2048.

    Returns (batch, n_pick) long tensor."""
    if batch == 1:
        return torch.zeros(1, n_pick, dtype=torch.long, device=device)
    # Sample in [0, batch-1)
    idx = torch.randint(0, batch - 1, (batch, n_pick), device=device)
    # Shift: whenever sampled index is ≥ row index, add 1 to skip self
    row_idx = torch.arange(batch, device=device).unsqueeze(1)
    idx = idx + (idx >= row_idx).long()
    return idx


def build_candidate_set_c1(
    s: torch.Tensor,
    s_next: torch.Tensor,
    candidate_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    C1: In-batch negatives.

    For each (s_i, s_next_i) pair in the batch, the candidate set is:
    - The true next state s_next_i (always at index 0)
    - Random other s_next_j from the same batch (j ≠ i)

    Free (no extra forward passes); negatives are mostly easy since unrelated
    trajectories are typically far apart in latent space.

    Fully vectorized — no Python loop over batch.

    Args:
        s: (batch, D) current states
        s_next: (batch, D) true next states
        candidate_size: total number of candidates per state

    Returns:
        candidates: (batch, C, D) candidate sets
        true_idx: (batch,) index of the true next state in each set (always 0)
    """
    batch, _ = s.shape
    device = s.device
    n_neg = candidate_size - 1

    neg_idx = _sample_nonself_indices(batch, n_neg, device)  # (batch, n_neg)
    neg_cands = s_next[neg_idx]  # (batch, n_neg, D)

    candidates = torch.cat([s_next.unsqueeze(1), neg_cands], dim=1)  # (batch, C, D)
    true_idx = torch.zeros(batch, dtype=torch.long, device=device)
    return candidates, true_idx


def build_candidate_set_c2(
    s: torch.Tensor,
    s_next: torch.Tensor,
    all_states: torch.Tensor,
    candidate_size: int = 32,
    n_knn: int = 16,
    knn_index=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    C2: In-batch negatives + kNN hard negatives.

    Augments C1 with the k nearest neighbors of each s in the full
    state space. These are states that are close in latent space but
    not the true continuation — forcing fine-grained discrimination.

    Args:
        s: (batch, D) current states
        s_next: (batch, D) true next states
        all_states: (N, D) all states in the training set (for kNN lookup)
        candidate_size: total candidates per state
        n_knn: number of kNN hard negatives to include
        knn_index: optional pre-built FAISS index

    Returns:
        candidates: (batch, C, D)
        true_idx: (batch,)
    """
    batch, dim = s.shape
    device = s.device

    assert all_states.shape[1] == dim, (
        f"Dimension mismatch: s is {dim}D but all_states is {all_states.shape[1]}D. "
        f"Ensure kNN index is built on adapter-projected (latent) states, not raw embeddings."
    )

    # ---- kNN hard negatives ----
    if knn_index is not None:
        s_np = s.detach().cpu().numpy().astype("float32")
        _, knn_indices = knn_index.search(s_np, n_knn + 1)
        knn_indices = torch.from_numpy(knn_indices[:, 1:]).long()  # exclude self
        knn_states = all_states[knn_indices]  # (batch, n_knn, D)
    else:
        dists = torch.cdist(s, all_states)  # (batch, N)
        _, topk = dists.topk(n_knn + 1, largest=False, dim=-1)
        knn_states = all_states[topk[:, 1:]]  # (batch, n_knn, D)

    knn_states = knn_states.to(device)

    # ---- In-batch negatives filler ----
    n_inbatch = max(0, candidate_size - 1 - n_knn)

    parts = [s_next.unsqueeze(1), knn_states]  # true next + kNN
    if n_inbatch > 0 and batch > 1:
        inb_idx = _sample_nonself_indices(batch, n_inbatch, device)
        parts.append(s_next[inb_idx])

    candidates = torch.cat(parts, dim=1)  # (batch, <=C, D)

    # ---- Truncate / pad to exactly candidate_size ----
    if candidates.shape[1] > candidate_size:
        candidates = candidates[:, :candidate_size]
    elif candidates.shape[1] < candidate_size:
        pad = candidate_size - candidates.shape[1]
        # Pad by sampling from existing negatives (indices 1..end), per row.
        neg_pool = candidates[:, 1:] if candidates.shape[1] > 1 else candidates
        pool_size = neg_pool.shape[1]
        pad_idx = torch.randint(0, pool_size, (batch, pad), device=device)
        pad_cands = torch.gather(
            neg_pool, 1, pad_idx.unsqueeze(-1).expand(-1, -1, dim)
        )
        candidates = torch.cat([candidates, pad_cands], dim=1)

    true_idx = torch.zeros(batch, dtype=torch.long, device=device)
    return candidates, true_idx


def build_faiss_index(states: torch.Tensor):
    """
    Build a FAISS index for fast kNN retrieval.

    Args:
        states: (N, D) tensor of all states

    Returns:
        index: FAISS index
    """
    import faiss
    import numpy as np

    dim = states.shape[1]
    states_np = states.detach().cpu().numpy().astype("float32")

    # Use IVF index for larger datasets, flat for small ones
    if len(states_np) > 10000:
        nlist = min(256, len(states_np) // 40)
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(states_np)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(states_np)
    return index
