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


def build_candidate_set_c1(
    s: torch.Tensor,
    s_next: torch.Tensor,
    candidate_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    C1: In-batch negatives.

    For each (s_i, s_next_i) pair in the batch, the candidate set is:
    - The true next state s_next_i
    - Random other s_next_j from the same batch (j ≠ i)

    This is computationally free (no extra forward passes) but provides
    mostly easy negatives, since unrelated trajectories are typically
    far apart in latent space.

    Args:
        s: (batch, D) current states
        s_next: (batch, D) true next states
        candidate_size: total number of candidates per state

    Returns:
        candidates: (batch, C, D) candidate sets
        true_idx: (batch,) index of the true next state in each set (always 0)
    """
    batch, dim = s.shape
    n_neg = min(candidate_size - 1, batch - 1)

    # For each item, pick n_neg random negatives from other items in batch
    candidates_list = []
    for i in range(batch):
        # True next state is always first
        cands = [s_next[i].unsqueeze(0)]

        # Sample negatives: random indices excluding i
        neg_indices = torch.randperm(batch, device=s.device)
        neg_indices = neg_indices[neg_indices != i][:n_neg]
        cands.append(s_next[neg_indices])

        # Pad if we don't have enough negatives
        current = 1 + len(neg_indices)
        if current < candidate_size:
            # Repeat random negatives to fill
            extra = candidate_size - current
            extra_idx = torch.randint(0, len(neg_indices), (extra,), device=s.device)
            cands.append(s_next[neg_indices[extra_idx]])

        candidates_list.append(torch.cat(cands, dim=0)[:candidate_size])

    candidates = torch.stack(candidates_list)  # (batch, C, D)
    true_idx = torch.zeros(batch, dtype=torch.long, device=s.device)

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

    # Get kNN hard negatives
    if knn_index is not None:
        # Use FAISS index
        s_np = s.detach().cpu().numpy().astype("float32")
        _, knn_indices = knn_index.search(s_np, n_knn + 1)
        knn_indices = torch.from_numpy(knn_indices[:, 1:])  # exclude self
        knn_states = all_states[knn_indices.long()]  # (batch, n_knn, D)
    else:
        # Brute-force kNN (for small datasets)
        dists = torch.cdist(s, all_states)  # (batch, N)
        _, topk = dists.topk(n_knn + 1, largest=False, dim=-1)
        knn_states = all_states[topk[:, 1:]]  # (batch, n_knn, D)

    knn_states = knn_states.to(device)

    # Combine: true_next + kNN + in-batch negatives
    n_inbatch = candidate_size - 1 - n_knn
    n_inbatch = max(0, n_inbatch)

    candidates_list = []
    for i in range(batch):
        cands = [s_next[i].unsqueeze(0)]  # True next
        cands.append(knn_states[i])       # kNN negatives

        if n_inbatch > 0:
            # Fill remaining with in-batch negatives
            neg_idx = torch.randperm(batch, device=device)
            neg_idx = neg_idx[neg_idx != i][:n_inbatch]
            if len(neg_idx) > 0:
                cands.append(s_next[neg_idx])

        combined = torch.cat(cands, dim=0)[:candidate_size]

        # Pad if needed
        if combined.shape[0] < candidate_size:
            pad = candidate_size - combined.shape[0]
            pad_idx = torch.randint(1, combined.shape[0], (pad,), device=device)
            combined = torch.cat([combined, combined[pad_idx]], dim=0)

        candidates_list.append(combined)

    candidates = torch.stack(candidates_list)
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
