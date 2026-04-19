"""Hidden-state memory + retrieval-based readout evaluation.

The M2 milestone's central claim — "the projection preserves the
local structure of the LM hidden state" — is tested by showing that
the top-k neighbours of a query, computed in the full 4096-D space,
are substantially recovered when the query is first passed through
Ψ(Φ(·)). This module provides the memory index, the disjoint
doc-id split, and the evaluation function.

Revision note: memory and query doc-ids are *strictly disjoint* by
construction (see :func:`build_memory_and_queries`) so the
retrieval metric cannot be gamed by "retrieve yourself". The metric is
a *stability* ratio — overlap between the projected top-k and the
baseline top-k, under the same fixed memory.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ==========================================================================
# Memory
# ==========================================================================

class HiddenStateMemory:
    """Cosine-similarity nearest-neighbour index over hidden states.

    Uses FAISS ``IndexFlatIP`` with L2-normalised vectors when N > 1000,
    else falls back to a brute-force torch matmul so unit tests don't
    depend on FAISS.
    """

    def __init__(
        self,
        hidden_states: torch.Tensor,
        doc_ids: List[str],
        positions: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        if hidden_states.ndim != 2:
            raise ValueError(
                f"hidden_states must be (N, d_model); got {tuple(hidden_states.shape)}"
            )
        if len(doc_ids) != hidden_states.shape[0]:
            raise ValueError(
                f"len(doc_ids)={len(doc_ids)} != N={hidden_states.shape[0]}"
            )
        self._h = hidden_states.detach().float().cpu().contiguous()
        self._h_norm = torch.nn.functional.normalize(self._h, dim=-1)
        self._doc_ids = list(doc_ids)
        self._positions = list(positions) if positions is not None else [(-1, -1)] * len(doc_ids)
        self._use_faiss = self._h.shape[0] > 1000
        self._faiss_index = None
        if self._use_faiss:
            try:
                import faiss  # type: ignore
                dim = self._h.shape[1]
                self._faiss_index = faiss.IndexFlatIP(dim)
                self._faiss_index.add(self._h_norm.numpy().astype("float32"))
            except Exception as exc:
                logger.warning(
                    "FAISS unavailable (%s); falling back to brute-force torch", exc,
                )
                self._use_faiss = False
                self._faiss_index = None

    def __len__(self) -> int:
        return int(self._h.shape[0])

    def doc_id_at(self, idx: int) -> str:
        return self._doc_ids[idx]

    @property
    def doc_ids(self) -> List[str]:
        return list(self._doc_ids)

    def search(
        self, query: torch.Tensor, top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(indices, scores)`` of shape ``(B, top_k)``.

        Scores are cosine similarities in ``[-1, 1]``.
        """
        if query.ndim == 1:
            query = query.unsqueeze(0)
        q = torch.nn.functional.normalize(query.detach().float().cpu(), dim=-1)
        top_k = min(int(top_k), len(self))

        if self._use_faiss and self._faiss_index is not None:
            scores_np, idx_np = self._faiss_index.search(
                q.numpy().astype("float32"), top_k,
            )
            return torch.from_numpy(idx_np).long(), torch.from_numpy(scores_np).float()

        # Brute force: (B, D) @ (D, N) = (B, N)
        sims = q @ self._h_norm.T
        scores, idx = sims.topk(top_k, dim=-1)
        return idx, scores


# ==========================================================================
# Disjoint-doc-id memory / query construction
# ==========================================================================

def _sample_states_from_doc_ids(
    reader,
    allowed_doc_ids: set,
    n_states: int,
    seed: int,
) -> List[dict]:
    """Return up to ``n_states`` items ``{h, doc_id, positions}`` whose
    ``doc_id`` is in ``allowed_doc_ids``.

    Reservoir-samples across the shard stream so the order on disk does
    not bias the selection.
    """
    if n_states <= 0:
        return []
    rng = random.Random(seed)
    collected: List[dict] = []
    total = 0
    for item in reader.iter_items():
        doc_id = item.get("doc_id")
        if doc_id not in allowed_doc_ids:
            continue
        # hidden_states layout may be flat or nested (branching).
        hs = item.get("hidden_states")
        positions = item.get("token_positions") or []
        if hs is None and isinstance(item.get("trajectory_a"), dict):
            hs = item["trajectory_a"].get("hidden_states")
            positions = item["trajectory_a"].get("token_positions") or []
        if hs is None:
            continue
        for row_idx, row in enumerate(hs):
            pos = positions[row_idx] if row_idx < len(positions) else (-1, -1)
            entry = {
                "h": row.float().cpu(),
                "doc_id": doc_id,
                "position": tuple(pos),
            }
            if total < n_states:
                collected.append(entry)
            else:
                j = rng.randint(0, total)
                if j < n_states:
                    collected[j] = entry
            total += 1
    return collected


def build_memory_and_queries(
    reader,
    memory_size: int,
    n_queries: int,
    memory_fraction: float,
    seed: int,
) -> Tuple[HiddenStateMemory, torch.Tensor, List[str]]:
    """Build a memory and a query set from *disjoint* doc-id subsets.

    A deterministic permutation of all ``doc_id``s is split by
    ``memory_fraction``: the first slice feeds the memory, the second
    provides queries. An ``assert`` after construction re-checks that
    no doc-id appears in both.

    Args:
        reader: M1 ``TrajectoryShardReader``.
        memory_size: Number of hidden states to pack into the memory.
        n_queries: Number of query states to draw.
        memory_fraction: Fraction of doc-ids assigned to the memory
            (``0.6`` ⇒ 60% memory, 40% query).
        seed: Deterministic seed for permutation + row-level sampling.

    Returns:
        ``(memory, queries_h, queries_doc_ids)`` where:
          - ``memory`` is a :class:`HiddenStateMemory`;
          - ``queries_h`` is ``(n_queries, d_model)`` on CPU float32;
          - ``queries_doc_ids`` is a list of length ``n_queries``.

    Raises:
        ValueError: If the memory and query doc-id partitions overlap.
    """
    if not 0.0 < memory_fraction < 1.0:
        raise ValueError(
            f"memory_fraction must be in (0, 1); got {memory_fraction}"
        )
    all_doc_ids = sorted(reader.get_doc_ids())
    if len(all_doc_ids) < 2:
        raise ValueError(
            f"Need at least 2 doc_ids in the dataset; got {len(all_doc_ids)}"
        )

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(all_doc_ids), generator=g).tolist()
    n_memory_docs = max(1, int(len(all_doc_ids) * memory_fraction))
    n_memory_docs = min(n_memory_docs, len(all_doc_ids) - 1)

    memory_doc_ids = {all_doc_ids[i] for i in perm[:n_memory_docs]}
    query_doc_ids = {all_doc_ids[i] for i in perm[n_memory_docs:]}

    overlap = memory_doc_ids & query_doc_ids
    if overlap:
        raise ValueError(
            f"Memory/query doc_id overlap detected: {len(overlap)} docs. "
            "This would cause retrieval leakage."
        )

    memory_items = _sample_states_from_doc_ids(
        reader, memory_doc_ids, memory_size, seed,
    )
    if not memory_items:
        raise RuntimeError("No memory items sampled — dataset may be empty.")
    memory = HiddenStateMemory(
        hidden_states=torch.stack([it["h"] for it in memory_items]),
        doc_ids=[it["doc_id"] for it in memory_items],
        positions=[it["position"] for it in memory_items],
    )

    query_items = _sample_states_from_doc_ids(
        reader, query_doc_ids, n_queries, seed + 1,
    )
    if not query_items:
        raise RuntimeError("No query items sampled — dataset may be empty.")
    queries_h = torch.stack([it["h"] for it in query_items])
    queries_doc_ids = [it["doc_id"] for it in query_items]

    # Post-construction assertion: memory doc-ids and query doc-ids
    # are strictly disjoint even after sampling.
    memory_doc_id_set = set(memory.doc_ids)
    query_doc_id_set = set(queries_doc_ids)
    leak = memory_doc_id_set & query_doc_id_set
    assert not leak, (
        f"Post-construction leakage check failed: {len(leak)} shared doc-ids"
    )

    return memory, queries_h, queries_doc_ids


# ==========================================================================
# Retrieval evaluation
# ==========================================================================

def retrieval_evaluation(
    autoencoder,
    memory: HiddenStateMemory,
    queries_h: torch.Tensor,
    queries_doc_ids: List[str],
    topk_list: List[int],
) -> dict:
    """Measure how much of the baseline top-k the projected readout recovers.

    For every query ``h_q``:

      - *baseline* ranks memory items by ``cos(h_q, ·)``.
      - *projected* ranks memory items by ``cos(Ψ(Φ(h_q)), ·)``.

    We then compute, for each ``k`` in ``topk_list``, the mean fraction
    of projected top-k indices that also appear in the baseline top-k.
    A value of 1.0 means the projection perfectly preserves the local
    neighbourhood; the M2 hard gate requires this ratio to be ≥ 0.80
    at ``k = 5``.

    The returned dict also carries raw per-k baseline and projected
    top-k overlap sizes for reporting.
    """
    device = next(autoencoder.parameters()).device
    with torch.no_grad():
        s = autoencoder.encode(queries_h.to(device))
        h_hat = autoencoder.decode(s).detach().cpu()

    results = {
        "n_queries": int(queries_h.shape[0]),
        "memory_size": len(memory),
        "projected_fraction_of_baseline_topk": {},
        "baseline_topk_recalls": {},
        "projected_topk_recalls": {},
    }
    for k in topk_list:
        baseline_idx, _ = memory.search(queries_h, top_k=k)  # (B, k)
        projected_idx, _ = memory.search(h_hat, top_k=k)     # (B, k)

        B = baseline_idx.shape[0]
        overlap_fracs = []
        for b in range(B):
            base_set = set(baseline_idx[b].tolist())
            proj_set = set(projected_idx[b].tolist())
            k_eff = max(1, min(k, len(base_set), len(proj_set)))
            overlap_fracs.append(len(base_set & proj_set) / float(k_eff))
        results["projected_fraction_of_baseline_topk"][str(k)] = float(
            sum(overlap_fracs) / max(1, B)
        )
        # Report the raw mean overlap size (sanity metric).
        results["baseline_topk_recalls"][str(k)] = float(k)
        results["projected_topk_recalls"][str(k)] = float(
            sum(overlap_fracs) * k / max(1, len(overlap_fracs))
        )
    return results


def on_manifold_drift(
    autoencoder,
    h_sample: torch.Tensor,
) -> float:
    """Mean ``||Ψ(Φ(h)) - h|| / std(h)`` over a held-out sample.

    Expressed in units of the per-dimension standard deviation of the
    ambient hidden states, so the metric is scale-free. Zero means
    perfect reconstruction; small values indicate the decoder lands
    inside the ambient distribution.
    """
    device = next(autoencoder.parameters()).device
    h = h_sample.to(device).float()
    with torch.no_grad():
        _, h_hat = autoencoder(h)
    diff_norm = (h - h_hat).norm(dim=-1)
    # Scale by the mean per-dim std of the sample (averaged across dims).
    std_scalar = h.std(dim=0).mean().clamp_min(1e-8)
    return float((diff_norm / std_scalar).mean().item())
