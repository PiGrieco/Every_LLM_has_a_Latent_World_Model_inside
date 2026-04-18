"""Post-run sanity checks for produced trajectory datasets.

Each function loads a :class:`~.storage.TrajectoryShardReader`, samples
up to a bounded number of items, and returns a dict of summary
statistics. Violations of reasonable bounds are logged as warnings so a
large overnight run does not fail but still flags problems.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List

import torch

from .storage import TrajectoryShardReader

logger = logging.getLogger(__name__)


def _sample_items(reader: TrajectoryShardReader, n: int, seed: int = 0) -> List[dict]:
    total = len(reader)
    if total == 0:
        return []
    n = min(n, total)
    # Stream once, reservoir-sample n items.
    rng = random.Random(seed)
    sample: List[dict] = []
    for i, item in enumerate(reader.iter_items()):
        if i < n:
            sample.append(item)
        else:
            j = rng.randint(0, i)
            if j < n:
                sample[j] = item
    return sample


def _norm_stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    t = torch.tensor(vals, dtype=torch.float32)
    return {
        "n": int(t.numel()),
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
        "min": float(t.min()),
        "max": float(t.max()),
    }


def validate_trajectory_statistics(reader: TrajectoryShardReader) -> dict:
    """Compute finiteness, norm and window-count statistics on a forward dataset.

    Samples up to 1000 trajectories, looks at ``hidden_states``, and
    reports:
      - ``fraction_finite``: share of trajectories with all-finite hidden states
      - ``norm_stats``: descriptive stats of ``||h||_2`` per window
      - ``n_windows_stats``: descriptive stats of the pooled trajectory length
      - ``unique_doc_ids``: number of distinct articles represented

    Warns if any hidden states are non-finite or the per-window norm
    falls outside ``[1, 200]`` (too small → dead features; too large →
    numerical issues).
    """
    items = _sample_items(reader, 1000)
    n_finite = 0
    per_window_norms: List[float] = []
    n_windows: List[int] = []
    doc_ids = set()

    for it in items:
        hs = _get_hidden(it)
        if hs is None:
            continue
        finite = bool(torch.isfinite(hs).all())
        if finite:
            n_finite += 1
            per_window_norms.extend(hs.float().norm(dim=-1).tolist())
        n_windows.append(int(hs.shape[0]))
        if "doc_id" in it:
            doc_ids.add(it["doc_id"])

    fraction_finite = n_finite / max(1, len(items))
    stats = {
        "n_items_sampled": len(items),
        "fraction_finite": fraction_finite,
        "norm_stats": _norm_stats(per_window_norms),
        "n_windows_stats": _norm_stats([float(x) for x in n_windows]),
        "unique_doc_ids": len(doc_ids),
    }
    if fraction_finite < 1.0:
        logger.warning(
            "Trajectory stats: only %.3f of sampled trajectories are all-finite",
            fraction_finite,
        )
    mean_norm = stats["norm_stats"]["mean"]
    if mean_norm > 0 and not (1.0 <= mean_norm <= 200.0):
        logger.warning(
            "Trajectory stats: mean per-window norm %.2f outside [1, 200] "
            "(possible scaling issue)", mean_norm,
        )
    logger.info("Trajectory stats: %s", stats)
    return stats


def validate_branching_divergence(reader: TrajectoryShardReader) -> dict:
    """Check that branching pairs actually separate after the branching point.

    For each sampled pair, aligns the two per-window trajectories on
    overlapping positions, and reports:
      - ``mean_divergence``: average ``||a_t - b_t||`` over t
      - ``fraction_positive``: share of pairs with positive divergence
    """
    pairs = _sample_items(reader, 500)
    divergences: List[float] = []

    for it in pairs:
        a = it.get("trajectory_a", {}).get("hidden_states")
        b = it.get("trajectory_b", {}).get("hidden_states")
        if a is None or b is None:
            continue
        t_shared = min(a.shape[0], b.shape[0])
        if t_shared == 0:
            continue
        d = (a[:t_shared].float() - b[:t_shared].float()).norm(dim=-1).mean().item()
        divergences.append(d)

    if not divergences:
        stats = {"n_pairs_sampled": 0, "mean_divergence": 0.0, "fraction_positive": 0.0}
    else:
        t = torch.tensor(divergences, dtype=torch.float32)
        fraction_positive = float((t > 0).float().mean())
        stats = {
            "n_pairs_sampled": len(divergences),
            "mean_divergence": float(t.mean()),
            "fraction_positive": fraction_positive,
        }
        if fraction_positive < 0.7:
            logger.warning(
                "Branching divergence: only %.2f of pairs diverge (< 0.7 target)",
                fraction_positive,
            )
    logger.info("Branching stats: %s", stats)
    return stats


def validate_reversed_differ(reader: TrajectoryShardReader) -> dict:
    """Confirm forward and reversed trajectories differ nontrivially.

    Computes, per pair, the mean cosine similarity between
    ``forward[t]`` and ``reversed[T-1-t]``. If the overall mean exceeds
    0.95, warns loudly: at that point the representation is not
    direction-sensitive and the whole "arrow of time" story is in
    trouble.
    """
    pairs = _sample_items(reader, 500)
    similarities: List[float] = []

    for it in pairs:
        fwd = it.get("forward_hidden")
        rev = it.get("reversed_hidden")
        if fwd is None or rev is None:
            continue
        T = min(fwd.shape[0], rev.shape[0])
        if T == 0:
            continue
        # Align reversed[T-1-t] with forward[t]
        a = fwd[:T].float()
        b = rev.flip(0)[-T:].float()  # rev reversed so rev[T-1-t] == rev.flip(0)[t]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
        similarities.append(cos)

    if not similarities:
        stats = {"n_pairs_sampled": 0, "mean_cosine_similarity": 0.0}
    else:
        t = torch.tensor(similarities, dtype=torch.float32)
        stats = {
            "n_pairs_sampled": len(similarities),
            "mean_cosine_similarity": float(t.mean()),
            "max_cosine_similarity": float(t.max()),
        }
        if stats["mean_cosine_similarity"] > 0.95:
            logger.warning(
                "Reversed check: mean cosine similarity %.3f > 0.95 — "
                "forward and reversed trajectories are too close",
                stats["mean_cosine_similarity"],
            )
    logger.info("Reversed stats: %s", stats)
    return stats


def _get_hidden(item: dict):
    """Best-effort: return a (T, D) hidden_states tensor from any item shape."""
    if "hidden_states" in item and isinstance(item["hidden_states"], torch.Tensor):
        return item["hidden_states"]
    if "trajectory_a" in item and isinstance(item["trajectory_a"], dict):
        return item["trajectory_a"].get("hidden_states")
    if "forward_hidden" in item and isinstance(item["forward_hidden"], torch.Tensor):
        return item["forward_hidden"]
    return None
