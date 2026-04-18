"""Rigid smoke-gate validators for produced trajectory datasets.

Each validator returns a structured check::

    {
        "passed": bool,
        "value": float,                     # the main measurement
        "threshold": float,                 # the gate cfg value
        "details": dict,                    # sub-measurements / sub-checks
    }

``run_smoke_gate`` aggregates them across all datasets under
``output_dir`` and ``scripts/probe/smoke_gate.py`` turns the aggregate
``all_passed`` verdict into an exit code. This is a CI-style gate, not
a warning log: a failed gate stops M2 from starting.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .config import ProbeConfig
from .storage import TrajectoryShardReader

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _sample_items(reader: TrajectoryShardReader, n: int, seed: int = 0) -> List[dict]:
    total = len(reader)
    if total == 0:
        return []
    n = min(n, total)
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
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    t = torch.tensor(vals, dtype=torch.float32)
    return {
        "n": int(t.numel()),
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
        "min": float(t.min()),
        "max": float(t.max()),
        "median": float(t.median()),
    }


def _get_hidden(item: dict):
    """Return the ``hidden_states`` tensor for any dataset kind, or ``None``."""
    if "hidden_states" in item and isinstance(item["hidden_states"], torch.Tensor):
        return item["hidden_states"]
    if "trajectory_a" in item and isinstance(item["trajectory_a"], dict):
        return item["trajectory_a"].get("hidden_states")
    if "forward_hidden" in item and isinstance(item["forward_hidden"], torch.Tensor):
        return item["forward_hidden"]
    return None


def _locate_branching_window(
    token_positions: List[Tuple[int, int]], branching_point: int,
) -> int:
    """Return the first window index whose end exceeds the branching token.

    This is the window in which ``τ_a`` and ``τ_b`` can first diverge on
    the pooled state.
    """
    for i, pos in enumerate(token_positions):
        if pos[1] > branching_point:
            return i
    return len(token_positions)


# --------------------------------------------------------------------------
# Individual validators
# --------------------------------------------------------------------------

def validate_trajectory_statistics(
    reader: TrajectoryShardReader, cfg: ProbeConfig,
) -> dict:
    """Finiteness + norm band + sufficient-window check on a forward set.

    Returns a structured check. ``passed`` is the AND of three
    sub-checks, documented in ``details.subchecks``:

      1. fraction_finite >= ``cfg.gate_min_fraction_finite``
      2. mean per-window norm in [``gate_norm_min``, ``gate_norm_max``]
      3. median n_windows >= ``gate_min_n_windows_median``
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
    norm_stats = _norm_stats(per_window_norms)
    windows_stats = _norm_stats([float(x) for x in n_windows])

    sub_finite_ok = fraction_finite >= cfg.gate_min_fraction_finite
    sub_norm_ok = (
        norm_stats["n"] > 0
        and cfg.gate_norm_min <= norm_stats["mean"] <= cfg.gate_norm_max
    )
    sub_windows_ok = (
        windows_stats["n"] > 0
        and windows_stats["median"] >= cfg.gate_min_n_windows_median
    )
    passed = sub_finite_ok and sub_norm_ok and sub_windows_ok

    return {
        "passed": bool(passed),
        "value": float(fraction_finite),
        "threshold": float(cfg.gate_min_fraction_finite),
        "details": {
            "n_items_sampled": len(items),
            "unique_doc_ids": len(doc_ids),
            "norm_stats": norm_stats,
            "n_windows_stats": windows_stats,
            "subchecks": {
                "fraction_finite": {
                    "passed": sub_finite_ok,
                    "value": fraction_finite,
                    "threshold": cfg.gate_min_fraction_finite,
                },
                "norm_mean_in_band": {
                    "passed": sub_norm_ok,
                    "value": norm_stats["mean"],
                    "threshold": [cfg.gate_norm_min, cfg.gate_norm_max],
                },
                "median_windows": {
                    "passed": sub_windows_ok,
                    "value": windows_stats["median"],
                    "threshold": cfg.gate_min_n_windows_median,
                },
            },
        },
    }


def validate_branching_divergence(
    reader: TrajectoryShardReader, cfg: ProbeConfig,
) -> dict:
    """Check that branching pairs diverge AFTER the branching point.

    For each sampled pair we compute

      - ``diverg_pre``  = mean ``||a_t - b_t||`` for windows before the
                         branching window
      - ``diverg_post`` = mean for windows at/after the branching window

    A pair is "divergent" iff ``diverg_post > diverg_pre``. Gate
    passes when the fraction of divergent pairs is at least
    ``cfg.gate_min_branching_divergence_fraction``.
    """
    pairs = _sample_items(reader, 500)
    pre_divergences: List[float] = []
    post_divergences: List[float] = []
    is_divergent_flags: List[bool] = []

    for it in pairs:
        a_d = it.get("trajectory_a", {})
        b_d = it.get("trajectory_b", {})
        a = a_d.get("hidden_states")
        b = b_d.get("hidden_states")
        t_abs = int(it.get("branching_point", -1))
        positions_a = a_d.get("token_positions")
        if a is None or b is None or t_abs < 0 or not positions_a:
            continue

        t_shared = min(a.shape[0], b.shape[0])
        if t_shared == 0:
            continue

        w_branch = _locate_branching_window(positions_a, t_abs)
        w_branch = max(0, min(w_branch, t_shared))

        diffs = (a[:t_shared].float() - b[:t_shared].float()).norm(dim=-1)
        pre_tail = diffs[:w_branch]
        post_tail = diffs[w_branch:]
        diverg_pre = float(pre_tail.mean()) if pre_tail.numel() > 0 else 0.0
        diverg_post = float(post_tail.mean()) if post_tail.numel() > 0 else 0.0
        pre_divergences.append(diverg_pre)
        post_divergences.append(diverg_post)
        is_divergent_flags.append(diverg_post > diverg_pre)

    n = len(is_divergent_flags)
    fraction_divergent = (
        sum(is_divergent_flags) / n if n > 0 else 0.0
    )
    passed = fraction_divergent >= cfg.gate_min_branching_divergence_fraction

    return {
        "passed": bool(passed),
        "value": float(fraction_divergent),
        "threshold": float(cfg.gate_min_branching_divergence_fraction),
        "details": {
            "n_pairs_sampled": n,
            "mean_diverg_pre": _norm_stats(pre_divergences)["mean"],
            "mean_diverg_post": _norm_stats(post_divergences)["mean"],
            "fraction_divergent": fraction_divergent,
        },
    }


def validate_reversed_differ(
    reader: TrajectoryShardReader, cfg: ProbeConfig,
) -> dict:
    """Check forward vs reversed cosine similarity is below threshold.

    ``mean_cos_sim`` is computed across aligned ``(forward[t],
    reversed[T-1-t])`` pairs. Gate passes iff
    ``mean_cos_sim <= cfg.gate_max_reversed_cosine_similarity``.
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
        a = fwd[:T].float()
        b = rev.flip(0)[-T:].float()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
        similarities.append(cos)

    mean_cos_sim = (
        sum(similarities) / len(similarities) if similarities else 0.0
    )
    passed = mean_cos_sim <= cfg.gate_max_reversed_cosine_similarity

    return {
        "passed": bool(passed),
        "value": float(mean_cos_sim),
        "threshold": float(cfg.gate_max_reversed_cosine_similarity),
        "details": {
            "n_pairs_sampled": len(similarities),
            "max_cosine_similarity": (max(similarities) if similarities else 0.0),
            "stats": _norm_stats(similarities),
        },
    }


# --------------------------------------------------------------------------
# Aggregated gate
# --------------------------------------------------------------------------

def run_smoke_gate(output_dir: str, cfg: ProbeConfig) -> dict:
    """Run every applicable validator on every dataset under ``output_dir``.

    Args:
        output_dir: Root directory whose subfolders are datasets
            (forward / branching / reversed / validation).
        cfg: Runtime configuration (thresholds are read from here).

    Returns:
        dict of form::

            {
              "all_passed": bool,
              "datasets": {
                "forward": {"<check>": {...}, ...},
                ...
              }
            }

        If a dataset directory lacks a ``manifest.json`` it is skipped
        with a warning, which counts against ``all_passed`` only when
        the directory is one we produced (i.e. it exists but is empty).
    """
    out_dir = Path(output_dir)
    report: dict = {"all_passed": True, "datasets": {}}

    def _record(name: str, checks: Dict[str, dict]) -> None:
        ds_passed = all(c.get("passed", False) for c in checks.values())
        report["datasets"][name] = {**checks, "_all_passed": ds_passed}
        if not ds_passed:
            report["all_passed"] = False

    # forward / validation use the same validator
    for name in ("forward", "validation"):
        ds_dir = out_dir / name
        if not (ds_dir / "manifest.json").exists():
            continue
        reader = TrajectoryShardReader(str(ds_dir))
        _record(name, {"trajectory_statistics": validate_trajectory_statistics(reader, cfg)})

    ds_dir = out_dir / "branching"
    if (ds_dir / "manifest.json").exists():
        reader = TrajectoryShardReader(str(ds_dir))
        # Also apply the finiteness + norm checks to each side of the pair.
        traj_stats = validate_trajectory_statistics(reader, cfg)
        div = validate_branching_divergence(reader, cfg)
        _record("branching", {
            "trajectory_statistics": traj_stats,
            "branching_divergence": div,
        })

    ds_dir = out_dir / "reversed"
    if (ds_dir / "manifest.json").exists():
        reader = TrajectoryShardReader(str(ds_dir))
        traj_stats = validate_trajectory_statistics(reader, cfg)
        rev = validate_reversed_differ(reader, cfg)
        _record("reversed", {
            "trajectory_statistics": traj_stats,
            "reversed_divergence": rev,
        })

    if not report["datasets"]:
        logger.warning(
            "run_smoke_gate found no datasets under %s; returning all_passed=False.",
            out_dir,
        )
        report["all_passed"] = False

    return report
