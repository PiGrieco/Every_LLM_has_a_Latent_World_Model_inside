"""Linear identity probe: can a linear map ``Φ(h) → doc_id`` generalise?

**Not** a hard gate — in M2 this is a *diagnostic* with warning and
error thresholds. With ~1 000 classes and ~8 examples per class a
single linear layer has enormous variance; a low accuracy does not
invalidate M2 provided the retrieval and reconstruction hard gates
pass.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _collect_by_doc(
    reader, samples_per_class: int, max_classes: int, seed: int,
) -> Tuple[List[torch.Tensor], List[int], List[str], List[int]]:
    """Collect up to ``samples_per_class`` rows per doc_id, capped at
    ``max_classes`` doc_ids.

    Returns:
        features: list of ``(d_model,)`` tensors.
        labels: list of integer class indices.
        class_names: mapping from class index → doc_id.
        counts: per-class counts (for diagnostics).
    """
    rng = random.Random(seed)
    per_doc: Dict[str, List[torch.Tensor]] = {}
    for item in reader.iter_items():
        doc_id = item.get("doc_id")
        if doc_id is None:
            continue
        hs = item.get("hidden_states")
        if hs is None and isinstance(item.get("trajectory_a"), dict):
            hs = item["trajectory_a"].get("hidden_states")
        if hs is None:
            continue
        bucket = per_doc.setdefault(doc_id, [])
        if len(bucket) >= samples_per_class:
            continue
        for row in hs:
            if len(bucket) >= samples_per_class:
                break
            bucket.append(row.float().cpu())

    usable = [
        (doc_id, rows) for doc_id, rows in per_doc.items()
        if len(rows) >= samples_per_class
    ]
    rng.shuffle(usable)
    usable = usable[:max_classes]

    features: List[torch.Tensor] = []
    labels: List[int] = []
    class_names: List[str] = []
    counts: List[int] = []
    for cls_idx, (doc_id, rows) in enumerate(usable):
        class_names.append(doc_id)
        counts.append(len(rows))
        for row in rows:
            features.append(row)
            labels.append(cls_idx)
    return features, labels, class_names, counts


def train_identity_probe(
    autoencoder,
    reader,
    cfg,
) -> dict:
    """Train a linear probe on Φ(h) → doc_id and report held-out accuracy.

    Gradients do NOT flow into the autoencoder — Φ is used as a frozen
    feature extractor. Returns a dict with accuracy, the number of
    usable classes, and the random-chance baseline ``1/n_classes``.

    If fewer than two classes are usable (tiny smoke datasets), the
    function returns ``{"accuracy": float("nan"), ...}`` with
    ``"skipped": True`` so downstream reports can note the skip
    rather than raising.
    """
    features, labels, class_names, counts = _collect_by_doc(
        reader,
        samples_per_class=cfg.identity_probe_samples_per_class,
        max_classes=cfg.identity_probe_n_classes,
        seed=cfg.random_seed,
    )

    n_classes = len(class_names)
    if n_classes < 2:
        logger.warning(
            "Identity probe: only %d usable classes — skipping.", n_classes,
        )
        return {
            "accuracy": float("nan"),
            "n_classes": n_classes,
            "n_samples": len(features),
            "random_baseline": (1.0 / n_classes) if n_classes else 0.0,
            "skipped": True,
        }

    X = torch.stack(features, dim=0)
    y = torch.tensor(labels, dtype=torch.long)

    # 80/20 stratified-ish split: first k-1 of each class → train, rest → test.
    train_idx, test_idx = [], []
    by_class: Dict[int, List[int]] = {}
    for i, c in enumerate(labels):
        by_class.setdefault(c, []).append(i)
    for c, rows in by_class.items():
        split = max(1, int(len(rows) * 0.8))
        train_idx.extend(rows[:split])
        test_idx.extend(rows[split:])

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    device = next(autoencoder.parameters()).device
    # Extract features once (no grad); train probe on CPU or GPU.
    autoencoder.eval()
    with torch.no_grad():
        z_tr = autoencoder.encode(X_tr.to(device)).detach()
        z_te = autoencoder.encode(X_te.to(device)).detach()

    latent_dim = z_tr.shape[-1]
    probe = nn.Linear(latent_dim, n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-5)
    bs = min(512, z_tr.shape[0])
    n_tr = z_tr.shape[0]
    for _ in range(cfg.identity_probe_epochs):
        perm = torch.randperm(n_tr, device=device)
        for start in range(0, n_tr, bs):
            sel = perm[start:start + bs]
            logits = probe(z_tr[sel])
            loss = F.cross_entropy(logits, y_tr[sel].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(z_te).argmax(dim=-1).cpu()
    acc = float((preds == y_te).float().mean().item()) if len(y_te) > 0 else float("nan")

    return {
        "accuracy": acc,
        "n_classes": n_classes,
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "random_baseline": 1.0 / n_classes,
        "class_count_stats": {
            "min": int(min(counts)), "max": int(max(counts)),
            "mean": float(sum(counts) / len(counts)),
        },
        "skipped": False,
    }
