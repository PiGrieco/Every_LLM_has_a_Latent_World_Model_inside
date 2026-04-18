"""Shard-based writer / reader for trajectory datasets.

Each dataset (forward, branching, reversed, validation) lives in its
own directory under ``cfg.output_dir``. Items are accumulated into a
small buffer and flushed to disk as torch-pickled shards whenever the
buffer reaches ``cfg.shard_size``. A JSON manifest alongside the
shards keeps fast metadata access without loading anything.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator, List, Optional, Set  # noqa: F401

import torch

logger = logging.getLogger(__name__)


_SAVE_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _cast_tensors(obj: Any, target: torch.dtype) -> Any:
    """Recursively cast any float32/float64 tensor in ``obj`` to ``target``."""
    if isinstance(obj, torch.Tensor):
        if obj.dtype in (torch.float32, torch.float64):
            return obj.to(dtype=target)
        return obj
    if isinstance(obj, dict):
        return {k: _cast_tensors(v, target) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cast_tensors(v, target) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cast_tensors(v, target) for v in obj)
    return obj


class TrajectoryShardWriter:
    """Append-only shard writer with a JSON manifest.

    Usage::

        metadata = {
            "environment": capture_environment(),
            "model_metadata": capture_model_metadata(model, tokenizer, cfg),
            "probe_config_snapshot": config_snapshot(cfg),
        }
        with TrajectoryShardWriter(dir_path, shard_size=100,
                                   save_dtype="float16",
                                   metadata=metadata,
                                   dataset_name="forward") as w:
            for item in items:
                w.add(item)

    The context manager guarantees a final flush on exit. The manifest
    ``manifest.json`` is rewritten atomically on every flush and carries
    the exact library versions, model commit sha, and probe config used
    to produce the shards.
    """

    def __init__(
        self,
        dataset_dir: str,
        shard_size: int = 100,
        save_dtype: str = "float16",
        metadata: Optional[dict] = None,
        dataset_name: Optional[str] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self.save_dtype_name = save_dtype
        self.save_dtype = _SAVE_DTYPE_MAP.get(save_dtype.lower(), torch.float16)
        self._buffer: List[dict] = []
        self._metadata = dict(metadata) if metadata else {}
        # Derive dataset_name from the dir unless explicitly given.
        self._dataset_name = dataset_name or self.dataset_dir.name

        # Manifest state — reload if we're resuming an existing directory.
        self._manifest_path = self.dataset_dir / "manifest.json"
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path) as f:
                    self._manifest = json.load(f)
                self._shard_idx = int(self._manifest.get("n_shards", 0))
                self._n_items_total = int(self._manifest.get("n_items_total", 0))
                logger.info(
                    "Resuming writer at %s: %d existing shards, %d items",
                    self.dataset_dir, self._shard_idx, self._n_items_total,
                )
                # Refresh top-level metadata so the manifest always reflects
                # the CURRENT run's env (which matters if someone upgrades
                # transformers between resumptions — mismatches surface in
                # reviews of manifest.json diffs).
                self._merge_metadata()
            except Exception as exc:
                logger.warning("Could not read existing manifest (%s); starting fresh", exc)
                self._init_manifest()
        else:
            self._init_manifest()

    def _merge_metadata(self) -> None:
        """Update the manifest's metadata block, keeping created_at stable."""
        for k, v in self._metadata.items():
            self._manifest[k] = v
        self._manifest.setdefault("dataset_name", self._dataset_name)

    def _init_manifest(self) -> None:
        self._shard_idx = 0
        self._n_items_total = 0
        self._manifest = {
            "dataset_dir": str(self.dataset_dir),
            "dataset_name": self._dataset_name,
            "n_shards": 0,
            "n_items_total": 0,
            "shards": [],
            "save_dtype": self.save_dtype_name,
            "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        # Fold caller-supplied reproducibility metadata in.
        self._merge_metadata()

    # ---- context manager ----
    def __enter__(self) -> "TrajectoryShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.flush()

    # ---- public API ----
    def existing_doc_ids(self) -> Set[str]:
        """Return the set of ``doc_id`` values already persisted (any shard)."""
        return set(d for s in self._manifest.get("shards", []) for d in s.get("doc_ids", []))

    def add(self, item: dict) -> None:
        """Append an item and flush automatically if the shard is full."""
        if not isinstance(item, dict):
            raise TypeError(f"add() expects a dict; got {type(item).__name__}")
        self._buffer.append(item)
        if len(self._buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        """Persist the current buffer as a shard and update the manifest."""
        if not self._buffer:
            return
        shard_name = f"shard_{self._shard_idx:05d}.pt"
        shard_path = self.dataset_dir / shard_name
        items_cast = [_cast_tensors(it, self.save_dtype) for it in self._buffer]
        torch.save(
            {"items": items_cast, "shard_idx": self._shard_idx},
            shard_path,
        )
        doc_ids = [str(it.get("doc_id", "")) for it in self._buffer]
        self._manifest["shards"].append({
            "idx": self._shard_idx,
            "path": shard_name,
            "n_items": len(self._buffer),
            "doc_ids": doc_ids,
        })
        self._shard_idx += 1
        self._n_items_total += len(self._buffer)
        self._manifest["n_shards"] = self._shard_idx
        self._manifest["n_items_total"] = self._n_items_total
        self._buffer.clear()

        tmp = self._manifest_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._manifest, f, indent=2)
        os.replace(tmp, self._manifest_path)
        logger.info(
            "Flushed %s (%d items); total so far %d",
            shard_name, self._manifest["shards"][-1]["n_items"], self._n_items_total,
        )


class TrajectoryShardReader:
    """Read-only companion to :class:`TrajectoryShardWriter`."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self._manifest_path = self.dataset_dir / "manifest.json"
        if not self._manifest_path.exists():
            raise FileNotFoundError(
                f"No manifest.json under {dataset_dir}. Has the dataset been built?"
            )
        with open(self._manifest_path) as f:
            self._manifest = json.load(f)

    def __len__(self) -> int:
        return int(self._manifest.get("n_items_total", 0))

    @property
    def manifest(self) -> dict:
        return dict(self._manifest)

    def iter_items(self) -> Iterator[dict]:
        """Yield items from every shard in manifest order."""
        for shard_meta in self._manifest.get("shards", []):
            path = self.dataset_dir / shard_meta["path"]
            data = torch.load(path, weights_only=False)
            for item in data["items"]:
                yield item

    def load_shard(self, idx: int) -> List[dict]:
        """Return the full list of items from shard ``idx``."""
        shards = self._manifest.get("shards", [])
        if idx < 0 or idx >= len(shards):
            raise IndexError(f"Shard {idx} out of range (have {len(shards)}).")
        path = self.dataset_dir / shards[idx]["path"]
        return list(torch.load(path, weights_only=False)["items"])

    def get_doc_ids(self) -> Set[str]:
        """Union of doc_ids across all shards."""
        return set(d for s in self._manifest.get("shards", []) for d in s.get("doc_ids", []))

    def get_metadata(self) -> dict:
        """Return environment + model_metadata + probe_config_snapshot."""
        m = self._manifest
        return {
            "environment": m.get("environment", {}),
            "model_metadata": m.get("model_metadata", {}),
            "probe_config_snapshot": m.get("probe_config_snapshot", {}),
            "dataset_name": m.get("dataset_name", ""),
            "created_at": m.get("created_at", ""),
        }
