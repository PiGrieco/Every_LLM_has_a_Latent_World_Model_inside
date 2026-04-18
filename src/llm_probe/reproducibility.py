"""Capture environment + model metadata for every shard manifest.

One of the failure modes we refuse to permit on v2: running a full
24–60 h Llama-3 probe on two different ``transformers`` versions and
only noticing when something doesn't line up six months later. Every
manifest carries the exact library versions, GPU, model revision, and
config hash used to produce it.
"""

from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import hashlib
import json
import logging
import platform as _platform
import sys as _sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any

import torch

from .config import ProbeConfig

logger = logging.getLogger(__name__)


def _safe_version(pkg: str) -> str:
    try:
        return _pkg_version(pkg)
    except PackageNotFoundError:
        return "not-installed"


def capture_environment() -> dict:
    """Snapshot library versions and hardware at the time of call.

    The return dict is safe to ``json.dump`` — only primitive types.
    It is attached to every shard manifest so a dataset can always be
    retraced to the exact Python / torch / transformers / CUDA / GPU
    combination that produced it.
    """
    cuda_version = getattr(torch.version, "cuda", None)
    gpu_name = None
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    return {
        "python_version": _sys.version.split()[0],
        "platform": _platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": _safe_version("transformers"),
        "tokenizers_version": _safe_version("tokenizers"),
        "datasets_version": _safe_version("datasets"),
        "accelerate_version": _safe_version("accelerate"),
        "safetensors_version": _safe_version("safetensors"),
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def _get_commit_sha(obj: Any) -> Any:
    """Best-effort extraction of an HF commit sha from a model/tokenizer."""
    for attr in ("_commit_hash", "commit_hash"):
        if hasattr(obj, attr):
            v = getattr(obj, attr)
            if v:
                return v
    # Older HF puts it under .config
    cfg = getattr(obj, "config", None)
    if cfg is not None:
        for attr in ("_commit_hash", "commit_hash"):
            if hasattr(cfg, attr):
                v = getattr(cfg, attr)
                if v:
                    return v
    return None


def _config_hash(cfg: ProbeConfig) -> str:
    """SHA1 of the cfg dict after stripping secrets (hf_token)."""
    payload = _dc.asdict(cfg)
    payload.pop("hf_token", None)
    # Canonicalise (sorted keys) so equal configs produce equal hashes.
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def capture_model_metadata(model, tokenizer, cfg: ProbeConfig) -> dict:
    """Capture identifying metadata for the loaded model.

    Unlike :func:`capture_environment`, this depends on the live model
    handle: it records architecture dims, the layer we probe, the
    exact revision / commit hash when recoverable, and a SHA1 of the
    full :class:`ProbeConfig` (minus the HF token).

    Args:
        model: Loaded causal LM.
        tokenizer: Matching tokenizer.
        cfg: Runtime configuration.

    Returns:
        JSON-serialisable dict.
    """
    model_sha = _get_commit_sha(model)
    tok_sha = _get_commit_sha(tokenizer)
    if model_sha is None:
        logger.warning(
            "Could not recover model commit sha for %s (@%s). Manifest "
            "will record revision but not a pinned sha.",
            cfg.model_name, cfg.model_revision,
        )

    try:
        probe_layer_class = str(type(model.model.layers[cfg.probe_layer]))
    except Exception:
        probe_layer_class = "unknown"

    return {
        "model_name": cfg.model_name,
        "model_revision": cfg.model_revision,
        "model_commit_sha": model_sha,
        "tokenizer_commit_sha": tok_sha,
        "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
        "n_layers": int(model.config.num_hidden_layers),
        "d_model": int(model.config.hidden_size),
        "vocab_size": int(model.config.vocab_size),
        "probe_layer_idx": int(cfg.probe_layer),
        "probe_layer_class": probe_layer_class,
        "dtype": cfg.dtype,
        "device": cfg.device,
        "config_hash": _config_hash(cfg),
    }


def config_snapshot(cfg: ProbeConfig) -> dict:
    """Redacted, JSON-serialisable snapshot of the probe config."""
    payload = _dc.asdict(cfg)
    payload.pop("hf_token", None)
    return payload
