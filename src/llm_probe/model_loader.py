"""Llama-3 loading + forward-hook installation for layer probing.

Loads the causal LM in half precision on a single GPU, registers a
forward hook on the requested decoder layer, and exposes helpers to
validate the assumptions the rest of the probing pipeline relies on
(layer indexing, block structure).
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Tuple

import torch
import torch.nn as nn

from .config import ProbeConfig
from .reproducibility import capture_model_metadata

logger = logging.getLogger(__name__)


_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def load_model(
    cfg: ProbeConfig,
) -> Tuple[nn.Module, Any, dict]:
    """Load model + tokenizer and return (model, tokenizer, info).

    The HF access token is read from ``cfg.hf_token`` and, if unset,
    from the ``HF_TOKEN`` environment variable. Model is placed on a
    single GPU via ``device_map={"": 0}`` and switched to ``eval()``.
    SDPA is preferred over Flash Attention 2 because it is the official
    path for Llama-3 in transformers 4.44.x.

    Args:
        cfg: Runtime configuration.

    Returns:
        Tuple ``(model, tokenizer, info)`` where ``info`` is the output
        of :func:`~.reproducibility.capture_model_metadata` and includes
        the model revision / commit sha, architecture dims, dtype, and
        a redacted hash of the probe config.

    Raises:
        RuntimeError: If no Hugging Face token is available and the
            model is gated (Llama-3 is).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = cfg.hf_token or os.environ.get("HF_TOKEN")
    if token is None and "llama" in cfg.model_name.lower():
        raise RuntimeError(
            "No Hugging Face token available. Set cfg.hf_token or the "
            "HF_TOKEN environment variable. Llama-3 is a gated model."
        )

    dtype = _DTYPE_MAP.get(cfg.dtype.lower())
    if dtype is None:
        raise ValueError(f"Unknown dtype {cfg.dtype!r}; expected one of {list(_DTYPE_MAP)}")

    logger.info(
        "Loading tokenizer %s @ revision=%s", cfg.model_name, cfg.model_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, revision=cfg.model_revision, token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(
        "Loading model %s @ revision=%s (dtype=%s) on %s",
        cfg.model_name, cfg.model_revision, cfg.dtype, cfg.device,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        revision=cfg.model_revision,
        torch_dtype=dtype,
        device_map={"": 0} if cfg.device == "cuda" else cfg.device,
        attn_implementation="sdpa",
        token=token,
    )
    model.eval()

    info = capture_model_metadata(model, tokenizer, cfg)
    logger.info("Model loaded: %s", info)
    return model, tokenizer, info


def install_activation_hook(
    model: nn.Module,
    layer_idx: int,
) -> Tuple[torch.utils.hooks.RemovableHandle, List[torch.Tensor]]:
    """Install a forward hook capturing hidden states post-layer.

    The captured list is cleared at the start of every hook invocation,
    so after a single forward pass ``captured[0]`` is the hidden-state
    tensor of shape ``(batch, seq_len, d_model)``. The caller is
    responsible for calling ``handle.remove()`` when finished.

    Args:
        model: A causal LM whose ``model.model.layers[layer_idx]`` is
            a standard decoder block. Use :func:`validate_model_structure`
            to double-check.
        layer_idx: 0-indexed layer to probe.

    Returns:
        Tuple of ``(handle, captured_list)``.
    """
    layers = model.model.layers
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(
            f"probe_layer={layer_idx} out of range; model has {len(layers)} layers"
        )

    captured: List[torch.Tensor] = []

    def _hook(_module, _inputs, output):
        # Decoder layer returns a tuple: (hidden_states, *extras)
        hs = output[0] if isinstance(output, tuple) else output
        captured.clear()
        captured.append(hs)

    handle = layers[layer_idx].register_forward_hook(_hook)
    logger.debug("Installed activation hook on layer %d", layer_idx)
    return handle, captured


def validate_model_structure(model: nn.Module, cfg: ProbeConfig) -> None:
    """Sanity check the model layout against our probing assumptions.

    Raises:
        RuntimeError: If the layer layout does not match what the rest
            of the module expects (no ``model.model.layers`` sequence,
            or the target block is missing a standard decoder interface).
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(
            "Model does not expose .model.layers — only standard "
            "decoder-only LMs are supported by this probe."
        )
    layers = model.model.layers
    if cfg.probe_layer < 0 or cfg.probe_layer >= len(layers):
        raise RuntimeError(
            f"probe_layer={cfg.probe_layer} out of range for model with "
            f"{len(layers)} layers."
        )
    block = layers[cfg.probe_layer]
    for name in ("self_attn", "mlp"):
        if not hasattr(block, name):
            raise RuntimeError(
                f"Layer {cfg.probe_layer} is not a standard decoder block: "
                f"missing .{name}"
            )
    logger.info(
        "Model structure validated: layer %d of %d, d_model=%d",
        cfg.probe_layer, len(layers), model.config.hidden_size,
    )
