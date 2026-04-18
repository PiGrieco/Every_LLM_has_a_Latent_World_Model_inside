"""Window-pool hidden states into trajectory sequences.

Per-token hidden states from one decoder layer are pooled into
overlapping windows to produce a manageable, fixed-size trajectory.
The resulting tensor is the raw material for the v2 geometric stack
that follows in later milestones.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch

from .config import ProbeConfig

logger = logging.getLogger(__name__)


_SAVE_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def window_pool(
    hidden_states: torch.Tensor,
    window: int,
    stride: int,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Mean-pool a per-token hidden-state matrix into overlapping windows.

    Args:
        hidden_states: ``(seq_len, d_model)`` tensor.
        window: window length in tokens.
        stride: hop between consecutive windows.

    Returns:
        pooled: ``(n_windows, d_model)`` tensor of mean-pooled features.
        positions: list of ``(start, end)`` token offsets for each
            window, half-open (``end`` exclusive).

    Notes:
        Returns ``(Tensor(0, d_model), [])`` when ``seq_len < window``
        rather than raising, so the caller can skip short trajectories
        without a try/except.
    """
    if hidden_states.dim() != 2:
        raise ValueError(
            f"window_pool expects (seq_len, d_model); got shape {tuple(hidden_states.shape)}"
        )
    seq_len, d_model = hidden_states.shape
    if seq_len < window:
        return hidden_states.new_zeros((0, d_model)), []

    n_windows = (seq_len - window) // stride + 1
    pooled = hidden_states.new_zeros((n_windows, d_model))
    positions: List[Tuple[int, int]] = []
    for i in range(n_windows):
        start = i * stride
        end = start + window
        pooled[i] = hidden_states[start:end].mean(dim=0)
        positions.append((start, end))
    return pooled, positions


def extract_trajectory_states(
    model,
    input_ids: torch.Tensor,
    captured_list: List[torch.Tensor],
    cfg: ProbeConfig,
) -> dict:
    """Run a forward pass and pool the captured hidden states.

    Args:
        model: model carrying an installed hook (from
            :func:`install_activation_hook`). The hook writes post-layer
            hidden states into ``captured_list``.
        input_ids: token tensor of shape ``(seq_len,)`` or
            ``(1, seq_len)``. Moved to the model's device internally.
        captured_list: mutable list shared with the forward hook.
        cfg: probe configuration (window sizes, save dtype).

    Returns:
        dict with keys:
            - ``hidden_states``: ``(n_windows, d_model)`` CPU tensor in
              ``cfg.save_dtype``
            - ``token_positions``: list of ``(start, end)`` offsets
            - ``seq_len``: integer sequence length actually fed

    Notes:
        All returned tensors live on CPU so GPU memory does not grow
        across many calls in a long run.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    elif input_ids.dim() != 2:
        raise ValueError(
            f"input_ids must be 1D or 2D; got shape {tuple(input_ids.shape)}"
        )

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    captured_list.clear()
    use_amp = cfg.dtype.lower() in ("float16", "fp16", "bfloat16", "bf16")
    amp_dtype = _SAVE_DTYPE_MAP.get(cfg.dtype.lower(), torch.float16)

    with torch.no_grad():
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _ = model(input_ids, use_cache=False)
        else:
            _ = model(input_ids, use_cache=False)

    if not captured_list:
        raise RuntimeError(
            "Forward hook did not capture anything. Was the hook installed "
            "on the right module?"
        )
    hs = captured_list[0]
    # Expect (1, seq_len, d_model) — drop batch.
    if hs.dim() == 3:
        hs = hs.squeeze(0)
    seq_len = hs.shape[0]

    pooled, positions = window_pool(hs, cfg.window_size, cfg.window_stride)
    save_dtype = _SAVE_DTYPE_MAP.get(cfg.save_dtype.lower(), torch.float16)
    pooled = pooled.to(dtype=save_dtype).cpu()

    return {
        "hidden_states": pooled,
        "token_positions": positions,
        "seq_len": int(seq_len),
    }
