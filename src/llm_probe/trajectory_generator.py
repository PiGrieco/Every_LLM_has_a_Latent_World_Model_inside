"""Trajectory generators: forward, branching pairs, reversed.

Each generator consumes one article and produces one or more
trajectories captured at ``cfg.probe_layer`` via the pre-installed
forward hook. All three are wrapped so a per-article failure logs a
warning and returns ``None`` rather than aborting an overnight run.
"""

from __future__ import annotations

import logging
import traceback
from typing import List, Optional

import torch

from .activation_extractor import extract_trajectory_states
from .config import ProbeConfig

logger = logging.getLogger(__name__)


def _gen_config_dict(cfg: ProbeConfig, seed_used: int, num_return_sequences: int) -> dict:
    """Effective generation params recorded into every trajectory item."""
    return {
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "seed_used": int(seed_used),
        "num_return_sequences": int(num_return_sequences),
        "max_new_tokens": cfg.continuation_tokens,
        "prompt_tokens": cfg.prompt_tokens,
    }


def _prompt_ids(article: dict, tokenizer, cfg: ProbeConfig, device) -> torch.Tensor:
    """Encode an article into the first ``cfg.prompt_tokens`` tokens."""
    enc = tokenizer(
        article["text"],
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
    )
    ids = enc.input_ids[0, : cfg.prompt_tokens].to(device)
    if ids.numel() == 0:
        raise ValueError(f"Article {article.get('doc_id')} produced empty prompt_ids")
    return ids


def generate_forward_trajectories(
    model,
    tokenizer,
    article: dict,
    cfg: ProbeConfig,
    doc_idx: int,
    hook_handle,
    captured_list: List[torch.Tensor],
) -> Optional[List[dict]]:
    """Sample ``k_trajectories`` continuations from the same prompt.

    Generates ``K = cfg.k_trajectories`` forward trajectories from the
    article's first ``prompt_tokens`` tokens, then runs a *separate*
    forward pass per sequence with the hook active to capture
    per-layer hidden states. We do it this way (not during generate)
    because the hook would otherwise fire on each step's KV-cache
    forward and mix the autoregressive expansions.

    Args:
        model: Causal LM with an installed activation hook.
        tokenizer: Matching tokenizer.
        article: Pool entry ``{doc_id, title, text, token_count}``.
        cfg: Probe configuration.
        doc_idx: Article index, used to seed generation.
        hook_handle: Opaque handle (kept for API symmetry; unused here).
        captured_list: List shared with the forward hook.

    Returns:
        List of ``cfg.k_trajectories`` dicts, or ``None`` on failure.
        Each dict contains keys ``doc_id``, ``trajectory_idx``,
        ``prompt_tokens``, ``full_tokens``, ``hidden_states``,
        ``token_positions``, ``seq_len``.
    """
    del hook_handle  # unused; kept for uniform call signatures
    try:
        device = next(model.parameters()).device
        prompt_ids = _prompt_ids(article, tokenizer, cfg, device)

        seed_used = cfg.gen_seed_base + doc_idx
        torch.manual_seed(seed_used)
        with torch.no_grad():
            out = model.generate(
                input_ids=prompt_ids.unsqueeze(0),
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_new_tokens=cfg.continuation_tokens,
                num_return_sequences=cfg.k_trajectories,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        sequences = out.sequences  # (K, <= prompt + continuation)
        gen_cfg = _gen_config_dict(cfg, seed_used, cfg.k_trajectories)

        results: List[dict] = []
        for k in range(sequences.shape[0]):
            full_tokens = sequences[k]
            # Drop any trailing pads (all EOS == pad for us).
            full_tokens = _strip_trailing_pad(full_tokens, tokenizer.eos_token_id)
            if full_tokens.numel() < cfg.window_size:
                # Too short to produce any window — skip this k but keep going.
                continue
            traj = extract_trajectory_states(
                model, full_tokens, captured_list, cfg,
            )
            results.append({
                "doc_id": article["doc_id"],
                "trajectory_idx": k,
                "prompt_tokens": prompt_ids.cpu(),
                "full_tokens": full_tokens.cpu(),
                "hidden_states": traj["hidden_states"],
                "token_positions": traj["token_positions"],
                "seq_len": traj["seq_len"],
                "generation_config": gen_cfg,
            })
        if not results:
            logger.warning("Article %s produced no usable forward trajectories", article["doc_id"])
            return None
        return results
    except Exception as exc:
        logger.warning(
            "forward failed on doc %s (idx=%d): %s\n%s",
            article.get("doc_id"), doc_idx, exc, traceback.format_exc(),
        )
        return None


def generate_branching_pairs(
    model,
    tokenizer,
    article: dict,
    cfg: ProbeConfig,
    doc_idx: int,
    hook_handle,
    captured_list: List[torch.Tensor],
) -> Optional[List[dict]]:
    """Produce branching trajectory pairs that split at high-entropy tokens.

    For each of ``cfg.n_pairs_per_article`` pairs:
      1. sample a single base continuation ``τ_base``;
      2. run a logits-only forward over ``τ_base`` to locate the first
         position inside ``[branching_window_start, branching_window_end)``
         with entropy above ``cfg.entropy_threshold`` (fallback: argmax
         of entropy if no position exceeds the threshold);
      3. pick an alternative top-k token at that position and resample
         the rest of the sequence with a different seed;
      4. extract hidden states for both ``τ_a`` and ``τ_b`` separately.

    Returns ``None`` on per-article failure; individual failed pairs are
    skipped silently and the rest of the list is still returned.
    """
    del hook_handle
    try:
        device = next(model.parameters()).device
        prompt_ids = _prompt_ids(article, tokenizer, cfg, device)

        out: List[dict] = []
        for pair_idx in range(cfg.n_pairs_per_article):
            try:
                # 1. Base trajectory.
                seed_base_traj = cfg.gen_seed_base + doc_idx + pair_idx * 100
                torch.manual_seed(seed_base_traj)
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=prompt_ids.unsqueeze(0),
                        do_sample=True,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        max_new_tokens=cfg.continuation_tokens,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                    )
                full_tokens = _strip_trailing_pad(gen.sequences[0], tokenizer.eos_token_id)

                # 2. Find branching point by entropy.
                with torch.no_grad():
                    logits = model(
                        full_tokens.unsqueeze(0), use_cache=False,
                    ).logits[0]
                start = cfg.prompt_tokens + cfg.branching_window_start
                end = min(
                    cfg.prompt_tokens + cfg.branching_window_end,
                    full_tokens.shape[0] - 1,
                )
                if start >= end:
                    logger.warning(
                        "Article %s pair %d: branching window empty (start=%d end=%d seq=%d)",
                        article["doc_id"], pair_idx, start, end, full_tokens.shape[0],
                    )
                    continue
                probs = torch.softmax(logits[start:end], dim=-1)
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
                mask = entropy > cfg.entropy_threshold
                t_rel = int(mask.float().argmax().item()) if bool(mask.any()) else int(entropy.argmax().item())
                t_abs = start + t_rel

                # 3. Pick a top-5 alternative.
                topk = logits[t_abs - 1].topk(5)
                original_token = int(full_tokens[t_abs].item())
                alternatives = [
                    int(t.item()) for t in topk.indices if int(t.item()) != original_token
                ][:4]
                if not alternatives:
                    continue
                alt_token = alternatives[(pair_idx + doc_idx) % len(alternatives)]

                # 4. Build the alt prefix and generate the rest.
                prefix_alt = torch.cat([
                    full_tokens[:t_abs].clone(),
                    torch.tensor([alt_token], device=device, dtype=full_tokens.dtype),
                ])
                target_total = prompt_ids.shape[0] + cfg.continuation_tokens
                remaining = max(1, target_total - prefix_alt.shape[0])
                seed_alt = cfg.gen_seed_base + doc_idx + pair_idx * 100 + 77
                torch.manual_seed(seed_alt)
                with torch.no_grad():
                    gen_alt = model.generate(
                        input_ids=prefix_alt.unsqueeze(0),
                        do_sample=True,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        max_new_tokens=remaining,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                    )
                full_tokens_alt = _strip_trailing_pad(
                    gen_alt.sequences[0], tokenizer.eos_token_id,
                )

                # 5. Extract states for both.
                if (full_tokens.numel() < cfg.window_size
                        or full_tokens_alt.numel() < cfg.window_size):
                    continue
                traj_a = extract_trajectory_states(model, full_tokens, captured_list, cfg)
                traj_b = extract_trajectory_states(model, full_tokens_alt, captured_list, cfg)

                out.append({
                    "doc_id": article["doc_id"],
                    "pair_idx": pair_idx,
                    "branching_point": int(t_abs),
                    "original_token": original_token,
                    "intervention_token": alt_token,
                    "trajectory_a": {
                        "full_tokens": full_tokens.cpu(),
                        "hidden_states": traj_a["hidden_states"],
                        "token_positions": traj_a["token_positions"],
                        "seq_len": traj_a["seq_len"],
                    },
                    "trajectory_b": {
                        "full_tokens": full_tokens_alt.cpu(),
                        "hidden_states": traj_b["hidden_states"],
                        "token_positions": traj_b["token_positions"],
                        "seq_len": traj_b["seq_len"],
                    },
                    "generation_config": {
                        "base": _gen_config_dict(cfg, seed_base_traj, 1),
                        "alt": _gen_config_dict(cfg, seed_alt, 1),
                    },
                })
            except Exception as exc:
                logger.warning(
                    "branching pair failed on doc %s pair %d: %s",
                    article.get("doc_id"), pair_idx, exc,
                )
                continue
        if not out:
            return None
        return out
    except Exception as exc:
        logger.warning(
            "branching failed on doc %s (idx=%d): %s\n%s",
            article.get("doc_id"), doc_idx, exc, traceback.format_exc(),
        )
        return None


def extract_reversed_pair(
    model,
    tokenizer,
    article: dict,
    cfg: ProbeConfig,
    doc_idx: int,
    hook_handle,
    captured_list: List[torch.Tensor],
) -> Optional[dict]:
    """Extract hidden states for a passage in forward and reversed order.

    Samples a random ``cfg.reversed_passage_tokens``-long window from
    the tokenized article (seeded deterministically), runs two separate
    forward passes with the hook active, and returns both trajectories.

    Returns ``None`` when the article is shorter than the passage window.
    """
    del hook_handle
    try:
        device = next(model.parameters()).device
        enc = tokenizer(
            article["text"],
            return_tensors="pt",
            add_special_tokens=True,
            truncation=False,
        )
        tokens = enc.input_ids[0].to(device)
        if tokens.shape[0] <= cfg.reversed_passage_tokens:
            logger.warning(
                "Article %s too short for reversed passage (%d <= %d)",
                article.get("doc_id"), tokens.shape[0], cfg.reversed_passage_tokens,
            )
            return None

        seed_rev = cfg.gen_seed_base + doc_idx + 999_999
        torch.manual_seed(seed_rev)
        max_start = int(tokens.shape[0] - cfg.reversed_passage_tokens)
        start = int(torch.randint(0, max_start + 1, (1,)).item())
        slice_ids = tokens[start : start + cfg.reversed_passage_tokens]
        reversed_ids = slice_ids.flip(0)

        fwd = extract_trajectory_states(model, slice_ids, captured_list, cfg)
        rev = extract_trajectory_states(model, reversed_ids, captured_list, cfg)

        return {
            "doc_id": article["doc_id"],
            "forward_tokens": slice_ids.cpu(),
            "reversed_tokens": reversed_ids.cpu(),
            "forward_hidden": fwd["hidden_states"],
            "reversed_hidden": rev["hidden_states"],
            "forward_positions": fwd["token_positions"],
            "reversed_positions": rev["token_positions"],
            "forward_seq_len": fwd["seq_len"],
            "reversed_seq_len": rev["seq_len"],
            "generation_config": {
                "seed_used": int(seed_rev),
                "passage_tokens": int(cfg.reversed_passage_tokens),
                "start_offset": int(start),
            },
        }
    except Exception as exc:
        logger.warning(
            "reversed failed on doc %s (idx=%d): %s\n%s",
            article.get("doc_id"), doc_idx, exc, traceback.format_exc(),
        )
        return None


def _strip_trailing_pad(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Drop trailing ``pad_id`` tokens so sequences reflect their real
    (post-EOS) length. Keeps at least one token."""
    if tokens.numel() == 0:
        return tokens
    # Find the last non-pad position.
    mask = tokens != pad_id
    if not bool(mask.any()):
        return tokens[:1]
    last = int(mask.nonzero(as_tuple=False)[-1].item())
    return tokens[: last + 1]
