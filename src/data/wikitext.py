"""
WikiText-103 loader for D2 (real-text trajectories).

Loads WikiText-103 from HuggingFace, splits articles into paragraphs,
encodes each paragraph with a sentence-transformer, and optionally
computes LM log-probabilities for the semantic term.

The output is a trajectory dataset where each trajectory is one article,
and each event is one paragraph's embedding.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm


class WikiTextTrajectoryDataset(Dataset):
    """
    Dataset of real-text trajectories from WikiText-103.
    Each trajectory = one article, each state = one paragraph embedding.
    """

    def __init__(
        self,
        embeddings: List[torch.Tensor],
        log_probs: Optional[List[torch.Tensor]] = None,
        texts: Optional[List[List[str]]] = None,
        article_titles: Optional[List[str]] = None,
    ):
        """
        Args:
            embeddings: list of (T_i, D) tensors, one per article
            log_probs: list of (T_i - 1,) tensors of transition log-probs
            texts: list of list of paragraph strings (for decoding/debugging)
            article_titles: list of article title strings
        """
        self.embeddings = embeddings
        self.log_probs = log_probs
        self.texts = texts
        self.article_titles = article_titles or [""] * len(embeddings)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        item = {
            "trajectory": self.embeddings[idx],
            "labels": {"article_title": self.article_titles[idx], "idx": idx},
        }
        if self.log_probs is not None:
            item["log_probs"] = self.log_probs[idx]
        return item


def load_wikitext_articles(
    split: str = "train",
    min_paragraphs: int = 5,
    max_paragraphs: int = 100,
    max_articles: Optional[int] = None,
) -> List[Tuple[str, List[str]]]:
    """
    Load WikiText-103 and split into (title, [paragraphs]).

    When max_articles is set, uses streaming mode to avoid downloading
    the full ~500MB dataset — only reads what's needed.
    """
    from datasets import load_dataset

    # Stream whenever we're loading a subset: avoids downloading the full
    # ~500MB parquet. Full corpus → non-streaming (we need all of it anyway).
    use_streaming = max_articles is not None
    ds = load_dataset(
        "wikitext", "wikitext-103-raw-v1",
        split=split,
        streaming=use_streaming,
    )

    if use_streaming:
        text_iter = (row["text"] for row in ds)
    else:
        text_iter = ds["text"]

    articles = []
    current_title = ""
    current_paragraphs = []

    for line in text_iter:
        line = line.strip()

        if line.startswith("= ") and line.endswith(" =") and not line.startswith("= ="):
            if current_paragraphs and min_paragraphs <= len(current_paragraphs) <= max_paragraphs:
                articles.append((current_title, current_paragraphs))
                if max_articles and len(articles) >= max_articles:
                    break

            current_title = line.strip("= ").strip()
            current_paragraphs = []

        elif len(line) > 50:
            current_paragraphs.append(line)

    if (not max_articles or len(articles) < max_articles):
        if current_paragraphs and min_paragraphs <= len(current_paragraphs) <= max_paragraphs:
            articles.append((current_title, current_paragraphs))

    return articles


def encode_articles(
    articles: List[Tuple[str, List[str]]],
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    device: str = "cuda",
    cache_path: Optional[str] = None,
) -> List[torch.Tensor]:
    """
    Encode each paragraph of each article using a sentence-transformer.

    Returns a list of (T_i, D) tensors where T_i is the number of
    paragraphs in article i and D is the encoder dimension.
    """
    # Check cache
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached embeddings from {cache_path}")
        data = torch.load(cache_path)
        return data["embeddings"]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(encoder_name, device=device)

    # Flatten all paragraphs with article indices for batching
    all_paragraphs = []
    article_lengths = []
    for title, paragraphs in articles:
        all_paragraphs.extend(paragraphs)
        article_lengths.append(len(paragraphs))

    # Encode in batches
    print(f"Encoding {len(all_paragraphs)} paragraphs...")
    all_embeddings = model.encode(
        all_paragraphs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
    )
    all_embeddings = all_embeddings.cpu()

    # Split back into per-article tensors
    embeddings = []
    offset = 0
    for length in article_lengths:
        embeddings.append(all_embeddings[offset : offset + length])
        offset += length

    # Cache
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": embeddings}, cache_path)
        print(f"Cached embeddings to {cache_path}")

    return embeddings


def compute_lm_log_probs(
    articles: List[Tuple[str, List[str]]],
    lm_name: str = "gpt2-medium",
    max_context: int = 512,
    batch_size: int = 8,
    device: str = "cuda",
    cache_path: Optional[str] = None,
) -> List[torch.Tensor]:
    """
    Compute -log P_LM(d_{t+1} | d_{<=t}) for each transition in each article.

    For each paragraph d_{t+1}, we condition on the concatenation of all
    previous paragraphs d_0, ..., d_t (truncated to max_context tokens)
    and compute the average per-token log-probability of d_{t+1}.

    Returns a list of (T_i - 1,) tensors of negative log-probabilities.
    """
    # Check cache
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached LM scores from {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        return data["log_probs"]

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(lm_name).to(device)
    model.eval()

    max_positions = getattr(model.config, 'n_positions', 1024)

    all_log_probs = []
    n_skipped = 0
    n_computed = 0

    # Resume from partial results if available
    partial_path = str(cache_path) + ".partial" if cache_path else None
    start_idx = 0
    if partial_path and Path(partial_path).exists():
        partial = torch.load(partial_path, weights_only=False)
        all_log_probs = partial["log_probs"]
        start_idx = len(all_log_probs)
        print(f"  Resuming from article {start_idx}/{len(articles)}")

    for art_idx, (title, paragraphs) in enumerate(
        tqdm(articles, desc="Computing LM log-probs")
    ):
        if art_idx < start_idx:
            continue

        article_logprobs = []

        for t in range(len(paragraphs) - 1):
            context = " ".join(paragraphs[: t + 1])
            continuation = " " + paragraphs[t + 1]

            context_ids = tokenizer.encode(context, add_special_tokens=False)
            cont_ids = tokenizer.encode(continuation, add_special_tokens=False)

            # Truncate continuation if it alone exceeds the window
            if len(cont_ids) > max_positions - 10:
                cont_ids = cont_ids[: max_positions - 10]

            # Truncate context from the left to fit
            max_ctx_len = min(max_context, max_positions - len(cont_ids))
            if max_ctx_len < 10:
                max_ctx_len = 10
            context_ids = context_ids[-max_ctx_len:]

            input_ids = torch.tensor(
                [context_ids + cont_ids], device=device
            )

            # Hard guard: clamp to model's positional embedding size
            if input_ids.shape[1] > max_positions:
                input_ids = input_ids[:, -max_positions:]
                context_ids = context_ids[-(max_positions - len(cont_ids)):]

            ctx_len = len(context_ids)

            try:
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits

                    log_probs_per_token = torch.log_softmax(
                        logits[0, ctx_len - 1 : -1], dim=-1
                    )
                    target_ids = input_ids[0, ctx_len:]
                    token_log_probs = log_probs_per_token.gather(
                        1, target_ids.unsqueeze(1)
                    ).squeeze(1)

                    neg_log_prob = -token_log_probs.mean().item()
                n_computed += 1
            except Exception as e:
                neg_log_prob = float("nan")
                n_skipped += 1
                if n_skipped <= 5:
                    print(f"  [WARN] Skipped transition (article={art_idx}, t={t}): {e}")

            article_logprobs.append(neg_log_prob)

        all_log_probs.append(torch.tensor(article_logprobs))

        # Periodic VRAM cleanup + partial save
        if (art_idx + 1) % 500 == 0:
            if device != "cpu":
                torch.cuda.empty_cache()
            if partial_path:
                torch.save({"log_probs": all_log_probs}, partial_path)

    # Replace NaN scores with the dataset median
    all_valid = [v for lp in all_log_probs for v in lp.tolist() if not np.isnan(v)]
    if all_valid:
        median_score = float(np.median(all_valid))
        for i, lp in enumerate(all_log_probs):
            nan_mask = torch.isnan(lp)
            if nan_mask.any():
                lp[nan_mask] = median_score
                all_log_probs[i] = lp

    print(f"  LM scoring complete: {n_computed} computed, {n_skipped} skipped")

    # Cache final results
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"log_probs": all_log_probs}, cache_path)
        print(f"Cached LM scores to {cache_path}")
    # Clean up partial file
    if partial_path and Path(partial_path).exists():
        Path(partial_path).unlink()

    return all_log_probs


def build_d2_dataset(
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    lm_name: str = "gpt2-medium",
    split: str = "train",
    min_paragraphs: int = 5,
    max_paragraphs: int = 100,
    max_articles: Optional[int] = None,
    device: str = "cuda",
    cache_dir: str = "./cache",
) -> WikiTextTrajectoryDataset:
    """
    Full pipeline: load WikiText-103, encode, compute LM scores.
    """
    print("Loading WikiText-103 articles...")
    articles = load_wikitext_articles(
        split=split,
        min_paragraphs=min_paragraphs,
        max_paragraphs=max_paragraphs,
        max_articles=max_articles,
    )
    print(f"Found {len(articles)} articles with {min_paragraphs}-{max_paragraphs} paragraphs")

    embeddings = encode_articles(
        articles,
        encoder_name=encoder_name,
        device=device,
        cache_path=f"{cache_dir}/wikitext_embeddings.pt",
    )

    log_probs = compute_lm_log_probs(
        articles,
        lm_name=lm_name,
        device=device,
        cache_path=f"{cache_dir}/wikitext_lm_scores.pt",
    )

    titles = [title for title, _ in articles]
    texts = [paragraphs for _, paragraphs in articles]

    return WikiTextTrajectoryDataset(
        embeddings=embeddings,
        log_probs=log_probs,
        texts=texts,
        article_titles=titles,
    )
