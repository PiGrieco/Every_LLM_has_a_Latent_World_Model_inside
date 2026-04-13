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

    use_streaming = max_articles is not None and max_articles < 5000
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
        data = torch.load(cache_path)
        return data["log_probs"]

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(lm_name).to(device)
    model.eval()

    all_log_probs = []

    for title, paragraphs in tqdm(articles, desc="Computing LM log-probs"):
        article_logprobs = []

        for t in range(len(paragraphs) - 1):
            # Context: concatenation of paragraphs 0..t
            context = " ".join(paragraphs[: t + 1])
            continuation = " " + paragraphs[t + 1]

            # Tokenize context + continuation together
            context_ids = tokenizer.encode(context, add_special_tokens=False)
            cont_ids = tokenizer.encode(continuation, add_special_tokens=False)

            # Truncate context from the left to fit max_context
            max_ctx_len = max_context - len(cont_ids)
            if max_ctx_len < 10:
                max_ctx_len = 10
            context_ids = context_ids[-max_ctx_len:]

            input_ids = torch.tensor([context_ids + cont_ids], device=device)
            ctx_len = len(context_ids)

            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits  # (1, seq_len, vocab_size)

                # Compute log-prob of each continuation token
                # logits[0, ctx_len-1:-1] predicts tokens at positions ctx_len..end
                log_probs_per_token = torch.log_softmax(logits[0, ctx_len - 1 : -1], dim=-1)
                target_ids = input_ids[0, ctx_len:]
                token_log_probs = log_probs_per_token.gather(1, target_ids.unsqueeze(1)).squeeze(1)

                # Average negative log-probability (the semantic cost)
                neg_log_prob = -token_log_probs.mean().item()

            article_logprobs.append(neg_log_prob)

        all_log_probs.append(torch.tensor(article_logprobs))

    # Cache
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"log_probs": all_log_probs}, cache_path)
        print(f"Cached LM scores to {cache_path}")

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
