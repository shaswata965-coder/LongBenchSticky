import hashlib
import json
import random
from datasets import load_dataset
from transformers import AutoTokenizer


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_wikitext103_drift_blocks(
    tokenizer_name: str,
    num_samples: int = 5,
    min_tokens: int = 1024,
    seed: int = 42,
    split: str = "train",
):

    print(f"Loading local WikiText-103 from /kaggle/input/wiki-text-103-train/wiki.train.tokens...")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    ds = load_dataset(
        "text",
        data_files={"train": "/kaggle/input/wiki-text-103-train/wiki.train.tokens"},
        split=split,
        streaming=False,
    )

    rng = random.Random(seed)

    print("Assembling WikiText articles...")
    all_articles = []
    current_text = []

    for entry in ds:
        text = entry["text"].strip()
        if text == "":
            if current_text:
                all_articles.append("\n".join(current_text).strip())
                current_text = []
        else:
            current_text.append(text)
            
    if current_text:
        all_articles.append("\n".join(current_text).strip())
        
    print(f"Found {len(all_articles)} total articles. Shuffling with seed {seed}...")
    rng.shuffle(all_articles)

    samples = []
    article_idx = 0

    print(f"Scanning shuffled articles for {num_samples} samples with >= {min_tokens} tokens...")
    for article_text in all_articles:
        tokens = tokenizer(
            article_text,
            add_special_tokens=False,
        ).input_ids
        
        token_len = len(tokens)

        if token_len >= min_tokens:
            samples.append({
                "text": article_text,
                "token_count": token_len,
                "sha256": sha256(article_text),
                "article_index": article_idx,
            })
            article_idx += 1

            if len(samples) >= num_samples:
                break

    print(f"Collected {len(samples)} reproducible drift blocks")

    return samples


def get_fixed_prompt():
    """
    Optional conditioning prefix.
    Intentionally minimal and neutral for ICML evaluation.
    """
    return ""