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
    min_tokens: int = 2048,
    seed: int = 42,
    split: str = "train",
):

    print(f"Loading local WikiText-103 from Datasets/wiki.train.tokens...")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    ds = load_dataset(
        "text",
        data_files={"train": "Datasets/wiki.train.tokens"},
        split=split,
        streaming=True,
    )

    rng = random.Random(seed)

    samples = []
    current_text = ""
    current_tokens = 0
    article_idx = 0

    for entry in ds:
        text = entry["text"].strip()

        # WikiText article boundaries are marked by empty lines
        if text == "":
            if current_tokens >= min_tokens:
                samples.append({
                    "text": current_text.strip(),
                    "token_count": current_tokens,
                    "sha256": sha256(current_text),
                    "article_index": article_idx,
                })

            current_text = ""
            current_tokens = 0
            article_idx += 1

            if len(samples) >= num_samples:
                break

            continue

        token_len = len(
            tokenizer(
                text,
                add_special_tokens=False,
            ).input_ids
        )

        current_text += text + "\n"
        current_tokens += token_len

    print(f"Collected {len(samples)} reproducible drift blocks")

    return samples


def get_fixed_prompt():
    """
    Optional conditioning prefix.
    Intentionally minimal and neutral for ICML evaluation.
    """
    return ""