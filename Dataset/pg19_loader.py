import hashlib
import json
import random
from datasets import Dataset
from transformers import AutoTokenizer


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_pg19_blocks(
    tokenizer_name: str,
    num_samples: int = 5,
    min_tokens: int = 2048,
    seed: int = 42,
    arrow_path: str = "/kaggle/input/notebooks/shaswatabhattacharya/pg-19-test-split/pg19_eval/data-00000-of-00001.arrow",
):

    print(f"Loading PG-19 from {arrow_path}...")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    ds = Dataset.from_file(arrow_path)

    print(f"Shuffling dataset with seed {seed}...")
    ds = ds.shuffle(seed=seed)

    samples = []
    article_idx = 0

    for entry in ds:
        text = entry["text"].strip()
        
        # Fast heuristic: 1 token ~ max 5 characters for English text.
        # We grab enough raw characters to guarantee we have `min_tokens + 50` tokens.
        target_tokens = min_tokens + 50
        raw_chunk = text[:target_tokens * 6]
        
        # Tokenize the chunk to precisely cut the text to the target token length
        tokens = tokenizer(
            raw_chunk,
            add_special_tokens=False,
            truncation=True,
            max_length=target_tokens
        ).input_ids

        token_len = len(tokens)

        if token_len >= min_tokens:
            # Decode the exact `target_tokens` span to get clean, truncated string
            exact_text = tokenizer.decode(tokens, clean_up_tokenization_spaces=False).strip()
            
            samples.append({
                "text": exact_text,
                "token_count": token_len,
                "sha256": sha256(exact_text), # Hash the exact text block 
                "article_index": article_idx,
            })

            article_idx += 1

            if len(samples) >= num_samples:
                break

    print(f"Collected {len(samples)} reproducible drift blocks from PG-19")

    return samples


def get_fixed_prompt():
    """
    Optional conditioning prefix.
    """
    return ""
