import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np
import os
import glob
from sticky_config import dataset_tracker
from npz_io import save_results_npz
import sticky_config as config

OUTPUT_FILE = "pure_vanilla_baseline_results.npz"
TRACKED_LAYERS = config.TRACKED_LAYERS
TRACKED_HEADS = list(range(config.NUM_KV_HEADS))


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    setup_seed(config.SEEDS[0])

    for f in glob.glob(OUTPUT_FILE.replace('.npz', '*.npz')):
        print(f"Removing existing {f} to prevent appending bugs...")
        os.remove(f)

    print(f"Loading pure HuggingFace causal LM (no sticky) from {config.MODEL_PATH}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_tracker == 1:
        from pg19_loader import get_pg19_blocks
        samples = get_pg19_blocks(config.MODEL_PATH, num_samples=config.NUM_SAMPLES, min_tokens=config.DATASET_MIN_TOKENS)
    else:
        from wiki_text_loader import get_wikitext103_drift_blocks
        samples = get_wikitext103_drift_blocks(config.MODEL_PATH, num_samples=config.NUM_SAMPLES, min_tokens=config.DATASET_MIN_TOKENS)

    results = []

    for idx, sample in enumerate(samples):
        text = sample["text"]
        truncate_pct = random.uniform(0.5, 0.6)
        target_len = int(len(text) * truncate_pct)

        cut_idx = text.rfind('.', 0, target_len)
        if cut_idx == -1:
            cut_idx = text.rfind(' ', 0, target_len)
        if cut_idx == -1:
            cut_idx = target_len

        truncation_char_index = cut_idx + 1
        truncated_text = text[:truncation_char_index].strip()
        remaining_text = text[cut_idx + 1:].strip()

        gt_tokens = tokenizer(remaining_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        num_gt_tokens = min(config.GENERATION_CONFIG.get("max_new_tokens", 512), len(gt_tokens))

        if num_gt_tokens == 0:
            print(f"Warning: No remaining text for sample {idx}. Skipping.")
            continue

        messages = [{"role": "user", "content": f"Please write a comprehensive, detailed 200-word continuation expanding on the following text. Do not stop early:\n\n{truncated_text}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        results.append({
            "metadata": {
                "sha256": sample["sha256"],
                "article_index": sample["article_index"],
                "token_count_input": int(inputs.input_ids.shape[1]),
                "generated_token_count": int(num_gt_tokens),
                "generated_token_ids": gt_tokens[:num_gt_tokens].tolist(),
                "truncation_char_index": truncation_char_index,
                "teacher_forcing": True,
            },
            "tracked_layers": TRACKED_LAYERS,
            "tracked_heads": TRACKED_HEADS,
            "prefill_attention": {},
            "prefill_window_scores": {},
            "generation_attention": [],
            "generation_window_scores": [],
            "generation_attention_fresh": [],
            "generation_window_scores_fresh": [],
        })

    save_results_npz(results, OUTPUT_FILE)
    print(f"Saved pure vanilla baseline metadata to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
