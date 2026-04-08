import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from sticky_llama_model import STICKYLlamaForCausalLM
from configuration_sticky_llama import LlamaConfig
import json
import random
import numpy as np
import os
import gc
import glob
from tqdm import tqdm
from sticky_config import OMEGA, SINK_TOKENS, dataset_tracker
from npz_io import save_results_npz
import sticky_config as config

# --- Configuration ---
OUTPUT_FILE = "vanilla_baseline_results.npz"

GROUP_SIZE = config.NUM_Q_HEADS // config.NUM_KV_HEADS
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
    
    print(f"Loading STICKYLlama (eviction DISABLED) from {config.MODEL_PATH}...")
    try:
        model_config = LlamaConfig.from_pretrained(config.MODEL_PATH)
        
        if hasattr(model_config, "rope_scaling") and model_config.rope_scaling is not None:
            if "rope_type" in model_config.rope_scaling and "type" not in model_config.rope_scaling:
                model_config.rope_scaling["type"] = model_config.rope_scaling["rope_type"]
        
        model_config.rope_theta = getattr(model_config, "rope_theta", config.ROPE_THETA)
            
        # CRITICAL: Set r_ratio=100 to keep 100% of tokens (no eviction)
        model_config.r_ratio = 100   # <-- This disables eviction
        model_config.start_idx = 0

        model = STICKYLlamaForCausalLM.from_pretrained(
            config.MODEL_PATH, 
            config=model_config,
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Model loaded (r_ratio=100, no eviction). Tracking {len(config.TRACKED_LAYERS)} layers and {len(TRACKED_HEADS)} heads.")

    # Load samples using the local PG-19 or WikiText loader
    if dataset_tracker == 1:
        from pg19_loader import get_pg19_blocks
        samples = get_pg19_blocks(config.MODEL_PATH, num_samples=config.NUM_SAMPLES, min_tokens=config.DATASET_MIN_TOKENS)
    else:
        from wiki_text_loader import get_wikitext103_drift_blocks
        samples = get_wikitext103_drift_blocks(config.MODEL_PATH, num_samples=config.NUM_SAMPLES, min_tokens=config.DATASET_MIN_TOKENS)
    
    results = []

    for idx, sample in enumerate(samples):
        text = sample["text"]
        
        # Truncate between 50% and 90% of the text length
        truncate_pct = random.uniform(0.5, 0.6)
        target_len = int(len(text) * truncate_pct)
        
        # Snap strictly to the nearest grammatical period to ensure the instruct model is fed a complete thought
        cut_idx = text.rfind('.', 0, target_len)
        if cut_idx == -1: cut_idx = text.rfind(' ', 0, target_len)
        if cut_idx == -1: cut_idx = target_len
        
        # Keep the period so the block is fully grammatical
        truncation_char_index = cut_idx + 1
        truncated_text = text[:truncation_char_index].strip()
        remaining_text = text[cut_idx + 1:].strip()
        
        print(f"Processing sample {idx + 1}/{len(samples)} (original={len(text)}, truncated={len(truncated_text)} chars, remaining={len(remaining_text)} chars)...")
        
        # --- Tokenize the REMAINING article text to get ground-truth continuation tokens ---
        gt_tokens = tokenizer(remaining_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        num_gt_tokens = min(config.GENERATION_CONFIG.get("max_new_tokens", 512), len(gt_tokens))
        
        if num_gt_tokens == 0:
            print(f"  Warning: No remaining text for sample {idx}. Skipping.")
            continue
        
        gt_continuation = gt_tokens[:num_gt_tokens]  # [num_gt_tokens]
        print(f"  Ground-truth continuation: {num_gt_tokens} tokens")

        messages = [
            {"role": "user", "content": f"Please write a comprehensive, detailed 200-word continuation expanding on the following text. Do not stop early:\n\n{truncated_text}"}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prefill_len = inputs.input_ids.shape[1]
        
        # Reset Cache (same as before)
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "_clean_cache"):
                layer.self_attn._clean_cache()
            layer.self_attn.kv_cache._prefill_done = False
            layer.self_attn.kv_cache.global_token_counter.zero_()
            layer.self_attn.kv_cache.token_ledger.fill_(-1.0)
            layer.self_attn.kv_cache.global_score_history.fill_(-1.0)

        # === STEP 1: PREFILL — Single forward pass with the full prompt ===
        print("  Running prefill...")
        with torch.no_grad():
            prefill_outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                use_cache=True,
                output_attentions=True
            )
            past_kv = prefill_outputs.past_key_values
            prefill_attentions = prefill_outputs.attentions  # tuple of [bsz, heads, q_len, kv_len] per layer

        # --- Extract Prefill Data ---

        prefill_data = {}
        prefill_window_scores = {}
        
        for layer_idx in TRACKED_LAYERS:
            layer_module = model.model.layers[layer_idx]
            ws = layer_module.self_attn.kv_cache.window_scores.detach().cpu().numpy()
            
            layer_attn = prefill_attentions[layer_idx]
            current_seq_len = layer_attn.shape[-1]
            

            # FIX ISSUE 4: Capture true entire prefill attention
            q_importance = layer_attn[0, :, :current_seq_len, :].sum(dim=1).float()
            kv_importance = q_importance.view(config.NUM_KV_HEADS, GROUP_SIZE, -1).mean(dim=1).cpu().numpy()
            
            # Build output dicts — window scores are already batched in ws[head]
            layer_data = {}
            layer_ws_data = {}
            for kv_head_idx in TRACKED_HEADS:
                layer_data[str(kv_head_idx)] = kv_importance[kv_head_idx].tolist()
                head_ws = ws[kv_head_idx]
                valid_ws = head_ws[~np.isnan(head_ws[:, 1])]  # filter in one line
                layer_ws_data[str(kv_head_idx)] = valid_ws[:, :2].tolist()
            
            prefill_data[str(layer_idx)] = layer_data
            prefill_window_scores[str(layer_idx)] = layer_ws_data

        # === STEP 2: TEACHER-FORCING GENERATION — Feed GT tokens one at a time ===
        print(f"  Running teacher-forcing generation ({num_gt_tokens} steps)...")
        generation_data = []
        generation_window_scores = []
        generation_attention_fresh = []
        generation_window_scores_fresh = []
        generated_token_ids = []
        
        for step in range(num_gt_tokens):
            # Feed the ground-truth token as the next input
            next_token_id = gt_continuation[step].item()
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=model.device)
            
            with torch.no_grad():
                gen_output = model(
                    input_ids=next_token_tensor,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_attentions=True
                )
                past_kv = gen_output.past_key_values
            
            generated_token_ids.append(next_token_id)
            
            # --- Snapshot Ledger & Window Scores (BATCHED) ---
            step_data = {}
            step_ws_data = {}
            for layer_idx in TRACKED_LAYERS:
                layer_module = model.model.layers[layer_idx]
                ledger = layer_module.self_attn.kv_cache.token_ledger
                ws = layer_module.self_attn.kv_cache.window_scores.detach().cpu().numpy()
                total_tokens = int(layer_module.self_attn.kv_cache.global_token_counter.item())
                
                # Batch: compute live mask once, extract all heads' scores at once
                live_mask = (ledger[:total_tokens, 1] >= 0)
                live_indices = torch.where(live_mask)[0].cpu().numpy()
                
                # Batch extract scores for ALL heads: columns [3, 3+1, ..., 3+7]
                all_full_rows = np.zeros((config.NUM_KV_HEADS, total_tokens), dtype=np.float32)
                if len(live_indices) > 0:
                    score_cols = [2 + config.NUM_KV_HEADS + h for h in TRACKED_HEADS]
                    all_live_scores = ledger[live_indices][:, score_cols].cpu().numpy()  # [num_live, 8]
                    all_full_rows[:, live_indices] = all_live_scores.T  # [8, total_tokens]
                
                layer_step_data = {}
                layer_ws_data_inner = {}
                for head_idx in TRACKED_HEADS:
                    layer_step_data[str(head_idx)] = all_full_rows[head_idx].tolist()
                    head_ws = ws[head_idx]
                    valid_ws = head_ws[~np.isnan(head_ws[:, 1])]
                    layer_ws_data_inner[str(head_idx)] = valid_ws[:, :2].tolist()
                
                step_data[str(layer_idx)] = layer_step_data
                step_ws_data[str(layer_idx)] = layer_ws_data_inner
                
            generation_data.append(step_data)
            generation_window_scores.append(step_ws_data)
            
            # --- Fresh Per-Step Attention (non-cumulative, from this step only) ---
            gen_attentions = gen_output.attentions
            step_attn_fresh = {}
            step_ws_fresh = {}
            for layer_idx in TRACKED_LAYERS:
                layer_attn = gen_attentions[layer_idx]  # [1, 32, 1, seq_len]
                per_head = layer_attn[0, :, 0, :].float()  # [32, seq_len]
                per_kv_head = per_head.view(config.NUM_KV_HEADS, GROUP_SIZE, -1).mean(dim=1).cpu().numpy()
                
                # Batch fresh attention: compute window scores for ALL heads at once
                evictable_all = per_kv_head[:, SINK_TOKENS:]  # [8, evictable_len]
                num_complete_windows = evictable_all.shape[1] // OMEGA
                
                if num_complete_windows > 0:
                    windowed = evictable_all[:, :num_complete_windows * OMEGA].reshape(config.NUM_KV_HEADS, num_complete_windows, OMEGA)
                    win_mags_all = windowed.sum(axis=2)  # [8, num_complete_windows]
                    win_ids = np.arange(num_complete_windows, dtype=np.float32)
                
                layer_attn_fresh = {}
                layer_ws_fresh = {}
                for kv_head_idx in TRACKED_HEADS:
                    layer_attn_fresh[str(kv_head_idx)] = per_kv_head[kv_head_idx].tolist()
                    if num_complete_windows > 0:
                        layer_ws_fresh[str(kv_head_idx)] = np.stack([win_mags_all[kv_head_idx], win_ids], axis=1).tolist()
                    else:
                        layer_ws_fresh[str(kv_head_idx)] = []
                
                step_attn_fresh[str(layer_idx)] = layer_attn_fresh
                step_ws_fresh[str(layer_idx)] = layer_ws_fresh
            
            generation_attention_fresh.append(step_attn_fresh)
            generation_window_scores_fresh.append(step_ws_fresh)

        results.append({
            "metadata": {
                "sha256": sample["sha256"],
                "article_index": sample["article_index"],
                "token_count_input": prefill_len,
                "generated_token_count": len(generation_data),
                "generated_token_ids": generated_token_ids,
                "truncation_char_index": truncation_char_index,
                "teacher_forcing": True,
            },
            "tracked_layers": TRACKED_LAYERS,
            "tracked_heads": TRACKED_HEADS,
            "prefill_attention": prefill_data,
            "prefill_window_scores": prefill_window_scores,
            "generation_attention": generation_data,
            "generation_window_scores": generation_window_scores,
            "generation_attention_fresh": generation_attention_fresh,
            "generation_window_scores_fresh": generation_window_scores_fresh,
        })
        
        # Free GPU memory
        del prefill_outputs, prefill_attentions, past_kv, inputs, prefill_data, generation_data, prefill_window_scores, generation_window_scores, generation_attention_fresh, generation_window_scores_fresh
        gc.collect()
        torch.cuda.empty_cache()

    # Save results as compressed NPZ
    save_results_npz(results, OUTPUT_FILE)
        
    print(f"Saved vanilla baseline results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()