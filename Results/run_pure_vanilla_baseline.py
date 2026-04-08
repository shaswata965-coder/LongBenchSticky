import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random
import numpy as np
import os
import gc
import glob
from sticky_config import OMEGA, SINK_TOKENS, dataset_tracker
from npz_io import save_results_npz
import sticky_config as config

# --- Configuration ---
OUTPUT_FILE = "pure_vanilla_baseline_results.npz"

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
        
    STICKY_OUTPUT_FILE = "sticky_baseline_results.npz"
    for f in glob.glob(STICKY_OUTPUT_FILE.replace('.npz', '*.npz')):
        print(f"Removing existing {f} to force a synchronized regeneration run...")
        os.remove(f)
    
    print(f"Loading Pure HuggingFace LLaMA (No sticky cache logic) from {config.MODEL_PATH}...")
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig as HFLlamaConfig
        with open(os.path.join(config.MODEL_PATH, "config.json"), "r") as f:
            v_config_dict = json.load(f)
        rope_scaling_config = v_config_dict.get("rope_scaling", None)
        if "rope_scaling" in v_config_dict:
            del v_config_dict["rope_scaling"]
        v_config = HFLlamaConfig(**v_config_dict)

        model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_PATH, 
            config=v_config,
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
        
        # --- MONKEY PATCH LLAMA 3 ROPE ---
        if rope_scaling_config is not None:
            rope_type = rope_scaling_config.get("type", rope_scaling_config.get("rope_type", ""))
            if rope_type == "llama3":
                print("Monkey-patching HuggingFace 4.35 model with custom Llama 3 RoPE...")
                from sticky_llama_attention import Llama3RotaryEmbedding
                dim = v_config.hidden_size // v_config.num_attention_heads
                max_pos = v_config.max_position_embeddings
                base = getattr(v_config, "rope_theta", 500000.0)
                factor = rope_scaling_config.get("factor", 8.0)
                low_freq = rope_scaling_config.get("low_freq_factor", 1.0)
                high_freq = rope_scaling_config.get("high_freq_factor", 4.0)
                orig_max_pos = rope_scaling_config.get("original_max_position_embeddings", config.ORIGINAL_MAX_POSITION_EMBEDDINGS)
                for layer in model.model.layers:
                    layer.self_attn.rotary_emb = Llama3RotaryEmbedding(
                        dim=dim, max_position_embeddings=max_pos, base=base,
                        scaling_factor=factor, low_freq_factor=low_freq,
                        high_freq_factor=high_freq, original_max_position_embeddings=orig_max_pos
                    ).to(device=layer.self_attn.q_proj.weight.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Model loaded. Tracking {len(config.TRACKED_LAYERS)} layers and {len(TRACKED_HEADS)} heads.")

    # Load samples — min_tokens=2560 ensures enough remaining for 512-token generation
    # after 0.5-0.7 truncation (worst case: 2560*0.3 = 768 tokens remaining)
    # Note: Cannot go higher due to O(N²) attention memory with output_attentions=True
    if dataset_tracker == 1:
        from pg19_loader import get_pg19_blocks
        samples = get_pg19_blocks(config.MODEL_PATH, num_samples=config.NUM_SAMPLES, min_tokens=config.DATASET_MIN_TOKENS)
    else:
        from wiki_text_loader import get_wikitext103_drift_blocks
        samples = get_wikitext103_drift_blocks(config.MODEL_PATH, num_samples=config.NUM_SAMPLES, min_tokens=config.DATASET_MIN_TOKENS)
    
    results = []

    for idx, sample in enumerate(samples):
        text = sample["text"]
        
        # --- Unified truncation: 0.5-0.7 random cutoff with period-snap ---
        truncate_pct = random.uniform(0.5, 0.7)
        target_len = int(len(text) * truncate_pct)
        
        # Snap to nearest period for grammatical completeness
        cut_idx = text.rfind('.', 0, target_len)
        if cut_idx == -1: cut_idx = text.rfind(' ', 0, target_len)
        if cut_idx == -1: cut_idx = target_len
        
        # Keep the period so the block is fully grammatical
        truncation_char_index = cut_idx + 1
        truncated_text = text[:truncation_char_index].strip()
        remaining_text = text[truncation_char_index:].strip()
        
        print(f"Processing sample {idx + 1}/{len(samples)} (original={len(text)}, truncated={len(truncated_text)} chars, remaining={len(remaining_text)} chars)...")
        
        gt_tokens = tokenizer(remaining_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        num_gt_tokens = min(config.GENERATION_CONFIG.get("max_new_tokens", 512), len(gt_tokens))
        
        if num_gt_tokens == 0:
            print(f"  Warning: No remaining text for sample {idx}. Skipping.")
            continue
            
        gt_continuation = gt_tokens[:num_gt_tokens]  # [num_gt_tokens]
        print(f"  Ground-truth continuation: {num_gt_tokens} tokens")
        
        messages = [{"role": "user", "content": f"Please write a comprehensive, detailed 200-word continuation expanding on the following text. Do not stop early:\n\n{truncated_text}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prefill_len = inputs.input_ids.shape[1]
        
        max_seq_len = prefill_len + num_gt_tokens
        
        # Track cumulative attention per token per layer/head
        token_ledger_scores = {layer: np.zeros((config.NUM_KV_HEADS, max_seq_len), dtype=np.float32) for layer in config.TRACKED_LAYERS}

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
        
        for layer_idx in config.TRACKED_LAYERS:
            layer_attn = prefill_attentions[layer_idx] # [1, 32, prefill_len, prefill_len]
            current_seq_len = prefill_len
            
            # Cumulative: Match sticky exactly (Mean across Q-heads first)
            layer_attn_view = layer_attn[0, :, :current_seq_len, :].view(config.NUM_KV_HEADS, GROUP_SIZE, current_seq_len, -1)
            
            # 1. Compute mean across Q-heads -> [8, prefill_len, prefill_len]
            # Must enforce contiguous layout so bfloat16 summation vectorization utilizes equivalent SIMD buffers!
            scores_for_cache = layer_attn_view.mean(dim=1).contiguous()
            
            # To achieve 100% parity with Sticky's internal mechanism, we must sum the 5 window tokens in bfloat16 BEFORE float casting!
            # And we MUST ALSO slice the tensor before calling .sum(dim=1) so the PyTorch kernel uses identical vectorization strides!
            
            # Full sum for cumulative ledger tracking (float32)
            kv_importance_bf16_full = scores_for_cache.sum(dim=1) # [8, prefill_len]
            kv_importance = kv_importance_bf16_full.float().cpu().numpy() # [8, prefill_len]
            
            # Add to running cumulative sum (float32 for generation parity)
            token_ledger_scores[layer_idx][:, :current_seq_len] += kv_importance
            
            # Compute Window Scores for JSON
            num_windows = max(0, (current_seq_len - SINK_TOKENS) // OMEGA)
            actual_review_end = SINK_TOKENS + num_windows * OMEGA
            
            # --- STRICT PARITY SLICE ---
            # Sticky slices the actual_review_end BEFORE summing queries.
            scores_slice = scores_for_cache[:, :, SINK_TOKENS:actual_review_end] # [8, seq_len, num_windows * omega]
            obs_sum = scores_slice.sum(dim=1) # [8, num_windows * omega] 
            win_scores = obs_sum.view(config.NUM_KV_HEADS, num_windows, OMEGA).sum(dim=2).float().cpu().numpy() # [8, num_windows]
            
            # Build output dicts from batched arrays (no per-window Python loop)
            layer_data = {}
            layer_ws_data = {}
            win_ids = np.arange(num_windows, dtype=np.float32)
            for kv_head_idx in TRACKED_HEADS:
                layer_data[str(kv_head_idx)] = kv_importance[kv_head_idx].tolist()
                if num_windows > 0:
                    layer_ws_data[str(kv_head_idx)] = np.stack([win_scores[kv_head_idx], win_ids], axis=1).tolist()
                else:
                    layer_ws_data[str(kv_head_idx)] = []
            
            prefill_data[str(layer_idx)] = layer_data
            prefill_window_scores[str(layer_idx)] = layer_ws_data

        # === STEP 2: TEACHER-FORCING GENERATION — Feed GT tokens one at a time ===
        print(f"  Running teacher-forcing generation ({num_gt_tokens} steps)...")
        generation_data = []
        generation_window_scores = []
        generated_token_ids = []
        
        for step in range(num_gt_tokens):
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
                gen_attentions = gen_output.attentions
            
            generated_token_ids.append(next_token_id)
            current_seq_len = prefill_len + step + 1
            
            # --- Snapshot Ledger & Window Scores (BATCHED across all heads) ---
            step_data = {}
            step_ws_data = {}
            for layer_idx in config.TRACKED_LAYERS:
                layer_attn = gen_attentions[layer_idx] # [1, 32, 1, current_seq_len]
                
                # Vectorized: compute KV importance for ALL heads at once
                q_importance = layer_attn[0, :, 0, :].float() # [32, current_seq_len]
                kv_importance = q_importance.view(config.NUM_KV_HEADS, GROUP_SIZE, -1).mean(dim=1).cpu().numpy() # [8, current_seq_len]
                
                # Accumulate cumulative scores
                token_ledger_scores[layer_idx][:, :current_seq_len] += kv_importance
                
                num_windows = max(0, (current_seq_len - SINK_TOKENS) // OMEGA)
                
                # Batch export: slice all heads at once -> [8, current_seq_len]
                all_scores = token_ledger_scores[layer_idx][:, :current_seq_len]
                
                # Batch window scores: reshape evictable region into windows and sum
                if num_windows > 0:
                    evictable_end = SINK_TOKENS + num_windows * OMEGA
                    evictable = all_scores[:, SINK_TOKENS:evictable_end]  # [8, num_windows * OMEGA]
                    win_scores = evictable.reshape(config.NUM_KV_HEADS, num_windows, OMEGA).sum(axis=2)  # [8, num_windows]
                    win_ids = np.arange(num_windows, dtype=np.float32)  # [num_windows]
                else:
                    win_scores = np.zeros((config.NUM_KV_HEADS, 0), dtype=np.float32)
                    win_ids = np.array([], dtype=np.float32)
                
                # Build dicts from batched arrays (no per-window Python loop)
                layer_step_data = {}
                layer_ws_data_inner = {}
                for head_idx in TRACKED_HEADS:
                    layer_step_data[str(head_idx)] = token_ledger_scores[layer_idx][head_idx, :current_seq_len].tolist()
                    if num_windows > 0:
                        layer_ws_data_inner[str(head_idx)] = np.stack([win_scores[head_idx], win_ids], axis=1).tolist()
                    else:
                        layer_ws_data_inner[str(head_idx)] = []
                
                step_data[str(layer_idx)] = layer_step_data
                step_ws_data[str(layer_idx)] = layer_ws_data_inner
                
            generation_data.append(step_data)
            generation_window_scores.append(step_ws_data)

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
            "tracked_layers": config.TRACKED_LAYERS,
            "tracked_heads": TRACKED_HEADS,
            "prefill_attention": prefill_data,
            "prefill_window_scores": prefill_window_scores,
            "generation_attention": generation_data,
            "generation_window_scores": generation_window_scores,
        })
        
        # Free GPU memory
        del prefill_outputs, prefill_attentions, past_kv, inputs, prefill_data, generation_data, prefill_window_scores, generation_window_scores
        gc.collect()
        torch.cuda.empty_cache()

    # Save results as compressed NPZ
    save_results_npz(results, OUTPUT_FILE)
        
    print(f"Saved pure vanilla baseline results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
