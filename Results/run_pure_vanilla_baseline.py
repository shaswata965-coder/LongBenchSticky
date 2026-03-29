import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import random
import numpy as np
import os
import gc
from pg19_loader import get_pg19_blocks
from sticky_config import OMEGA, SINK_TOKENS

# --- Configuration ---
MODEL_PATH = "/kaggle/input/llama-3.2/transformers/1b-instruct/1"
NUM_SAMPLES = 5
MAX_NEW_TOKENS = 128
SEED = 42
OUTPUT_FILE = "pure_vanilla_baseline_results.json"

# We hardcode these so they are consistent across runs if we re-run this script
TRACKED_LAYERS = [1, 3, 5, 8, 10, 11, 14, 15] # 8 random layers from 0-15

# KV-head indices (0-7) — matches the granularity used by sticky eviction
# Llama 3.2 1B: 32 Q-heads, 8 KV-heads, group_size=4
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS
TRACKED_HEADS = list(range(NUM_KV_HEADS))  # [0, 1, 2, 3, 4, 5, 6, 7]

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    setup_seed(SEED)
    
    if os.path.exists(OUTPUT_FILE):
        print(f"Removing existing {OUTPUT_FILE} to prevent appending bugs...")
        os.remove(OUTPUT_FILE)
        
    STICKY_OUTPUT_FILE = "sticky_baseline_results.json"
    if os.path.exists(STICKY_OUTPUT_FILE):
        print(f"Removing existing {STICKY_OUTPUT_FILE} to force a synchronized regeneration run...")
        os.remove(STICKY_OUTPUT_FILE)
    
    print(f"Loading Pure HuggingFace LLaMA (No sticky cache logic) from {MODEL_PATH}...")
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig as HFLlamaConfig
        with open(os.path.join(MODEL_PATH, "config.json"), "r") as f:
            v_config_dict = json.load(f)
        rope_scaling_config = v_config_dict.get("rope_scaling", None)
        if "rope_scaling" in v_config_dict:
            del v_config_dict["rope_scaling"]
        v_config = HFLlamaConfig(**v_config_dict)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, 
            config=v_config,
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
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
                orig_max_pos = rope_scaling_config.get("original_max_position_embeddings", 8192)
                for layer in model.model.layers:
                    layer.self_attn.rotary_emb = Llama3RotaryEmbedding(
                        dim=dim, max_position_embeddings=max_pos, base=base,
                        scaling_factor=factor, low_freq_factor=low_freq,
                        high_freq_factor=high_freq, original_max_position_embeddings=orig_max_pos
                    ).to(device=model.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Model loaded. Tracking {len(TRACKED_LAYERS)} layers and {len(TRACKED_HEADS)} heads.")

    # Load samples using PG-19 loader
    samples = get_pg19_blocks(MODEL_PATH, num_samples=NUM_SAMPLES, min_tokens=2560)
    
    results = []

    for idx, sample in enumerate(samples):
        text = sample["text"]
        
        # We want the max tokens received (prefill) to be <= 512.
        # We start with ~380 words to leave room for the prompt template.
        words = text.split()
        truncated_text = " ".join(words[:380])
        
        while True:
            messages = [{"role": "user", "content": f"Please write a comprehensive, detailed 200-word continuation expanding on the following text. Do not stop early:\n\n{truncated_text}"}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            if inputs.input_ids.shape[1] <= 512:
                break
            
            # Back off by a few words until we fit under 512 tokens
            words = words[:-5]
            truncated_text = " ".join(words)
            
        remaining_text = text[len(truncated_text):].strip()
        
        print(f"Processing sample {idx + 1}/{len(samples)} (truncated={len(truncated_text)} chars, remaining={len(remaining_text)} chars)...")
        
        gt_tokens = tokenizer(remaining_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        num_gt_tokens = min(MAX_NEW_TOKENS, len(gt_tokens))
        
        if num_gt_tokens == 0:
            print(f"  Warning: No remaining text for sample {idx}. Skipping.")
            continue
            
        gt_continuation = gt_tokens[:num_gt_tokens]  # [num_gt_tokens]
        print(f"  Ground-truth continuation: {num_gt_tokens} tokens")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prefill_len = inputs.input_ids.shape[1]
        
        max_seq_len = prefill_len + num_gt_tokens
        
        # Track cumulative attention per token per layer/head
        token_ledger_scores = {layer: np.zeros((NUM_KV_HEADS, max_seq_len), dtype=np.float32) for layer in TRACKED_LAYERS}

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
            layer_attn = prefill_attentions[layer_idx] # [1, 32, prefill_len, prefill_len]
            current_seq_len = prefill_len
            
            # Cumulative: Match sticky exactly (Mean across Q-heads first)
            layer_attn_view = layer_attn[0, :, :current_seq_len, :].view(NUM_KV_HEADS, GROUP_SIZE, current_seq_len, -1)
            
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
            win_scores = obs_sum.view(NUM_KV_HEADS, num_windows, OMEGA).sum(dim=2).float().cpu().numpy() # [8, num_windows]
            
            layer_data = {}
            layer_ws_data = {}
            for kv_head_idx in TRACKED_HEADS:
                layer_data[str(kv_head_idx)] = kv_importance[kv_head_idx].tolist()
                
                head_ws = []
                for w in range(num_windows):
                    w_score = win_scores[kv_head_idx, w]
                    head_ws.append([float(w_score), float(w)])  # Format: [Score, ID] matching cumulative schema
                
                layer_ws_data[str(kv_head_idx)] = head_ws
            
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
            
            # --- Snapshot Ledger & Window Scores ---
            step_data = {}
            step_ws_data = {}
            for layer_idx in TRACKED_LAYERS:
                layer_attn = gen_attentions[layer_idx] # [1, 32, 1, current_seq_len]
                
                # Squeeze the single query dim and sum over it (it's essentially a sum of 1)
                q_importance = layer_attn[0, :, 0, :].float() # [32, current_seq_len]
                kv_importance = q_importance.view(NUM_KV_HEADS, GROUP_SIZE, -1).mean(dim=1).cpu().numpy() # [8, current_seq_len]
                
                # Accumulate! The essence of Cumulative vanilla metric tracking
                token_ledger_scores[layer_idx][:, :current_seq_len] += kv_importance
                
                num_windows = max(0, (current_seq_len - SINK_TOKENS) // OMEGA)
                
                layer_step_data = {}
                layer_ws_data_inner = {}
                for head_idx in TRACKED_HEADS:
                    kv_head_idx = head_idx
                    
                    # For generation step data, we just export the full tracked cumulative token scores up to current_seq_len
                    full_row = token_ledger_scores[layer_idx][kv_head_idx, :current_seq_len]
                    layer_step_data[str(head_idx)] = full_row.tolist()
                    
                    head_ws = []
                    for w in range(num_windows):
                        w_start = SINK_TOKENS + w * OMEGA
                        w_end = w_start + OMEGA
                        w_score = np.sum(token_ledger_scores[layer_idx][kv_head_idx, w_start:w_end])
                        head_ws.append([float(w_score), float(w)])  # Format: [Score, ID] matching cumulative schema
                    
                    layer_ws_data_inner[str(head_idx)] = head_ws
                
                step_data[str(layer_idx)] = layer_step_data
                step_ws_data[str(layer_idx)] = layer_ws_data_inner
                
            generation_data.append(step_data)
            generation_window_scores.append(step_ws_data)

        # Decode the full sequence for reference
        full_sequence = torch.cat([inputs.input_ids[0], torch.tensor(generated_token_ids, device=model.device)])
        decoded_output = tokenizer.decode(full_sequence, skip_special_tokens=True)

        results.append({
            "metadata": {
                "sha256": sample["sha256"],
                "article_index": sample["article_index"],
                "token_count_input": prefill_len,
                "generated_token_count": len(generation_data),
                "generated_token_ids": generated_token_ids,
                "truncated_text": truncated_text,
                "teacher_forcing": True,
            },
            "tracked_layers": TRACKED_LAYERS,
            "tracked_heads": TRACKED_HEADS,
            "prefill_attention": prefill_data,
            "prefill_window_scores": prefill_window_scores,
            "generation_attention": generation_data,
            "generation_window_scores": generation_window_scores,
            "full_text": decoded_output
        })
        
        # Free GPU memory
        del prefill_outputs, prefill_attentions, past_kv, inputs, prefill_data, generation_data, prefill_window_scores, generation_window_scores
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved pure vanilla baseline results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
