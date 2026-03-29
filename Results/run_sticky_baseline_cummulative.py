import torch
from transformers import AutoTokenizer, AutoConfig
from sticky_llama_model import STICKYLlamaForCausalLM
from configuration_sticky_llama import LlamaConfig
import json
import os
from tqdm import tqdm
import numpy as np
import random
from sticky_config import OMEGA, SINK_TOKENS


# --- Configuration ---
MODEL_PATH = "/kaggle/input/llama-3.2/transformers/1b-instruct/1"
VANILLA_RESULTS_PATH = "vanilla_baseline_results.json"
OUTPUT_FILE = "sticky_baseline_results.json"
MAX_NEW_TOKENS = 1024

def main():
    # 1. Load Vanilla Results to get the exact samples
    if not os.path.exists(VANILLA_RESULTS_PATH):
        print(f"Error: {VANILLA_RESULTS_PATH} not found. Run vanilla baseline first.")
        return

    if os.path.exists(OUTPUT_FILE):
        print(f"Removing existing {OUTPUT_FILE} to prevent appending bugs...")
        os.remove(OUTPUT_FILE)

    with open(VANILLA_RESULTS_PATH, "r") as f:
        vanilla_data = json.load(f)

    # 2. Initialize Sticky Model
    print(f"Loading StickyLlama from {MODEL_PATH}...")
    try:
        config = LlamaConfig.from_pretrained(MODEL_PATH)
        
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
             if "rope_type" in config.rope_scaling and "type" not in config.rope_scaling:
                 config.rope_scaling["type"] = config.rope_scaling["rope_type"]
        
        config.rope_theta = getattr(config, "rope_theta", 500000.0)
            
        config.r_ratio = 50
        config.start_idx = 0

        model = STICKYLlamaForCausalLM.from_pretrained(
            MODEL_PATH, 
            config=config, 
            torch_dtype=torch.bfloat16, # Match vanilla baseline dtype
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Model loaded.")

    results = []

    # Map for tracked heads (Attention Head -> KV Head)
    # Llama 3.2 1B has 32 Q heads and 8 KV heads -> Group size 4
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    group_size = num_q_heads // num_kv_heads

    # Extract tracked layers from vanilla data
    tracked_layers = vanilla_data[0]["tracked_layers"]
    
    # Use KV-head indices (0-7) to match eviction granularity
    tracked_heads = list(range(num_kv_heads))  # [0, 1, 2, 3, 4, 5, 6, 7]

    # RELOAD SAMPLES using the same method as vanilla baseline
    # This ensures inputs are identical.
    from pg19_loader import get_pg19_blocks
    # Important: Use same seed as vanilla baseline if loader uses randomness
    # But get_wikitext103_drift_blocks seems deterministic for a given set?
    # run_vanilla_baseline used SEED=42.
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    raw_samples = get_pg19_blocks(MODEL_PATH, num_samples=len(vanilla_data), min_tokens=2560)
    
    # Verify alignment
    if len(raw_samples) != len(vanilla_data):
        print("Warning: Number of loaded samples does not match vanilla results.")
    

    for idx, (raw_sample, v_result) in enumerate(zip(raw_samples, vanilla_data)):
        if raw_sample["sha256"] != v_result["metadata"]["sha256"]:
            print(f"Mismatch at index {idx}! SHA256 inconsistent.")
            continue
            
        if "truncated_text" in v_result["metadata"]:
            prompt = v_result["metadata"]["truncated_text"]
            print(f"Loaded truncated prompt from metadata (len {len(prompt)})")
        else:
            prompt = raw_sample["text"]
            print(f"Warning: truncated_text not found in metadata, using raw sample (len {len(prompt)})")
        
        # --- Get the SAME ground-truth continuation tokens as vanilla ---
        gt_token_ids = v_result["metadata"]["generated_token_ids"]
        num_gt_tokens = len(gt_token_ids)
        
        if num_gt_tokens == 0:
            print(f"  Warning: No GT tokens for sample {idx}. Skipping.")
            continue
        
        print(f"  Teacher-forcing with {num_gt_tokens} GT tokens from vanilla results.")
        
        messages = [
            {"role": "user", "content": f"Please write a comprehensive, detailed 200-word continuation expanding on the following text. Do not stop early:\n\n{prompt}"}
        ]
        
        chat_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        prefill_len = inputs.input_ids.shape[1]
        
        # Reset Cache
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

        # --- Extract Prefill Data ---

        print("  Extracting Prefill data and Window Scores...")
        prefill_data = {}
        prefill_window_scores = {}
        
        for layer_idx in tracked_layers:
            layer_module = model.model.layers[layer_idx]
            prefill_tensor = layer_module.self_attn.kv_cache.prefill_attention_matrix
            ws = layer_module.self_attn.kv_cache.window_scores.detach().cpu().numpy()
            
            if prefill_tensor is None:
                print(f"  Warning: No prefill_attention_matrix for layer {layer_idx}")
                prefill_data[str(layer_idx)] = {str(h): [] for h in tracked_heads}
                prefill_window_scores[str(layer_idx)] = {str(h): [] for h in tracked_heads}
                continue
            
            current_seq_len = prefill_tensor.shape[-1]
            

            # FIX ISSUE 4: Capture true entire prefill attention, aligning bfloat16 fp-commutation perfectly with Vanilla!
            prefill_tensor_view = prefill_tensor[:, :current_seq_len, :].view(num_kv_heads, group_size, current_seq_len, -1)
            scores_for_cache = prefill_tensor_view.mean(dim=1).contiguous()
            kv_importance_val = scores_for_cache.sum(dim=1).float().cpu().numpy()
            
            layer_data = {}
            layer_ws_data = {}
            for kv_head_idx in tracked_heads:
                layer_data[str(kv_head_idx)] = kv_importance_val[kv_head_idx].tolist()
                
                head_ws = ws[kv_head_idx]
                valid_mask = ~np.isnan(head_ws[:, 1])
                valid_ws = head_ws[valid_mask]
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
            next_token_id = gt_token_ids[step]
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
            
            # --- Snapshot Ledger & Window Scores ---
            step_data = {}
            step_ws_data = {}
            for layer_idx in tracked_layers:
                layer_module = model.model.layers[layer_idx]
                ledger = layer_module.self_attn.kv_cache.token_ledger
                ws = layer_module.self_attn.kv_cache.window_scores.detach().cpu().numpy()
                total_tokens = int(layer_module.self_attn.kv_cache.global_token_counter.item())
                
                layer_step_data = {}
                layer_ws_data_inner = {}
                num_kv_heads_cache = layer_module.self_attn.kv_cache.num_heads
                for head_idx in tracked_heads:
                    kv_head_idx = head_idx
                    
                    full_row = np.zeros(total_tokens, dtype=np.float32)
                    # Per-head physical ID column: 2 + head_idx
                    phys_col = 2 + kv_head_idx
                    live_mask = (ledger[:total_tokens, phys_col] >= 0)
                    live_indices = torch.where(live_mask)[0].cpu().numpy()
                    
                    if len(live_indices) > 0:
                        # Per-head score column: 2 + num_heads + head_idx
                        score_col = 2 + num_kv_heads_cache + kv_head_idx
                        live_scores = ledger[live_indices, score_col].cpu().numpy()
                        full_row[live_indices] = live_scores
                    
                    layer_step_data[str(head_idx)] = full_row.tolist()
                    
                    head_ws = ws[kv_head_idx]
                    valid_mask = ~np.isnan(head_ws[:, 1])
                    valid_ws = head_ws[valid_mask]
                    layer_ws_data_inner[str(head_idx)] = valid_ws[:, :2].tolist()
                
                step_data[str(layer_idx)] = layer_step_data
                step_ws_data[str(layer_idx)] = layer_ws_data_inner
                
            generation_data.append(step_data)
            generation_window_scores.append(step_ws_data)
            
            # --- Fresh Per-Step Attention (non-cumulative, from this step only) ---
            gen_attentions = gen_output.attentions
            step_attn_fresh = {}
            step_ws_fresh = {}
            for layer_idx in tracked_layers:
                layer_module = model.model.layers[layer_idx]
                layer_attn = gen_attentions[layer_idx]  # [1, num_q_heads, 1, compressed_seq_len]
                compressed_seq_len = layer_attn.shape[-1]
                per_head = layer_attn[0, :, 0, :].float()  # [num_q_heads, compressed_seq_len]
                per_kv_head = per_head.view(num_kv_heads, group_size, -1).mean(dim=1).cpu().numpy()
                
                ledger_fresh = layer_module.self_attn.kv_cache.token_ledger
                ws_fresh = layer_module.self_attn.kv_cache.window_scores.detach().cpu().numpy()
                total_tokens_fresh = int(layer_module.self_attn.kv_cache.global_token_counter.item())
                
                layer_attn_fresh = {}
                layer_ws_fresh = {}
                for head_idx in tracked_heads:
                    kv_head_idx = head_idx
                    head_attn = per_kv_head[kv_head_idx]  # [compressed_seq_len]
                    
                    # Per-token fresh attention: map compressed positions back to global
                    full_row_fresh = np.zeros(total_tokens_fresh, dtype=np.float32)
                    phys_col = 2 + kv_head_idx
                    live_mask = (ledger_fresh[:total_tokens_fresh, phys_col] >= 0)
                    live_indices = torch.where(live_mask)[0]
                    if len(live_indices) > 0:
                        phys_positions = ledger_fresh[live_indices, phys_col].long().cpu().numpy()
                        valid = phys_positions < compressed_seq_len
                        live_np = live_indices.cpu().numpy()
                        full_row_fresh[live_np[valid]] = head_attn[phys_positions[valid]]
                    
                    layer_attn_fresh[str(head_idx)] = full_row_fresh.tolist()
                    
                    # Per-window fresh magnitude: use window_scores logical IDs
                    head_ws_f = ws_fresh[kv_head_idx]
                    valid_ws_mask = ~np.isnan(head_ws_f[:, 1])
                    valid_ws_f = head_ws_f[valid_ws_mask]
                    
                    window_fresh = []
                    for w_idx in range(len(valid_ws_f)):
                        logical_id = int(valid_ws_f[w_idx, 1])
                        phys_start = SINK_TOKENS + w_idx * OMEGA
                        phys_end = phys_start + OMEGA
                        if phys_end <= compressed_seq_len:
                            win_mag = float(head_attn[phys_start:phys_end].sum())
                            window_fresh.append([win_mag, float(logical_id)])
                    
                    layer_ws_fresh[str(head_idx)] = window_fresh
                
                step_attn_fresh[str(layer_idx)] = layer_attn_fresh
                step_ws_fresh[str(layer_idx)] = layer_ws_fresh
            
            generation_attention_fresh.append(step_attn_fresh)
            generation_window_scores_fresh.append(step_ws_fresh)

        # Decode the full sequence for reference
        full_sequence = torch.cat([inputs.input_ids[0], torch.tensor(generated_token_ids, device=model.device)])
        decoded_output = tokenizer.decode(full_sequence, skip_special_tokens=True)

        result_entry = {
            "metadata": {
                "sha256": v_result["metadata"]["sha256"],
                "article_index": v_result["metadata"]["article_index"],
                "token_count_input": prefill_len,
                "generated_token_count": len(generation_data),
                "generated_token_ids": generated_token_ids,
                "truncated_text": v_result["metadata"]["truncated_text"],
                "teacher_forcing": True,
            },
            "tracked_layers": tracked_layers,
            "tracked_heads": tracked_heads,
            "prefill_attention": prefill_data,
            "prefill_window_scores": prefill_window_scores,
            "generation_attention": generation_data,
            "generation_window_scores": generation_window_scores,
            "generation_attention_fresh": generation_attention_fresh,
            "generation_window_scores_fresh": generation_window_scores_fresh,
            "full_text": decoded_output 
        }
        results.append(result_entry)
        
        # Explicitly Free GPU Memory
        import gc
        del prefill_outputs, past_kv
        del inputs
        del prefill_data
        del generation_data
        del generation_attention_fresh, generation_window_scores_fresh
        gc.collect()
        torch.cuda.empty_cache()

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved sticky baseline results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()