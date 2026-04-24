import torch
from transformers import AutoTokenizer, AutoConfig
import os
import sys
import gc
import numpy as np
import time as _time

from sticky_llama_model import STICKYLlamaForCausalLM
from configuration_sticky_llama import LlamaConfig
import sticky_config as config
from npz_io import save_results_npz

def main():
    p_ratio = getattr(config, 'P_RATIO', 'None')
    r_ratio = getattr(config, 'R_RATIO', 'None')
    local_tokens = getattr(config, 'LOCAL_NUM_TOKENS', 'None')
    
    output_dir = f"p_{p_ratio}_rd_{r_ratio}_local_{local_tokens}"
    os.makedirs(output_dir, exist_ok=True)
    
    for maxTokens in [200,400,600,800,1000]:
        OUTPUT_FILE = os.path.join(output_dir, f"sticky_mt{maxTokens}.npz")
        if os.path.exists(OUTPUT_FILE):
            print(f"Removing existing {OUTPUT_FILE} to prevent appending bugs...")
            os.remove(OUTPUT_FILE)

        # 1. Initialize Sticky Model
        print(f"Loading StickyLlama from {config.MODEL_PATH}...")
        try:
            model_config = LlamaConfig.from_pretrained(config.MODEL_PATH)
            if hasattr(model_config, "rope_scaling") and model_config.rope_scaling is not None:
                if "rope_type" in model_config.rope_scaling and "type" not in model_config.rope_scaling:
                    model_config.rope_scaling["type"] = model_config.rope_scaling["rope_type"]
            model_config.rope_theta = getattr(model_config, "rope_theta", 500000.0)
            model_config.r_ratio = getattr(config, "R_RATIO", 50)
            
            # Add local / P_RATIO settings if they exist natively in the file
            if hasattr(config, "P_RATIO"):
                model_config.p_ratio = config.P_RATIO
            elif hasattr(config, "LOCAL_NUM_TOKENS"):
                model_config.local_num_tokens = config.LOCAL_NUM_TOKENS
                
            model_config.start_idx = getattr(config, "S_IDX", 0)

            model = STICKYLlamaForCausalLM.from_pretrained(
                config.MODEL_PATH, 
                config=model_config,
                max_new_tokens=maxTokens, 
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        print("Model loaded.")

        results = []

        num_q_heads = model_config.num_attention_heads
        num_kv_heads = model_config.num_key_value_heads
        group_size = num_q_heads // num_kv_heads

        tracked_layers = config.TRACKED_LAYERS
        tracked_heads = list(range(num_kv_heads))

        import random
        seed = config.SEEDS[0]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if config.dataset_tracker == 1:
            from pg19_loader import get_pg19_blocks
            raw_samples = get_pg19_blocks(config.MODEL_PATH, num_samples=config.NUM_SAMPLES, min_tokens=config.DATASET_MIN_TOKENS)
        else:
            from wiki_text_loader import get_wikitext103_drift_blocks
            raw_samples = get_wikitext103_drift_blocks(config.MODEL_PATH, num_samples=config.NUM_SAMPLES, min_tokens=config.DATASET_MIN_TOKENS)
        
        for idx, raw_sample in enumerate(raw_samples):
            print(f"\n--- Processing Sample {idx+1}/{len(raw_samples)} ---")
            prompt = raw_sample["text"].strip()
            
            messages = [
                {"role": "user", "content": f"Provide a detailed continuation of the following text. Your response must be approximately 200 words long.\n\nText:\n{prompt}"}
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

            # === STEP 1: PREFILL ===
            print(f"  Running prefill ({prefill_len} tokens) ...")
            with torch.no_grad():
                prefill_outputs = model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    use_cache=True,
                    output_attentions=False
                )
                past_kv = prefill_outputs.past_key_values

            print("  Extracting Prefill data and Window Scores...")
            prefill_data = {}
            prefill_window_scores = {}
            
            for layer_idx in tracked_layers:
                layer_module = model.model.layers[layer_idx]
                prefill_tensor = layer_module.self_attn.kv_cache.prefill_attention_matrix
                ws = layer_module.self_attn.kv_cache.window_scores.detach().cpu().numpy()
                
                if prefill_tensor is None:
                    prefill_data[str(layer_idx)] = {str(h): [] for h in tracked_heads}
                    prefill_window_scores[str(layer_idx)] = {str(h): [] for h in tracked_heads}
                    continue
                
                current_seq_len = prefill_tensor.shape[-1]
                prefill_tensor_view = prefill_tensor[:, :current_seq_len, :].view(num_kv_heads, group_size, current_seq_len, -1)
                scores_for_cache = prefill_tensor_view.mean(dim=1).contiguous()
                kv_importance_val = scores_for_cache.sum(dim=1).float().cpu().numpy()
                
                kv_cache = layer_module.self_attn.kv_cache
                total_tokens = int(kv_cache.global_token_counter.item())
                ledger_np = kv_cache.token_ledger[:total_tokens].detach().cpu().numpy()
                
                layer_data = {}
                layer_ws_data = {}
                for kv_head_idx in tracked_heads:
                    phys_col = 2 + kv_head_idx
                    head_importance = kv_importance_val[kv_head_idx].copy()
                    
                    live_mask = ledger_np[:, phys_col] >= 0
                    dead_mask = ~live_mask
                    valid_dead = np.where(dead_mask)[0]
                    valid_dead = valid_dead[valid_dead < current_seq_len]
                    head_importance[valid_dead] = 0.0
                    
                    layer_data[str(kv_head_idx)] = head_importance.tolist()
                    
                    head_ws = ws[kv_head_idx]
                    valid_mask = ~np.isnan(head_ws[:, 1])
                    valid_ws = head_ws[valid_mask]
                    layer_ws_data[str(kv_head_idx)] = valid_ws[:, :2].tolist()
                
                prefill_data[str(layer_idx)] = layer_data
                prefill_window_scores[str(layer_idx)] = layer_ws_data

            # Initialize generation arrays for tracking
            # We explicitly use the `maxTokens` loop variable so each iteration tests the new length
            max_new_tokens = maxTokens
            do_sample = config.GENERATION_CONFIG.get("do_sample", False)
            temperature = config.GENERATION_CONFIG.get("temperature", 1.0)
            
            max_total_tokens = prefill_len + max_new_tokens
            cumulative_gen_scores = {}
            for layer_idx in tracked_layers:
                cumulative_gen_scores[layer_idx] = np.zeros((num_kv_heads, max_total_tokens), dtype=np.float32)
                pre_layer = prefill_data.get(str(layer_idx), {})
                for kv_head_idx in tracked_heads:
                    pre_head = pre_layer.get(str(kv_head_idx), [])
                    if len(pre_head) > 0:
                        pre_arr = np.array(pre_head, dtype=np.float32)
                        cumulative_gen_scores[layer_idx][kv_head_idx, :len(pre_arr)] = pre_arr

            # Determine the first generated token from the prefill logits!
            first_token_logits = prefill_outputs.logits[0, -1, :]
            if do_sample:
                first_token_logits = first_token_logits / temperature
                probs = torch.nn.functional.softmax(first_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token_id = torch.argmax(first_token_logits, dim=-1).item()

            # === STEP 2: OPEN-ENDED GENERATION ===
            print(f"  Running open-ended autoregressive generation (Max steps: {max_new_tokens})...")
            generation_data = []
            generation_window_scores = []
            generated_token_ids = []
            
            _gen_t0 = _time.time()
            actual_steps = 0
            
            for step in range(max_new_tokens):
                if next_token_id == tokenizer.eos_token_id:
                    print(f"    Early stopping at step {step} due to EOS token.")
                    break
                    
                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=model.device)
                
                # --- CAPTURE PRE-EVICTION LEDGER FOR ACCURATE ATTENTION ALIGNMENT ---
                pre_step_ledgers = {}
                for layer_idx in tracked_layers:
                    kv_cache = model.model.layers[layer_idx].self_attn.kv_cache
                    tot_toks = int(kv_cache.global_token_counter.item())
                    pre_step_ledgers[layer_idx] = kv_cache.token_ledger[:tot_toks].detach().cpu().numpy().copy()
                
                with torch.no_grad():
                    gen_output = model(
                        input_ids=next_token_tensor,
                        past_key_values=past_kv,
                        use_cache=True,
                        output_attentions=True
                    )
                    past_kv = gen_output.past_key_values
                
                generated_token_ids.append(next_token_id)
                gen_attentions = gen_output.attentions
                
                step_data = {}
                step_ws_data = {}
                
                # Only take memory-heavy snapshots at OMEGA intervals to save RAM and disk size
                is_save_step = (step % config.OMEGA == 0) or (step == max_new_tokens - 1)
                
                for layer_idx in tracked_layers:
                    layer_module = model.model.layers[layer_idx]
                    kv_cache = layer_module.self_attn.kv_cache
                    
                    layer_attn = gen_attentions[layer_idx]
                    q_importance = layer_attn[0, :, 0, :].float()
                    
                    # Group average for GQA mapping
                    num_kv_heads_cache = kv_cache.num_heads
                    group_size_map = q_importance.shape[0] // num_kv_heads_cache
                    kv_importance = q_importance.view(num_kv_heads_cache, group_size_map, -1).mean(dim=1).cpu().numpy()
                    
                    total_tokens = int(kv_cache.global_token_counter.item())
                    ws_np = kv_cache.window_scores.detach().cpu().numpy()
                    pre_ledger_np = pre_step_ledgers[layer_idx]
                    
                    all_full_rows = np.zeros((num_kv_heads_cache, total_tokens), dtype=np.float32)
                    
                    layer_step_data = {}
                    layer_ws_data_inner = {}
                    
                    for head_idx in tracked_heads:
                        phys_col = 2 + head_idx
                        
                        # 1. Map existing tokens using PRE-EVICTION physical storage IDs
                        live_mask = pre_ledger_np[:, phys_col] >= 0
                        live_indices = np.where(live_mask)[0]
                        if len(live_indices) > 0:
                            phys_indices = pre_ledger_np[live_indices, phys_col].astype(int)
                            valid_bounds = phys_indices < kv_importance.shape[1]
                            all_full_rows[head_idx, live_indices[valid_bounds]] = kv_importance[head_idx, phys_indices[valid_bounds]]
                        
                        # 2. Map the newly generated open-ended token
                        new_global_id = pre_ledger_np.shape[0]
                        new_phys_idx = kv_importance.shape[1] - 1
                        if new_global_id < total_tokens and new_phys_idx >= 0:
                            all_full_rows[head_idx, new_global_id] = kv_importance[head_idx, new_phys_idx]
                        
                        # 3. Accumulate scores into cumulative array EVERY step
                        cumulative_gen_scores[layer_idx][head_idx, :total_tokens] += all_full_rows[head_idx]
                        
                        # 4. Only take the snapshot of the array if it's a save step
                        if is_save_step:
                            evicted_mask = pre_ledger_np[:, phys_col] < 0
                            evicted_indices = np.where(evicted_mask)[0]
                            evicted_indices = evicted_indices[evicted_indices < total_tokens]
                            
                            cumulative_snapshot = cumulative_gen_scores[layer_idx][head_idx, :total_tokens].copy()
                            cumulative_snapshot[evicted_indices] = 0.0
                            layer_step_data[str(head_idx)] = cumulative_snapshot
                            
                            head_ws = ws_np[head_idx]
                            valid_ws_mask = ~np.isnan(head_ws[:, 1])
                            valid_ws = head_ws[valid_ws_mask]
                            layer_ws_data_inner[str(head_idx)] = valid_ws[:, :2].tolist() if len(valid_ws) > 0 else []
                    
                    if is_save_step:
                        step_data[str(layer_idx)] = layer_step_data
                        step_ws_data[str(layer_idx)] = layer_ws_data_inner
                
                generation_data.append(step_data)
                generation_window_scores.append(step_ws_data)
                actual_steps += 1
                
                if (step + 1) % 25 == 0 or step == 0:
                    elapsed = _time.time() - _gen_t0
                    rate = (step + 1) / elapsed
                    print(f"    Step {step + 1}/{max_new_tokens} ({rate:.1f} steps/s)")

                # Predict next token for autoregressive forward step
                step_logits = gen_output.logits[0, -1, :]
                if do_sample:
                    step_logits = step_logits / temperature
                    probs = torch.nn.functional.softmax(step_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token_id = torch.argmax(step_logits, dim=-1).item()

            result_entry = {
                "metadata": {
                    "sha256": raw_sample["sha256"],
                    "article_index": raw_sample["article_index"],
                    "token_count_input": prefill_len,
                    "generated_token_count": actual_steps,
                    "generated_token_ids": generated_token_ids,
                    "truncation_char_index": len(prompt),
                    "teacher_forcing": False,
                },
                "tracked_layers": tracked_layers,
                "tracked_heads": tracked_heads,
                "prefill_attention": prefill_data,
                "prefill_window_scores": prefill_window_scores,
                "generation_attention": generation_data,
                "generation_window_scores": generation_window_scores,
            }
            results.append(result_entry)
            
            # Explicitly Free Memory
            del prefill_outputs, past_kv
            del inputs
            del prefill_data
            del generation_data
            gc.collect()
            torch.cuda.empty_cache()

        save_results_npz(results, OUTPUT_FILE)
        print(f"\nSaved Open-Ended Sticky baseline results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
