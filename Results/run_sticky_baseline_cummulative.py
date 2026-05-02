import torch
from transformers import AutoConfig, AutoTokenizer
from sticky_qwen2_model import STICKYQwen2ForCausalLM
from configuration_sticky_qwen2 import StickyQwen2Config
import json
import os
from tqdm import tqdm
import numpy as np
import random
import glob
from sticky_config import OMEGA, SINK_TOKENS, dataset_tracker
from npz_io import save_results_npz, load_results_npz
import sticky_config as config

# --- Configuration ---
VANILLA_RESULTS_PATH = "pure_vanilla_baseline_results.npz"
OUTPUT_FILE = "sticky_baseline_results.npz"

def main():
    # 1. Load Vanilla Results to get the exact samples
    if not os.path.exists(VANILLA_RESULTS_PATH):
        print(f"Error: {VANILLA_RESULTS_PATH} not found. Run vanilla baseline first.")
        return

    for f in glob.glob(OUTPUT_FILE.replace('.npz', '*.npz')):
        print(f"Removing existing {f} to prevent appending bugs...")
        os.remove(f)

    vanilla_data = load_results_npz(VANILLA_RESULTS_PATH, metadata_only=True)

    # 2. Initialize Sticky Model
    print(f"Loading STICKYQwen2 from {config.MODEL_PATH}...")
    try:
        model_config = StickyQwen2Config(**AutoConfig.from_pretrained(config.MODEL_PATH).to_dict())
        
        if hasattr(model_config, "rope_scaling") and model_config.rope_scaling is not None:
             if "rope_type" in model_config.rope_scaling and "type" not in model_config.rope_scaling:
                 model_config.rope_scaling["type"] = model_config.rope_scaling["rope_type"]
            
        model_config.r_ratio = getattr(config, "R_RATIO", 50)
        model_config.start_idx = getattr(config, "S_IDX", 0)
        model_config.use_fast_attention = False

        model = STICKYQwen2ForCausalLM.from_pretrained(
            config.MODEL_PATH, 
            config=model_config, 
            torch_dtype=torch.bfloat16, # Match vanilla baseline dtype
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

    # Map for tracked heads (Attention Head -> KV Head)
    # Qwen2.5-7B: 28 Q heads, 4 KV heads -> group size 7
    num_q_heads = model_config.num_attention_heads
    num_kv_heads = model_config.num_key_value_heads
    group_size = num_q_heads // num_kv_heads

    # Extract tracked layers from vanilla data
    tracked_layers = vanilla_data[0]["tracked_layers"]
    
    # Use KV-head indices (0-7) to match eviction granularity
    tracked_heads = list(range(num_kv_heads))  # [0, 1, 2, 3, 4, 5, 6, 7]

    # RELOAD SAMPLES using the same method as vanilla baseline
    # This ensures inputs are identical.
    import random
    seed = config.SEEDS[0]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if dataset_tracker == 1:
        from pg19_loader import get_pg19_blocks
        raw_samples = get_pg19_blocks(config.MODEL_PATH, num_samples=len(vanilla_data), min_tokens=config.DATASET_MIN_TOKENS)
    else:
        from wiki_text_loader import get_wikitext103_drift_blocks
        raw_samples = get_wikitext103_drift_blocks(config.MODEL_PATH, num_samples=len(vanilla_data), min_tokens=config.DATASET_MIN_TOKENS)
    
    # Verify alignment
    if len(raw_samples) != len(vanilla_data):
        print("Warning: Number of loaded samples does not match vanilla results.")
    

    for idx, (raw_sample, v_result) in enumerate(zip(raw_samples, vanilla_data)):
        if raw_sample["sha256"] != v_result["metadata"]["sha256"]:
            print(f"Mismatch at index {idx}! SHA256 inconsistent.")
            continue
            
        # Reconstruct the exact truncated prompt from the character index
        truncation_char_index = v_result["metadata"]["truncation_char_index"]
        prompt = raw_sample["text"][:truncation_char_index].strip()
        print(f"Reconstructed truncated prompt from char index {truncation_char_index} (len {len(prompt)})")
        
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
                output_attentions=False  # KV cache captures attention internally; returned tuple is unused
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
                raise RuntimeError(
                    f"prefill_attention_matrix is None for layer {layer_idx}. "
                    "This indicates the fast-attention backend (or tracking disabled) was used. "
                    "For cumulative evaluation, ensure model_config.use_fast_attention = False and tracking is enabled."
                )
            
            current_seq_len = prefill_tensor.shape[-1]
            
            # FIX ISSUE 4: Capture true entire prefill attention, aligning bfloat16 fp-commutation perfectly with Vanilla!
            prefill_tensor_view = prefill_tensor[:, :current_seq_len, :].view(num_kv_heads, group_size, current_seq_len, -1)
            scores_for_cache = prefill_tensor_view.mean(dim=1).contiguous()
            kv_importance_val = scores_for_cache.sum(dim=1).float().cpu().numpy()
            
            # --- APPLY PREFILL LEDGER MASK ---
            kv_cache = layer_module.self_attn.kv_cache
            total_tokens = int(kv_cache.global_token_counter.item())
            ledger_np = kv_cache.token_ledger[:total_tokens].detach().cpu().numpy()
            
            layer_data = {}
            layer_ws_data = {}
            for kv_head_idx in tracked_heads:
                phys_col = 2 + kv_head_idx
                head_importance = kv_importance_val[kv_head_idx].copy()
                
                # Zero out indices explicitly evicted by Sticky during prefill grouping
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

        # --- Initialize cumulative score accumulators seeded from prefill ---
        # This ensures generation attention is exported as total score accumulated
        # up until that point (prefill + all gen steps so far), just like Jaccard.
        max_total_tokens = prefill_len + num_gt_tokens
        cumulative_gen_scores = {}
        for layer_idx in tracked_layers:
            cumulative_gen_scores[layer_idx] = np.zeros((num_kv_heads, max_total_tokens), dtype=np.float32)
            # Seed with prefill cumulative scores
            pre_layer = prefill_data.get(str(layer_idx), {})
            for kv_head_idx in tracked_heads:
                pre_head = pre_layer.get(str(kv_head_idx), [])
                if len(pre_head) > 0:
                    pre_arr = np.array(pre_head, dtype=np.float32)
                    cumulative_gen_scores[layer_idx][kv_head_idx, :len(pre_arr)] = pre_arr

        # === STEP 2: TEACHER-FORCING GENERATION — Feed GT tokens one at a time ===
        print(f"  Running teacher-forcing generation ({num_gt_tokens} steps)...")
        generation_data = []
        generation_window_scores = []
        generated_token_ids = []
        
        import time as _time
        _gen_t0 = _time.time()
        
        for step in range(num_gt_tokens):
            next_token_id = gt_token_ids[step]
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=model.device)
            
            # --- CAPTURE PRE-EVICTION LEDGER FOR ACCURATE ATTENTION ALIGNMENT ---
            pre_step_ledgers = {}
            for layer_idx in tracked_layers:
                kv_cache = model.model.layers[layer_idx].self_attn.kv_cache
                tot_toks = int(kv_cache.global_token_counter.item())
                # Copy required to prevent tensor view permutation during compaction
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
            
            for layer_idx in tracked_layers:
                layer_module = model.model.layers[layer_idx]
                kv_cache = layer_module.self_attn.kv_cache
                
                layer_attn = gen_attentions[layer_idx]
                q_importance = layer_attn[0, :, 0, :].float()
                # Group average
                group_size = q_importance.shape[0] // model_config.num_key_value_heads
                kv_importance = q_importance.view(model_config.num_key_value_heads, group_size, -1).mean(dim=1).cpu().numpy()
                
                total_tokens = int(kv_cache.global_token_counter.item())
                num_kv_heads_cache = kv_cache.num_heads
                
                ws_np = kv_cache.window_scores.detach().cpu().numpy()
                pre_ledger_np = pre_step_ledgers[layer_idx]
                
                all_full_rows = np.zeros((num_kv_heads_cache, total_tokens), dtype=np.float32)
                
                layer_step_data = {}
                layer_ws_data_inner = {}
                
                for head_idx in tracked_heads:
                    phys_col = 2 + head_idx
                    
                    # 1. Map all existing tokens natively using PRE-EVICTION coordinates
                    live_mask = pre_ledger_np[:, phys_col] >= 0
                    live_indices = np.where(live_mask)[0]
                    if len(live_indices) > 0:
                        phys_indices = pre_ledger_np[live_indices, phys_col].astype(int)
                        valid_bounds = phys_indices < kv_importance.shape[1]
                        all_full_rows[head_idx, live_indices[valid_bounds]] = kv_importance[head_idx, phys_indices[valid_bounds]]
                    
                    # 2. Map the newly generated token identically to the trailing boundary
                    new_global_id = pre_ledger_np.shape[0]
                    new_phys_idx = kv_importance.shape[1] - 1
                    if new_global_id < total_tokens and new_phys_idx >= 0:
                        all_full_rows[head_idx, new_global_id] = kv_importance[head_idx, new_phys_idx]
                    
                    # 3. Accumulate into cumulative scores and export cumulative
                    cumulative_gen_scores[layer_idx][head_idx, :total_tokens] += all_full_rows[head_idx]
                    # Zero out evicted positions in the cumulative snapshot
                    evicted_mask = pre_ledger_np[:, phys_col] < 0
                    evicted_indices = np.where(evicted_mask)[0]
                    evicted_indices = evicted_indices[evicted_indices < total_tokens]
                    cumulative_snapshot = cumulative_gen_scores[layer_idx][head_idx, :total_tokens].copy()
                    cumulative_snapshot[evicted_indices] = 0.0
                    layer_step_data[str(head_idx)] = cumulative_snapshot
                    
                    # Window scores read post-step cleanly
                    head_ws = ws_np[head_idx]
                    valid_ws_mask = ~np.isnan(head_ws[:, 1])
                    valid_ws = head_ws[valid_ws_mask]
                    layer_ws_data_inner[str(head_idx)] = valid_ws[:, :2].tolist() if len(valid_ws) > 0 else []
                
                step_data[str(layer_idx)] = layer_step_data
                step_ws_data[str(layer_idx)] = layer_ws_data_inner
            
            generation_data.append(step_data)
            generation_window_scores.append(step_ws_data)
            
            if (step + 1) % 50 == 0 or step == 0:
                elapsed = _time.time() - _gen_t0
                rate = (step + 1) / elapsed
                print(f"    Step {step + 1}/{num_gt_tokens} ({rate:.1f} steps/s, elapsed {elapsed:.0f}s)")

        result_entry = {
            "metadata": {
                "sha256": v_result["metadata"]["sha256"],
                "article_index": v_result["metadata"]["article_index"],
                "token_count_input": prefill_len,
                "generated_token_count": len(generation_data),
                "generated_token_ids": generated_token_ids,
                "truncation_char_index": truncation_char_index,
                "teacher_forcing": True,
            },
            "tracked_layers": tracked_layers,
            "tracked_heads": tracked_heads,
            "prefill_attention": prefill_data,
            "prefill_window_scores": prefill_window_scores,
            "generation_attention": generation_data,
            "generation_window_scores": generation_window_scores,
        }
        results.append(result_entry)
        
        # Explicitly Free GPU Memory
        import gc
        del prefill_outputs, past_kv
        del inputs
        del prefill_data
        del generation_data
        gc.collect()
        torch.cuda.empty_cache()

    save_results_npz(results, OUTPUT_FILE)
    
    print(f"Saved sticky baseline results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()