import torch
from transformers import AutoTokenizer, AutoConfig
from sticky_llama_model import STICKYLlamaForCausalLM
from configuration_sticky_llama import LlamaConfig
import json
import os
from tqdm import tqdm
import numpy as np
import random
import glob
from sticky_config import OMEGA, SINK_TOKENS, dataset_tracker
from npz_io import save_results_npz, load_results_npz


# --- Configuration ---
MODEL_PATH = "/kaggle/input/llama-3.2/transformers/1b-instruct/1"
VANILLA_RESULTS_PATH = "pure_vanilla_baseline_results.npz"
OUTPUT_FILE = "sticky_baseline_results.npz"
MAX_NEW_TOKENS = 1024

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
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    if dataset_tracker == 1:
        from pg19_loader import get_pg19_blocks
        raw_samples = get_pg19_blocks(MODEL_PATH, num_samples=len(vanilla_data), min_tokens=1024)
    else:
        from wiki_text_loader import get_wikitext103_drift_blocks
        raw_samples = get_wikitext103_drift_blocks(MODEL_PATH, num_samples=len(vanilla_data), min_tokens=1024)
    
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
        generated_token_ids = []
        
        import time as _time
        _gen_t0 = _time.time()
        
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
            
            # Cache attention outputs for fresh-attention extraction
            gen_attentions = gen_output.attentions
            
            # --- Snapshot Ledger & Window Scores ---
            # Single bulk GPU→CPU transfer per layer.
            step_data = {}
            step_ws_data = {}
            
            for layer_idx in tracked_layers:
                layer_module = model.model.layers[layer_idx]
                kv_cache = layer_module.self_attn.kv_cache
                total_tokens = int(kv_cache.global_token_counter.item())
                num_kv_heads_cache = kv_cache.num_heads
                
                # === OPTIMIZATION: Single bulk GPU→CPU transfer per layer ===
                # (Previously: 8+ per-head GPU reads for ledger + 2 redundant ws reads)
                ledger_np = kv_cache.token_ledger[:total_tokens].detach().cpu().numpy()
                ws_np = kv_cache.window_scores.detach().cpu().numpy()
                
                # Pre-allocate output arrays for all heads
                all_full_rows = np.zeros((num_kv_heads, total_tokens), dtype=np.float32)
                
                layer_step_data = {}
                layer_ws_data_inner = {}
                
                for head_idx in tracked_heads:
                    phys_col = 2 + head_idx
                    score_col = 2 + num_kv_heads_cache + head_idx
                    
                    # --- Ledger snapshot: CPU-only numpy ops (no GPU sync) ---
                    live_mask = ledger_np[:, phys_col] >= 0
                    live_indices = np.where(live_mask)[0]
                    if len(live_indices) > 0:
                        all_full_rows[head_idx, live_indices] = ledger_np[live_indices, score_col]
                    
                    layer_step_data[str(head_idx)] = all_full_rows[head_idx].copy()
                    
                    # --- Window scores (read once, used for both ledger and fresh) ---
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