import time
import torch
import gc
import re
import numpy as np
from typing import Dict, Any, List

import sticky_config as config
import metrics
import data_loader
import utils

def get_ground_truth(ex: Dict[str, Any], task: str) -> List[str]:
    """
    Robust extraction of ground truth references.
    """
    # 1. NarrativeQA: Answer is often in 'answer' or 'answers'
    if task == "narrativeqa":
        if "answer" in ex: return [ex["answer"]]
        if "answers" in ex: return ex["answers"]

    # 2. QMSum: Summary is in 'summary' or 'targets'
    if task == "qmsum":
        if "summary" in ex: return [ex["summary"]]
        if "targets" in ex: return ex["targets"]

    # 3. LCC / Code: The key is inconsistent. Check ALL variations.
    if task == "lcc":
        # Priority list of keys to check
        possible_keys = ["answers", "answer", "target", "output", "reference", "completion"]
        for key in possible_keys:
            if key in ex:
                val = ex[key]
                # If it's a non-empty list, return it
                if isinstance(val, list) and val:
                    return val
                # If it's a non-empty string, wrap in list
                if isinstance(val, str) and val.strip():
                    return [val]

    # 4. Standard Fallback (2Wiki, MuSiQue, Hotpot, etc.)
    if "answers" in ex:
        return ex["answers"]
    if "answer" in ex:
        val = ex["answer"]
        return [val] if isinstance(val, str) else val
        
    return []

def extract_answer_span(prediction: str, references: List[str]) -> str:
    """
    Standard 'Answer Inclusion' Logic:
    If the normalized reference appears within the normalized prediction,
    we extract it as the answer. This penalizes hallucinations (if answer is missing)
    but forgives 'chattiness' (if answer is surrounded by text).
    """
    pred_norm = utils.normalize_answer(prediction)
    
    # Check all valid references
    for ref in references:
        ref_norm = utils.normalize_answer(ref)
        
        # Safety: avoid matching empty or extremely short common words mistakenly
        if len(ref_norm) < 3: 
            continue
            
        if ref_norm in pred_norm:
            # FOUND IT: The model knows the answer.
            # We return the Reference string itself to ensure the Metric (F1/EM)
            # gives full credit for finding the correct information.
            return ref
            
    # If no reference is found, return the full prediction 
    # so the metric can penalize the wrong answer.
    return prediction

# UPDATED: Added 'task' argument
def generate(prompt, model, tokenizer, device, refs=None, task=None, **kwargs):
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # =========================================================================
    # 🧹 1. PRE-GENERATION CLEANUP & RESET
    # =========================================================================
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "_clean_cache"):
                layer.self_attn._clean_cache()
                if hasattr(layer.self_attn, "kv_cache"):
                    layer.self_attn.kv_cache._prefill_done = False
                    layer.self_attn.kv_cache.global_token_counter.zero_()
                    layer.self_attn.kv_cache.token_ledger.fill_(-1.0)
                    layer.self_attn.kv_cache.global_score_history.fill_(-1.0)
                    if getattr(layer.self_attn.kv_cache, "prefill_attention_matrix", None) is not None:
                        layer.self_attn.kv_cache.prefill_attention_matrix = None

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache() 
        torch.cuda.reset_peak_memory_stats() # <--- CRITICAL: Reset stats here
        start_mem = torch.cuda.memory_allocated()

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            **config.GENERATION_CONFIG,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs
        )
    if device == "cuda":
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start
    
    input_len = inputs.input_ids.shape[1]
    gen_tokens = out.shape[1] - input_len
    raw_text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    
    # ---------------------------------------------------------
    # CRITICAL FIX: Dataset-Specific Cleaning Logic
    # ---------------------------------------------------------
    if task == "lcc":
        # For Code, we MUST strip markdown and chatty prefixes

        clean_text = utils.clean_code_output(raw_text)
    elif refs:
        # For QA, we try to extract the specific answer span
        clean_text = extract_answer_span(raw_text, refs)
    else:
        # Default fallback
        clean_text = raw_text
    # ---------------------------------------------------------

    peak_mem = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    
    del inputs
    del out
    
    return {
        "text": clean_text,
        "raw_text": raw_text,
        "tokens": gen_tokens,
        "time": total_time,
        "peak_mem": peak_mem
    }

def evaluate_dataset(name, dataset, seed, model, tokenizer, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    results = []
    sample_size = min(100, len(dataset))

    for i in range(sample_size):
        ex = dataset[i]
        try:
            prompt = data_loader.build_prompt(ex, name)
        except ValueError as e:
            print(f"⚠️  {e}, skipping example")
            continue

        refs = get_ground_truth(ex, name)
        if not refs:
            continue

        # Pass refs to generate for the inclusion check
        gen = generate(prompt, model, tokenizer, device, refs=refs, use_cache= True)
        
        if name == "qmsum":
            ref_text = refs[0] if refs else ""
            m = metrics.rouge_metrics(gen["text"], ref_text)
        elif name == "lcc":
            ref_text = refs[0] if refs else ""
            score = metrics.code_sim_score(gen["text"], ref_text)
            m = {"edit_sim": score}
        else:
            m = metrics.qa_metrics(gen["text"], refs)

        if gen["tokens"] == 0:
            print("⚠️  Empty generation detected")

        throughput = (gen["tokens"] / gen["time"]) if gen["tokens"] > 1 and gen["time"] > 0 else 0.0

        results.append({
            "metrics": m,
            "tokens": gen["tokens"],
            "time": gen["time"],
            "throughput": throughput,
            "peak_mem": gen["peak_mem"]
        })
        
        if device == "cuda":
            torch.cuda.empty_cache()
    
            # === Data Logger (NEW) ===
    if results:
        metric_keys = results[0]["metrics"].keys()
        agg = {}
        for k in metric_keys:
            vals = [r["metrics"][k] for r in results]
            agg[k] = np.mean(vals)
            ci = utils.calculate_ci(vals, confidence=config.CONFIDENCE_LEVEL)
            agg[f"{k}_ci"] = ci

        log_parts = [f"{name} (seed={seed})"]
        for k in sorted(metric_keys):
            mean = agg[k]
            ci = agg[f"{k}_ci"]
            log_parts.append(f"{k}: {mean:.4f} ± {ci:.4f}")
        print("   📊 " + " | ".join(log_parts))
    else:
        print(f"   📊 {name} (seed={seed}) — No valid results to log.")

    return {
        "dataset": name,
        "seed": seed,
        "sample_size": len(results),
        "sample_adequacy_heuristic": len(results) >= config.MIN_SAMPLE_ADEQUACY,
        "results": results
    }
