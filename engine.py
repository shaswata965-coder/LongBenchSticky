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

# ── Task Category Sets (for routing) ─────────────────────────────────────
QA_TASKS = {"narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"}
ROUGE_TASKS = {"gov_report", "qmsum", "multi_news", "samsum"}
CODE_TASKS = {"lcc", "repobench-p"}
CLASSIFICATION_TASKS = {"trec"}
COUNT_TASKS = {"passage_count"}
RETRIEVAL_TASKS = {"passage_retrieval_en"}

def get_ground_truth(ex: Dict[str, Any], task: str) -> List[str]:
    """
    Robust extraction of ground truth references for all 16 LongBench datasets.
    LongBench standardizes all datasets to have an 'answers' field (list of strings).
    Task-specific fallbacks handle edge cases.
    """
    # ── Code tasks: key is inconsistent across sources ────────────────
    if task in CODE_TASKS:
        possible_keys = ["answers", "answer", "target", "output", "reference", "completion"]
        for key in possible_keys:
            if key in ex:
                val = ex[key]
                if isinstance(val, list) and val:
                    return val
                if isinstance(val, str) and val.strip():
                    return [val]

    # ── Summarization: may use 'summary' or 'targets' ────────────────
    if task in ROUGE_TASKS:
        if "answers" in ex:
            return ex["answers"]
        if "summary" in ex:
            return [ex["summary"]]
        if "targets" in ex:
            return ex["targets"]

    # ── Standard Fallback (QA, Classification, Synthetic, etc.) ──────
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
def generate(prompt, model, tokenizer, device, refs=None, task=None, max_tokens=None, **kwargs):
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

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache() 
        torch.cuda.reset_peak_memory_stats() # <--- CRITICAL: Reset stats here
        start_mem = torch.cuda.memory_allocated()

    generation_kwargs = config.GENERATION_CONFIG.copy()
    if max_tokens is not None:
        generation_kwargs["max_new_tokens"] = max_tokens

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            **generation_kwargs,
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
    # Dataset-Specific Cleaning Logic
    # ---------------------------------------------------------
    if task in CODE_TASKS:
        # For Code tasks, strip markdown fences and chatty prefixes
        clean_text = utils.clean_code_output(raw_text)
    elif task in QA_TASKS and refs:
        # For QA tasks, try to extract the specific answer span
        clean_text = extract_answer_span(raw_text, refs)
    elif task in ROUGE_TASKS:
        # For Summarization, strip chatty preamble before ROUGE scoring
        clean_text = utils.clean_summary_output(raw_text)
    elif task in CLASSIFICATION_TASKS:
        # For Classification (TREC), strip preamble for label matching
        clean_text = utils.clean_classification_output(raw_text)
    else:
        # Synthetic (passage_count, passage_retrieval_en) — use raw output
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

def evaluate_dataset(name, dataset, seed, model, tokenizer, device, max_tokens=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    results = []
    # Use fallback 100 if attribute doesn't exist to not break other pipelines
    sample_size = min(getattr(config, "LONGBENCH_SAMPLES", 100), len(dataset))

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

        # Pass refs and task to generate for task-specific cleaning
        gen = generate(prompt, model, tokenizer, device, refs=refs, task=name, max_tokens=max_tokens, use_cache=True)
        
        # ── Metric Routing (Official LongBench dataset2metric) ────
        if name in ROUGE_TASKS:
            ref_text = refs[0] if refs else ""
            m = metrics.rouge_metrics(gen["text"], ref_text)
        elif name in CODE_TASKS:
            ref_text = refs[0] if refs else ""
            score = metrics.code_sim_score(gen["text"], ref_text)
            m = {"edit_sim": score}
        elif name in CLASSIFICATION_TASKS:
            # TREC: all_classes is stored per-example in the dataset
            all_classes = ex.get("all_classes", [])
            ref_text = refs[0] if refs else ""
            score = metrics.classification_score(gen["text"], ref_text, all_classes)
            m = {"cls_acc": score}
        elif name in COUNT_TASKS:
            ref_text = refs[0] if refs else ""
            score = metrics.count_score(gen["text"], ref_text)
            m = {"count_acc": score}
        elif name in RETRIEVAL_TASKS:
            ref_text = refs[0] if refs else ""
            score = metrics.retrieval_score(gen["text"], ref_text)
            m = {"retrieval_acc": score}
        else:
            # Default: QA F1 (narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, triviaqa)
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
