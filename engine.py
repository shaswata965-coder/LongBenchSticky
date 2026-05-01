import time
import sticky_config as config
import torch
import numpy as np
from typing import Dict, Any, List


import metrics
import data_loader

# =============================================================================
# dataset2metric dispatch table  (identical to official LongBench eval.py)
# Routes every task name to its canonical scoring function.
# =============================================================================
dataset2metric = {
    "narrativeqa":          metrics.qa_f1_score,
    "qasper":               metrics.qa_f1_score,
    "multifieldqa_en":      metrics.qa_f1_score,
    "multifieldqa_zh":      metrics.qa_f1_zh_score,
    "hotpotqa":             metrics.qa_f1_score,
    "2wikimqa":             metrics.qa_f1_score,
    "musique":              metrics.qa_f1_score,
    "dureader":             metrics.rouge_zh_score,
    "gov_report":           metrics.rouge_score,
    "qmsum":                metrics.rouge_score,
    "multi_news":           metrics.rouge_score,
    "vcsum":                metrics.rouge_zh_score,
    "trec":                 metrics.classification_score,
    "triviaqa":             metrics.qa_f1_score,
    "samsum":               metrics.rouge_score,
    "lsht":                 metrics.classification_score,
    "passage_retrieval_en": metrics.retrieval_score,
    "passage_count":        metrics.count_score,
    "passage_retrieval_zh": metrics.retrieval_zh_score,
    "lcc":                  metrics.code_sim_score,
    "repobench-p":          metrics.code_sim_score,
}

# Tasks where the model tends to generate multi-line chat; score only line 0.
_FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum", "lsht"}

# Tasks that need raw completion (no chat template, no BOS token) -- matches DefensiveKV
_RAW_COMPLETION_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}



# =============================================================================
# Ground-truth extraction
# =============================================================================

def get_ground_truth(ex: Dict[str, Any], task: str) -> List[str]:
    """
    Robust extraction of ground-truth references from a dataset example.
    Handles the inconsistent key conventions across LongBench sub-tasks.
    """
    # NarrativeQA: answer key varies
    if task == "narrativeqa":
        if "answer"  in ex: return [ex["answer"]]
        if "answers" in ex: return ex["answers"]

    # QMSum: summary lives under 'summary' or 'targets'
    if task == "qmsum":
        if "summary" in ex: return [ex["summary"]]
        if "targets"  in ex: return ex["targets"]

    # LCC / Code: reference key is inconsistent across dataset versions
    if task == "lcc":
        for key in ("answers", "answer", "target", "output", "reference", "completion"):
            if key in ex:
                val = ex[key]
                if isinstance(val, list) and val:            return val
                if isinstance(val, str)  and val.strip():    return [val]

    # Standard fallback (2wikimqa, musique, hotpotqa, …)
    if "answers" in ex:
        return ex["answers"]
    if "answer" in ex:
        val = ex["answer"]
        return [val] if isinstance(val, str) else val

    return []


def get_all_classes(ex: Dict[str, Any]) -> Any:
    """Return the 'all_classes' field used by classification tasks, or None."""
    return ex.get("all_classes", None)




# =============================================================================
# score_example  (official scorer() logic, per-example)
# =============================================================================

def score_example(dataset: str, prediction: str,
                  ground_truths: List[str], all_classes: Any) -> float:
    """
    Score a single (prediction, ground_truths) pair using the official
    LongBench dispatch pattern:
      • Apply first-line truncation for chat-heavy tasks.
      • Take the max score over all ground-truth references.
    Returns a raw score in [0, 1].
    """
    if dataset not in dataset2metric:
        raise ValueError(f"Unknown dataset '{dataset}' — not in dataset2metric.")

    # Official truncation for tasks that generate multi-line chat replies
    if dataset in _FIRST_LINE_TASKS:
        prediction = prediction.lstrip("\n").split("\n")[0]

    score = 0.0
    for ground_truth in ground_truths:
        score = max(
            score,
            dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes),
        )
    return score


# =============================================================================
# scorer_e  (official scorer_e() logic — LongBench-E length-stratified scoring)
# =============================================================================

def scorer_e(dataset: str, predictions: List[str], answers: List[List[str]],
             lengths: List[int], all_classes: Any) -> Dict[str, float]:
    """
    Length-stratified scorer for LongBench-E (mirrors eval.py scorer_e exactly).

    Buckets each example by its input length and returns the mean score × 100
    for each bucket:
        "0-4k"  — inputs shorter than 4 000 tokens
        "4-8k"  — inputs between 4 000 and 8 000 tokens
        "8k+"   — inputs longer than 8 000 tokens

    Args:
        dataset:     Task name key into dataset2metric.
        predictions: List of model prediction strings.
        answers:     List of ground-truth lists (one list per example).
        lengths:     List of input token-lengths (one per example).
        all_classes: Class list used by classification tasks (or None).

    Returns:
        Dict with keys "0-4k", "4-8k", "8k+" mapped to rounded scores (0–100).
    """
    if dataset not in dataset2metric:
        raise ValueError(f"Unknown dataset '{dataset}' — not in dataset2metric.")

    scores: Dict[str, list] = {"0-4k": [], "4-8k": [], "8k+": []}

    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0

        # Official truncation for chat-heavy tasks
        if dataset in _FIRST_LINE_TASKS:
            prediction = prediction.lstrip("\n").split("\n")[0]

        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes),
            )

        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)

    # Average each bucket and scale to 0–100 (empty buckets → 0.0)
    return {
        key: round(100 * np.mean(vals), 2) if vals else 0.0
        for key, vals in scores.items()
    }


# =============================================================================
# Generation
# =============================================================================

def generate(prompt, model, tokenizer, device, refs=None, task=None, **kwargs):
    """
    Run one forward pass through the model and return the decoded text plus
    performance counters (token count, wall-clock time, peak VRAM).
    No output post-processing -- raw decoded text is returned directly,
    matching DefensiveKV's pipeline.
    """
    if task in _RAW_COMPLETION_TASKS:
        # FIX (M1): Use add_special_tokens=False instead of mutating bos_token.
        # Setting bos_token="" has no effect on PreTrainedTokenizerFast (Llama 3)
        # which controls BOS prepending via the add_bos_token attribute.
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    else:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # =========================================================================
    # 🧹 Pre-generation cache cleanup & reset
    # =========================================================================
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "_clean_cache"):
                layer.self_attn._clean_cache()

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()   # CRITICAL: reset before timing

    gen_kwargs = {**config.GENERATION_CONFIG}
    gen_kwargs.update(kwargs)
    gen_kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)

    # DefensiveKV-style: stop samsum generation at newline (single-line summary)
    if task == "samsum":
        nl_token_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id, nl_token_id]

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            **gen_kwargs
        )
    if device == "cuda":
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start

    input_len  = inputs.input_ids.shape[1]
    gen_tokens = out.shape[1] - input_len
    raw_text   = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

    clean_text = raw_text

    peak_mem = torch.cuda.max_memory_allocated() if device == "cuda" else 0

    del inputs
    del out

    return {
        "text":     clean_text,
        "raw_text": raw_text,
        "tokens":   gen_tokens,
        "time":     total_time,
        "peak_mem": peak_mem,
    }


# =============================================================================
# evaluate_dataset  (main evaluation loop)
# =============================================================================

def evaluate_dataset(name, dataset, seed, model, tokenizer, device):
    """
    Run the full evaluation loop for one LongBench sub-task.

    Scoring follows the official eval.py pattern:
      • dataset2metric dispatch
      • max-over-references
      • first-line truncation for chat-heavy tasks
      • all_classes passed as kwarg for classification tasks

    Additional project-specific features (all preserved):
      • Seeded RNG for reproducibility
      • Per-example throughput & peak-VRAM tracking
      • CI-width logging at the end
      • sample_adequacy_heuristic flag in the returned dict
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    results = []
    sample_size = min(config.LONGBENCH_SAMPLES, len(dataset))
    print(f"Running {sample_size} samples from {config.LONGBENCH_SAMPLES} configured (dataset size: {len(dataset)})")


    for i in range(sample_size):
        ex = dataset[i]

        try:
            prompt = data_loader.build_prompt(ex, name)
        except ValueError as e:
            print(f"⚠️  {e}, skipping example")
            continue

        refs        = get_ground_truth(ex, name)
        all_classes = get_all_classes(ex)

        if not refs:
            continue

        # Generate — passes task name so post-processing is applied correctly
        task_max_tokens = data_loader.max_new_tokens.get(name, 128)
        gen = generate(prompt, model, tokenizer, device, refs=refs, task=name, use_cache=True, max_new_tokens=task_max_tokens)

        if gen["tokens"] == 0:
            print("⚠️  Empty generation detected")

        # -----------------------------------------------------------------
        # Score with the official dispatch pattern (mirrors eval.py scorer)
        # -----------------------------------------------------------------
        try:
            raw_score = score_example(name, gen["text"], refs, all_classes)
        except ValueError as e:
            print(f"⚠️  Scoring error: {e}, skipping example")
            continue

        # Derive the metric-key name from the scoring function for logging
        scorer_fn  = dataset2metric.get(name)
        metric_key = getattr(scorer_fn, "__name__", "score")
        m = {metric_key: raw_score}
        # -----------------------------------------------------------------

        throughput = (
            gen["tokens"] / gen["time"]
            if gen["tokens"] >= 1 and gen["time"] > 0
            else 0.0
        )

        results.append({
            "metrics":    m,
            "tokens":     gen["tokens"],
            "time":       gen["time"],
            "throughput": throughput,
            "peak_mem":   gen["peak_mem"],
        })

        if device == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Aggregate logging with confidence intervals
    # -------------------------------------------------------------------------
    if results:
        metric_keys = results[0]["metrics"].keys()
        agg = {}
        for k in metric_keys:
            vals    = [r["metrics"][k] for r in results]
            agg[k]  = np.mean(vals)
            ci      = metrics.calculate_ci(vals, confidence=config.CONFIDENCE_LEVEL)
            agg[f"{k}_ci"] = ci

        log_parts = [f"{name} (seed={seed})"]
        for k in sorted(metric_keys):
            log_parts.append(f"{k}: {agg[k]:.4f} ± {agg[f'{k}_ci']:.4f}")
        print("   📊 " + " | ".join(log_parts))
    else:
        print(f"   📊 {name} (seed={seed}) — No valid results to log.")

    return {
        "dataset":                  name,
        "seed":                     seed,
        "sample_size":              len(results),
        "sample_adequacy_heuristic": len(results) >= config.MIN_SAMPLE_ADEQUACY,
        "results":                  results,
    }