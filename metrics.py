from collections import Counter
from typing import Dict, List
import difflib
from rouge_score import rouge_scorer
import utils

# Initialize scorer once
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def qa_metrics(pred: str, refs: List[str]) -> Dict[str, float]:
    """
    Standard QA metrics: Exact Match (EM) and F1 Score.
    Used for: 2wikimqa, musique, narrativeqa, qasper
    """
    pred_tokens = utils.normalize(pred)
    best_f1 = 0.0
    em = 0.0

    for r in refs:
        r_tokens = utils.normalize(r)
        if pred.strip() == r.strip():
            em = 1.0
        common = Counter(pred_tokens) & Counter(r_tokens)
        num_same = sum(common.values())

        if not pred_tokens or not r_tokens:
            f1 = 0.0
        else:
            p = num_same / len(pred_tokens)
            r = num_same / len(r_tokens)
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        best_f1 = max(best_f1, f1)

    return {"em": em, "f1": best_f1}

def rouge_metrics(pred: str, ref: str) -> Dict[str, float]:
    """
    Summarization metrics.
    Used for: qmsum
    """
    scores = rouge.score(ref, pred)
    return {k: v.fmeasure for k, v in scores.items()}

def code_sim_score(pred: str, ref: str) -> float:
    """
    Edit Similarity for code generation.
    Used for: lcc
    Calculates the Levenshtein edit similarity normalized by the length of the longer string.
    """
    # Standard LongBench implementation uses strict character matching without tokenization
    # to capture syntax sensitivity.
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
        
    matcher = difflib.SequenceMatcher(None, pred, ref)
    return matcher.ratio()