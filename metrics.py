"""
metrics.py — Self-contained evaluation module.

Structure mirrors the official LongBench metrics.py:
  1. Imports
  2. Text normalisation helpers  (EN + ZH)
  3. Primitive scoring functions (one per task type)
  4. Project-level wrappers      (qa_metrics, rouge_metrics, code_sim_score)
  5. Utility helpers             (calculate_ci, clean_code_output)

engine.py should import only this file; utils.py is no longer needed.
"""

# =============================================================================
# 1. Imports
# =============================================================================
import re
import string
import difflib
import numpy as np
from scipy import stats
from collections import Counter
from typing import Dict, List

from fuzzywuzzy import fuzz
from rouge import Rouge

try:
    import jieba
except ImportError:
    jieba = None

try:
    from rouge_score import rouge_scorer as _rouge_scorer
except ImportError:
    _rouge_scorer = None

# =============================================================================
# 2. Text Normalisation  (identical to official LongBench)
# =============================================================================

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    """Lower text and remove punctuation, extra whitespace (Chinese)."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def normalize(text: str) -> List[str]:
    """Return a list of normalised tokens (convenience wrapper)."""
    return normalize_answer(text).split()


# =============================================================================
# 3. Primitive Scoring Functions  (identical to official LongBench)
# =============================================================================

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall    = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction   = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens   = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens   = list(jieba.cut(prediction,   cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens   = [normalize_zh_answer(t) for t in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(t) for t in ground_truth_tokens]
    prediction_tokens   = [t for t in prediction_tokens   if len(t) > 0]
    ground_truth_tokens = [t for t in ground_truth_tokens if len(t) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)


def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    prediction   = " ".join(list(jieba.cut(prediction,   cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    return rouge_score(prediction, ground_truth)


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs.get("all_classes") or []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = 1.0 / len(em_match_list)
    else:
        score = 0.0
    return score


def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def code_sim_score(prediction, ground_truth, **kwargs):
    """
    Code similarity for lcc / repobench-p.
    Matches DefensiveKV: extract first non-comment/non-backtick line, then fuzz.ratio.
    """
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100


# =============================================================================
# 4. Project-Level High-Level Wrappers  (called from engine.py)
# =============================================================================

# Initialise the multi-metric ROUGE scorer once at module load.
_rouge_multi = _rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True) if _rouge_scorer else None


def qa_metrics(pred: str, refs: List[str]) -> Dict[str, float]:
    """
    Standard QA metrics: Exact Match (EM) and best-of-N F1.
    Used for: 2wikimqa, musique, narrativeqa, qasper, hotpotqa, etc.
    """
    pred_norm   = normalize_answer(pred)
    pred_tokens = normalize(pred)
    best_f1 = 0.0
    em = 0.0

    for ref in refs:
        ref_norm   = normalize_answer(ref)
        ref_tokens = normalize(ref)

        # EM: fully normalised comparison (SQuAD / LongBench spec)
        if pred_norm == ref_norm:
            em = 1.0

        common   = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())

        if not pred_tokens or not ref_tokens:
            f1 = 0.0
        else:
            precision = num_same / len(pred_tokens)
            recall    = num_same / len(ref_tokens)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        best_f1 = max(best_f1, f1)

    return {"em": em, "f1": best_f1}


def rouge_metrics(pred: str, ref: str) -> Dict[str, float]:
    """
    Full ROUGE suite (rouge1, rouge2, rougeL) for summarisation.
    Used for: qmsum
    """
    scores = _rouge_multi.score(ref, pred)
    return {k: v.fmeasure for k, v in scores.items()}


# =============================================================================
# 5. Utility Helpers  (formerly in utils.py)
# =============================================================================

def calculate_ci(values: List[float], confidence: float = 0.95) -> float:
    """Half-width of a t-distribution confidence interval."""
    if len(values) < 2:
        return 0.0
    sem = np.std(values, ddof=1) / np.sqrt(len(values))
    return sem * stats.t.ppf((1 + confidence) / 2.0, len(values) - 1)


def clean_code_output(text: str) -> str:
    """
    Adaptive cleaning for model-generated code output.  Handles three formats:

    1. Triple backticks  (```)  — standard Markdown fence
    2. Single backtick   (`)   — lazy Markdown inline wrapper
    3. No delimiters           — plain text with a chatty prefix
    """
    text = text.strip()

    # --- STRATEGY 1: Triple Backticks ---
    if "```" in text:
        start_idx  = text.find("```")
        newline_idx = text.find("\n", start_idx)
        code_start  = newline_idx + 1 if newline_idx != -1 else start_idx + 3
        end_idx     = text.rfind("```")
        if end_idx <= start_idx:          # truncated — no closing fence
            return text[code_start:].strip()
        return text[code_start:end_idx].strip()

    # --- STRATEGY 2: Single Backtick Wrapper ---
    first_backtick = text.find("`")
    if first_backtick != -1 and first_backtick < 50:
        last_backtick = text.rfind("`")
        if last_backtick > first_backtick + 10:
            content = text[first_backtick + 1 : last_backtick].strip()
            if "\n" in content or len(content) > 20:  # genuine block, not inline code
                return content

    # --- STRATEGY 3: Fallback — strip conversational prefixes ---
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(
            ("here is", "sure", "below is", "certainly", "i have", "please", "continuation")
        ):
            continue
        return "\n".join(lines[i:]).strip()

    return text