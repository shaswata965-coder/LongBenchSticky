import os
import json
from typing import Dict, Any
from datasets import Dataset

# Standardized Prompts (Official THUDM LongBench Templates)
PROMPT_TEMPLATES = {
    # ── Single/Multi-Document QA ──────────────────────────────────────
    "qa": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "{context}\n\n"
        "Question: {input}\n\n"
        "Answer:"
    ),
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question based on the story.\n\n"
        "Story:\n{context}\n\n"
        "Question:\n{input}\n\n"
        "Answer:"
    ),

    # ── Summarization ─────────────────────────────────────────────────
    "qmsum": (
        "You are a helpful assistant. Read the meeting transcript below and write a detailed summary "
        "specifically answering the following query.\n\n"
        "Meeting Transcript:\n{context}\n\n"
        "Query: {input}\n\n"
        "Summary:"
    ),
    "gov_report": (
        "You are given a report by a government agency. "
        "Write a summary of the report.\n\n"
        "Report:\n{context}\n\n"
        "Summary:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news.\n\n"
        "News:\n{context}\n\n"
        "Now, write a one-page summary of all the news.\n\n"
        "Summary:"
    ),

    # ── Few-shot Learning ─────────────────────────────────────────────
    "few_shot": (
        "{context}\n{input}"
    ),

    # ── Synthetic ─────────────────────────────────────────────────────
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
        "Please carefully read these paragraphs and determine how many unique paragraphs there are "
        "after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n"
        "{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. "
        "The output format should only contain the number, such as 1, 2, 3, and so on.\n\n"
        "The final answer is: "
    ),
    "passage_retrieval": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. "
        "Please determine which paragraph the abstract is from.\n\n"
        "{context}\n\n"
        "The following is an abstract.\n\n"
        "{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. "
        "The answer format must be like \"Paragraph 1\".\n"
    ),

    # ── Code ──────────────────────────────────────────────────────────
    "code": (
        "You are a code completion engine. Continue the code provided below. "
        "Output ONLY the code continuation. "
        "Do NOT output markdown backticks (```). "
        "Do NOT add any explanations or conversational text.\n\n"
        "{input}"
    ),
    "code_completion": (
        "Please complete the code given below.\n\n"
        "{context}\n"
    ),
}

# CONFIG: Context length is now unrestricted for standardized LongBench testing.

def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return Dataset.from_list(records)

def load_datasets(root: str) -> Dict[str, Dataset]:
    names = [
        # Single-Document QA
        "narrativeqa", "qasper", "multifieldqa_en",
        # Multi-Document QA
        "hotpotqa", "2wikimqa", "musique",
        # Summarization
        "gov_report", "qmsum", "multi_news",
        # Few-shot Learning
        "trec", "triviaqa", "samsum",
        # Synthetic
        "passage_count", "passage_retrieval_en",
        # Code
        "lcc", "repobench-p",
    ]
    out = {}
    for n in names:
        p = os.path.join(root, f"{n}.jsonl")
        try:
            ds = load_jsonl(p)
            assert len(ds) > 0
            out[n] = ds
            print(f"✓ Loaded {n} ({len(ds)})")
        except Exception as e:
            print(f"✗ Skipping {n}: {e}")
    if not out:
        raise RuntimeError("No datasets loaded.")
    return out

def build_prompt(example: Dict[str, Any], task: str) -> str:
    """Build the evaluation prompt for a LongBench task using official THUDM templates."""
    # 1. Fetch raw content
    ctx = example.get("context") or example.get("document") or ""
    inp = example.get("input") or example.get("question") or ""

    # ── Single-Document QA ────────────────────────────────────────────
    if task == "narrativeqa":
        return PROMPT_TEMPLATES["narrativeqa"].format(context=ctx, input=inp)

    # ── Single/Multi-Document QA (shared template) ───────────────────
    if task in ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"]:
        return PROMPT_TEMPLATES["qa"].format(context=ctx, input=inp)

    # ── Summarization ─────────────────────────────────────────────────
    if task == "qmsum":
        return PROMPT_TEMPLATES["qmsum"].format(context=ctx, input=inp)

    if task == "gov_report":
        return PROMPT_TEMPLATES["gov_report"].format(context=ctx)

    if task == "multi_news":
        return PROMPT_TEMPLATES["multi_news"].format(context=ctx)

    # ── Few-shot Learning ─────────────────────────────────────────────
    # TREC, TriviaQA, SAMSum: context = few-shot examples, input = query
    if task in ["trec", "triviaqa", "samsum"]:
        return PROMPT_TEMPLATES["few_shot"].format(context=ctx, input=inp)

    # ── Synthetic ─────────────────────────────────────────────────────
    if task == "passage_count":
        return PROMPT_TEMPLATES["passage_count"].format(context=ctx)

    if task == "passage_retrieval_en":
        return PROMPT_TEMPLATES["passage_retrieval"].format(context=ctx, input=inp)

    # ── Code ──────────────────────────────────────────────────────────
    if task == "lcc":
        # LCC stores code in 'context', not 'input'
        code_source = ctx if ctx else inp
        return PROMPT_TEMPLATES["code"].format(input=code_source)

    if task == "repobench-p":
        return PROMPT_TEMPLATES["code_completion"].format(context=ctx)

    raise ValueError(f"Unknown task: {task}")