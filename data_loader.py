import os
import json
from typing import Dict, Any
from datasets import Dataset

# Standardized Prompts
PROMPT_TEMPLATES = {
    "qa": (
        "Answer the question directly on the context below. Do NOT explain your reasoning. "
        "Do NOT say 'According to the text'. Keep the answer under 10 words.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n\n"
        "Answer:"
    ),
    "summary": (
        "You are a helpful assistant. Summarize the following text efficiently.\n\n"
        "Text:\n{input}\n\n"
        "Summary:"
    ),
    "qmsum": (
        "You are a helpful assistant. Read the meeting transcript below and write a detailed summary "
        "specifically answering the following query.\n\n"
        "Meeting Transcript:\n{context}\n\n"
        "Query: {input}\n\n"
        "Summary:"
    ),
    "code": (
        "You are a code completion engine. Continue the code provided below. "
        "Output ONLY the code continuation. "
        "Do NOT output markdown backticks (```). "
        "Do NOT add any explanations or conversational text.\n\n"
        "{input}"
    )
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
    names = ["2wikimqa", "qasper", "qmsum", "musique", "multifieldqa_en", "lcc"]
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
    # 1. Fetch raw content
    ctx = example.get("context") or example.get("document") or ""
    inp = example.get("input") or example.get("question") or ""


    # 3. Apply Templates
    if task in ["2wikimqa", "musique", "multifieldqa_en", "qasper", "hotpotqa"]:
        return PROMPT_TEMPLATES["qa"].format(context=ctx, input=inp)
    
    if task == "qmsum":
        # FIX: Pass BOTH context (transcript) and input (query)
        return PROMPT_TEMPLATES["qmsum"].format(context=ctx, input=inp)
        
    if task == "lcc":
        # CRITICAL FIX: LCC stores the code in 'context', not 'input'.
        # We must use 'ctx' as the source for the prompt.
        code_source = ctx if ctx else inp
        
            
        return PROMPT_TEMPLATES["code"].format(input=code_source)

    raise ValueError(f"Unknown task: {task}")