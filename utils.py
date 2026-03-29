import numpy as np
from scipy import stats
from typing import List
import string
import re

def calculate_ci(values: List[float], confidence=0.95) -> float:
    if len(values) < 2:
        return 0.0
    mean = np.mean(values)
    sem = np.std(values, ddof=1) / np.sqrt(len(values))
    return sem * stats.t.ppf((1 + confidence) / 2., len(values) - 1)

def normalize_answer(s: str) -> str:
    """Official LongBench/SQuAD normalization logic."""
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

def normalize(text: str) -> List[str]:
    """Wrapper that returns tokens for the QA metric function."""
    return normalize_answer(text).split()

import re

def clean_code_output(text: str) -> str:
    """
    Adaptive cleaning. Handles:
    1. Triple backticks (Standard Markdown)
    2. Single backticks (Lazy Markdown)
    3. No backticks (Just chatty text)
    """
    text = text.strip()
    
    # --- STRATEGY 1: Triple Backticks (```) ---
    # This is the gold standard. Check for it first.
    if "```" in text:
        start_idx = text.find("```")
        # Find the end of the language tag (e.g. ```python\n)
        newline_idx = text.find("\n", start_idx)
        
        # Determine where the actual code starts
        if newline_idx != -1:
            code_start = newline_idx + 1
        else:
            code_start = start_idx + 3
            
        # Find the closing fence (from the right/end)
        end_idx = text.rfind("```")
        
        # If no closing fence (truncated) or same as opening, take to end
        if end_idx <= start_idx:
            return text[code_start:].strip()
        return text[code_start:end_idx].strip()

    # --- STRATEGY 2: Single Backtick Wrappers (`) ---
    # Only if triple failed. Be careful not to match inline code comments.
    # We look for a backtick near the start and a backtick near the end.
    
    # Check if the text *starts* with a backtick (ignoring headers like "Here is code:")
    # Or if there is a backtick on a line by itself or early in the text.
    first_backtick = text.find("`")
    
    if first_backtick != -1:
        # Check if this backtick seems to be a wrapper.
        # It's a wrapper if it's near the start OR preceded by a newline/colon
        is_likely_wrapper = (first_backtick < 50) 
        
        if is_likely_wrapper:
            # Look for the last backtick
            last_backtick = text.rfind("`")
            
            # Ensure they are distinct and span a decent length
            if last_backtick > first_backtick + 10:
                # Extract everything between them
                content = text[first_backtick+1 : last_backtick].strip()
                
                # Double check: Did we just strip `variable_name` inside a sentence?
                # If the result is one word, it was probably just inline code. 
                # If it has newlines, it's definitely the code block.
                if "\n" in content or len(content) > 20:
                    return content

    # --- STRATEGY 3: Fallback (No Delimiters) ---
    # Strip conversational prefixes
    lines = text.split('\n')
    for i, line in enumerate(lines):
        # Skip lines that look like chat
        if line.strip().lower().startswith(("here is", "sure", "below is", "certainly", "i have", "please", "continuation")):
            continue
        # Stop at the first real line of code
        return "\n".join(lines[i:]).strip()
        
    return text