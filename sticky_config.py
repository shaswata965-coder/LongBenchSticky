import torch

MODEL_PATH = "/kaggle/input/llama-3.2/transformers/1b-instruct/1"

# --- STICKY SPECIFIC RATIOS ---
# Adjust these to match the VRAM usage of your quantized setup
R_RATIO = 50  # Total KV cache budget (e.g., 25% of sequence length)

# To use a percentage of the cache for local windows, set P_RATIO (e.g., 50) and comment out LOCAL_NUM_TOKENS
# P_RATIO = 50 # Local/Recent window size as % of total budget

# To use a fixed number of tokens for local windows, set LOCAL_NUM_TOKENS (e.g., 256) and comment out P_RATIO
LOCAL_NUM_TOKENS = 0

OMEGA = 5  # Window size for KV cache grouping
SINK_TOKENS = 0  # Number of permanently protected sink tokens
tracking_flag = 1

S_IDX = 0     # Starting index for window tracking
SEEDS = [42]
MIN_SAMPLE_ADEQUACY = 10
CONFIDENCE_LEVEL = 0.95
MAX_CONTEXT_WARNING_TOKENS = 8192

GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "do_sample": False,
    "temperature": 1.0,
}