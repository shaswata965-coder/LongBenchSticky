
# import torch

MODEL_PATH = "/kaggle/input/llama-3.2/transformers/1b-instruct/1"

# --- STICKY SPECIFIC RATIOS ---
# Adjust these to match the VRAM usage of your quantized setup
R_RATIO = 20  # Total KV cache budget (e.g., 25% of sequence length)

# To use a percentage of the cache for local windows, set P_RATIO (e.g., 50) and comment out LOCAL_NUM_TOKENS
P_RATIO = 50 # Local/Recent window size as % of total budget

# Percentage of total cache budget reserved for int8-quantized evicted tokens.
# This carves out from the window allocation: e.g., if windows would get 25% of the
# budget, setting Q_RATIO=10 means windows get 15% and quantized slots get 10%.
# Int8 provides ~2x compression vs fp16, so the effective q-cache capacity is ~2x q_num.
Q_RATIO = 10  # Set to e.g. 10 for 10% of total budget allocated to quantized evicted tokens

# Quantization bit-width for the evicted (q-cache) tokens.
# 8 → standard INT8 (1 byte/element, 2x compression vs fp16) — backward-compatible default.
# 4 → packed INT4 (0.5 bytes/element, 4x compression vs fp16) — doubles q_windows_count.
QUANTIZATION_BIT_WIDTH = 8

# To use a fixed number of tokens for local windows, set LOCAL_NUM_TOKENS (e.g., 256) and comment out P_RATIO
# LOCAL_NUM_TOKENS = 32

OMEGA = 8  # Window size for KV cache grouping
SINK_TOKENS = 4  # Number of permanently protected sink tokens
tracking_flag = 1
dataset_tracker = 1

S_IDX = 0     # Starting index for window tracking
SEEDS = [42]
MIN_SAMPLE_ADEQUACY = 10
CONFIDENCE_LEVEL = 0.95
MAX_CONTEXT_WARNING_TOKENS = 131072
MAX_POSITION_EMBEDDINGS = 131072
ORIGINAL_MAX_POSITION_EMBEDDINGS = 8192
ROPE_THETA = 500000.0
ROPE_SCALING_FACTOR = 8.0
ROPE_LOW_FREQ_FACTOR = 1.0
ROPE_HIGH_FREQ_FACTOR = 4.0
DATASET_MIN_TOKENS = 50
GENERATION_CONFIG = {
    "max_new_tokens": 100,
    "do_sample": False,
    "temperature": 1.0,
}

DATA_DIR = "/kaggle/input/datasets/shaswatabhattacharya/longbench-12/1LongBenchData"


# --- EVALUATION SCRIPT CONFIGURATIONS ---
NUM_SAMPLES = 10
LONGBENCH_SAMPLES = 20
TRACKED_LAYERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8