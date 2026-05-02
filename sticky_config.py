import os

# Qwen2-7B-Instruct (local directory with config + weights)
MODEL_PATH = "/home/ee/phd/eez228470/Qwen2-7B-Instruct"

# --- STICKY SPECIFIC RATIOS ---
# Env-var overrides let launch_all.sh pass per-job hyperparameters.
R_RATIO = int(os.environ.get("STICKY_R_RATIO", 20))
Q_RATIO = int(os.environ.get("STICKY_Q_RATIO", 10))
OMEGA   = int(os.environ.get("STICKY_OMEGA", 8))

# LOCAL_NUM_TOKENS from env takes priority; otherwise fall back to P_RATIO.
_env_local = os.environ.get("STICKY_LOCAL_NUM_TOKENS")
if _env_local is not None:
    LOCAL_NUM_TOKENS = int(_env_local)
    P_RATIO = None
else:
    P_RATIO = 50
    # LOCAL_NUM_TOKENS = 32

# Quantization bit-width for the evicted (q-cache) tokens.
# 8 → standard INT8 (1 byte/element, 2x compression vs fp16) — backward-compatible default.
# 4 → packed INT4 (0.5 bytes/element, 4x compression vs fp16) — doubles q_windows_count.
QUANTIZATION_BIT_WIDTH = 4

SINK_TOKENS = 4
tracking_flag = 1
dataset_tracker = 1

S_IDX = 0
SEEDS = [42]
MIN_SAMPLE_ADEQUACY = 10
CONFIDENCE_LEVEL = 0.95

# Qwen2-7B-Instruct: max_position_embeddings=32768, rope_theta=1e6, no rope_scaling
MAX_CONTEXT_WARNING_TOKENS = 32768
MAX_POSITION_EMBEDDINGS = 32768
ORIGINAL_MAX_POSITION_EMBEDDINGS = 32768
ROPE_THETA = 1000000.0
ROPE_SCALING_FACTOR = 1.0
ROPE_LOW_FREQ_FACTOR = 1.0
ROPE_HIGH_FREQ_FACTOR = 1.0
DATASET_MIN_TOKENS = 50
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "do_sample": False,
    # NOTE: temperature is ignored by HuggingFace generate() when do_sample=False.
    # Kept here so that switching to do_sample=True gives deterministic (temp=1.0) results.
    "temperature": 1.0,
}

DATA_DIR = "/home/ee/visitor/man_misn.visitor/defensive_kv_new/DefensiveKV/Final_LongBench_Dataset"


# --- EVALUATION SCRIPT CONFIGURATIONS ---
NUM_SAMPLES = 10
LONGBENCH_SAMPLES = int(os.environ.get("STICKY_LONGBENCH_SAMPLES", 500))
TRACKED_LAYERS = list(range(28))
NUM_Q_HEADS = 28
NUM_KV_HEADS = 4