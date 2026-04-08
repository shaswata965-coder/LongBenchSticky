<div align="center">
  <h1>🧠 LongBench Sticky KV Cache</h1>
  <p><i>An advanced evaluation and inference framework for long-context Large Language Models using Cumulative Sticky Attention Eviction with custom CUDA kernel acceleration.</i></p>

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
  [![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
  [![Transformers](https://img.shields.io/badge/HuggingFace-Transformers_4.35.2-yellow.svg)](https://huggingface.co/)
  [![FlashAttention](https://img.shields.io/badge/FlashAttention-2.0-orange.svg)](https://github.com/Dao-AILab/flash-attention)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

---

## 📖 Overview

As Large Language Models (LLMs) scale to handle massive context windows (128K+ tokens), the Key-Value (KV) cache becomes a major memory and compute bottleneck. **LongBench Sticky KV** is an experimental evaluation repository that implements an advanced KV-cache eviction strategy: **Cumulative Sticky Attention**.

Instead of naively discarding old tokens or keeping the full sequence in memory, the "Sticky KV" algorithm retains the most historically important tokens by maintaining a running ledger of attention scores over multiple observation windows. This ensures high performance on long-context benchmarks while drastically reducing the KV-cache memory footprint.

**Key innovation**: The eviction pipeline is accelerated by **custom CUDA kernels** (modeled after [DefensiveKV](https://github.com/FFY0/DefensiveKV)) that fuse per-head KV cache gathering, window score reduction, and vote accumulation into optimized GPU operations — delivering significant speedups on Ampere+ hardware (A6000, A100).

This repository runs standardized configurations of the [LongBench](https://github.com/THUDM/LongBench) suite, comparing unadulterated baseline models against the Sticky KV optimized versions.

---

## ✨ Key Features

- **Cumulative Sticky Attention Cache**: A dynamic KV cache eviction mechanism that preserves crucial context using rolling window-based attention score accumulation without causing CUDA OOMs.
- **Custom CUDA Kernel Acceleration**: Three purpose-built CUDA kernels (`per_head_kv_gather`, `window_score_reduce`, `vote_accumulate`) that accelerate the eviction hotpath, with automatic fallback to pure PyTorch.
- **Granular Token Ledger**: Meticulously tracks attention scores across windows, globally across the context sequence.
- **Layer Information Retention (LIR) Metrics**: Custom metric pipelines to quantitatively analyze the retention of important tokens layer-by-layer.
- **Attention Jaccard Similarity**: Determines the overlap and fidelity of the Sticky KV cache compared against the pure, uncompressed Vanilla baseline.
- **Flash Attention 2.0 Integration**: A native, OOM-safe `fast_attention` variant utilizing `flash_attn` to accelerate prefill stages on Ampere+ hardware.
- **Unrestricted Context Evaluations**: Capable of processing raw LongBench datasets with zero mid-truncation or chunking for pure, standardized benchmarking.

---

## 🗄️ Supported Datasets

The evaluation suite seamlessly supports subsets of the LongBench and PG-19 datasets, categorized by task:

| Category | Datasets |
|---|---|
| **Single-Document QA** | `qasper`, `multifieldqa_en` |
| **Multi-Document QA** | `2wikimqa`, `musique` |
| **Summarization** | `qmsum` |
| **Code Completion** | `lcc` |
| **Language Modeling** | `PG-19` |

---

## 🏗️ Architecture

### Project Structure

```
StickyLLmCummulative/
├── csrc/                                  # Custom CUDA kernels
│   ├── csrc/
│   │   ├── sticky_kv_kernels.cu          # 3 CUDA kernels + C++ launchers
│   │   └── static_switch.h              # fp16/bf16/fp32 dispatch macros
│   ├── include/
│   │   └── sticky_kv_api.h              # Debug & safety check macros
│   ├── build.py                          # Build script (sm_80/sm_86)
│   ├── Makefile                          # Build shortcut
│   └── test_kernels.py                   # Kernel unit tests
├── Metrices/
│   ├── calculate_layer_information_retention.py   # LIR metric computation
│   ├── calculate_window_jaccard.py                # Jaccard similarity
│   ├── visualize_attention_similarity.py          # Attention heatmaps
│   ├── visualize_lir.py                           # LIR visualizations
│   └── visualize_per_head_jaccard_divergence.py   # Per-head analysis
├── Results/
│   ├── run_longbench_sticky.py           # LongBench evaluation (Sticky)
│   ├── run_longbench_vanilla.py          # LongBench evaluation (Vanilla)
│   ├── run_sticky_baseline_cummulative.py # Sticky baseline runner
│   ├── run_pure_vanilla_baseline.py       # Pure vanilla runner
│   └── npz_io.py                          # NPZ data I/O utilities
├── sticky_config.py                       # Central configuration
├── sticky_kv_cuda_ops.py                  # Python wrapper for CUDA kernels
├── sticky_kv_logic_fast_attention.py      # Fast attention KV cache logic
├── sticky_kv_logic_cummulative.py         # Cumulative KV cache logic (research)
├── sticky_llama_attention_fast_attention.py # Flash Attention attention module
├── sticky_llama_attention.py              # Standard attention module
├── sticky_llama_model.py                  # Custom LlamaForCausalLM wrapper
├── configuration_sticky_llama.py          # Custom config class
├── engine.py                              # Core evaluation driver
├── data_loader.py                         # LongBench dataset parser
├── main.py                                # Main entry point
├── metrics.py                             # F1/Rouge/Edit metrics
├── utils.py                               # Utility functions
└── requirements.txt                       # Python dependencies
```

### Component Diagram

```
┌───────────────────────────────────────────────────────────────────┐
│  Runner Scripts (Results/)                                         │
│  run_longbench_sticky.py / run_sticky_baseline_cummulative.py      │
└───────────────────────┬───────────────────────────────────────────┘
                        │ imports
┌───────────────────────▼───────────────────────────────────────────┐
│  STICKYLlamaForCausalLM (sticky_llama_model.py)                    │
│  └─ Replaces each layer's attention with STICKYLlamaAttention      │
└───────────────────────┬───────────────────────────────────────────┘
                        │ imports
┌───────────────────────▼───────────────────────────────────────────┐
│  STICKYLlamaAttention (sticky_llama_attention_fast_attention.py)    │
│  └─ Flash Attention 2 for output, chunked scoring for eviction     │
└───────────────────────┬───────────────────────────────────────────┘
                        │ calls
┌───────────────────────▼───────────────────────────────────────────┐
│  STICKYKVCache_LayerWise (sticky_kv_logic_fast_attention.py)       │
│  └─ Window scoring, vote accumulation, physical eviction           │
│     └─ Uses CUDA-accelerated ops via sticky_kv_cuda_ops.py         │
└───────────────────────┬───────────────────────────────────────────┘
                        │ imports
┌───────────────────────▼───────────────────────────────────────────┐
│  sticky_kv_cuda_ops.py (Python wrapper)                            │
│  ├─ per_head_kv_gather()    → fused K+V cache gathering            │
│  ├─ window_score_reduce()   → fused view + sum                     │
│  └─ vote_accumulate()       → in-place score accumulation          │
│     └─ Auto-fallback to PyTorch if CUDA extension unavailable      │
└───────────────────────┬───────────────────────────────────────────┘
                        │ C++ binding
┌───────────────────────▼───────────────────────────────────────────┐
│  sticky_kv_cuda (csrc/csrc/sticky_kv_kernels.cu)                   │
│  ├─ per_head_kv_gather_kernel   (Grid: heads × thread_groups)      │
│  ├─ window_score_reduce_kernel  (Grid: heads × windows)            │
│  └─ vote_accumulate_kernel      (Grid: heads × positions)          │
└───────────────────────────────────────────────────────────────────┘
```

---

## ⚡ CUDA Kernel Acceleration

### Overview

The eviction pipeline's GPU-bound hotspots are accelerated by 3 custom CUDA kernels, inspired by the [DefensiveKV](https://github.com/FFY0/DefensiveKV) (ICLR'26) architecture. These kernels target the **cache manipulation infrastructure** around attention, not attention computation itself (which is handled by Flash Attention 2).

### Kernel 1: `per_head_kv_gather_kernel`

**Biggest performance win.** Fuses two separate `torch.gather` calls (for K and V) into a single kernel launch, eliminating the intermediate expanded index tensor.

```cuda
// Each thread copies one element across both K and V caches
template <typename tensor_t>
__global__ void per_head_kv_gather_kernel(
    tensor_t *key_out, tensor_t *value_out,
    const tensor_t *key_cache, const tensor_t *value_cache,
    const int *indices, const int seq_len,
    const int max_kept, const int head_dim) {

  const int head_idx = blockIdx.x;
  const int tid = threadIdx.x + blockIdx.y * blockDim.x;
  const int total_elements = max_kept * head_dim;

  for (int i = tid; i < total_elements; i += blockDim.x * gridDim.y) {
    int src_pos = indices[head_idx * max_kept + i / head_dim];
    src_pos = max(0, min(src_pos, seq_len - 1));  // bounds check

    int src_off = head_idx * seq_len * head_dim + src_pos * head_dim + i % head_dim;
    int dst_off = head_idx * max_kept * head_dim + i;

    key_out[dst_off]   = key_cache[src_off];
    value_out[dst_off] = value_cache[src_off];
  }
}
```

**Replaces this Python code:**
```python
# Before (2 kernel launches + index tensor allocation)
gather_idx = torch.clamp(indices, 0, seq_len-1).unsqueeze(-1).expand(-1, -1, head_dim)
k_kept = torch.gather(past_key_values[0][0], 1, gather_idx)
v_kept = torch.gather(past_key_values[1][0], 1, gather_idx)

# After (1 fused kernel launch)
k_kept, v_kept = per_head_kv_gather(past_key_values[0][0], past_key_values[1][0], indices)
```

### Kernel 2: `window_score_reduce_kernel`

Fuses the `view + sum` pattern for computing per-window attention scores. Each thread handles one (head, window) pair and sums `omega` consecutive elements.

```cuda
__global__ void window_score_reduce_kernel(
    float *scores_out, const float *scores_in,
    const int num_tokens, const int num_windows, const int omega) {

  const int head_idx = blockIdx.x;
  const int win_idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (win_idx >= num_windows) return;

  float sum = 0.0f;
  for (int t = 0; t < omega; t++) {
    int pos = win_idx * omega + t;
    if (pos < num_tokens) sum += scores_in[head_idx * num_tokens + pos];
  }
  scores_out[head_idx * num_windows + win_idx] = sum;
}
```

**Replaces:**
```python
# Before
win_scores = scores.view(num_heads, num_windows, omega).sum(dim=2).to(float32)

# After
win_scores = window_score_reduce(scores, num_windows, omega)
```

### Kernel 3: `vote_accumulate_kernel`

Simple elementwise in-place addition for generation-step vote accumulation. No race conditions — each (head, position) is handled by exactly one thread.

```cuda
__global__ void vote_accumulate_kernel(
    float *running_votes, const float *new_scores,
    const int seq_len, const int max_context, const int score_stride) {

  const int head_idx = blockIdx.x;
  const int pos = blockIdx.y * blockDim.x + threadIdx.x;
  if (pos >= seq_len) return;

  running_votes[head_idx * max_context + pos] += new_scores[head_idx * score_stride + pos];
}
```

### Graceful Fallback

The Python wrapper (`sticky_kv_cuda_ops.py`) automatically detects whether the CUDA extension is available:

```python
from sticky_kv_cuda_ops import per_head_kv_gather, window_score_reduce, vote_accumulate

# These functions automatically use:
#   - CUDA kernels if compiled extension (sticky_kv_cuda) is available
#   - Pure PyTorch ops otherwise (identical numerical results)
```

You'll see one of these messages on import:
```
[sticky_kv_cuda_ops] CUDA extension loaded — using accelerated kernels
[sticky_kv_cuda_ops] CUDA extension not found — using PyTorch fallback
```

### Safety Guarantees

| Feature | Detail |
|---|---|
| **Bounds checking** | All index accesses clamped to `[0, seq_len-1]` |
| **No shared memory** | Global memory patterns only — zero race conditions |
| **No warp primitives** | Avoids fragile `__shfl` operations |
| **Debug mode** | `CUDA_ERROR_CHECK` after every kernel launch |
| **Dtype support** | fp16, bf16, fp32 via compile-time dispatch |
| **Graceful fallback** | Identical PyTorch fallback if extension unavailable |

---

## 🚀 Quickstart

### Prerequisites

- **GPU**: CUDA-compatible (Ampere/Hopper recommended — A6000, A100, H100)
- **CUDA Toolkit**: ≥ 11.8
- **Python**: ≥ 3.10
- **PyTorch**: ≥ 2.0 with CUDA support

```bash
git clone https://github.com/shaswata965-coder/LongBenchSticky.git
cd LongBenchSticky
pip install -r requirements.txt
```

### Building CUDA Kernels (Optional — HPC Only)

On your HPC node with A6000/A100 GPUs:

```bash
cd csrc
make                    # Compiles the sticky_kv_cuda extension
python test_kernels.py  # Validates all 3 kernels against PyTorch reference
cd ..
```

> **Note**: If you skip this step, the codebase works identically using pure PyTorch operations. The CUDA kernels provide a performance boost but are not required.

### Configuration

All parameters are centralized in `sticky_config.py`:

```python
# --- Core Sticky KV Parameters ---
R_RATIO = 50              # Total KV cache budget (% of sequence length)
LOCAL_NUM_TOKENS = 0       # Fixed number of local/recent tokens (0 = use P_RATIO)
OMEGA = 5                  # Window size for KV cache grouping
SINK_TOKENS = 0            # Number of permanently protected sink tokens

# --- Model ---
MODEL_PATH = "/path/to/llama-3.2/1b-instruct"

# --- Generation ---
GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "do_sample": False,
    "temperature": 1.0,
}
```

### Running Evaluations

#### 1. Vanilla Baseline (No Eviction)

Establishes ground-truth inference and metrics:

```bash
python Results/run_pure_vanilla_baseline.py
```

#### 2. Sticky KV Baseline (Cumulative Eviction)

Runs with the Sticky KV eviction policy + CUDA acceleration:

```bash
python Results/run_sticky_baseline_cummulative.py
```

#### 3. LongBench Full Evaluation

Runs the complete LongBench suite with the Sticky KV cache:

```bash
python Results/run_longbench_sticky.py
```

#### 4. Metrics & Visualizations

After generating results, compute quality metrics:

```bash
# Layer Information Retention
python Metrices/calculate_layer_information_retention.py

# Window Jaccard Similarity
python Metrices/calculate_window_jaccard.py

# Visualizations
python Metrices/visualize_lir.py
python Metrices/visualize_attention_similarity.py
python Metrices/visualize_per_head_jaccard_divergence.py
```

---

## 🔬 How Sticky KV Eviction Works

### The Algorithm

1. **Prefill Stage**: Process the full prompt using Flash Attention 2. Accumulate attention scores across all query positions. Group scores into `omega`-sized windows and rank windows by cumulative importance.

2. **Initial Eviction**: After prefill, evict the least-important windows to bring the cache within the `R_RATIO` budget. Permanently protect `SINK_TOKENS` initial positions and `LOCAL_NUM_TOKENS` recent positions.

3. **Generation Stage**: For each new token generated:
   - Accumulate its attention distribution into `running_attention_votes`
   - Every `omega` tokens, evaluate window scores and evict the worst-performing window
   - Physically compact the KV cache using per-head index gathering

### Code Flow (Generation)

```python
# Inside STICKYKVCache_LayerWise.__call__() during generation:

# 1. Accumulate votes (CUDA-accelerated)
vote_accumulate(self.running_attention_votes, new_scores, seq_len)

# 2. Every OMEGA tokens, evaluate and evict
if self.tokens_since_last_review == self.omega:
    # Compute per-window scores (CUDA-accelerated)
    scores_slice = self.running_attention_votes[:, sink:review_end]
    win_scores = window_score_reduce(scores_slice, num_windows, self.omega)
    
    # Select worst window via top-k
    worst_windows = torch.topk(win_scores, k=1, largest=False)
    
    # Build survivor indices (sink + kept windows + local)
    all_indices = torch.cat([sink_idx, kept_window_idx, local_idx], dim=1)
    
    # Physical eviction (CUDA-accelerated)
    k_kept, v_kept = per_head_kv_gather(key_cache, value_cache, all_indices)
```

---

## 📊 Metrics

### Layer Information Retention (LIR)

Measures how much of the original attention probability mass is retained after eviction, computed layer-by-layer:

```
LIR(layer) = Σ attention_mass(retained_tokens) / Σ attention_mass(all_tokens)
```

### Attention Jaccard Similarity

Compares the top-k attended token sets between vanilla (full cache) and sticky (evicted cache) per window:

```
J(window) = |TopK_vanilla ∩ TopK_sticky| / |TopK_vanilla ∪ TopK_sticky|
```

---

## ⚙️ Advanced: CUDA Extension Details

### Build Configuration

The CUDA extension compiles for both A100 (`sm_80`) and A6000 (`sm_86`):

```python
# csrc/build.py — Key compiler flags
cc_flag = [
    "-gencode", "arch=compute_80,code=sm_80",  # A100
    "-gencode", "arch=compute_86,code=sm_86",  # A6000
]
nvcc_flags = [
    "-O3", "-std=c++17", "--use_fast_math",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",      # Enable bf16
    "--expt-relaxed-constexpr",                 # Lambda support
]
```

### Testing

The test suite (`csrc/test_kernels.py`) validates each kernel against PyTorch references across:

- **Head counts**: 4, 8
- **Head dimensions**: 64, 128
- **Sequence lengths**: 256 — 4096
- **Window sizes (omega)**: 1, 5, 16, 32
- **Data types**: bf16, fp16
- **Edge cases**: empty gather, zero windows, single element

```bash
cd csrc && python test_kernels.py
# Expected output:
# ✓ per_head_kv_gather: heads=8, seq=4096, kept=500, dim=128, dtype=torch.bfloat16
# ✓ window_score_reduce: heads=8, windows=500, omega=5, max_diff=1.2e-07
# ✓ vote_accumulate: heads=8, seq=4096, max_ctx=131072, max_diff=0.0e+00
# ALL TESTS PASSED
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! If you find bugs or want to benchmark a new dataset against the Sticky KV eviction algorithm, feel free to open a PR.

When contributing CUDA kernel changes, please:
1. Add corresponding test cases in `csrc/test_kernels.py`
2. Verify the PyTorch fallback still produces identical results
3. Test on at least one Ampere GPU before submitting

## 📜 License

[MIT License](LICENSE)
