<div align="center">
  <h1>🧠 LongBench Sticky KV Cache</h1>
  <p><i>An advanced evaluation and inference framework for long-context Large Language Models using Cumulative Sticky Attention Eviction.</i></p>

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
  [![Transformers](https://img.shields.io/badge/HuggingFace-Transformers_4.35.2-yellow.svg)](https://huggingface.co/)
  [![FlashAttention](https://img.shields.io/badge/FlashAttention-2.0-orange.svg)](https://github.com/Dao-AILab/flash-attention)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

---

## 📖 Overview

As Large Language Models (LLMs) scale to handle massive context windows, calculating and storing the Key-Value (KV) cache becomes a major memory and compute bottleneck. 

**LongBench Sticky KV** is an experimental evaluation repository that prototypes an advanced KV-cache eviction strategy: **Cumulative Sticky Attention**. Instead of naively discarding old tokens or keeping a massive sequence in memory, the "Sticky KV" algorithm retains the most historically important tokens by maintaining a running ledger of attention scores. This ensures high performance on long-context benchmarks while drastically reducing the KV-cache memory footprint.

This repository runs standardized configurations of the [LongBench](https://github.com/THUDM/LongBench) suite, comparing unadulterated baseline models against the Sticky KV optimized versions.

---

## ⚙️ How it Works: The Architecture

To solve the memory bottleneck, Sticky KV divides the context into distinct, manageable zones and evaluates them periodically.

### 1. The Token Zones
Tokens in the cache are categorized into four dynamic zones:
- **Sink Zone:** Permanent anchor tokens at the start of the sequence (e.g., system prompts). These are never evicted.
- **Local Zone (The Bubble):** A sliding window of the most recently generated tokens. These are temporarily protected from eviction to ensure smooth, immediate generation.
- **Sticky Zone (Main Cache):** The highest-scoring historical token windows. These are retained in full precision (FP16/BF16) because the model frequently attends to them.
- **Q-Cache (Quantized Zone):** The "runner-up" historical windows. Instead of being completely discarded, they are highly compressed into INT8 format and kept as a backup.

### 2. The Eviction Engine
Eviction decisions aren't made every token—they are made in blocks of `OMEGA` tokens (e.g., every 64 tokens). 
1. **Cumulative Scoring:** As the model generates text, it accumulates attention scores for every window. When the Local Zone fills up, the oldest block of tokens exits the "bubble" and becomes a *Challenger*.
2. **The Competition:** The Challenger's scores are compared against the historical scores of the Sticky Zone and the Q-Cache. The top scorers win and stay in the Sticky Zone.
3. **Zero-Degradation Demotion:** If a Sticky window loses the competition, it is demoted to the Q-Cache. To maintain accuracy, its Key cache is quantized **per-channel** (respecting RoPE dimensions), and its Value cache is quantized **per-token**.
4. **Seamless Promotion:** If the model suddenly needs a Q-Cache window again, its attention score will surge. It wins the next competition and is instantly dequantized and promoted back to the FP16 Sticky Zone.

### 3. GPU-Native Performance
Previous eviction methods stalled because they transferred massive cache indexes back and forth between the GPU and CPU. Sticky KV eliminates this entirely:
- **O(1) Block-Boundary Arithmetic:** Instead of iterating over every single token to rebuild the cache, the algorithm samples boundary markers to instantly move large blocks of memory.
- **Vectorized Routing:** Complex Q-Cache promotions use native PyTorch operations (`torch.isin` and `.nonzero()`) strictly on the GPU. 
- **Result:** Instantaneous eviction cycles with zero performance hangs, even on 200,000+ token contexts.

---

## ✨ Key Features

- **Cumulative Sticky Attention Cache**: A dynamic KV cache eviction mechanism that preserves crucial context without causing CUDA OOMs.
- **Zero-Degradation INT8 Q-Cache**: Seamlessly demotes runner-up windows to INT8 and promotes them back to FP16 precisely when needed.
- **GPU-Native O(1) Eviction Pipeline**: Ultra-optimized memory routing to eliminate `.cpu().numpy()` synchronization bottlenecks.
- **Granular Token Ledger**: Meticulously tracks attention scores across windows globally.
- **Layer Information Retention (LIR) Metrics**: Custom metric pipelines to quantitatively analyze the retention of important tokens layer-by-layer.
- **Attention Jaccard Similarity**: Determines the overlap and fidelity of the Sticky KV cache compared exactly against the pure, uncompressed Vanilla baseline.
- **Flash Attention 2.0 Integration**: A native, OOM-safe `fast_attention` variant utilizing `flash_attn` to accelerate prefill stages on Ampere+ hardware without compromising rigorous eviction tracking.
- **Unrestricted Context Evaluations**: Capable of processing raw LongBench datasets with zero mid-truncation or chunking for pure, standardized benchmarking.

## 🗄️ Supported Datasets

The evaluation suite seamlessly supports subsets of the LongBench and PG-19 datasets, categorized by task:

- **Single-Document QA**: `qasper`, `multifieldqa_en`
- **Multi-Document QA**: `2wikimqa`, `musique`
- **Summarization**: `qmsum`
- **Code Completion**: `lcc`
- **Language Modeling**: `PG-19`

## 🚀 Quickstart

### Prerequisites

Ensure you have a machine with a CUDA-compatible GPU (Ampere/Hopper recommended for FA2 operations) and PyTorch installed.

```bash
git clone https://github.com/shaswata965-coder/LongBenchSticky.git
cd LongBenchSticky
pip install -r requirements.txt # (Installs transformers==4.35.2, flash_attn natively, etc.)
```

### Running Evaluations

1. **Vanilla Baseline Testing**
   Run the pure baseline (no KV cache eviction) to establish the ground-truth inference and metrics.
   ```bash
   python Results/run_pure_vanilla_baseline.py
   ```

2. **Sticky KV Testing**
   Run the identically configured test using the Cumulative Sticky cache eviction policy.
   ```bash
   python Results/run_sticky_baseline_cummulative.py
   ```

3. **Metrics & Visualizations**
   After generating the result JSONs, calculate Layer Information Retention (LIR) or Jaccard similarities:
   ```bash
   python Metrices/calculate_layer_information_retention.py
   python Metrices/visualize_attention_similarity.py
   ```
   Visualizations will be securely exported to the respective `Jaccard/` and `LIR/` directories.

## 🏗️ Architecture Design

* **`engine.py`**: The core driver for generating text and calculating QA/Rouge/Edit Similarity metrics across LongBench subsets.
* **`sticky_kv_logic_fast_attention.py` & `sticky_kv_logic_cummulative.py`**: The engine rooms of the eviction methodology. Handles the complex competitive scoring between the Local Zone, Main Cache, and Q-Cache to orchestrate zero-degradation promotions and demotions seamlessly.
* **`sticky_llama_attention_fast_attention.py`**: Native Flash Attention 2 implementations enabling ultra-fast, memory-safe evaluation.
* **`data_loader.py`**: A clean, unrestricted parser for routing LongBench multi-task datasets, applying tailored prompts seamlessly.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! If you find bugs or want to benchmark a new dataset against the Sticky KV eviction algorithm, feel free to open a PR.

## 📜 License

[MIT License](LICENSE)
