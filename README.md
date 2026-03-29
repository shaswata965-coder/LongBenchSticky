<div align="center">
  <h1>🧠 LongBench Sticky KV Cache</h1>
  <p><i>An advanced evaluation and inference framework for long-context Large Language Models using Cumulative Sticky Attention Eviction.</i></p>

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
  [![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

---

## 📖 Overview

As Large Language Models (LLMs) scale to handle massive context windows, calculating and storing Key-Value (KV) cache becomes a major memory and compute bottleneck. **LongBench Sticky KV** is an experimental evaluation repository that prototypes an advanced KV-cache eviction strategy: **Cumulative Sticky Attention**.

Instead of naively discarding old tokens or keeping a massive sequence in memory, the "Sticky KV" algorithm retains the most historically important tokens by maintaining a running ledger of attention scores over multiple observation windows. This ensures high performance on long-context benchmarks while drastically reducing the KV-cache memory footprint.

This repository runs standardized configurations of the [LongBench](https://github.com/THUDM/LongBench) suite, comparing unadulterated baseline models against the Sticky KV optimized versions.

## ✨ Key Features

- **Cumulative Sticky Attention Cache**: A dynamic KV cache eviction mechanism that preserves crucial context without causing CUDA OOMs.
- **Granular Token Ledger**: Meticulously tracks attention scores across windows, globally across the context sequence.
- **Layer Information Retention (LIR) Metrics**: Custom metric pipelines to quantitatively analyze the retention of important tokens layer-by-layer.
- **Attention Jaccard Similarity**: Determines the overlap and fidelity of the Sticky KV cache compared exactly against the pure, uncompressed Vanilla baseline.
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

Ensure you have a machine with a CUDA-compatible GPU and PyTorch installed.

```bash
git clone https://github.com/shaswata965-coder/LongBenchSticky.git
cd LongBenchSticky
pip install -r requirements.txt # (Ensure torch, transformers, datasets, numpy are installed)
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
* **`sticky_kv_logic_cummulative.py`**: The engine room of the eviction algorithm. Contains the `STICKYKVCache_LayerWise` class that drops non-essential KV data at generation time based on accumulated window scores.
* **`data_loader.py`**: A clean, unrestricted parser for routing LongBench multi-task datasets, applying tailored prompts seamlessly.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! If you find bugs or want to benchmark a new dataset against the Sticky KV eviction algorithm, feel free to open a PR.

## 📜 License

[MIT License](LICENSE)
