# Sticky KV Qwen2 Bug Report

This document tracks **known issues, structural limitations, and resolved bugs** in the Qwen2 Sticky KV integration in `C:\qwen_adapted`.

---

## 1. Flash Attention prefill ignores padding masks (batched prefill corruption)
**Severity:** Moderate (Incorrect outputs for padded batched prefill)  
**Location:** `sticky_qwen2_attention_fast_attention.py` (FlashAttention path, `q_len > 1`)  
**Status:** 🚧 KNOWN STRUCTURAL LIMITATION

### What happens
During prefill, the fast attention implementation calls `flash_attn_func(..., causal=True)` and **does not incorporate** the model’s `attention_mask`. This makes padding behave like real tokens during prefill, degrading results for batched inputs with unequal lengths.

### Recommended future action
Migrate the batched prefill path to `flash_attn_varlen_func` and compute `cu_seqlens_q/cu_seqlens_k` from the padding mask (unpad → attend → repad).

---

## 2. `position_ids` / RoPE misalignment after sticky eviction
**Severity:** Critical (Garbled generation in long context)  
**Location:** `sticky_qwen2_attention.py`, `sticky_qwen2_attention_fast_attention.py`  
**Status:** ✅ RESOLVED (single-sequence / uniform-length batches)

### What happened
Sticky eviction truncates the **physical** KV cache, which previously caused `position_ids` to reset/misalign and permanently break RoPE.

### Resolution
Both attention modules override `position_ids` during decode using `kv_cache.global_token_counter` so RoPE reflects the **logical** sequence length.

---

## 3. `global_token_counter` desync during chunked/speculative decoding (`q_len > 1`)
**Severity:** High (Breaks RoPE after first chunked step)  
**Location:** `sticky_kv_logic_fast_attention.py`, `sticky_kv_logic_cummulative.py`  
**Status:** ✅ RESOLVED

### What happened
Generation-stage logic previously did `global_token_counter += 1` even when `q_len > 1`, causing the logical counter to drift and breaking subsequent RoPE.

### Resolution
Both KV logic modules now increment `global_token_counter` by **`q_len`** during generation. The cumulative backend’s ledger registration also now registers all `q_len` tokens.

---

## 4. Fast attention backend ignores q-cache during chunked/spec decode (`q_len > 1`)
**Severity:** Moderate (Evicted tokens are ignored; q-cache gets no votes)  
**Location:** `sticky_qwen2_attention_fast_attention.py`  
**Status:** ✅ RESOLVED (correctness-first fallback)

### What happened
The `q_len > 1` FlashAttention path did not incorporate q-cache, so after eviction, chunked/spec decode could not attend to evicted tokens.

### Resolution
When q-cache is present and `q_len > 1`, the fast-attention module now **falls back to the manual attention path** (non-FlashAttention) so q-cache participates in attention/scoring.

---

## 5. `prefill_attention_matrix` is `None` in fast attention backend
**Severity:** High (Breaks cumulative evaluation expectations)  
**Location:** `sticky_kv_logic_fast_attention.py`, `Results/run_*_cummulative*.py`  
**Status:** 🚧 STRUCTURAL LIMITATION + FAIL-FAST GUARD ADDED

### What happens
The fast-attention backend does not materialize a full NxN prefill attention matrix in Python memory, so `kv_cache.prefill_attention_matrix` remains `None`.

### Current safe behavior
The cumulative evaluation scripts now **raise a clear error** if `prefill_attention_matrix` is `None` to prevent silent generation of invalid evaluation artifacts.

### Recommended future action
If prefill NxN matrices are required from the fast backend, add an optional “materialize prefill matrix” mode (slow / research-only), or implement an efficient export using varlen attention + explicit score extraction.

---

## 6. `use_fast_attention` config not persisting through HF serialization
**Severity:** Moderate (Ignores developer override)  
**Location:** `configuration_sticky_qwen2.py`  
**Status:** ✅ RESOLVED

### Resolution
`StickyQwen2Config` now pops/stores `use_fast_attention`, ensuring it survives HuggingFace `from_pretrained` config cycles.

---

## 7. Cumulative backend prefill control-flow bug (attention compute gated on `past_kv`)
**Severity:** Critical (Prefill can crash or skip compute)  
**Location:** `sticky_qwen2_attention.py`  
**Status:** ✅ RESOLVED

### What happened
The main attention compute block was accidentally indented under `if past_kv is not None`, so prefill (`past_kv is None`, `q_len > 1`) could fail.

### Resolution
The attention compute block now runs for both prefill and decode; only KV concatenation remains gated on `past_kv`.

---

## 8. Padding mask handling is incomplete across backends
**Severity:** High for padded/ragged batching  
**Location:** `sticky_qwen2_attention.py`, `sticky_qwen2_attention_fast_attention.py`  
**Status:** ⚠️ PARTIALLY MITIGATED

### What happens
Historically both attention modules dropped the caller-provided padding `attention_mask`. This corrupted behavior for left-padded batches even outside FlashAttention.

### Current state
Both attention modules now preserve and apply **2D padding masks** (converted to additive form) in the manual attention path.  
The FlashAttention prefill path still cannot correctly handle padding (see Issue #1).

---

## 9. Device placement risks with `device_map="auto"` / offload
**Severity:** High (device mismatch / unexpected transfers)  
**Location:** `sticky_kv_logic_fast_attention.py`, `sticky_kv_logic_cummulative.py`  
**Status:** 🚨 KNOWN ISSUE (NOT YET FIXED)

### What happens
KV-cache modules allocate buffers on `cuda` if available inside `__init__`, which can disagree with the layer’s dispatched device under HF `device_map`.

### Recommended future action
Register buffers without forcing a device in `__init__`, and always create temporary tensors on `hidden_states.device` at use-time.

---

## 10. Batched generation RoPE correctness (different lengths per batch element)
**Severity:** High (silent quality collapse)  
**Location:** both attention backends  
**Status:** 🚧 LIMITATION (single logical counter per layer)

### What happens
`global_token_counter` is a scalar per layer. In batched generation where examples have different past lengths, a scalar counter cannot represent per-example `position_ids`.

### Recommended future action
Track logical lengths per batch element (vector `[bsz]`) or derive per-example positions from HF cache metadata (`cache_position` / per-example seen tokens).

---

## 11. Unconditional `flash_attn` import can crash even when fast attention is disabled
**Severity:** Critical (import-time crash on machines without FlashAttention)  
**Location:** `sticky_qwen2_attention.py`, `sticky_qwen2_attention_fast_attention.py`  
**Status:** 🚨 KNOWN ISSUE (NOT YET FIXED)

### What happens
Both attention modules import `flash_attn` at module import time. If `flash_attn` is not installed, Python raises `ModuleNotFoundError` immediately — even if you intend to use the cumulative/manual backend.

### Recommended fix
Wrap the `flash_attn` import in `try/except` and only require it when the FlashAttention path is actually used. Provide a clear error message or automatic fallback when unavailable.

---

## 12. Mixed-device buffer initialization in cumulative KV logic (CPU vs CUDA)
**Severity:** High (likely runtime crash / implicit transfers)  
**Location:** `sticky_kv_logic_cummulative.py` (`token_ledger`, `global_score_history`)  
**Status:** 🚨 KNOWN ISSUE (NOT YET FIXED)

### What happens
In the cumulative KV logic, most buffers are created on a selected device (CUDA if available), but `token_ledger` and `global_score_history` are created without an explicit device. This can place them on CPU while other buffers and runtime tensors are on CUDA, causing device-mismatch errors once tracking code touches them.

### Recommended fix
Ensure all registered buffers in the module live on the same device, and avoid forcing device selection inside `__init__` (see Issue #9).

---

## 13. `main.py` uses `model.device` with `device_map=\"auto\"` (unsafe for sharded models)
**Severity:** Medium–High (environment-dependent crash)  
**Location:** `main.py`  
**Status:** 🚨 KNOWN ISSUE (NOT YET FIXED)

### What happens
With `device_map=\"auto\"`, a model can be sharded/offloaded across multiple devices. `model.device` may not reflect the device of the input embedding layer. Forcing inputs via `.to(model.device)` can therefore put inputs on the wrong device and crash when the first module runs.

### Recommended fix
Move inputs to the embedding/input device (e.g. `model.model.embed_tokens.weight.device`) or avoid forcing input `.to(...)` and let the HF/Accelerate dispatch handle it.

---

## 14. `cache_position` is accepted but ignored by custom attention modules
**Severity:** Medium (compatibility risk with some `transformers` versions / generation modes)  
**Location:** `sticky_qwen2_attention.py`, `sticky_qwen2_attention_fast_attention.py`  
**Status:** 🚨 KNOWN ISSUE (NOT YET FIXED)

### What happens
The attention forward signatures accept `cache_position` but do not use it. Some `transformers` cache implementations/generation modes rely on `cache_position` for correct positional handling.

### Recommended fix
If `cache_position` is provided, use it as the source of logical positions (or validate it against `global_token_counter`). Document the supported `transformers` versions and cache types.
