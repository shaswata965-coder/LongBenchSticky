# Sticky KV Cache Optimization Plan (Bug-Aware Revision)

This plan audits the current optimization work around `run_longbench_sticky.py`, sticky attention, KV eviction, and q-cache quantization. The current code has implemented some intended optimizations, but several changes are not safe yet. Fix correctness first, then optimize further.

## Active LongBench Path

`Results/run_longbench_sticky.py` imports `STICKYLlamaForCausalLM` from `sticky_llama_model.py`. That model replaces each layer's attention with `STICKYLlamaAttention` from `sticky_llama_attention_fast_attention.py`, which uses `STICKYKVCache_LayerWise` from `sticky_kv_logic_fast_attention.py`.

So the primary LongBench path is:

```text
Results/run_longbench_sticky.py
  -> sticky_llama_model.py
  -> sticky_llama_attention_fast_attention.py
  -> sticky_kv_logic_fast_attention.py
```

The cumulative files still matter because baseline/research scripts import them, but they are not the default `run_longbench_sticky.py` path.

## 0. Blocker Bugs To Fix First

### 0.1 `sticky_kv_logic_cummulative.py` Does Not Parse

**Severity:** P0  
**Evidence:** `sticky_kv_logic_cummulative.py:641-643` has an unexpected indent after the per-head q-cache promotion rewrite.

Current broken shape:

```python
for qi in promoted_qi_indices.tolist():
    q_wid = self.q_cache_ids[h, qi]
        # Dequantize using per-window index qi directly
        k_deq = self._dequantize_from_quant(
```

**Risk:** Any cumulative path import fails before runtime. This breaks scripts that use `sticky_llama_attention.py` or cumulative baseline tooling.

**Fix direction:**

Dedent the dequantization body under `for qi`, and keep the per-head promotion mask:

```python
promo_mask = (
    self.q_cache_ids.long().unsqueeze(2)
    == final_ids.long().unsqueeze(1)
).any(dim=2)

for h in range(self.num_heads):
    promoted_q_data_k[h] = []
    promoted_q_data_v[h] = []
    promoted_qi_indices = promo_mask[h].nonzero(as_tuple=True)[0]
    for qi in promoted_qi_indices.tolist():
        q_wid = self.q_cache_ids[h, qi]
        k_deq = self._dequantize_from_quant(...)
        v_deq = self._dequantize_from_quant(...)
```

### 0.2 Empty Q-Cache Path Can Crash

**Severity:** P1  
**Evidence:** In `sticky_kv_logic_fast_attention.py`, `promoted_q_data_k` and `promoted_q_data_v` are initialized as empty dicts at `540-541`, filled only inside `if self.q_cache_ids is not None` at `542-570`, but consumed unconditionally at `779-780`:

```python
_prom_k = {(_h, int(w)): k for _h in range(self.num_heads) for w, k in promoted_q_data_k[_h]}
```

**Risk:** If q-cache is disabled, empty, or no q-cache losers were captured, generation eviction can raise `KeyError`.

**Fix direction:**

Initialize per-head lists before the `if`:

```python
promoted_q_data_k = {h: [] for h in range(self.num_heads)}
promoted_q_data_v = {h: [] for h in range(self.num_heads)}

if self.q_cache_ids is not None:
    ...
```

Apply the same pattern in `sticky_kv_logic_cummulative.py`.

### 0.3 Challenger Windows Are Still Not Safely Resolved

**Severity:** P1  
**Evidence:**
- `sticky_kv_logic_fast_attention.py:468-506` constructs the challenger with `last_id_tensor = last_id_val` and appends it into `competing_ids`.
- `sticky_kv_logic_fast_attention.py:520-524` gathers `surviving_ids` into `final_ids`, so `final_ids` can contain `last_id_val`.
- `sticky_kv_logic_fast_attention.py:822-844` handles `not_in_main_mask` by first checking q-cache promotion dictionaries, then computing `local_base_lid = last_id_val + 1`.
- Therefore, a selected challenger with `wid_val == last_id_val` is not matched by the local fallback:

Current risky shape:

```python
local_base_lid = last_id_val + 1
local_offset = (wid_val - local_base_lid) * self.omega

p_k = _prom_k.get((h_idx, wid_val))
p_v = _prom_v.get((h_idx, wid_val))
if p_k is not None:
    new_k[...] = p_k
    new_v[...] = p_v
elif 0 <= local_offset < local_tokens_count:
    ...
else:
    raise RuntimeError(...)
```

**Risk:** The previous silent zero-copy corruption has been improved into a fail-fast error, but the valid challenger window is still not preserved. `run_longbench_sticky.py` can still crash at the first review cycle where the challenger wins and is not in old sticky/q-cache.

**Current partial fix status:** The code no longer blindly sets a full logical id over zero KV when no source exists, but it still uses the wrong local base for the challenger case.

**Fix direction:**

Stop deriving physical positions from `last_id_val + 1` alone. Resolve `wid_val` by asking the current logical map where that window physically lives.

Recommended resolver:

```python
def _find_logical_window_span(self, h, wid_val, seq_len):
    positions = (self.logical_id_map[h] == int(wid_val)).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        return None

    start = int(positions.min().item())
    end = int(positions.max().item()) + 1

    # Only accept spans that are contiguous and full-window-sized for sticky/q-cache promotion.
    expected = torch.arange(start, end, device=positions.device)
    if positions.numel() != (end - start) or not torch.equal(positions, expected):
        raise RuntimeError(f"Logical window {wid_val} has non-contiguous physical positions")

    if end - start != self.omega:
        return None  # defer partial windows; do not promote them into sticky/q-cache

    if end > seq_len:
        return None
    return start, end
```

Then the physical fallback becomes:

```python
span = self._find_logical_window_span(h_idx, wid_val, seq_len)
if span is not None:
    old_start, old_end = span
    new_k[0, h_idx, new_pos:new_pos+self.omega] = past_key_values[0][0, h_idx, old_start:old_end]
    new_v[0, h_idx, new_pos:new_pos+self.omega] = past_key_values[1][0, h_idx, old_start:old_end]
    new_logical_id_map[h_idx, new_pos:new_pos+self.omega] = wid_val
else:
    raise RuntimeError(...)
```

This directly answers: "where is this logical window right now?" It also avoids off-by-one assumptions around `last_id_val`.

### 0.4 Q-Cache Rebuild Still Misses Challenger Windows

**Severity:** P1  
**Evidence:**
- `sticky_kv_logic_fast_attention.py:526-537` builds `new_q_loser_ids` from the non-surviving `competing_ids`.
- `competing_ids` can include the challenger because `sticky_kv_logic_fast_attention.py:504-506` concatenates `last_id_tensor` into the competition set.
- `sticky_kv_logic_fast_attention.py:668-686` uses `local_base_lid = last_id_val + 1`, so `wid_val == last_id_val` falls through old sticky search and local fallback.

```python
local_base_lid = last_id_val + 1
local_offset = (wid_val - local_base_lid) * self.omega
...
elif 0 <= local_offset < local_tokens_count:
    ...
else:
    raise RuntimeError(...)
```

**Risk:** The code no longer quantizes all-zero tensors for this case, which is safer than before. However, valid challenger/local q-cache losers can still raise and stop generation. If the q-cache slot should be retained, the code must gather the real physical KV window before quantization.

**Fix direction:**

Reuse the logical-map resolver from section 0.3 for the q-cache rebuild. Routing order:

1. retained q-cache: copy old quantized tensors directly;
2. old main sticky block: gather from `block_wids/block_starts`;
3. current logical-map span: gather from `self.logical_id_map`;
4. archived metadata path: only reuse scale/zp if a valid source span was found;
5. missing or partial source: do not quantize zeros. Either drop the q-cache candidate or raise an explicit assertion.

Example:

```python
span = self._find_logical_window_span(h, wid_val, seq_len)
if span is None:
    raise RuntimeError(f"Q-cache rebuild failed: source for window {wid_val} missing or partial")

old_start, old_end = span
k_fp = past_key_values[0][0, h, old_start:old_end]
v_fp = past_key_values[1][0, h, old_start:old_end]
```

### 0.5 Partial Local Windows Can Still Leave Incomplete Sticky Slots

**Severity:** P2  
**Evidence:**
- `sticky_kv_logic_fast_attention.py:837-842` copies only `copy_len` tokens into the destination sticky slot when the local fallback matches:

```python
copy_len = min(self.omega, seq_len - old_pos)
new_k[0, h_idx, new_pos:new_pos+copy_len] = ...
new_v[0, h_idx, new_pos:new_pos+copy_len] = ...
new_logical_id_map[h_idx, new_pos:new_pos+copy_len] = wid_val
```

- `new_k`, `new_v`, and `new_logical_id_map` were initialized at `sticky_kv_logic_fast_attention.py:753-755`, so any destination positions after `copy_len` remain zero/default.
- The cumulative path mirrors this at `sticky_kv_logic_cummulative.py:932-938`.

**Risk:** If a partial local window enters `final_ids` or q-cache losers, the cache contains a half-valid sticky/q-cache window. That corrupts attention while making the logical map look partly valid.

**Fix direction:**

Do not promote partial windows into sticky or q-cache storage. Require `end - start == self.omega` in the resolver. If the current logical map contains only a partial local window, keep it in the local zone until it becomes complete. This fits the existing chunk invariant that sticky/q-cache windows are `omega` tokens.

### 0.6 Cumulative Path Mirrors The Remaining Source-Routing Risks

**Severity:** P2 after parse fix  
**Evidence:**
- `sticky_kv_logic_cummulative.py:917-940` uses the same `local_base_lid = last_id_val + 1` fallback and partial `copy_len` behavior as the fast path.
- `sticky_kv_logic_cummulative.py:932-938` writes only partial local spans to `new_k`, `new_v`, `new_logical_id_map`, and `mapping`.
- `sticky_kv_logic_cummulative.py:943-976` then consumes `mapping` to update token ledger physical positions.

**Risk:** Cumulative now parses, but logical source routing is still not safe. Bad or partial mapping can corrupt research ledger data even if generation appears to continue.

**Fix direction:**

Apply the same full-window logical-map resolver to cumulative. Additionally, verify `mapping` is written for every survivor copied from sinks, old sticky blocks, and local spans. Promoted q-cache windows should remain unmapped only if they truly had no live old physical position.

## 1. Safe Optimization: Per-Head Q-Cache Promotion Mask

**Status:** Good direction, but fix cumulative indentation and empty-dict initialization first.

**Evidence:**
- Old cumulative path used per-head `torch.isin(q_wid, final_ids[h])`, slow but correct.
- Earlier fast path used global `torch.isin(self.q_cache_ids.long(), final_ids.long())`, which could match across heads.

**Safe implementation:**

```python
promoted_q_data_k = {h: [] for h in range(self.num_heads)}
promoted_q_data_v = {h: [] for h in range(self.num_heads)}

if self.q_cache_ids is not None:
    promo_mask = (
        self.q_cache_ids.long().unsqueeze(2)
        == final_ids.long().unsqueeze(1)
    ).any(dim=2)

    for h in range(self.num_heads):
        for qi in promo_mask[h].nonzero(as_tuple=True)[0].tolist():
            ...
```

This preserves per-head correctness and avoids crashing when q-cache is absent.

## 2. Safe Optimization: Physical Eviction Vectorization

**Status:** Partially implemented in fast and cumulative, but still unsafe until logical-map source routing replaces `last_id_val + 1`.

**Evidence:**
- Fast path now vectorizes sinks at `sticky_kv_logic_fast_attention.py:789-792`.
- Fast path vectorizes main sticky hits at `795-815`.
- Fast path copies local zone at `831-839`.
- The remaining bug is `not_in_main_mask` local/challenger routing at `817-844`; it uses `local_base_lid = last_id_val + 1`, but the challenger itself is `last_id_val`.

**Safe implementation shape:**

1. Allocate `new_k`, `new_v`, `new_logical_id_map`.
2. Copy sink tokens across all heads.
3. Vector-copy `found_in_main` sticky windows from old sticky blocks.
4. For `not_in_main_mask`, route each window:
   - q-cache promoted data;
   - full-window logical-map span from `self.logical_id_map`;
   - explicit error for missing source.
5. Copy local-zone tail.
6. In cumulative mode, update `mapping` for sinks, main sticky hits, and local-zone tail.

Do not allow a code path that sets `new_logical_id_map` without also writing valid KV tensors for that same span.

## 3. Safe Optimization: Grouped Matmul Instead Of `repeat_kv`

**Status:** Implemented in both sticky attention files and structurally reasonable.

**Evidence:**
- `sticky_llama_attention_fast_attention.py:306-318` computes main logits without `repeat_kv`.
- `sticky_llama_attention_fast_attention.py:320` initializes `q_scores_for_cache = None`.
- `sticky_llama_attention_fast_attention.py:321-378` handles q-cache logits and output with grouped matmul while preserving joint softmax.
- `sticky_llama_attention.py:227` also initializes `q_scores_for_cache = None`.

**Remaining checks:**
- Compare outputs for one short prompt against the old `repeat_kv` path with q-cache disabled.
- Compare q-cache-enabled generation after the first eviction cycle.
- Verify shape behavior for `num_heads == num_key_value_heads`.

This optimization does not appear to corrupt eviction logic directly because `scores_for_cache` and `q_scores_for_cache` are still passed to the KV cache.

## 4. Q-Cache Quantization Integrity

**Status:** Retained q-cache copy path is good; fresh/non-retained source routing is safer than before because it raises instead of quantizing zeros, but it still misses challenger windows.

**Evidence:**
- Retained q-cache windows copy raw quantized tensors and scale/zp at `sticky_kv_logic_fast_attention.py:629-645`.
- Fresh/non-retained windows now raise when not found, but `sticky_kv_logic_fast_attention.py:668-686` can still fail on valid challenger ids because it uses `last_id_val + 1`.

**Fix direction:**

Create one shared helper for both physical promotion and q-cache rebuild:

```python
def _gather_window_from_current_kv(self, past_key_values, h, wid_val, *, seq_len):
    span = self._find_logical_window_span(h, wid_val, seq_len)
    if span is None:
        return None
    start, end = span
    return (
        past_key_values[0][0, h, start:end],
        past_key_values[1][0, h, start:end],
    )
```

Use it everywhere a logical window id must become float KV data. This reduces duplicate routing logic and prevents the two corruption classes above from drifting apart.

## 5. Deferred Optimizations

### 5.1 Dictionary Replacement

Keep this deferred. `_prom_k/_prom_v` dictionaries are not the main bottleneck after main physical copy is vectorized. They are also useful for clarity while fixing the routing bugs.

### 5.2 `torch.unique` In Prefill

Keep this deferred. The current prefill compression deduplicates per head and then stacks rows using `safe_len`. A vectorized replacement must preserve equal row length without padding duplicate KV tokens or dropping protected sink/sticky tokens incorrectly.

### 5.3 Full Q-Cache Rebuild Vectorization

Do not vectorize further until source routing is correct. Once correctness is fixed, the remaining loop over `new_q_count` and heads can be optimized in batches.

## Recommended Order Of Work

1. Fix `sticky_kv_logic_cummulative.py` syntax.
2. Initialize promoted q-cache dictionaries per head in both KV files.
3. Add a shared full-window source resolver based on `self.logical_id_map`, not `last_id_val + 1`.
4. Use the resolver in physical eviction fallback.
5. Use the resolver in q-cache rebuild fresh/non-retained path.
6. Add assertions/tests that no selected `final_ids` window produces all-zero KV unless the source KV was actually all-zero.
7. Only then benchmark grouped matmul and physical vector copy performance on `run_longbench_sticky.py`.
