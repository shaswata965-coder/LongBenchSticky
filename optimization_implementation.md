# Sticky KV Logic — Optimization Implementation Plan

## Audit Result

Review update: the source-routing blocker bugs from the earlier plan are now addressed in both KV files, but the optimization plan itself still needs correction before implementation.

Evidence checked:
- Syntax: `ast.parse` succeeds for `sticky_kv_logic_fast_attention.py`, `sticky_kv_logic_cummulative.py`, `sticky_llama_attention_fast_attention.py`, and `sticky_llama_attention.py`. `python -m py_compile` could not be used because the workspace denied writing into `__pycache__`.
- Source routing: both KV files now define `_find_logical_window_span` and `_gather_window_from_current_kv`, require a full contiguous `omega` window, and use that resolver in physical eviction fallback and q-cache rebuild.
- Empty q-cache promotion dictionaries are initialized per head before q-cache promotion in both files.
- Pending optimization work remains: OPT-1, OPT-2, OPT-4, OPT-5, OPT-6, and OPT-7 are mostly plan-level proposals, not applied source changes yet.

Review verdict: the approach is directionally good, but do not implement it exactly as written. See the comments below, especially OPT-4 and OPT-7b/7c.

---

## Scope of Optimizations

Six targeted changes that eliminate Python-level overhead and redundant tensor allocations.  
**Core eviction logic, scoring, quantization math, and window promotion semantics are not touched.**

| # | Optimization | Files | Net Benefit |
|---|---|---|---|
| OPT-1 | Vectorize `_create_mask_and_evict` dedup loop | Both | Fewer per-head `torch.unique` calls; still has per-head compaction |
| OPT-2 | Batch `.item()` in `not_in_main_mask` fallback | Both | O(count) CPU syncs → 1 |
| OPT-3 | ~~Sticky zone copy refactor~~ | ~~Both~~ | **Skipped — leave as-is** |
| OPT-4 | Compact first `score_end` clamp only | Both | Minor cleanup; final omega snap must remain |
| OPT-5 | In-place `masked_fill_` for NaN-gather | Both | 1 fewer `[H,k_windows]` tensor/cycle |
| OPT-6 | Precompute `_quant_bytes_len` once | Both | Eliminates repeated conditional |
| OPT-7 | Vectorize `tracking_flag` ledger loops | **Cumulative only** | O(N_live) Python → GPU scatter |

### Review Comments With Evidence

| Item | Verdict | Comment and Evidence |
|------|---------|----------------------|
| Prior source-routing bugs | Looks fixed | Fast path evidence: `_find_logical_window_span` at `sticky_kv_logic_fast_attention.py:233`, q-cache rebuild calls `_gather_window_from_current_kv` at `sticky_kv_logic_fast_attention.py:688`, and physical fallback uses the resolver at `sticky_kv_logic_fast_attention.py:842`. Cumulative mirrors this at `sticky_kv_logic_cummulative.py:267`, `:780`, and `:936`. |
| Syntax | Looks fixed | AST parse completed for both KV files and both attention files. `py_compile` was blocked only by `__pycache__` write permissions, not by Python syntax. |
| OPT-1 | Needs benchmark/semantic check | Current source still uses per-head `torch.unique` in `sticky_kv_logic_fast_attention.py:1010-1022` and `sticky_kv_logic_cummulative.py:1130-1140`. The replacement removes per-head `torch.unique`, but the boolean compaction still loops per head, so the "H kernels -> 2" claim is too strong. |
| OPT-2 | Safe but small | Current source still calls `.item()` inside the fallback loop at `sticky_kv_logic_fast_attention.py:831-833` and `sticky_kv_logic_cummulative.py:925-927`. Batching `final_ids[heads, indices].tolist()` is safe because it only changes scalar extraction. |
| OPT-4 | Comment corrected | The second `score_end` snap is not dead arithmetic. It is required after `min(..., attn_score_cache.shape[3])` in case the attention length is not `omega`-aligned. Current source has this alignment in `sticky_kv_logic_fast_attention.py:324-328` and `sticky_kv_logic_cummulative.py:366-370`. At most combine the first assignment and `min`; do not remove the final aligned recalculation. |
| OPT-5 | Safe if dtype/device preserved | Current source still uses `torch.where(..., torch.zeros_like(...))` at `sticky_kv_logic_fast_attention.py:484-486`; cumulative has the same pattern. The in-place `masked_fill_` replacement is safe because `old_w_gen_scores` is a fresh gather/zero tensor, not an alias that is reused elsewhere. |
| OPT-6 | Safe but currently absent | Current source computes `quant_bytes_len` locally in q-cache rebuild at `sticky_kv_logic_fast_attention.py:643` and `sticky_kv_logic_cummulative.py:735`; `_update_k_win_and_local_num` does not set `_quant_bytes_len` (`sticky_kv_logic_fast_attention.py:1038-1059`, `sticky_kv_logic_cummulative.py:1154-1178`). Add a reset/default path if this field is introduced. |
| OPT-7a | Reasonable, with existing edge case | Current cumulative prefill tracking loop is at `sticky_kv_logic_cummulative.py:430-438`. The proposed vectorization preserves the head-0 physical-index behavior, but it also preserves the existing lack of `phys_idx >= 0` validation. Add that check if changing correctness is allowed. |
| OPT-7b | Not strictly equivalent | Current code skips a whole head when any live physical index is out of range (`valid_phys.max() < attn_score_cache.size(-1)` at `sticky_kv_logic_cummulative.py:510`). The proposed version updates valid in-range elements and masks only invalid elements. That is likely better, but it is a behavior change and should be called out/tested. |
| OPT-7c | Needs dtype fix | Current mapping is float (`mapping[...] = torch.arange(..., dtype=torch.float32)` around `sticky_kv_logic_cummulative.py:895` and `:956`). The proposed `torch.full_like(new_phys, -1.0)` will create a long tensor if `new_phys` is produced from long indexing; assign/cast to the ledger dtype before writing. |

---

## Detailed Change Specifications

---

### OPT-1 — Vectorize dedup loop in `_create_mask_and_evict_from_kv_cache_prompt_stage`

**Both files**  
FA: lines ~1010–1023 | Cumulative: lines ~1130–1142

#### Current code (both files identical)
```python
sorted_indices = []
for h in range(self.num_heads):
    unique = torch.unique(all_indices_clamped[h])
    sorted_indices.append(unique)

safe_len = min(len(u) for u in sorted_indices)
final_indices = torch.stack(
    [u[:safe_len] for u in sorted_indices], dim=0
)
```

#### Replacement code
```python
# Sort all heads in one batched kernel instead of H separate torch.unique() calls
sorted_all, _ = torch.sort(all_indices_clamped, dim=1)   # [H, N]
# Mark the first occurrence of each value per row (diff != 0 means value changed)
diff = torch.cat([
    torch.ones(self.num_heads, 1, device=device, dtype=torch.bool),
    sorted_all[:, 1:] != sorted_all[:, :-1]
], dim=1)                                                  # [H, N]
# Per-head unique count → global safe_len (one .item() sync, unavoidable)
per_head_counts = diff.sum(dim=1)                          # [H]
safe_len = int(per_head_counts.min().item())
# Compact: for each head, pull only the unique values up to safe_len
final_indices = torch.stack([
    sorted_all[h][diff[h]][:safe_len]
    for h in range(self.num_heads)
], dim=0)
```

**Why this is safe:** `torch.unique` internally does `sort + adjacent-diff`. We do the same thing explicitly but share the sort across all heads in one kernel. The semantic result (sorted unique indices, min-length truncation) should be identical.

**Review comment:** This is still not a fully vectorized compaction because `sorted_all[h][diff[h]][:safe_len]` runs in a Python loop per head. Treat the claimed benefit as "remove H `torch.unique` calls" unless benchmarking proves the end-to-end kernel count and latency improvement.

---

### OPT-2 — Batch `.item()` calls in `not_in_main_mask` fallback loop

**Both files**  
FA: lines ~829–849 | Cumulative: lines ~923–944

#### Current code (both files identical)
```python
if not_in_main_mask.any():
    heads, indices = not_in_main_mask.nonzero(as_tuple=True)
    for h_idx, i_idx in zip(heads.tolist(), indices.tolist()):
        wid_val = int(final_ids[h_idx, i_idx].item())   # ← CPU sync per iteration
        new_pos = self.sink_tokens + i_idx * self.omega
        ...
```

#### Replacement code
```python
if not_in_main_mask.any():
    heads, indices = not_in_main_mask.nonzero(as_tuple=True)
    # Extract all wid_vals in one .tolist() call — eliminates O(count) .item() syncs
    all_wid_vals = final_ids[heads, indices].long().tolist()
    heads_list = heads.tolist()
    indices_list = indices.tolist()
    for h_idx, i_idx, wid_val in zip(heads_list, indices_list, all_wid_vals):
        new_pos = self.sink_tokens + i_idx * self.omega
        ...  # (rest of loop body unchanged)
```

**Why this is safe:** `.tolist()` on a small tensor (typically 0–5 elements) does one blocking CPU transfer instead of one per element. All loop body logic is unchanged.

---

### OPT-4 — Compact but preserve the final `score_end` alignment in prefill

**Both files**  
FA: lines ~320–328 | Cumulative: lines ~366–370

#### Current code (both files identical)
```python
score_end = self.sink_tokens + (num_windows * self.omega)
score_end = min(score_end, attn_score_cache.shape[3])   # clamp to available score length

num_windows = (score_end - self.sink_tokens) // self.omega
score_end = self.sink_tokens + (num_windows * self.omega)  # required omega alignment after clamp
```

#### Replacement code
```python
score_end = min(
    self.sink_tokens + (num_windows * self.omega),
    attn_score_cache.shape[3]
)
num_windows = (score_end - self.sink_tokens) // self.omega
score_end = self.sink_tokens + num_windows * self.omega
```

**Why this is safe:** This only combines the initial assignment and clamp. The final recalculation is still required because `attn_score_cache.shape[3]` may not be aligned to `omega`; the floor division and rebuild snap `score_end` back to a complete-window boundary.

---

### OPT-5 — In-place `masked_fill_` to replace `torch.where` allocation

**Both files**  
FA: lines ~478–487 | Cumulative: lines ~564–573

#### Current code (both files identical)
```python
old_ids = torch.nan_to_num(raw_ids, nan=0.0)
old_scores_hist = torch.nan_to_num(raw_scores, nan=0.0)

old_w_gen_scores = torch.gather(scoreboard, 1, old_ids.long()) if valid_old_windows > 0 \
    else torch.zeros_like(old_scores_hist)
# torch.where allocates a third tensor of size [H, k_windows]
old_w_gen_scores = torch.where(is_valid_slot, old_w_gen_scores, torch.zeros_like(old_w_gen_scores))
old_scores = old_scores_hist + old_w_gen_scores
```

#### Replacement code
```python
safe_ids = raw_ids.nan_to_num(nan=0.0).long()
old_scores_hist = raw_scores.nan_to_num(nan=0.0)

if valid_old_windows > 0:
    old_w_gen_scores = scoreboard.gather(1, safe_ids)
    old_w_gen_scores.masked_fill_(~is_valid_slot, 0.0)  # in-place, no extra allocation
else:
    old_w_gen_scores = old_scores_hist.new_zeros(old_scores_hist.shape)
old_scores = old_scores_hist + old_w_gen_scores
```

**Why this is safe:** `torch.where(cond, x, zeros)` is equivalent to `x.masked_fill_(~cond, 0)`. The in-place version avoids allocating `torch.zeros_like(old_w_gen_scores)` — a `[num_heads, k_windows]` tensor that is otherwise allocated and immediately discarded every eviction cycle.

---

### OPT-6 — Precompute `_quant_bytes_len` once

**Both files**  
Target method: `_update_k_win_and_local_num`

#### Change to `_update_k_win_and_local_num` (add one line at end, both files)
```python
def _update_k_win_and_local_num(self, new_tokens, max_tokens):
    ...
    self.k_windows = max(0, available_sticky_tokens // self.omega)
    # NEW: cache the quant byte width so rebuild blocks don't re-evaluate the conditional
    self._quant_bytes_len = self.head_dim if self.quant_bit_width == 8 else (self.head_dim // 2)
```

#### Usage in q-cache rebuild block (both files — replace local computation)
```python
# BEFORE:
quant_bytes_len = head_dim if self.quant_bit_width == 8 else (head_dim // 2)

# AFTER:
quant_bytes_len = self._quant_bytes_len
```

**Why this is safe:** `quant_bit_width` never changes within a document generation pass. Precomputing at allocation time makes the intent explicit and removes the repeated branch.

---

### OPT-7 — Vectorize `tracking_flag` ledger loops

**Cumulative file only**  
Target: prefill ledger loop (lines ~430–438) and generation ledger loop (lines ~501–513)

> [!NOTE]
> This optimization applies **only to `sticky_kv_logic_cummulative.py`**.  
> The fast-attention file has no `tracking_flag` path.

#### 7a — Prefill importance recording (lines ~430–438)

##### Current code
```python
if self.tracking_flag:
    importance_map = attn_score_cache[0, :, :seq_len, :].sum(dim=1)
    active_mask = (self.token_ledger[:, 2:2+self.num_heads] >= 0).any(dim=1) \
                & (self.token_ledger[:, 0] >= 0)
    active_g_ids = torch.where(active_mask)[0]
    for g_id in active_g_ids:
        pre_eviction_phys_idx = self.token_ledger[g_id, 2].long()
        if pre_eviction_phys_idx < importance_map.shape[1]:
            self.token_ledger[g_id, 2+self.num_heads:2+2*self.num_heads] = \
                importance_map[:, pre_eviction_phys_idx]
            self.global_score_history[g_id, :] = importance_map[:, pre_eviction_phys_idx]
```

##### Replacement code
```python
if self.tracking_flag:
    importance_map = attn_score_cache[0, :, :seq_len, :].sum(dim=1)  # [H, seq_len]
    active_mask = (self.token_ledger[:, 2:2+self.num_heads] >= 0).any(dim=1) \
                & (self.token_ledger[:, 0] >= 0)
    active_g_ids = torch.where(active_mask)[0]          # [N_active]
    if active_g_ids.numel() > 0:
        phys_idx = self.token_ledger[active_g_ids, 2].long()  # [N_active] — use head-0 column
        valid = phys_idx < importance_map.shape[1]
        v_gids = active_g_ids[valid]
        v_phys = phys_idx[valid]                                # [N_valid]
        # Gather scores for all valid tokens at once: [H, N_valid] → transpose → [N_valid, H]
        scores_for_valid = importance_map[:, v_phys].T          # [N_valid, H]
        self.token_ledger[v_gids, 2+self.num_heads:2+2*self.num_heads] = scores_for_valid
        self.global_score_history[v_gids, :] = scores_for_valid
```

#### 7b — Generation per-token attention accumulation (lines ~501–513)

##### Current code
```python
if self.tracking_flag:
    live_mask = (self.token_ledger[:, 2:2+self.num_heads] >= 0).any(dim=1)
    live_g_ids = torch.where(live_mask)[0]

    for head_idx in range(self.num_heads):
        phys_indices = self.token_ledger[live_g_ids, 2 + head_idx].long()
        valid_mask = phys_indices >= 0
        valid_phys = phys_indices[valid_mask]
        valid_g_ids = live_g_ids[valid_mask]

        if len(valid_phys) > 0 and valid_phys.max() < attn_score_cache.size(-1):
            head_scores = attn_score_cache[0, head_idx, 0, valid_phys]
            self.token_ledger[valid_g_ids, 2 + self.num_heads + head_idx] += head_scores.float()
            self.global_score_history[valid_g_ids, head_idx] += head_scores.float()
```

##### Replacement code
```python
if self.tracking_flag:
    live_mask = (self.token_ledger[:, 2:2+self.num_heads] >= 0).any(dim=1)
    live_g_ids = torch.where(live_mask)[0]              # [N_live]

    if live_g_ids.numel() > 0:
        # Gather all physical indices for all heads at once: [N_live, H]
        all_phys = self.token_ledger[live_g_ids, 2:2+self.num_heads].long()  # [N_live, H]
        valid_phys_mask = all_phys >= 0                  # [N_live, H]

        # Extract 1-step attention scores for all heads: [H, kv_seq_len]
        gen_scores = attn_score_cache[0, :, 0, :]       # [H, kv_seq_len]
        kv_seq = gen_scores.shape[1]

        # Clamp physical indices to valid range for safe gather, zero out invalid slots after
        safe_phys = all_phys.clamp(min=0, max=kv_seq - 1)  # [N_live, H]

        # Gather: for each (token, head) pair, get the attention score
        # gen_scores.T is [kv_seq, H]; we need [N_live, H]
        # Use advanced indexing: gen_scores[head_idx, phys_idx] for each pair
        head_idx_grid = torch.arange(self.num_heads, device=gen_scores.device)\
            .unsqueeze(0).expand(live_g_ids.shape[0], -1)  # [N_live, H]
        gathered = gen_scores[head_idx_grid, safe_phys]    # [N_live, H]

        # Zero out invalid slots (phys < 0 or phys >= kv_seq)
        in_range = valid_phys_mask & (all_phys < kv_seq)
        gathered = gathered * in_range.to(gathered.dtype)  # [N_live, H]

        # Accumulate into ledger and history (both are [N_live, H] shaped writes)
        self.token_ledger[live_g_ids, 2+self.num_heads:2+2*self.num_heads] += gathered.float()
        self.global_score_history[live_g_ids, :] += gathered.float()
```

**Review comment:** This is not strictly identical to the current guard. The current loop skips an entire head when any valid physical index for that head is out of range; this replacement masks invalid elements individually and still updates valid elements. That is probably more robust, but it is a behavior change and must be tested against ledger output with intentionally stale/out-of-range physical indices.

#### 7c — Generation eviction ledger update (lines ~961–981 in Cumulative)

##### Current code
```python
if self.tracking_flag:
    for head_idx in range(self.num_heads):
        phys_col = 2 + head_idx
        live_mask = (self.token_ledger[:, 2:2+self.num_heads] >= 0).any(dim=1)
        g_ids = torch.where(live_mask)[0]
        old_phys = self.token_ledger[g_ids, phys_col].long()

        valid_old = (old_phys >= 0) & (old_phys < seq_len)
        valid_old_phys = old_phys[valid_old]
        valid_g_ids = g_ids[valid_old]

        if len(valid_g_ids) > 0:
            head_scores_acc = self.running_attention_votes[head_idx, valid_old_phys]
            self.token_ledger[valid_g_ids, 2 + self.num_heads + head_idx] += head_scores_acc.float()
            self.global_score_history[valid_g_ids, head_idx] += head_scores_acc.float()

        new_phys = mapping[head_idx, valid_old_phys]
        self.token_ledger[valid_g_ids, phys_col] = new_phys
```

##### Replacement code
```python
if self.tracking_flag:
    live_mask = (self.token_ledger[:, 2:2+self.num_heads] >= 0).any(dim=1)
    g_ids = torch.where(live_mask)[0]                   # [N_live]

    if g_ids.numel() > 0:
        # All physical positions across all heads at once: [N_live, H]
        all_phys = self.token_ledger[g_ids, 2:2+self.num_heads].long()
        valid_mask = (all_phys >= 0) & (all_phys < seq_len)   # [N_live, H]
        safe_phys = all_phys.clamp(min=0, max=seq_len - 1)    # [N_live, H]

        # Gather running attention votes for all (token, head) pairs: [N_live, H]
        head_grid = torch.arange(self.num_heads, device=device)\
            .unsqueeze(0).expand(g_ids.shape[0], -1)           # [N_live, H]
        votes = self.running_attention_votes[head_grid, safe_phys]  # [N_live, H]
        votes = votes * valid_mask.to(votes.dtype)             # zero invalid slots

        # Accumulate final omega-step scores before eviction
        self.token_ledger[g_ids, 2+self.num_heads:2+2*self.num_heads] += votes.float()
        self.global_score_history[g_ids, :] += votes.float()

        # Update physical positions using the mapping tensor: [N_live, H]
        new_phys = mapping[head_grid, safe_phys]               # [N_live, H], mapping dtype
        # For invalid old slots, set new_phys to -1 (evicted)
        new_phys = torch.where(valid_mask, new_phys, torch.full_like(new_phys, -1.0))
        self.token_ledger[g_ids, 2:2+self.num_heads] = new_phys
```

**Review comment:** Keep `new_phys` in `mapping`/ledger dtype. Do not derive `new_phys` from a long tensor and then use `torch.full_like(new_phys, -1.0)`, because that would create a long `-1` fill and then rely on implicit assignment casting into the float ledger. The version above is safe only because `mapping[head_grid, safe_phys]` is the source tensor.

---

## Files to Edit

| File | OPTs Applied |
|------|-------------|
| `sticky_kv_logic_fast_attention.py` | OPT-1, OPT-2, OPT-4, OPT-5, OPT-6 |
| `sticky_kv_logic_cummulative.py` | OPT-1, OPT-2, OPT-4, OPT-5, OPT-6, OPT-7a, OPT-7b, OPT-7c |

---

## Verification Plan

### Automated Tests
1. Run `python main.py --task qasper` — medium context, exercises full prefill + 1+ eviction cycles.
2. Run `python main.py --task gov_report` — long context (35k+), exercises q-cache rebuild with promotion.
3. Confirm metrics match the pre-optimization baseline (±0.01 tolerance for float rounding).

### Ledger Validation (tracking_flag path)
4. Set `tracking_flag = 1` in `sticky_config.py`, run a short task, call `get_ledger_data()`, and confirm the returned `attention_score` columns have no rows of all-zeros that weren't all-zeros before.

### Performance Check
5. Profile with `torch.profiler` on a 40k-token input and confirm reduction in CPU-side blocking time vs. pre-optimization baseline.

---

## Exclusions (by user decision)

- **OPT-3 (sticky zone copy refactor)**: Left as-is. The `found_in_main` boolean-mask + advanced-indexing path is already GPU-native and runs only once per `omega` tokens. For GQA models with small KV head count (e.g. 4), the intermediate tensor is negligible.
