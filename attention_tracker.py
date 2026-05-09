"""
attention_tracker.py
====================
Non-intrusive, window-granularity attention recorder for the Sticky KV cache.

Records per-head window attention at:
  - Prefill end (first eviction)
  - Every omega-step eviction during generation

All three zones (sticky, quantized, local) are stored with an identical schema.
All tensors are moved to CPU immediately after recording.

Public API (called on STICKYKVCache_LayerWise.attn_tracker):
    Snapshots:
        get_snapshot("prefill")    -> AttentionSnapshot | None
        get_snapshot(N)            -> AttentionSnapshot | None  (N = 0,1,2,... gen eviction index)
        list_snapshots()           -> list of keys

    Ledger (incremental global window lifecycle record):
        ledger.get_all(head)       -> dict[int, WindowEntry]  — all windows ever seen for that head
        ledger.get_alive(head)     -> list[WindowEntry]        — currently alive windows
        ledger.get_evicted(head)   -> list[WindowEntry]        — evicted windows
        ledger.entries             -> list[dict] of length num_heads
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional


# ============================================================
# Snapshot — one eviction event
# ============================================================

@dataclass
class AttentionSnapshot:
    """
    One eviction-event snapshot. All tensors are on CPU.

    Shapes:
        window_ids        [H, W]  int64   — logical window IDs (0 = first window after sinks)
        step_scores       [H, W]  float32 — attention received this eviction step only
        cumulative_scores [H, W]  float32 — total cumulative attention since sequence start
        zones             List[str] len W  — "sticky" | "quantized" | "local"  (same for all heads)

    Window ID semantics:
        Window W covers physical token positions [sink_tokens + W*omega, sink_tokens + (W+1)*omega).
        Window IDs are 0-indexed from the first token after sink tokens.

    eviction_key:
        "prefill"  -> snapshot taken at end of prefill (first eviction)
        0, 1, 2... -> 0-based index of generation-phase eviction cycles
    """
    layer_idx:         int
    eviction_key:      Union[str, int]
    num_heads:         int
    omega:             int
    sink_tokens:       int

    window_ids:        torch.Tensor   # [H, W] int64,   CPU
    step_scores:       torch.Tensor   # [H, W] float32, CPU
    cumulative_scores: torch.Tensor   # [H, W] float32, CPU
    zones:             List[str]      # len W


# ============================================================
# Window Ledger — global incremental lifecycle record
# ============================================================

@dataclass
class WindowEntry:
    """
    Lifecycle record for one logical window, per head.

    Fields:
        window_id            : logical window index (0-indexed after sink tokens)
        first_seen           : eviction_key when this window was first recorded ("prefill" or int)
        last_seen            : eviction_key of the most recent snapshot where it was present
        current_zone         : zone label at last_seen ("sticky" | "quantized" | "local")
        last_cumulative_score: cumulative attention score from the last snapshot where it was alive
        status               : "alive" — present in the most recent snapshot
                               "evicted" — was present before but absent from latest snapshot
        evicted_at           : eviction_key of the step at which it was first found missing
                               (None while status == "alive")
    """
    window_id:             int
    first_seen:            Union[str, int]
    last_seen:             Union[str, int]
    current_zone:          str
    last_cumulative_score: float                   # cumulative attention at last_seen step
    status:                str                     # "alive" | "evicted"
    evicted_at:            Optional[Union[str, int]]  # None while alive


class WindowLedger:
    """
    Incremental per-head global record of every window ever seen.

    Updated in-place after every snapshot (prefill + every omega generation step).
    Never recreated — only mutated.

    Access:
        ledger.entries[h]          -> dict[int, WindowEntry]  for head h
        ledger.get_all(h)          -> same dict
        ledger.get_alive(h)        -> list of alive WindowEntry for head h
        ledger.get_evicted(h)      -> list of evicted WindowEntry for head h
    """

    def __init__(self, layer_idx: int, num_heads: int) -> None:
        self.layer_idx  = layer_idx
        self.num_heads  = num_heads
        # One dict per head: window_id (int) -> WindowEntry
        self.entries: List[Dict[int, WindowEntry]] = [{} for _ in range(num_heads)]

    # ------------------------------------------------------------------
    # Update — called after every snapshot
    # ------------------------------------------------------------------

    def update(self, snapshot: AttentionSnapshot) -> None:
        """
        Incrementally update the ledger from a new snapshot.

        Rules:
          - Windows present in snapshot  → created or updated; status set to "alive",
            last_cumulative_score updated from snapshot.cumulative_scores.
          - Windows that were "alive" in the previous step but absent now
            → status set to "evicted", evicted_at = snapshot.eviction_key.
          - A window that was "evicted" and reappears (e.g. re-promoted from q-cache)
            → status restored to "alive", evicted_at cleared.
        """
        key   = snapshot.eviction_key
        zones = snapshot.zones   # list[str], length W — shared across heads

        for h in range(self.num_heads):
            head_dict    = self.entries[h]
            wids_tensor  = snapshot.window_ids[h]           # [W] int64 on CPU
            cumul_tensor = snapshot.cumulative_scores[h]    # [W] float32 on CPU
            wids_list    = wids_tensor.tolist()              # pure Python ints
            cumul_list   = cumul_tensor.tolist()             # pure Python floats

            current_set: set[int] = set(wids_list)

            # --- Pass 1: update / create entries for windows present this step ---
            for col, wid in enumerate(wids_list):
                zone  = zones[col]
                cumul = cumul_list[col]
                if wid not in head_dict:
                    head_dict[wid] = WindowEntry(
                        window_id             = wid,
                        first_seen            = key,
                        last_seen             = key,
                        current_zone          = zone,
                        last_cumulative_score = cumul,
                        status                = "alive",
                        evicted_at            = None,
                    )
                else:
                    e = head_dict[wid]
                    e.last_seen            = key
                    e.current_zone         = zone
                    e.last_cumulative_score = cumul
                    e.status               = "alive"
                    e.evicted_at           = None   # clear if it was previously evicted

            # --- Pass 2: mark previously-alive windows that disappeared as evicted ---
            for wid, entry in head_dict.items():
                if entry.status == "alive" and wid not in current_set:
                    entry.status     = "evicted"
                    entry.evicted_at = key

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_all(self, head: int) -> Dict[int, WindowEntry]:
        """All windows ever seen for the given head, keyed by window_id."""
        return self.entries[head]

    def get_alive(self, head: int) -> List[WindowEntry]:
        """Windows that are currently alive (present in the most recent snapshot)."""
        return [e for e in self.entries[head].values() if e.status == "alive"]

    def get_evicted(self, head: int) -> List[WindowEntry]:
        """Windows that have been evicted (absent from the most recent snapshot)."""
        return [e for e in self.entries[head].values() if e.status == "evicted"]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all records (called on document boundary by _clean_scores)."""
        self.entries = [{} for _ in range(self.num_heads)]


# ============================================================
# Attention Tracker — owns both snapshots and ledger
# ============================================================

class AttentionTracker:
    """
    Stores ordered AttentionSnapshot objects and maintains the WindowLedger.

    Instantiated inside STICKYKVCache_LayerWise.__init__ as self.attn_tracker.
    Recording hooks call record_prefill() / record_generation() — do not call externally.

    Access:
        attn_tracker.get_snapshot("prefill")
        attn_tracker.get_snapshot(N)
        attn_tracker.list_snapshots()
        attn_tracker.ledger.get_all(head)
        attn_tracker.ledger.get_alive(head)
        attn_tracker.ledger.get_evicted(head)
    """

    def __init__(self, layer_idx: int, num_heads: int) -> None:
        self.layer_idx  = layer_idx
        self.num_heads  = num_heads
        self.snapshots: Dict[Union[str, int], AttentionSnapshot] = {}
        self.ledger     = WindowLedger(layer_idx=layer_idx, num_heads=num_heads)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_prefill(self, snapshot: AttentionSnapshot) -> None:
        """Store prefill snapshot and update the ledger."""
        self.snapshots["prefill"] = snapshot
        self.ledger.update(snapshot)

    def record_generation(self, snapshot: AttentionSnapshot) -> None:
        """Store generation-phase snapshot and update the ledger."""
        self.snapshots[snapshot.eviction_key] = snapshot
        self.ledger.update(snapshot)

    # ------------------------------------------------------------------
    # Snapshot retrieval
    # ------------------------------------------------------------------

    def get_snapshot(self, key: Union[str, int]) -> Optional[AttentionSnapshot]:
        return self.snapshots.get(key)

    def list_snapshots(self) -> List[Union[str, int]]:
        return list(self.snapshots.keys())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all snapshots and ledger (called on document boundary)."""
        self.snapshots.clear()
        self.ledger.reset()
