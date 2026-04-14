"""
debug_pratio_bugs.py
====================
Demonstrates 4 logical inconsistencies in p_ratio mode (LOCAL_NUM_TOKENS
commented out, P_RATIO=50).

Uses synthetic data — no model weights or .npz files needed.
Run from project root:  python debug_pratio_bugs.py
"""
import torch
import numpy as np


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────
def header(title):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def ok(msg):   print(f"  ✅ {msg}")
def fail(msg): print(f"  ❌ {msg}")
def info(msg): print(f"     {msg}")


# ────────────────────────────────────────────────────────────────────────
# BUG 1 — P_RATIO wiring
# ────────────────────────────────────────────────────────────────────────
def test_bug1():
    header("BUG 1: P_RATIO wiring — does the cache read sticky_config.P_RATIO?")

    import sticky_config as sc
    from sticky_kv_logic_cummulative import STICKYKVCache_LayerWise

    sc_p = getattr(sc, "P_RATIO", "NOT SET")

    # Create a cache and check what local_cache_ratio it actually got
    class Cfg:
        max_position_embeddings = 131072
    cache = STICKYKVCache_LayerWise(
        p_ratio=999, r_ratio=50, start_idx=0,  # intentionally wrong p_ratio
        num_heads=2, layer_idx=0, config=Cfg(),
    )
    actual = cache.local_cache_ratio

    print(f"  sticky_config.P_RATIO          = {sc_p}")
    print(f"  cache.local_cache_ratio        = {actual}")
    print(f"  (constructor was given p_ratio=999 as a decoy)")
    print()

    if sc_p != "NOT SET" and actual == sc_p:
        ok(f"Cache correctly reads P_RATIO={sc_p} from sticky_config (ignores constructor arg)")
        return False  # bug fixed
    elif sc_p == "NOT SET":
        ok(f"P_RATIO not in sticky_config — cache fell back to constructor arg ({actual})")
        return False
    else:
        fail(f"Cache has {actual}, expected {sc_p} from sticky_config")
        return True


# ────────────────────────────────────────────────────────────────────────
# BUG 2-4 common setup
# ────────────────────────────────────────────────────────────────────────
def build_cache_and_prefill():
    """Return (cache, prefill_result, constants)."""
    import sticky_config as sc
    from sticky_kv_logic_cummulative import STICKYKVCache_LayerWise

    OMEGA     = sc.OMEGA            # 5
    SINK      = sc.SINK_TOKENS      # 0
    NUM_HEADS = 2
    HEAD_DIM  = 16
    PROMPT    = 500                  # 100 complete windows of size 5
    P_RATIO   = 50
    R_RATIO   = 50

    # Use same device the cache buffers will live on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Cfg:
        max_position_embeddings = 131072

    cache = STICKYKVCache_LayerWise(
        p_ratio=P_RATIO, r_ratio=R_RATIO, start_idx=0,
        num_heads=NUM_HEADS, layer_idx=0, config=Cfg(),
    )
    cache.tracking_flag = False          # simplify — no ledger

    # ── KV tensors (on same device as cache buffers) ──
    k = torch.randn(1, NUM_HEADS, PROMPT, HEAD_DIM, device=device)
    v = torch.randn(1, NUM_HEADS, PROMPT, HEAD_DIM, device=device)

    # ── Synthetic attention (NxN, softmax-normalised) ──
    # Boost END windows (80-99) → forces top-K sticky to overlap with
    # the local range, which starts around window 63 for local_num≈37.
    attn = torch.rand(1, NUM_HEADS, PROMPT, PROMPT, device=device) * 0.001
    for h in range(NUM_HEADS):
        attn[0, h, :, 400:500] += 0.50     # windows 80-99 very high
        attn[0, h, :,   0: 90] += 0.20     # windows  0-17 medium
    attn = attn / attn.sum(dim=-1, keepdim=True)

    # ── Prefill ──
    result = cache((k, v), attn)

    consts = dict(OMEGA=OMEGA, SINK=SINK, NUM_HEADS=NUM_HEADS,
                  HEAD_DIM=HEAD_DIM, PROMPT=PROMPT,
                  num_w=PROMPT // OMEGA)

    return cache, result, consts


# ────────────────────────────────────────────────────────────────────────
# BUG 2 — Physical↔Logical mapping corruption from sticky-local overlap
# ────────────────────────────────────────────────────────────────────────
def test_bug2(cache, result, C):
    header("BUG 2: Sticky↔Local overlap corrupts physical↔logical mapping")

    num_w   = C["num_w"]     # 100
    phys    = result[0].shape[2]
    phys_w  = phys // C["OMEGA"]

    local_lo = max(0, num_w - cache.local_num)
    local_hi = num_w
    local_set = set(range(local_lo, local_hi))

    print(f"  Budget:  k_windows={cache.k_windows} sticky, local_num={cache.local_num} local")
    print(f"  Prompt:  {C['PROMPT']} tokens  →  {num_w} logical windows")
    print(f"  Post-eviction physical cache: {phys} tokens  ({phys_w} windows)")
    print(f"  Local window ID range: [{local_lo}, {local_hi})")
    print()

    bug_found = False
    for h in range(C["NUM_HEADS"]):
        ws     = cache.window_scores[h, :cache.k_windows, 1]
        valid  = ~torch.isnan(ws)
        sticky = set(ws[valid].long().tolist())
        overlap = sorted(sticky & local_set)

        non_overlap = sorted(sticky - local_set)
        print(f"  Head {h}:")
        print(f"    Sticky IDs (top-{cache.k_windows}):      {sorted(sticky)}")
        print(f"    Overlap with local [{local_lo},{local_hi}): {overlap}")

        if overlap:
            bug_found = True
            fail(f"{len(overlap)} sticky windows sit physically in the local zone")
            info("After dedup+sort they have HIGH physical positions.")
            info("But window_scores lists them as low-index competing entries.")
            print()

            # which index in window_scores does the first overlapper occupy?
            id_list = ws[valid].long().tolist()
            ov_id   = overlap[0]
            ws_idx  = id_list.index(ov_id)

            info("Concrete collision during generation review:")
            info(f"  window_scores entry [{h}, {ws_idx}] → logical ID {ov_id}")
            info(f"  win_scores[{h}, {ws_idx}]           → attention for physical window {ws_idx}")

            if ws_idx < len(non_overlap):
                phys_log_id = non_overlap[ws_idx]
                info(f"  Physical window {ws_idx} = logical ID {phys_log_id} (non-overlapping sticky)")
                info(f"  ➜ Attention for window {phys_log_id} + history for window {ov_id}")
                fail("CROSS-CONTAMINATION between two different windows")
            else:
                info(f"  Physical window {ws_idx} has unknown identity → mapping corrupt")
        else:
            ok("No overlap this run")
        print()

    return bug_found


# ────────────────────────────────────────────────────────────────────────
# BUG 3 — last_id_val off by 2 (stale -2*omega)
# ────────────────────────────────────────────────────────────────────────
def test_bug3():
    header("BUG 3: last_id_val — verify correct window transitions into competition")
    print("  (Runs isolated cache + prefill + gen cycle, observes actual behavior)")
    print()

    import sticky_config as sc
    from sticky_kv_logi import STICKYKVCache_LayerWise

    OMEGA     = sc.OMEGA
    SINK      = sc.SINK_TOKENS
    NUM_HEADS = 2
    HEAD_DIM  = 16
    PROMPT    = 500
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Cfg:
        max_position_embeddings = 131072

    cache = STICKYKVCache_LayerWise(
        p_ratio=50, r_ratio=50, start_idx=0,
        num_heads=NUM_HEADS, layer_idx=0, config=Cfg(),
    )
    cache.tracking_flag = False

    # Prefill with synthetic attention (boost end windows for overlap)
    k = torch.randn(1, NUM_HEADS, PROMPT, HEAD_DIM, device=device)
    v = torch.randn(1, NUM_HEADS, PROMPT, HEAD_DIM, device=device)
    attn = torch.rand(1, NUM_HEADS, PROMPT, PROMPT, device=device) * 0.001
    for h in range(NUM_HEADS):
        attn[0, h, :, 400:500] += 0.50
        attn[0, h, :,   0: 90] += 0.20
    attn = attn / attn.sum(dim=-1, keepdim=True)
    result = cache((k, v), attn)

    print(f"  Prefill done: {cache.num_of_tokens_without_eviction} tokens seen")
    print(f"  Budget: k_windows={cache.k_windows}, local_num={cache.local_num}")
    print(f"  Physical cache: {result[0].shape[2]} tokens")

    # Snapshot local_history BEFORE generation
    lh_before = cache.local_history.clone()

    # Run exactly OMEGA gen steps to trigger one review cycle
    past_kv = result
    for step in range(OMEGA):
        new_k = torch.randn(1, NUM_HEADS, 1, HEAD_DIM, device=device)
        new_v = torch.randn(1, NUM_HEADS, 1, HEAD_DIM, device=device)
        past_kv = (torch.cat([past_kv[0], new_k], dim=2),
                   torch.cat([past_kv[1], new_v], dim=2))
        seq = past_kv[0].shape[2]
        gen_attn = torch.rand(1, NUM_HEADS, 1, seq, device=device)
        gen_attn = gen_attn / gen_attn.sum(dim=-1, keepdim=True)
        out = cache(past_kv, gen_attn)
        if out is not None:
            past_kv = out

    # Observe: which local_history slot was consumed (zeroed) by the review?
    lh_after = cache.local_history.clone()
    consumed_mask = (lh_before.sum(dim=0) > 0) & (lh_after.sum(dim=0) == 0)
    consumed = torch.where(consumed_mask)[0].tolist()

    # Ground truth: the expected transitioning window ID
    # = total_logical_windows - local_num
    total_tok = cache.num_of_tokens_without_eviction
    total_lw  = total_tok // OMEGA
    expected  = total_lw - cache.local_num

    print(f"\n  After {OMEGA} gen tokens: {total_tok} tokens → {total_lw} logical windows")
    print(f"  Local zone = [{total_lw - cache.local_num}, {total_lw - 1}]")
    print(f"  Expected transitioning window ID = {expected}")
    print(f"  local_history slots actually consumed (zeroed): {consumed}")
    print()

    if expected in consumed:
        ok(f"Correct window {expected} transitioned into competition")
        return False   # bug fixed
    elif consumed:
        fail(f"Window(s) {consumed} consumed, but expected {expected}")
        info("The code is reading the wrong local_history slot.")
        return True    # bug present
    else:
        info("No slots consumed — review may not have had enough competing windows")
        return False


# ────────────────────────────────────────────────────────────────────────
# BUG 4 — local_history indexed by physical position (gen) vs logical (prefill)
# ────────────────────────────────────────────────────────────────────────
def test_bug4(cache, result, C):
    header("BUG 4: local_history uses physical indices (gen) vs logical (prefill)")

    OMEGA = C["OMEGA"]
    SINK  = C["SINK"]
    num_w = C["num_w"]
    phys  = result[0].shape[2]
    phys_w = phys // OMEGA

    # What prefill seeded
    seeded = torch.where(cache.local_history[0] > 0)[0]
    lo_seed, hi_seed = seeded.min().item(), seeded.max().item()
    print(f"  Prefill seeded local_history[{lo_seed}..{hi_seed}]  (LOGICAL window IDs)")

    # What generation WILL use
    gen_tok = OMEGA
    sim_seq = phys + gen_tok
    ltc     = cache.local_num * OMEGA
    ncw     = max(1, (sim_seq - SINK - ltc) // OMEGA)
    ls      = SINK + ncw * OMEGA
    lid_s   = (ls - SINK) // OMEGA          # physical-based index
    lw      = (sim_seq - ls) // OMEGA
    lid_e   = lid_s + lw

    print(f"  Generation review (seq_len={sim_seq}):")
    print(f"    num_competing_windows = {ncw}")
    print(f"    local zone physical start = position {ls}")
    print(f"    local_id_start (PHYSICAL) = {lid_s}")
    print(f"    Will write to local_history[{lid_s}:{lid_e}]")
    print()
    print(f"  Physical windows after compaction = {phys_w}")
    print(f"  Logical windows from prompt       = {num_w}")

    if phys_w != num_w:
        print()
        fail(f"Physical ({phys_w}) ≠ Logical ({num_w}) — index spaces diverge")
        info(f"local_history[{lid_s}] was seeded during prefill with logical window {lid_s}'s score")
        info(f"Generation adds physical window {lid_s}'s attention to the SAME slot")
        info(f"These are DIFFERENT windows → cumulative score corrupted")
        return True
    else:
        ok("Physical = logical this run (no compaction) — bug latent")
        return False


# ────────────────────────────────────────────────────────────────────────
# LIVE TEST — Actually run omega generation steps and inspect state
# ────────────────────────────────────────────────────────────────────────
def test_live_generation(cache, result, C):
    header(f"LIVE TEST: Run {C['OMEGA']} gen steps → trigger review cycle")

    OMEGA     = C["OMEGA"]
    NUM_HEADS = C["NUM_HEADS"]
    HEAD_DIM  = C["HEAD_DIM"]

    past_kv = result
    ltc     = cache.local_num * OMEGA

    # Snapshots before generation
    lh_before = cache.local_history.clone()
    phys_before = past_kv[0].shape[2]

    print(f"  Pre-gen physical cache: {phys_before} tokens")
    print(f"  tokens_since_last_review = {cache.tokens_since_last_review}")
    print()

    device = past_kv[0].device

    for step in range(OMEGA):
        new_k = torch.randn(1, NUM_HEADS, 1, HEAD_DIM, device=device)
        new_v = torch.randn(1, NUM_HEADS, 1, HEAD_DIM, device=device)
        past_kv = (torch.cat([past_kv[0], new_k], dim=2),
                   torch.cat([past_kv[1], new_v], dim=2))

        seq = past_kv[0].shape[2]
        gen_attn = torch.rand(1, NUM_HEADS, 1, seq, device=device)
        gen_attn = gen_attn / gen_attn.sum(dim=-1, keepdim=True)

        out = cache(past_kv, gen_attn)
        if out is not None:
            past_kv = out

        review_fired = (step + 1 == OMEGA)
        status = "← REVIEW FIRED" if review_fired else ""
        print(f"  Step {step}: seq_len={past_kv[0].shape[2]}  "
              f"tokens_since_last_review={cache.tokens_since_last_review}  {status}")

    print()
    print(f"  Post-gen physical cache: {past_kv[0].shape[2]} tokens")

    # Observe what actually changed in local_history
    lh_after = cache.local_history.clone()
    diff = (lh_after - lh_before).abs().sum(dim=0)
    changed  = torch.where(diff > 1e-8)[0].tolist()

    consumed_mask = (lh_before.sum(dim=0) > 0) & (lh_after.sum(dim=0) == 0)
    consumed = torch.where(consumed_mask)[0].tolist()

    added_mask = (lh_before.sum(dim=0) == 0) & (lh_after.sum(dim=0) > 0)
    added = torch.where(added_mask)[0].tolist()

    print(f"  local_history slots modified:          {changed[:20]}{'...' if len(changed)>20 else ''}")
    print(f"  local_history slots consumed (zeroed): {consumed}")
    print(f"  local_history slots newly written:     {added[:20]}{'...' if len(added)>20 else ''}")

    # Post-review window_scores
    for h in range(NUM_HEADS):
        ws_v = ~torch.isnan(cache.window_scores[h, :, 1])
        ids  = cache.window_scores[h, ws_v, 1].long().tolist()
        print(f"  Head {h} surviving IDs after gen review: {ids}")


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Sticky KV  ·  P_RATIO Mode Bug Reproduction                      ║")
    print("║  Config: P_RATIO=50, R_RATIO=50, OMEGA=5, SINK_TOKENS=0           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    bugs = {}

    # Bug 1 — wiring
    bugs[1] = test_bug1()

    # Shared setup for bugs 2-4
    cache, result, C = build_cache_and_prefill()

    # Bug 2 — overlap
    bugs[2] = test_bug2(cache, result, C)

    # Bug 3 — last_id_val (runs its own isolated cache + gen cycle)
    bugs[3] = test_bug3()

    # Bug 4 — local_history index space
    bugs[4] = test_bug4(cache, result, C)

    # Live gen cycle
    test_live_generation(cache, result, C)

    # Verdict
    header("SUMMARY")
    labels = {
        1: "P_RATIO config wiring",
        2: "Phys↔Logic mapping (sticky-local overlap)",
        3: "last_id_val off-by-2 (-2·omega stale)",
        4: "local_history index space (phys vs logical)",
    }
    total = 0
    for i in sorted(bugs):
        status = "❌ CONFIRMED" if bugs[i] else "⚠️  Latent / not triggered this run"
        if bugs[i]:
            total += 1
        print(f"  Bug {i}: {labels[i]:50s} {status}")
    print()
    print(f"  Confirmed: {total}/4")
    if total > 0:
        print()
        print("  ROOT CAUSE: After prefill dedup-compaction, physical positions")
        print("  diverge from logical window IDs. The generation code assumes")
        print("  they are identical. With LOCAL_NUM_TOKENS=0 (local_num=0),")
        print("  there is no overlap and no compaction → bugs are latent.")
        print("  Enabling P_RATIO (local_num > 0) activates all of them.")
    print()


if __name__ == "__main__":
    main()
