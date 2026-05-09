"""
npz_io.py — Shared NPZ I/O helpers for baseline result serialization.

Replaces JSON serialization of large nested attention/window-score data
with NumPy compressed .npz format for ~10-20x file size reduction.

The load function reconstructs the same nested dict structure that the
metric consumers (calculate_window_jaccard, calculate_layer_information_retention)
expect, making the migration transparent to downstream code.
"""

import numpy as np
import os


def save_results_npz(results_list, filepath, extra_arrays=None):
    """
    Save a list of sample result dicts to a compressed .npz file.

    Expected dict schema per sample:
        metadata:
            sha256, article_index, token_count_input,
            generated_token_ids, truncation_char_index, teacher_forcing
        tracked_layers: list[int]
        tracked_heads: list[int]
        prefill_attention:           {layer_str: {head_str: [float, ...]}}
        prefill_window_scores:       {layer_str: {head_str: [[score, id], ...]}}
        generation_attention:        [step: {layer_str: {head_str: [float, ...]}}]
        generation_window_scores:    [step: {layer_str: {head_str: [[score, id], ...]}}]
        (optional) generation_attention_fresh / generation_window_scores_fresh
    """
    arrays = {}

    # Version tag — downstream scripts check this to ensure correct code was used
    arrays["__version__"] = np.frombuffer(b"v3_alive_filter", dtype=np.uint8)

    # Structural arrays (shared across samples)
    if results_list:
        arrays["tracked_layers"] = np.array(results_list[0]["tracked_layers"], dtype=np.int32)
        arrays["tracked_heads"] = np.array(results_list[0]["tracked_heads"], dtype=np.int32)

    for i, result in enumerate(results_list):
        meta = result["metadata"]

        # --- Scalar metadata ---
        arrays[f"meta_{i}_sha256"] = np.frombuffer(meta["sha256"].encode("utf-8"), dtype=np.uint8)
        arrays[f"meta_{i}_article_index"] = np.array(meta["article_index"], dtype=np.int32)
        arrays[f"meta_{i}_token_count_input"] = np.array(meta["token_count_input"], dtype=np.int32)
        arrays[f"meta_{i}_generated_token_ids"] = np.array(meta["generated_token_ids"], dtype=np.int32)
        arrays[f"meta_{i}_truncation_char_index"] = np.array(meta["truncation_char_index"], dtype=np.int32)
        arrays[f"meta_{i}_teacher_forcing"] = np.array(meta.get("teacher_forcing", True), dtype=np.bool_)

        # --- Prefill attention: {layer: {head: [scores]}} ---
        prefill_attn = result.get("prefill_attention", {})
        for l_str, heads in prefill_attn.items():
            for h_str, scores in heads.items():
                key = f"prefill_attn_{i}_L{l_str}_H{h_str}"
                arrays[key] = np.array(scores, dtype=np.float32)

        # --- Prefill window scores: {layer: {head: [[score, id], ...]}} ---
        prefill_ws = result.get("prefill_window_scores", {})
        for l_str, heads in prefill_ws.items():
            for h_str, ws_list in heads.items():
                key = f"prefill_ws_{i}_L{l_str}_H{h_str}"
                if ws_list:
                    arrays[key] = np.array(ws_list, dtype=np.float32)
                else:
                    arrays[key] = np.zeros((0, 2), dtype=np.float32)

        # --- Generation attention (per-step snapshots) ---
        gen_attn = result.get("generation_attention", [])
        for s, step_data in enumerate(gen_attn):
            for l_str, heads in step_data.items():
                for h_str, scores in heads.items():
                    key = f"gen_attn_{i}_s{s}_L{l_str}_H{h_str}"
                    arrays[key] = np.array(scores, dtype=np.float32)

        # --- Generation window scores (per-step) ---
        gen_ws = result.get("generation_window_scores", [])
        for s, step_data in enumerate(gen_ws):
            for l_str, heads in step_data.items():
                for h_str, ws_list in heads.items():
                    key = f"gen_ws_{i}_s{s}_L{l_str}_H{h_str}"
                    if ws_list:
                        arrays[key] = np.array(ws_list, dtype=np.float32)
                    else:
                        arrays[key] = np.zeros((0, 2), dtype=np.float32)

        # --- Fresh attention (sticky-only, optional) ---
        gen_attn_fresh = result.get("generation_attention_fresh", [])
        for s, step_data in enumerate(gen_attn_fresh):
            for l_str, heads in step_data.items():
                for h_str, scores in heads.items():
                    key = f"gen_attn_fresh_{i}_s{s}_L{l_str}_H{h_str}"
                    arrays[key] = np.array(scores, dtype=np.float32)

        # --- Fresh window scores (sticky-only, optional) ---
        gen_ws_fresh = result.get("generation_window_scores_fresh", [])
        for s, step_data in enumerate(gen_ws_fresh):
            for l_str, heads in step_data.items():
                for h_str, ws_list in heads.items():
                    key = f"gen_ws_fresh_{i}_s{s}_L{l_str}_H{h_str}"
                    if ws_list:
                        arrays[key] = np.array(ws_list, dtype=np.float32)
                    else:
                        arrays[key] = np.zeros((0, 2), dtype=np.float32)

    # Merge any extra arrays (e.g. tracker data) in the same pass
    if extra_arrays:
        arrays.update(extra_arrays)

    np.savez_compressed(filepath, **arrays)
    # np.savez_compressed auto-appends .npz if not present
    actual_path = filepath if filepath.endswith(".npz") else filepath + ".npz"
    size_mb = os.path.getsize(actual_path) / (1024 * 1024)
    print(f"Saved {len(results_list)} samples to {actual_path} ({size_mb:.2f} MB)")


def _parse_npy_bytes(raw_npy):
    """Parse raw .npy bytes into a numpy array, ~100x faster than np.load."""
    import re
    if len(raw_npy) < 10:
        return np.array([])
    major = raw_npy[6]
    if major == 1:
        hdr_len = int.from_bytes(raw_npy[8:10], 'little')
        hdr_start, data_off = 10, 10 + hdr_len
    elif major == 2:
        hdr_len = int.from_bytes(raw_npy[8:12], 'little')
        hdr_start, data_off = 12, 12 + hdr_len
    else:
        import io
        return np.load(io.BytesIO(raw_npy))
    hdr = raw_npy[hdr_start:data_off]
    dm = re.search(rb"'descr':\s*'([^']*)'", hdr)
    sm = re.search(rb"'shape':\s*\(([^)]*)\)", hdr)
    if not dm or not sm:
        import io
        return np.load(io.BytesIO(raw_npy))
    dtype = np.dtype(dm.group(1).decode('ascii'))
    s_str = sm.group(1).decode('ascii').strip().rstrip(',').strip()
    shape = tuple(int(x.strip()) for x in s_str.split(',') if x.strip()) if s_str else ()
    data = raw_npy[data_off:]
    n = 1
    for d in shape:
        n *= d
    if n == 0 or len(data) == 0:
        return np.zeros(shape, dtype=dtype)
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def load_results_npz(filepath, metadata_only=False, skip_attention=False):
    """
    Load a .npz file and reconstruct the same list-of-dicts structure
    that the old JSON format produced.

    Uses raw zip byte parsing + C-level zlib to bypass Python's zipfile
    per-entry overhead, achieving ~200x speedup for files with many
    small arrays (600K+ entries).

    Args:
        filepath: Path to the .npz file.
        metadata_only: If True, skip all attention/window-score data and
                       only load metadata + tracked_layers/heads.
        skip_attention: If True, skip loading attention arrays (prefill_attn,
                        gen_attn, gen_attn_fresh) but still load window_scores.
                        Use this for metrics that only need window scores.

    Returns: list[dict] with the same schema as save_results_npz input.
    """
    import io, time, struct, zlib, zipfile

    t0 = time.time()

    # ── Phase 1: Read entire file into memory ──
    with open(filepath, 'rb') as f:
        zpf = f.read()
    t_read = time.time()
    print(f"  File read: {len(zpf)/1024/1024:.1f} MB in {t_read-t0:.1f}s")

    # ── Phase 2: Build zip directory using zipfile (fast, reads central dir only) ──
    zf = zipfile.ZipFile(io.BytesIO(zpf))
    all_infos = zf.infolist()
    all_file_keys = set(info.filename[:-4] if info.filename.endswith('.npy')
                        else info.filename for info in all_infos)
    zf.close()  # close immediately — we won't use zf.read()

    # ── Phase 3: Extract arrays by reading raw zip bytes (bypass zipfile.read) ──
    data = {}
    skipped = 0
    total = len(all_infos)
    for idx, info in enumerate(all_infos):
        fname = info.filename
        key = fname[:-4] if fname.endswith('.npy') else fname

        # Filter early
        if metadata_only:
            if not (key.startswith('meta_') or key in ('tracked_layers', 'tracked_heads')):
                skipped += 1
                continue
        elif skip_attention and '_attn_' in key:
            skipped += 1
            continue

        # Read raw compressed bytes directly from memory buffer
        off = info.header_offset
        fn_len = struct.unpack_from('<H', zpf, off + 26)[0]
        ex_len = struct.unpack_from('<H', zpf, off + 28)[0]
        method = struct.unpack_from('<H', zpf, off + 8)[0]
        data_off = off + 30 + fn_len + ex_len
        comp = zpf[data_off : data_off + info.compress_size]

        if method == 0:          # stored (no compression)
            npy_bytes = comp
        elif method == 8:        # deflated
            npy_bytes = zlib.decompress(comp, -15)
        else:
            # Rare fallback
            zf2 = zipfile.ZipFile(io.BytesIO(zpf))
            npy_bytes = zf2.read(fname)
            zf2.close()

        data[key] = _parse_npy_bytes(npy_bytes)

        if (idx + 1) % 100000 == 0:
            print(f"  Extracting: {idx+1}/{total} ({100*(idx+1)//total}%)")

    del zpf  # free raw bytes

    all_keys = set(data.keys())

    t1 = time.time()
    print(f"  Loaded {len(data)} arrays (skipped {skipped}) in {t1-t0:.1f}s")

    # ── Phase 4: Reconstruct nested dict structure ──
    tracked_layers = data["tracked_layers"].tolist()
    tracked_heads = data["tracked_heads"].tolist()

    sample_indices = []
    for i in range(10000):
        if f"meta_{i}_sha256" in all_keys:
            sample_indices.append(i)
        else:
            break

    results = []
    for i in sample_indices:
        # --- Reconstruct metadata ---
        sha256_bytes = data[f"meta_{i}_sha256"].tobytes().decode("utf-8")
        gen_token_ids = data[f"meta_{i}_generated_token_ids"]
        metadata = {
            "sha256": sha256_bytes,
            "article_index": int(data[f"meta_{i}_article_index"]),
            "token_count_input": int(data[f"meta_{i}_token_count_input"]),
            "generated_token_ids": gen_token_ids.tolist(),
            "truncation_char_index": int(data[f"meta_{i}_truncation_char_index"]),
            "teacher_forcing": bool(data[f"meta_{i}_teacher_forcing"]),
        }

        # --- Fast path: metadata only ---
        if metadata_only:
            results.append({
                "metadata": metadata,
                "tracked_layers": tracked_layers,
                "tracked_heads": tracked_heads,
            })
            continue

        # --- Reconstruct prefill attention ---
        prefill_attention = {}
        prefill_window_scores = {}
        for l in tracked_layers:
            l_str = str(l)
            layer_attn = {}
            layer_ws = {}
            for h in tracked_heads:
                h_str = str(h)
                if not skip_attention:
                    attn_key = f"prefill_attn_{i}_L{l_str}_H{h_str}"
                    if attn_key in all_keys:
                        layer_attn[h_str] = data[attn_key].tolist()

                ws_key = f"prefill_ws_{i}_L{l_str}_H{h_str}"
                if ws_key in all_keys:
                    arr = data[ws_key]
                    layer_ws[h_str] = arr.tolist() if arr.size > 0 else []

            if layer_attn:
                prefill_attention[l_str] = layer_attn
            if layer_ws:
                prefill_window_scores[l_str] = layer_ws

        # --- Compute gen step count ---
        num_gen_steps = len(gen_token_ids)
        if num_gen_steps > 0:
            probe_l = str(tracked_layers[0])
            probe_h = str(tracked_heads[0])
            probe_prefix = "gen_ws" if skip_attention else "gen_attn"
            probe_key_set = all_keys if skip_attention else all_file_keys
            while num_gen_steps > 0:
                if f"{probe_prefix}_{i}_s{num_gen_steps - 1}_L{probe_l}_H{probe_h}" in probe_key_set:
                    break
                num_gen_steps -= 1

        # Check for fresh data by probing one key
        has_fresh = f"gen_ws_fresh_{i}_s0_L{str(tracked_layers[0])}_H{str(tracked_heads[0])}" in all_keys

        # --- Reconstruct generation attention + window scores ---
        generation_attention = []
        generation_window_scores = []
        for s in range(num_gen_steps):
            step_attn = {}
            step_ws = {}
            for l in tracked_layers:
                l_str = str(l)
                layer_attn = {}
                layer_ws = {}
                for h in tracked_heads:
                    h_str = str(h)
                    if not skip_attention:
                        attn_key = f"gen_attn_{i}_s{s}_L{l_str}_H{h_str}"
                        if attn_key in all_keys:
                            layer_attn[h_str] = data[attn_key].tolist()

                    ws_key = f"gen_ws_{i}_s{s}_L{l_str}_H{h_str}"
                    if ws_key in all_keys:
                        arr = data[ws_key]
                        layer_ws[h_str] = arr.tolist() if arr.size > 0 else []

                if layer_attn:
                    step_attn[l_str] = layer_attn
                if layer_ws:
                    step_ws[l_str] = layer_ws

            generation_attention.append(step_attn)
            generation_window_scores.append(step_ws)

        # --- Reconstruct fresh attention (sticky-only) ---
        generation_attention_fresh = []
        generation_window_scores_fresh = []
        if has_fresh:
            for s in range(num_gen_steps):
                step_attn = {}
                step_ws = {}
                for l in tracked_layers:
                    l_str = str(l)
                    layer_attn = {}
                    layer_ws = {}
                    for h in tracked_heads:
                        h_str = str(h)
                        if not skip_attention:
                            attn_key = f"gen_attn_fresh_{i}_s{s}_L{l_str}_H{h_str}"
                            if attn_key in all_keys:
                                layer_attn[h_str] = data[attn_key].tolist()

                        ws_key = f"gen_ws_fresh_{i}_s{s}_L{l_str}_H{h_str}"
                        if ws_key in all_keys:
                            arr = data[ws_key]
                            layer_ws[h_str] = arr.tolist() if arr.size > 0 else []

                    if layer_attn:
                        step_attn[l_str] = layer_attn
                    if layer_ws:
                        step_ws[l_str] = layer_ws

                generation_attention_fresh.append(step_attn)
                generation_window_scores_fresh.append(step_ws)

        result = {
            "metadata": metadata,
            "tracked_layers": tracked_layers,
            "tracked_heads": tracked_heads,
            "prefill_attention": prefill_attention,
            "prefill_window_scores": prefill_window_scores,
            "generation_attention": generation_attention,
            "generation_window_scores": generation_window_scores,
        }

        if generation_attention_fresh:
            result["generation_attention_fresh"] = generation_attention_fresh
            result["generation_window_scores_fresh"] = generation_window_scores_fresh

        results.append(result)

    t2 = time.time()
    print(f"Loaded {len(results)} samples from {filepath} "
          f"(total: {t2-t0:.1f}s, reconstruct: {t2-t1:.1f}s)")
    return results


def _run_round_trip_test():
    """Self-test: create synthetic data, save, reload, verify equality."""
    import tempfile

    print("Running round-trip test...")

    fake_results = [{
        "metadata": {
            "sha256": "abc123def456",
            "article_index": 7,
            "token_count_input": 100,
            "generated_token_ids": [1, 2, 3, 4, 5],
            "truncation_char_index": 42,
            "teacher_forcing": True,
        },
        "tracked_layers": [1, 5, 10],
        "tracked_heads": [0, 1, 2],
        "prefill_attention": {
            "1": {"0": [0.1, 0.2, 0.3], "1": [0.4, 0.5, 0.6], "2": [0.7, 0.8, 0.9]},
            "5": {"0": [1.0, 1.1, 1.2], "1": [1.3, 1.4, 1.5], "2": [1.6, 1.7, 1.8]},
            "10": {"0": [2.0, 2.1], "1": [2.2, 2.3], "2": [2.4, 2.5]},
        },
        "prefill_window_scores": {
            "1": {"0": [[0.5, 0.0], [0.3, 1.0]], "1": [[0.4, 0.0]], "2": []},
            "5": {"0": [[0.9, 0.0]], "1": [], "2": [[0.2, 0.0]]},
            "10": {"0": [], "1": [], "2": []},
        },
        "generation_attention": [
            {"1": {"0": [0.1, 0.2], "1": [0.3, 0.4], "2": [0.5, 0.6]},
             "5": {"0": [0.7], "1": [0.8], "2": [0.9]},
             "10": {"0": [1.0], "1": [1.1], "2": [1.2]}},
            {"1": {"0": [0.11, 0.22], "1": [0.33, 0.44], "2": [0.55, 0.66]},
             "5": {"0": [0.77], "1": [0.88], "2": [0.99]},
             "10": {"0": [1.01], "1": [1.11], "2": [1.22]}},
        ],
        "generation_window_scores": [
            {"1": {"0": [[0.1, 0.0]], "1": [], "2": [[0.3, 0.0]]},
             "5": {"0": [], "1": [], "2": []},
             "10": {"0": [], "1": [], "2": []}},
            {"1": {"0": [[0.2, 0.0]], "1": [[0.1, 1.0]], "2": []},
             "5": {"0": [], "1": [], "2": []},
             "10": {"0": [], "1": [], "2": []}},
        ],
    }]

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_results.npz")

    save_results_npz(fake_results, path)
    loaded = load_results_npz(path)

    assert len(loaded) == 1, f"Expected 1 sample, got {len(loaded)}"

    orig = fake_results[0]
    recon = loaded[0]

    # Metadata
    assert orig["metadata"]["sha256"] == recon["metadata"]["sha256"]
    assert orig["metadata"]["article_index"] == recon["metadata"]["article_index"]
    assert orig["metadata"]["generated_token_ids"] == recon["metadata"]["generated_token_ids"]
    assert orig["metadata"]["truncation_char_index"] == recon["metadata"]["truncation_char_index"]

    # Prefill attention
    for l_str in orig["prefill_attention"]:
        for h_str in orig["prefill_attention"][l_str]:
            orig_vals = orig["prefill_attention"][l_str][h_str]
            recon_vals = recon["prefill_attention"][l_str][h_str]
            assert np.allclose(orig_vals, recon_vals, atol=1e-6), \
                f"Prefill attn mismatch at L{l_str} H{h_str}"

    # Generation attention
    assert len(orig["generation_attention"]) == len(recon["generation_attention"])
    for s in range(len(orig["generation_attention"])):
        for l_str in orig["generation_attention"][s]:
            for h_str in orig["generation_attention"][s][l_str]:
                orig_vals = orig["generation_attention"][s][l_str][h_str]
                recon_vals = recon["generation_attention"][s][l_str][h_str]
                assert np.allclose(orig_vals, recon_vals, atol=1e-6), \
                    f"Gen attn mismatch at step {s} L{l_str} H{h_str}"

    # Window scores
    for s in range(len(orig["generation_window_scores"])):
        for l_str in orig["generation_window_scores"][s]:
            for h_str in orig["generation_window_scores"][s][l_str]:
                orig_ws = orig["generation_window_scores"][s][l_str][h_str]
                recon_ws = recon["generation_window_scores"][s][l_str][h_str]
                assert len(orig_ws) == len(recon_ws), \
                    f"WS length mismatch at step {s} L{l_str} H{h_str}"
                if orig_ws:
                    assert np.allclose(orig_ws, recon_ws, atol=1e-6)

    # Cleanup
    os.remove(path)
    os.rmdir(tmpdir)

    print("✅ Round-trip test PASSED — all data verified identical.")


if __name__ == "__main__":
    _run_round_trip_test()
