"""
test_kernels.py — Unit tests for Sticky KV CUDA kernels.

Validates each kernel against a pure-PyTorch reference implementation.
Run on HPC after building:
    cd csrc
    python test_kernels.py

Tests sweep across relevant parameter ranges for Llama 3.2 (1B/3B).
"""

import torch
import random
import sys

try:
    import sticky_kv_cuda
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False
    print("WARNING: sticky_kv_cuda extension not found. "
          "Build it first: cd csrc && make")
    sys.exit(1)


# ============================================================================
# Test 1: per_head_kv_gather
# ============================================================================

def test_per_head_kv_gather(num_heads, seq_len, max_kept, head_dim, dtype):
    """Test that CUDA gather matches PyTorch torch.gather exactly."""

    # Create random KV cache
    key_cache = torch.randn(
        (num_heads, seq_len, head_dim), dtype=dtype, device="cuda"
    )
    value_cache = torch.randn(
        (num_heads, seq_len, head_dim), dtype=dtype, device="cuda"
    )

    # Create random per-head indices (sorted, unique within each head)
    indices_list = []
    for h in range(num_heads):
        # Sample max_kept unique positions from [0, seq_len)
        actual_kept = min(max_kept, seq_len)
        idx = torch.randperm(seq_len, device="cuda")[:actual_kept].sort()[0]
        # Pad if necessary
        if len(idx) < max_kept:
            pad = idx[-1:].expand(max_kept - len(idx))
            idx = torch.cat([idx, pad])
        indices_list.append(idx)
    indices = torch.stack(indices_list, dim=0).to(torch.int32)

    # PyTorch reference
    gather_idx = indices.long().unsqueeze(-1).expand(-1, -1, head_dim)
    ref_key = torch.gather(key_cache, 1, gather_idx)
    ref_val = torch.gather(value_cache, 1, gather_idx)

    # CUDA kernel
    key_cache_c = key_cache.contiguous()
    value_cache_c = value_cache.contiguous()
    indices_c = indices.contiguous()
    cuda_results = sticky_kv_cuda.per_head_kv_gather(
        key_cache_c, value_cache_c, indices_c
    )
    cuda_key = cuda_results[0]
    cuda_val = cuda_results[1]

    # Bit-exact comparison (gather is a copy, no arithmetic)
    assert torch.equal(ref_key, cuda_key), (
        f"Key mismatch! max_diff={torch.max(torch.abs(ref_key - cuda_key)).item()}"
    )
    assert torch.equal(ref_val, cuda_val), (
        f"Value mismatch! max_diff={torch.max(torch.abs(ref_val - cuda_val)).item()}"
    )

    print(f"  ✓ per_head_kv_gather: heads={num_heads}, seq={seq_len}, "
          f"kept={max_kept}, dim={head_dim}, dtype={dtype}")


# ============================================================================
# Test 2: window_score_reduce
# ============================================================================

def test_window_score_reduce(num_heads, num_windows, omega):
    """Test that CUDA window reduction matches PyTorch view+sum."""

    num_tokens = num_windows * omega
    scores = torch.randn(
        (num_heads, num_tokens), dtype=torch.float32, device="cuda"
    )

    # PyTorch reference
    ref_out = scores.view(num_heads, num_windows, omega).sum(dim=2)

    # CUDA kernel
    scores_c = scores.contiguous()
    cuda_out = sticky_kv_cuda.window_score_reduce(scores_c, num_windows, omega)

    # Tolerance-based comparison (float32 reduction)
    max_diff = torch.max(torch.abs(ref_out - cuda_out)).item()
    assert max_diff < 1e-4, (
        f"Window score mismatch! max_diff={max_diff}"
    )

    print(f"  ✓ window_score_reduce: heads={num_heads}, "
          f"windows={num_windows}, omega={omega}, max_diff={max_diff:.2e}")


# ============================================================================
# Test 3: vote_accumulate
# ============================================================================

def test_vote_accumulate(num_heads, seq_len, max_context):
    """Test that CUDA vote accumulation matches PyTorch += ."""

    running_votes = torch.randn(
        (num_heads, max_context), dtype=torch.float32, device="cuda"
    )
    new_scores = torch.randn(
        (num_heads, seq_len), dtype=torch.float32, device="cuda"
    )

    # PyTorch reference
    ref_votes = running_votes.clone()
    ref_votes[:, :seq_len] += new_scores[:, :seq_len]

    # CUDA kernel (in-place)
    cuda_votes = running_votes.clone().contiguous()
    new_scores_c = new_scores.contiguous()
    sticky_kv_cuda.vote_accumulate(cuda_votes, new_scores_c, seq_len)

    # Tolerance-based comparison
    max_diff = torch.max(torch.abs(ref_votes - cuda_votes)).item()
    assert max_diff < 1e-6, (
        f"Vote accumulation mismatch! max_diff={max_diff}"
    )

    # Verify elements beyond seq_len are untouched
    if max_context > seq_len:
        beyond_diff = torch.max(
            torch.abs(running_votes[:, seq_len:] - cuda_votes[:, seq_len:])
        ).item()
        assert beyond_diff == 0.0, (
            f"Vote accumulation modified elements beyond seq_len!"
        )

    print(f"  ✓ vote_accumulate: heads={num_heads}, "
          f"seq={seq_len}, max_ctx={max_context}, max_diff={max_diff:.2e}")


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_edge_cases():
    """Test boundary conditions."""

    # Empty gather (max_kept = 0) — should not crash
    key = torch.randn((8, 100, 64), dtype=torch.bfloat16, device="cuda")
    val = torch.randn((8, 100, 64), dtype=torch.bfloat16, device="cuda")
    idx = torch.empty((8, 0), dtype=torch.int32, device="cuda")
    result = sticky_kv_cuda.per_head_kv_gather(key, val, idx)
    assert result[0].shape == (8, 0, 64)
    assert result[1].shape == (8, 0, 64)
    print("  ✓ edge: empty gather (max_kept=0)")

    # Zero windows
    scores = torch.randn((8, 50), dtype=torch.float32, device="cuda")
    out = sticky_kv_cuda.window_score_reduce(scores, 0, 5)
    assert out.shape == (8, 0)
    print("  ✓ edge: zero windows")

    # Zero seq_len accumulation
    votes = torch.randn((8, 1000), dtype=torch.float32, device="cuda")
    ns = torch.randn((8, 10), dtype=torch.float32, device="cuda")
    votes_before = votes.clone()
    sticky_kv_cuda.vote_accumulate(votes, ns, 0)
    assert torch.equal(votes, votes_before)
    print("  ✓ edge: zero seq_len accumulation")

    # Single element per head
    key = torch.randn((4, 10, 128), dtype=torch.float16, device="cuda")
    val = torch.randn((4, 10, 128), dtype=torch.float16, device="cuda")
    idx = torch.tensor([[5], [3], [9], [0]], dtype=torch.int32, device="cuda")
    result = sticky_kv_cuda.per_head_kv_gather(key, val, idx)
    for h in range(4):
        assert torch.equal(result[0][h, 0], key[h, idx[h, 0].long()])
        assert torch.equal(result[1][h, 0], val[h, idx[h, 0].long()])
    print("  ✓ edge: single element gather")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    print("=" * 60)
    print("Sticky KV CUDA Kernel Tests")
    print("=" * 60)

    # Seed for reproducibility
    for seed in [42, 123, 777]:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        print(f"\n--- Seed: {seed} ---")

        # Test per_head_kv_gather across parameter sweep
        print("\n[Kernel 1] per_head_kv_gather:")
        for num_heads in [4, 8]:
            for head_dim in [64, 128]:
                for seq_len in [256, 1024, 4096]:
                    for max_kept in [1, 50, 200, min(500, seq_len)]:
                        for dtype in [torch.bfloat16, torch.float16]:
                            test_per_head_kv_gather(
                                num_heads, seq_len, max_kept, head_dim, dtype
                            )

        # Test window_score_reduce
        print("\n[Kernel 2] window_score_reduce:")
        for num_heads in [4, 8]:
            for omega in [1, 5, 16, 32]:
                for num_windows in [1, 10, 100, 500]:
                    test_window_score_reduce(num_heads, num_windows, omega)

        # Test vote_accumulate
        print("\n[Kernel 3] vote_accumulate:")
        for num_heads in [4, 8]:
            for seq_len in [1, 100, 1024, 4096]:
                for max_context in [seq_len, seq_len + 1000, 131072]:
                    test_vote_accumulate(num_heads, seq_len, max_context)

    # Edge cases
    print("\n[Edge Cases]:")
    test_edge_cases()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
