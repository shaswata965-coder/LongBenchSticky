"""
sticky_kv_cuda_ops.py — Python wrapper for Sticky KV CUDA kernels.

Provides three optimized operations with automatic fallback to pure PyTorch
if the CUDA extension is not compiled/available.

Usage:
    from sticky_kv_cuda_ops import per_head_kv_gather, window_score_reduce, vote_accumulate

Functions:
    per_head_kv_gather(key_cache, value_cache, indices)
        -> (key_out, value_out)

    window_score_reduce(scores, num_windows, omega)
        -> window_scores

    vote_accumulate(running_votes, new_scores, seq_len)
        -> None (in-place)
"""

import torch

# ============================================================================
# Try to import the compiled CUDA extension
# ============================================================================

_HAS_CUDA_EXT = False
_sticky_kv_cuda = None

try:
    import sticky_kv_cuda as _sticky_kv_cuda
    _HAS_CUDA_EXT = True
except ImportError:
    pass

# Print status once on import
if _HAS_CUDA_EXT:
    print("[sticky_kv_cuda_ops] CUDA extension loaded — using accelerated kernels", flush=True)
else:
    print("[sticky_kv_cuda_ops] CUDA extension not found — using PyTorch fallback", flush=True)


# ============================================================================
# Operation 1: Per-Head KV Cache Gather (Eviction)
# ============================================================================

def per_head_kv_gather(key_cache, value_cache, indices):
    """
    Gather KV cache entries per-head using either CUDA kernel or PyTorch fallback.

    Args:
        key_cache:   [num_heads, seq_len, head_dim] — K cache tensor
        value_cache: [num_heads, seq_len, head_dim] — V cache tensor
        indices:     [num_heads, max_kept] — per-head gather indices (int32 or int64)

    Returns:
        (key_out, value_out) — each [num_heads, max_kept, head_dim]
    """
    if _HAS_CUDA_EXT and key_cache.is_cuda:
        # Ensure contiguity and correct dtype for indices
        key_c = key_cache.contiguous()
        val_c = value_cache.contiguous()
        idx_c = indices.to(torch.int32).contiguous()

        results = _sticky_kv_cuda.per_head_kv_gather(key_c, val_c, idx_c)
        return results[0], results[1]
    else:
        return _per_head_kv_gather_pytorch(key_cache, value_cache, indices)


def _per_head_kv_gather_pytorch(key_cache, value_cache, indices):
    """Pure PyTorch fallback for per_head_kv_gather."""
    head_dim = key_cache.shape[-1]
    gather_idx = indices.long().unsqueeze(-1).expand(-1, -1, head_dim)
    key_out = torch.gather(key_cache, 1, gather_idx)
    value_out = torch.gather(value_cache, 1, gather_idx)
    return key_out, value_out


# ============================================================================
# Operation 2: Window Score Reduction
# ============================================================================

def window_score_reduce(scores, num_windows, omega):
    """
    Reduce attention scores into per-window sums.

    Replaces:  scores.view(num_heads, num_windows, omega).sum(dim=2)

    Args:
        scores:      [num_heads, num_tokens] — float32 attention scores
        num_windows: int — number of windows
        omega:       int — window size

    Returns:
        [num_heads, num_windows] — float32 per-window scores
    """
    if _HAS_CUDA_EXT and scores.is_cuda:
        scores_c = scores.to(torch.float32).contiguous()
        return _sticky_kv_cuda.window_score_reduce(scores_c, num_windows, omega)
    else:
        return _window_score_reduce_pytorch(scores, num_windows, omega)


def _window_score_reduce_pytorch(scores, num_windows, omega):
    """Pure PyTorch fallback for window_score_reduce."""
    num_heads = scores.shape[0]
    # Only take the first num_windows * omega tokens
    usable = scores[:, :num_windows * omega]
    return usable.view(num_heads, num_windows, omega).sum(dim=2).to(torch.float32)


# ============================================================================
# Operation 3: Vote Accumulation
# ============================================================================

def vote_accumulate(running_votes, new_scores, seq_len):
    """
    Accumulate generation-step attention votes in-place.

    Replaces:  running_votes[:, :seq_len] += new_scores[:, :seq_len]

    Args:
        running_votes: [num_heads, max_context] — float32, modified in-place
        new_scores:    [num_heads, score_len]   — any dtype (converted to float32)
        seq_len:       int — number of positions to accumulate

    Returns:
        None (running_votes is modified in-place)
    """
    if _HAS_CUDA_EXT and running_votes.is_cuda:
        # Ensure float32 and contiguity
        new_f32 = new_scores[:, :seq_len].to(torch.float32).contiguous()

        # running_votes is a registered buffer allocated via torch.zeros(),
        # so it is always contiguous. Assert rather than silently copy back.
        assert running_votes.is_contiguous(), (
            "running_votes must be contiguous for in-place CUDA kernel"
        )
        _sticky_kv_cuda.vote_accumulate(running_votes, new_f32, seq_len)
    else:
        _vote_accumulate_pytorch(running_votes, new_scores, seq_len)


def _vote_accumulate_pytorch(running_votes, new_scores, seq_len):
    """Pure PyTorch fallback for vote_accumulate."""
    running_votes[:, :seq_len] += new_scores[:, :seq_len].float()
