"""
Sticky CUDA Kernels — JIT-compiled fused operations for sticky KV cache.

Provides GPU-accelerated replacements for hot-path PyTorch operations:
  1. fused_scoreboard_scatter — replaces mask + scatter_add_
  2. fused_quantize_k/v_int8 — replaces per-window quantization + packing
  3. fused_dequantize_int8 — replaces dequantization
  4. fused_eviction_copy_* — replaces physical eviction gather/scatter

Falls back to pure-PyTorch implementations if CUDA compilation fails OR if
the tensor is too small to amortize kernel-launch overhead (~5-20 µs fixed cost).
Threshold tuning:
  - Qasper avg context ~3-8K tokens → compressed_len ~400-1000 → uses PyTorch
  - NarrativeQA / BookSum >20K tokens → compressed_len >2500 → uses CUDA
"""

import os
import torch

_cuda_ops = None
_load_attempted = False


def _try_load():
    """Attempt to JIT-compile the CUDA extension once."""
    global _cuda_ops, _load_attempted
    if _load_attempted:
        return _cuda_ops
    _load_attempted = True
    try:
        from torch.utils.cpp_extension import load
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        _cuda_ops = load(
            name="sticky_cuda_ops",
            sources=[os.path.join(kernel_dir, "sticky_kernels.cu")],
            verbose=False,
        )
        print("[StickyKernels] CUDA kernels compiled and loaded successfully.")
    except Exception as e:
        print(f"[StickyKernels] CUDA compilation failed, using PyTorch fallbacks: {e}")
        _cuda_ops = None
    return _cuda_ops


def is_available():
    """Check if CUDA kernels are available."""
    return _try_load() is not None


# ---------------------------------------------------------------------------
# Pre-load at module import time.
# All public functions reference _OPS directly — zero per-call overhead.
# ---------------------------------------------------------------------------
_OPS = _try_load()

# Minimum element counts below which kernel-launch overhead exceeds compute savings.
# At 256 threads/block and ~10 µs launch latency on A100, break-even ≈ 8K float32 ops.
_SCATTER_THRESHOLD = 8_192    # votes.numel() — 8 heads × 1024 compressed tokens
_EVICT_THRESHOLD   = 16_384   # H × tokens × D elements moved per eviction kernel


# ============================================================
# 1. Scoreboard Scatter
# ============================================================

def scoreboard_scatter(votes, logical_ids, max_windows, out=None):
    """
    Fused scoreboard scatter: routes votes to window scoreboard via logical IDs.

    Args:
        votes:       [H, compressed_len] float32
        logical_ids: [H, compressed_len] long
        max_windows: int
        out:         optional pre-allocated [H, max_windows] float32 buffer.
                     If provided, it is zeroed in-place and used as output
                     (avoids cudaMalloc on every call).
    Returns:
        scoreboard: [H, max_windows] float32  (== out if out was provided)
    """
    if _OPS is not None and votes.numel() >= _SCATTER_THRESHOLD:
        if out is not None:
            # Zero only the live buffer — no cudaMalloc, no memset on a fresh tensor
            out.zero_()
            _OPS.fused_scoreboard_scatter_inplace(
                votes.contiguous(), logical_ids.contiguous(), out
            )
            return out
        return _OPS.fused_scoreboard_scatter(
            votes.contiguous(), logical_ids.contiguous(), max_windows
        )

    # PyTorch fallback — also uses out buffer when provided to avoid torch.zeros alloc
    H = votes.shape[0]
    device = votes.device
    is_chunk_token = (logical_ids >= 0) & (logical_ids < max_windows)
    routed_votes = torch.where(is_chunk_token, votes, torch.zeros_like(votes))
    safe_ids = torch.where(is_chunk_token, logical_ids, torch.zeros_like(logical_ids)).long()
    if out is not None:
        scoreboard = out
        scoreboard.zero_()
    else:
        scoreboard = torch.zeros((H, max_windows), device=device, dtype=torch.float32)
    scoreboard.scatter_add_(1, safe_ids, routed_votes)
    return scoreboard


def scoreboard_scatter_new_tokens(
    votes_slice, scoreboard,
    num_tokens_without_eviction, omega, sink_tokens,
    compressed_len, seq_len
):
    """
    Scatter votes for new tokens (not yet in logical_id_map) into scoreboard.

    Args:
        votes_slice: [H, n_new] float32
        scoreboard:  [H, max_windows] float32 — mutated in-place
    """
    if _OPS is not None and votes_slice.numel() >= _SCATTER_THRESHOLD:
        _OPS.fused_scoreboard_scatter_new_tokens(
            votes_slice.contiguous(), scoreboard,
            int(num_tokens_without_eviction), omega, sink_tokens,
            compressed_len, seq_len
        )
        return

    # PyTorch fallback
    n_new = votes_slice.shape[1]
    device = votes_slice.device
    H = votes_slice.shape[0]
    js = torch.arange(n_new, device=device, dtype=torch.long)
    raw_new_lids = (num_tokens_without_eviction - omega + js - sink_tokens) // omega
    valid_new = (raw_new_lids >= 0) & (raw_new_lids < scoreboard.shape[1])
    if valid_new.any():
        scoreboard.scatter_add_(
            1,
            raw_new_lids[valid_new].unsqueeze(0).expand(H, -1),
            votes_slice[:, valid_new]
        )


# ============================================================
# 2. Quantization
# ============================================================

def quantize_k_int8(tensor):
    """
    Quantize K cache per-channel per-window with RoPE-paired dimension tying.

    Args:
        tensor: [H, W, omega, D] float16/bfloat16
    Returns:
        (quant [H,W,omega,D] uint8, scale [H,W,1,D] fp16, zp [H,W,1,D] fp16)
    """
    if _OPS is not None and tensor.dtype in (torch.float16, torch.bfloat16) and tensor.is_cuda:
        result = _OPS.fused_quantize_k_int8(tensor.contiguous())
        return result[0], result[1], result[2]

    # PyTorch fallback — exact replica of STICKYKVCache_LayerWise._quantize_k_per_window
    t_min = tensor.amin(dim=2, keepdim=True)
    t_max = tensor.amax(dim=2, keepdim=True)
    half_d = tensor.shape[-1] // 2
    t_min_h1, t_min_h2 = t_min[..., :half_d], t_min[..., half_d:]
    t_max_h1, t_max_h2 = t_max[..., :half_d], t_max[..., half_d:]
    t_min_tied = torch.min(t_min_h1, t_min_h2)
    t_max_tied = torch.max(t_max_h1, t_max_h2)
    t_min = torch.cat([t_min_tied, t_min_tied], dim=-1)
    t_max = torch.cat([t_max_tied, t_max_tied], dim=-1)
    scale = torch.clamp((t_max - t_min) / 255.0, min=1e-8)
    quantized = torch.round((tensor - t_min) / scale).clamp(0, 255).to(torch.uint8)
    return quantized, scale.to(tensor.dtype), t_min.to(tensor.dtype)


def quantize_v_int8(tensor):
    """
    Quantize V cache per-token per-window.

    Args:
        tensor: [H, W, omega, D] float16/bfloat16
    Returns:
        (quant [H,W,omega,D] uint8, scale [H,W,omega,1] fp16, zp [H,W,omega,1] fp16)
    """
    if _OPS is not None and tensor.dtype in (torch.float16, torch.bfloat16) and tensor.is_cuda:
        result = _OPS.fused_quantize_v_int8(tensor.contiguous())
        return result[0], result[1], result[2]

    # PyTorch fallback
    t_min = tensor.amin(dim=3, keepdim=True)
    t_max = tensor.amax(dim=3, keepdim=True)
    scale = torch.clamp((t_max - t_min) / 255.0, min=1e-8)
    quantized = torch.round((tensor - t_min) / scale).clamp(0, 255).to(torch.uint8)
    return quantized, scale.to(tensor.dtype), t_min.to(tensor.dtype)


def dequantize_int8(quant_tensor, scale, zero_point, per_channel=True, omega=1):
    """
    Dequantize int8 tensor back to fp16.

    Args:
        quant_tensor: uint8
        scale, zero_point: fp16/bfloat16
        per_channel: True for K (scale has last dim D), False for V (last dim 1)
        omega: shared scale stride for K kernels
    Returns:
        float16/bfloat16 tensor
    """
    if _OPS is not None and quant_tensor.is_cuda and scale.dtype in (torch.float16, torch.bfloat16):
        return _OPS.fused_dequantize_int8(
            quant_tensor.contiguous(), scale.contiguous(),
            zero_point.contiguous(), per_channel, omega
        )

    # PyTorch fallback
    return quant_tensor.to(scale.dtype) * scale + zero_point


# ============================================================
# 3. Physical Eviction
# ============================================================

def eviction_copy_sinks(old_k, old_v, new_k, new_v, old_lid, new_lid, sink_tokens):
    """Copy sink tokens from old to new KV cache. Operates on [H, seq, D] views.
    Caller must ensure old_k and old_v are already contiguous."""
    total_elems = old_k.shape[0] * sink_tokens * old_k.shape[2]
    if (
        _OPS is not None
        and old_k.dtype in (torch.float16, torch.bfloat16)
        and old_k.is_cuda
        and total_elems >= _EVICT_THRESHOLD
    ):
        _OPS.fused_eviction_copy_sinks(
            old_k, old_v,                    # caller guarantees contiguous
            new_k, new_v,
            old_lid.contiguous(), new_lid,
            sink_tokens
        )
        return

    # PyTorch fallback — optimal for small sink counts (typically 4)
    new_k[:, :sink_tokens] = old_k[:, :sink_tokens]
    new_v[:, :sink_tokens] = old_v[:, :sink_tokens]
    new_lid[:, :sink_tokens] = old_lid[:, :sink_tokens]


def eviction_copy_sticky(
    old_k, old_v, new_k, new_v,
    first_phys, found_mask, final_ids, new_lid,
    curr_k, omega, sink_tokens
):
    """Copy surviving sticky windows from old to new KV cache.
    Caller must ensure old_k and old_v are already contiguous."""
    H, _, D = old_k.shape
    total_elems = H * curr_k * omega * D
    if (
        _OPS is not None
        and old_k.dtype in (torch.float16, torch.bfloat16)
        and old_k.is_cuda
        and total_elems >= _EVICT_THRESHOLD
    ):
        _OPS.fused_eviction_copy_sticky(
            old_k, old_v,                    # caller guarantees contiguous
            new_k, new_v,
            first_phys.contiguous().long(), found_mask.contiguous(),
            final_ids.contiguous().long(), new_lid,
            curr_k, omega, sink_tokens
        )
        return

    # PyTorch fallback
    device = old_k.device
    target_starts = sink_tokens + torch.arange(curr_k, device=device, dtype=torch.long) * omega
    target_starts = target_starts.unsqueeze(0).expand(H, -1)
    offsets = torch.arange(omega, device=device, dtype=torch.long)
    phys_gather    = (first_phys.unsqueeze(2) + offsets).view(H, -1)
    target_scatter = (target_starts.unsqueeze(2) + offsets).view(H, -1)
    mask = found_mask.unsqueeze(2).expand(-1, -1, omega).reshape(H, -1)
    if mask.any():
        valid_phys   = phys_gather[mask]
        valid_target = target_scatter[mask]
        head_indices = torch.arange(H, device=device).unsqueeze(1).expand(-1, curr_k * omega)
        valid_heads  = head_indices[mask]
        new_k[valid_heads, valid_target] = old_k[valid_heads, valid_phys]
        new_v[valid_heads, valid_target] = old_v[valid_heads, valid_phys]
        flat_ids = final_ids.unsqueeze(2).expand(-1, -1, omega).reshape(H, -1)
        new_lid[valid_heads, valid_target] = flat_ids[mask].long()


def eviction_copy_local(
    old_k, old_v, new_k, new_v, new_lid,
    local_count, old_local_start, new_local_start,
    local_start_wid, omega
):
    """Copy local zone tokens from old to new KV cache.
    Caller must ensure old_k and old_v are already contiguous."""
    H, _, D = old_k.shape
    total_elems = H * local_count * D
    if (
        _OPS is not None
        and old_k.dtype in (torch.float16, torch.bfloat16)
        and old_k.is_cuda
        and total_elems >= _EVICT_THRESHOLD
    ):
        _OPS.fused_eviction_copy_local(
            old_k, old_v,                    # caller guarantees contiguous
            new_k, new_v, new_lid,
            local_count, old_local_start, new_local_start,
            int(local_start_wid), omega
        )
        return

    # PyTorch fallback
    actual_local = min(local_count, old_k.shape[1] - old_local_start)
    new_k[:, new_local_start:new_local_start + actual_local] = \
        old_k[:, old_local_start:old_local_start + actual_local]
    new_v[:, new_local_start:new_local_start + actual_local] = \
        old_v[:, old_local_start:old_local_start + actual_local]
    offsets    = torch.arange(actual_local, device=new_lid.device, dtype=torch.long)
    local_lids = local_start_wid + (offsets // omega)
    new_lid[:, new_local_start:new_local_start + actual_local] = local_lids.unsqueeze(0)
