// sticky_kv_kernels.cu — CUDA kernels for Sticky KV Cache acceleration
//
// Three kernels:
//   1. per_head_kv_gather_kernel  — Fused per-head KV cache eviction gather
//   2. window_score_reduce_kernel — Window-level attention score reduction
//   3. vote_accumulate_kernel     — Generation-step vote accumulation
//
// Target: NVIDIA A6000 (Ampere, sm_86)
// Safety: All memory accesses are bounds-checked. No shared memory mutations.
//         No warp-level primitives. Simple global memory patterns only.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/python.h>

#include "sticky_kv_api.h"
#include "static_switch.h"

// ============================================================================
// Kernel 1: Per-Head KV Cache Gather (Eviction)
// ============================================================================
//
// Replaces the Python-level pattern:
//   gather_idx = indices.unsqueeze(-1).expand(-1, -1, head_dim)
//   k_out = torch.gather(key_cache, 1, gather_idx)
//   v_out = torch.gather(value_cache, 1, gather_idx)
//
// Fuses both K and V gathers into a single kernel launch, avoiding
// the intermediate expanded index tensor allocation.
//
// Memory layout (all contiguous):
//   key_cache, value_cache: [num_heads, seq_len, head_dim]
//   key_out, value_out:     [num_heads, max_kept, head_dim]
//   indices:                [num_heads, max_kept] as int32
//
// Grid:  (num_heads, num_thread_groups)
// Block: (num_threads)
//
// Each x-block handles one head. Multiple y-blocks provide parallelism
// within a head (same pattern as DefensiveKV).

template <typename tensor_t>
__global__ void per_head_kv_gather_kernel(
    tensor_t *__restrict__ key_out,
    tensor_t *__restrict__ value_out,
    const tensor_t *__restrict__ key_cache,
    const tensor_t *__restrict__ value_cache,
    const int *__restrict__ indices,
    const int seq_len,
    const int max_kept,
    const int head_dim) {

  const int head_idx = blockIdx.x;
  const int tid = threadIdx.x + blockIdx.y * blockDim.x;
  const int num_threads = blockDim.x * gridDim.y;

  // Total elements to copy per head (for both K and V)
  const int total_elements = max_kept * head_dim;

  // Precompute base offsets for this head
  const int src_head_off = head_idx * seq_len * head_dim;
  const int dst_head_off = head_idx * max_kept * head_dim;
  const int idx_head_off = head_idx * max_kept;

  // Iterate over all elements for this head
  for (int i = tid; i < total_elements; i += num_threads) {
    const int kept_pos = i / head_dim;
    const int d = i % head_dim;

    // Read the source position for this kept entry (bounds-clamped)
    int src_pos = indices[idx_head_off + kept_pos];

    // Safety: clamp to valid range [0, seq_len-1]
    if (src_pos < 0) src_pos = 0;
    if (src_pos >= seq_len) src_pos = seq_len - 1;

    const int src_offset = src_head_off + src_pos * head_dim + d;
    const int dst_offset = dst_head_off + i;

    // Copy both K and V in the same thread
    key_out[dst_offset] = key_cache[src_offset];
    value_out[dst_offset] = value_cache[src_offset];
  }
}


// ============================================================================
// Kernel 2: Window Score Reduction
// ============================================================================
//
// Replaces the Python-level pattern:
//   scores.view(num_heads, num_windows, omega).sum(dim=2)
//
// Each thread computes the sum for one (head, window) pair by iterating
// over omega elements. This is optimal for small omega (e.g., omega=5).
//
// Memory layout:
//   scores_in:  [num_heads, num_tokens]  (float32, contiguous)
//   scores_out: [num_heads, num_windows] (float32, contiguous)
//
// Grid:  (num_heads, ceil(num_windows / blockDim.x))
// Block: (256)

__global__ void window_score_reduce_kernel(
    float *__restrict__ scores_out,
    const float *__restrict__ scores_in,
    const int num_tokens,
    const int num_windows,
    const int omega) {

  const int head_idx = blockIdx.x;
  const int win_idx = blockIdx.y * blockDim.x + threadIdx.x;

  if (win_idx >= num_windows) return;

  // Compute sum for this window
  float sum = 0.0f;
  const int base = win_idx * omega;
  const int head_offset = head_idx * num_tokens;

  #pragma unroll 8
  for (int t = 0; t < omega; t++) {
    const int pos = base + t;
    // Safety: bounds check against actual token count
    if (pos < num_tokens) {
      sum += scores_in[head_offset + pos];
    }
  }

  scores_out[head_idx * num_windows + win_idx] = sum;
}


// ============================================================================
// Kernel 3: Vote Accumulation
// ============================================================================
//
// Replaces the Python-level pattern:
//   running_votes[:, :seq_len] += new_scores[:, :seq_len]
//
// Simple elementwise addition. No race conditions since each (head, pos)
// pair is handled by exactly one thread.
//
// Memory layout:
//   running_votes: [num_heads, max_context] (float32, contiguous)
//   new_scores:    [num_heads, score_stride] (float32, contiguous)
//     - Only first seq_len elements per head are read
//
// Grid:  (num_heads, ceil(seq_len / blockDim.x))
// Block: (256)

__global__ void vote_accumulate_kernel(
    float *__restrict__ running_votes,
    const float *__restrict__ new_scores,
    const int seq_len,
    const int max_context,
    const int score_stride) {

  const int head_idx = blockIdx.x;
  const int pos = blockIdx.y * blockDim.x + threadIdx.x;

  if (pos >= seq_len) return;

  // Each (head, pos) is unique — no race condition
  running_votes[head_idx * max_context + pos] += new_scores[head_idx * score_stride + pos];
}


// ============================================================================
// C++ Launcher Functions (called from Python via pybind11)
// ============================================================================

// Launcher for Kernel 1: per_head_kv_gather
std::vector<torch::Tensor> per_head_kv_gather(
    torch::Tensor &key_cache,    // [num_heads, seq_len, head_dim]
    torch::Tensor &value_cache,  // [num_heads, seq_len, head_dim]
    torch::Tensor &indices) {    // [num_heads, max_kept] int32

  CHECK_INPUT(key_cache);
  CHECK_INPUT(value_cache);
  CHECK_INPUT(indices);
  TORCH_CHECK(indices.dtype() == torch::kInt32,
              "indices must be int32, got ", indices.dtype());
  TORCH_CHECK(key_cache.dim() == 3,
              "key_cache must be 3D [num_heads, seq_len, head_dim]");
  TORCH_CHECK(value_cache.dim() == 3,
              "value_cache must be 3D [num_heads, seq_len, head_dim]");
  TORCH_CHECK(indices.dim() == 2,
              "indices must be 2D [num_heads, max_kept]");
  TORCH_CHECK(key_cache.dtype() == value_cache.dtype(),
              "key_cache and value_cache must have the same dtype");

  const int num_heads = key_cache.size(0);
  const int seq_len = key_cache.size(1);
  const int head_dim = key_cache.size(2);
  const int max_kept = indices.size(1);

  TORCH_CHECK(key_cache.size(0) == indices.size(0),
              "num_heads mismatch between cache and indices");
  TORCH_CHECK(value_cache.size(0) == num_heads,
              "num_heads mismatch between key and value cache");
  TORCH_CHECK(value_cache.size(1) == seq_len,
              "seq_len mismatch between key and value cache");
  TORCH_CHECK(value_cache.size(2) == head_dim,
              "head_dim mismatch between key and value cache");

  // Allocate output tensors
  auto key_out = torch::empty({num_heads, max_kept, head_dim}, key_cache.options());
  auto value_out = torch::empty({num_heads, max_kept, head_dim}, value_cache.options());

  if (max_kept == 0) {
    return {key_out, value_out};
  }

  // Configure launch parameters (follow DefensiveKV's pattern)
  const int num_threads = 256;
  // Adapt thread groups to workload size
  const int total_work = max_kept * head_dim;
  int num_thread_groups = (total_work + num_threads - 1) / num_threads;
  num_thread_groups = std::min(num_thread_groups, 128);
  num_thread_groups = std::max(num_thread_groups, 1);

  dim3 grid(num_heads, num_thread_groups);
  dim3 block(num_threads);

  auto stream = at::cuda::getCurrentCUDAStream();

  // Dispatch based on dtype
  DTYPE_SWITCH(key_cache.scalar_type(), [&] {
    per_head_kv_gather_kernel<elem_type><<<grid, block, 0, stream>>>(
        (elem_type *)key_out.data_ptr(),
        (elem_type *)value_out.data_ptr(),
        (const elem_type *)key_cache.data_ptr(),
        (const elem_type *)value_cache.data_ptr(),
        (const int *)indices.data_ptr(),
        seq_len, max_kept, head_dim);
  });

  // Check for launch errors
  CUDA_ERROR_CHECK(cudaGetLastError());

  return {key_out, value_out};
}


// Launcher for Kernel 2: window_score_reduce
torch::Tensor window_score_reduce(
    torch::Tensor &scores,     // [num_heads, num_tokens] float32
    int num_windows,
    int omega) {

  CHECK_INPUT(scores);
  TORCH_CHECK(scores.dtype() == torch::kFloat32,
              "scores must be float32, got ", scores.dtype());
  TORCH_CHECK(scores.dim() == 2,
              "scores must be 2D [num_heads, num_tokens]");
  TORCH_CHECK(num_windows >= 0, "num_windows must be non-negative");
  TORCH_CHECK(omega > 0, "omega must be positive");
  TORCH_CHECK(num_windows * omega <= scores.size(1),
              "num_windows * omega (", num_windows * omega,
              ") exceeds num_tokens (", scores.size(1), ")");

  const int num_heads = scores.size(0);
  const int num_tokens = scores.size(1);

  auto out = torch::empty({num_heads, num_windows}, scores.options());

  if (num_windows == 0) {
    return out;
  }

  const int num_threads = 256;
  dim3 grid(num_heads, (num_windows + num_threads - 1) / num_threads);
  dim3 block(num_threads);

  auto stream = at::cuda::getCurrentCUDAStream();

  window_score_reduce_kernel<<<grid, block, 0, stream>>>(
      (float *)out.data_ptr(),
      (const float *)scores.data_ptr(),
      num_tokens, num_windows, omega);

  CUDA_ERROR_CHECK(cudaGetLastError());

  return out;
}


// Launcher for Kernel 3: vote_accumulate
void vote_accumulate(
    torch::Tensor &running_votes,  // [num_heads, max_context] float32, modified in-place
    torch::Tensor &new_scores,     // [num_heads, score_stride] float32
    int seq_len) {

  CHECK_INPUT(running_votes);
  CHECK_INPUT(new_scores);
  TORCH_CHECK(running_votes.dtype() == torch::kFloat32,
              "running_votes must be float32");
  TORCH_CHECK(new_scores.dtype() == torch::kFloat32,
              "new_scores must be float32");
  TORCH_CHECK(running_votes.dim() == 2, "running_votes must be 2D");
  TORCH_CHECK(new_scores.dim() == 2, "new_scores must be 2D");
  TORCH_CHECK(seq_len >= 0, "seq_len must be non-negative");

  const int num_heads = running_votes.size(0);
  const int max_context = running_votes.size(1);
  const int score_stride = new_scores.size(1);

  TORCH_CHECK(new_scores.size(0) == num_heads,
              "num_heads mismatch between running_votes and new_scores");
  TORCH_CHECK(seq_len <= max_context,
              "seq_len exceeds max_context");
  TORCH_CHECK(seq_len <= score_stride,
              "seq_len exceeds new_scores length");

  if (seq_len == 0) return;

  const int num_threads = 256;
  dim3 grid(num_heads, (seq_len + num_threads - 1) / num_threads);
  dim3 block(num_threads);

  auto stream = at::cuda::getCurrentCUDAStream();

  vote_accumulate_kernel<<<grid, block, 0, stream>>>(
      (float *)running_votes.data_ptr(),
      (const float *)new_scores.data_ptr(),
      seq_len, max_context, score_stride);

  CUDA_ERROR_CHECK(cudaGetLastError());
}


// ============================================================================
// Python Module Registration
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("per_head_kv_gather", &per_head_kv_gather,
        "Per-head KV cache gather for eviction (fused K+V)");
  m.def("window_score_reduce", &window_score_reduce,
        "Reduce attention scores into per-window sums");
  m.def("vote_accumulate", &vote_accumulate,
        "Accumulate generation-step attention votes in-place");
}
