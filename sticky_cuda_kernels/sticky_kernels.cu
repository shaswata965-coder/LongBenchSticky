#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// ============================================================
// Kernel 1: Fused Scoreboard Scatter
// Replaces: is_chunk_token mask + torch.where + scatter_add_
// Each thread handles one (head, position) pair.
// ============================================================
__global__ void scoreboard_scatter_kernel(
    const float* __restrict__ votes,       // [H, compressed_len]
    const long*  __restrict__ logical_ids,  // [H, compressed_len]
    float*       __restrict__ scoreboard,   // [H, max_windows]
    int H, int compressed_len, int max_windows
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * compressed_len;
    if (idx >= total) return;

    int h = idx / compressed_len;
    int p = idx % compressed_len;

    long lid = logical_ids[h * compressed_len + p];
    if (lid >= 0 && lid < max_windows) {
        float v = votes[h * compressed_len + p];
        atomicAdd(&scoreboard[h * max_windows + lid], v);
    }
}

// Also scatter votes for new tokens not yet in logical_id_map
__global__ void scoreboard_scatter_new_tokens_kernel(
    const float* __restrict__ votes,        // [H, max_context] — read [H, compressed_len..seq_len]
    float*       __restrict__ scoreboard,   // [H, max_windows]
    int H, int compressed_len, int seq_len, int max_windows,
    long num_tokens_without_eviction, int omega, int sink_tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_new = seq_len - compressed_len;
    int total = H * n_new;
    if (idx >= total) return;

    int h = idx / n_new;
    int j = idx % n_new;

    long raw_lid = (num_tokens_without_eviction - omega + (long)j - sink_tokens) / omega;
    if (raw_lid >= 0 && raw_lid < max_windows) {
        int vote_idx = h * seq_len + compressed_len + j;  // offset into votes buffer at max_context stride
        // Actually votes is running_attention_votes with stride max_context, but we pass the pointer
        // offset to [h, compressed_len] already. Let caller handle pointer math.
        // Simpler: pass votes as [H, n_new] sub-view
        float v = votes[h * n_new + j];
        atomicAdd(&scoreboard[h * max_windows + raw_lid], v);
    }
}

// ============================================================
// Kernel 2: Fused INT8 Quantize K (per-channel, RoPE-paired)
// Input:  tensor [H, W, omega, D] float16
// Output: quant  [H, W, omega, D] uint8, scale [H,W,1,D] fp16, zp [H,W,1,D] fp16
// ============================================================
__global__ void quantize_k_int8_kernel(
    const at::Half* __restrict__ input,   // [H*W, omega, D]
    uint8_t*        __restrict__ output,  // [H*W, omega, D]
    at::Half*       __restrict__ scale,   // [H*W, 1, D]
    at::Half*       __restrict__ zp,      // [H*W, 1, D]
    int HW, int omega, int D
) {
    // Each block handles one (hw, d) pair — reduces across omega
    int hw = blockIdx.x;
    int d  = blockIdx.y * blockDim.x + threadIdx.x;
    if (hw >= HW || d >= D) return;

    int half_d = D / 2;
    // RoPE pairing: pair d with d+half_d or d-half_d
    int d_pair = (d < half_d) ? (d + half_d) : (d - half_d);

    // Find min/max across omega for both d and d_pair
    float vmin_d = 1e30f, vmax_d = -1e30f;
    float vmin_p = 1e30f, vmax_p = -1e30f;
    for (int t = 0; t < omega; t++) {
        float val_d = __half2float(input[hw * omega * D + t * D + d]);
        float val_p = __half2float(input[hw * omega * D + t * D + d_pair]);
        vmin_d = fminf(vmin_d, val_d);
        vmax_d = fmaxf(vmax_d, val_d);
        vmin_p = fminf(vmin_p, val_p);
        vmax_p = fmaxf(vmax_p, val_p);
    }

    // Tied min/max
    float tied_min = fminf(vmin_d, vmin_p);
    float tied_max = fmaxf(vmax_d, vmax_p);

    float s = fmaxf((tied_max - tied_min) / 255.0f, 1e-8f);
    float inv_s = 1.0f / s;

    // Write scale and zp
    scale[hw * D + d] = __float2half(s);
    zp[hw * D + d]    = __float2half(tied_min);

    // Quantize
    for (int t = 0; t < omega; t++) {
        float val = __half2float(input[hw * omega * D + t * D + d]);
        float q = roundf((val - tied_min) * inv_s);
        q = fminf(fmaxf(q, 0.0f), 255.0f);
        output[hw * omega * D + t * D + d] = (uint8_t)q;
    }
}

// ============================================================
// Kernel 3: Fused INT8 Quantize V (per-token)
// Input:  tensor [H, W, omega, D] float16
// Output: quant  [H, W, omega, D] uint8, scale [H,W,omega,1] fp16, zp [H,W,omega,1] fp16
// ============================================================
__global__ void quantize_v_int8_kernel(
    const at::Half* __restrict__ input,   // [H*W*omega, D]
    uint8_t*        __restrict__ output,  // [H*W*omega, D]
    at::Half*       __restrict__ scale,   // [H*W*omega, 1]
    at::Half*       __restrict__ zp,      // [H*W*omega, 1]
    int total_rows, int D
) {
    int row = blockIdx.x;
    if (row >= total_rows) return;

    // Each block processes one row (one token), threads handle columns
    int d = threadIdx.x;

    // Use shared memory for reduction
    extern __shared__ float smem[];
    float* s_min = smem;
    float* s_max = smem + blockDim.x;

    float local_min = 1e30f, local_max = -1e30f;
    for (int col = d; col < D; col += blockDim.x) {
        float val = __half2float(input[row * D + col]);
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }
    s_min[d] = local_min;
    s_max[d] = local_max;
    __syncthreads();

    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (d < stride) {
            s_min[d] = fminf(s_min[d], s_min[d + stride]);
            s_max[d] = fmaxf(s_max[d], s_max[d + stride]);
        }
        __syncthreads();
    }

    float row_min = s_min[0];
    float row_max = s_max[0];
    float s_val = fmaxf((row_max - row_min) / 255.0f, 1e-8f);
    float inv_s = 1.0f / s_val;

    if (d == 0) {
        scale[row] = __float2half(s_val);
        zp[row]    = __float2half(row_min);
    }
    __syncthreads();

    // Quantize all columns
    for (int col = d; col < D; col += blockDim.x) {
        float val = __half2float(input[row * D + col]);
        float q = roundf((val - row_min) * inv_s);
        q = fminf(fmaxf(q, 0.0f), 255.0f);
        output[row * D + col] = (uint8_t)q;
    }
}

// ============================================================
// Kernel 4: Fused INT8 Dequantize
// quant [N, D] uint8 + scale + zp -> output [N, D] float16
// ============================================================
__global__ void dequantize_int8_kernel(
    const uint8_t*  __restrict__ quant,
    const at::Half* __restrict__ scale,
    const at::Half* __restrict__ zp,
    at::Half*       __restrict__ output,
    int N, int D,
    int scale_stride_last,  // D for K (per-channel), 1 for V (per-token)
    int omega               // K: scale shared across omega rows; V: set to 1
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;

    int row = idx / D;
    int col = idx % D;

    int scale_row = row / omega;  // K: groups of omega share one scale row; V: omega=1
    int scale_col = (scale_stride_last == 1) ? 0 : col;
    float s = __half2float(scale[scale_row * scale_stride_last + scale_col]);
    float z = __half2float(zp[scale_row * scale_stride_last + scale_col]);
    float val = (float)quant[idx] * s + z;
    output[idx] = __float2half(val);
}

// ============================================================
// Kernel 5: Fused Physical Eviction — copy sticky windows
// For each (head, window_slot), copies omega tokens from old to new position
// ============================================================
__global__ void eviction_copy_sticky_kernel(
    const at::Half* __restrict__ old_k,   // [H, old_seq, D]
    const at::Half* __restrict__ old_v,
    at::Half*       __restrict__ new_k,   // [H, new_seq, D]
    at::Half*       __restrict__ new_v,
    const long*     __restrict__ first_phys,   // [H, curr_k] — source start per window
    const bool*     __restrict__ found_mask,   // [H, curr_k]
    const long*     __restrict__ final_ids,    // [H, curr_k] — logical window IDs
    long*           __restrict__ new_lid_map,  // [H, new_seq]
    int H, int curr_k, int omega, int D,
    int old_seq, int new_seq, int sink_tokens
) {
    // Grid: (H * curr_k * omega) blocks of D threads
    int block_id = blockIdx.x;
    int total_blocks = H * curr_k * omega;
    if (block_id >= total_blocks) return;

    int h = block_id / (curr_k * omega);
    int rem = block_id % (curr_k * omega);
    int wi = rem / omega;
    int ti = rem % omega;
    int d = threadIdx.x;
    if (d >= D) return;

    if (!found_mask[h * curr_k + wi]) return;

    long src_pos = first_phys[h * curr_k + wi] + ti;
    long dst_pos = sink_tokens + wi * omega + ti;

    if (src_pos >= old_seq || dst_pos >= new_seq) return;

    long src_idx = (long)h * old_seq * D + src_pos * D + d;
    long dst_idx = (long)h * new_seq * D + dst_pos * D + d;

    new_k[dst_idx] = old_k[src_idx];
    new_v[dst_idx] = old_v[src_idx];

    // Write logical ID (only first thread per token to avoid races)
    if (d == 0) {
        new_lid_map[h * new_seq + dst_pos] = final_ids[h * curr_k + wi];
    }
}

// Copy sinks
__global__ void eviction_copy_sinks_kernel(
    const at::Half* __restrict__ old_k,
    const at::Half* __restrict__ old_v,
    at::Half*       __restrict__ new_k,
    at::Half*       __restrict__ new_v,
    const long*     __restrict__ old_lid,
    long*           __restrict__ new_lid,
    int H, int sink_tokens, int D, int old_seq, int new_seq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * sink_tokens * D;
    if (idx >= total) return;

    int h = idx / (sink_tokens * D);
    int rem = idx % (sink_tokens * D);
    int t = rem / D;
    int d = rem % D;

    long src = (long)h * old_seq * D + t * D + d;
    long dst = (long)h * new_seq * D + t * D + d;
    new_k[dst] = old_k[src];
    new_v[dst] = old_v[src];

    if (d == 0) {
        new_lid[h * new_seq + t] = old_lid[h * old_seq + t];
    }
}

// Copy local zone
__global__ void eviction_copy_local_kernel(
    const at::Half* __restrict__ old_k,
    const at::Half* __restrict__ old_v,
    at::Half*       __restrict__ new_k,
    at::Half*       __restrict__ new_v,
    long*           __restrict__ new_lid,
    int H, int local_count, int D,
    int old_seq, int new_seq,
    int old_local_start, int new_local_start,
    long local_start_wid, int omega
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * local_count * D;
    if (idx >= total) return;

    int h = idx / (local_count * D);
    int rem = idx % (local_count * D);
    int t = rem / D;
    int d = rem % D;

    long src_pos = old_local_start + t;
    long dst_pos = new_local_start + t;
    if (src_pos >= old_seq || dst_pos >= new_seq) return;

    long src = (long)h * old_seq * D + src_pos * D + d;
    long dst = (long)h * new_seq * D + dst_pos * D + d;
    new_k[dst] = old_k[src];
    new_v[dst] = old_v[src];

    if (d == 0) {
        long lid = local_start_wid + (long)t / omega;
        new_lid[h * new_seq + dst_pos] = lid;
    }
}


// ============================================================
// C++ wrapper functions
// ============================================================

torch::Tensor fused_scoreboard_scatter(
    torch::Tensor votes,         // [H, compressed_len]
    torch::Tensor logical_ids,   // [H, compressed_len]
    int max_windows
) {
    int H = votes.size(0);
    int compressed_len = votes.size(1);
    auto scoreboard = torch::zeros({H, max_windows}, votes.options());

    int total = H * compressed_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    scoreboard_scatter_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        votes.data_ptr<float>(),
        logical_ids.data_ptr<long>(),
        scoreboard.data_ptr<float>(),
        H, compressed_len, max_windows
    );
    return scoreboard;
}

void fused_scoreboard_scatter_new_tokens(
    torch::Tensor votes_slice,   // [H, n_new] — already sliced
    torch::Tensor scoreboard,    // [H, max_windows] — mutated in-place
    long num_tokens_without_eviction, int omega, int sink_tokens,
    int compressed_len, int seq_len
) {
    int H = votes_slice.size(0);
    int n_new = votes_slice.size(1);
    int max_windows = scoreboard.size(1);
    int total = H * n_new;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    scoreboard_scatter_new_tokens_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        votes_slice.data_ptr<float>(),
        scoreboard.data_ptr<float>(),
        H, compressed_len, seq_len, max_windows,
        num_tokens_without_eviction, omega, sink_tokens
    );
}

std::vector<torch::Tensor> fused_quantize_k_int8(
    torch::Tensor input  // [H, W, omega, D] float16
) {
    int H = input.size(0);
    int W = input.size(1);
    int omega = input.size(2);
    int D = input.size(3);
    int HW = H * W;

    auto output = torch::zeros({H, W, omega, D}, input.options().dtype(torch::kUInt8));
    auto scale  = torch::zeros({H, W, 1, D}, input.options());
    auto zp     = torch::zeros({H, W, 1, D}, input.options());

    // Grid: (HW, ceil(D/256)), Block: min(D, 256)
    int threads = std::min(D, 256);
    dim3 grid(HW, (D + threads - 1) / threads);

    quantize_k_int8_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const at::Half*)input.data_ptr(),
        output.data_ptr<uint8_t>(),
        (at::Half*)scale.data_ptr(),
        (at::Half*)zp.data_ptr(),
        HW, omega, D
    );
    return {output, scale, zp};
}

std::vector<torch::Tensor> fused_quantize_v_int8(
    torch::Tensor input  // [H, W, omega, D] float16
) {
    int H = input.size(0);
    int W = input.size(1);
    int omega = input.size(2);
    int D = input.size(3);
    int total_rows = H * W * omega;

    auto output = torch::zeros({H, W, omega, D}, input.options().dtype(torch::kUInt8));
    auto scale  = torch::zeros({H, W, omega, 1}, input.options());
    auto zp     = torch::zeros({H, W, omega, 1}, input.options());

    int threads = std::min(D, 256);
    // Need power-of-2 for reduction
    int red_threads = 1;
    while (red_threads < threads) red_threads <<= 1;
    if (red_threads > 256) red_threads = 256;

    quantize_v_int8_kernel<<<total_rows, red_threads, 2 * red_threads * sizeof(float),
        at::cuda::getCurrentCUDAStream()>>>(
        (const at::Half*)input.data_ptr(),
        output.data_ptr<uint8_t>(),
        (at::Half*)scale.data_ptr(),
        (at::Half*)zp.data_ptr(),
        total_rows, D
    );
    return {output, scale, zp};
}

torch::Tensor fused_dequantize_int8(
    torch::Tensor quant,   // [..., D] uint8
    torch::Tensor scale,   // [..., D] or [..., 1] float16
    torch::Tensor zp,      // same shape as scale
    bool per_channel,       // true for K (scale_last_dim=D), false for V (scale_last_dim=1)
    int omega               // K: scale shared across omega rows; V: set to 1
) {
    auto output = torch::zeros_like(quant, scale.options());
    int N = quant.numel() / quant.size(-1);
    int D = quant.size(-1);
    int scale_stride_last = per_channel ? D : 1;

    int total = N * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    dequantize_int8_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        quant.data_ptr<uint8_t>(),
        (const at::Half*)scale.data_ptr(),
        (const at::Half*)zp.data_ptr(),
        (at::Half*)output.data_ptr(),
        N, D, scale_stride_last, omega
    );
    return output;
}

void fused_eviction_copy_sinks(
    torch::Tensor old_k, torch::Tensor old_v,
    torch::Tensor new_k, torch::Tensor new_v,
    torch::Tensor old_lid, torch::Tensor new_lid,
    int sink_tokens
) {
    int H = old_k.size(0);
    int D = old_k.size(2);
    int old_seq = old_k.size(1);
    int new_seq = new_k.size(1);

    int total = H * sink_tokens * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    eviction_copy_sinks_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const at::Half*)old_k.data_ptr(), (const at::Half*)old_v.data_ptr(),
        (at::Half*)new_k.data_ptr(), (at::Half*)new_v.data_ptr(),
        old_lid.data_ptr<long>(), new_lid.data_ptr<long>(),
        H, sink_tokens, D, old_seq, new_seq
    );
}

void fused_eviction_copy_sticky(
    torch::Tensor old_k, torch::Tensor old_v,
    torch::Tensor new_k, torch::Tensor new_v,
    torch::Tensor first_phys, torch::Tensor found_mask,
    torch::Tensor final_ids, torch::Tensor new_lid,
    int curr_k, int omega, int sink_tokens
) {
    int H = old_k.size(0);
    int D = old_k.size(2);
    int old_seq = old_k.size(1);
    int new_seq = new_k.size(1);

    int total_blocks_needed = H * curr_k * omega;
    int threads = std::min(D, 256);

    eviction_copy_sticky_kernel<<<total_blocks_needed, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const at::Half*)old_k.data_ptr(), (const at::Half*)old_v.data_ptr(),
        (at::Half*)new_k.data_ptr(), (at::Half*)new_v.data_ptr(),
        first_phys.data_ptr<long>(), found_mask.data_ptr<bool>(),
        final_ids.data_ptr<long>(), new_lid.data_ptr<long>(),
        H, curr_k, omega, D, old_seq, new_seq, sink_tokens
    );
}

void fused_eviction_copy_local(
    torch::Tensor old_k, torch::Tensor old_v,
    torch::Tensor new_k, torch::Tensor new_v,
    torch::Tensor new_lid,
    int local_count, int old_local_start, int new_local_start,
    long local_start_wid, int omega
) {
    int H = old_k.size(0);
    int D = old_k.size(2);
    int old_seq = old_k.size(1);
    int new_seq = new_k.size(1);

    int total = H * local_count * D;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    eviction_copy_local_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const at::Half*)old_k.data_ptr(), (const at::Half*)old_v.data_ptr(),
        (at::Half*)new_k.data_ptr(), (at::Half*)new_v.data_ptr(),
        new_lid.data_ptr<long>(),
        H, local_count, D, old_seq, new_seq,
        old_local_start, new_local_start,
        local_start_wid, omega
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_scoreboard_scatter", &fused_scoreboard_scatter);
    m.def("fused_scoreboard_scatter_new_tokens", &fused_scoreboard_scatter_new_tokens);
    m.def("fused_quantize_k_int8", &fused_quantize_k_int8);
    m.def("fused_quantize_v_int8", &fused_quantize_v_int8);
    m.def("fused_dequantize_int8", &fused_dequantize_int8);
    m.def("fused_eviction_copy_sinks", &fused_eviction_copy_sinks);
    m.def("fused_eviction_copy_sticky", &fused_eviction_copy_sticky);
    m.def("fused_eviction_copy_local", &fused_eviction_copy_local);
}
