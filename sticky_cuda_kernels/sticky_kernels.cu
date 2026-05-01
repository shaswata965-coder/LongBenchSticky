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
    const float* __restrict__ votes,        // [H, n_new] — read [H, n_new]
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
        float v = votes[h * n_new + j];
        atomicAdd(&scoreboard[h * max_windows + raw_lid], v);
    }
}

// ============================================================
// Kernel 2: Fused INT8 Quantize K (per-channel, RoPE-paired)
// Input:  tensor [H, W, omega, D]
// Output: quant  [H, W, omega, D] uint8, scale [H,W,1,D], zp [H,W,1,D]
// ============================================================
template <typename scalar_t>
__global__ void quantize_k_int8_kernel(
    const scalar_t* __restrict__ input,   // [H*W, omega, D]
    uint8_t*        __restrict__ output,  // [H*W, omega, D]
    scalar_t*       __restrict__ scale,   // [H*W, 1, D]
    scalar_t*       __restrict__ zp,      // [H*W, 1, D]
    int HW, int omega, int D
) {
    int hw = blockIdx.x;
    int d  = blockIdx.y * blockDim.x + threadIdx.x;
    if (hw >= HW || d >= D) return;

    int half_d = D / 2;
    int d_pair = (d < half_d) ? (d + half_d) : (d - half_d);

    float vmin_d = 1e30f, vmax_d = -1e30f;
    float vmin_p = 1e30f, vmax_p = -1e30f;
    for (int t = 0; t < omega; t++) {
        float val_d = static_cast<float>(input[hw * omega * D + t * D + d]);
        float val_p = static_cast<float>(input[hw * omega * D + t * D + d_pair]);
        vmin_d = fminf(vmin_d, val_d);
        vmax_d = fmaxf(vmax_d, val_d);
        vmin_p = fminf(vmin_p, val_p);
        vmax_p = fmaxf(vmax_p, val_p);
    }

    float tied_min = fminf(vmin_d, vmin_p);
    float tied_max = fmaxf(vmax_d, vmax_p);

    float s = fmaxf((tied_max - tied_min) / 255.0f, 1e-8f);
    float inv_s = 1.0f / s;

    scale[hw * D + d] = static_cast<scalar_t>(s);
    zp[hw * D + d]    = static_cast<scalar_t>(tied_min);

    for (int t = 0; t < omega; t++) {
        float val = static_cast<float>(input[hw * omega * D + t * D + d]);
        float q = roundf((val - tied_min) * inv_s);
        q = fminf(fmaxf(q, 0.0f), 255.0f);
        output[hw * omega * D + t * D + d] = (uint8_t)q;
    }
}

// ============================================================
// Kernel 3: Fused INT8 Quantize V (per-token)
// Input:  tensor [H, W, omega, D]
// Output: quant  [H, W, omega, D] uint8, scale [H,W,omega,1], zp [H,W,omega,1]
// ============================================================
template <typename scalar_t>
__global__ void quantize_v_int8_kernel(
    const scalar_t* __restrict__ input,   // [H*W*omega, D]
    uint8_t*        __restrict__ output,  // [H*W*omega, D]
    scalar_t*       __restrict__ scale,   // [H*W*omega, 1]
    scalar_t*       __restrict__ zp,      // [H*W*omega, 1]
    int total_rows, int D
) {
    int row = blockIdx.x;
    if (row >= total_rows) return;

    int d = threadIdx.x;

    extern __shared__ float smem[];
    float* s_min = smem;
    float* s_max = smem + blockDim.x;

    // BUG-5 FIX: When red_threads is rounded up to the next power-of-2 and D is
    // not itself a power-of-2, threads with d >= D must not iterate over any
    // columns.  Their sentinel values (1e30f / -1e30f) are the correct identity
    // elements for fminf/fmaxf, so the reduction remains correct.
    float local_min = 1e30f, local_max = -1e30f;
    if (d < D) {
        for (int col = d; col < D; col += blockDim.x) {
            float val = static_cast<float>(input[row * D + col]);
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
        }
    }
    s_min[d] = local_min;
    s_max[d] = local_max;
    __syncthreads();

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
        scale[row] = static_cast<scalar_t>(s_val);
        zp[row]    = static_cast<scalar_t>(row_min);
    }
    __syncthreads();

    for (int col = d; col < D; col += blockDim.x) {
        float val = static_cast<float>(input[row * D + col]);
        float q = roundf((val - row_min) * inv_s);
        q = fminf(fmaxf(q, 0.0f), 255.0f);
        output[row * D + col] = (uint8_t)q;
    }
}

// ============================================================
// Kernel 4: Fused INT8 Dequantize
// quant [N, D] uint8 + scale + zp -> output [N, D]
// ============================================================
template <typename scalar_t>
__global__ void dequantize_int8_kernel(
    const uint8_t*  __restrict__ quant,
    const scalar_t* __restrict__ scale,
    const scalar_t* __restrict__ zp,
    scalar_t*       __restrict__ output,
    int N, int D,
    int scale_stride_last,  
    int omega               
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;

    int row = idx / D;
    int col = idx % D;

    int scale_row = row / omega;  
    int scale_col = (scale_stride_last == 1) ? 0 : col;
    float s = static_cast<float>(scale[scale_row * scale_stride_last + scale_col]);
    float z = static_cast<float>(zp[scale_row * scale_stride_last + scale_col]);
    float val = (float)quant[idx] * s + z;
    output[idx] = static_cast<scalar_t>(val);
}

// ============================================================
// Kernel 5: Fused Physical Eviction — copy sticky windows
// ============================================================
template <typename scalar_t>
__global__ void eviction_copy_sticky_kernel(
    const scalar_t* __restrict__ old_k,   
    const scalar_t* __restrict__ old_v,
    scalar_t*       __restrict__ new_k,   
    scalar_t*       __restrict__ new_v,
    const long*     __restrict__ first_phys,   
    const bool*     __restrict__ found_mask,   
    const long*     __restrict__ final_ids,    
    long*           __restrict__ new_lid_map,  
    int H, int curr_k, int omega, int D,
    int old_seq, int new_seq, int sink_tokens
) {
    int block_id = blockIdx.x;
    int total_blocks = H * curr_k * omega;
    if (block_id >= total_blocks) return;

    int h = block_id / (curr_k * omega);
    int rem = block_id % (curr_k * omega);
    int wi = rem / omega;
    int ti = rem % omega;

    if (!found_mask[h * curr_k + wi]) return;

    long src_pos = first_phys[h * curr_k + wi] + ti;
    long dst_pos = sink_tokens + wi * omega + ti;

    if (src_pos >= old_seq || dst_pos >= new_seq) return;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        long src_idx = (long)h * old_seq * D + src_pos * D + d;
        long dst_idx = (long)h * new_seq * D + dst_pos * D + d;

        new_k[dst_idx] = old_k[src_idx];
        new_v[dst_idx] = old_v[src_idx];
    }

    if (threadIdx.x == 0) {
        new_lid_map[h * new_seq + dst_pos] = final_ids[h * curr_k + wi];
    }
}

// Copy sinks
template <typename scalar_t>
__global__ void eviction_copy_sinks_kernel(
    const scalar_t* __restrict__ old_k,
    const scalar_t* __restrict__ old_v,
    scalar_t*       __restrict__ new_k,
    scalar_t*       __restrict__ new_v,
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
template <typename scalar_t>
__global__ void eviction_copy_local_kernel(
    const scalar_t* __restrict__ old_k,
    const scalar_t* __restrict__ old_v,
    scalar_t*       __restrict__ new_k,
    scalar_t*       __restrict__ new_v,
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

#define DISPATCH_HALF_AND_BFLOAT16(TYPE, NAME, ...) \
    if (TYPE == torch::kFloat16) { \
        using scalar_t = at::Half; \
        __VA_ARGS__(); \
    } else if (TYPE == torch::kBFloat16) { \
        using scalar_t = at::BFloat16; \
        __VA_ARGS__(); \
    } else { \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }

torch::Tensor fused_scoreboard_scatter(
    torch::Tensor votes,         
    torch::Tensor logical_ids,   
    int max_windows
) {
    TORCH_CHECK(votes.is_cuda(), "votes must be a CUDA tensor");
    TORCH_CHECK(logical_ids.is_cuda(), "logical_ids must be a CUDA tensor");
    TORCH_CHECK(votes.is_contiguous(), "votes must be contiguous");
    TORCH_CHECK(logical_ids.is_contiguous(), "logical_ids must be contiguous");
    TORCH_CHECK(votes.scalar_type() == torch::kFloat32, "votes must be float32");
    TORCH_CHECK(logical_ids.scalar_type() == torch::kInt64, "logical_ids must be int64");

    int H = votes.size(0);
    int compressed_len = votes.size(1);
    auto scoreboard = torch::zeros({H, max_windows}, votes.options());

    int total = H * compressed_len;
    if (total == 0) return scoreboard;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    scoreboard_scatter_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        votes.data_ptr<float>(),
        logical_ids.data_ptr<long>(),
        scoreboard.data_ptr<float>(),
        H, compressed_len, max_windows
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return scoreboard;
}

void fused_scoreboard_scatter_new_tokens(
    torch::Tensor votes_slice,   
    torch::Tensor scoreboard,    
    long num_tokens_without_eviction, int omega, int sink_tokens,
    int compressed_len, int seq_len
) {
    TORCH_CHECK(votes_slice.is_cuda(), "votes_slice must be a CUDA tensor");
    TORCH_CHECK(scoreboard.is_cuda(), "scoreboard must be a CUDA tensor");
    TORCH_CHECK(votes_slice.is_contiguous(), "votes_slice must be contiguous");
    TORCH_CHECK(scoreboard.is_contiguous(), "scoreboard must be contiguous");
    TORCH_CHECK(votes_slice.scalar_type() == torch::kFloat32, "votes_slice must be float32");
    TORCH_CHECK(scoreboard.scalar_type() == torch::kFloat32, "scoreboard must be float32");

    int H = votes_slice.size(0);
    int n_new = votes_slice.size(1);
    int max_windows = scoreboard.size(1);
    int total = H * n_new;
    if (total == 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    scoreboard_scatter_new_tokens_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        votes_slice.data_ptr<float>(),
        scoreboard.data_ptr<float>(),
        H, compressed_len, seq_len, max_windows,
        num_tokens_without_eviction, omega, sink_tokens
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// In-place variant — caller pre-allocates and pre-zeros `out` [H, max_windows].
// Eliminates cudaMalloc + memset overhead on every eviction cycle.
void fused_scoreboard_scatter_inplace(
    torch::Tensor votes,         // [H, compressed_len] float32 contiguous
    torch::Tensor logical_ids,   // [H, compressed_len] int64  contiguous
    torch::Tensor out            // [H, max_windows]  float32  pre-zeroed
) {
    TORCH_CHECK(votes.is_cuda() && logical_ids.is_cuda() && out.is_cuda(),
                "all tensors must be CUDA tensors");
    TORCH_CHECK(votes.is_contiguous() && logical_ids.is_contiguous() && out.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(votes.scalar_type() == torch::kFloat32, "votes must be float32");
    TORCH_CHECK(logical_ids.scalar_type() == torch::kInt64, "logical_ids must be int64");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");

    int H = votes.size(0);
    int compressed_len = votes.size(1);
    int max_windows = out.size(1);
    int total = H * compressed_len;
    if (total == 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    scoreboard_scatter_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        votes.data_ptr<float>(),
        logical_ids.data_ptr<long>(),
        out.data_ptr<float>(),
        H, compressed_len, max_windows
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> fused_quantize_k_int8(
    torch::Tensor input  
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be a 4D tensor");
    TORCH_CHECK(input.size(3) % 2 == 0, "D dimension must be divisible by 2 for RoPE pairing");

    int H = input.size(0);
    int W = input.size(1);
    int omega = input.size(2);
    int D = input.size(3);
    int HW = H * W;

    auto output = torch::zeros({H, W, omega, D}, input.options().dtype(torch::kUInt8));
    auto scale  = torch::zeros({H, W, 1, D}, input.options());
    auto zp     = torch::zeros({H, W, 1, D}, input.options());

    if (HW == 0 || omega == 0 || D <= 0) return {output, scale, zp};

    int threads = std::min(D, 256);
    dim3 grid(HW, (D + threads - 1) / threads);

    DISPATCH_HALF_AND_BFLOAT16(input.scalar_type(), fused_quantize_k_int8, [&] {
        quantize_k_int8_kernel<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<uint8_t>(),
            scale.data_ptr<scalar_t>(),
            zp.data_ptr<scalar_t>(),
            HW, omega, D
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return {output, scale, zp};
}

std::vector<torch::Tensor> fused_quantize_v_int8(
    torch::Tensor input  
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be a 4D tensor");

    int H = input.size(0);
    int W = input.size(1);
    int omega = input.size(2);
    int D = input.size(3);
    int total_rows = H * W * omega;

    auto output = torch::zeros({H, W, omega, D}, input.options().dtype(torch::kUInt8));
    auto scale  = torch::zeros({H, W, omega, 1}, input.options());
    auto zp     = torch::zeros({H, W, omega, 1}, input.options());

    if (total_rows == 0 || D <= 0) return {output, scale, zp};

    int threads = std::min(D, 256);
    int red_threads = 1;
    while (red_threads < threads) red_threads <<= 1;
    if (red_threads > 256) red_threads = 256;

    DISPATCH_HALF_AND_BFLOAT16(input.scalar_type(), fused_quantize_v_int8, [&] {
        quantize_v_int8_kernel<scalar_t><<<total_rows, red_threads, 2 * red_threads * sizeof(float),
            at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<uint8_t>(),
            scale.data_ptr<scalar_t>(),
            zp.data_ptr<scalar_t>(),
            total_rows, D
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return {output, scale, zp};
}

torch::Tensor fused_dequantize_int8(
    torch::Tensor quant,   
    torch::Tensor scale,   
    torch::Tensor zp,      
    bool per_channel,       
    int omega               
) {
    TORCH_CHECK(quant.is_cuda() && scale.is_cuda() && zp.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(quant.is_contiguous() && scale.is_contiguous() && zp.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(quant.scalar_type() == torch::kUInt8, "quant must be uint8");
    TORCH_CHECK(scale.scalar_type() == zp.scalar_type(), "scale and zp must have the same dtype");
    TORCH_CHECK(omega > 0, "omega must be strictly positive");

    auto output = torch::zeros_like(quant, scale.options());
    int N = quant.numel() / quant.size(-1);
    int D = quant.size(-1);
    int scale_stride_last = per_channel ? D : 1;

    int total = N * D;
    if (total == 0 || D <= 0) return output;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    DISPATCH_HALF_AND_BFLOAT16(scale.scalar_type(), fused_dequantize_int8, [&] {
        dequantize_int8_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            quant.data_ptr<uint8_t>(),
            scale.data_ptr<scalar_t>(),
            zp.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, D, scale_stride_last, omega
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return output;
}

void fused_eviction_copy_sinks(
    torch::Tensor old_k, torch::Tensor old_v,
    torch::Tensor new_k, torch::Tensor new_v,
    torch::Tensor old_lid, torch::Tensor new_lid,
    int sink_tokens
) {
    TORCH_CHECK(old_k.is_cuda() && new_k.is_cuda() && old_lid.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(old_v.is_cuda() && new_v.is_cuda() && new_lid.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(old_k.is_contiguous() && new_k.is_contiguous(), "K tensors must be contiguous");
    TORCH_CHECK(old_v.is_contiguous() && new_v.is_contiguous(), "V tensors must be contiguous");
    TORCH_CHECK(old_lid.is_contiguous() && new_lid.is_contiguous(), "lid tensors must be contiguous");
    TORCH_CHECK(old_k.scalar_type() == old_v.scalar_type() && old_k.scalar_type() == new_k.scalar_type() && old_k.scalar_type() == new_v.scalar_type(), "KV tensors must have the same dtype");
    TORCH_CHECK(old_lid.scalar_type() == torch::kInt64 && new_lid.scalar_type() == torch::kInt64, "lid tensors must be int64");

    int H = old_k.size(0);
    int D = old_k.size(2);
    int old_seq = old_k.size(1);
    int new_seq = new_k.size(1);

    int total = H * sink_tokens * D;
    if (total == 0 || D <= 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    DISPATCH_HALF_AND_BFLOAT16(old_k.scalar_type(), fused_eviction_copy_sinks, [&] {
        eviction_copy_sinks_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            old_k.data_ptr<scalar_t>(), old_v.data_ptr<scalar_t>(),
            new_k.data_ptr<scalar_t>(), new_v.data_ptr<scalar_t>(),
            old_lid.data_ptr<long>(), new_lid.data_ptr<long>(),
            H, sink_tokens, D, old_seq, new_seq
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

void fused_eviction_copy_sticky(
    torch::Tensor old_k, torch::Tensor old_v,
    torch::Tensor new_k, torch::Tensor new_v,
    torch::Tensor first_phys, torch::Tensor found_mask,
    torch::Tensor final_ids, torch::Tensor new_lid,
    int curr_k, int omega, int sink_tokens
) {
    TORCH_CHECK(old_k.is_cuda() && new_k.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(old_v.is_cuda() && new_v.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(first_phys.is_cuda() && found_mask.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(final_ids.is_cuda() && new_lid.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(old_k.is_contiguous() && new_k.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(old_v.is_contiguous() && new_v.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(first_phys.is_contiguous() && found_mask.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(final_ids.is_contiguous() && new_lid.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(old_k.scalar_type() == old_v.scalar_type() && old_k.scalar_type() == new_k.scalar_type() && old_k.scalar_type() == new_v.scalar_type(), "KV tensors must have the same dtype");
    TORCH_CHECK(first_phys.scalar_type() == torch::kInt64 && final_ids.scalar_type() == torch::kInt64 && new_lid.scalar_type() == torch::kInt64, "ID tensors must be int64");
    TORCH_CHECK(found_mask.scalar_type() == torch::kBool, "found_mask must be bool");

    int H = old_k.size(0);
    int D = old_k.size(2);
    int old_seq = old_k.size(1);
    int new_seq = new_k.size(1);

    int total_blocks_needed = H * curr_k * omega;
    if (total_blocks_needed == 0 || D <= 0) return;

    int threads = std::min(D, 256);

    DISPATCH_HALF_AND_BFLOAT16(old_k.scalar_type(), fused_eviction_copy_sticky, [&] {
        eviction_copy_sticky_kernel<scalar_t><<<total_blocks_needed, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            old_k.data_ptr<scalar_t>(), old_v.data_ptr<scalar_t>(),
            new_k.data_ptr<scalar_t>(), new_v.data_ptr<scalar_t>(),
            first_phys.data_ptr<long>(), found_mask.data_ptr<bool>(),
            final_ids.data_ptr<long>(), new_lid.data_ptr<long>(),
            H, curr_k, omega, D, old_seq, new_seq, sink_tokens
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

void fused_eviction_copy_local(
    torch::Tensor old_k, torch::Tensor old_v,
    torch::Tensor new_k, torch::Tensor new_v,
    torch::Tensor new_lid,
    int local_count, int old_local_start, int new_local_start,
    long local_start_wid, int omega
) {
    TORCH_CHECK(old_k.is_cuda() && new_k.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(old_v.is_cuda() && new_v.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(new_lid.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(old_k.is_contiguous() && new_k.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(old_v.is_contiguous() && new_v.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(new_lid.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(old_k.scalar_type() == old_v.scalar_type() && old_k.scalar_type() == new_k.scalar_type() && old_k.scalar_type() == new_v.scalar_type(), "KV tensors must have the same dtype");
    TORCH_CHECK(new_lid.scalar_type() == torch::kInt64, "new_lid must be int64");

    int H = old_k.size(0);
    int D = old_k.size(2);
    int old_seq = old_k.size(1);
    int new_seq = new_k.size(1);

    int total = H * local_count * D;
    if (total == 0 || D <= 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    DISPATCH_HALF_AND_BFLOAT16(old_k.scalar_type(), fused_eviction_copy_local, [&] {
        eviction_copy_local_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            old_k.data_ptr<scalar_t>(), old_v.data_ptr<scalar_t>(),
            new_k.data_ptr<scalar_t>(), new_v.data_ptr<scalar_t>(),
            new_lid.data_ptr<long>(),
            H, local_count, D, old_seq, new_seq,
            old_local_start, new_local_start,
            local_start_wid, omega
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_scoreboard_scatter", &fused_scoreboard_scatter);
    m.def("fused_scoreboard_scatter_inplace", &fused_scoreboard_scatter_inplace);
    m.def("fused_scoreboard_scatter_new_tokens", &fused_scoreboard_scatter_new_tokens);
    m.def("fused_quantize_k_int8", &fused_quantize_k_int8);
    m.def("fused_quantize_v_int8", &fused_quantize_v_int8);
    m.def("fused_dequantize_int8", &fused_dequantize_int8);
    m.def("fused_eviction_copy_sinks", &fused_eviction_copy_sinks);
    m.def("fused_eviction_copy_sticky", &fused_eviction_copy_sticky);
    m.def("fused_eviction_copy_local", &fused_eviction_copy_local);
}
