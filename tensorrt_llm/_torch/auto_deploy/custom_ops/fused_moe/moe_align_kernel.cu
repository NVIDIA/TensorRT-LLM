// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Inspired by vLLM's moe_align_kernel.cu and ported to TensorRT-LLM

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <torch/extension.h>

#define CEILDIV(x, y) (((x) + (y) -1) / (y))
#define WARP_SIZE 32

namespace auto_deploy
{
namespace moe
{

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(scalar_t const* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts, int32_t padded_num_experts,
    int32_t experts_per_warp, int32_t block_size, size_t numel, int32_t* __restrict__ cumsum,
    int32_t max_num_tokens_padded)
{
    extern __shared__ int32_t shared_counts[];

    // Initialize sorted_token_ids with numel
    for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x)
    {
        sorted_token_ids[it] = numel;
    }

    int const warp_id = threadIdx.x / WARP_SIZE;
    int const my_expert_start = warp_id * experts_per_warp;

    for (int i = 0; i < experts_per_warp; ++i)
    {
        if (my_expert_start + i < padded_num_experts)
        {
            shared_counts[warp_id * experts_per_warp + i] = 0;
        }
    }

    __syncthreads();

    const size_t tid = threadIdx.x;
    const size_t stride = blockDim.x;

    for (size_t i = tid; i < numel; i += stride)
    {
        int expert_id = topk_ids[i];
        int warp_idx = expert_id / experts_per_warp;
        int expert_offset = expert_id % experts_per_warp;
        atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
    }

    __syncthreads();

    // Compute prefix sum over token counts per expert
    using BlockScan = cub::BlockScan<int32_t, 1024>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int expert_count = 0;
    int expert_id = threadIdx.x;
    if (expert_id < num_experts)
    {
        int warp_idx = expert_id / experts_per_warp;
        int expert_offset = expert_id % experts_per_warp;
        expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
        expert_count = CEILDIV(expert_count, block_size) * block_size;
    }

    int cumsum_val;
    BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);
    if (expert_id <= num_experts)
    {
        cumsum[expert_id] = cumsum_val;
    }

    if (expert_id == num_experts)
    {
        *total_tokens_post_pad = cumsum_val;
    }

    __syncthreads();

    if (threadIdx.x < num_experts)
    {
        for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size)
        {
            expert_ids[i / block_size] = threadIdx.x;
        }
    }

    // Fill remaining expert_ids with 0
    const size_t fill_start_idx = cumsum[num_experts] / block_size + threadIdx.x;
    const size_t expert_ids_size = CEILDIV(max_num_tokens_padded, block_size);
    for (size_t i = fill_start_idx; i < expert_ids_size; i += blockDim.x)
    {
        expert_ids[i] = 0;
    }
}

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(scalar_t const* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ cumsum_buffer, size_t numel)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < numel; i += stride)
    {
        int32_t expert_id = topk_ids[i];
        int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
        sorted_token_ids[rank_post_pad] = i;
    }
}

template <typename scalar_t>
__global__ void moe_align_block_size_small_batch_expert_kernel(scalar_t const* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts, int32_t block_size, size_t numel,
    int32_t max_num_tokens_padded)
{
    // Initialize sorted_token_ids with numel
    for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x)
    {
        sorted_token_ids[it] = numel;
    }

    const size_t tid = threadIdx.x;
    const size_t stride = blockDim.x;

    extern __shared__ int32_t shared_mem[];
    int32_t* cumsum = shared_mem;
    int32_t* tokens_cnts = (int32_t*) (shared_mem + num_experts + 1);

    for (int i = 0; i < num_experts; ++i)
    {
        tokens_cnts[(threadIdx.x + 1) * num_experts + i] = 0;
    }

    for (size_t i = tid; i < numel; i += stride)
    {
        ++tokens_cnts[(threadIdx.x + 1) * num_experts + topk_ids[i]];
    }

    __syncthreads();

    if (threadIdx.x < num_experts)
    {
        tokens_cnts[threadIdx.x] = 0;
        for (int i = 1; i <= blockDim.x; ++i)
        {
            tokens_cnts[i * num_experts + threadIdx.x] += tokens_cnts[(i - 1) * num_experts + threadIdx.x];
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        cumsum[0] = 0;
        for (int i = 1; i <= num_experts; ++i)
        {
            cumsum[i] = cumsum[i - 1] + CEILDIV(tokens_cnts[blockDim.x * num_experts + i - 1], block_size) * block_size;
        }
        *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
    }

    __syncthreads();

    if (threadIdx.x < num_experts)
    {
        for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size)
        {
            expert_ids[i / block_size] = threadIdx.x;
        }
    }

    // Fill remaining expert_ids with 0
    const size_t fill_start_idx = cumsum[num_experts] / block_size + threadIdx.x;
    const size_t expert_ids_size = CEILDIV(max_num_tokens_padded, block_size);
    for (size_t i = fill_start_idx; i < expert_ids_size; i += blockDim.x)
    {
        expert_ids[i] = 0;
    }

    for (size_t i = tid; i < numel; i += stride)
    {
        int32_t expert_id = topk_ids[i];
        int32_t rank_post_pad = tokens_cnts[threadIdx.x * num_experts + expert_id] + cumsum[expert_id];
        sorted_token_ids[rank_post_pad] = i;
        ++tokens_cnts[threadIdx.x * num_experts + expert_id];
    }
}

// CuTeDSL-optimized kernel: outputs token IDs, sorted weights, and cu_seqlens directly
template <typename scalar_t, typename weight_t>
__global__ void moe_align_cutedsl_kernel(scalar_t const* __restrict__ topk_ids,
    weight_t const* __restrict__ topk_weights, int32_t* __restrict__ sorted_token_ids,
    weight_t* __restrict__ sorted_weights, int32_t* __restrict__ cu_seqlens,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts, int32_t top_k, size_t numel,
    int32_t max_num_tokens_padded)
{
    extern __shared__ int32_t shared_mem[];
    int32_t* shared_counts = shared_mem;
    int32_t* cumsum = shared_mem + num_experts;

    // Initialize output arrays for padding entries
    // Padding entries have token_id=0 and weight=0 so they don't affect results
    for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x)
    {
        sorted_token_ids[it] = 0;                      // Valid token ID for safe indexing
        sorted_weights[it] = static_cast<weight_t>(0); // Zero weight for padding
    }

    // Initialize shared memory counts
    if (threadIdx.x < num_experts)
    {
        shared_counts[threadIdx.x] = 0;
    }
    __syncthreads();

    // Count tokens per expert
    const size_t tid = threadIdx.x;
    const size_t stride = blockDim.x;
    for (size_t i = tid; i < numel; i += stride)
    {
        int expert_id = topk_ids[i];
        atomicAdd(&shared_counts[expert_id], 1);
    }
    __syncthreads();

    // Compute cumulative sum (cu_seqlens)
    using BlockScan = cub::BlockScan<int32_t, 1024>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int expert_count = 0;
    if (threadIdx.x < num_experts)
    {
        expert_count = shared_counts[threadIdx.x];
    }

    int cumsum_val;
    BlockScan(temp_storage).ExclusiveSum(expert_count, cumsum_val);

    // Write cu_seqlens (this is what CuTeDSL needs!)
    if (threadIdx.x <= num_experts)
    {
        cu_seqlens[threadIdx.x] = cumsum_val;
        cumsum[threadIdx.x] = cumsum_val;
    }

    if (threadIdx.x == num_experts)
    {
        *total_tokens_post_pad = cumsum_val;
    }
    __syncthreads();

    // Sort tokens and weights by expert
    // Each thread processes tokens and stores token_id (not flattened index) and weight
    for (size_t i = tid; i < numel; i += stride)
    {
        int32_t expert_id = topk_ids[i];
        int32_t token_id = i / top_k; // Convert flattened index to token ID
        weight_t weight = topk_weights[i];

        // Atomically get position in sorted array for this expert
        int32_t rank = atomicAdd(&cumsum[expert_id], 1);

        // Store token ID (not flattened index!) and weight
        sorted_token_ids[rank] = token_id;
        sorted_weights[rank] = weight;
    }
}

} // namespace moe
} // namespace auto_deploy

void moe_align_block_size_cuda(torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    torch::Tensor sorted_token_ids, torch::Tensor experts_ids, torch::Tensor num_tokens_post_pad)
{

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int64_t padded_num_experts = ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    int experts_per_warp = WARP_SIZE;
    int threads = 1024;
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    // BlockScan uses 1024 threads and assigns one thread per expert.
    TORCH_CHECK(padded_num_experts < 1024, "padded_num_experts must be less than 1024");

    AT_DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_block_size_kernel",
        [&]
        {
            // calc needed amount of shared mem for `cumsum` tensors
            auto options_int = torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
            torch::Tensor cumsum_buffer = torch::empty({num_experts + 1}, options_int);
            bool small_batch_expert_mode = (topk_ids.numel() < 1024) && (num_experts <= 64);

            if (small_batch_expert_mode)
            {
                const int32_t threads = std::max((int32_t) num_experts, (int32_t) WARP_SIZE);
                const int32_t shared_mem_size = ((threads + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);

                auto small_batch_expert_kernel
                    = auto_deploy::moe::moe_align_block_size_small_batch_expert_kernel<scalar_t>;
                small_batch_expert_kernel<<<1, threads, shared_mem_size, stream>>>(topk_ids.data_ptr<scalar_t>(),
                    sorted_token_ids.data_ptr<int32_t>(), experts_ids.data_ptr<int32_t>(),
                    num_tokens_post_pad.data_ptr<int32_t>(), num_experts, block_size, topk_ids.numel(),
                    sorted_token_ids.size(0));
            }
            else
            {
                auto align_kernel = auto_deploy::moe::moe_align_block_size_kernel<scalar_t>;

                size_t num_warps = CEILDIV(padded_num_experts, experts_per_warp);
                size_t shared_mem_size = num_warps * experts_per_warp * sizeof(int32_t);

                align_kernel<<<1, threads, shared_mem_size, stream>>>(topk_ids.data_ptr<scalar_t>(),
                    sorted_token_ids.data_ptr<int32_t>(), experts_ids.data_ptr<int32_t>(),
                    num_tokens_post_pad.data_ptr<int32_t>(), num_experts, padded_num_experts, experts_per_warp,
                    block_size, topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>(), sorted_token_ids.size(0));

                const int block_threads = std::min(256, (int) threads);
                const int num_blocks = (topk_ids.numel() + block_threads - 1) / block_threads;
                const int max_blocks = 65535;
                const int actual_blocks = std::min(num_blocks, max_blocks);

                auto sort_kernel = auto_deploy::moe::count_and_sort_expert_tokens_kernel<scalar_t>;
                sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(topk_ids.data_ptr<scalar_t>(),
                    sorted_token_ids.data_ptr<int32_t>(), cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel());
            }
        });
}

void moe_align_cutedsl_cuda(torch::Tensor topk_ids, torch::Tensor topk_weights, int64_t num_experts, int64_t top_k,
    torch::Tensor sorted_token_ids, torch::Tensor sorted_weights, torch::Tensor cu_seqlens,
    torch::Tensor num_tokens_post_pad)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Use 1024 threads for BlockScan (supports up to 1024 experts)
    int threads = 1024;
    TORCH_CHECK(num_experts <= 1024, "num_experts must be <= 1024 for CuTeDSL kernel");

    // Shared memory: counts[num_experts] + cumsum[num_experts + 1]
    size_t shared_mem_size = (num_experts + num_experts + 1) * sizeof(int32_t);

    AT_DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_cutedsl_kernel",
        [&]
        {
            // Dispatch based on weight dtype
            if (topk_weights.scalar_type() == torch::kFloat32)
            {
                auto kernel = auto_deploy::moe::moe_align_cutedsl_kernel<scalar_t, float>;
                kernel<<<1, threads, shared_mem_size, stream>>>(topk_ids.data_ptr<scalar_t>(),
                    topk_weights.data_ptr<float>(), sorted_token_ids.data_ptr<int32_t>(),
                    sorted_weights.data_ptr<float>(), cu_seqlens.data_ptr<int32_t>(),
                    num_tokens_post_pad.data_ptr<int32_t>(), num_experts, top_k, topk_ids.numel(),
                    sorted_token_ids.size(0));
            }
            else if (topk_weights.scalar_type() == torch::kFloat16)
            {
                auto kernel = auto_deploy::moe::moe_align_cutedsl_kernel<scalar_t, at::Half>;
                kernel<<<1, threads, shared_mem_size, stream>>>(topk_ids.data_ptr<scalar_t>(),
                    topk_weights.data_ptr<at::Half>(), sorted_token_ids.data_ptr<int32_t>(),
                    sorted_weights.data_ptr<at::Half>(), cu_seqlens.data_ptr<int32_t>(),
                    num_tokens_post_pad.data_ptr<int32_t>(), num_experts, top_k, topk_ids.numel(),
                    sorted_token_ids.size(0));
            }
            else if (topk_weights.scalar_type() == torch::kBFloat16)
            {
                auto kernel = auto_deploy::moe::moe_align_cutedsl_kernel<scalar_t, at::BFloat16>;
                kernel<<<1, threads, shared_mem_size, stream>>>(topk_ids.data_ptr<scalar_t>(),
                    topk_weights.data_ptr<at::BFloat16>(), sorted_token_ids.data_ptr<int32_t>(),
                    sorted_weights.data_ptr<at::BFloat16>(), cu_seqlens.data_ptr<int32_t>(),
                    num_tokens_post_pad.data_ptr<int32_t>(), num_experts, top_k, topk_ids.numel(),
                    sorted_token_ids.size(0));
            }
            else
            {
                TORCH_CHECK(false, "Unsupported weight dtype for CuTeDSL kernel");
            }
        });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("moe_align_block_size", &moe_align_block_size_cuda, "MoE align block size (CUDA)");
    m.def("moe_align_cutedsl", &moe_align_cutedsl_cuda, "MoE align for CuTeDSL (CUDA)");
}
