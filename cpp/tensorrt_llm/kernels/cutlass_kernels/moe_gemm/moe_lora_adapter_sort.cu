/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_lora_adapter_sort.h"

#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels
{

namespace
{

constexpr int kBlockSize = 256;

// Largest histogram (in shared-memory ints) we run without opting into extra
// shared memory. 3 arrays (hist, base, cursor) of (num_slots + 1) ints must fit
// in the default 48 KiB block budget, so num_slots is capped accordingly.
constexpr int kMaxSharedInts = (48 * 1024) / static_cast<int>(sizeof(int));

// One block per expert. Counting-sorts (groups) the permuted rows in the
// expert's block [s, e) by adapter slot, rewriting the forward map for that
// range and the inverse map for the touched rows. expert_first_token_offset is
// not modified - the block boundaries are fixed, only intra-block order changes.
__global__ void moeLoraAdapterRegroupKernel(int* __restrict__ p2u_out, int* __restrict__ u2p,
    int const* __restrict__ p2u_in, int64_t const* __restrict__ expert_first_token_offset,
    int32_t const* __restrict__ token_to_slot, int64_t num_tokens, int num_slots)
{
    int const expert = blockIdx.x;
    int64_t const s = expert_first_token_offset[expert];
    int64_t const e = expert_first_token_offset[expert + 1];
    if (e <= s)
    {
        return;
    }

    // Bin num_slots is the catch-all for no-adapter / out-of-range tokens, so
    // those rows still group together (a rank-0 run for the builder).
    int const num_bins = num_slots + 1;
    extern __shared__ int smem[];
    int* hist = smem;                  // [num_bins]
    int* base = smem + num_bins;       // [num_bins] exclusive prefix of hist
    int* cursor = smem + 2 * num_bins; // [num_bins] atomic placement cursor

    for (int b = threadIdx.x; b < num_bins; b += blockDim.x)
    {
        hist[b] = 0;
    }
    __syncthreads();

    // Phase 1: histogram of adapter slots within this expert block.
    for (int64_t p = s + threadIdx.x; p < e; p += blockDim.x)
    {
        int const u = p2u_in[p];
        int const tok = static_cast<int>(static_cast<int64_t>(u) % num_tokens);
        int const slot = token_to_slot[tok];
        int const bin = (slot >= 0 && slot < num_slots) ? slot : num_slots;
        atomicAdd(&hist[bin], 1);
    }
    __syncthreads();

    // Phase 2: exclusive prefix sum -> per-bin within-block base offset. num_bins
    // is small, so a single-thread scan is cheap and avoids a block-scan dep.
    if (threadIdx.x == 0)
    {
        int acc = 0;
        for (int b = 0; b < num_bins; ++b)
        {
            base[b] = acc;
            cursor[b] = acc;
            acc += hist[b];
        }
    }
    __syncthreads();

    // Phase 3: scatter rows into their slot's contiguous sub-range. Grouping is
    // sufficient (order within a slot is irrelevant downstream), so a plain
    // atomic cursor per bin is used.
    for (int64_t p = s + threadIdx.x; p < e; p += blockDim.x)
    {
        int const u = p2u_in[p];
        int const tok = static_cast<int>(static_cast<int64_t>(u) % num_tokens);
        int const slot = token_to_slot[tok];
        int const bin = (slot >= 0 && slot < num_slots) ? slot : num_slots;
        int const local = atomicAdd(&cursor[bin], 1);
        int64_t const new_pos = s + static_cast<int64_t>(local);
        p2u_out[new_pos] = u;
        u2p[u] = static_cast<int>(new_pos);
    }
}

} // namespace

bool launchMoeLoraAdapterRegroup(int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row,
    int* p2u_scratch, int64_t const* expert_first_token_offset, int32_t const* token_to_slot,
    int num_experts_per_node, int64_t num_tokens, int64_t expanded_num_rows, int num_slots, cudaStream_t stream)
{
    if (num_experts_per_node <= 0 || expanded_num_rows <= 0 || num_tokens <= 0 || num_slots <= 0)
    {
        return false;
    }
    TLLM_CHECK_WITH_INFO(permuted_row_to_unpermuted_row != nullptr, "permuted_row_to_unpermuted_row must be non-null");
    TLLM_CHECK_WITH_INFO(unpermuted_row_to_permuted_row != nullptr, "unpermuted_row_to_permuted_row must be non-null");
    TLLM_CHECK_WITH_INFO(p2u_scratch != nullptr, "p2u_scratch must be non-null");
    TLLM_CHECK_WITH_INFO(expert_first_token_offset != nullptr, "expert_first_token_offset must be non-null");
    TLLM_CHECK_WITH_INFO(token_to_slot != nullptr, "token_to_slot must be non-null");

    int const num_bins = num_slots + 1;
    if (static_cast<int64_t>(num_bins) * 3 > kMaxSharedInts)
    {
        // Too many adapter slots to histogram in the default shared-memory
        // budget; skip regrouping (Piece A still applies with shorter runs).
        return false;
    }
    size_t const smem_bytes = static_cast<size_t>(num_bins) * 3 * sizeof(int);

    // Stable read source: copy the current forward map so the kernel's reads
    // (p2u_scratch) never alias its writes (permuted_row_to_unpermuted_row). The
    // async D2D copy is CUDA-graph-capture-safe and ordered before the kernel on
    // the same stream.
    TLLM_CUDA_CHECK(cudaMemcpyAsync(p2u_scratch, permuted_row_to_unpermuted_row,
        static_cast<size_t>(expanded_num_rows) * sizeof(int), cudaMemcpyDeviceToDevice, stream));

    moeLoraAdapterRegroupKernel<<<num_experts_per_node, kBlockSize, smem_bytes, stream>>>(
        permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row, p2u_scratch, expert_first_token_offset,
        token_to_slot, num_tokens, num_slots);
    sync_check_cuda_error(stream);
    return true;
}

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
