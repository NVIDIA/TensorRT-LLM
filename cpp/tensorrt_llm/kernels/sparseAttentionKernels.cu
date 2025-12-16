/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/sparseAttentionKernels.h"
#include <cub/cub.cuh>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
template <int THREADS_PER_BLOCK, int MAX_NUM_PAGES>
__global__ void gatherKvPageOffsetsKernel(
    int32_t* output_kv_page_offsets, // [num_head_kv, batch_size, 2, max_num_pages_per_seq]
    int32_t* output_seq_lengths,     // [num_head_kv, batch_size]
    int32_t const* kv_page_offsets,  // [batch_size, 2, max_num_pages_per_seq]
    int32_t const* seq_lengths,      // [batch_size]
    SparseAttentionParams const sparse_params, int32_t const batch_size, int32_t const tokens_per_page,
    int32_t const max_num_pages_per_seq)
{
    // Each CUDA block processes one sequence from the batch for one head.
    int32_t const head_idx = blockIdx.x;
    int32_t const batch_idx = blockIdx.y;
    int32_t const indices_block_size = sparse_params.sparse_attn_indices_block_size;
    if (batch_idx >= batch_size)
    {
        return;
    }

    using BlockScan = cub::BlockScan<int32_t, THREADS_PER_BLOCK>;
    using BlockReduce = cub::BlockReduce<Pair, THREADS_PER_BLOCK>;

    __shared__ typename BlockScan::TempStorage temp_storage_scan;
    __shared__ typename BlockReduce::TempStorage temp_storage_reduce;

    __shared__ int32_t s_page_mask[MAX_NUM_PAGES];
    __shared__ int32_t s_cu_page_mask[MAX_NUM_PAGES];
    __shared__ int32_t s_scan_total; // Store total count from scan

    // Get the range of sparse indices and the sequence length.
    int32_t const start_offset = sparse_params.sparse_attn_offsets[batch_idx];
    int32_t const end_offset = sparse_params.sparse_attn_offsets[batch_idx + 1];
    int32_t const sparse_attn_indices_stride = sparse_params.sparse_attn_indices_stride;
    int32_t const num_sparse_indices = end_offset - start_offset;
    int32_t const original_seq_len = seq_lengths[batch_idx];
    int32_t const ori_valid_pages = (original_seq_len + tokens_per_page - 1) / tokens_per_page;
    int32_t const page_loops = (ori_valid_pages + MAX_NUM_PAGES - 1) / MAX_NUM_PAGES;

    // Get global sparse index.
    int32_t const sparse_idx_global = head_idx * sparse_attn_indices_stride + start_offset;

    // Get the base memory offset. shape: [batch_size, 2, max_num_pages_per_seq]
    size_t const src_base_offset = (size_t) batch_idx * 2 * max_num_pages_per_seq;
    size_t const dst_base_offset = (size_t) head_idx * batch_size * 2 * max_num_pages_per_seq + src_base_offset;

    // Initialize the local max page index and number of valid pages.
    int32_t local_max_page_index = -1;
    int32_t local_num_valid_pages = 0;

    int32_t src_page_idx_offset = 0;
    int32_t dst_page_idx_offset = 0;
    for (int32_t loop_idx = 0; loop_idx < page_loops; loop_idx++)
    {
        src_page_idx_offset = loop_idx * MAX_NUM_PAGES;
        int32_t loop_num_valid_pages = min(MAX_NUM_PAGES, ori_valid_pages - src_page_idx_offset);
        for (int32_t i = threadIdx.x; i < MAX_NUM_PAGES; i += blockDim.x)
        {
            s_page_mask[i] = 0;
        }
        __syncthreads();

        for (int32_t i = threadIdx.x; i < num_sparse_indices; i += blockDim.x)
        {
            int32_t const src_idx = sparse_params.sparse_attn_indices[sparse_idx_global + i];
            int32_t const src_idx_start = src_idx * indices_block_size;
            int32_t const src_idx_end = min(src_idx_start + indices_block_size, original_seq_len);
            for (int32_t j = src_idx_start; j < src_idx_end; j++)
            {
                int32_t const src_page_idx = j / tokens_per_page;
                if (src_page_idx >= src_page_idx_offset && src_page_idx < src_page_idx_offset + loop_num_valid_pages)
                {
                    atomicExch(&s_page_mask[src_page_idx - src_page_idx_offset], 1);
                }
            }
        }
        __syncthreads();

        // Handle case when loop_num_valid_pages > blockDim.x by processing in chunks
        int32_t scan_offset = 0;
        int32_t const scan_chunks = (loop_num_valid_pages + blockDim.x - 1) / blockDim.x;

        for (int32_t chunk_idx = 0; chunk_idx < scan_chunks; chunk_idx++)
        {
            int32_t const chunk_start = chunk_idx * blockDim.x;
            int32_t const chunk_size = min((int32_t) blockDim.x, loop_num_valid_pages - chunk_start);

            int32_t thread_data = (threadIdx.x < chunk_size) ? s_page_mask[chunk_start + threadIdx.x] : 0;
            int32_t thread_output;
            int32_t aggregate;

            BlockScan(temp_storage_scan).ExclusiveSum(thread_data, thread_output, aggregate);
            __syncthreads();

            if (threadIdx.x < chunk_size)
            {
                s_cu_page_mask[chunk_start + threadIdx.x] = thread_output + scan_offset;
            }
            __syncthreads();

            // Update scan offset for next chunk
            scan_offset += aggregate;
        }

        if (threadIdx.x == 0)
        {
            s_scan_total = scan_offset;
        }
        __syncthreads();

        // Perform the gather operation.
        for (int32_t i = threadIdx.x; i < loop_num_valid_pages; i += blockDim.x)
        {
            // Skip if the page is not valid.
            if (s_page_mask[i] == 0)
            {
                continue;
            }

            int32_t const src_idx = src_page_idx_offset + i;
            int32_t const dst_idx = dst_page_idx_offset + s_cu_page_mask[i];

            local_max_page_index = max(local_max_page_index, src_idx);
            local_num_valid_pages++;

            size_t const src_offset_dim0 = src_base_offset + 0 * max_num_pages_per_seq + src_idx;
            size_t const src_offset_dim1 = src_base_offset + 1 * max_num_pages_per_seq + src_idx;
            size_t const dst_offset_dim0 = dst_base_offset + 0 * max_num_pages_per_seq + dst_idx;
            size_t const dst_offset_dim1 = dst_base_offset + 1 * max_num_pages_per_seq + dst_idx;

            output_kv_page_offsets[dst_offset_dim0] = kv_page_offsets[src_offset_dim0];
            output_kv_page_offsets[dst_offset_dim1] = kv_page_offsets[src_offset_dim1];
        }
        __syncthreads();

        // Update dst offset using the total count from scan
        dst_page_idx_offset += s_scan_total;
    }

    // Reduce the local max page indices and number of valid pages.
    Pair local_pair = {local_max_page_index, local_num_valid_pages};
    Pair result = BlockReduce(temp_storage_reduce).Reduce(local_pair, PairReduceOp());

    // Update sequence length for this head and batch.
    if (threadIdx.x == 0)
    {
        int32_t const max_page_index = result.max_val;
        int32_t const num_valid_pages = result.sum_val;
        size_t const seq_len_offset = (size_t) head_idx * batch_size + batch_idx;
        int32_t seq_len = 0;
        if (num_valid_pages > 0)
        {
            if (max_page_index == ori_valid_pages - 1)
            {
                seq_len = (num_valid_pages - 1) * tokens_per_page
                    + (original_seq_len - (ori_valid_pages - 1) * tokens_per_page);
            }
            else
            {
                seq_len = num_valid_pages * tokens_per_page;
            }
        }
        output_seq_lengths[seq_len_offset] = seq_len;
    }
}

// Host-side launcher function
void invokeGatherKvPageOffsets(int32_t* output_kv_page_offsets, int32_t* output_seq_lengths,
    int32_t const* kv_page_offsets, int32_t const* seq_lengths, SparseAttentionParams const sparse_params,
    int32_t const batch_size, int32_t const num_head_kv, int32_t const tokens_per_page,
    int32_t const max_num_pages_per_seq, cudaStream_t stream)
{
    // The grid.
    dim3 grid(num_head_kv, batch_size, 1);
    // The block.
    dim3 block(256, 1, 1);

    gatherKvPageOffsetsKernel<256, 512><<<grid, block, 0, stream>>>(output_kv_page_offsets, output_seq_lengths,
        kv_page_offsets, seq_lengths, sparse_params, batch_size, tokens_per_page, max_num_pages_per_seq);
}
} // namespace kernels

TRTLLM_NAMESPACE_END
