#include "tensorrt_llm/kernels/sparseAttentionKernels.h"
#include <cub/cub.cuh>

namespace tensorrt_llm
{
namespace kernels
{
template <int THREADS_PER_BLOCK>
__global__ void gatherKvPageOffsetsKernel(
    int32_t* output_kv_page_offsets, // [num_head_kv, batch_size, 2, max_num_pages_per_seq]
    int32_t* output_seq_lengths,     // [num_head_kv, batch_size]
    int32_t const* kv_page_offsets,  // [batch_size, 2, max_num_pages_per_seq]
    int32_t const* seq_lengths,      // [batch_size]
    SparseAttentionParams sparse_params)
{
    // Each CUDA block processes one sequence from the batch for one head.
    int32_t const head_idx = blockIdx.x;
    int32_t const batch_idx = blockIdx.y;
    int32_t const batch_size = sparse_params.batch_size;
    if (batch_idx >= batch_size)
    {
        return;
    }

    // Shared memory for reduction.
    __shared__ typename cub::BlockReduce<Pair, THREADS_PER_BLOCK>::TempStorage temp_storage;

    // Get the range of sparse indices and the sequence length.
    int32_t const start_offset = sparse_params.sparse_attn_offsets[batch_idx];
    int32_t const end_offset = sparse_params.sparse_attn_offsets[batch_idx + 1];
    int32_t const total_pages = sparse_params.sparse_attn_offsets[batch_size];
    int32_t const num_sparse_pages = end_offset - start_offset;
    int32_t const original_seq_len = seq_lengths[batch_idx];

    // Get global sparse index.
    int32_t const sparse_idx_global = head_idx * total_pages + start_offset;

    // Get the base memory offset. shape: [batch_size, 2, max_num_pages_per_seq]
    int32_t const max_num_pages_per_seq = sparse_params.max_num_pages_per_seq;
    size_t const src_base_offset = (size_t) batch_idx * 2 * max_num_pages_per_seq;
    size_t const dst_base_offset = (size_t) head_idx * batch_size * 2 * max_num_pages_per_seq + src_base_offset;

    // Initialize the local max page index and number of valid pages.
    int32_t local_max_page_index = -1;
    int32_t local_num_valid_pages = 0;

    // Perform the gather operation.
    for (int32_t i = threadIdx.x; i < num_sparse_pages; i += blockDim.x)
    {
        // Get the source idx and offset.
        int32_t const src_idx = sparse_params.sparse_attn_indices[sparse_idx_global + i];
        if (src_idx < 0)
        {
            continue;
        }

        // Update the local max page index.
        local_max_page_index = max(local_max_page_index, src_idx);
        local_num_valid_pages++;

        // Get the source and destination offsets.
        size_t const src_offset_dim0 = src_base_offset + 0 * max_num_pages_per_seq + src_idx;
        size_t const src_offset_dim1 = src_base_offset + 1 * max_num_pages_per_seq + src_idx;
        size_t const dst_offset_dim0 = dst_base_offset + 0 * max_num_pages_per_seq + i;
        size_t const dst_offset_dim1 = dst_base_offset + 1 * max_num_pages_per_seq + i;

        // Perform the gather operation: read from the sparse location and write to the dense location.
        output_kv_page_offsets[dst_offset_dim0] = kv_page_offsets[src_offset_dim0];
        output_kv_page_offsets[dst_offset_dim1] = kv_page_offsets[src_offset_dim1];
    }

    // Reduce the local max page indices and number of valid pages.
    Pair local_pair = {local_max_page_index, local_num_valid_pages};
    Pair result = cub::BlockReduce<Pair, THREADS_PER_BLOCK>(temp_storage).Reduce(local_pair, PairReduceOp());

    // Update sequence length for this head and batch.
    if (threadIdx.x == 0)
    {
        int32_t const max_page_index = result.max_val;
        int32_t const num_valid_pages = result.sum_val;
        int32_t const tokens_per_page = sparse_params.tokens_per_page;
        int32_t const ori_valid_pages = (original_seq_len + tokens_per_page - 1) / tokens_per_page;
        size_t const seq_len_offset = (size_t) head_idx * batch_size + batch_idx;
        if (num_valid_pages > 0)
        {
            int32_t seq_len = original_seq_len - (ori_valid_pages - num_valid_pages) * tokens_per_page;
            if (max_page_index != ori_valid_pages - 1)
            {
                seq_len += tokens_per_page - original_seq_len % tokens_per_page;
            }
            output_seq_lengths[seq_len_offset] = seq_len;
        }
        else
        {
            output_seq_lengths[seq_len_offset] = 0;
        }
    }
}

// Host-side launcher function
void invokeGatherKvPageOffsets(int32_t* output_kv_page_offsets, int32_t* output_seq_lengths,
    int32_t const* kv_page_offsets, int32_t const* seq_lengths, SparseAttentionParams sparse_params,
    cudaStream_t stream)
{
    // The grid.
    dim3 grid(sparse_params.num_head_kv, sparse_params.batch_size, 1);
    // The block.
    dim3 block(256, 1, 1);
    // Shared memory size.
    size_t smem_size = sizeof(Pair) * 256;

    // Launch the kernel.
    gatherKvPageOffsetsKernel<256><<<grid, block, smem_size, stream>>>(
        output_kv_page_offsets, output_seq_lengths, kv_page_offsets, seq_lengths, sparse_params);
}
} // namespace kernels
} // namespace tensorrt_llm
