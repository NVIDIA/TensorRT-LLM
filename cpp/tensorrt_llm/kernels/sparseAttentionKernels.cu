#include "tensorrt_llm/kernels/sparseAttentionKernels.h"

namespace tensorrt_llm
{
namespace kernels
{
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
    if (batch_idx >= sparse_params.batch_size)
    {
        return;
    }

    // Get the range of sparse indices.
    int32_t const start_offset = sparse_params.sparse_attn_offsets[batch_idx];
    int32_t const end_offset = sparse_params.sparse_attn_offsets[batch_idx + 1];
    int32_t const num_sparse_pages = end_offset - start_offset;

    // Get the base memory offset. shape: [batch_size, 2, max_num_pages_per_seq]
    int32_t const max_num_pages_per_seq = sparse_params.max_num_pages_per_seq;
    size_t const src_base_offset = (size_t) batch_idx * 2 * max_num_pages_per_seq;
    size_t const dst_base_offset
        = (size_t) head_idx * sparse_params.batch_size * 2 * max_num_pages_per_seq + src_base_offset;

    // Initialize sequence length for this head and batch to 0.
    size_t const seq_len_offset = (size_t) head_idx * sparse_params.batch_size + batch_idx;
    if (threadIdx.x == 0)
    {
        output_seq_lengths[seq_len_offset] = 0;
    }
    __syncthreads();

    // Count valid pages and accumulate sequence length as we gather.
    int32_t const tokens_per_page = sparse_params.tokens_per_page;
    int32_t const original_seq_len = seq_lengths[batch_idx];
    int32_t const total_pages = (original_seq_len + tokens_per_page - 1) / tokens_per_page;

    // Perform the gather operation.
    for (int32_t i = threadIdx.x; i < num_sparse_pages; i += blockDim.x)
    {
        // Get the source idx and offset.
        int32_t const sparse_idx_global = (start_offset + i) * sparse_params.num_head_kv + head_idx;
        int32_t const src_idx = sparse_params.sparse_attn_indices[sparse_idx_global];
        if (src_idx == -1)
        {
            continue;
        }

        // Calculate tokens to add for this page
        int32_t tokens_to_add = tokens_per_page;
        // Check if this is the last page of the original sequence
        if (src_idx == total_pages - 1)
        {
            // For the last page, only add the remaining tokens
            int32_t remaining_tokens = original_seq_len - (total_pages - 1) * tokens_per_page;
            tokens_to_add = remaining_tokens;
        }

        // Atomically add tokens to the sequence length
        atomicAdd(&output_seq_lengths[seq_len_offset], tokens_to_add);

        size_t const src_offset_dim0 = src_base_offset + 0 * max_num_pages_per_seq + src_idx;
        size_t const src_offset_dim1 = src_base_offset + 1 * max_num_pages_per_seq + src_idx;

        // Get the destination offset.
        size_t const dst_offset_dim0 = dst_base_offset + 0 * max_num_pages_per_seq + i;
        size_t const dst_offset_dim1 = dst_base_offset + 1 * max_num_pages_per_seq + i;

        // Perform the gather operation: read from the sparse location and write to the dense location.
        output_kv_page_offsets[dst_offset_dim0] = kv_page_offsets[src_offset_dim0];
        output_kv_page_offsets[dst_offset_dim1] = kv_page_offsets[src_offset_dim1];
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

    // Launch the kernel.
    gatherKvPageOffsetsKernel<<<grid, block, 0, stream>>>(
        output_kv_page_offsets, output_seq_lengths, kv_page_offsets, seq_lengths, sparse_params);
}
} // namespace kernels
} // namespace tensorrt_llm
