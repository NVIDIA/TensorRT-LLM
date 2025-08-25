#include "tensorrt_llm/kernels/sparseAttentionKernels.h"

namespace tensorrt_llm
{
namespace kernels
{
__global__ void gatherKvPageOffsetsKernel(
    int* output_kv_page_offsets, // [num_head_kv, batch_size, 2, max_num_pages_per_seq]
    int* output_seq_lengths,     // [batch_size]
    int const* kv_page_offsets,  // [batch_size, 2, max_num_pages_per_seq]
    int const* seq_lengths,      // [batch_size]
    SparseAttentionParams const& sparse_params)
{
    // Each CUDA block processes one sequence from the batch.
    int const head_idx = blockIdx.x;
    int const batch_idx = blockIdx.y;
    if (batch_idx >= sparse_params.batch_size)
    {
        return;
    }

    // Get the range of sparse indices.
    int const start_offset = sparse_params.sparse_attn_offsets[batch_idx];
    int const end_offset = sparse_params.sparse_attn_offsets[batch_idx + 1];
    int const num_sparse_pages = end_offset - start_offset;

    // Get the base memory offset. shape: [batch_size, 2, max_num_pages_per_seq]
    int const max_num_pages_per_seq = sparse_params.max_num_pages_per_seq;
    size_t const src_base_offset = (size_t) batch_idx * 2 * max_num_pages_per_seq;
    size_t const dst_base_offset
        = (size_t) head_idx * sparse_params.batch_size * 2 * max_num_pages_per_seq + src_base_offset;

    // Set the sequence length.
    if (threadIdx.x == 0)
    {
        int const tokens_per_page = sparse_params.tokens_per_page;
        int const num_pages = (seq_lengths[batch_idx] + tokens_per_page - 1) / tokens_per_page;
        output_seq_lengths[batch_idx] = seq_lengths[batch_idx] - (num_pages - num_sparse_pages) * tokens_per_page;
    }

    // Perform the gather operation.
    for (int i = threadIdx.x; i < num_sparse_pages; i += blockDim.x)
    {
        // Get the source idx and offset.
        int const sparse_idx_global = (start_offset + i) * sparse_params.num_head_kv + head_idx;
        int const src_idx = sparse_params.sparse_attn_indices[sparse_idx_global];
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
void invokeGatherKvPageOffsets(int* output_kv_page_offsets, int* output_seq_lengths, int const* kv_page_offsets,
    int const* seq_lengths, SparseAttentionParams const& sparse_params, cudaStream_t stream)
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
