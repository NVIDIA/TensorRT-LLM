#include <gtest/gtest.h>

#include "tensorrt_llm/kernels/sparseAttentionKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include <memory>
#include <vector>

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace
{
class sparseAttentionKernelsTest : public ::testing::Test
{
public:
    void SetUp() override
    {
        mStream = std::make_shared<CudaStream>();
        mBufferManager = std::make_shared<BufferManager>(mStream);
    }

    void TearDown() override {}

protected:
    std::shared_ptr<CudaStream> mStream;
    std::shared_ptr<BufferManager> mBufferManager;
};

TEST_F(sparseAttentionKernelsTest, GatherKvPageOffsetsKernelTest)
{
    // Test parameters
    constexpr int max_batch_size = 4;
    constexpr int batch_size = 2;
    constexpr int num_head_kv = 4;
    constexpr int max_num_pages_per_seq = 8;
    constexpr int tokens_per_page = 64;
    // Batch 0 has 8 sparse tokens, Batch 1 has 6 sparse tokens, total = 14
    constexpr int total_sparse_tokens = 14;

    // Create input buffers
    auto kv_page_offsets
        = mBufferManager->gpu(ITensor::makeShape({batch_size, 2, max_num_pages_per_seq}), nvinfer1::DataType::kINT32);
    auto seq_lengths = mBufferManager->gpu(ITensor::makeShape({batch_size}), nvinfer1::DataType::kINT32);
    // Shape: [num_head_kv, total_sparse_tokens] - flattened across all batches
    auto sparse_indices
        = mBufferManager->gpu(ITensor::makeShape({num_head_kv, total_sparse_tokens}), nvinfer1::DataType::kINT32);
    auto sparse_indices_offsets = mBufferManager->gpu(ITensor::makeShape({batch_size + 1}), nvinfer1::DataType::kINT32);

    // Create output buffers
    auto output_kv_page_offsets = mBufferManager->gpu(
        ITensor::makeShape({num_head_kv, batch_size, 2, max_num_pages_per_seq}), nvinfer1::DataType::kINT32);
    auto output_seq_lengths
        = mBufferManager->gpu(ITensor::makeShape({num_head_kv, batch_size}), nvinfer1::DataType::kINT32);

    // Create pinned host buffers for data initialization
    auto kv_page_offsets_host = mBufferManager->pinned(
        ITensor::makeShape({batch_size, 2, max_num_pages_per_seq}), nvinfer1::DataType::kINT32);
    auto seq_lengths_host = mBufferManager->pinned(ITensor::makeShape({batch_size}), nvinfer1::DataType::kINT32);
    auto sparse_indices_host
        = mBufferManager->pinned(ITensor::makeShape({num_head_kv, total_sparse_tokens}), nvinfer1::DataType::kINT32);
    auto sparse_indices_offsets_host
        = mBufferManager->pinned(ITensor::makeShape({batch_size + 1}), nvinfer1::DataType::kINT32);

    // Initialize test data
    auto kv_page_offsets_ptr = bufferCast<int32_t>(*kv_page_offsets_host);
    auto seq_lengths_ptr = bufferCast<int>(*seq_lengths_host);
    auto sparse_indices_ptr = bufferCast<int>(*sparse_indices_host);
    auto sparse_indices_offsets_ptr = bufferCast<int>(*sparse_indices_offsets_host);

    // Initialize KV page offsets with test data
    for (int b = 0; b < batch_size; ++b)
    {
        for (int d = 0; d < 2; ++d)
        {
            for (int p = 0; p < max_num_pages_per_seq; ++p)
            {
                int offset = b * 2 * max_num_pages_per_seq + d * max_num_pages_per_seq + p;
                kv_page_offsets_ptr[offset] = 1000 + b * 100 + d * 10 + p;
            }
        }
    }

    // Initialize sequence lengths
    seq_lengths_ptr[0] = 2 * tokens_per_page + 18; // 3 pages (146 tokens) for batch 0
    seq_lengths_ptr[1] = 3 * tokens_per_page + 3;  // 4 pages (195 tokens) for batch 1

    // Initialize sparse indices with token-level indices (indices_block_size = 1)
    // Shape: [num_head_kv, total_sparse_tokens]
    // All heads have the same number of sparse tokens: 8 for batch 0, 6 for batch 1
    // Memory layout: sparse_indices_ptr[head_idx * total_sparse_tokens + token_offset]
    std::vector<std::vector<int>> sparse_tokens_per_head
        = {// Head 0: Batch 0 [10,20,70,75,90,95,100,105] -> pages [0,0,1,1,1,1,1,1] -> unique [0,1]
           //         Batch 1 [64,65,128,129,192,193] -> pages [1,1,2,2,3,3] -> unique [1,2,3]
            {10, 20, 70, 75, 90, 95, 100, 105, 64, 65, 128, 129, 192, 193},

            // Head 1: Batch 0 [5,6,65,66,130,131,135,140] -> pages [0,0,1,1,2,2,2,2] -> unique [0,1,2]
            //         Batch 1 [70,71,128,129,190,191] -> pages [1,1,2,2,2,2] -> unique [1,2]
            {5, 6, 65, 66, 130, 131, 135, 140, 70, 71, 128, 129, 190, 191},

            // Head 2: Batch 0 [20,21,80,81,85,86,90,91] -> pages [0,0,1,1,1,1,1,1] -> unique [0,1]
            //         Batch 1 [64,65,66,67,68,69] -> pages [1,1,1,1,1,1] -> unique [1]
            {20, 21, 80, 81, 85, 86, 90, 91, 64, 65, 66, 67, 68, 69},

            // Head 3: Batch 0 [70,71,72,73,74,75,76,77] -> pages [1,1,1,1,1,1,1,1] -> unique [1]
            //         Batch 1 [192,193,194,195,196,197] -> pages [3,3,3,3,3,3] -> unique [3]
            {70, 71, 72, 73, 74, 75, 76, 77, 192, 193, 194, 195, 196, 197}};

    // Fill sparse_indices_ptr using the defined data
    for (int head = 0; head < num_head_kv; ++head)
    {
        for (int token_idx = 0; token_idx < total_sparse_tokens; ++token_idx)
        {
            sparse_indices_ptr[head * total_sparse_tokens + token_idx] = sparse_tokens_per_head[head][token_idx];
        }
    }

    // Initialize sparse indices offsets (these are per-batch offsets into the flattened array)
    sparse_indices_offsets_ptr[0] = 0;  // Start of batch 0
    sparse_indices_offsets_ptr[1] = 8;  // Start of batch 1 (batch 0 has 8 sparse tokens)
    sparse_indices_offsets_ptr[2] = 14; // End (batch 1 has 6 sparse tokens, total = 14)

    // Copy data to GPU
    mBufferManager->copy(*kv_page_offsets_host, *kv_page_offsets);
    mBufferManager->copy(*seq_lengths_host, *seq_lengths);
    mBufferManager->copy(*sparse_indices_host, *sparse_indices);
    mBufferManager->copy(*sparse_indices_offsets_host, *sparse_indices_offsets);

    SparseAttentionParams sparse_params;
    sparse_params.sparse_attn_indices = bufferCast<int32_t>(*sparse_indices);
    sparse_params.sparse_attn_offsets = bufferCast<int32_t>(*sparse_indices_offsets);
    sparse_params.sparse_attn_indices_block_size = 1; // Token-level indexing
    sparse_params.sparse_attn_indices_stride = total_sparse_tokens;

    // Launch the kernel
    invokeGatherKvPageOffsets(bufferCast<int32_t>(*output_kv_page_offsets), bufferCast<int32_t>(*output_seq_lengths),
        bufferCast<int32_t>(*kv_page_offsets), bufferCast<int32_t>(*seq_lengths), sparse_params, batch_size,
        num_head_kv, tokens_per_page, max_num_pages_per_seq, mStream->get());

    // Wait for completion
    mStream->synchronize();

    // Copy results back to host for verification
    auto output_kv_page_offsets_host = mBufferManager->pinned(
        ITensor::makeShape({num_head_kv, batch_size, 2, max_num_pages_per_seq}), nvinfer1::DataType::kINT32);
    auto output_seq_lengths_host
        = mBufferManager->pinned(ITensor::makeShape({num_head_kv, batch_size}), nvinfer1::DataType::kINT32);

    mBufferManager->copy(*output_kv_page_offsets, *output_kv_page_offsets_host);
    mBufferManager->copy(*output_seq_lengths, *output_seq_lengths_host);

    // Wait for completion
    mStream->synchronize();

    auto output_kv_offsets_ptr = bufferCast<int32_t>(*output_kv_page_offsets_host);
    auto output_seq_len_ptr = bufferCast<int>(*output_seq_lengths_host);

    // Define expected results for each head and batch
    // Format: {num_pages, {page_indices...}, seq_len}
    struct ExpectedResult
    {
        int num_pages;
        std::vector<int> page_indices;
        int seq_len;
    };

    ExpectedResult expected_results[4][2] = {// Head 0
        {                                    // Batch 0: tokens on pages [0,1] -> 2 pages, seq_len = 2 * 64 = 128
            {2, {0, 1}, 2 * tokens_per_page},
            // Batch 1: tokens on pages [1,2,3] -> 3 pages, max_page=3 (last page)
            // seq_len = 195 - (4-3)*64 = 131 (no padding needed, max_page is last page)
            {3, {1, 2, 3}, 131}},
        // Head 1
        {// Batch 0: tokens on pages [0,1,2] -> 3 pages (all), seq_len = 146
            {3, {0, 1, 2}, 2 * tokens_per_page + 18},
            // Batch 1: tokens on pages [1,2] -> 2 pages, max_page=2 (not last page)
            // seq_len = 195 - (4-2)*64 = 67, padding: 67 + (64-3) = 128
            {2, {1, 2}, 2 * tokens_per_page}},
        // Head 2
        {// Batch 0: tokens on pages [0,1] -> 2 pages, seq_len = 128
            {2, {0, 1}, 2 * tokens_per_page},
            // Batch 1: tokens on page [1] -> 1 page, max_page=1 (not last page)
            // seq_len = 195 - (4-1)*64 = 3, padding: 3 + (64-3) = 64
            {1, {1}, tokens_per_page}},
        // Head 3
        {// Batch 0: tokens on page [1] -> 1 page, seq_len = 64
            {1, {1}, tokens_per_page},
            // Batch 1: tokens on page [3] -> 1 page, max_page=3 (last page)
            // seq_len = 195 - (4-1)*64 = 3 (no padding needed, max_page is last page)
            {1, {3}, 3}}};

    // Verify sequence lengths for each head and batch
    for (int h = 0; h < num_head_kv; ++h)
    {
        for (int b = 0; b < batch_size; ++b)
        {
            int seq_len_idx = h * batch_size + b;
            EXPECT_EQ(output_seq_len_ptr[seq_len_idx], expected_results[h][b].seq_len)
                << "Sequence length mismatch at head=" << h << ", batch=" << b
                << ", expected=" << expected_results[h][b].seq_len << ", got=" << output_seq_len_ptr[seq_len_idx];
        }
    }

    // Verify gathered KV page offsets
    for (int h = 0; h < num_head_kv; ++h)
    {
        for (int b = 0; b < batch_size; ++b)
        {
            auto const& expected = expected_results[h][b];

            for (int d = 0; d < 2; ++d)
            {
                for (int p = 0; p < expected.num_pages; ++p)
                {
                    int src_page_idx = expected.page_indices[p];

                    // Calculate output offset
                    size_t output_offset = h * batch_size * 2 * max_num_pages_per_seq + b * 2 * max_num_pages_per_seq
                        + d * max_num_pages_per_seq + p;

                    int expected_value = 1000 + b * 100 + d * 10 + src_page_idx;

                    EXPECT_EQ(output_kv_offsets_ptr[output_offset], expected_value)
                        << "KV page offset mismatch at head=" << h << ", batch=" << b << ", dim=" << d << ", page=" << p
                        << ", expected_page_idx=" << src_page_idx << ", expected_value=" << expected_value
                        << ", got=" << output_kv_offsets_ptr[output_offset];
                }
            }
        }
    }
}

} // namespace
