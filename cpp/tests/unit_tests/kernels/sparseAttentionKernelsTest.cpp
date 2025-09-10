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
    constexpr int total_sparse_pages = max_batch_size * max_num_pages_per_seq; // Total sparse pages across all batches

    // Create input buffers
    auto kv_page_offsets
        = mBufferManager->gpu(ITensor::makeShape({batch_size, 2, max_num_pages_per_seq}), nvinfer1::DataType::kINT32);
    auto seq_lengths = mBufferManager->gpu(ITensor::makeShape({batch_size}), nvinfer1::DataType::kINT32);
    auto sparse_indices
        = mBufferManager->gpu(ITensor::makeShape({total_sparse_pages, num_head_kv}), nvinfer1::DataType::kINT32);
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
        = mBufferManager->pinned(ITensor::makeShape({total_sparse_pages, num_head_kv}), nvinfer1::DataType::kINT32);
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
                kv_page_offsets_ptr[offset] = 1000 + b * 100 + d * 10 + p; // Test pattern
            }
        }
    }

    // Initialize sequence lengths
    seq_lengths_ptr[0] = 2 * tokens_per_page + 18; // 3 pages for batch 0
    seq_lengths_ptr[1] = 3 * tokens_per_page + 3;  // 4 pages for batch 1

    // Initialize sparse indices with different patterns for different heads
    // Shape: {total_sparse_pages, num_head_kv}
    // Each head can have its own sparse pattern
    int num_sparse_pages = 5;
    int sparse_page_indices[5][4] = {{1, 0, 0, 1}, {2, 1, 1, -1}, {-1, 2, -1, -1}, {0, 1, 2, 3}, {3, 3, 3, -1}};
    for (int page = 0; page < num_sparse_pages; ++page)
    {
        for (int head = 0; head < num_head_kv; ++head)
        {
            int idx = page * num_head_kv + head;
            sparse_indices_ptr[idx] = sparse_page_indices[page][head];
        }
    }

    // Initialize sparse indices offsets
    sparse_indices_offsets_ptr[0] = 0; // Start of batch 0
    sparse_indices_offsets_ptr[1] = 3; // Start of batch 1 (3 sparse pages for batch 0)
    sparse_indices_offsets_ptr[2] = 5; // End (3 sparse pages for batch 1)

    // Copy data to GPU
    mBufferManager->copy(*kv_page_offsets_host, *kv_page_offsets);
    mBufferManager->copy(*seq_lengths_host, *seq_lengths);
    mBufferManager->copy(*sparse_indices_host, *sparse_indices);
    mBufferManager->copy(*sparse_indices_offsets_host, *sparse_indices_offsets);

    // Set up sparse attention parameters
    SparseAttentionParams sparse_params;
    sparse_params.sparse_attn_indices = bufferCast<int>(*sparse_indices);
    sparse_params.sparse_attn_offsets = bufferCast<int>(*sparse_indices_offsets);
    sparse_params.batch_size = batch_size;
    sparse_params.num_head_kv = num_head_kv;
    sparse_params.tokens_per_page = tokens_per_page;
    sparse_params.max_num_pages_per_seq = max_num_pages_per_seq;

    // Launch the kernel
    invokeGatherKvPageOffsets(bufferCast<int32_t>(*output_kv_page_offsets), bufferCast<int32_t>(*output_seq_lengths),
        bufferCast<int32_t>(*kv_page_offsets), bufferCast<int32_t>(*seq_lengths), sparse_params, mStream->get());

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

    // Verify sequence lengths for each head and batch
    int expected_seq_lens[4][2] = {
        {tokens_per_page + 18, tokens_per_page + 3},     // Head 0: batch 0 has 2 pages, batch 1 has 0 pages
        {2 * tokens_per_page + 18, tokens_per_page + 3}, // Head 1: batch 0 has 3 pages, batch 1 has 0 pages
        {2 * tokens_per_page, tokens_per_page + 3},      // Head 2: batch 0 has 2 pages, batch 1 has 0 pages
        {tokens_per_page, 3}                             // Head 3: batch 0 has 2 pages, batch 1 has 0 pages
    };

    for (int h = 0; h < num_head_kv; ++h)
    {
        for (int b = 0; b < batch_size; ++b)
        {
            int seq_len_idx = h * batch_size + b;
            EXPECT_EQ(output_seq_len_ptr[seq_len_idx], expected_seq_lens[h][b])
                << "Sequence length mismatch at head=" << h << ", batch=" << b;
        }
    }

    // Verify gathered KV page offsets
    for (int h = 0; h < num_head_kv; ++h)
    {
        for (int b = 0; b < batch_size; ++b)
        {
            int num_sparse_pages_batch = sparse_indices_offsets_ptr[b + 1] - sparse_indices_offsets_ptr[b];
            for (int d = 0; d < 2; ++d)
            {
                for (int p = 0; p < num_sparse_pages_batch; ++p)
                {
                    // Calculate expected value (from the sparse index)
                    int sparse_idx_global = sparse_indices_offsets_ptr[b] + p;
                    int src_page_idx = sparse_indices_ptr[sparse_idx_global * num_head_kv + h];

                    if (src_page_idx == -1)
                    {
                        continue; // Skip invalid indices
                    }

                    // Calculate output offset
                    size_t output_offset = h * batch_size * 2 * max_num_pages_per_seq + b * 2 * max_num_pages_per_seq
                        + d * max_num_pages_per_seq + p;

                    int expected_value = 1000 + b * 100 + d * 10 + src_page_idx;

                    EXPECT_EQ(output_kv_offsets_ptr[output_offset], expected_value)
                        << "Mismatch at head=" << h << ", batch=" << b << ", dim=" << d << ", page=" << p;
                }
            }
        }
    }
}

} // namespace
