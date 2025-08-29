/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

using namespace tensorrt_llm::kernels;

class SparseKvCacheTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mStream = nullptr;
        TLLM_CUDA_CHECK(cudaStreamCreate(&mStream));

        // Test parameters
        mBatchSize = 2;
        mNumKvHeads = 4;
        mHeadSize = 128;
        mMaxSeqLen = 512;
        mTokensPerBlock = 64;
        mMaxBlocksPerSeq = 8;

        // Allocate test data
        setupTestData();
    }

    void TearDown() override
    {
        cleanup();
        if (mStream)
        {
            TLLM_CUDA_CHECK(cudaStreamDestroy(mStream));
        }
    }

    void setupTestData()
    {
        // Allocate device memory for sparse KV indices and offsets
        size_t sparse_indices_size = 20 * mNumKvHeads * sizeof(int); // 20 sparse tokens max
        size_t sparse_offsets_size = (mBatchSize + 1) * sizeof(int);

        TLLM_CUDA_CHECK(cudaMalloc(&mSparseKvIndicesDevice, sparse_indices_size));
        TLLM_CUDA_CHECK(cudaMalloc(&mSparseKvOffsetsDevice, sparse_offsets_size));
        TLLM_CUDA_CHECK(cudaMalloc(&mSeqLensDevice, mBatchSize * sizeof(int)));
        TLLM_CUDA_CHECK(cudaMalloc(&mCacheSeqLensDevice, mBatchSize * sizeof(int)));

        // Create sparse indices in the correct format: [sparse_token_idx][head_idx]
        // Total sparse tokens: 5 (batch 0) + 3 (batch 1) = 8
        std::vector<int> sparseKvIndicesHost;

        // Batch 0: 5 sparse tokens per head
        std::vector<std::vector<int>> batch0_indices = {
            {1, 2, 3, 4, 5},  // head 0
            {3, 4, 5, 6, 8},  // head 1
            {0, 1, 3, 5, 8},  // head 2
            {1, 3, 5, 10, 11} // head 3
        };

        // Batch 1: 3 sparse tokens per head
        std::vector<std::vector<int>> batch1_indices = {
            {1, 4, 7}, // head 0
            {0, 2, 3}, // head 1
            {1, 2, 7}, // head 2
            {1, 3, 7}  // head 3
        };

        // Flatten in the format: [sparse_token_idx * num_heads + head_idx]
        for (size_t token = 0; token < batch0_indices[0].size(); ++token)
        {
            for (int head = 0; head < mNumKvHeads; ++head)
            {
                sparseKvIndicesHost.push_back(batch0_indices[head][token]);
            }
        }

        for (size_t token = 0; token < batch1_indices[0].size(); ++token)
        {
            for (int head = 0; head < mNumKvHeads; ++head)
            {
                sparseKvIndicesHost.push_back(batch1_indices[head][token]);
            }
        }

        std::vector<int> sparseKvOffsetsHost = {0, 5, 8}; // Batch 0: 5 tokens, Batch 1: 3 tokens
        std::vector<int> seqLensHost = {12, 8};           // Original sequence lengths
        std::vector<int> cacheSeqLensHost = {12, 8};      // Cache sequence lengths

        TLLM_CUDA_CHECK(cudaMemcpy(mSparseKvIndicesDevice, sparseKvIndicesHost.data(),
            sparseKvIndicesHost.size() * sizeof(int), cudaMemcpyHostToDevice));
        TLLM_CUDA_CHECK(cudaMemcpy(mSparseKvOffsetsDevice, sparseKvOffsetsHost.data(),
            sparseKvOffsetsHost.size() * sizeof(int), cudaMemcpyHostToDevice));
        TLLM_CUDA_CHECK(
            cudaMemcpy(mSeqLensDevice, seqLensHost.data(), seqLensHost.size() * sizeof(int), cudaMemcpyHostToDevice));
        TLLM_CUDA_CHECK(cudaMemcpy(mCacheSeqLensDevice, cacheSeqLensHost.data(), cacheSeqLensHost.size() * sizeof(int),
            cudaMemcpyHostToDevice));
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        // Setup KV cache buffer using KVBlockArray
        setupKvCacheBuffer();
    }

    void setupKvCacheBuffer()
    {
        // Calculate memory requirements
        auto const elemSize = sizeof(half);
        auto const sizePerToken = mNumKvHeads * mHeadSize * elemSize;
        auto const bytesPerBlock = mTokensPerBlock * sizePerToken;
        auto const totalBlocks = mBatchSize * mMaxBlocksPerSeq;
        auto const poolSize = totalBlocks * bytesPerBlock * 2; // K and V

        // Allocate primary pool
        TLLM_CUDA_CHECK(cudaMalloc(&mKvCachePool, poolSize));
        TLLM_CUDA_CHECK(cudaMemset(mKvCachePool, 0, poolSize));

        // Allocate block offsets
        size_t offsetsSize = mBatchSize * mMaxBlocksPerSeq * 2 * sizeof(KVCacheIndex);
        TLLM_CUDA_CHECK(cudaMalloc(&mBlockOffsetsDevice, offsetsSize));

        // Initialize block offsets (simple linear mapping for test)
        std::vector<KVCacheIndex> blockOffsetsHost;
        blockOffsetsHost.reserve(mBatchSize * mMaxBlocksPerSeq * 2);

        for (int batch = 0; batch < mBatchSize; ++batch)
        {
            for (int block = 0; block < mMaxBlocksPerSeq; ++block)
            {
                // K cache block offset
                int kBlockIdx = batch * mMaxBlocksPerSeq * 2 + block;
                blockOffsetsHost.emplace_back(kBlockIdx, false);
            }
            for (int block = 0; block < mMaxBlocksPerSeq; ++block)
            {
                // V cache block offset
                int vBlockIdx = batch * mMaxBlocksPerSeq * 2 + mMaxBlocksPerSeq + block;
                blockOffsetsHost.emplace_back(vBlockIdx, false);
            }
        }

        TLLM_CUDA_CHECK(cudaMemcpy(mBlockOffsetsDevice, blockOffsetsHost.data(),
            blockOffsetsHost.size() * sizeof(KVCacheIndex), cudaMemcpyHostToDevice));

        TLLM_CUDA_CHECK(cudaDeviceSynchronize());
        // Create KVBlockArray with correct parameter order:
        // (batchSize, maxBlocksPerSeq, tokensPerBlock, bytesPerToken,
        //  maxAttentionWindow, maxAttentionWindowAllLayer, sinkTokenLen, canUseOneMoreBlock,
        //  primaryPoolPtr, secondaryPoolPtr, data)
        mKvCacheBuffer = KVBlockArray(mBatchSize, mMaxBlocksPerSeq, mTokensPerBlock, sizePerToken, mMaxSeqLen,
            mMaxSeqLen, 0, false, mKvCachePool, nullptr, mBlockOffsetsDevice);
    }

    void cleanup()
    {
        if (mSparseKvIndicesDevice)
            cudaFree(mSparseKvIndicesDevice);
        if (mSparseKvOffsetsDevice)
            cudaFree(mSparseKvOffsetsDevice);
        if (mSeqLensDevice)
            cudaFree(mSeqLensDevice);
        if (mCacheSeqLensDevice)
            cudaFree(mCacheSeqLensDevice);
        if (mKvCachePool)
            cudaFree(mKvCachePool);
        if (mBlockOffsetsDevice)
            cudaFree(mBlockOffsetsDevice);
        if (mQkvInputDevice)
            cudaFree(mQkvInputDevice);
    }

    // Test parameters
    int mBatchSize;
    int mNumKvHeads;
    int mHeadSize;
    int mMaxSeqLen;
    int mTokensPerBlock;
    int mMaxBlocksPerSeq;

    // Device memory
    int* mSparseKvIndicesDevice = nullptr;
    int* mSparseKvOffsetsDevice = nullptr;
    int* mSeqLensDevice = nullptr;
    int* mCacheSeqLensDevice = nullptr;
    void* mKvCachePool = nullptr;
    KVCacheIndex* mBlockOffsetsDevice = nullptr;
    half* mQkvInputDevice = nullptr;

    KVBlockArray mKvCacheBuffer;
    cudaStream_t mStream;

    // Helper functions for verification
    bool verifySparseKvCacheMapping(std::vector<half> const& originalKvCache);
    void performHostSparseMapping(std::vector<half> const& originalKvCache, std::vector<half>& expectedKvCache);
    void extractKvCacheFromGpu(std::vector<half>& kvCacheHost);
    void initializeKvCacheWithPattern();
};

TEST_F(SparseKvCacheTest, UpdateSparseKvCacheAfterFmha)
{
    // Allocate dummy QKV input (normally this would come from the attention computation)
    size_t qkvInputSize = mBatchSize * mMaxSeqLen * 3 * mNumKvHeads * mHeadSize * sizeof(half);
    TLLM_CUDA_CHECK(cudaMalloc(&mQkvInputDevice, qkvInputSize));

    // Initialize with test pattern
    std::vector<half> qkvInputHost(qkvInputSize / sizeof(half));
    for (size_t i = 0; i < qkvInputHost.size(); ++i)
    {
        qkvInputHost[i] = half(float(i % 1000) / 100.0f); // Simple test pattern
    }
    TLLM_CUDA_CHECK(cudaMemcpy(mQkvInputDevice, qkvInputHost.data(), qkvInputSize, cudaMemcpyHostToDevice));

    // Initialize KV cache with initial data pattern for testing
    initializeKvCacheWithPattern();

    // Extract the original KV cache data before kernel execution for verification
    size_t totalKvElements = mBatchSize * mMaxSeqLen * mNumKvHeads * mHeadSize * 2; // K and V
    std::vector<half> originalKvCache(totalKvElements);
    extractKvCacheFromGpu(originalKvCache);

    // Setup QKVPreprocessingParams
    QKVPreprocessingParams<half, KVBlockArray> params;
    memset(&params, 0, sizeof(params));

    params.qkv_input = mQkvInputDevice;
    params.kv_cache_buffer = mKvCacheBuffer;
    params.sparse_kv_indices = mSparseKvIndicesDevice;
    params.sparse_kv_offsets = mSparseKvOffsetsDevice;
    params.seq_lens = mSeqLensDevice;
    params.cache_seq_lens = mCacheSeqLensDevice;

    params.batch_size = mBatchSize;
    params.head_num = mNumKvHeads; // For Q heads, assuming same as KV heads for this test
    params.kv_head_num = mNumKvHeads;
    params.size_per_head = mHeadSize;
    params.cache_type = KvCacheDataType::BASE;
    params.rotary_embedding_dim = 0; // No rotary embedding for this test

    params.setCommonParameters();

    // Verify sparse indices and offsets on host
    std::vector<int> hostSparseIndices(8 * mNumKvHeads);
    TLLM_CUDA_CHECK(cudaMemcpy(hostSparseIndices.data(), mSparseKvIndicesDevice, hostSparseIndices.size() * sizeof(int),
        cudaMemcpyDeviceToHost));

    std::vector<int> hostSparseOffsets(mBatchSize + 1);
    TLLM_CUDA_CHECK(cudaMemcpy(hostSparseOffsets.data(), mSparseKvOffsetsDevice, hostSparseOffsets.size() * sizeof(int),
        cudaMemcpyDeviceToHost));
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t pre_kernel_error = cudaGetLastError();
    if (pre_kernel_error != cudaSuccess)
    {
        printf("Debug: CUDA error before kernel call: %s\n", cudaGetErrorString(pre_kernel_error));
    }

    invokeUpdateSparseKvCacheAfterFmha<half, half, KVBlockArray>(params, mStream);

    TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));
    cudaError_t post_kernel_error = cudaGetLastError();
    if (post_kernel_error != cudaSuccess)
    {
        printf("Debug: CUDA error after kernel call: %s\n", cudaGetErrorString(post_kernel_error));
    }
    else
    {
        printf("Debug: Kernel call completed, no immediate CUDA errors\n");
    }

    // Verification: Compare GPU result with CPU reference implementation
    EXPECT_TRUE(verifySparseKvCacheMapping(originalKvCache));
}

// Implementation of verification functions
bool SparseKvCacheTest::verifySparseKvCacheMapping(std::vector<half> const& originalKvCache)
{
    // Perform host-side sparse mapping to get expected result
    size_t totalKvElements = originalKvCache.size();
    std::vector<half> expectedKvCache{originalKvCache};
    performHostSparseMapping(originalKvCache, expectedKvCache);

    // Extract actual result from GPU after sparse kernel execution
    std::vector<half> actualKvCache(totalKvElements);
    extractKvCacheFromGpu(actualKvCache);

    // Compare results with tolerance - only for valid sparse tokens
    float const tolerance = 1e-5f;
    bool passed = true;
    int errorCount = 0;
    int const maxErrorsToShow = 10;
    size_t totalValidElements = 0;

    // Only compare the sparse tokens that should have been reorganized
    std::vector<int> sparseTokenCounts = {5, 3}; // Batch 0: 5 tokens, Batch 1: 3 tokens

    for (int batch = 0; batch < mBatchSize; ++batch)
    {
        int numSparseTokens = sparseTokenCounts[batch];

        for (int kv = 0; kv < 2; ++kv) // K and V
        {
            for (int token = 0; token < numSparseTokens; ++token)
            {
                for (int head = 0; head < mNumKvHeads; ++head)
                {
                    for (int dim = 0; dim < mHeadSize; ++dim)
                    {
                        // Calculate index in flat array
                        // Layout: [batch][kv][token][head][dim]
                        size_t idx = batch * (2 * mMaxSeqLen * mNumKvHeads * mHeadSize)
                            + kv * (mMaxSeqLen * mNumKvHeads * mHeadSize) + token * (mNumKvHeads * mHeadSize)
                            + head * mHeadSize + dim;

                        if (idx < totalKvElements)
                        {
                            float expected = float(expectedKvCache[idx]);
                            float actual = float(actualKvCache[idx]);
                            float diff = std::abs(expected - actual);

                            if (diff > tolerance)
                            {
                                if (errorCount < maxErrorsToShow)
                                {
                                    printf(
                                        "Mismatch at batch=%d, kv=%d, token=%d, head=%d, dim=%d: expected %.6f, got "
                                        "%.6f, diff %.6f\n",
                                        batch, kv, token, head, dim, expected, actual, diff);
                                }
                                errorCount++;
                                passed = false;
                            }
                            totalValidElements++;
                        }
                    }
                }
            }
        }
    }

    if (errorCount > 0)
    {
        printf("Total errors: %d out of %zu valid sparse token elements\n", errorCount, totalValidElements);
    }
    else
    {
        printf("Verification passed: all %zu valid sparse token elements match within tolerance %.2e\n",
            totalValidElements, tolerance);
    }

    return passed;
}

void SparseKvCacheTest::performHostSparseMapping(
    std::vector<half> const& originalKvCache, std::vector<half>& expectedKvCache)
{
    // Host-side reference implementation of sparse KV cache mapping
    // This is a naive but correct implementation for verification

    // Sparse indices from the test setup - matching the actual flattened format
    std::vector<std::vector<std::vector<int>>> sparseIndices = {// Batch 0
        {
            {1, 2, 3, 4, 5},                                    // head 0
            {3, 4, 5, 6, 8},                                    // head 1
            {0, 1, 3, 5, 8},                                    // head 2
            {1, 3, 5, 10, 11}                                   // head 3
        },
        // Batch 1
        {
            {1, 4, 7}, // head 0
            {0, 2, 3}, // head 1
            {1, 2, 7}, // head 2
            {1, 3, 7}  // head 3
        }};

    // Process each batch
    for (int batch = 0; batch < mBatchSize; ++batch)
    {
        // Process each head
        for (int head = 0; head < mNumKvHeads; ++head)
        {
            auto const& indices = sparseIndices[batch][head];

            // Process both K and V cache
            for (int kv = 0; kv < 2; ++kv) // 0 = K, 1 = V
            {
                // For each sparse token
                for (size_t sparseIdx = 0; sparseIdx < indices.size(); ++sparseIdx)
                {
                    int originalTokenIdx = indices[sparseIdx];
                    int continuousTokenIdx = static_cast<int>(sparseIdx);

                    // Copy from original position to continuous position
                    for (int dim = 0; dim < mHeadSize; ++dim)
                    {
                        // Calculate indices in the flat array
                        // Layout: [batch][kv][token][head][dim]
                        size_t srcIdx = batch * (2 * mMaxSeqLen * mNumKvHeads * mHeadSize)
                            + kv * (mMaxSeqLen * mNumKvHeads * mHeadSize) + originalTokenIdx * (mNumKvHeads * mHeadSize)
                            + head * mHeadSize + dim;

                        size_t dstIdx = batch * (2 * mMaxSeqLen * mNumKvHeads * mHeadSize)
                            + kv * (mMaxSeqLen * mNumKvHeads * mHeadSize)
                            + continuousTokenIdx * (mNumKvHeads * mHeadSize) + head * mHeadSize + dim;

                        if (srcIdx < originalKvCache.size() && dstIdx < expectedKvCache.size())
                        {
                            expectedKvCache[dstIdx] = originalKvCache[srcIdx];
                        }
                    }
                }
            }
        }
    }
}

void SparseKvCacheTest::extractKvCacheFromGpu(std::vector<half>& kvCacheHost)
{
    // Extract KV cache data from GPU KVBlockArray structure
    // This is a simplified extraction for testing purposes

    // Calculate total size needed
    size_t totalElements = mBatchSize * mMaxSeqLen * mNumKvHeads * mHeadSize * 2;
    kvCacheHost.resize(totalElements);

    // For testing, we'll use a simplified approach to read back the cache
    // In a real implementation, this would need to handle the block structure properly

    // Calculate pool size
    auto const elemSize = sizeof(half);
    auto const sizePerToken = mNumKvHeads * mHeadSize * elemSize;
    auto const bytesPerBlock = mTokensPerBlock * sizePerToken;
    auto const totalBlocks = mBatchSize * mMaxBlocksPerSeq;
    auto const poolSize = totalBlocks * bytesPerBlock * 2; // K and V

    // Create temporary buffer to read entire pool
    std::vector<half> poolData(poolSize / sizeof(half));
    TLLM_CUDA_CHECK(cudaMemcpy(poolData.data(), mKvCachePool, poolSize, cudaMemcpyDeviceToHost));
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());
    // Reorganize from block structure to linear structure for comparison
    // This is a simplified mapping - in reality you'd need to handle block indexing
    for (int batch = 0; batch < mBatchSize; ++batch)
    {
        for (int token = 0; token < mMaxSeqLen; ++token)
        {
            for (int head = 0; head < mNumKvHeads; ++head)
            {
                for (int dim = 0; dim < mHeadSize; ++dim)
                {
                    for (int kv = 0; kv < 2; ++kv)
                    {
                        // Calculate block coordinates
                        int blockIdx = token / mTokensPerBlock;
                        int tokenInBlock = token % mTokensPerBlock;

                        // Calculate source index in pool (simplified)
                        // The layout of a block in the pool is [mTokensPerBlock, mNumKvHeads, mHeadSize]
                        size_t block_base_pool_idx
                            = (size_t) (batch * mMaxBlocksPerSeq * 2 + kv * mMaxBlocksPerSeq + blockIdx)
                            * mTokensPerBlock * mNumKvHeads * mHeadSize;

                        size_t inner_block_pool_idx
                            = (size_t) head * mTokensPerBlock * mHeadSize + (size_t) tokenInBlock * mHeadSize + dim;

                        size_t poolIdx = block_base_pool_idx + inner_block_pool_idx;

                        // Calculate destination index in linear layout
                        size_t linearIdx = (size_t) batch * (2 * mMaxSeqLen * mNumKvHeads * mHeadSize)
                            + (size_t) kv * (mMaxSeqLen * mNumKvHeads * mHeadSize)
                            + (size_t) token * (mNumKvHeads * mHeadSize) + (size_t) head * mHeadSize + dim;

                        if (poolIdx < poolData.size() && linearIdx < kvCacheHost.size())
                        {
                            kvCacheHost[linearIdx] = poolData[poolIdx];
                        }
                    }
                }
            }
        }
    }
}

void SparseKvCacheTest::initializeKvCacheWithPattern()
{

    // Calculate pool size
    auto const elemSize = sizeof(half);
    auto const sizePerToken = mNumKvHeads * mHeadSize * elemSize;
    auto const bytesPerBlock = mTokensPerBlock * sizePerToken;
    auto const totalBlocks = mBatchSize * mMaxBlocksPerSeq;
    auto const poolSize = totalBlocks * bytesPerBlock * 2; // K and V

    // Create host data with recognizable pattern
    std::vector<half> poolData(poolSize / sizeof(half));
    for (size_t i = 0; i < poolData.size(); ++i)
    {
        poolData[i] = half(float(i) / 1000.0f);
    }

    // Copy to GPU
    TLLM_CUDA_CHECK(cudaMemcpy(mKvCachePool, poolData.data(), poolSize, cudaMemcpyHostToDevice));
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());
}

// Main function for standalone compilation
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
