/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaKernels.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunnerParams.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/prepareCustomMask.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"

namespace
{

using tensorrt_llm::kernels::FmhaKernelType;
using tensorrt_llm::kernels::runPrepareCustomMask;
using tensorrt_llm::kernels::TllmGenFmhaKernelMetaInfo;
using tensorrt_llm::kernels::TllmGenFmhaRunnerParams;
using tensorrt_llm::runtime::BufferManager;
using tensorrt_llm::runtime::CudaStream;
using tensorrt_llm::runtime::MemoryType;
using tensorrt_llm::runtime::bufferCast;

inline int32_t ceilDiv(int32_t dividend, int32_t divisor)
{
    return (dividend + divisor - 1) / divisor;
}

// CPU reference implementation for preparing custom mask buffers
std::tuple<std::vector<uint32_t>, std::vector<int64_t>, std::vector<int32_t>> prepareCustomMaskBuffersCPU(
    int32_t batchSize, int32_t numHeadsQPerKv, int32_t tileSizeQ, int32_t tileSizeKv, int32_t numInstsQ,
    int32_t numInstsKv, std::vector<int32_t> const& seqLensQ, std::vector<int32_t> const& seqLensKv,
    std::vector<int32_t> const& firstSparseMaskOffsetsKv,
    std::vector<int32_t> const& inputTreeMask) // Non-packed mask [bs, seqLenQ, seqLenQ]
{
    // Pad tileSizeKv to multiple of 32 for keepsMmaAb kernel
    int32_t tileSizeKvPadded = ceilDiv(tileSizeKv, 32) * 32;
    int32_t tileSizeQPerCta = tileSizeQ * numInstsQ;
    int32_t tileSizeKvPerCta = tileSizeKvPadded * numInstsKv;

    std::vector<int32_t> cumSeqLensQ(batchSize + 1, 0);
    for (int32_t i = 0; i < batchSize; ++i)
    {
        cumSeqLensQ[i + 1] = cumSeqLensQ[i] + seqLensQ[i];
    }

    std::vector<int64_t> customMaskOffsets(batchSize, 0);
    std::vector<int32_t> adjustedFirstSparseMaskOffsetsKv(batchSize, 0);

    int64_t totalMaskSize = 0;
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        int32_t seqLenQ = seqLensQ[batchIdx];
        int32_t seqLenKv = seqLensKv[batchIdx];
        int32_t firstSparseMaskOffsetKv = firstSparseMaskOffsetsKv[batchIdx];

        int32_t numTilesQ = ceilDiv(seqLenQ * numHeadsQPerKv, tileSizeQPerCta);
        int32_t firstSparseTile = firstSparseMaskOffsetKv / tileSizeKvPerCta;
        int32_t numCustomMaskTilesKv = ceilDiv(seqLenKv, tileSizeKvPerCta) - firstSparseTile;

        customMaskOffsets[batchIdx] = totalMaskSize;
        adjustedFirstSparseMaskOffsetsKv[batchIdx] = firstSparseTile * tileSizeKvPerCta;

        int64_t maskSize = static_cast<int64_t>(numTilesQ) * numCustomMaskTilesKv * numInstsQ * numInstsKv
            * (tileSizeQ * tileSizeKvPadded) / 32;
        totalMaskSize += maskSize;
    }

    std::vector<uint32_t> customMask(totalMaskSize, 0);

    // Fill custom mask from input packed mask
    for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        int32_t seqLenQ = seqLensQ[batchIdx];
        int32_t seqLenKv = seqLensKv[batchIdx];
        int32_t firstSparseMaskOffsetKv = firstSparseMaskOffsetsKv[batchIdx];
        int32_t adjustedFirstSparseMaskOffsetKv = adjustedFirstSparseMaskOffsetsKv[batchIdx];

        int32_t numTilesQ = ceilDiv(seqLenQ * numHeadsQPerKv, tileSizeQPerCta);
        int32_t firstSparseTile = firstSparseMaskOffsetKv / tileSizeKvPerCta;
        int32_t numCustomMaskTilesKv = ceilDiv(seqLenKv, tileSizeKvPerCta) - firstSparseTile;

        uint32_t* localCustomMask = customMask.data() + customMaskOffsets[batchIdx];

        // Tree mask layout: [bs, seqLenQ, seqLenQ] (non-packed)
        int32_t batchMaskOffset = batchIdx * seqLenQ * seqLenQ;

        for (int32_t tokenIdxQ = 0; tokenIdxQ < seqLenQ; ++tokenIdxQ)
        {
            for (int32_t tokenIdxKv = 0; tokenIdxKv < seqLenKv; ++tokenIdxKv)
            {

                bool randomMask = false;
                if (tokenIdxKv < firstSparseMaskOffsetKv)
                {
                    randomMask = true; // Dense region (always attend)
                }
                else
                {
                    int32_t qPosInTree = tokenIdxKv - firstSparseMaskOffsetKv;
                    if (qPosInTree < seqLenQ)
                    {
                        int32_t maskIdx = batchMaskOffset + tokenIdxQ * seqLenQ + qPosInTree;
                        randomMask = static_cast<bool>(inputTreeMask[maskIdx]);
                    }
                    else
                    {
                        randomMask = false;
                    }
                }

                // Only process custom mask region (excluding dense region before adjustedFirstSparseMaskOffsetKv)
                if (tokenIdxKv >= adjustedFirstSparseMaskOffsetKv)
                {
                    int32_t customMaskTokenIdxKv = tokenIdxKv - adjustedFirstSparseMaskOffsetKv;
                    int32_t tileIdxKv = customMaskTokenIdxKv / tileSizeKvPerCta;
                    int32_t instIdxKv = (customMaskTokenIdxKv % tileSizeKvPerCta) / tileSizeKvPadded;
                    int32_t tokenIdxInTileKv = (customMaskTokenIdxKv % tileSizeKvPerCta) % tileSizeKvPadded;

                    for (int32_t headIdxInGrp = 0; headIdxInGrp < numHeadsQPerKv; ++headIdxInGrp)
                    {
                        int32_t customMaskTokenIdxQ = tokenIdxQ * numHeadsQPerKv + headIdxInGrp;
                        int32_t tileIdxQ = customMaskTokenIdxQ / tileSizeQPerCta;
                        int32_t instIdxQ = (customMaskTokenIdxQ % tileSizeQPerCta) / tileSizeQ;
                        int32_t tokenIdxInTileQ = (customMaskTokenIdxQ % tileSizeQPerCta) % tileSizeQ;

                        // Calculate mask offset
                        int64_t tileOffset = tileIdxQ * numCustomMaskTilesKv + tileIdxKv;
                        int64_t instOffset = tileOffset * numInstsQ * numInstsKv + (instIdxQ * numInstsKv + instIdxKv);
                        int64_t maskOffset = instOffset * tileSizeQ * tileSizeKvPadded
                            + (tokenIdxInTileQ * tileSizeKvPadded + tokenIdxInTileKv);

                        int64_t offsetAsUInt32 = maskOffset >> 5;
                        int64_t bitPosInUInt32 = maskOffset & 0x1F;

                        localCustomMask[offsetAsUInt32] |= (uint32_t(randomMask) << bitPosInUInt32);
                    }
                }
            }
        }
    }

    return std::make_tuple(customMask, customMaskOffsets, adjustedFirstSparseMaskOffsetsKv);
}

class PrepareCustomMaskTest : public ::testing::Test
{
protected:
    static bool shouldSkip()
    {
        return !tensorrt_llm::common::isSM100Family();
    }

    void SetUp() override
    {
        if (shouldSkip())
        {
            GTEST_SKIP() << "Skipping due to not SM100 family GPU";
        }
        mStream = std::make_shared<CudaStream>();
        mBufferManager = std::make_shared<BufferManager>(mStream);
    }

    void TearDown() override
    {
        if (mStream)
        {
            cudaStreamSynchronize(mStream->get());
        }
        cudaDeviceSynchronize();
        mBufferManager.reset();
        mStream.reset();
    }

    void testPrepareCustomMask(int32_t batchSize, int32_t maxSeqLenQ, int32_t maxSeqLenKv, int32_t numHeadsQPerKv,
        int32_t tileSizeQ = 128, int32_t tileSizeKv = 128, int32_t numInstsQ = 2, int32_t numInstsKv = 1)
    {

        std::mt19937 gen(42);
        std::uniform_int_distribution<> seqLenQDist(1, maxSeqLenQ);
        std::uniform_int_distribution<> seqLenKvDist(maxSeqLenQ, maxSeqLenKv);

        std::vector<int32_t> seqLensQ(batchSize);
        std::vector<int32_t> seqLensKv(batchSize);
        std::vector<int32_t> firstSparseMaskOffsetsKv(batchSize);
        std::vector<int32_t> cumSeqLensQ(batchSize + 1, 0);
        std::vector<int32_t> specDecodingGenerationLengths(batchSize);

        // Generate a uniform seqLenQ for all batches
        int32_t uniformSeqLenQ = seqLenQDist(gen);

        for (int32_t i = 0; i < batchSize; ++i)
        {
            seqLensQ[i] = uniformSeqLenQ;
            seqLensKv[i] = seqLenKvDist(gen);
            firstSparseMaskOffsetsKv[i] = seqLensKv[i] - seqLensQ[i];
            cumSeqLensQ[i + 1] = cumSeqLensQ[i] + seqLensQ[i];
            specDecodingGenerationLengths[i] = seqLensQ[i];
        }

        // Generate random tree mask input
        // Non-packed mask shape: [bs, seqLensQ, seqLensQ]
        int32_t totalTreeMaskSize = batchSize * uniformSeqLenQ * uniformSeqLenQ;
        std::vector<int32_t> inputTreeMaskHost(totalTreeMaskSize, 0);
        std::uniform_int_distribution<int32_t> binaryDist(0, 1);

        for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            int32_t batchOffset = batchIdx * uniformSeqLenQ * uniformSeqLenQ;
            for (int32_t i = 0; i < uniformSeqLenQ * uniformSeqLenQ; ++i)
            {
                inputTreeMaskHost[batchOffset + i] = binaryDist(gen); // Random 0 or 1
            }
        }

        // Pack the tree mask for GPU kernel input
        // Packed mask shape: [bs, seqLensQ, ceilDiv(seqLensQ, 32)]
        int32_t const numBitsPerPackedMask = 32;
        int32_t const numPackedMasksPerToken = ceilDiv(uniformSeqLenQ, numBitsPerPackedMask);
        int32_t totalPackedMaskSize = batchSize * uniformSeqLenQ * numPackedMasksPerToken;
        std::vector<int32_t> inputPackedMaskHost(totalPackedMaskSize, 0);

        for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            int32_t treeMaskBatchOffset = batchIdx * uniformSeqLenQ * uniformSeqLenQ;
            int32_t packedBatchOffset = batchIdx * uniformSeqLenQ * numPackedMasksPerToken;

            for (int32_t i = 0; i < uniformSeqLenQ; ++i)
            {
                for (int32_t j = 0; j < numPackedMasksPerToken; ++j)
                {
                    int32_t mask = 0;
                    for (int32_t k = 0; k < numBitsPerPackedMask; ++k)
                    {
                        int32_t const bitIndex = j * numBitsPerPackedMask + k;
                        if (bitIndex < uniformSeqLenQ)
                        {
                            int32_t maskFlag = inputTreeMaskHost[treeMaskBatchOffset + i * uniformSeqLenQ + bitIndex];
                            mask |= (maskFlag << k);
                        }
                    }
                    inputPackedMaskHost[packedBatchOffset + i * numPackedMasksPerToken + j] = mask;
                }
            }
        }

        auto seqLensQDevice = mBufferManager->copyFrom(seqLensQ, MemoryType::kGPU);
        auto seqLensKvDevice = mBufferManager->copyFrom(seqLensKv, MemoryType::kGPU);
        auto cumSeqLensQDevice = mBufferManager->copyFrom(cumSeqLensQ, MemoryType::kGPU);
        auto specDecodingGenerationLengthsDevice
            = mBufferManager->copyFrom(specDecodingGenerationLengths, MemoryType::kGPU);
        auto firstSparseMaskOffsetsKvDevice = mBufferManager->copyFrom(firstSparseMaskOffsetsKv, MemoryType::kGPU);
        auto inputPackedMaskDevice = mBufferManager->copyFrom(inputPackedMaskHost, MemoryType::kGPU);

        // Calculate output buffer sizes using conservative upper bound
        int32_t tileSizeKvPadded = ceilDiv(tileSizeKv, 32) * 32;
        int32_t tileSizeQPerCta = tileSizeQ * numInstsQ;
        int32_t tileSizeKvPerCta = tileSizeKvPadded * numInstsKv;

        // Find max values across all batches
        int32_t actualMaxSeqLenQ = *std::max_element(seqLensQ.begin(), seqLensQ.end());
        int32_t actualMaxSeqLenKv = *std::max_element(seqLensKv.begin(), seqLensKv.end());
        int32_t minFirstSparseMaskOffsetKv
            = *std::min_element(firstSparseMaskOffsetsKv.begin(), firstSparseMaskOffsetsKv.end());

        // Calculate conservative upper bounds
        int32_t maxNumTilesQ = ceilDiv(actualMaxSeqLenQ * numHeadsQPerKv, tileSizeQPerCta);
        int32_t firstSparseTile = minFirstSparseMaskOffsetKv / tileSizeKvPerCta;
        int32_t maxNumCustomMaskTilesKv = ceilDiv(actualMaxSeqLenKv, tileSizeKvPerCta) - firstSparseTile;

        // Total size in uint32 elements
        int64_t totalMaskSize = static_cast<int64_t>(batchSize) * maxNumTilesQ * maxNumCustomMaskTilesKv * numInstsQ
            * numInstsKv * (tileSizeQ * tileSizeKvPadded) / 32;

        auto customMaskOffsetsDevice = mBufferManager->gpu(batchSize, nvinfer1::DataType::kINT64);
        auto customMaskDevice = mBufferManager->gpu(totalMaskSize, nvinfer1::DataType::kINT32);

        // Clear GPU buffers to ensure no stale data from previous tests
        cudaMemsetAsync(bufferCast<int64_t>(*customMaskOffsetsDevice), 0, batchSize * sizeof(int64_t), mStream->get());
        cudaMemsetAsync(bufferCast<int32_t>(*customMaskDevice), 0, totalMaskSize * sizeof(int32_t), mStream->get());
        cudaStreamSynchronize(mStream->get());

        // Setup kernel parameters
        TllmGenFmhaKernelMetaInfo kernelMeta{};
        kernelMeta.mTileSizeQ = tileSizeQ;
        kernelMeta.mTileSizeKv = tileSizeKv;
        kernelMeta.mStepQ = tileSizeQ * numInstsQ;
        kernelMeta.mStepKv = tileSizeKv * numInstsKv;
        kernelMeta.mKernelType = static_cast<int>(FmhaKernelType::KeepsMmaAbForGeneration);

        TllmGenFmhaRunnerParams runnerParams;
        runnerParams.mBatchSize = batchSize;
        runnerParams.mNumHeadsQPerKv = numHeadsQPerKv;
        runnerParams.mMaxSeqLenQ = uniformSeqLenQ; // All batches have same Q length
        runnerParams.mMaxSeqLenKv = *std::max_element(seqLensKv.begin(), seqLensKv.end());
        runnerParams.seqLensKvPtr = bufferCast<int32_t>(*seqLensKvDevice);
        runnerParams.cumSeqLensQPtr = bufferCast<int32_t>(*cumSeqLensQDevice);
        runnerParams.seqlensQPtr = bufferCast<int32_t>(*specDecodingGenerationLengthsDevice);
        runnerParams.firstSparseMaskOffsetsKvPtr = bufferCast<int32_t>(*firstSparseMaskOffsetsKvDevice);
        runnerParams.generalPackedCustoMaskPtr = bufferCast<int32_t>(*inputPackedMaskDevice);
        runnerParams.customMaskOffsetsPtr = bufferCast<int64_t>(*customMaskOffsetsDevice);
        runnerParams.customMaskPtr = reinterpret_cast<uint32_t*>(bufferCast<int32_t>(*customMaskDevice));

        runPrepareCustomMask(kernelMeta, runnerParams, mStream->get());
        cudaError_t cudaErr = cudaStreamSynchronize(mStream->get());
        if (cudaErr != cudaSuccess)
        {
            FAIL() << "CUDA error: " << cudaGetErrorString(cudaErr);
        }

        // Get GPU results
        auto customMaskOffsetsHost = mBufferManager->copyFrom(*customMaskOffsetsDevice, MemoryType::kCPU);
        auto customMaskHost = mBufferManager->copyFrom(*customMaskDevice, MemoryType::kCPU);

        // Run CPU reference with non-packed tree mask
        auto [cpuMask, cpuOffsets, cpuAdjustedOffsets]
            = prepareCustomMaskBuffersCPU(batchSize, numHeadsQPerKv, tileSizeQ, tileSizeKv, numInstsQ, numInstsKv,
                seqLensQ, seqLensKv, firstSparseMaskOffsetsKv, inputTreeMaskHost);

        auto* gpuOffsets = bufferCast<int64_t>(*customMaskOffsetsHost);
        auto* gpuMask = reinterpret_cast<uint32_t*>(bufferCast<int32_t>(*customMaskHost));
        auto firstSparseMaskOffsetsKvHost = mBufferManager->copyFrom(*firstSparseMaskOffsetsKvDevice, MemoryType::kCPU);
        auto* gpuAdjustedOffsets = bufferCast<int32_t>(*firstSparseMaskOffsetsKvHost);

        // Compare only the effective portion
        for (int32_t i = 0; i < cpuMask.size(); ++i)
        {
            EXPECT_EQ(gpuMask[i], cpuMask[i]);
        }

        for (int32_t i = 0; i < cpuOffsets.size(); ++i)
        {
            EXPECT_EQ(gpuOffsets[i], cpuOffsets[i]);
        }
        for (int32_t i = 0; i < cpuAdjustedOffsets.size(); ++i)
        {
            EXPECT_EQ(gpuAdjustedOffsets[i], cpuAdjustedOffsets[i]);
        }
    }

    std::shared_ptr<CudaStream> mStream;
    std::shared_ptr<BufferManager> mBufferManager;
};

TEST_F(PrepareCustomMaskTest, SmallBatch)
{
    testPrepareCustomMask(/* batchSize */ 2,
        /* maxSeqLenQ */ 16,
        /* maxSeqLenKv */ 128,
        /* numHeadsQPerKv */ 4);
}

TEST_F(PrepareCustomMaskTest, MediumBatch)
{
    testPrepareCustomMask(/* batchSize */ 4,
        /* maxSeqLenQ */ 32,
        /* maxSeqLenKv */ 256,
        /* numHeadsQPerKv */ 8);
}

} // namespace
