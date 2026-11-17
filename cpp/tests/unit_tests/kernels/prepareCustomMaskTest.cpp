/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <optional>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaKernels.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunnerParams.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/prepareCustomMask.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"

namespace
{

using tensorrt_llm::kernels::FmhaKernelType;
using tensorrt_llm::kernels::getCustomMaskCapacityUpperBound;
using tensorrt_llm::kernels::getCustomMaskSizeUpperBound;
using tensorrt_llm::kernels::runPrepareCustomMask;
using tensorrt_llm::kernels::sTllmGenFmhaKernelMetaInfos;
using tensorrt_llm::kernels::TllmGenFmhaRunnerParams;
using tensorrt_llm::kernels::TrtllmGenAttentionMaskType;
using tensorrt_llm::runtime::BufferManager;
using tensorrt_llm::runtime::CudaStream;
using tensorrt_llm::runtime::MemoryType;

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

        int32_t const stepQ = tileSizeQ * numInstsQ;
        int32_t const stepKv = tileSizeKv * numInstsKv;
        int64_t const maskCapacity = std::max(totalMaskSize,
            getCustomMaskSizeUpperBound(FmhaKernelType::KeepsMmaAbForGeneration, batchSize, uniformSeqLenQ,
                numHeadsQPerKv, stepQ, stepKv, tileSizeQ, tileSizeKv));

        auto customMaskOffsetsDevice = mBufferManager->gpu(batchSize, tensorrt_llm::DataType::kINT64);
        auto customMaskDevice = mBufferManager->gpu(maskCapacity, tensorrt_llm::DataType::kINT32);

        // Clear GPU buffers to ensure no stale data from previous tests
        cudaMemsetAsync(
            customMaskOffsetsDevice->getBufferPtr<int64_t>(), 0, batchSize * sizeof(int64_t), mStream->get());
        cudaMemsetAsync(customMaskDevice->getBufferPtr<int32_t>(), 0, maskCapacity * sizeof(int32_t), mStream->get());
        cudaStreamSynchronize(mStream->get());

        TllmGenFmhaRunnerParams runnerParams;
        runnerParams.mBatchSize = batchSize;
        runnerParams.mNumHeadsQPerKv = numHeadsQPerKv;
        runnerParams.mMaxSeqLenQ = uniformSeqLenQ; // All batches have same Q length
        runnerParams.mMaxSeqLenKv = *std::max_element(seqLensKv.begin(), seqLensKv.end());
        runnerParams.seqLensKvPtr = seqLensKvDevice->getBufferPtr<int32_t>();
        runnerParams.cumSeqLensQPtr = cumSeqLensQDevice->getBufferPtr<int32_t>();
        runnerParams.seqLensQPtr = specDecodingGenerationLengthsDevice->getBufferPtr<int32_t>();
        runnerParams.firstSparseMaskOffsetsKvPtr = firstSparseMaskOffsetsKvDevice->getBufferPtr<int32_t>();
        runnerParams.generalPackedCustoMaskPtr = inputPackedMaskDevice->getBufferPtr<int32_t>();
        runnerParams.customMaskOffsetsPtr = customMaskOffsetsDevice->getBufferPtr<int64_t>();
        runnerParams.customMaskPtr = reinterpret_cast<uint32_t*>(customMaskDevice->getBufferPtr<int32_t>());
        runnerParams.customMaskCapacity = maskCapacity;

        runPrepareCustomMask(runnerParams, FmhaKernelType::KeepsMmaAbForGeneration, stepQ, stepKv, tileSizeQ,
            tileSizeKv, mStream->get());
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

        auto* gpuOffsets = customMaskOffsetsHost->getBufferPtr<int64_t>();
        auto* gpuMask = reinterpret_cast<uint32_t*>(customMaskHost->getBufferPtr<int32_t>());
        auto firstSparseMaskOffsetsKvHost = mBufferManager->copyFrom(*firstSparseMaskOffsetsKvDevice, MemoryType::kCPU);
        auto* gpuAdjustedOffsets = firstSparseMaskOffsetsKvHost->getBufferPtr<int32_t>();

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

    // Fills every tree bit and checks the SwapsMmaAb kernels write nothing past the C++ capacity bound.
    void testSwapsMmaAbStaysWithinCapacity(int32_t batchSize, int32_t seqLenQ, int32_t maxSeqLenKv,
        int32_t numHeadsQPerKv, int32_t tileSizeQ, int32_t numInstsKv)
    {
        int32_t const tileSizeKv = 128;
        int32_t const stepQ = tileSizeQ;
        int32_t const stepKv = tileSizeKv * numInstsKv;

        std::mt19937 gen(7);
        std::uniform_int_distribution<> seqLenKvDist(seqLenQ, maxSeqLenKv);
        std::vector<int32_t> seqLensQ(batchSize, seqLenQ);
        std::vector<int32_t> seqLensKv(batchSize);
        std::vector<int32_t> firstSparseMaskOffsetsKv(batchSize);
        std::vector<int32_t> cumSeqLensQ(batchSize + 1, 0);
        for (int32_t i = 0; i < batchSize; ++i)
        {
            seqLensKv[i] = seqLenKvDist(gen);
            firstSparseMaskOffsetsKv[i] = seqLensKv[i] - seqLenQ;
            cumSeqLensQ[i + 1] = cumSeqLensQ[i] + seqLenQ;
        }
        std::vector<int32_t> packedMaskHost(batchSize * seqLenQ * ceilDiv(seqLenQ, 32), -1);

        int64_t const capacity = getCustomMaskSizeUpperBound(FmhaKernelType::SwapsMmaAbForGeneration, batchSize,
            seqLenQ, numHeadsQPerKv, stepQ, stepKv, tileSizeQ, tileSizeKv);
        EXPECT_LE(capacity, getCustomMaskCapacityUpperBound(batchSize, seqLenQ, numHeadsQPerKv));
        int64_t const guardSize = 4096;

        auto seqLensQDevice = mBufferManager->copyFrom(seqLensQ, MemoryType::kGPU);
        auto seqLensKvDevice = mBufferManager->copyFrom(seqLensKv, MemoryType::kGPU);
        auto cumSeqLensQDevice = mBufferManager->copyFrom(cumSeqLensQ, MemoryType::kGPU);
        auto firstSparseMaskOffsetsKvDevice = mBufferManager->copyFrom(firstSparseMaskOffsetsKv, MemoryType::kGPU);
        auto packedMaskDevice = mBufferManager->copyFrom(packedMaskHost, MemoryType::kGPU);
        auto customMaskOffsetsDevice = mBufferManager->gpu(batchSize, tensorrt_llm::DataType::kINT64);
        auto customMaskDevice = mBufferManager->gpu(capacity + guardSize, tensorrt_llm::DataType::kINT32);
        cudaMemsetAsync(
            customMaskDevice->getBufferPtr<int32_t>(), 0, (capacity + guardSize) * sizeof(int32_t), mStream->get());

        TllmGenFmhaRunnerParams runnerParams;
        runnerParams.mBatchSize = batchSize;
        runnerParams.mNumHeadsQPerKv = numHeadsQPerKv;
        runnerParams.mMaxSeqLenQ = seqLenQ;
        runnerParams.mMaxSeqLenKv = *std::max_element(seqLensKv.begin(), seqLensKv.end());
        runnerParams.seqLensKvPtr = seqLensKvDevice->getBufferPtr<int32_t>();
        runnerParams.cumSeqLensQPtr = cumSeqLensQDevice->getBufferPtr<int32_t>();
        runnerParams.seqLensQPtr = seqLensQDevice->getBufferPtr<int32_t>();
        runnerParams.firstSparseMaskOffsetsKvPtr = firstSparseMaskOffsetsKvDevice->getBufferPtr<int32_t>();
        runnerParams.generalPackedCustoMaskPtr = packedMaskDevice->getBufferPtr<int32_t>();
        runnerParams.customMaskOffsetsPtr = customMaskOffsetsDevice->getBufferPtr<int64_t>();
        runnerParams.customMaskPtr = reinterpret_cast<uint32_t*>(customMaskDevice->getBufferPtr<int32_t>());
        runnerParams.customMaskCapacity = capacity;

        runPrepareCustomMask(runnerParams, FmhaKernelType::SwapsMmaAbForGeneration, stepQ, stepKv, tileSizeQ,
            tileSizeKv, mStream->get());
        ASSERT_EQ(cudaStreamSynchronize(mStream->get()), cudaSuccess);

        auto customMaskHost = mBufferManager->copyFrom(*customMaskDevice, MemoryType::kCPU);
        auto* mask = reinterpret_cast<uint32_t*>(customMaskHost->getBufferPtr<int32_t>());
        EXPECT_TRUE(std::any_of(mask, mask + capacity, [](uint32_t word) { return word != 0; }));
        EXPECT_TRUE(std::all_of(mask + capacity, mask + capacity + guardSize, [](uint32_t word) { return word == 0; }));
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

TEST_F(PrepareCustomMaskTest, LargeBatchParallelOffsetScan)
{
    testPrepareCustomMask(/* batchSize */ 128,
        /* maxSeqLenQ */ 16,
        /* maxSeqLenKv */ 128,
        /* numHeadsQPerKv */ 4);
}

TEST_F(PrepareCustomMaskTest, SwapsMmaAbStaysWithinCapacity)
{
    for (int32_t const seqLenQ : {13, 31, 60})
    {
        for (int32_t const numHeadsQPerKv : {2, 4, 8})
        {
            for (int32_t const tileSizeQ : {8, 16, 32})
            {
                for (int32_t const numInstsKv : {1, 2})
                {
                    SCOPED_TRACE(::testing::Message() << "seqLenQ " << seqLenQ << " numHeadsQPerKv " << numHeadsQPerKv
                                                      << " tileSizeQ " << tileSizeQ << " numInstsKv " << numInstsKv);
                    testSwapsMmaAbStaysWithinCapacity(
                        /* batchSize */ 8, seqLenQ, /* maxSeqLenKv */ 1024, numHeadsQPerKv, tileSizeQ, numInstsKv);
                }
            }
        }
    }
}

// The capacity check runs on the host before any launch, so this needs no SM100 GPU.
TEST(PrepareCustomMaskCapacityTest, RejectsUndersizedBuffer)
{
    int32_t const batchSize = 2;
    int32_t const seqLenQ = 13;
    int32_t const numHeadsQPerKv = 2;
    int32_t const tileSizeQ = 8;
    int32_t const tileSizeKv = 128;
    int32_t const stepKv = 256;

    TllmGenFmhaRunnerParams runnerParams;
    runnerParams.mBatchSize = batchSize;
    runnerParams.mNumHeadsQPerKv = numHeadsQPerKv;
    runnerParams.mMaxSeqLenQ = seqLenQ;
    runnerParams.customMaskCapacity = getCustomMaskSizeUpperBound(FmhaKernelType::SwapsMmaAbForGeneration, batchSize,
                                          seqLenQ, numHeadsQPerKv, tileSizeQ, stepKv, tileSizeQ, tileSizeKv)
        - 1;

    EXPECT_THROW(runPrepareCustomMask(runnerParams, FmhaKernelType::SwapsMmaAbForGeneration, tileSizeQ, stepKv,
                     tileSizeQ, tileSizeKv, /* stream */ nullptr),
        tensorrt_llm::common::TllmException);
}

//! Returns the mask type in the kernel name, which follows the QKV layout (e.g. "PagedKvCausalP32").
std::optional<TrtllmGenAttentionMaskType> maskTypeFromName(std::string_view name)
{
    static constexpr std::string_view kLayouts[] = {"SeparateQkv", "PackedQkv", "PagedKv", "ContiguousKv"};
    static constexpr std::pair<std::string_view, TrtllmGenAttentionMaskType> kMasks[] = {
        {"Dense", TrtllmGenAttentionMaskType::Dense},
        {"Causal", TrtllmGenAttentionMaskType::Causal},
        {"SlidingOrChunkedCausal", TrtllmGenAttentionMaskType::SlidingOrChunkedCausal},
        {"Custom", TrtllmGenAttentionMaskType::Custom},
        {"SlidingWindowCustom", TrtllmGenAttentionMaskType::SlidingWindowCustom},
        {"VariableWindow", TrtllmGenAttentionMaskType::VariableWindow},
    };
    for (auto const layout : kLayouts)
    {
        auto const pos = name.find(layout);
        if (pos == std::string_view::npos)
        {
            continue;
        }
        auto const rest = name.substr(pos + layout.size());
        for (auto const& [maskName, maskType] : kMasks)
        {
            if (rest.starts_with(maskName))
            {
                return maskType;
            }
        }
    }
    return std::nullopt;
}

// The cubin lookup is keyed on the registered mask type, so it must match the one in the kernel name.
TEST(FmhaKernelMetaInfoTest, MaskTypeMatchesKernelName)
{
    for (auto const& meta : sTllmGenFmhaKernelMetaInfos)
    {
        std::string_view const name{meta.mFuncName};
        auto const nameMask = maskTypeFromName(name);
        ASSERT_TRUE(nameMask.has_value()) << "No mask type in kernel name " << name;

        auto const mask = static_cast<TrtllmGenAttentionMaskType>(meta.mMaskType);
        // algoFilterForCubinPath requests Causal for Dense non-MLA generation kernels.
        bool const isDenseGenRegisteredAsCausal
            = static_cast<FmhaKernelType>(meta.mKernelType) != FmhaKernelType::Context
            && *nameMask == TrtllmGenAttentionMaskType::Dense && mask == TrtllmGenAttentionMaskType::Causal;
        EXPECT_TRUE(mask == *nameMask || isDenseGenRegisteredAsCausal)
            << name << " is registered with mMaskType " << meta.mMaskType;
    }
}

} // namespace
