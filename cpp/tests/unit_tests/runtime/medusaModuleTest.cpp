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

#include "tensorrt_llm/runtime/medusaModule.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/utils/speculativeChoicesUtils.h"

#include <NvInferRuntime.h>

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>

namespace tensorrt_llm::runtime
{
using TensorPtr = ITensor::SharedPtr;

class MedusaModuleTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    MedusaModuleTest() {}

    void SetUp() override
    {
        mStream = std::make_unique<CudaStream>();
        mManager = std::make_unique<BufferManager>(mStream);
    }

    void runTest(MedusaModule::MedusaChoices& choices, std::vector<SizeType32> const& refTopKs,
        std::vector<SizeType32> const& refPositionOffsets, std::vector<SizeType32> const& refTreeIds,
        std::vector<std::vector<SizeType32>> const& refPaths, std::vector<std::vector<int32_t>> const& refPackedMask)
    {
        auto maxMedusaTokens = static_cast<SizeType32>(choices.size());
        auto medusaHeads = static_cast<SizeType32>(std::max_element(choices.begin(), choices.end(),
            [](std::vector<SizeType32> const& a, std::vector<SizeType32> const& b)
            { return a.size() < b.size(); })->size());
        std::cout << medusaHeads << " " << maxMedusaTokens << std::endl;

        MedusaModule medusaModule(medusaHeads, maxMedusaTokens);
        auto const numPackedMasks = medusaModule.getNumPackedMasks();
        auto const tokensPerStep = medusaModule.getMaxDecodingTokens();

        // batch size = 1 here.
        TensorPtr medusaGenerationLengthsHost = mManager->pinned(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
        TensorPtr medusaPositionOffsetsHost
            = mManager->pinned(ITensor::makeShape({tokensPerStep}), nvinfer1::DataType::kINT32);
        TensorPtr medusaTreeIdsHost = mManager->pinned(ITensor::makeShape({tokensPerStep}), nvinfer1::DataType::kINT32);
        TensorPtr medusaPathsHost
            = mManager->pinned(ITensor::makeShape({tokensPerStep, medusaHeads + 1}), nvinfer1::DataType::kINT32);
        TensorPtr attentionPackedMaskHost
            = mManager->pinned(ITensor::makeShape({tokensPerStep, numPackedMasks}), nvinfer1::DataType::kINT32);

        std::vector<SizeType32> topKs;
        utils::initTensorsFromChoices(medusaModule, choices, topKs, medusaGenerationLengthsHost,
            medusaPositionOffsetsHost, medusaTreeIdsHost, medusaPathsHost, attentionPackedMaskHost);

        std::cout << "medusaPositionOffsetsHost " << *medusaPositionOffsetsHost << std::endl;
        std::cout << "medusaTreeIdsHost " << *medusaTreeIdsHost << std::endl;
        std::cout << "medusaPathsHost " << *medusaPathsHost << std::endl;
        for (SizeType32 hi = 0; hi < medusaHeads; ++hi)
        {
            EXPECT_EQ(topKs[hi], refTopKs[hi]);
        }

        // batch size = 1 here.
        EXPECT_EQ(bufferCast<SizeType32>(*medusaGenerationLengthsHost)[0], tokensPerStep);

        for (SizeType32 hi = 0; hi < tokensPerStep; ++hi)
        {
            EXPECT_EQ(bufferCast<SizeType32>(*medusaPositionOffsetsHost)[hi], refPositionOffsets[hi]);
        }

        for (SizeType32 hi = 0; hi < tokensPerStep - 1; ++hi)
        {
            EXPECT_EQ(bufferCast<SizeType32>(*medusaTreeIdsHost)[hi], refTreeIds[hi]);
        }

        for (SizeType32 ri = 0; ri < refPaths.size(); ++ri)
        {
            for (SizeType32 vi = 0; vi < medusaHeads + 1; ++vi)
            {
                auto const out = bufferCast<SizeType32>(*medusaPathsHost)[ri * (medusaHeads + 1) + vi];
                auto const ref = refPaths[ri][vi];
                EXPECT_EQ(out, ref);
            }
        }

        auto packedMaskPtr = bufferCast<int32_t>(*attentionPackedMaskHost);
        for (SizeType32 ti = 0; ti < tokensPerStep; ++ti)
        {
            for (SizeType32 mi = 0; mi < numPackedMasks; ++mi)
            {
                EXPECT_EQ(packedMaskPtr[ti * numPackedMasks + mi], refPackedMask[ti][mi]);
            }
        }
    }

    std::unique_ptr<BufferManager> mManager;
    BufferManager::CudaStreamPtr mStream;
};

TEST_F(MedusaModuleTest, simpleChoices)
{
    MedusaModule::MedusaChoices choices
        = {{0}, {0, 1}, {0, 0}, {0, 2}, {1, 0, 0}, {1, 0, 1}, {0, 0, 0}, {1}, {1, 0}, {1, 1}};
    // Ref is generated with python code
    std::vector<SizeType32> refTopKs = {2, 3, 2, 0};
    std::vector<SizeType32> refPositionOffsets = {0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3};
    std::vector<SizeType32> refTreeIds = {0, 1, 2, 3, 4, 2, 3, 5, 5, 6};
    std::vector<std::vector<SizeType32>> refPaths
        = {{0, 1, 3, 8}, {0, 1, 4, -1}, {0, 1, 5, -1}, {0, 2, 6, 9}, {0, 2, 6, 10}, {0, 2, 7, -1}};
    std::vector<std::vector<int32_t>> refPackedMask
        = {{1}, {3}, {5}, {11}, {19}, {35}, {69}, {133}, {267}, {581}, {1093}};

    runTest(choices, refTopKs, refPositionOffsets, refTreeIds, refPaths, refPackedMask);
}

TEST_F(MedusaModuleTest, mcSim7663Choices)
{
    MedusaModule::MedusaChoices choices = {{0}, {0, 0}, {1}, {0, 1}, {2}, {0, 0, 0}, {1, 0}, {0, 2}, {3}, {0, 3}, {4},
        {0, 4}, {2, 0}, {0, 5}, {0, 0, 1}, {5}, {0, 6}, {6}, {0, 7}, {0, 1, 0}, {1, 1}, {7}, {0, 8}, {0, 0, 2}, {3, 0},
        {0, 9}, {8}, {9}, {1, 0, 0}, {0, 2, 0}, {1, 2}, {0, 0, 3}, {4, 0}, {2, 1}, {0, 0, 4}, {0, 0, 5}, {0, 0, 0, 0},
        {0, 1, 1}, {0, 0, 6}, {0, 3, 0}, {5, 0}, {1, 3}, {0, 0, 7}, {0, 0, 8}, {0, 0, 9}, {6, 0}, {0, 4, 0}, {1, 4},
        {7, 0}, {0, 1, 2}, {2, 0, 0}, {3, 1}, {2, 2}, {8, 0}, {0, 5, 0}, {1, 5}, {1, 0, 1}, {0, 2, 1}, {9, 0},
        {0, 6, 0}, {0, 0, 0, 1}, {1, 6}, {0, 7, 0}};
    // Ref is generated with python code
    std::vector<SizeType32> refTopKs = {10, 10, 10, 2};
    std::vector<SizeType32> refPositionOffsets
        = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4};
    std::vector<SizeType32> refTreeIds = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 10, 11,
        12, 13, 14, 15, 16, 10, 11, 12, 10, 11, 10, 10, 10, 10, 10, 10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 20, 21,
        22, 20, 21, 20, 20, 20, 20, 20, 20, 21, 20, 30, 31};
    std::vector<std::vector<SizeType32>> refPaths = {{0, 1, 11, 39, 62}, {0, 1, 11, 39, 63}, {0, 1, 11, 40, -1},
        {0, 1, 11, 41, -1}, {0, 1, 11, 42, -1}, {0, 1, 11, 43, -1}, {0, 1, 11, 44, -1}, {0, 1, 11, 45, -1},
        {0, 1, 11, 46, -1}, {0, 1, 11, 47, -1}, {0, 1, 11, 48, -1}, {0, 1, 12, 49, -1}, {0, 1, 12, 50, -1},
        {0, 1, 12, 51, -1}, {0, 1, 13, 52, -1}, {0, 1, 13, 53, -1}, {0, 1, 14, 54, -1}, {0, 1, 15, 55, -1},
        {0, 1, 16, 56, -1}, {0, 1, 17, 57, -1}, {0, 1, 18, 58, -1}, {0, 1, 19, -1, -1}, {0, 1, 20, -1, -1},
        {0, 2, 21, 59, -1}, {0, 2, 21, 60, -1}, {0, 2, 22, -1, -1}, {0, 2, 23, -1, -1}, {0, 2, 24, -1, -1},
        {0, 2, 25, -1, -1}, {0, 2, 26, -1, -1}, {0, 2, 27, -1, -1}, {0, 3, 28, 61, -1}, {0, 3, 29, -1, -1},
        {0, 3, 30, -1, -1}, {0, 4, 31, -1, -1}, {0, 4, 32, -1, -1}, {0, 5, 33, -1, -1}, {0, 6, 34, -1, -1},
        {0, 7, 35, -1, -1}, {0, 8, 36, -1, -1}, {0, 9, 37, -1, -1}, {0, 10, 38, -1, -1}};
    std::vector<std::vector<int32_t>> refPackedMask = {{1, 0}, {3, 0}, {5, 0}, {9, 0}, {17, 0}, {33, 0}, {65, 0},
        {129, 0}, {257, 0}, {513, 0}, {1025, 0}, {2051, 0}, {4099, 0}, {8195, 0}, {16387, 0}, {32771, 0}, {65539, 0},
        {131075, 0}, {262147, 0}, {524291, 0}, {1048579, 0}, {2097157, 0}, {4194309, 0}, {8388613, 0}, {16777221, 0},
        {33554437, 0}, {67108869, 0}, {134217733, 0}, {268435465, 0}, {536870921, 0}, {1073741833, 0}, {-2147483631, 0},
        {17, 1}, {33, 2}, {65, 4}, {129, 8}, {257, 16}, {513, 32}, {1025, 64}, {2051, 128}, {2051, 256}, {2051, 512},
        {2051, 1024}, {2051, 2048}, {2051, 4096}, {2051, 8192}, {2051, 16384}, {2051, 32768}, {2051, 65536},
        {4099, 131072}, {4099, 262144}, {4099, 524288}, {8195, 1048576}, {8195, 2097152}, {16387, 4194304},
        {32771, 8388608}, {65539, 16777216}, {131075, 33554432}, {262147, 67108864}, {2097157, 134217728},
        {2097157, 268435456}, {268435465, 536870912}, {2051, 1073741952}, {2051, -2147483520}};

    runTest(choices, refTopKs, refPositionOffsets, refTreeIds, refPaths, refPackedMask);
}

} // namespace tensorrt_llm::runtime
