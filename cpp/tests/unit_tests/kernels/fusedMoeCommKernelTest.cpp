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

#include <atomic>
#include <chrono>
#include <functional>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/fusedMoeCommKernels.h"

using namespace tensorrt_llm::kernels;

class FusedMoeCommTestBase : public ::testing::Test
{
protected:
    static bool shouldSkip()
    {
        int deviceCount = tensorrt_llm::common::getDeviceCount();
        if (deviceCount <= 0)
        {
            return true;
        }
        int sm = tensorrt_llm::common::getSMVersion();
        if (sm < 90)
        {
            return true;
        }
        return false;
    }

    void SetUp() override
    {
        if (shouldSkip())
        {
            skipped = true;
            GTEST_SKIP() << "Skipping due to no/unsupported GPU";
        }
        TLLM_CUDA_CHECK(cudaStreamCreate(&stream));
        std::srand(42); // Initialize random seed
    }

    void TearDown() override
    {
        if (!skipped)
        {
            TLLM_CUDA_CHECK(cudaStreamDestroy(stream));
        }
    }

    cudaDataType_t getCudaDataType(int elementSize)
    {
        switch (elementSize)
        {
        case 1: return CUDA_R_8U;
        case 2: return CUDA_R_16F;
        case 4: return CUDA_R_32F;
        case 8: return CUDA_R_64F;
        case 16: return CUDA_C_64F;
        default: TLLM_THROW("Unsupported element size: %d", elementSize);
        };
    }

    bool skipped = false;
    cudaStream_t stream = nullptr;

    // Helper function to allocate and initialize test data
    template <typename T>
    void allocateAndInitializeData(
        T** hostPtr, T** devicePtr, size_t count, std::function<T(size_t)> generator = nullptr)
    {
        *hostPtr = new T[count];
        TLLM_CUDA_CHECK(cudaMalloc(devicePtr, count * sizeof(T)));

        if (generator)
        {
            for (size_t i = 0; i < count; i++)
            {
                (*hostPtr)[i] = generator(i);
            }
        }
        else
        {
            // Default initialization with random values
            for (size_t i = 0; i < count; i++)
            {
                if constexpr (std::is_same_v<T, float>)
                {
                    (*hostPtr)[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
                }
                else if constexpr (std::is_same_v<T, int>)
                {
                    (*hostPtr)[i] = rand() % 1000;
                }
                else
                {
                    (*hostPtr)[i] = static_cast<T>(rand() % 100);
                }
            }
        }

        TLLM_CUDA_CHECK(cudaMemcpy(*devicePtr, *hostPtr, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void cleanup(void* hostPtr, void* devicePtr)
    {
        delete[] static_cast<char*>(hostPtr);
        TLLM_CUDA_CHECK(cudaFree(devicePtr));
    }

    // Generate a one-to-one mapping, extending with random permutation if needed
    std::vector<int> generateOneToOneMapping(std::vector<int> const& partialMapping, int totalSize)
    {
        std::vector<int> fullMapping(totalSize);
        std::vector<bool> used(totalSize, false);

        // First, copy the provided mapping and mark used indices
        int providedSize = static_cast<int>(partialMapping.size());
        for (int i = 0; i < std::min(providedSize, totalSize); i++)
        {
            int target = partialMapping[i];
            if (target >= 0 && target < totalSize && !used[target])
            {
                fullMapping[i] = target;
                used[target] = true;
            }
            else
            {
                // Invalid mapping, will be handled later
                fullMapping[i] = -1;
            }
        }

        // Collect unused indices
        std::vector<int> unusedIndices;
        for (int i = 0; i < totalSize; i++)
        {
            if (!used[i])
            {
                unusedIndices.push_back(i);
            }
        }

        // Shuffle unused indices for random assignment
        std::srand(42); // Fixed seed for reproducible tests
        std::random_shuffle(unusedIndices.begin(), unusedIndices.end());

        // Fill in any invalid mappings and extend with remaining unused indices
        int unusedIdx = 0;
        for (int i = 0; i < totalSize; i++)
        {
            if (i < providedSize && fullMapping[i] == -1)
            {
                // Fix invalid mapping
                if (unusedIdx < unusedIndices.size())
                {
                    fullMapping[i] = unusedIndices[unusedIdx++];
                }
            }
            else if (i >= providedSize)
            {
                // Extend mapping
                if (unusedIdx < unusedIndices.size())
                {
                    fullMapping[i] = unusedIndices[unusedIdx++];
                }
                else
                {
                    // Fallback: identity mapping for remaining
                    fullMapping[i] = i;
                }
            }
        }

        return fullMapping;
    }
};

// Test class for launchSingleG2S function
class FusedMoeCommG2STest : public FusedMoeCommTestBase
{
protected:
    void runG2STest(int topK, bool hasScales, bool hasBasicFields, int sendFieldCount,
        std::vector<size_t> const& elementSizes, std::vector<uint16_t> const& vectorSizes, int tokenCount = 4,
        int warpsPerBlock = 2)
    {
        // Setup expert parallel info
        MoeExpertParallelInfo expertParallelInfo;
        expertParallelInfo.topK = topK;
        expertParallelInfo.expertCount = 8;

        // Setup send field info
        FusedMoeFieldInfo sendFieldInfo = {};
        sendFieldInfo.isBasicInterleaved = false;
        sendFieldInfo.fieldCount = sendFieldCount;

        // Allocate token selected slots and expert scales if needed
        int* hostTokenSlots = nullptr;
        int* deviceTokenSlots = nullptr;
        float* hostScales = nullptr;
        float* deviceScales = nullptr;

        if (hasBasicFields)
        {
            allocateAndInitializeData<int>(&hostTokenSlots, &deviceTokenSlots, tokenCount * topK,
                [](size_t i) { return static_cast<int>(i % 8); });
            sendFieldInfo.tokenSelectedSlots = deviceTokenSlots;

            if (hasScales)
            {
                allocateAndInitializeData<float>(&hostScales, &deviceScales, tokenCount * topK,
                    [](size_t i) -> float { return 1.0f + static_cast<float>(i) * 0.1f; });
                sendFieldInfo.expertScales = deviceScales;
            }
        }

        // Setup send field info using new fillFieldInfo helper
        std::vector<void*> hostFieldPtrs(sendFieldCount);
        std::vector<void*> deviceFieldPtrs(sendFieldCount);

        for (int i = 0; i < sendFieldCount; i++)
        {
            size_t elementSize = elementSizes[i % elementSizes.size()];
            uint16_t vectorSize = vectorSizes[i % vectorSizes.size()];
            size_t fieldSize = elementSize * vectorSize * tokenCount;

            // Allocate field data
            uint8_t* hostField;
            uint8_t* deviceField;
            allocateAndInitializeData<uint8_t>(&hostField, &deviceField, fieldSize,
                [i](size_t idx) { return static_cast<uint8_t>((i * 100 + idx) % 128); });

            hostFieldPtrs[i] = hostField;
            deviceFieldPtrs[i] = deviceField;

            // Use the new fillFieldInfo helper function
            sendFieldInfo.fieldsInfo[i].fillFieldInfo(
                deviceField, elementSize, vectorSize, vectorSize, getCudaDataType(elementSize));
        }

        // Fill field placement info
        sendFieldInfo.fillFieldPlacementInfo(topK, hasBasicFields);

        // Compute shared memory size and allocate output buffer
        int warpShmSize = sendFieldInfo.computeSingleUncompactSize(topK, hasScales, hasBasicFields);
        size_t shmDumpSize = tokenCount * warpShmSize;
        size_t shmDumpIntCount = shmDumpSize / sizeof(int);

        int* hostShmDump;
        int* deviceShmDump;
        allocateAndInitializeData<int>(&hostShmDump, &deviceShmDump, shmDumpIntCount, [](size_t) { return 0; });

        // Launch G2S kernel with new signature
        fused_moe_comm_tests::launchSingleG2S(
            sendFieldInfo, expertParallelInfo, tokenCount, deviceShmDump, warpsPerBlock, hasBasicFields, stream);

        TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Copy back results
        int* resultShmDump = new int[shmDumpIntCount];
        TLLM_CUDA_CHECK(
            cudaMemcpy(resultShmDump, deviceShmDump, shmDumpIntCount * sizeof(int), cudaMemcpyDeviceToHost));

        // Verify results
        verifyG2SResults(resultShmDump, hostTokenSlots, hostScales, hostFieldPtrs, topK, hasScales, hasBasicFields,
            sendFieldCount, elementSizes, vectorSizes, tokenCount, warpsPerBlock, warpShmSize);

        // Cleanup
        if (hasBasicFields)
        {
            cleanup(hostTokenSlots, deviceTokenSlots);
            if (hasScales)
            {
                cleanup(hostScales, deviceScales);
            }
        }
        for (int i = 0; i < sendFieldCount; i++)
        {
            cleanup(hostFieldPtrs[i], deviceFieldPtrs[i]);
        }
        cleanup(hostShmDump, deviceShmDump);
        delete[] resultShmDump;
    }

private:
    void verifyG2SResults(int const* shmDump, int const* expectedTokenSlots, float const* expectedScales,
        std::vector<void*> const& expectedFields, int topK, bool hasScales, bool hasBasicFields, int sendFieldCount,
        std::vector<size_t> const& elementSizes, std::vector<uint16_t> const& vectorSizes, int tokenCount,
        int warpsPerBlock, int warpShmSize)
    {
        for (int tokenId = 0; tokenId < tokenCount; tokenId++)
        {
            int const* warpShmData = shmDump + tokenId * warpShmSize / sizeof(int);

            // Verify token slots and scales only if hasBasicFields is true
            if (hasBasicFields)
            {
                // Verify token slots
                if (expectedTokenSlots)
                {
                    for (int k = 0; k < topK; k++)
                    {
                        int expected = expectedTokenSlots[tokenId * topK + k];
                        int actual = warpShmData[k];
                        EXPECT_EQ(expected, actual) << "Token slot mismatch at warp=" << tokenId << ", k=" << k;
                    }
                }

                // Verify scales if present
                if (hasScales && expectedScales)
                {
                    for (int k = 0; k < topK; k++)
                    {
                        float expected = expectedScales[tokenId * topK + k];
                        float actual = reinterpret_cast<float const*>(warpShmData)[topK + k];
                        EXPECT_NEAR(expected, actual, 1e-6f) << "Scale mismatch at warp=" << tokenId << ", k=" << k;
                    }
                }
            }

            // Additional field verification can be added here if needed
            // For now, we just verify that the operation completed successfully
        }
    }
};

// Test class for launchSingleS2G function
class FusedMoeCommS2GTest : public FusedMoeCommTestBase
{
protected:
    void runS2GTest(int topK, bool hasScales, bool hasBasicFields, int recvFieldCount,
        std::vector<size_t> const& elementSizes, std::vector<uint16_t> const& vectorSizes, int tokenCount = 4,
        int warpsPerBlock = 2)
    {
        // Setup expert parallel info
        MoeExpertParallelInfo expertParallelInfo;
        expertParallelInfo.topK = topK;
        expertParallelInfo.expertCount = 8;

        // Setup recv field info
        FusedMoeFieldInfo recvFieldInfo = {};
        recvFieldInfo.isBasicInterleaved = false;
        recvFieldInfo.fieldCount = recvFieldCount;

        // Allocate token selected slots and expert scales if needed
        int* hostTokenSlots = nullptr;
        int* deviceTokenSlots = nullptr;
        float* hostScales = nullptr;
        float* deviceScales = nullptr;

        if (hasBasicFields)
        {
            allocateAndInitializeData<int>(&hostTokenSlots, &deviceTokenSlots, tokenCount * topK,
                [](size_t) { return 0; }); // Initialize to zero, will be filled by S2G
            recvFieldInfo.tokenSelectedSlots = deviceTokenSlots;

            if (hasScales)
            {
                allocateAndInitializeData<float>(&hostScales, &deviceScales, tokenCount * topK,
                    [](size_t) { return 0.0f; }); // Initialize to zero, will be filled by S2G
                recvFieldInfo.expertScales = deviceScales;
            }
        }

        // Setup recv field info using new fillFieldInfo helper
        std::vector<void*> hostFieldPtrs(recvFieldCount);
        std::vector<void*> deviceFieldPtrs(recvFieldCount);

        for (int i = 0; i < recvFieldCount; i++)
        {
            size_t elementSize = elementSizes[i % elementSizes.size()];
            uint16_t vectorSize = vectorSizes[i % vectorSizes.size()];
            size_t fieldSize = elementSize * vectorSize * tokenCount;

            // Allocate field data (initialize to zero, will be filled by S2G)
            uint8_t* hostField;
            uint8_t* deviceField;
            allocateAndInitializeData<uint8_t>(
                &hostField, &deviceField, fieldSize, [](size_t) { return static_cast<uint8_t>(0); });

            hostFieldPtrs[i] = hostField;
            deviceFieldPtrs[i] = deviceField;

            // Use the new fillFieldInfo helper function
            recvFieldInfo.fieldsInfo[i].fillFieldInfo(
                deviceField, elementSize, vectorSize, vectorSize, getCudaDataType(elementSize));
        }

        // Fill field placement info
        recvFieldInfo.fillFieldPlacementInfo(topK, hasBasicFields);

        // Compute shared memory size and prepare input data
        int warpShmSize = recvFieldInfo.computeSingleUncompactSize(topK, hasScales, hasBasicFields);
        size_t shmPreloadSize = tokenCount * warpShmSize;
        size_t shmPreloadIntCount = shmPreloadSize / sizeof(int);

        int* hostShmPreload;
        int* deviceShmPreload;
        allocateAndInitializeData<int>(&hostShmPreload, &deviceShmPreload, shmPreloadIntCount,
            [this, topK, hasScales, hasBasicFields, shmPreloadIntCount](size_t idx)
            { return this->generateShmPreloadData(idx, topK, hasScales, hasBasicFields, shmPreloadIntCount); });

        // Launch S2G kernel with new signature
        fused_moe_comm_tests::launchSingleS2G(
            recvFieldInfo, expertParallelInfo, tokenCount, deviceShmPreload, warpsPerBlock, hasBasicFields, stream);

        TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Copy back results only if hasBasicFields
        int* resultTokenSlots = nullptr;
        float* resultScales = nullptr;

        if (hasBasicFields)
        {
            resultTokenSlots = new int[tokenCount * topK];
            TLLM_CUDA_CHECK(cudaMemcpy(
                resultTokenSlots, deviceTokenSlots, tokenCount * topK * sizeof(int), cudaMemcpyDeviceToHost));

            if (hasScales)
            {
                resultScales = new float[tokenCount * topK];
                TLLM_CUDA_CHECK(
                    cudaMemcpy(resultScales, deviceScales, tokenCount * topK * sizeof(float), cudaMemcpyDeviceToHost));
            }
        }

        // Verify results
        verifyS2GResults(resultTokenSlots, resultScales, hostShmPreload, topK, hasScales, hasBasicFields, tokenCount,
            warpsPerBlock, warpShmSize);

        // Cleanup
        if (hasBasicFields)
        {
            cleanup(hostTokenSlots, deviceTokenSlots);
            if (hasScales)
            {
                cleanup(hostScales, deviceScales);
            }
        }
        for (int i = 0; i < recvFieldCount; i++)
        {
            cleanup(hostFieldPtrs[i], deviceFieldPtrs[i]);
        }
        cleanup(hostShmPreload, deviceShmPreload);
        if (resultTokenSlots)
        {
            delete[] resultTokenSlots;
        }
        if (resultScales)
        {
            delete[] resultScales;
        }
    }

private:
    int generateShmPreloadData(size_t idx, int topK, bool hasScales, bool hasBasicFields, int shmPreloadIntCount)
    {
        size_t warpIdx = idx / shmPreloadIntCount;
        size_t offsetInWarp = idx % shmPreloadIntCount;

        if (hasBasicFields)
        {
            if (offsetInWarp < topK)
            {
                // Token slots area
                return static_cast<int>(warpIdx * 10 + offsetInWarp);
            }
            else if (hasScales && offsetInWarp < topK * 2)
            {
                // Scales area
                float scale
                    = 1.0f + static_cast<float>(warpIdx) * 0.1f + static_cast<float>(offsetInWarp - topK) * 0.01f;
                return *reinterpret_cast<int*>(&scale);
            }
            else
            {
                // Other field data
                return static_cast<int>((warpIdx * 1000 + offsetInWarp) % 128);
            }
        }
        else
        {
            // Only field data when no basic fields
            return static_cast<int>((warpIdx * 1000 + offsetInWarp) % 128);
        }
    }

    void verifyS2GResults(int const* resultTokenSlots, float const* resultScales, int const* shmPreloadData, int topK,
        bool hasScales, bool hasBasicFields, int tokenCount, int warpsPerBlock, int warpShmSize)
    {
        if (!hasBasicFields)
        {
            // For non-basic fields tests, just verify that the operation completed successfully
            // without errors. The actual field data verification would require more complex setup.
            return;
        }

        for (int tokenId = 0; tokenId < tokenCount; tokenId++)
        {
            int const* warpShmData = shmPreloadData + tokenId * warpShmSize / sizeof(int);

            // Verify token slots were written correctly
            if (resultTokenSlots)
            {
                for (int k = 0; k < topK; k++)
                {
                    int expected = warpShmData[k];
                    int actual = resultTokenSlots[tokenId * topK + k];
                    EXPECT_EQ(expected, actual) << "Token slot mismatch at warp=" << tokenId << ", k=" << k;
                }
            }

            // Verify scales if present
            if (hasScales && resultScales)
            {
                for (int k = 0; k < topK; k++)
                {
                    float expected = reinterpret_cast<float const*>(warpShmData)[topK + k];
                    float actual = resultScales[tokenId * topK + k];
                    EXPECT_NEAR(expected, actual, 1e-6f) << "Scale mismatch at warp=" << tokenId << ", k=" << k;
                }
            }
        }
    }
};

// Test class for launchLoopback function (loopback test)
class FusedMoeCommLoopbackTest : public FusedMoeCommTestBase
{
protected:
    void runLoopbackTest(int topK, bool hasScales, bool hasBasicFields, int fieldCount,
        std::vector<size_t> const& elementSizes, std::vector<uint16_t> const& vectorSizes,
        std::vector<int> const& recvIndexMappingVec, int tokenCount = 4, int warpsPerBlock = 2)
    {
        // Setup expert parallel info
        MoeExpertParallelInfo expertParallelInfo;
        expertParallelInfo.topK = topK;
        expertParallelInfo.expertCount = 8;

        // Setup field info - for loopback test, send and recv fields should be identical
        FusedMoeFieldInfo sendFieldInfo = {};
        sendFieldInfo.isBasicInterleaved = false;
        sendFieldInfo.fieldCount = fieldCount;

        FusedMoeFieldInfo recvFieldInfo = {};
        recvFieldInfo.isBasicInterleaved = false;
        recvFieldInfo.fieldCount = fieldCount;

        // Allocate token selected slots and expert scales if needed
        int* hostSendTokenSlots = nullptr;
        int* deviceSendTokenSlots = nullptr;
        float* hostSendScales = nullptr;
        float* deviceSendScales = nullptr;
        int* hostRecvTokenSlots = nullptr;
        int* deviceRecvTokenSlots = nullptr;
        float* hostRecvScales = nullptr;
        float* deviceRecvScales = nullptr;

        if (hasBasicFields)
        {
            // Send side basic fields
            allocateAndInitializeData<int>(&hostSendTokenSlots, &deviceSendTokenSlots, tokenCount * topK,
                [](size_t i) { return static_cast<int>(i % 8); });
            sendFieldInfo.tokenSelectedSlots = deviceSendTokenSlots;

            // Recv side basic fields (initialized to zero, will be filled by loopback)
            allocateAndInitializeData<int>(
                &hostRecvTokenSlots, &deviceRecvTokenSlots, tokenCount * topK, [](size_t) { return 0; });
            recvFieldInfo.tokenSelectedSlots = deviceRecvTokenSlots;

            if (hasScales)
            {
                allocateAndInitializeData<float>(&hostSendScales, &deviceSendScales, tokenCount * topK,
                    [](size_t i) -> float { return 1.0f + static_cast<float>(i) * 0.1f; });
                sendFieldInfo.expertScales = deviceSendScales;

                allocateAndInitializeData<float>(
                    &hostRecvScales, &deviceRecvScales, tokenCount * topK, [](size_t) { return 0.0f; });
                recvFieldInfo.expertScales = deviceRecvScales;
            }
        }

        // Setup field info - both send and recv use same layout for loopback
        std::vector<void*> hostSendFieldPtrs(fieldCount);
        std::vector<void*> deviceSendFieldPtrs(fieldCount);
        std::vector<void*> hostRecvFieldPtrs(fieldCount);
        std::vector<void*> deviceRecvFieldPtrs(fieldCount);

        for (int i = 0; i < fieldCount; i++)
        {
            size_t elementSize = elementSizes[i % elementSizes.size()];
            uint16_t vectorSize = vectorSizes[i % vectorSizes.size()];
            size_t fieldSize = elementSize * vectorSize * tokenCount;

            // Allocate send field data with specific pattern
            uint8_t* hostSendField;
            uint8_t* deviceSendField;
            allocateAndInitializeData<uint8_t>(&hostSendField, &deviceSendField, fieldSize,
                [i](size_t idx) { return static_cast<uint8_t>((i * 100 + idx + 1) % 128); });

            // Allocate recv field data (initially zero, will be filled by loopback)
            uint8_t* hostRecvField;
            uint8_t* deviceRecvField;
            allocateAndInitializeData<uint8_t>(
                &hostRecvField, &deviceRecvField, fieldSize, [](size_t) { return static_cast<uint8_t>(0); });

            hostSendFieldPtrs[i] = hostSendField;
            deviceSendFieldPtrs[i] = deviceSendField;
            hostRecvFieldPtrs[i] = hostRecvField;
            deviceRecvFieldPtrs[i] = deviceRecvField;

            // Fill field info for both send and recv
            sendFieldInfo.fieldsInfo[i].fillFieldInfo(
                deviceSendField, elementSize, vectorSize, vectorSize, getCudaDataType(elementSize));
            recvFieldInfo.fieldsInfo[i].fillFieldInfo(
                deviceRecvField, elementSize, vectorSize, vectorSize, getCudaDataType(elementSize));
        }

        // Fill field placement info
        sendFieldInfo.fillFieldPlacementInfo(topK, hasBasicFields);
        recvFieldInfo.fillFieldPlacementInfo(topK, hasBasicFields);

        // Setup recvIndexMapping - ensure one-to-one mapping
        std::vector<int> fullMapping = generateOneToOneMapping(recvIndexMappingVec, tokenCount);
        int* hostRecvIndexMapping;
        int* deviceRecvIndexMapping;
        allocateAndInitializeData<int>(&hostRecvIndexMapping, &deviceRecvIndexMapping, tokenCount,
            [&fullMapping](size_t i) { return fullMapping[i]; });

        // Launch loopback kernel
        fused_moe_comm_tests::launchLoopback(sendFieldInfo, recvFieldInfo, expertParallelInfo, deviceRecvIndexMapping,
            tokenCount, warpsPerBlock, hasBasicFields, stream);

        TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Copy back results and verify
        verifyLoopbackResults(hostSendTokenSlots, hostSendScales, hostSendFieldPtrs, hostRecvFieldPtrs,
            deviceRecvTokenSlots, deviceRecvScales, deviceRecvFieldPtrs, fullMapping, topK, hasScales, hasBasicFields,
            fieldCount, elementSizes, vectorSizes, tokenCount);

        // Cleanup
        if (hasBasicFields)
        {
            cleanup(hostSendTokenSlots, deviceSendTokenSlots);
            cleanup(hostRecvTokenSlots, deviceRecvTokenSlots);
            if (hasScales)
            {
                cleanup(hostSendScales, deviceSendScales);
                cleanup(hostRecvScales, deviceRecvScales);
            }
        }
        for (int i = 0; i < fieldCount; i++)
        {
            cleanup(hostSendFieldPtrs[i], deviceSendFieldPtrs[i]);
            cleanup(hostRecvFieldPtrs[i], deviceRecvFieldPtrs[i]);
        }
        cleanup(hostRecvIndexMapping, deviceRecvIndexMapping);
    }

private:
    void verifyLoopbackResults(int const* expectedSendTokenSlots, float const* expectedSendScales,
        std::vector<void*> const& expectedSendFields, std::vector<void*> const& hostRecvFields,
        int* deviceRecvTokenSlots, float* deviceRecvScales, std::vector<void*> const& deviceRecvFields,
        std::vector<int> const& fullMapping, int topK, bool hasScales, bool hasBasicFields, int fieldCount,
        std::vector<size_t> const& elementSizes, std::vector<uint16_t> const& vectorSizes, int tokenCount)
    {
        // Copy back device results for verification
        int* resultRecvTokenSlots = nullptr;
        float* resultRecvScales = nullptr;

        if (hasBasicFields)
        {
            resultRecvTokenSlots = new int[tokenCount * topK];
            TLLM_CUDA_CHECK(cudaMemcpy(
                resultRecvTokenSlots, deviceRecvTokenSlots, tokenCount * topK * sizeof(int), cudaMemcpyDeviceToHost));

            if (hasScales)
            {
                resultRecvScales = new float[tokenCount * topK];
                TLLM_CUDA_CHECK(cudaMemcpy(
                    resultRecvScales, deviceRecvScales, tokenCount * topK * sizeof(float), cudaMemcpyDeviceToHost));
            }
        }

        // Copy back field data
        std::vector<uint8_t*> resultRecvFields(fieldCount);
        for (int i = 0; i < fieldCount; i++)
        {
            size_t elementSize = elementSizes[i % elementSizes.size()];
            uint16_t vectorSize = vectorSizes[i % vectorSizes.size()];
            size_t fieldSize = elementSize * vectorSize * tokenCount;

            resultRecvFields[i] = new uint8_t[fieldSize];
            TLLM_CUDA_CHECK(cudaMemcpy(resultRecvFields[i], deviceRecvFields[i], fieldSize, cudaMemcpyDeviceToHost));
        }

        // Verify the loopback: recv[fullMapping[sendIndex]] should equal send[sendIndex]
        int tokenSlotErrorCount = 0;
        int scaleErrorCount = 0;
        std::vector<int> fieldErrorCounts(fieldCount, 0);

        for (int sendIndex = 0; sendIndex < tokenCount; sendIndex++)
        {
            int recvIndex = fullMapping[sendIndex];
            ASSERT_GE(recvIndex, 0) << "Invalid recv index mapping at " << sendIndex;
            ASSERT_LT(recvIndex, tokenCount) << "Recv index out of bounds at " << sendIndex;

            // Verify basic fields if present
            if (hasBasicFields)
            {
                // Verify token slots
                if (expectedSendTokenSlots && resultRecvTokenSlots)
                {
                    for (int k = 0; k < topK; k++)
                    {
                        int expected = expectedSendTokenSlots[sendIndex * topK + k];
                        int actual = resultRecvTokenSlots[recvIndex * topK + k];
                        EXPECT_EQ(expected, actual) << "Token slot loopback mismatch: send[" << sendIndex << "][" << k
                                                    << "] -> recv[" << recvIndex << "][" << k << "]";
                    }
                }

                // Verify scales if present
                if (hasScales && expectedSendScales && resultRecvScales)
                {
                    for (int k = 0; k < topK; k++)
                    {
                        float expected = expectedSendScales[sendIndex * topK + k];
                        float actual = resultRecvScales[recvIndex * topK + k];
                        EXPECT_NEAR(expected, actual, 1e-6f) << "Scale loopback mismatch: send[" << sendIndex << "]["
                                                             << k << "] -> recv[" << recvIndex << "][" << k << "]";
                    }
                }
            }

            // Verify field data
            for (int fieldIdx = 0; fieldIdx < fieldCount; fieldIdx++)
            {
                size_t elementSize = elementSizes[fieldIdx % elementSizes.size()];
                uint16_t vectorSize = vectorSizes[fieldIdx % vectorSizes.size()];
                size_t fieldSize = elementSize * vectorSize;

                uint8_t const* expectedSendField = static_cast<uint8_t const*>(expectedSendFields[fieldIdx]);
                uint8_t const* actualRecvField = resultRecvFields[fieldIdx];

                for (size_t byteIdx = 0; byteIdx < fieldSize; byteIdx++)
                {
                    uint8_t expected = expectedSendField[sendIndex * fieldSize + byteIdx];
                    uint8_t actual = actualRecvField[recvIndex * fieldSize + byteIdx];
                    EXPECT_EQ(expected, actual)
                        << "Field loopback mismatch: field[" << fieldIdx << "] send[" << sendIndex << "][" << byteIdx
                        << "] -> recv[" << recvIndex << "][" << byteIdx << "]";
                }
            }
        }

        // Cleanup temporary arrays
        if (resultRecvTokenSlots)
            delete[] resultRecvTokenSlots;
        if (resultRecvScales)
            delete[] resultRecvScales;
        for (int i = 0; i < fieldCount; i++)
        {
            if (resultRecvFields[i])
                delete[] resultRecvFields[i];
        }
    }
};

// Tests for G2S functionality
TEST_F(FusedMoeCommG2STest, BasicG2SWithoutScales)
{
    runG2STest(2, false, true, 1, {4}, {64}); // topK=2, no scales, has basic fields, 1 field, 4-byte elements, 64 units
}

TEST_F(FusedMoeCommG2STest, BasicG2SWithScales)
{
    runG2STest(
        4, true, true, 1, {4}, {32}); // topK=4, with scales, has basic fields, 1 field, 4-byte elements, 32 units
}

TEST_F(FusedMoeCommG2STest, MultipleFieldsVariousAlignments)
{
    runG2STest(2, true, true, 3, {1, 2, 4}, {16, 32, 64}); // Multiple fields with different element sizes
}

TEST_F(FusedMoeCommG2STest, LargeTopK)
{
    runG2STest(8, true, true, 2, {4, 8}, {128, 256}); // Large topK value
}

TEST_F(FusedMoeCommG2STest, PerfectAlignmentFields)
{
    runG2STest(4, false, true, 2, {16}, {32}); // 16-byte aligned fields
}

TEST_F(FusedMoeCommG2STest, MixedAlignmentTypes)
{
    runG2STest(3, true, true, 4, {8, 4, 2, 1}, {64, 32, 16, 8});     // All alignment types
    runG2STest(3, true, true, 4, {1, 2, 4, 8}, {63, 30, 17, 9}, 32); // All alignment types
}

TEST_F(FusedMoeCommG2STest, SingleByteAlignment)
{
    runG2STest(2, false, true, 2, {1}, {128}); // Single byte alignment
}

TEST_F(FusedMoeCommG2STest, EdgeCaseTopKOne)
{
    runG2STest(1, false, true, 1, {4}, {16}); // Minimal topK
}

TEST_F(FusedMoeCommG2STest, EdgeCaseNoExtraFields)
{
    runG2STest(2, true, true, 0, {}, {}); // Only basic fields (token slots + scales)
}

TEST_F(FusedMoeCommG2STest, LargeTokenCount)
{
    runG2STest(4, true, true, 2, {4, 8}, {64, 128}, 16, 4); // 16 tokens, 4 warps per block
}

// New tests for no basic fields scenario
TEST_F(FusedMoeCommG2STest, G2SWithoutBasicFields)
{
    runG2STest(0, false, false, 2, {4, 8}, {32, 64}); // No basic fields, only field data
}

TEST_F(FusedMoeCommG2STest, G2SWithoutBasicFieldsLargeFields)
{
    runG2STest(0, false, false, 3, {1, 4, 16}, {128, 256, 512}); // No basic fields, large field data
}

// Tests for S2G functionality
TEST_F(FusedMoeCommS2GTest, BasicS2GWithoutScales)
{
    runS2GTest(2, false, true, 1, {4}, {64}); // topK=2, no scales, has basic fields, 1 field, 4-byte elements
}

TEST_F(FusedMoeCommS2GTest, BasicS2GWithScales)
{
    runS2GTest(4, true, true, 1, {4}, {32}); // topK=4, with scales, has basic fields, 1 field, 4-byte elements
}

TEST_F(FusedMoeCommS2GTest, MultipleFieldsVariousAlignments)
{
    runS2GTest(2, true, true, 3, {1, 2, 4}, {16, 32, 64}); // Multiple fields with different element sizes
}

TEST_F(FusedMoeCommS2GTest, LargeTopK)
{
    runS2GTest(8, true, true, 2, {4, 8}, {128, 256}); // Large topK value
}

TEST_F(FusedMoeCommS2GTest, PerfectAlignmentFields)
{
    runS2GTest(4, false, true, 2, {16}, {32}); // 16-byte aligned fields
}

TEST_F(FusedMoeCommS2GTest, MixedAlignmentTypes)
{
    runS2GTest(3, true, true, 4, {1, 2, 4, 8}, {8, 16, 32, 64});     // All alignment types
    runS2GTest(3, true, true, 4, {1, 2, 4, 8}, {63, 30, 17, 9}, 32); // All alignment types
}

TEST_F(FusedMoeCommS2GTest, SingleByteAlignment)
{
    runS2GTest(2, false, true, 2, {1}, {128}); // Single byte alignment
}

TEST_F(FusedMoeCommS2GTest, EdgeCaseTopKOne)
{
    runS2GTest(1, false, true, 1, {4}, {16}); // Minimal topK
}

TEST_F(FusedMoeCommS2GTest, EdgeCaseNoExtraFields)
{
    runS2GTest(2, true, true, 0, {}, {}); // Only basic fields (token slots + scales)
}

TEST_F(FusedMoeCommS2GTest, LargeTokenCount)
{
    runS2GTest(4, true, true, 2, {4, 8}, {64, 128}, 16, 4); // 16 tokens, 4 warps per block
}

// New tests for no basic fields scenario
TEST_F(FusedMoeCommS2GTest, S2GWithoutBasicFields)
{
    runS2GTest(0, false, false, 2, {4, 8}, {32, 64}); // No basic fields, only field data
}

TEST_F(FusedMoeCommS2GTest, S2GWithoutBasicFieldsLargeFields)
{
    runS2GTest(0, false, false, 3, {1, 4, 16}, {128, 256, 512}); // No basic fields, large field data
}

// Tests for G2S+Pack+Unpack+S2G loopback functionality
TEST_F(FusedMoeCommLoopbackTest, BasicLoopbackWithoutScales)
{
    std::vector<int> mapping = {0, 1, 2, 3}; // Identity mapping
    runLoopbackTest(2, false, true, 1, {4}, {64}, mapping);
}

TEST_F(FusedMoeCommLoopbackTest, BasicLoopbackWithScales)
{
    std::vector<int> mapping = {0, 1, 2, 3}; // Identity mapping
    runLoopbackTest(4, true, true, 1, {4}, {32}, mapping);
}

TEST_F(FusedMoeCommLoopbackTest, LoopbackWithReordering)
{
    std::vector<int> mapping = {3, 0, 2, 1}; // Reorder mapping
    runLoopbackTest(2, true, true, 2, {4, 8}, {32, 64}, mapping);
}

TEST_F(FusedMoeCommLoopbackTest, LoopbackWithReverseMapping)
{
    std::vector<int> mapping = {3, 2, 1, 0}; // Reverse mapping
    runLoopbackTest(3, false, true, 1, {2}, {128}, mapping);
}

TEST_F(FusedMoeCommLoopbackTest, LoopbackMultipleFieldsVariousAlignments)
{
    std::vector<int> mapping = {1, 3, 0, 2}; // Complex reordering
    runLoopbackTest(2, true, true, 3, {1, 2, 4}, {16, 32, 64}, mapping);
}

TEST_F(FusedMoeCommLoopbackTest, LoopbackLargeTopK)
{
    std::vector<int> mapping = {2, 0, 3, 1}; // Reorder mapping
    runLoopbackTest(8, true, true, 2, {4, 8}, {128, 256}, mapping);
}

TEST_F(FusedMoeCommLoopbackTest, LoopbackPerfectAlignmentFields)
{
    std::vector<int> mapping = {0, 2, 1, 3}; // Partial reordering
    runLoopbackTest(4, false, true, 2, {16}, {32}, mapping);
}

TEST_F(FusedMoeCommLoopbackTest, LoopbackMixedAlignmentTypes)
{
    std::vector<int> mapping = {1, 0, 3, 2}; // Pair swap
    runLoopbackTest(3, true, true, 4, {1, 2, 4, 8}, {8, 16, 32, 64}, mapping);
}

TEST_F(FusedMoeCommLoopbackTest, LoopbackSingleByteAlignment)
{
    std::vector<int> mapping = {2, 3, 0, 1}; // Cyclic shift
    runLoopbackTest(2, false, true, 2, {1}, {128}, mapping);
}

TEST_F(FusedMoeCommLoopbackTest, LoopbackEdgeCaseTopKOne)
{
    std::vector<int> mapping = {1, 0, 3, 2}; // Simple reordering
    runLoopbackTest(1, false, true, 1, {4}, {16}, mapping);
}

TEST_F(FusedMoeCommLoopbackTest, LoopbackEdgeCaseNoExtraFields)
{
    std::vector<int> mapping = {3, 1, 0, 2};            // Random reordering
    runLoopbackTest(2, true, true, 0, {}, {}, mapping); // Only basic fields
}

TEST_F(FusedMoeCommLoopbackTest, LoopbackLargeTokenCount)
{
    std::vector<int> mapping = {7, 0, 5, 2, 3, 6, 1, 4, 15, 8, 11, 10, 9, 14, 13, 12}; // Complex 16-token mapping
    runLoopbackTest(4, true, true, 2, {4, 8}, {64, 128}, mapping, 16, 4);
}

// New tests for no basic fields scenario
TEST_F(FusedMoeCommLoopbackTest, LoopbackWithoutBasicFields)
{
    std::vector<int> mapping = {1, 3, 0, 2};                        // Reorder mapping
    runLoopbackTest(0, false, false, 2, {4, 8}, {32, 64}, mapping); // No basic fields, only field data
}

TEST_F(FusedMoeCommLoopbackTest, LoopbackWithoutBasicFieldsLargeFields)
{
    std::vector<int> mapping = {2, 0, 3, 1};                                   // Reorder mapping
    runLoopbackTest(0, false, false, 3, {1, 4, 16}, {128, 256, 512}, mapping); // No basic fields, large field data
}

// Test class for launchLocalFifoSendRecv function (FIFO-based local send/recv test)
class FusedMoeCommLocalFifoSendRecvTest : public FusedMoeCommTestBase
{
protected:
    void runLocalFifoSendRecvTest(int topK, bool hasScales, bool hasBasicFields, int fieldCount,
        std::vector<size_t> const& elementSizes, std::vector<uint16_t> const& vectorSizes,
        std::vector<int> const& sendIndexMappingVec, std::vector<int> const& recvIndexMappingVec, int tokenCount = 4,
        int warpsPerBlock = 2, int blockChannelCount = 1)
    {
        // Setup expert parallel info
        MoeExpertParallelInfo expertParallelInfo;
        expertParallelInfo.topK = topK;
        expertParallelInfo.expertCount = 8;

        // Setup field info for send and receive sides
        FusedMoeFieldInfo sendFieldInfo = {};
        sendFieldInfo.isBasicInterleaved = false;
        sendFieldInfo.fieldCount = fieldCount;

        FusedMoeFieldInfo recvFieldInfo = {};
        recvFieldInfo.isBasicInterleaved = false;
        recvFieldInfo.fieldCount = fieldCount;

        // Allocate token selected slots and expert scales if needed
        int* hostSendTokenSlots = nullptr;
        int* deviceSendTokenSlots = nullptr;
        float* hostSendScales = nullptr;
        float* deviceSendScales = nullptr;
        int* hostRecvTokenSlots = nullptr;
        int* deviceRecvTokenSlots = nullptr;
        float* hostRecvScales = nullptr;
        float* deviceRecvScales = nullptr;

        if (hasBasicFields)
        {
            // Send side basic fields
            allocateAndInitializeData<int>(&hostSendTokenSlots, &deviceSendTokenSlots, tokenCount * topK,
                [](size_t i) { return static_cast<int>(i % 8); });
            sendFieldInfo.tokenSelectedSlots = deviceSendTokenSlots;

            // Recv side basic fields (initialized to zero, will be filled by communication)
            allocateAndInitializeData<int>(
                &hostRecvTokenSlots, &deviceRecvTokenSlots, tokenCount * topK, [](size_t) { return 0; });
            recvFieldInfo.tokenSelectedSlots = deviceRecvTokenSlots;

            if (hasScales)
            {
                allocateAndInitializeData<float>(&hostSendScales, &deviceSendScales, tokenCount * topK,
                    [](size_t i) -> float { return 1.0f + static_cast<float>(i) * 0.1f; });
                sendFieldInfo.expertScales = deviceSendScales;

                allocateAndInitializeData<float>(
                    &hostRecvScales, &deviceRecvScales, tokenCount * topK, [](size_t) { return 0.0f; });
                recvFieldInfo.expertScales = deviceRecvScales;
            }
        }

        // Setup field info for additional fields
        std::vector<void*> hostSendFieldPtrs(fieldCount);
        std::vector<void*> deviceSendFieldPtrs(fieldCount);
        std::vector<void*> hostRecvFieldPtrs(fieldCount);
        std::vector<void*> deviceRecvFieldPtrs(fieldCount);

        for (int i = 0; i < fieldCount; i++)
        {
            size_t elementSize = elementSizes[i % elementSizes.size()];
            uint16_t vectorSize = vectorSizes[i % vectorSizes.size()];
            size_t fieldSize = elementSize * vectorSize * tokenCount;

            // Allocate send field data with specific pattern
            uint8_t* hostSendField;
            uint8_t* deviceSendField;
            allocateAndInitializeData<uint8_t>(&hostSendField, &deviceSendField, fieldSize,
                [i](size_t idx) { return static_cast<uint8_t>((i * 100 + idx + 1) % 128); });

            // Allocate recv field data (initially zero, will be filled by communication)
            uint8_t* hostRecvField;
            uint8_t* deviceRecvField;
            allocateAndInitializeData<uint8_t>(
                &hostRecvField, &deviceRecvField, fieldSize, [](size_t) { return static_cast<uint8_t>(0); });

            hostSendFieldPtrs[i] = hostSendField;
            deviceSendFieldPtrs[i] = deviceSendField;
            hostRecvFieldPtrs[i] = hostRecvField;
            deviceRecvFieldPtrs[i] = deviceRecvField;

            // Fill field info
            sendFieldInfo.fieldsInfo[i].fillFieldInfo(
                deviceSendField, elementSize, vectorSize, vectorSize, getCudaDataType(elementSize));
            recvFieldInfo.fieldsInfo[i].fillFieldInfo(
                deviceRecvField, elementSize, vectorSize, vectorSize, getCudaDataType(elementSize));
        }

        // Fill field placement info
        sendFieldInfo.fillFieldPlacementInfo(topK, hasBasicFields);
        recvFieldInfo.fillFieldPlacementInfo(topK, hasBasicFields);

        // Setup sendIndexMapping and recvIndexMapping - ensure one-to-one mappings
        std::vector<int> fullSendMapping = generateOneToOneMapping(sendIndexMappingVec, tokenCount);
        std::vector<int> fullRecvMapping = generateOneToOneMapping(recvIndexMappingVec, tokenCount);

        int* hostSendIndexMapping;
        int* deviceSendIndexMapping;
        int* hostRecvIndexMapping;
        int* deviceRecvIndexMapping;

        allocateAndInitializeData<int>(&hostSendIndexMapping, &deviceSendIndexMapping, tokenCount,
            [&fullSendMapping](size_t i) { return fullSendMapping[i]; });
        allocateAndInitializeData<int>(&hostRecvIndexMapping, &deviceRecvIndexMapping, tokenCount,
            [&fullRecvMapping](size_t i) { return fullRecvMapping[i]; });

        // Setup workspace for FIFO communication
        FusedMoeWorkspace fusedMoeWorkspace;
        int totalChannelCount = blockChannelCount * warpsPerBlock;
        size_t workspaceSizePerRank = FusedMoeWorkspace::computeWorkspaceSizePreRank(1, totalChannelCount);
        size_t totalWorkspaceSize = workspaceSizePerRank;
        fusedMoeWorkspace.rankStrideInU64 = workspaceSizePerRank / sizeof(uint64_t);
        fusedMoeWorkspace.channelCount = totalChannelCount;

        TLLM_CUDA_CHECK(cudaMalloc(&fusedMoeWorkspace.workspacePtr, totalWorkspaceSize));

        // Initialize workspace
        FusedMoeWorldInfo worldInfo;
        worldInfo.epInfo.epRank = 0;
        worldInfo.epInfo.epSize = 1;
        fusedMoeWorkspace.initializeLocalWorkspace(worldInfo);

        // Launch FIFO send/recv kernel
        fused_moe_comm_tests::launchLocalFifoSendRecv(sendFieldInfo, recvFieldInfo, expertParallelInfo,
            deviceSendIndexMapping, deviceRecvIndexMapping, fusedMoeWorkspace, tokenCount, warpsPerBlock,
            blockChannelCount, hasBasicFields, stream);

        TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Copy back results and verify
        verifyLocalFifoSendRecvResults(hostSendTokenSlots, hostSendScales, hostSendFieldPtrs, hostRecvFieldPtrs,
            deviceRecvTokenSlots, deviceRecvScales, deviceRecvFieldPtrs, fullSendMapping, fullRecvMapping, topK,
            hasScales, hasBasicFields, fieldCount, elementSizes, vectorSizes, tokenCount);

        // Cleanup
        if (hasBasicFields)
        {
            cleanup(hostSendTokenSlots, deviceSendTokenSlots);
            cleanup(hostRecvTokenSlots, deviceRecvTokenSlots);
            if (hasScales)
            {
                cleanup(hostSendScales, deviceSendScales);
                cleanup(hostRecvScales, deviceRecvScales);
            }
        }
        for (int i = 0; i < fieldCount; i++)
        {
            cleanup(hostSendFieldPtrs[i], deviceSendFieldPtrs[i]);
            cleanup(hostRecvFieldPtrs[i], deviceRecvFieldPtrs[i]);
        }
        cleanup(hostSendIndexMapping, deviceSendIndexMapping);
        cleanup(hostRecvIndexMapping, deviceRecvIndexMapping);
        TLLM_CUDA_CHECK(cudaFree(fusedMoeWorkspace.workspacePtr));
    }

private:
    void verifyLocalFifoSendRecvResults(int const* expectedSendTokenSlots, float const* expectedSendScales,
        std::vector<void*> const& expectedSendFields, std::vector<void*> const& hostRecvFields,
        int* deviceRecvTokenSlots, float* deviceRecvScales, std::vector<void*> const& deviceRecvFields,
        std::vector<int> const& fullSendMapping, std::vector<int> const& fullRecvMapping, int topK, bool hasScales,
        bool hasBasicFields, int fieldCount, std::vector<size_t> const& elementSizes,
        std::vector<uint16_t> const& vectorSizes, int tokenCount)
    {
        // Copy back device results for verification
        int* resultRecvTokenSlots = nullptr;
        float* resultRecvScales = nullptr;

        if (hasBasicFields)
        {
            resultRecvTokenSlots = new int[tokenCount * topK];
            TLLM_CUDA_CHECK(cudaMemcpy(
                resultRecvTokenSlots, deviceRecvTokenSlots, tokenCount * topK * sizeof(int), cudaMemcpyDeviceToHost));

            if (hasScales)
            {
                resultRecvScales = new float[tokenCount * topK];
                TLLM_CUDA_CHECK(cudaMemcpy(
                    resultRecvScales, deviceRecvScales, tokenCount * topK * sizeof(float), cudaMemcpyDeviceToHost));
            }
        }

        // Copy back field data
        std::vector<uint8_t*> resultRecvFields(fieldCount);
        for (int i = 0; i < fieldCount; i++)
        {
            size_t elementSize = elementSizes[i % elementSizes.size()];
            uint16_t vectorSize = vectorSizes[i % vectorSizes.size()];
            size_t fieldSize = elementSize * vectorSize * tokenCount;

            resultRecvFields[i] = new uint8_t[fieldSize];
            TLLM_CUDA_CHECK(cudaMemcpy(resultRecvFields[i], deviceRecvFields[i], fieldSize, cudaMemcpyDeviceToHost));
        }

        // Verify the FIFO send/recv with independent mappings:
        // For logical index i:
        // - Send side reads from fullSendMapping[i]
        // - Recv side writes to fullRecvMapping[i]
        // So we need to verify: recv[fullRecvMapping[i]] should equal send[fullSendMapping[i]]
        int tokenSlotErrorCount = 0;
        int scaleErrorCount = 0;
        std::vector<int> fieldErrorCounts(fieldCount, 0);

        for (int logicalIndex = 0; logicalIndex < tokenCount; logicalIndex++)
        {
            int actualSendIndex = fullSendMapping[logicalIndex];
            int actualRecvIndex = fullRecvMapping[logicalIndex];
            if (actualSendIndex < 0 || actualSendIndex >= tokenCount || actualRecvIndex < 0
                || actualRecvIndex >= tokenCount)
                continue;

            // Verify token selected slots
            if (hasBasicFields)
            {
                for (int k = 0; k < topK; k++)
                {
                    int expectedSlot = expectedSendTokenSlots[actualSendIndex * topK + k];
                    int actualSlot = resultRecvTokenSlots[actualRecvIndex * topK + k];
                    if (expectedSlot != actualSlot)
                    {
                        tokenSlotErrorCount++;
                        if (tokenSlotErrorCount <= 16)
                        {
                            EXPECT_EQ(expectedSlot, actualSlot)
                                << "Token slot mismatch at logicalIndex=" << logicalIndex
                                << ", actualSendIndex=" << actualSendIndex << ", actualRecvIndex=" << actualRecvIndex
                                << ", k=" << k;
                        }
                    }
                }

                // Verify expert scales
                if (hasScales)
                {
                    for (int k = 0; k < topK; k++)
                    {
                        float expectedScale = expectedSendScales[actualSendIndex * topK + k];
                        float actualScale = resultRecvScales[actualRecvIndex * topK + k];
                        if (std::abs(expectedScale - actualScale) > 1e-6f)
                        {
                            scaleErrorCount++;
                            if (scaleErrorCount <= 16)
                            {
                                EXPECT_NEAR(expectedScale, actualScale, 1e-6f)
                                    << "Scale mismatch at logicalIndex=" << logicalIndex
                                    << ", actualSendIndex=" << actualSendIndex
                                    << ", actualRecvIndex=" << actualRecvIndex << ", k=" << k;
                            }
                        }
                    }
                }
            }

            // Verify additional fields
            for (int fieldIdx = 0; fieldIdx < fieldCount; fieldIdx++)
            {
                size_t elementSize = elementSizes[fieldIdx % elementSizes.size()];
                uint16_t vectorSize = vectorSizes[fieldIdx % vectorSizes.size()];
                size_t fieldSizePerToken = elementSize * vectorSize;

                uint8_t const* expectedFieldData = static_cast<uint8_t const*>(expectedSendFields[fieldIdx]);
                uint8_t const* actualFieldData = resultRecvFields[fieldIdx];

                for (size_t byteIdx = 0; byteIdx < fieldSizePerToken; byteIdx++)
                {
                    uint8_t expected = expectedFieldData[actualSendIndex * fieldSizePerToken + byteIdx];
                    uint8_t actual = actualFieldData[actualRecvIndex * fieldSizePerToken + byteIdx];
                    if (expected != actual)
                    {
                        fieldErrorCounts[fieldIdx]++;
                        if (fieldErrorCounts[fieldIdx] <= 16)
                        {
                            EXPECT_EQ(static_cast<int>(expected), static_cast<int>(actual))
                                << "Field[" << fieldIdx << "] mismatch at logicalIndex=" << logicalIndex
                                << ", actualSendIndex=" << actualSendIndex << ", actualRecvIndex=" << actualRecvIndex
                                << ", byteIdx=" << byteIdx;
                        }
                    }
                }
            }
        }

        // Print error summary for counts exceeding 16
        if (tokenSlotErrorCount > 16)
        {
            ADD_FAILURE() << "Token slot errors: Showed first 16 of " << tokenSlotErrorCount << " total mismatches.";
        }
        if (scaleErrorCount > 16)
        {
            ADD_FAILURE() << "Scale errors: Showed first 16 of " << scaleErrorCount << " total mismatches.";
        }
        for (int fieldIdx = 0; fieldIdx < fieldCount; fieldIdx++)
        {
            if (fieldErrorCounts[fieldIdx] > 16)
            {
                ADD_FAILURE() << "Field[" << fieldIdx << "] errors: Showed first 16 of " << fieldErrorCounts[fieldIdx]
                              << " total mismatches.";
            }
        }

        // Cleanup temporary arrays
        if (resultRecvTokenSlots)
            delete[] resultRecvTokenSlots;
        if (resultRecvScales)
            delete[] resultRecvScales;
        for (int i = 0; i < fieldCount; i++)
        {
            if (resultRecvFields[i])
                delete[] resultRecvFields[i];
        }
    }
};

// Tests for Local FIFO Send/Recv functionality with Packed Protocol
TEST_F(FusedMoeCommLocalFifoSendRecvTest, BasicFifoSendRecvPackedProtocol)
{
    std::vector<int> sendMapping = {0, 1, 2, 3}; // Identity mapping for send
    std::vector<int> recvMapping = {2, 3, 0, 1}; // Rotate mapping for recv
    runLocalFifoSendRecvTest(2, false, true, 1, {4}, {64}, sendMapping, recvMapping, 4, 1, 1); // Packed protocol
}

TEST_F(FusedMoeCommLocalFifoSendRecvTest, BasicFifoSendRecvWithScalesPackedProtocol)
{
    std::vector<int> sendMapping = {1, 2, 3, 0};                         // Rotate send mapping
    std::vector<int> recvMapping = {3, 0, 1, 2};                         // Opposite rotation for recv
    runLocalFifoSendRecvTest(
        4, true, true, 1, {4}, {32}, sendMapping, recvMapping, 4, 2, 1); // With scales, Packed protocol
}

TEST_F(FusedMoeCommLocalFifoSendRecvTest, FifoSendRecvWithReorderingPackedProtocol)
{
    std::vector<int> sendMapping = {3, 0, 2, 1}; // Random send reorder
    std::vector<int> recvMapping = {0, 3, 1, 2}; // Different recv reorder
    runLocalFifoSendRecvTest(2, true, true, 2, {4, 8}, {32, 64}, sendMapping, recvMapping, 256, 2, 2);
}

TEST_F(FusedMoeCommLocalFifoSendRecvTest, FifoSendRecvMultipleFieldsPackedProtocol)
{
    std::vector<int> mapping = {1, 3, 0, 2}; // Complex reordering
    runLocalFifoSendRecvTest(2, true, true, 3, {1, 2, 4}, {16, 32, 64}, mapping, mapping, 256, 2, 2);
}

TEST_F(FusedMoeCommLocalFifoSendRecvTest, FifoSendRecvLargeTopKPackedProtocol)
{
    std::vector<int> mapping = {2, 0, 3, 1}; // Reorder mapping
    runLocalFifoSendRecvTest(8, true, true, 2, {4, 8}, {128, 256}, mapping, mapping, 512, 3, 2);
}

TEST_F(FusedMoeCommLocalFifoSendRecvTest, FifoSendRecvWithoutBasicFieldsPackedProtocol)
{
    std::vector<int> sendMapping = {1, 3, 0, 2};                                    // Send reorder mapping
    std::vector<int> recvMapping = {3, 2, 1, 0};                                    // Reverse recv mapping
    runLocalFifoSendRecvTest(
        0, false, false, 2, {4, 8}, {32, 64}, sendMapping, recvMapping, 256, 2, 2); // No basic fields, Packed protocol
}

// Mixed alignment tests
TEST_F(FusedMoeCommLocalFifoSendRecvTest, FifoSendRecvMixedAlignmentsPackedProtocol)
{
    std::vector<int> mapping = {1, 0, 3, 2}; // Pair swap
    runLocalFifoSendRecvTest(3, true, true, 4, {1, 2, 4, 8}, {8, 16, 32, 64}, mapping, mapping, 512, 2, 2);
}

// Edge cases
TEST_F(FusedMoeCommLocalFifoSendRecvTest, FifoSendRecvEdgeCaseTopKOnePackedProtocol)
{
    std::vector<int> mapping = {1, 0, 3, 2}; // Simple reordering
    runLocalFifoSendRecvTest(1, false, true, 1, {4}, {16}, mapping, mapping, 128, 2, 1);
}

// Only basic fields cases
TEST_F(FusedMoeCommLocalFifoSendRecvTest, FifoSendRecvEdgeCaseNoExtraFieldsPackedProtocol)
{
    std::vector<int> mapping = {3, 1, 0, 2};                                         // Random reordering
    runLocalFifoSendRecvTest(2, true, true, 0, {}, {}, mapping, mapping, 256, 2, 2); // Only basic fields
}

// Large scale tests
TEST_F(FusedMoeCommLocalFifoSendRecvTest, FifoSendRecvLargeTokenCountPackedProtocol)
{
    std::vector<int> sendMapping = {7, 0, 5, 2, 3, 6, 1, 4, 15, 8, 11, 10, 9, 14, 13, 12}; // Complex send mapping
    std::vector<int> recvMapping = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}; // Reverse recv mapping
    runLocalFifoSendRecvTest(
        4, true, true, 2, {4, 8}, {64, 128}, sendMapping, recvMapping, 1024, 3, 3);        // Large scale test
}

// Perfect alignment tests
TEST_F(FusedMoeCommLocalFifoSendRecvTest, FifoSendRecvPerfectAlignmentPackedProtocol)
{
    std::vector<int> sendMapping = {2, 0, 3, 1}; // Different send reordering
    std::vector<int> recvMapping = {1, 3, 0, 2}; // Different recv reordering
    runLocalFifoSendRecvTest(4, false, true, 2, {16}, {32}, sendMapping, recvMapping, 256, 2, 3);
}

// Single byte alignment tests
TEST_F(FusedMoeCommLocalFifoSendRecvTest, FifoSendRecvSmallSingleByteAlignmentPackedProtocol)
{
    std::vector<int> mapping = {2, 3, 0, 1}; // Cyclic shift
    runLocalFifoSendRecvTest(2, false, true, 1, {1}, {127}, mapping, mapping, 4, 1, 1);
}

TEST_F(FusedMoeCommLocalFifoSendRecvTest, FifoSendRecvSingleByteAlignmentPackedProtocol)
{
    std::vector<int> mapping = {2, 3, 0, 1}; // Cyclic shift
    runLocalFifoSendRecvTest(2, false, true, 2, {1}, {127}, mapping, mapping, 256, 3, 1);
}

// Stress tests
TEST_F(FusedMoeCommLocalFifoSendRecvTest, FifoSendRecvStressTestManyChannelsPackedProtocol)
{
    std::vector<int> mapping = {7, 2, 5, 0, 3, 6, 1, 4};                                          // Complex mapping
    runLocalFifoSendRecvTest(4, true, true, 2, {8, 16}, {128, 256}, mapping, mapping, 512, 3, 4); // Many channels
}

TEST_F(FusedMoeCommLocalFifoSendRecvTest, FifoSendRecvStressTest2ManyChannelsPackedProtocol)
{
    std::vector<int> mapping = {7, 2, 5, 0, 3, 6, 1, 4};                                  // Complex mapping
    runLocalFifoSendRecvTest(
        4, true, true, 2, {2, 4, 8, 16}, {7, 15, 31, 255}, mapping, mapping, 4096, 1, 2); // Many channels
}

TEST_F(FusedMoeCommLocalFifoSendRecvTest, FifoSendRecvStressTestManyWarpsPackedProtocol)
{
    std::vector<int> mapping = {1, 0, 3, 2};                                             // Simple reordering
    runLocalFifoSendRecvTest(2, false, true, 1, {4}, {64}, mapping, mapping, 256, 4, 2); // Many warps per block
}
