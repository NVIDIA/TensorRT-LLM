/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include "tests/kernels/sampling/samplingTest.h"
#include <random>

using namespace tensorrt_llm::tests::kernels::sampling;
using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;

namespace
{

static float constexpr HALF_FLT_MAX = 65504.F;

__global__ void generateRandomNumber(int32_t* vals, curandState_t* states, const int batch_size)
{
    int idx = threadIdx.x;
    if (idx < batch_size)
    {
        vals[idx] = curand(states + idx);
    }
}

class SamplingUtilsKernelTest : public SamplingKernelTest<float>
{
};

TEST_F(SamplingUtilsKernelTest, CurandInitialize)
{
    int32_t batchSize = 127;

    auto initSeedAndGenerateNumbers = [batchSize, this](uint64_t seed) -> auto
    {
        curandState_t* curandStates;
        cudaMalloc(&curandStates, sizeof(curandState_t) * batchSize);
        // Initialize curand states.
        tk::invokeCurandInitialize(curandStates, batchSize, seed, this->mStream->get());
        sync_check_cuda_error();

        // Generate random numbers using initialized curand states.MemoryType
        auto randValsDevice = this->mBufferManager->gpu(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
        generateRandomNumber<<<1, batchSize, 0, this->mStream->get()>>>(
            bufferCast<int32_t>(*randValsDevice), curandStates, batchSize);
        auto randValsHost = this->mBufferManager->copyFrom(*randValsDevice, MemoryType::kCPU);
        this->mStream->synchronize();

        cudaFree(curandStates);
        return std::move(randValsHost);
    };

    const auto randValsSeed0Host = initSeedAndGenerateNumbers(0);
    const auto randValsSeed1Host = initSeedAndGenerateNumbers(1);
    const auto randValsSeed0AltHost = initSeedAndGenerateNumbers(0);

    const auto randValsSeed0HostPtr = bufferCast<int32_t>(*randValsSeed0Host);
    const auto randValsSeed1HostPtr = bufferCast<int32_t>(*randValsSeed1Host);
    const auto randValsSeed0AltHostPtr = bufferCast<int32_t>(*randValsSeed0AltHost);

    // The different seed produces the different random number.
    for (size_t i = 0; i < batchSize; ++i)
    {
        EXPECT_TRUE(randValsSeed0HostPtr[i] != randValsSeed1HostPtr[i])
            << tc::fmtstr("Fail at val[%d]=%d <> val[%d]=%d", i, randValsSeed0HostPtr[i], i, randValsSeed1HostPtr[i]);
    }

    // The same seed produces the same random number.
    for (size_t i = 0; i < batchSize; ++i)
    {
        EXPECT_TRUE(randValsSeed0HostPtr[i] == randValsSeed0AltHostPtr[i]) << tc::fmtstr(
            "Fail at val[%d]=%d <> val[%d]=%d", i, randValsSeed0HostPtr[i], i, randValsSeed0AltHostPtr[i]);
    }
}

TEST_F(SamplingUtilsKernelTest, CurandBatchInitialize)
{
    int32_t batchSize = 127;

    curandState_t* curandStates;
    cudaMalloc(&curandStates, sizeof(curandState_t) * batchSize);

    auto randomSeedsHost = mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT64);
    auto randomSeedsHostPtr = bufferCast<int64_t>(*randomSeedsHost);
    const size_t periodSize = 3;
    for (size_t i = 0; i < batchSize; ++i)
    {
        randomSeedsHostPtr[i] = i / periodSize;
    }
    auto randomSeedsDevice = mBufferManager->copyFrom(*randomSeedsHost, MemoryType::kGPU);

    // Initialize curand states.
    tk::invokeCurandBatchInitialize(
        curandStates, batchSize, reinterpret_cast<uint64_t*>(bufferCast<int64_t>(*randomSeedsDevice)), mStream->get());
    sync_check_cuda_error();

    // Generate random numbers using initialized curand states.
    auto randValsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
    generateRandomNumber<<<1, batchSize, 0, this->mStream->get()>>>(
        bufferCast<int32_t>(*randValsDevice), curandStates, batchSize);
    const auto randValsHost = mBufferManager->copyFrom(*randValsDevice, MemoryType::kCPU);
    this->mStream->synchronize();

    const auto randValsHostPtr = bufferCast<int32_t>(*randValsHost);

    // The same seed produces the same random number.
    for (size_t i = 0; i + periodSize - 1 < batchSize; i += periodSize)
    {
        for (size_t j = 1; j < periodSize; ++j)
        {
            EXPECT_TRUE(randValsHostPtr[i] == randValsHostPtr[i + j])
                << tc::fmtstr("Fail at val[%d]=%d <> val[%d]=%d", i, randValsHostPtr[i], i + j, randValsHostPtr[i + j]);
        }
    }

    cudaFree(curandStates);
    sync_check_cuda_error();
}

TEST_F(SamplingUtilsKernelTest, invokeTopPInitialize)
{
    const int32_t batchSize = 8;
    const int32_t vocabSize = 256;

    const auto topPIdValsDevice
        = this->mBufferManager->gpu(ITensor::makeShape({batchSize, vocabSize}), nvinfer1::DataType::kINT32);
    const auto beginOffsetsDevice
        = this->mBufferManager->gpu(ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32);
    const auto endOffsetsDevice
        = this->mBufferManager->gpu(ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32);

    tk::invokeTopPInitialize(bufferCast<int32_t>(*topPIdValsDevice), bufferCast<int32_t>(*endOffsetsDevice),
        bufferCast<int32_t>(*beginOffsetsDevice), batchSize, vocabSize, this->mStream->get());

    const auto topPIdValsHost = this->mBufferManager->copyFrom(*topPIdValsDevice, MemoryType::kCPU);
    const auto endOffsetsHost = this->mBufferManager->copyFrom(*endOffsetsDevice, MemoryType::kCPU);
    const auto beginOffsetsHost = this->mBufferManager->copyFrom(*beginOffsetsDevice, MemoryType::kCPU);

    this->mStream->synchronize();

    const auto topPIdValsHostPtr = bufferCast<int32_t>(*topPIdValsHost);
    const auto endOffsetsHostPtr = bufferCast<int32_t>(*endOffsetsHost);
    const auto beginOffsetsHostPtr = bufferCast<int32_t>(*beginOffsetsHost);

    for (int32_t bi = 0; bi < batchSize + 1; ++bi)
    {
        EXPECT_EQ(endOffsetsHostPtr[bi], bi * vocabSize);
        EXPECT_EQ(beginOffsetsHostPtr[bi], bi * vocabSize);
    }
    for (int32_t bi = 0; bi < batchSize; ++bi)
    {
        for (int32_t vi = 0; vi < vocabSize; ++vi)
        {
            EXPECT_EQ(topPIdValsHostPtr[bi * vocabSize + vi], vi);
        }
    }
};

template <typename T>
class SamplingUtilsTypedKernelTest : public SamplingKernelTest<T>
{
public:
    void testAddBiasEndMaskSoftmax(bool hasBias, bool computeSoftmax)
    {
        int32_t batchSize = 16;
        int32_t vocabSize = 51000;
        int32_t vocabSizePadded = tc::divUp(vocabSize, 256) * 256;

        auto logitsHost = this->mBufferManager->pinned(ITensor::makeShape({batchSize, vocabSizePadded}),
            std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);
        auto refLogitsHost = this->mBufferManager->pinned(
            ITensor::makeShape({batchSize, vocabSizePadded}), nvinfer1::DataType::kFLOAT);
        auto biasHost = this->mBufferManager->pinned(ITensor::makeShape({vocabSize}),
            std::is_same_v<T, float> ? nvinfer1::DataType::kFLOAT : nvinfer1::DataType::kHALF);
        auto endIdsHost = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
        auto finishedHost = this->mBufferManager->pinned(
            ITensor::makeShape({batchSize}), TRTDataType<tk::FinishedState::UnderlyingType>::value);

        auto logitsHostPtr = bufferCast<T>(*logitsHost);
        auto refLogitsHostPtr = bufferCast<float>(*refLogitsHost);
        auto biasHostPtr = bufferCast<T>(*biasHost);
        initRandom(logitsHostPtr, batchSize * vocabSizePadded, -3.0f, 3.0f);
        initRandom(biasHostPtr, vocabSize, -3.0f, 3.0f);

        auto logitsDevice = this->mBufferManager->copyFrom(*logitsHost, MemoryType::kGPU);
        auto biasDevice = this->mBufferManager->copyFrom(*biasHost, MemoryType::kGPU);

        auto endIdsHostPtr = bufferCast<int32_t>(*endIdsHost);
        auto finishedHostPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*finishedHost));

        std::mt19937 gen(42);
        std::uniform_int_distribution<> endIdsDistr(
            0, vocabSize - 1); // -1 because uniform_int_distribution generates closed interval
        std::uniform_real_distribution<> finishedDist(0, 1); // uniform distribution between 0 and 1

        for (SizeType bi = 0; bi < batchSize; ++bi)
        {
            endIdsHostPtr[bi] = endIdsDistr(gen);

            finishedHostPtr[bi] = finishedDist(gen) < 0.3 ? tk::FinishedState::finished() : tk::FinishedState::empty();
        }

        auto endIdsDevice = this->mBufferManager->copyFrom(*endIdsHost, MemoryType::kGPU);
        auto finishedDevice = this->mBufferManager->copyFrom(*finishedHost, MemoryType::kGPU);

        const auto biasDevicePtr = hasBias ? bufferCast<T>(*biasDevice) : nullptr;
        if (!computeSoftmax)
        {
            tk::invokeAddBiasEndMask(bufferCast<T>(*logitsDevice), biasDevicePtr, bufferCast<int32_t>(*endIdsDevice),
                reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*finishedDevice)),
                batchSize, vocabSize, vocabSizePadded, this->mStream->get());
        }
        else
        {
            tk::invokeAddBiasSoftMax(bufferCast<T>(*logitsDevice), bufferCast<T>(*logitsDevice), biasDevicePtr,
                bufferCast<int32_t>(*endIdsDevice),
                reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*finishedDevice)),
                batchSize, vocabSize, vocabSizePadded, this->mStream->get());
        }

        this->mStream->synchronize();

        const auto outLogitsHost = this->mBufferManager->copyFrom(*logitsDevice, MemoryType::kCPU);
        const auto outLogitsHostPtr = bufferCast<T>(*outLogitsHost);

        const bool IS_FP16 = std::is_same<T, half>::value;
        const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
        for (SizeType bi = 0; bi < batchSize; ++bi)
        {
            float maxLogit = -1 * FLT_MAX;
            for (SizeType vi = 0; vi < vocabSizePadded; ++vi)
            {
                const auto idx = bi * vocabSizePadded + vi;
                auto refLogit = logitsHostPtr[idx];
                if (vi >= vocabSize)
                {
                    refLogit = -MAX_T_VAL;
                }
                else if (finishedHostPtr[bi].isFinished())
                {
                    refLogit = (vi == endIdsHostPtr[bi]) ? MAX_T_VAL : -MAX_T_VAL;
                }
                else if (hasBias)
                {
                    refLogit += biasHostPtr[vi];
                }
                refLogitsHostPtr[idx] = refLogit;
                maxLogit = std::max(static_cast<float>(refLogit), maxLogit);
            }
            if (computeSoftmax)
            {
                float sumExp = 0.f;
                for (SizeType vi = 0; vi < vocabSizePadded; ++vi)
                {
                    const auto idx = bi * vocabSizePadded + vi;
                    float refLogit = refLogitsHostPtr[idx];
                    refLogitsHostPtr[idx] = std::exp(refLogit - maxLogit);
                    sumExp += static_cast<float>(refLogitsHostPtr[idx]);
                }
                for (SizeType vi = 0; vi < vocabSizePadded; ++vi)
                {
                    const auto idx = bi * vocabSizePadded + vi;
                    float refLogit = refLogitsHostPtr[idx];
                    refLogitsHostPtr[idx] = refLogit / (sumExp + 1e-6f);
                }
            }
            for (SizeType vi = 0; vi < vocabSizePadded; ++vi)
            {
                const auto idx = bi * vocabSizePadded + vi;
                auto refLogit = refLogitsHostPtr[idx];
                auto outLogit = outLogitsHostPtr[idx];
                ASSERT_TRUE(almostEqual(outLogit, refLogit)) << "bi: " << bi << " vi: " << vi;
            }
        }
    }
};

TYPED_TEST_SUITE(SamplingUtilsTypedKernelTest, FloatAndHalfTypes);

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskWithBias)
{
    this->testAddBiasEndMaskSoftmax(true, false);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskWithoutBias)
{
    this->testAddBiasEndMaskSoftmax(false, false);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskSoftmaxWithBias)
{
    this->testAddBiasEndMaskSoftmax(true, true);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskSoftmaxWithoutBias)
{
    this->testAddBiasEndMaskSoftmax(false, true);
}

} // end of namespace
