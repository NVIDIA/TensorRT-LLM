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
        tk::invokeCurandInitialize(curandStates, nullptr, batchSize, seed, this->mStream->get());
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

    auto const randValsSeed0Host = initSeedAndGenerateNumbers(0);
    auto const randValsSeed1Host = initSeedAndGenerateNumbers(1);
    auto const randValsSeed0AltHost = initSeedAndGenerateNumbers(0);

    auto const randValsSeed0HostPtr = bufferCast<int32_t>(*randValsSeed0Host);
    auto const randValsSeed1HostPtr = bufferCast<int32_t>(*randValsSeed1Host);
    auto const randValsSeed0AltHostPtr = bufferCast<int32_t>(*randValsSeed0AltHost);

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
    size_t const periodSize = 3;
    for (size_t i = 0; i < batchSize; ++i)
    {
        randomSeedsHostPtr[i] = i / periodSize;
    }
    auto randomSeedsDevice = mBufferManager->copyFrom(*randomSeedsHost, MemoryType::kGPU);

    auto batchSlots = mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);

    auto batchSlotsPtr = bufferCast<int32_t>(*batchSlots);
    for (SizeType bi = 0; bi < batchSize; ++bi)
    {
        batchSlotsPtr[batchSize - bi - 1] = bi;
    }

    // Initialize curand states.
    tk::invokeCurandBatchInitialize(curandStates, batchSlotsPtr, batchSize,
        reinterpret_cast<uint64_t*>(bufferCast<int64_t>(*randomSeedsDevice)), mStream->get());
    sync_check_cuda_error();

    // Generate random numbers using initialized curand states.
    auto randValsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
    generateRandomNumber<<<1, batchSize, 0, this->mStream->get()>>>(
        bufferCast<int32_t>(*randValsDevice), curandStates, batchSize);
    auto const randValsHost = mBufferManager->copyFrom(*randValsDevice, MemoryType::kCPU);
    this->mStream->synchronize();

    auto const randValsHostPtr = bufferCast<int32_t>(*randValsHost);

    // The same seed produces the same random number.
    for (size_t i = 0; i + periodSize - 1 < batchSize; i += periodSize)
    {
        for (size_t j = 1; j < periodSize; ++j)
        {
            // FIXME(nkorobov): this has to be accessed via batchSlot
            EXPECT_TRUE(randValsHostPtr[i] == randValsHostPtr[i + j])
                << tc::fmtstr("Fail at val[%d]=%d <> val[%d]=%d", i, randValsHostPtr[i], i + j, randValsHostPtr[i + j]);
        }
    }

    cudaFree(curandStates);
    sync_check_cuda_error();
}

TEST_F(SamplingUtilsKernelTest, invokeTopPInitialize)
{
    int32_t const batchSize = 8;
    int32_t const vocabSize = 256;

    auto const topPIdValsDevice
        = this->mBufferManager->gpu(ITensor::makeShape({batchSize, vocabSize}), nvinfer1::DataType::kINT32);
    auto const beginOffsetsDevice
        = this->mBufferManager->gpu(ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32);
    auto const endOffsetsDevice
        = this->mBufferManager->gpu(ITensor::makeShape({batchSize + 1}), nvinfer1::DataType::kINT32);

    tk::invokeTopPInitialize(bufferCast<int32_t>(*topPIdValsDevice), bufferCast<int32_t>(*endOffsetsDevice),
        bufferCast<int32_t>(*beginOffsetsDevice), batchSize, vocabSize, this->mStream->get());

    auto const topPIdValsHost = this->mBufferManager->copyFrom(*topPIdValsDevice, MemoryType::kCPU);
    auto const endOffsetsHost = this->mBufferManager->copyFrom(*endOffsetsDevice, MemoryType::kCPU);
    auto const beginOffsetsHost = this->mBufferManager->copyFrom(*beginOffsetsDevice, MemoryType::kCPU);

    this->mStream->synchronize();

    auto const topPIdValsHostPtr = bufferCast<int32_t>(*topPIdValsHost);
    auto const endOffsetsHostPtr = bufferCast<int32_t>(*endOffsetsHost);
    auto const beginOffsetsHostPtr = bufferCast<int32_t>(*beginOffsetsHost);

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
    void testAddBiasEndMaskSoftmax(bool hasBias, bool computeSoftmax, bool useLogitsPtrs, SizeType beamWidth)
    {
        auto const dataType = TRTDataType<T>::value;
        auto const ptrType = TRTDataType<T*>::value;

        int32_t const batchSize = 16;
        int32_t const maxBatchSize = 2 * batchSize;
        int32_t const vocabSize = 51000;
        int32_t const vocabSizePadded = tc::divUp(vocabSize, 256) * 256;

        auto logitsHost
            = this->mBufferManager->pinned(ITensor::makeShape({batchSize, beamWidth, vocabSizePadded}), dataType);
        auto logitsHostPtrs = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), ptrType);
        auto refLogitsHost = this->mBufferManager->pinned(
            ITensor::makeShape({batchSize, beamWidth, vocabSizePadded}), nvinfer1::DataType::kFLOAT);
        auto biasHost = this->mBufferManager->pinned(ITensor::makeShape({vocabSize}), dataType);
        auto endIdsHost = this->mBufferManager->pinned(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
        auto finishedHost = this->mBufferManager->pinned(
            ITensor::makeShape({beamWidth, maxBatchSize}), TRTDataType<tk::FinishedState::UnderlyingType>::value);

        auto batchSlots = this->mBufferManager->pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);

        auto batchSlotsPtr = bufferCast<int32_t>(*batchSlots);
        for (SizeType bi = 0; bi < batchSize; ++bi)
        {
            batchSlotsPtr[bi] = 2 * bi;
        }

        auto logitsHostPtr = bufferCast<T>(*logitsHost);
        auto refLogitsHostPtr = bufferCast<float>(*refLogitsHost);
        auto biasHostPtr = bufferCast<T>(*biasHost);
        initRandom(logitsHostPtr, batchSize * beamWidth * vocabSizePadded, -3.0f, 3.0f);
        initRandom(biasHostPtr, vocabSize, -3.0f, 3.0f);

        auto logitsHostPtrsData = reinterpret_cast<T**>(bufferCast<int64_t>(*logitsHostPtrs));
        for (SizeType bi = 0; bi < batchSize; ++bi)
        {
            logitsHostPtrsData[bi] = logitsHostPtr + bi * beamWidth * vocabSizePadded;
        }

        auto logitsDevice = this->mBufferManager->copyFrom(*logitsHost, MemoryType::kGPU);
        auto biasDevice = this->mBufferManager->copyFrom(*biasHost, MemoryType::kGPU);

        auto endIdsHostPtr = bufferCast<int32_t>(*endIdsHost);
        auto finishedHostPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*finishedHost));

        std::mt19937 gen(42);
        std::uniform_int_distribution<> endIdsDistr(
            0, vocabSize - 1); // -1 because uniform_int_distribution generates closed interval
        std::uniform_real_distribution<> finishedDist(0, 1); // uniform distribution between 0 and 1

        for (SizeType bi = 0; bi < maxBatchSize; ++bi)
        {
            endIdsHostPtr[bi] = endIdsDistr(gen);
            for (SizeType bwi = 0; bwi < beamWidth; ++bwi)
            {
                finishedHostPtr[bwi * maxBatchSize + bi]
                    = finishedDist(gen) < 0.3 ? tk::FinishedState::finished() : tk::FinishedState::empty();
            }
        }

        auto endIdsDevice = this->mBufferManager->copyFrom(*endIdsHost, MemoryType::kGPU);
        auto finishedDevice = this->mBufferManager->copyFrom(*finishedHost, MemoryType::kGPU);

        auto const biasDevicePtr = hasBias ? bufferCast<T>(*biasDevice) : nullptr;
        auto const logitsDevicePtr = useLogitsPtrs ? (T*) nullptr : bufferCast<T>(*logitsDevice);
        auto const logitsPtrsDevicePtr = useLogitsPtrs ? logitsHostPtrsData : (T**) nullptr;
        tk::invokeAddBiasSoftMax(logitsDevicePtr, logitsPtrsDevicePtr, bufferCast<T>(*logitsDevice), biasDevicePtr,
            bufferCast<int32_t>(*endIdsDevice),
            reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*finishedDevice)),
            batchSlotsPtr, batchSize, maxBatchSize, beamWidth, vocabSize, vocabSizePadded, !computeSoftmax, false,
            this->mStream->get());

        this->mStream->synchronize();

        auto const outLogitsHost = this->mBufferManager->copyFrom(*logitsDevice, MemoryType::kCPU);
        auto const outLogitsHostPtr = bufferCast<T>(*outLogitsHost);

        bool const IS_FP16 = std::is_same<T, half>::value;
        T const MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
        for (SizeType bi = 0; bi < batchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            for (SizeType bwi = 0; bwi < beamWidth; ++bwi)
            {
                float maxLogit = -1 * FLT_MAX;
                for (SizeType vi = 0; vi < vocabSizePadded; ++vi)
                {
                    auto const idx = (bi * beamWidth + bwi) * vocabSizePadded + vi;
                    auto refLogit = logitsHostPtr[idx];
                    if (vi >= vocabSize)
                    {
                        refLogit = -MAX_T_VAL;
                    }
                    else if (finishedHostPtr[bwi * maxBatchSize + batchSlot].isFinished())
                    {
                        refLogit = (vi == endIdsHostPtr[batchSlot]) ? MAX_T_VAL : -MAX_T_VAL;
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
                        auto const idx = (bi * beamWidth + bwi) * vocabSizePadded + vi;
                        float refLogit = refLogitsHostPtr[idx];
                        refLogitsHostPtr[idx] = std::exp(refLogit - maxLogit);
                        sumExp += static_cast<float>(refLogitsHostPtr[idx]);
                    }
                    for (SizeType vi = 0; vi < vocabSizePadded; ++vi)
                    {
                        auto const idx = (bi * beamWidth + bwi) * vocabSizePadded + vi;
                        float refLogit = refLogitsHostPtr[idx];
                        refLogitsHostPtr[idx] = refLogit / (sumExp + 1e-6f);
                    }
                }
            }
        }
        for (SizeType bi = 0; bi < batchSize; ++bi)
        {
            for (SizeType bwi = 0; bwi < beamWidth; ++bwi)
            {
                for (SizeType vi = 0; vi < vocabSizePadded; ++vi)
                {
                    auto const idx = (bi * beamWidth + bwi) * vocabSizePadded + vi;
                    auto refLogit = refLogitsHostPtr[idx];
                    auto outLogit = outLogitsHostPtr[idx];
                    ASSERT_TRUE(almostEqual(outLogit, refLogit)) << "bi: " << bi << " beam: " << bwi << " vi: " << vi;
                }
            }
        }
    }
};

TYPED_TEST_SUITE(SamplingUtilsTypedKernelTest, FloatAndHalfTypes);

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskWithBias)
{
    this->testAddBiasEndMaskSoftmax(true, false, false, 1);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskWithoutBias)
{
    this->testAddBiasEndMaskSoftmax(false, false, false, 1);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskSoftmaxWithBias)
{
    this->testAddBiasEndMaskSoftmax(true, true, false, 1);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskSoftmaxWithoutBias)
{
    this->testAddBiasEndMaskSoftmax(false, true, false, 1);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskSoftmaxLogitsPtrsBw1)
{
    this->testAddBiasEndMaskSoftmax(false, true, true, 1);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskSoftmaxLogitsPtrsBw2)
{
    this->testAddBiasEndMaskSoftmax(false, true, true, 2);
}
} // end of namespace
