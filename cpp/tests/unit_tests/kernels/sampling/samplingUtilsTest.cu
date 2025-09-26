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

#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tests/unit_tests/kernels/sampling/samplingTest.h"
#include <random>

using namespace tensorrt_llm::tests::kernels::sampling;
using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;

namespace
{

__global__ void generateRandomNumber(
    SizeType32* vals, SizeType32 const* batchSlots, curandState_t* states, SizeType32 batchSize)
{
    auto const bid = static_cast<SizeType32>(threadIdx.x);
    if (bid < batchSize)
    {
        auto const batchSlot = batchSlots[bid];
        vals[bid] = curand(states + batchSlot);
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
        auto batchSlots = getDefaultBatchSlots(batchSize);
        auto batchSlotsPtr = bufferCast<SizeType32>(*batchSlots);
        tk::invokeCurandInitialize(curandStates, batchSlotsPtr, batchSize, seed, this->mStream->get());
        sync_check_cuda_error(this->mStream->get());

        // Generate random numbers using initialized curand states.MemoryType
        auto randValsDevice = this->mBufferManager->gpu(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
        generateRandomNumber<<<1, batchSize, 0, this->mStream->get()>>>(
            bufferCast<int32_t>(*randValsDevice), batchSlotsPtr, curandStates, batchSize);
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
            << tc::fmtstr("Fail at val[%lu]=%d <> val[%lu]=%d", i, randValsSeed0HostPtr[i], i, randValsSeed1HostPtr[i]);
    }

    // The same seed produces the same random number.
    for (size_t i = 0; i < batchSize; ++i)
    {
        EXPECT_TRUE(randValsSeed0HostPtr[i] == randValsSeed0AltHostPtr[i]) << tc::fmtstr(
            "Fail at val[%lu]=%d <> val[%lu]=%d", i, randValsSeed0HostPtr[i], i, randValsSeed0AltHostPtr[i]);
    }
}

TEST_F(SamplingUtilsKernelTest, CurandBatchInitialize)
{
    SizeType32 batchSize = 127;

    curandState_t* curandStates;
    cudaMalloc(&curandStates, sizeof(curandState_t) * 2 * batchSize);

    auto randomSeedsHost = mBufferManager->pinnedPool(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT64);
    auto randomSeedsHostPtr = bufferCast<int64_t>(*randomSeedsHost);
    size_t const periodSize = 3;
    for (size_t i = 0; i < batchSize; ++i)
    {
        randomSeedsHostPtr[i] = i / periodSize;
    }
    auto randomSeedsDevice = mBufferManager->copyFrom(*randomSeedsHost, MemoryType::kGPU);

    auto batchSlots = mBufferManager->pinnedPool(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);

    auto batchSlotsPtr = bufferCast<SizeType32>(*batchSlots);
    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        batchSlotsPtr[bi] = 2 * bi;
    }

    // Initialize curand states.
    tk::invokeCurandBatchInitialize(curandStates, batchSlotsPtr, batchSize,
        reinterpret_cast<uint64_t*>(bufferCast<int64_t>(*randomSeedsDevice)), mStream->get());
    sync_check_cuda_error(mStream->get());

    // Generate random numbers using initialized curand states.
    auto randValsDevice = mBufferManager->gpu(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
    generateRandomNumber<<<1, batchSize, 0, this->mStream->get()>>>(
        bufferCast<SizeType32>(*randValsDevice), batchSlotsPtr, curandStates, batchSize);
    auto const randValsHost = mBufferManager->copyFrom(*randValsDevice, MemoryType::kCPU);
    this->mStream->synchronize();

    auto* const randValsHostPtr = bufferCast<SizeType32>(*randValsHost);

    // The same seed produces the same random number.
    for (SizeType32 bi = 0; bi + periodSize - 1 < batchSize; bi += periodSize)
    {
        for (size_t pi = 1; pi < periodSize; ++pi)
        {
            EXPECT_TRUE(randValsHostPtr[bi] == randValsHostPtr[bi + pi]) << tc::fmtstr(
                "Fail at val[%d]=%d <> val[%lu]=%d", bi, randValsHostPtr[bi], bi + pi, randValsHostPtr[bi + pi]);
        }
    }

    cudaFree(curandStates);
    sync_check_cuda_error(mStream->get());
}

template <typename T>
class SamplingUtilsTypedKernelTest : public SamplingKernelTest<T>
{
public:
    void testAddBiasEndMaskSoftmax(bool hasBias, bool computeSoftmax, bool useLogitsPtrs, bool computeEntropy,
        bool hasTemperature, SizeType32 maxBeamWidth)
    {
        auto const dataType = TRTDataType<T>::value;
        auto const ptrType = TRTDataType<T*>::value;

        int32_t const batchSize = 16;
        int32_t const maxBatchSize = 2 * batchSize;
        int32_t const vocabSize = 51000;
        int32_t const vocabSizePadded = tc::divUp(vocabSize, 256) * 256;

        if (computeEntropy)
        {
            computeSoftmax = true;
        }

        auto logitsHost = this->mBufferManager->pinnedPool(
            ITensor::makeShape({batchSize, maxBeamWidth, vocabSizePadded}), dataType);
        ITensor::SharedPtr logitsHostPtrs = this->mBufferManager->pinnedPool(ITensor::makeShape({batchSize}), ptrType);
        auto refLogitsHost = this->mBufferManager->pinnedPool(
            ITensor::makeShape({batchSize, maxBeamWidth, vocabSizePadded}), nvinfer1::DataType::kFLOAT);
        auto refEntropyHost = this->mBufferManager->pinnedPool(
            ITensor::makeShape({maxBatchSize, maxBeamWidth}), nvinfer1::DataType::kFLOAT);
        auto entropyDevice
            = this->mBufferManager->gpu(ITensor::makeShape({maxBatchSize, maxBeamWidth}), nvinfer1::DataType::kFLOAT);

        auto biasHost = this->mBufferManager->pinnedPool(ITensor::makeShape({vocabSize}), dataType);
        auto temperatureHost
            = this->mBufferManager->pinnedPool(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kFLOAT);

        auto endIdsHost
            = this->mBufferManager->pinnedPool(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
        auto beamWidthsHost
            = this->mBufferManager->pinnedPool(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);

        ITensor::SharedPtr finishedHost = this->mBufferManager->pinnedPool(
            ITensor::makeShape({maxBeamWidth, maxBatchSize}), TRTDataType<tk::FinishedState::UnderlyingType>::value);

        auto batchSlots = this->mBufferManager->pinnedPool(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);

        auto batchSlotsPtr = bufferCast<int32_t>(*batchSlots);
        auto beamWidthsHostPtr = bufferCast<SizeType32>(*beamWidthsHost);

        std::mt19937 gen(42);
        std::uniform_int_distribution<> beamWidthDistr(1, maxBeamWidth);
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            batchSlotsPtr[bi] = 2 * bi;
            beamWidthsHostPtr[batchSlotsPtr[bi]] = beamWidthDistr(gen);
        }

        auto logitsHostPtr = bufferCast<T>(*logitsHost);
        auto refLogitsHostPtr = bufferCast<float>(*refLogitsHost);
        auto refEntropyHostPtr = bufferCast<float>(*refEntropyHost);
        auto biasHostPtr = bufferCast<T>(*biasHost);
        auto temperatureHostPtr = bufferCast<float>(*temperatureHost);
        initRandom(logitsHostPtr, batchSize * maxBeamWidth * vocabSizePadded, -3.0f, 3.0f);
        initRandom(biasHostPtr, vocabSize, -3.0f, 3.0f);
        initRandom(temperatureHostPtr, maxBatchSize, 0.8f, 2.0f);

        auto logitsHostPtrsData = reinterpret_cast<T**>(bufferCast<int64_t>(*logitsHostPtrs));

        auto logitsDevice = this->mBufferManager->copyFrom(*logitsHost, MemoryType::kGPU);
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto logitsDevicePtr = bufferCast<T>(*logitsDevice);
            logitsHostPtrsData[bi] = logitsDevicePtr + bi * maxBeamWidth * vocabSizePadded;
        }
        auto biasDevice = this->mBufferManager->copyFrom(*biasHost, MemoryType::kGPU);
        auto temperatureDevice = this->mBufferManager->copyFrom(*temperatureHost, MemoryType::kGPU);

        auto endIdsHostPtr = bufferCast<int32_t>(*endIdsHost);
        auto finishedHostPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*finishedHost));

        std::uniform_int_distribution<> endIdsDistr(
            0, vocabSize - 1); // -1 because uniform_int_distribution generates closed interval
        std::uniform_real_distribution<> finishedDist(0, 1); // uniform distribution between 0 and 1

        for (SizeType32 bi = 0; bi < maxBatchSize; ++bi)
        {
            endIdsHostPtr[bi] = endIdsDistr(gen);
            for (SizeType32 bwi = 0; bwi < maxBeamWidth; ++bwi)
            {
                finishedHostPtr[bwi * maxBatchSize + bi]
                    = finishedDist(gen) < 0.3 ? tk::FinishedState::finished() : tk::FinishedState::empty();
            }
        }

        auto endIdsDevice = this->mBufferManager->copyFrom(*endIdsHost, MemoryType::kGPU);
        auto finishedDevice = this->mBufferManager->copyFrom(*finishedHost, MemoryType::kGPU);
        auto beamWidthsDevice = this->mBufferManager->copyFrom(*beamWidthsHost, MemoryType::kGPU);

        auto const beamWidthsPtr = maxBeamWidth == 1 ? nullptr : bufferCast<SizeType32>(*beamWidthsDevice);
        auto const biasDevicePtr = hasBias ? bufferCast<T>(*biasDevice) : nullptr;

        auto const temperatureDevicePtr = hasTemperature ? bufferCast<float>(*temperatureDevice) : nullptr;
        auto const entropyDevicePtr = computeEntropy ? bufferCast<float>(*entropyDevice) : nullptr;

        auto const logitsDevicePtr = useLogitsPtrs ? (T*) nullptr : bufferCast<T>(*logitsDevice);
        auto const logitsPtrsDevicePtr = useLogitsPtrs ? logitsHostPtrsData : (T**) nullptr;

        tk::BiasSoftmaxParams<T> biasSoftmaxParams;
        biasSoftmaxParams.logits = logitsDevicePtr;
        biasSoftmaxParams.logitsPtrs = logitsPtrsDevicePtr;
        biasSoftmaxParams.probs = bufferCast<T>(*logitsDevice);
        biasSoftmaxParams.outputEntropy = entropyDevicePtr;
        biasSoftmaxParams.bias = biasDevicePtr;
        biasSoftmaxParams.temperatures = temperatureDevicePtr;
        biasSoftmaxParams.endIds = bufferCast<int32_t>(*endIdsDevice);
        biasSoftmaxParams.finished
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*finishedDevice));
        biasSoftmaxParams.beamWidths = beamWidthsPtr;
        biasSoftmaxParams.batchSlots = batchSlotsPtr;
        biasSoftmaxParams.batchSize = batchSize;
        biasSoftmaxParams.maxBatchSize = maxBatchSize;
        biasSoftmaxParams.maxBeamWidth = maxBeamWidth;
        biasSoftmaxParams.vocabSize = vocabSize;
        biasSoftmaxParams.vocabSizePadded = vocabSizePadded;
        biasSoftmaxParams.skipSoftMax = !computeSoftmax;
        biasSoftmaxParams.batchSlotsLogits = false;
        biasSoftmaxParams.checkParams();
        tk::invokeAddBiasSoftMax(biasSoftmaxParams, this->mStream->get());

        auto const outLogitsHost = this->mBufferManager->copyFrom(*logitsDevice, MemoryType::kCPU);
        auto const outLogitsHostPtr = bufferCast<T>(*outLogitsHost);

        float* outEntropyHostPtr{nullptr};
        ITensor::SharedPtr outputEntropyHost{nullptr};
        if (computeEntropy)
        {
            outputEntropyHost = this->mBufferManager->copyFrom(*entropyDevice, MemoryType::kCPU);
            outEntropyHostPtr = bufferCast<float>(*outputEntropyHost);
        }

        this->mStream->synchronize();

        bool const IS_FP16 = std::is_same<T, half>::value;
        T const MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
        float const EPSILON = (IS_FP16) ? 1e-3f : 1e-6f;
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            for (SizeType32 bwi = 0; bwi < beamWidthsHostPtr[batchSlot]; ++bwi)
            {
                float maxLogit = -1 * FLT_MAX;
                for (SizeType32 vi = 0; vi < vocabSizePadded; ++vi)
                {
                    auto const idx = (bi * maxBeamWidth + bwi) * vocabSizePadded + vi;
                    auto refLogit = logitsHostPtr[idx];
                    if (hasTemperature)
                    {
                        refLogit *= (1.f / (temperatureHostPtr[batchSlot] + EPSILON));
                    }
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
                    for (SizeType32 vi = 0; vi < vocabSizePadded; ++vi)
                    {
                        auto const idx = (bi * maxBeamWidth + bwi) * vocabSizePadded + vi;
                        float refLogit = refLogitsHostPtr[idx];
                        refLogitsHostPtr[idx] = std::exp(refLogit - maxLogit);
                        sumExp += static_cast<float>(refLogitsHostPtr[idx]);
                    }
                    for (SizeType32 vi = 0; vi < vocabSizePadded; ++vi)
                    {
                        auto const idx = (bi * maxBeamWidth + bwi) * vocabSizePadded + vi;
                        float refLogit = refLogitsHostPtr[idx];
                        refLogitsHostPtr[idx] = refLogit / (sumExp + EPSILON);
                    }
                }
                if (computeEntropy)
                {
                    float entropy{0.f};
                    for (SizeType32 vi = 0; vi < vocabSizePadded; ++vi)
                    {
                        auto const idx = (bi * maxBeamWidth + bwi) * vocabSizePadded + vi;
                        auto const prob = refLogitsHostPtr[idx];

                        entropy += prob * logf(prob + EPSILON);
                    }
                    refEntropyHostPtr[bi * maxBeamWidth + bwi] = -entropy;
                }
            }
        }
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            for (SizeType32 bwi = 0; bwi < beamWidthsHostPtr[batchSlot]; ++bwi)
            {
                for (SizeType32 vi = 0; vi < vocabSizePadded; ++vi)
                {
                    auto const idx = (bi * maxBeamWidth + bwi) * vocabSizePadded + vi;
                    auto refLogit = refLogitsHostPtr[idx];
                    auto outLogit = outLogitsHostPtr[idx];
                    ASSERT_TRUE(almostEqual(outLogit, refLogit, /* atol */ 1e-3f))
                        << "bi: " << bi << " beam: " << bwi << " vi: " << vi;
                }
                if (computeEntropy)
                {
                    auto const idxLocal = bi * maxBeamWidth + bwi;
                    auto const idxGlobal = batchSlot * maxBeamWidth + bwi;
                    auto refEntropy = refEntropyHostPtr[idxLocal];
                    auto outEntropy = outEntropyHostPtr[idxGlobal];
                    ASSERT_TRUE(almostEqual(outEntropy, refEntropy, /* atol */ 1e-3f))
                        << "bi: " << bi << " beam: " << bwi;
                }
            }
        }
    }
};

TYPED_TEST_SUITE(SamplingUtilsTypedKernelTest, FloatAndHalfTypes);

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskWithBias)
{
    this->testAddBiasEndMaskSoftmax(/* hasBias */ true, /* computeSoftmax */ false, /* useLogitsPtrs */ false,
        /* computeEntropy */ false, /* hasTemperature */ false, 1);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskWithTemperature)
{
    this->testAddBiasEndMaskSoftmax(/* hasBias */ false, /* computeSoftmax */ false, /* useLogitsPtrs */ false,
        /* computeEntropy */ false, /* hasTemperature */ true, 1);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskWithoutBias)
{
    this->testAddBiasEndMaskSoftmax(/* hasBias */ false, /* computeSoftmax */ false, /* useLogitsPtrs */ false,
        /* computeEntropy */ false, /* hasTemperature */ false, 1);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskSoftmaxWithBias)
{
    this->testAddBiasEndMaskSoftmax(/* hasBias */ true, /* computeSoftmax */ true, /* useLogitsPtrs */ false,
        /* computeEntropy */ false, /* hasTemperature */ false, 1);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskSoftmaxWithoutBias)
{
    this->testAddBiasEndMaskSoftmax(/* hasBias */ false, /* computeSoftmax */ true, /* useLogitsPtrs */ false,
        /* computeEntropy */ false, /* hasTemperature */ false, 1);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskSoftmaxLogitsPtrsBw1)
{
    this->testAddBiasEndMaskSoftmax(/* hasBias */ false, /* computeSoftmax */ true, /* useLogitsPtrs */ true,
        /* computeEntropy */ false, /* hasTemperature */ false, 1);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskSoftmaxLogitsPtrsEntropyBw1)
{
    this->testAddBiasEndMaskSoftmax(/* hasBias */ false, /* computeSoftmax */ true, /* useLogitsPtrs */ true,
        /* computeEntropy */ true, /* hasTemperature */ false, 1);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskSoftmaxLogitsPtrsTemperatureEntropyBw1)
{
    this->testAddBiasEndMaskSoftmax(/* hasBias */ false, /* computeSoftmax */ true, /* useLogitsPtrs */ true,
        /* computeEntropy */ true, /* hasTemperature */ true, 1);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskSoftmaxLogitsPtrsTemperatureEntropyBw16)
{
    this->testAddBiasEndMaskSoftmax(/* hasBias */ false, /* computeSoftmax */ true, /* useLogitsPtrs */ true,
        /* computeEntropy */ true, /* hasTemperature */ true, 16);
}

TYPED_TEST(SamplingUtilsTypedKernelTest, AddBiasEndMaskSoftmaxLogitsPtrsBw2)
{
    this->testAddBiasEndMaskSoftmax(/* hasBias */ false, /* computeSoftmax */ true, /* useLogitsPtrs */ true,
        /* computeEntropy */ false, /* hasTemperature */ false, 2);
}
} // end of namespace
