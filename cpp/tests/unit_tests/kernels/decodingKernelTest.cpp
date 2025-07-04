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

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/kernels/speculativeDecoding/externalDraftTokensKernels.h"
#include "tensorrt_llm/kernels/speculativeDecoding/medusaDecodingKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/decoderState.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

#include <curand_kernel.h>
#include <gtest/gtest.h>

#include <random>
#include <unordered_set>

namespace tk = tensorrt_llm::kernels;
namespace tksp = tensorrt_llm::kernels::speculative_decoding;
namespace tc = tensorrt_llm::common;
namespace trk = tensorrt_llm::runtime::kernels;

using namespace tensorrt_llm::runtime;

namespace
{

inline bool almostEqual(float a, float b, float atol = 1e-2, float rtol = 1e-3)
{
    // Params: a = value to compare and b = reference
    // This function follows implementation of numpy.isclose(), which checks
    //   abs(a - b) <= (atol + rtol * abs(b)).
    // Note that the inequality above is asymmetric where b is considered as
    // a reference value. To account into both absolute/relative errors, it
    // uses absolute tolerance and relative tolerance at the same time. The
    // default values of atol and rtol borrowed from numpy.isclose(). For the
    // case of nan value, the result will be true.
    if (isnan(a) && isnan(b))
    {
        return true;
    }
    return fabs(a - b) <= (atol + rtol * fabs(b));
}

std::vector<float> calculateGaussianKernel(float sigma, int size)
{
    std::vector<float> kernel(size);
    float sum = 0.f;

    for (int i = 0; i < size; ++i)
    {
        int x = i - size / 2;
        kernel[i] = std::exp(-0.5f * (x * x) / (sigma * sigma));
        sum += kernel[i];
    }

    // Normalize the kernel
    for (int i = 0; i < size; ++i)
    {
        kernel[i] /= sum;
    }

    return kernel;
}

template <typename T>
void applyGaussianFilter(T* result, float const* input, int n, float sigma)
{
    int size = static_cast<int>(std::ceil(6.f * sigma));
    size = (size % 2 == 0) ? size + 1 : size;

    std::vector<float> kernel = calculateGaussianKernel(sigma, size);
    int halfSize = size / 2;

    for (int i = 0; i < n; ++i)
    {
        result[i] = T{0};
    }

    // Convolution operation
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            int k = i - halfSize + j;
            if (k >= 0 && k < n)
            {
                result[i] += input[k] * kernel[j];
            }
        }
    }
}

template void applyGaussianFilter(float* result, float const* input, int n, float sigma);
template void applyGaussianFilter(__half* result, float const* input, int n, float sigma);

template <typename T>
void probsToLogits(T const* probs, T* logits, SizeType32 n)
{
    constexpr float eps = 1e-6f;
    for (SizeType32 ni = 0; ni < n; ++ni)
    {
        auto const prob = std::max(eps, static_cast<float>(probs[ni]));
        logits[ni] = std::log(prob / (1.f - prob));
    }
}

template <typename T>
void softmax(T const* logits, T* probs, int n)
{
    float epsilon = 1e-6f;

    // Find the maximum logit value
    float maxLogits = -std::numeric_limits<float>::max();
    for (int ii = 0; ii < n; ++ii)
    {
        maxLogits = std::max(maxLogits, static_cast<float>(logits[ii]));
    }

    // Calculate the numerator of the softmax formula
    float expSum = 0.0;
    for (int ii = 0; ii < n; ++ii)
    {
        expSum += std::exp(static_cast<float>(logits[ii]) - maxLogits);
    }

    // Calculate softmax probabilities
    for (int ii = 0; ii < n; ++ii)
    {
        float prob = std::exp(static_cast<float>(logits[ii]) - maxLogits) / (expSum + epsilon);
        probs[ii] = prob;
    }
}

template void probsToLogits(float const* probs, float* logits, SizeType32 n);
template void probsToLogits(__half const* probs, __half* logits, SizeType32 n);

template <typename T>
void checkEquality(DecodingOutput::TensorPtr src, DecodingOutput::TensorPtr dst, char const* bufferName,
    tensorrt_llm::runtime::BufferManager& bufferManager)
{
    auto srcHost = bufferManager.copyFrom(*src, MemoryType::kPINNEDPOOL);
    auto dstHost = bufferManager.copyFrom(*dst, MemoryType::kPINNEDPOOL);
    bufferManager.getStream().synchronize();
    auto srcPtr = bufferCast<T>(*srcHost);
    auto dstPtr = bufferCast<T>(*dstHost);
    for (SizeType32 ii = 0; ii < src->getSize(); ++ii)
    {
        // since it's a simple copy, floats support the simple equality
        EXPECT_EQ(srcPtr[ii], dstPtr[ii]) << "Unequal values in buffer " << bufferName << " at ii: " << ii
                                          << " with values: src " << srcPtr[ii] << " dst " << dstPtr[ii] << std::endl;
    }
}

template <typename T>
void fillBufferWithRandom(ITensor& buffer, tensorrt_llm::runtime::BufferManager& bufferManager, std::mt19937& randGen)
{
    auto cpuBuffer = bufferManager.cpu(buffer.getShape(), TRTDataType<T>::value);

    auto const size = cpuBuffer->getSize();
    auto rawPtr = bufferCast<T>(*cpuBuffer);

    std::uniform_int_distribution<> dis(0, 255);

    for (SizeType32 i = 0; i < size; ++i)
    {
        rawPtr[i] = static_cast<T>(dis(randGen));
    }
    bufferManager.copy(*cpuBuffer, buffer);
}

class TestBeamHypothesesCopy : public ::testing::Test
{
public:
    DecodingOutput::BeamHypotheses srcBeams;
    DecodingOutput::BeamHypotheses dstBeams;
    DecodingOutput::TensorPtr mSrcCumLogProbs;
    DecodingOutput::TensorPtr mDstCumLogProbs;
    SizeType32 mNumSMs;

    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;

    std::mt19937 gen;

    void SetUp() override
    {
        mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        mNumSMs = deviceProp.multiProcessorCount;
        gen.seed(42U);
    }

    void initializeBuffers(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxSeqLen)
    {

        srcBeams.empty(*mBufferManager);
        srcBeams.reshape(batchSize, beamWidth, maxSeqLen);
        mSrcCumLogProbs = mBufferManager->gpu(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kFLOAT);

        setBuffers(srcBeams, mSrcCumLogProbs, 2);

        dstBeams.empty(*mBufferManager);
        dstBeams.reshape(batchSize, beamWidth, maxSeqLen);
        mDstCumLogProbs = mBufferManager->gpu(ITensor::makeShape({batchSize, beamWidth}), nvinfer1::DataType::kFLOAT);

        setBuffers(dstBeams, mDstCumLogProbs, 1);
    }

    void setBuffers(DecodingOutput::BeamHypotheses currBeams, DecodingOutput::TensorPtr cumLogProbs, int value)
    {
        fillBufferWithRandom<SizeType32>(*currBeams.outputIdsCBA, *mBufferManager, gen);
        fillBufferWithRandom<float>(*currBeams.logProbsCBA, *mBufferManager, gen);
        fillBufferWithRandom<SizeType32>(*currBeams.sequenceLengthsCBA, *mBufferManager, gen);
        fillBufferWithRandom<float>(*currBeams.cumLogProbsCBA, *mBufferManager, gen);
        fillBufferWithRandom<float>(*currBeams.normedScoresCBA, *mBufferManager, gen);
        fillBufferWithRandom<SizeType32>(*currBeams.numBeamsCBA, *mBufferManager, gen);
        fillBufferWithRandom<float>(*currBeams.minNormedScoresCBA, *mBufferManager, gen);
        fillBufferWithRandom<bool>(*currBeams.batchDones, *mBufferManager, gen);
        fillBufferWithRandom<float>(*cumLogProbs, *mBufferManager, gen);
    }

    void checkAllEqual()
    {
        checkEquality<SizeType32>(srcBeams.outputIdsCBA, dstBeams.outputIdsCBA, "outputIdsCBA", *mBufferManager);
        checkEquality<float>(srcBeams.logProbsCBA, dstBeams.logProbsCBA, "logProbsCBA", *mBufferManager);
        checkEquality<SizeType32>(
            srcBeams.sequenceLengthsCBA, dstBeams.sequenceLengthsCBA, "sequenceLengthsCBA", *mBufferManager);
        checkEquality<float>(srcBeams.cumLogProbsCBA, dstBeams.cumLogProbsCBA, "cumLogProbsCBA", *mBufferManager);
        checkEquality<float>(srcBeams.normedScoresCBA, dstBeams.normedScoresCBA, "normedScoresCBA", *mBufferManager);
        checkEquality<SizeType32>(srcBeams.numBeamsCBA, dstBeams.numBeamsCBA, "numBeamsCBA", *mBufferManager);
        checkEquality<float>(
            srcBeams.minNormedScoresCBA, dstBeams.minNormedScoresCBA, "minNormedScoresCBA", *mBufferManager);
        checkEquality<bool>(srcBeams.batchDones, dstBeams.batchDones, "batchDones", *mBufferManager);
        checkEquality<float>(mSrcCumLogProbs, mDstCumLogProbs, "cumLogProbs", *mBufferManager);
    }
};

// Test for invokeCopyBeamHypotheses
TEST_F(TestBeamHypothesesCopy, FullBatchTest)
{
    SizeType32 const batchSize{1024};
    SizeType32 const beamWidth{64};
    SizeType32 const maxSeqLen{2048};

    initializeBuffers(batchSize, beamWidth, maxSeqLen);
    mStream->synchronize();

    tk::invokeCopyBeamHypotheses(srcBeams, dstBeams, *mSrcCumLogProbs, *mDstCumLogProbs, *mStream, mNumSMs);
    mStream->synchronize();

    checkAllEqual();
}

TEST_F(TestBeamHypothesesCopy, SingleBatchTest)
{
    SizeType32 const batchSize{1};
    SizeType32 const beamWidth{64};
    SizeType32 const maxSeqLen{16384};

    initializeBuffers(batchSize, beamWidth, maxSeqLen);
    mStream->synchronize();

    tk::invokeCopyBeamHypotheses(srcBeams, dstBeams, *mSrcCumLogProbs, *mDstCumLogProbs, *mStream, mNumSMs);
    mStream->synchronize();

    checkAllEqual();
}

/**
 * @brief Fills a slice of a tensor with data from a source array.
 *
 * This function writes to `tensor`  from source array `src` at index `idx.
 * It optionally flattens the tensor before performing the insertion.
 * For example tensor if we wanted to write 5 values in the 3rd row of [1,10,100]
 * We will use (tensor, 2, 5, src, true, mBufferManager) where src is a buffer with at least 5 elems.
 *
 * @tparam T The type of elements in the source array.
 * @param tensor A shared pointer to the tensor to be modified. Also need to be of type T.
 * @param idx The index at which to start inserting data into the tensor.
 * @param insertLen The number of elements to insert from the source array into the tensor.
 * @param src An array containing the data to be inserted into the tensor.
 * @param flattenFirst A boolean flag indicating whether to flatten the first dimension of the tensor before insertion.
 * @param bufferManager A shared pointer to a BufferManager responsible for managing memory operations.
 */
template <typename T>
void fillTensorAtIndex(ITensor::SharedPtr tensor, SizeType32 idx, std::vector<T> src, bool flattenFirst,
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> bufferManager)
{
    SizeType32 insertLen = src.size();
    ITensor::SharedPtr target = ITensor::view(tensor);
    if (flattenFirst)
    {
        target->squeeze(0);
    }

    target = ITensor::slice(target, idx, 1);
    target->squeeze(0);
    target = ITensor::slice(target, 0, insertLen);
    bufferManager->copy(src.data(), *target);
}

class TestGatherTree : public ::testing::Test
{
public:
    SizeType32 batchSize{1};
    SizeType32 beamWidth{5};
    SizeType32 maxSeqLen{20};

    using TensorPtr = ITensor::SharedPtr;

    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream{nullptr};
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager{nullptr};

    std::unique_ptr<tensorrt_llm::runtime::decoder::DecoderState> mDecodingState{nullptr};

    SamplingConfig mSamplingConfig;

    TensorPtr mTargetOut{nullptr};

    void SetUp() override
    {
        mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);
    }

    // create the empty buffers with the correct shapes and zero them
    void createBuffers()
    {
        SizeType32 constexpr tensorParallelism{1};
        SizeType32 constexpr pipelineParallelism{1};
        SizeType32 constexpr contextParallelism{1};
        SizeType32 constexpr localRank{0};
        WorldConfig const worldConfig{tensorParallelism, pipelineParallelism, contextParallelism, localRank};

        SizeType32 constexpr vocabSize{51200};
        SizeType32 constexpr nbAttentionLayers{2};
        SizeType32 constexpr nbRnnLayers{0};
        SizeType32 constexpr nbHeads{16};
        SizeType32 constexpr hiddenSize{1024};
        nvinfer1::DataType constexpr dtype{nvinfer1::DataType::kFLOAT};
        ModelConfig modelConfig{
            vocabSize, nbAttentionLayers + nbRnnLayers, nbAttentionLayers, nbRnnLayers, nbHeads, hiddenSize, dtype};

        mDecodingState = std::make_unique<tensorrt_llm::runtime::decoder::DecoderState>();
        mDecodingState->setup(
            batchSize, beamWidth, maxSeqLen, 0, maxSeqLen, dtype, modelConfig, worldConfig, *mBufferManager);

        auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
        auto const jointOutputIdsShape = ITensor::makeShape({batchSize, beamWidth, maxSeqLen});

        mTargetOut = mBufferManager->gpu(jointOutputIdsShape, nvTokenIdType);
        mBufferManager->setZero(*mTargetOut);
    }

    // clang-format off

    // hardcode the input data for the output_len = 10 case
    // this should not cause any beam swapping from the CBAs, just reorder the beams
    void hardcodeBuffersLen10()
    {
        auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
        auto constexpr nvSizeType = TRTDataType<SizeType32>::value;
        auto constexpr nvFloatType = TRTDataType<float>::value;

        auto const decodingInput = mDecodingState->getJointDecodingInput();
        auto const decodingOutput = mDecodingState->getJointDecodingOutput();

        std::vector<SizeType32> len = {3, 3, 3, 3, 3};
        TensorPtr inputLengths{ITensor::slice(constPointerCast(decodingInput.lengths), 0, 1)};
        mBufferManager->copy(len.data(),*inputLengths);

        std::vector<SizeType32> eid = {0};
        TensorPtr endIds{ITensor::slice(constPointerCast(decodingInput.endIds), 0, 1)};
        mBufferManager->copy(eid.data(),*endIds);

        std::vector<std::vector<float>> logProbs =
        {
            {-2.96689, -1.63675, -2.31329, -0.0377979, -2.2442, -1.57552, -0.310524, -0.696636, -2.41985},
            {-2.96689, -1.63675, -2.31329, -0.0377979, -1.31451, -2.63339, -0.534199, -0.493615, -2.61479},
            {-2.96689, -1.63675, -2.31329, -0.0377979, -2.2442, -1.57552, -0.310524, -3.11851, -1.01671},
            {-2.96689, -1.63675, -2.31329, -0.0377979, -1.31451, -2.63339, -0.534199, 0, 0},
            {-2.96689, -1.63675, -2.31329, -0.0377979, -2.2442, -1.57552, -0.310524, -0.696636, -3.62298}
        };
        for (SizeType32 it = 0; it < logProbs.size(); it++){
            fillTensorAtIndex(decodingOutput.logProbs, it, logProbs[it], true, mBufferManager);
        }

        std::vector<std::vector<float>> logProbsTiled =
        {
            {-2.70907, -2.96689, -3.27157, -3.37314, -3.50595},
            {-1.84733, -1.8942, -1.63675, -1.9567, -1.47513},
            {-0.305059, -0.765237, -2.31329, -2.37162, -2.48475},
            {-1.97517, -0.0377979, -2.0169, -2.42439, -2.27471},
            {-1.31451, -2.2442, -1.5831, -2.44732, -2.02409},
            {-1.57552, -2.63339, -2.11286, -2.57304, -3.85214},
            {-0.310524, -0.534199, -0.74379, -2.86232, -1.72914},
            {-0.696636, -0.493615, -0.237725, -3.07164, -3.11851},
            {-2.41985, -2.61479, -1.01671, -3.62298, -1.26586},
            {-0.844337, -0.922832, -0.427682, -0.419985, -1.85996}
        };
        TensorPtr logProbsTiledView = ITensor::view(decodingOutput.logProbsTiled,ITensor::makeShape({maxSeqLen*batchSize, beamWidth}));
        for (SizeType32 it = 0; it < logProbsTiled.size(); it++){
            auto logProbsSlice = ITensor::slice(logProbsTiledView, it+3,1);
            mBufferManager->copy(logProbsTiled[it].data(),*logProbsSlice);
        }

        std::vector<SizeType32> outputLenghts = {13, 13, 13, 13, 13};
        mBufferManager->copy(outputLenghts.data(),*decodingOutput.lengths);

        std::vector<float> cumLogProbs = {-15.0458, -15.4681, -15.8323, -15.8424, -16.0614};
        mBufferManager->copy(cumLogProbs.data(),*decodingOutput.cumLogProbs);

        std::vector<std::vector<TokenIdType>> outputIdsCBA =
        {
            {1, 864, 304, 367, 263, 760, 310, 278, 3815, 29973},
            {1, 864, 304, 367, 263, 760, 310, 1749, 3815, 29973}
        };
        for(SizeType32 it = 0; it < outputIdsCBA.size(); it++)
        {
            fillTensorAtIndex(decodingOutput.beamHypotheses.outputIdsCBA, it, outputIdsCBA[it], true, mBufferManager);
        }

        std::vector<std::vector<float>> logProbsCBA =
        {
            {0, 0, 0, -2.96689, -1.63675, -2.31329, -0.0377979, -1.31451, -2.63339, -0.534199, -2.19674},
            {0, 0, 0, -2.96689, -1.63675, -2.31329, -0.0377979, -2.2442, -1.57552, -0.310524, -2.81382,}
        };
        for(SizeType32 it = 0; it < logProbsCBA.size(); it++)
        {
            fillTensorAtIndex(decodingOutput.beamHypotheses.logProbsCBA, it, logProbsCBA[it], true, mBufferManager);
        }

        std::vector<SizeType32> sequenceLengthsCBA = {10, 10, 0, 0, 0, 0, 0, 0, 0, 0};
        mBufferManager->copy(sequenceLengthsCBA.data(), *decodingOutput.beamHypotheses.sequenceLengthsCBA);

        std::vector<float> cumLogProbsCBA = {-13.6336, -13.8988, 0, 0, 0, 0, 0, 0, 0, 0};
        mBufferManager->copy(cumLogProbsCBA.data(), *decodingOutput.beamHypotheses.cumLogProbsCBA);

        std::vector<float> normedScoresCBA = {-1.7042, -1.73735, 0, 0, 0, 0, 0, 0, 0, 0};
        mBufferManager->copy(normedScoresCBA.data(), *decodingOutput.beamHypotheses.normedScoresCBA);

        std::vector<SizeType32> numBeamsCBA = {2};
        mBufferManager->copy(numBeamsCBA.data(), *decodingOutput.beamHypotheses.numBeamsCBA);

        std::vector<float> minNormedScoresCBA = {-1.73735};
        mBufferManager->copy(minNormedScoresCBA.data(), *decodingOutput.beamHypotheses.minNormedScoresCBA);

        std::vector<SizeType32> batchDones = {0};
        mBufferManager->copy(batchDones.data(), *decodingOutput.beamHypotheses.batchDones);

        std::vector<uint8_t> finishReasons = {4, 4, 4, 4, 4};
        mBufferManager->copy(finishReasons.data(), *decodingOutput.finishReasons);

        std::vector<std::vector<TokenIdType>> ids =
        {
            {1, 864, 304, 1073, 825, 1048, 278, 278, 3815, 29973, 13, 4806, 526},
            {1, 864, 304, 367, 920, 304, 310, 1749, 3815, 29973, 13, 4806, 526},
            {1, 864, 304, 679, 263, 760, 679, 263, 29973, 13, 310, 526, 502},
            {1, 864, 304, 1207, 901, 278, 1749, 445, 3889, 393, 591, 13443, 276},
            {1, 864, 304, 1074, 263, 29973, 1207, 263, 2446, 12623, 1334, 29915, 30010}
        };
        for(SizeType32 it = 0; it < ids.size(); it++)
        {
            fillTensorAtIndex(decodingOutput.ids, it, ids[it], true, mBufferManager);
        }

        std::vector<std::vector<SizeType32>> parentIds =
        {
            {0, 0, 0, 0, 0, 3, 0, 1, 1, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 1, 2, 1, 0, 1, 1, 1, 1},
            {0, 0, 0, 0, 1, 2, 1, 4, 3, 2, 4, 4, 3},
            {0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 0, 0, 4},
            {0, 0, 0, 0, 3, 3, 1, 2, 0, 4, 0, 3, 0}
        };
        for(SizeType32 it = 0; it < parentIds.size(); it++)
        {
            fillTensorAtIndex(decodingOutput.parentIds, it, parentIds[it], true, mBufferManager);
        }

        std::vector<std::vector<TokenIdType>> targetOutput =
    {
            {1, 864, 304, 367, 263, 760, 310, 1749, 3815, 29973, 13, 4806, 526},
            {1, 864, 304, 367, 263, 760, 310, 278, 3815, 29973, 13, 4806, 526},
            {1, 864, 304, 367, 263, 760, 310, 1749, 3815, 29973, 13, 13443, 502},
            {1, 864, 304, 367, 263, 760, 310, 1749, 3815, 29973, 591, 29915, 276},
            {1, 864, 304, 367, 263, 760, 310, 1749, 3815, 29973, 13, 4806, 30010}
        };
        for(SizeType32 it = 0; it < targetOutput.size(); it++)
        {
            fillTensorAtIndex(mTargetOut, it, targetOutput[it], true, mBufferManager);
        }
    }

    // this case has the output_len = 8, and tests that the beams from the CBAs are correctly swapped.
    void hardcodeBuffersLen8()
    {
        auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
        auto constexpr nvSizeType = TRTDataType<SizeType32>::value;
        auto constexpr nvFloatType = TRTDataType<float>::value;

        auto const decodingInput = mDecodingState->getJointDecodingInput();
        auto const decodingOutput = mDecodingState->getJointDecodingOutput();

        std::vector<SizeType32> len = {3, 3, 3, 3, 3};
        TensorPtr inputLengths{ITensor::slice(constPointerCast(decodingInput.lengths), 0, 1)};
        mBufferManager->copy(len.data(),*inputLengths);

        std::vector<SizeType32> eid = {0};
        TensorPtr endIds{ITensor::slice(constPointerCast(decodingInput.endIds), 0, 1)};
        mBufferManager->copy(eid.data(),*endIds);

        std::vector<std::vector<float> >logProbs =
        {
            {-2.96689, -1.63675, -2.31329, -0.0377979, -2.2442, -1.57552, -0.310524},
            {-2.96689, -1.63675, -2.31329, -0.0377979, -1.31451, -2.63339, -0.534199},
            {-2.96689, -1.63675, -2.31329, -0.0377979, -2.44732, -2.11286, -0.74379},
            {-2.96689, -1.63675, -2.31329, -0.0377979, -1.31451, -2.63339, -2.86232},
            {-2.96689, -1.63675, -2.31329, -0.0377979, -1.31451, -3.85214, -1.72914}
        };
        for (SizeType32 it = 0; it < logProbs.size(); it++){
            fillTensorAtIndex(decodingOutput.logProbs, it, logProbs[it], true, mBufferManager);
        }

        std::vector<std::vector<float>> logProbsTiled =
        {
            {-2.70907, -2.96689, -3.27157, -3.37314, -3.50595},
            {-1.84733, -1.8942, -1.63675, -1.9567, -1.47513},
            {-0.305059, -0.765237, -2.31329, -2.37162, -2.48475},
            {-1.97517, -0.0377979, -2.0169, -2.42439, -2.27471},
            {-1.31451, -2.2442, -1.5831, -2.44732, -2.02409},
            {-1.57552, -2.63339, -2.11286, -2.57304, -3.85214},
            {-0.310524, -0.534199, -0.74379, -2.86232, -1.72914},
            {-0.696636, -0.493615, -0.237725, -3.07164, -3.11851}
        };
        TensorPtr logProbsTiledView = ITensor::view(decodingOutput.logProbsTiled,ITensor::makeShape({maxSeqLen*batchSize, beamWidth}));
        for (SizeType32 it = 0; it < logProbsTiled.size(); it++){
            auto logProbsSlice = ITensor::slice(logProbsTiledView, it+3,1);
            mBufferManager->copy(logProbsTiled[it].data(),*logProbsSlice);
        }
        std::vector<SizeType32> outputLenghts = {11, 11, 11, 11, 11};
        mBufferManager->copy(outputLenghts.data(),*decodingOutput.lengths);

        std::vector<float> cumLogProbs = {-11.7816, -11.9304, -14.0883, -14.1566, -14.2035};
        mBufferManager->copy(cumLogProbs.data(),*decodingOutput.cumLogProbs);

        std::vector<std::vector<TokenIdType>> outputIdsCBA =
        {
            {1, 864, 304, 367, 263, 760, 310, 278, 3815, 29973},
            {1, 864, 304, 367, 263, 760, 310, 1749, 3815, 29973}
        };
        for(SizeType32 it = 0; it < outputIdsCBA.size(); it++)
        {
            fillTensorAtIndex(decodingOutput.beamHypotheses.outputIdsCBA, it, outputIdsCBA[it], true, mBufferManager);
        }

        std::vector<std::vector<float>> logProbsCBA =
        {
            {0, 0, 0, -2.96689, -1.63675, -2.31329, -0.0377979, -1.31451, -2.63339, -0.534199, -2.19674},
            {0, 0, 0, -2.96689, -1.63675, -2.31329, -0.0377979, -2.2442, -1.57552, -0.310524, -2.81382,}
        };
        for(SizeType32 it = 0; it < logProbsCBA.size(); it++)
        {
            fillTensorAtIndex(decodingOutput.beamHypotheses.logProbsCBA, it, logProbsCBA[it], true, mBufferManager);
        }

        std::vector<SizeType32> sequenceLengthsCBA  = {10, 10, 0, 0, 0, 0, 0, 0, 0, 0};
        mBufferManager->copy(sequenceLengthsCBA.data(), *decodingOutput.beamHypotheses.sequenceLengthsCBA);

        std::vector<float> cumLogProbsCBA = {-13.6336, -13.8988, 0, 0, 0, 0, 0, 0, 0, 0};
        mBufferManager->copy(cumLogProbsCBA.data(), *decodingOutput.beamHypotheses.cumLogProbsCBA);

        std::vector<float> normedScoresCBA = {-1.7042, -1.73735, 0, 0, 0, 0, 0, 0, 0, 0};
        mBufferManager->copy(normedScoresCBA.data(), *decodingOutput.beamHypotheses.normedScoresCBA);

        std::vector<SizeType32> numBeamsCBA = {2};
        mBufferManager->copy(numBeamsCBA.data(), *decodingOutput.beamHypotheses.numBeamsCBA);

        std::vector<float> minNormedScoresCBA = {-1.73735};
        mBufferManager->copy(minNormedScoresCBA.data(), *decodingOutput.beamHypotheses.minNormedScoresCBA);

        std::vector<SizeType32> batchDones = {0};
        mBufferManager->copy(batchDones.data(), *decodingOutput.beamHypotheses.batchDones);

        std::vector<uint8_t> finishReasons = {4, 4, 4, 4, 4};
        mBufferManager->copy(finishReasons.data(), *decodingOutput.finishReasons);

        std::vector<std::vector<TokenIdType>> ids =
        {
            {1, 864, 304, 1073, 825, 1048, 278, 278, 3815, 29973, 13},
            {1, 864, 304, 367, 920, 304, 310, 1749, 3815, 29973, 13},
            {1, 864, 304, 679, 263, 760, 679, 263, 29973, 13, 310},
            {1, 864, 304, 1207, 901, 278, 1749, 445, 3889, 393, 591},
            {1, 864, 304, 1074, 263, 29973, 1207, 263, 2446, 12623, 1334}
        };
        for(SizeType32 it = 0; it < ids.size(); it++)
        {
            fillTensorAtIndex(decodingOutput.ids, it, ids[it], true, mBufferManager);
        }

        std::vector<std::vector<SizeType32>> parentIds =
        {
            {0, 0, 0, 0, 0, 3, 0, 1, 1, 0, 0},
            {0, 0, 0, 0, 0, 1, 2, 1, 0, 1, 1},
            {0, 0, 0, 0, 1, 2, 1, 4, 3, 2, 4},
            {0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 0},
            {0, 0, 0, 0, 3, 3, 1, 2, 0, 4, 0}
        };
        for(SizeType32 it = 0; it < parentIds.size(); it++)
        {
            fillTensorAtIndex(decodingOutput.parentIds, it, parentIds[it], true, mBufferManager);
        }

        std::vector<std::vector<TokenIdType>> targetOutput =
        {
            {1, 864, 304, 367, 263, 760, 310, 1749, 3815, 29973, 13},
            {1, 864, 304, 367, 263, 760, 310, 278, 3815, 29973, 13},
            {1, 864, 304, 367, 263, 760, 310, 278, 3815, 29973, 0},
            {1, 864, 304, 367, 263, 760, 310, 1749, 3815, 29973, 0},
            {1, 864, 304, 367, 263, 760, 310, 278, 2446, 12623, 310}
        };
        for(SizeType32 it = 0; it < targetOutput.size(); it++)
        {
            fillTensorAtIndex(mTargetOut, it, targetOutput[it], true, mBufferManager);
        }
    }

    // clang-format on

    bool checkResult()
    {
        auto const reference = this->mBufferManager->copyFrom(*mTargetOut, tensorrt_llm::runtime::MemoryType::kCPU);
        auto referencePtr = bufferCast<TokenIdType>(*reference);

        auto const real = this->mBufferManager->copyFrom(
            *mDecodingState->getGatheredIds(), tensorrt_llm::runtime::MemoryType::kCPU);
        auto realPtr = bufferCast<TokenIdType>(*real);

        bool allEqual = true;
        for (SizeType32 iAssert = 0; iAssert < batchSize * beamWidth * maxSeqLen; iAssert++)
        {
            if (referencePtr[iAssert] != realPtr[iAssert])
            {
                TLLM_LOG_ERROR("Mismatch input value. Position of inputs: %d, expected value: %d, output value: %d",
                    iAssert, referencePtr[iAssert], realPtr[iAssert]);
                allEqual = false;
            }
        }
        return allEqual;
    }
};

TEST_F(TestGatherTree, GatherTreeNoSwap)
{
    createBuffers();
    hardcodeBuffersLen10();
    cudaDeviceSynchronize();
    kernels::gatherTree(
        mDecodingState->getJointDecodingOutput(), mDecodingState->getJointDecodingInput(), mSamplingConfig, *mStream);
    cudaDeviceSynchronize();

    EXPECT_TRUE(checkResult());
}

TEST_F(TestGatherTree, GatherTreeWithSwap)
{
    createBuffers();
    hardcodeBuffersLen8();
    cudaDeviceSynchronize();
    kernels::gatherTree(
        mDecodingState->getJointDecodingOutput(), mDecodingState->getJointDecodingInput(), mSamplingConfig, *mStream);
    cudaDeviceSynchronize();

    EXPECT_TRUE(checkResult());
}

enum AcceptKernelMode
{
    BY_IDS,
    BY_LOGITS,
    BY_IDS_WITH_PATH
};

struct DecodingKernelTestParam
{
    SizeType32 mBatchSize{128};
    SizeType32 mMaxBatchSize{2 * mBatchSize};
    SizeType32 mBeamWidth{1};
    SizeType32 mMaxSeqLen{16};
    SizeType32 mVocabSize{32};
    SizeType32 mMaxDraftTokens{8};
    SizeType32 mMaxNumHeads{0};
    SizeType32 mMaxDraftSeqPerStep{1};
    AcceptKernelMode mAcceptMode{AcceptKernelMode::BY_IDS};

    DecodingKernelTestParam& setBatchSize(SizeType32 bs)
    {
        mBatchSize = bs;
        mMaxBatchSize = 2 * mBatchSize;
        return *this;
    }

    DecodingKernelTestParam& setVocabSize(SizeType32 vs)
    {
        mVocabSize = vs;
        return *this;
    }

    DecodingKernelTestParam& setMaxSeqLen(SizeType32 msl)
    {
        mMaxSeqLen = msl;
        return *this;
    }

    DecodingKernelTestParam& setMaxDraftTokens(SizeType32 dt)
    {
        mMaxDraftTokens = dt;
        return *this;
    }

    DecodingKernelTestParam& setMaxNumHeads(SizeType32 mnh)
    {
        mMaxNumHeads = mnh;
        return *this;
    }

    DecodingKernelTestParam& setMaxDraftSeqPerStep(SizeType32 tps)
    {
        mMaxDraftSeqPerStep = tps;
        return *this;
    }

    DecodingKernelTestParam& setAcceptMode(AcceptKernelMode const& mode)
    {
        mAcceptMode = mode;
        return *this;
    }
};

template <typename T>
class DecodingKernelsTest : public testing::Test
{
public:
    using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;

    void SetUp() override
    {
        mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
        mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);
    }

    void TearDown() override {}

    void createBuffers()
    {
        auto const dataType = TRTDataType<T>::value;
        auto const ptrType = TRTDataType<T*>::value;

        mDraftTokens = mBufferManager->pinnedPool(
            ITensor::makeShape({mMaxBatchSize, mMaxDraftSeqlen}), nvinfer1::DataType::kINT32);
        mTargetTokens = mBufferManager->pinnedPool(
            ITensor::makeShape({mMaxBatchSize, mMaxTargetSeqlen}), nvinfer1::DataType::kINT32);
        mOutputTokens
            = mBufferManager->pinnedPool(ITensor::makeShape({mMaxBatchSize, mMaxSeqLen}), nvinfer1::DataType::kINT32);
        mNumsDraftTokens = mBufferManager->pinnedPool(
            ITensor::makeShape({mMaxBatchSize, mMaxDraftSeqPerStep}), nvinfer1::DataType::kINT32);
        mSequenceLengths = mBufferManager->pinnedPool(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        mAcceptedLengths = mBufferManager->pinnedPool(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        mContextLengths = mBufferManager->pinnedPool(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        mFinishedSteps = mBufferManager->pinnedPool(ITensor::makeShape({mMaxDraftTokens + 1, mMaxBatchSize}),
            TRTDataType<tk::FinishedState::UnderlyingType>::value);
        mFinishedFinal = mBufferManager->pinnedPool(
            ITensor::makeShape({mMaxBatchSize}), TRTDataType<tk::FinishedState::UnderlyingType>::value);
        mFinishedSum = mBufferManager->pinnedPool(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);

        mPaths = mBufferManager->pinnedPool(
            ITensor::makeShape({mMaxBatchSize, mMaxDraftSeqPerStep, mMaxDraftTokens}), nvinfer1::DataType::kINT32);
        mEndIds = mBufferManager->pinnedPool(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);

        mBatchSlots = mBufferManager->pinnedPool(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        auto batchSlotsRange = BufferRange<SizeType32>(*mBatchSlots);
        std::iota(batchSlotsRange.begin(), batchSlotsRange.end(), 0);

        mCurandStates = mBufferManager->gpu(
            ITensor::makeShape({mMaxBatchSize, sizeof(curandState_t)}), nvinfer1::DataType::kINT8);

        mAcceptedLen.resize(mMaxBatchSize);
        mOutputLen.resize(mMaxBatchSize);
        mAcceptedFinished.resize(mMaxBatchSize, tk::FinishedState::empty());

        // Buffers only for Logits comparison
        if (mAcceptMode == AcceptKernelMode::BY_LOGITS)
        {
            mDraftLogits = mBufferManager->pinnedPool(
                ITensor::makeShape({mMaxBatchSize, mMaxTotalDraftTokens, mVocabSize}), dataType);
            mTargetLogits = mBufferManager->pinnedPool(
                ITensor::makeShape({mMaxBatchSize, mMaxTotalDraftTokens, mVocabSize}), dataType);
            mTargetLogitsPtrs = mBufferManager->pinnedPool(ITensor::makeShape({mMaxBatchSize}), ptrType);
            mRefTargetLogits = mBufferManager->pinnedPool(
                ITensor::makeShape({mMaxBatchSize, mMaxTotalDraftTokens, mVocabSize}), dataType);

            mDraftProbs = mBufferManager->pinnedPool(
                ITensor::makeShape({mMaxBatchSize, mMaxTotalDraftTokens, mVocabSize}), dataType);
            mTargetProbs = mBufferManager->pinnedPool(
                ITensor::makeShape({mMaxBatchSize, mMaxTotalDraftTokens, mVocabSize}), dataType);
        }

        if (mAcceptMode == AcceptKernelMode::BY_IDS_WITH_PATH)
        {
            mMedusaLogitsPtrs = mBufferManager->pinnedPool(
                ITensor::makeShape({mMaxBatchSize, mMaxDraftSeqPerStep, mMaxNumHeads}), ptrType);
            mMedusaInputLogitsPtrs
                = mBufferManager->pinnedPool(ITensor::makeShape({mMaxBatchSize, mMaxNumHeads}), ptrType);
            mTokensPerStep
                = mBufferManager->pinnedPool(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
            mBestPaths = mBufferManager->pinnedPool(ITensor::makeShape({mMaxBatchSize}), nvinfer1::DataType::kINT32);
        }
    }

    void initData(SizeType32 seed)
    {
        std::mt19937 generator(seed);
        std::uniform_int_distribution<SizeType32> contextLenDistr(0, std::max(mMaxSeqLen - mMaxTotalDraftTokens, 0));
        std::uniform_int_distribution<SizeType32> numTotalDraftTokensDistr(1, mMaxTotalDraftTokens);
        std::uniform_int_distribution<SizeType32> numDraftTokensDistr(0, mMaxDraftTokens);
        std::uniform_int_distribution<SizeType32> vocabDistr(1, mVocabSize - 1);
        std::uniform_real_distribution<float> acceptTokenDistr(0.f, 1.f);

        trk::invokeFill(*mPaths, int32_t{-1}, *mStream);
        trk::invokeFill(*mFinishedFinal, tk::FinishedState::UnderlyingType{0}, *mStream);

        auto sequenceLengthsPtr = BufferRange<SizeType32>(*mSequenceLengths);
        auto contextLengthsPtr = BufferRange<SizeType32>(*mContextLengths);
        auto numsDraftTokensPtr = BufferRange<SizeType32>(*mNumsDraftTokens);
        auto draftTokensPtr = BufferRange<SizeType32>(*mDraftTokens);
        auto targetTokensPtr = BufferRange<SizeType32>(*mTargetTokens);
        auto finishedStepsPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedSteps));
        auto pathsPtr = BufferRange<SizeType32>(*mPaths);
        auto endIdsPtr = BufferRange<SizeType32>(*mEndIds);

        auto batchSlotsPtr = bufferCast<SizeType32>(*mBatchSlots);

        tk::invokeCurandInitialize(reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStates)), batchSlotsPtr,
            mMaxBatchSize, seed, this->mStream->get());

        auto generateAvoidingValues = [&vocabDistr, &generator](std::uniform_int_distribution<SizeType32>& distr,
                                          std::unordered_set<SizeType32> const& tokensToAvoid, SizeType32 maxTries = -1,
                                          SizeType32 defaultValue = -1)
        {
            // Avoid generating endId.
            auto token = distr(generator);
            SizeType32 tries = 0;
            while (tokensToAvoid.count(token) != 0 && ((maxTries >= 0 && tries < maxTries) || maxTries < 0))
            {
                token = distr(generator);
                tries++;
            }
            if (tries == maxTries)
            {
                token = defaultValue;
            }
            return token;
        };

        // Init batch slots
        for (SizeType32 bi = 0; bi < mBatchSize; ++bi)
        {
            batchSlotsPtr[bi] = 2 * bi;
        }

        // Init end ids
        for (SizeType32 bi = 0; bi < mMaxBatchSize; ++bi)
        {
            endIdsPtr[bi] = generateAvoidingValues(vocabDistr, {mPadId});
            TLLM_LOG_DEBUG("bi %d endIdsPtr[bi] %d", bi, endIdsPtr[bi]);

            // Randomly init context len for target and draft
            contextLengthsPtr[bi] = contextLenDistr(generator);
        }

        std::fill(draftTokensPtr.begin(), draftTokensPtr.begin() + mMaxBatchSize * mMaxDraftSeqlen, mPadId);
        std::fill(targetTokensPtr.begin(), targetTokensPtr.begin() + mMaxBatchSize * mMaxTargetSeqlen, mPadId);
        std::fill(pathsPtr.begin(), pathsPtr.begin() + mMaxBatchSize * mMaxDraftSeqPerStep * mMaxDraftTokens, -1);

        // Generate paths
        for (SizeType32 bi = 0; bi < mMaxBatchSize; ++bi)
        {
            auto const numTotalDraftTokens = std::min(mMaxDraftTokens, numTotalDraftTokensDistr(generator));
            std::uniform_int_distribution<SizeType32> pathIdDistr(0, numTotalDraftTokens);
            for (SizeType32 pi = 0; pi < mMaxDraftSeqPerStep; ++pi)
            {
                std::unordered_set<SizeType32> pathIds;
                auto const numDraftTokensAtStep = numDraftTokensDistr(generator);
                numsDraftTokensPtr[bi * mMaxDraftSeqPerStep + pi] = numDraftTokensAtStep;

                for (SizeType32 ti = 0; ti < numDraftTokensAtStep; ++ti)
                {
                    auto const pathIdx = tc::flat_index3(bi, pi, ti, mMaxDraftSeqPerStep, mMaxDraftTokens);
                    // Single linear path for BY_IDS and BY_LOGITS modes
                    auto const pathId = mAcceptMode == AcceptKernelMode::BY_IDS_WITH_PATH && ti != 0
                        ? generateAvoidingValues(pathIdDistr, pathIds, mMaxDraftTokens * 5, -1)
                        : ti;
                    pathsPtr[pathIdx] = pathId;
                    pathIds.insert(pathId);
                }
                if (bi == 2)
                {
                    TLLM_LOG_DEBUG("bi %d pi %d numsDraftTokensPtr[bi] %d", bi, pi, numDraftTokensAtStep);
                }
            }
        }

        for (SizeType32 ti = 0; ti < mMaxDraftSeqPerStep; ++ti)
        {
            std::vector<SizeType32> targetPredictedLen(mMaxBatchSize);
            std::vector<SizeType32> targetAcceptedLen(mMaxBatchSize);

            // Init number of draft tokens
            for (SizeType32 bi = 0; bi < mMaxBatchSize; ++bi)
            {
                // It can be shorter than num of draft tokens due to the EOS generation
                std::uniform_int_distribution<SizeType32> realDraftTokensDistr(
                    0, numsDraftTokensPtr[bi * mMaxDraftSeqPerStep + ti]);
                targetPredictedLen[bi] = realDraftTokensDistr(generator);
                // Accept ~ half of the tokens on avergae
                std::poisson_distribution<SizeType32> targetAcceptedDistr(targetPredictedLen[bi] / 2);
                targetAcceptedLen[bi] = std::min(targetAcceptedDistr(generator), targetPredictedLen[bi]);
                if (bi == 2)
                {
                    TLLM_LOG_DEBUG("bi %d ti %d targetPredictedLen[bi] %d targetAcceptedLen[bi] %d", bi, ti,
                        targetPredictedLen[bi], targetAcceptedLen[bi]);
                }
            }

            // Fill draft tokens
            for (SizeType32 bi = 0; bi < mMaxBatchSize; ++bi)
            {
                for (SizeType32 si = 0; si < numsDraftTokensPtr[bi * mMaxDraftSeqPerStep + ti]; ++si)
                {
                    auto const pathIdx = tc::flat_index3(bi, ti, si, mMaxDraftSeqPerStep, mMaxDraftTokens);
                    if (pathsPtr[pathIdx] == -1)
                    {
                        continue;
                    }
                    auto const draftTokenIdx = bi * mMaxDraftSeqlen + pathsPtr[pathIdx];
                    // Avoid generating endId. We'll insert in manually later if needed.
                    draftTokensPtr[draftTokenIdx] = generateAvoidingValues(vocabDistr, {mPadId, endIdsPtr[bi]});
                    if (bi == 2)
                    {
                        TLLM_LOG_DEBUG("bi %d ti %d si %d pathId %d draftToken %d", bi, ti, si, pathsPtr[pathIdx],
                            draftTokensPtr[draftTokenIdx]);
                    }
                }
            }

            for (SizeType32 bi = 0; bi < mMaxBatchSize; ++bi)
            {
                sequenceLengthsPtr[bi] = contextLengthsPtr[bi] + targetPredictedLen[bi];

                // Initialize finished states
                for (int di = 0; di < numsDraftTokensPtr[bi * mMaxDraftSeqPerStep + ti]; ++di)
                {
                    finishedStepsPtr[di * mMaxBatchSize + bi]
                        = (di < targetPredictedLen[bi]) ? tk::FinishedState::empty() : tk::FinishedState::finished();
                }

                // Init helper vectors
                mAcceptedLen[bi] = contextLengthsPtr[bi] + std::max(targetAcceptedLen[bi], 0);
                mOutputLen[bi] = std::min(sequenceLengthsPtr[bi], std::min(mAcceptedLen[bi] + 1, mMaxSeqLen));
                mAcceptedFinished[bi] = finishedStepsPtr[std::max(targetAcceptedLen[bi], 0) * mMaxBatchSize + bi];
                if (bi == 2)
                {
                    TLLM_LOG_DEBUG(
                        "bi %d ti %d contextLengthsPtr[bi] %d sequenceLengthsPtr[bi] %d mAcceptedLen[bi] %d "
                        "mOutputLen[bi] "
                        "%d",
                        bi, ti, contextLengthsPtr[bi], sequenceLengthsPtr[bi], mAcceptedLen[bi], mOutputLen[bi]);
                }
            }

            // Fill token arrays
            for (SizeType32 bi = 0; bi < mMaxBatchSize; ++bi)
            {
                // Draft:  [d0, d1, d2, ... for numsDraftTokensPtr[bi] ... , dK,
                //         padId, padId, .. to mMaxDraftSeqlen]
                // Target: [padId, padId, ... for contextLengthsPtr[bi] ... padId,
                //         d0, d1, d2, ... for targetAcceptedLen[bi],
                //         ti (!= di), ti+1 (!= di+1), ... for (targetPredictedLen[bi] - targetAcceptedLen[bi]),
                //         EOS, EOS, EOS, ... for (numsDraftTokensPtr[bi] - targetPredictedLen[bi])
                //         padId, padId, .. to mMaxSeqLen]
                auto numDraftTokens = numsDraftTokensPtr[bi * mMaxDraftSeqPerStep + ti];
                for (SizeType32 si = 0; si < numDraftTokens; ++si)
                {
                    auto const curPathIdx = tc::flat_index3(bi, ti, si, mMaxDraftSeqPerStep, mMaxDraftTokens);
                    auto const nextPathIdx = si + 1 < numDraftTokens
                        ? tc::flat_index3(bi, ti, si + 1, mMaxDraftSeqPerStep, mMaxDraftTokens)
                        : -1;
                    auto const curPathId = pathsPtr[curPathIdx];
                    auto nextPathId = curPathId;
                    if (mAcceptMode == AcceptKernelMode::BY_IDS_WITH_PATH)
                    {
                        nextPathId = nextPathIdx > -1 ? pathsPtr[nextPathIdx] : -1;
                    }

                    if (curPathId == -1)
                    {
                        continue;
                    }
                    auto const contextLen
                        = mAcceptMode == AcceptKernelMode::BY_IDS_WITH_PATH ? 0 : contextLengthsPtr[bi];
                    auto const draftTokenIdx = bi * mMaxDraftSeqlen + nextPathId;
                    auto const targetTokenIdx = bi * mMaxTargetSeqlen + contextLen + curPathId;
                    auto targetToken = mPadId;
                    if (0 <= si && si < targetAcceptedLen[bi] && nextPathId != -1)
                    {
                        // Use draft token up to the accepted len
                        targetToken = draftTokensPtr[draftTokenIdx];
                    }
                    else if (0 <= si && si < targetPredictedLen[bi])
                    {
                        // Do not use draft token token up to the generated len
                        std::unordered_set<SizeType32> avoidValues = {mPadId, endIdsPtr[bi]};
                        if (nextPathId != -1)
                        {
                            avoidValues.insert(draftTokensPtr[draftTokenIdx]);
                        }
                        targetToken = generateAvoidingValues(vocabDistr, avoidValues);
                    }
                    else if (targetPredictedLen[bi] <= si && si < numsDraftTokensPtr[bi])
                    {
                        // Fill with EOS from generated len to the draft len
                        targetToken = endIdsPtr[bi];
                    }
                    targetTokensPtr[targetTokenIdx] = targetToken;
                    if (bi == 2)
                    {
                        TLLM_LOG_DEBUG(
                            "bi %d ti %d si %d pathId %d targetToken %d", bi, ti, si, curPathId, targetToken);
                    }
                }
            }
        }

        if (mAcceptMode == AcceptKernelMode::BY_LOGITS)
        {
            initDataAndReferenceAcceptByLogits();
        }

        if (mAcceptMode == AcceptKernelMode::BY_IDS_WITH_PATH)
        {
            initDataAndReferenceAcceptByIdsWithPaths();
        }
        mSequenceLengthsCopy = mBufferManager->copyFrom(*mSequenceLengths, MemoryType::kCPU);
    }

    void initDataAndReferenceAcceptByIdsWithPaths()
    {
        auto const dataType = TRTDataType<T>::value;
        auto const ptrType = TRTDataType<T*>::value;

        auto pathsPtr = BufferRange<SizeType32>(*mPaths);
        auto endIdsPtr = BufferRange<SizeType32>(*mEndIds);
        auto contextLengthsPtr = BufferRange<SizeType32>(*mContextLengths);
        auto draftTokensPtr = BufferRange<SizeType32>(*mDraftTokens);
        auto targetTokensPtr = BufferRange<SizeType32>(*mTargetTokens);
        auto medusaInputLogitsPtr = BufferRange<T*>(*mMedusaInputLogitsPtrs);

        trk::invokeFill(*mMedusaLogitsPtrs, int64_t{0}, *mStream);
        trk::invokeFill(*mTokensPerStep, int32_t{mMaxTotalDraftTokens}, *mStream);
        trk::invokeFill(*mBestPaths, int32_t{-1}, *mStream);

        mAcceptedLen.resize(mMaxBatchSize);
        mAcceptedPathIdx.resize(mMaxBatchSize);
        mRefAcceptedTokens.resize(mMaxBatchSize);
        mFinishedByIdsPaths.resize(mMaxBatchSize);
        mLastTargetIdx.resize(mMaxBatchSize);
        for (SizeType32 bi = 0; bi < mMaxBatchSize; ++bi)
        {
            SizeType32 maxAcceptedLen = -1;
            SizeType32 maxAcceptedPath = -1;
            SizeType32 maxNextTargetTokenIdx = -1;
            bool maxFinished = false;
            std::vector<SizeType32> maxAcceptedTokens;
            for (SizeType32 ti = 0; ti < mMaxDraftSeqPerStep; ++ti)
            {
                std::vector<SizeType32> acceptedTokens;
                SizeType32 curAcceptedLen = mMaxDraftTokens;
                SizeType32 curAcceptedPath = ti;
                bool curFinished = false;

                auto const pathIdx = tc::flat_index3(bi, ti, 0, mMaxDraftSeqPerStep, mMaxDraftTokens);
                auto const pathId = pathsPtr[pathIdx];
                if (pathId == -1)
                {
                    continue;
                }
                auto targetTokenIdx = bi * mMaxTargetSeqlen + pathId;
                auto targetToken = targetTokensPtr[targetTokenIdx];
                auto curNextTargetTokenIdx = pathId;
                for (SizeType32 di = 1; di < mMaxDraftTokens; ++di)
                {
                    auto const pathIdx = tc::flat_index3(bi, ti, di, mMaxDraftSeqPerStep, mMaxDraftTokens);
                    auto const pathId = pathsPtr[pathIdx];
                    if (pathId == -1)
                    {
                        curAcceptedLen = di;
                        curAcceptedPath = ti;
                        curFinished = false;
                        acceptedTokens.push_back(targetToken);
                        break;
                    }
                    auto const draftTokenIdx = bi * mMaxDraftSeqlen + pathId - 1;
                    auto const targetTokenIdx = bi * mMaxTargetSeqlen + pathId;
                    auto const draftToken = draftTokensPtr[draftTokenIdx];
                    bool const hasEnd = targetToken == endIdsPtr[bi];
                    if (!hasEnd)
                    {
                        acceptedTokens.push_back(targetToken);
                    }
                    if (draftToken != targetToken || hasEnd)
                    {
                        auto const curLen = hasEnd ? di - 1 : di;
                        curAcceptedLen = curLen;
                        curAcceptedPath = ti;
                        curFinished = hasEnd;
                        break;
                    }
                    targetToken = targetTokensPtr[targetTokenIdx];
                    curNextTargetTokenIdx = pathId;
                }
                if (curAcceptedLen == mMaxDraftTokens)
                {
                    acceptedTokens.push_back(targetToken);
                }
                if (curAcceptedLen > maxAcceptedLen)
                {
                    maxAcceptedLen = curAcceptedLen;
                    maxAcceptedPath = curAcceptedPath;
                    maxAcceptedTokens = acceptedTokens;
                    maxFinished = curFinished;
                    maxNextTargetTokenIdx = curNextTargetTokenIdx;
                }
            }
            mAcceptedLen[bi] = maxAcceptedLen;
            mAcceptedPathIdx[bi] = maxAcceptedPath;
            mRefAcceptedTokens[bi] = maxAcceptedTokens;
            mFinishedByIdsPaths[bi] = maxFinished;
            mLastTargetIdx[bi] = maxNextTargetTokenIdx;
            for (SizeType32 hi = 0; hi < mMaxNumHeads; ++hi)
            {
                medusaInputLogitsPtr[bi * mMaxNumHeads + hi] = static_cast<T*>(nullptr)
                    + tc::flat_index4(hi, bi, 0, 0, mMaxBatchSize, mMaxDraftSeqPerStep, mVocabSize);
            }
            if (bi == 2)
            {
                TLLM_LOG_DEBUG("bi %d maxAcceptedLen %d maxAcceptedPath %d maxNextTargetTokenIdx %d", bi,
                    maxAcceptedLen, maxAcceptedPath, maxNextTargetTokenIdx);
                std::ostringstream ss;
                for (auto& tk : maxAcceptedTokens)
                {
                    ss << tk << " ";
                }
                TLLM_LOG_DEBUG(ss.str().c_str());
            }
        }
    }

    void initDataAndReferenceAcceptByLogits()
    {
        auto contextLengthsPtr = BufferRange<SizeType32>(*mContextLengths);
        auto numsDraftTokensPtr = BufferRange<SizeType32>(*mNumsDraftTokens);
        auto draftTokensPtr = BufferRange<SizeType32>(*mDraftTokens);
        auto targetTokensPtr = BufferRange<SizeType32>(*mTargetTokens);

        auto draftProbsPtr = BufferRange<T>(*mDraftProbs);
        auto targetProbsPtr = BufferRange<T>(*mTargetProbs);

        auto draftLogitsPtr = BufferRange<T>(*mDraftLogits);
        auto targetLogitsPtr = BufferRange<T>(*mTargetLogits);
        auto targetLogitsPtrsPtr = BufferRange<T*>(*mTargetLogitsPtrs);
        auto refTargetLogitsPtr = BufferRange<T>(*mRefTargetLogits);
        auto batchSlotsPtr = BufferRange<SizeType32>(*mBatchSlots);

        for (SizeType32 bi = 0; bi < mMaxBatchSize; ++bi)
        {
            // Init draft and target logits and probabilities
            for (SizeType32 si = 0; si < numsDraftTokensPtr[bi]; ++si)
            {
                std::vector<float> peakDraftProb(mVocabSize, 0.f);
                std::vector<float> peakTargetProb(mVocabSize, 0.f);

                auto const targetToken = targetTokensPtr[bi * mMaxSeqLen + contextLengthsPtr[bi] + si] % mVocabSize;
                auto const draftToken = draftTokensPtr[bi * mMaxDraftTokens + si] % mVocabSize;

                peakDraftProb[draftToken] = 1.f;
                peakTargetProb[targetToken] = 1.f;

                auto const logitsOffset = bi * mMaxDraftTokens * mVocabSize + si * mVocabSize;
                // Emulate some distribution around target token
                applyGaussianFilter(
                    draftProbsPtr.begin() + logitsOffset, peakDraftProb.data(), peakDraftProb.size(), 1.0f);
                applyGaussianFilter(
                    targetProbsPtr.begin() + logitsOffset, peakTargetProb.data(), peakTargetProb.size(), 1.0f);

                // Probabilities to logits
                probsToLogits(draftProbsPtr.begin() + logitsOffset, draftLogitsPtr.begin() + logitsOffset, mVocabSize);
                probsToLogits(
                    targetProbsPtr.begin() + logitsOffset, targetLogitsPtr.begin() + logitsOffset, mVocabSize);

                // Do softmax conversion back to emulate kernels accuracy
                softmax(draftLogitsPtr.begin() + logitsOffset, draftProbsPtr.begin() + logitsOffset, mVocabSize);
                softmax(targetLogitsPtr.begin() + logitsOffset, targetProbsPtr.begin() + logitsOffset, mVocabSize);
            }
        }

        for (SizeType32 bi = 0; bi < mMaxBatchSize; ++bi)
        {
            for (SizeType32 si = 0; si < mMaxDraftTokens; ++si)
            {
                auto const logitsOffset = bi * mMaxDraftTokens * mVocabSize + si * mVocabSize;
                auto const outputLen = mOutputLen[bi] - contextLengthsPtr[bi];
                auto const acceptedLen = mAcceptedLen[bi] - contextLengthsPtr[bi];
                if (si < acceptedLen)
                {
                    auto logitsStart = targetLogitsPtr.begin() + logitsOffset;
                    std::copy(logitsStart, logitsStart + mVocabSize, refTargetLogitsPtr.begin() + logitsOffset);
                }
                else if (si == acceptedLen)
                {
                    // When token is not accepted, correct probabilities and compute updated logits
                    float sumProb = 1e-6f;
                    for (SizeType32 vi = 0; vi < mVocabSize; ++vi)
                    {
                        auto const correctedProb = std::max(
                            static_cast<float>(targetProbsPtr[logitsOffset + vi] - draftProbsPtr[logitsOffset + vi]),
                            0.f);
                        sumProb += correctedProb;
                    }
                    for (SizeType32 vi = 0; vi < mVocabSize; ++vi)
                    {
                        auto prob = std::max(static_cast<float>(
                                                 targetProbsPtr[logitsOffset + vi] - draftProbsPtr[logitsOffset + vi]),
                                        0.f)
                            / sumProb;
                        if (prob < 1e-8)
                        {
                            prob = 0.f;
                        }
                        refTargetLogitsPtr[logitsOffset + vi] = std::log(prob / (1.f - prob));
                    }
                }
            }
        }
        for (SizeType32 bi = 0; bi < mBatchSize; ++bi)
        {
            targetLogitsPtrsPtr[bi] = targetLogitsPtr.begin() + batchSlotsPtr[bi] * mMaxDraftTokens * mVocabSize;
        }
    }

    void callAcceptByIds()
    {
        // tksp::invokeAcceptDraftTokensByIds(bufferCast<SizeType32>(*mDraftTokens),
        //     bufferCast<SizeType32>(*mTargetTokens), bufferCast<SizeType32>(*mContextLengths),
        //     bufferCast<SizeType32>(*mNumsDraftTokens), bufferCast<SizeType32>(*mSequenceLengths),
        //     reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedSteps)),
        //     reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedFinal)),
        //     bufferCast<SizeType32>(*mFinishedSum), bufferCast<SizeType32>(*mBatchSlots), mBatchSize, mMaxBatchSize,
        //     mBeamWidth, mMaxSeqLen, mMaxDraftTokens, mStream->get());
    }

    void callAcceptByLogits()
    {
        // tksp::acceptDraftTokensByLogits(bufferCast<T>(*mDraftLogits),
        //     reinterpret_cast<T**>(bufferCast<int64_t>(*mTargetLogitsPtrs)), bufferCast<T>(*mDraftProbs),
        //     bufferCast<T>(*mTargetProbs), bufferCast<SizeType32>(*mNumsDraftTokens),
        //     reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedSteps)),
        //     reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStates)),
        //     bufferCast<SizeType32>(*mBatchSlots), mBatchSize, mMaxBatchSize, mBeamWidth, mVocabSize, mVocabSize,
        //     mMaxDraftTokens, false, 0.9f, mStream->get());
    }

    void callAcceptByIdsWithPaths()
    {
        tksp::AcceptDraftTokensByIdsWithPathsParams<T> params;

        params.outputIds = bufferCast<SizeType32>(*mOutputTokens);
        params.draftIds = bufferCast<SizeType32>(*mDraftTokens);
        params.targetIds = bufferCast<SizeType32>(*mTargetTokens);
        params.sequenceLengths = bufferCast<SizeType32>(*mSequenceLengths);
        params.acceptedLengths = bufferCast<SizeType32>(*mAcceptedLengths);
        params.finishedFinal
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedFinal));
        params.batchSlots = bufferCast<SizeType32>(*mBatchSlots);
        params.paths = bufferCast<SizeType32>(*mPaths);
        params.endIds = bufferCast<SizeType32>(*mEndIds);
        params.medusaLogits = reinterpret_cast<T const**>(bufferCast<int64_t>(*mMedusaInputLogitsPtrs));
        params.logitsPtrs = reinterpret_cast<T const**>(bufferCast<int64_t>(*mMedusaLogitsPtrs));
        params.curTokensPerStep = bufferCast<SizeType32>(*mTokensPerStep);
        params.targetTokensPerStep = bufferCast<SizeType32>(*mTokensPerStep);
        params.bestPathIds = bufferCast<SizeType32>(*mBestPaths);
        params.batchSize = mBatchSize;
        params.maxBatchSize = mMaxBatchSize;
        params.vocabSize = mVocabSize;
        params.maxSeqLen = mMaxSeqLen;
        params.maxDraftPathLen = mMaxNumHeads;
        params.maxDecodingTokens = mMaxDraftSeqPerStep;
        params.stream = mStream->get();

        params.checkParams();

        tksp::acceptDraftTokensByIdsWithPaths(params);
    }

    void callTestedKernel()
    {
        switch (mAcceptMode)
        {
        case AcceptKernelMode::BY_IDS: callAcceptByIds(); break;
        case AcceptKernelMode::BY_LOGITS: callAcceptByLogits(); break;
        case AcceptKernelMode::BY_IDS_WITH_PATH: callAcceptByIdsWithPaths(); break;
        default: TLLM_CHECK(false); // Should never be here
        }
    }

    void verifyAcceptByIdsResults(SizeType32 seed)
    {
        auto finishedFinalPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedFinal));
        auto sequenceLengthsPtr = BufferRange<SizeType32>(*mSequenceLengths);
        auto finishedSumPtr = BufferRange<SizeType32>(*mFinishedSum);
        auto batchSlotsPtr = BufferRange<SizeType32>(*mBatchSlots);
        // Verify seqLen for accepted tokens
        for (SizeType32 bi = 0; bi < mBatchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            EXPECT_EQ(mOutputLen[batchSlot], sequenceLengthsPtr[batchSlot]) << " bi " << bi << " seed " << seed;
            EXPECT_EQ(mAcceptedFinished[batchSlot].isFinished(), finishedFinalPtr[batchSlot].isFinished())
                << " bi " << bi << " seed " << seed;
            EXPECT_EQ(mAcceptedFinished[batchSlot].isSkipDecoding(), finishedFinalPtr[batchSlot].isSkipDecoding())
                << " bi " << bi << " seed " << seed;
            EXPECT_EQ(static_cast<SizeType32>(mAcceptedFinished[batchSlot].isFinished()), finishedSumPtr[batchSlot]);
        }
    }

    void verifyAcceptByLogitsResults(SizeType32 seed)
    {
        auto finishedStepsPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedSteps));
        auto contextLengthsPtr = BufferRange<SizeType32>(*mContextLengths);
        auto outLogitsPtr = BufferRange<T>(*mTargetLogits);
        auto refLogitsPtr = BufferRange<T>(*mRefTargetLogits);
        auto numsDraftTokensPtr = BufferRange<SizeType32>(*mNumsDraftTokens);
        auto batchSlotsPtr = BufferRange<SizeType32>(*mBatchSlots);

        for (SizeType32 bi = 0; bi < mBatchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            for (SizeType32 si = 0; si < numsDraftTokensPtr[batchSlot]; ++si)
            {
                auto const outFinishedState = finishedStepsPtr[si * mMaxBatchSize + batchSlot];
                auto const logitsOffset = batchSlot * mMaxDraftTokens * mVocabSize + si * mVocabSize;
                if (si <= mAcceptedLen[batchSlot] - contextLengthsPtr[batchSlot])
                {
                    EXPECT_FALSE(outFinishedState.isSkipDecoding())
                        << " bi: " << bi << " si: " << si << " seed: " << seed;
                    for (SizeType32 vi = 0; vi < mVocabSize; ++vi)
                    {
                        auto const outLogit = static_cast<float>(outLogitsPtr[logitsOffset + vi]);
                        auto const refLogit = static_cast<float>(refLogitsPtr[logitsOffset + vi]);
                        EXPECT_FALSE((refLogit > -10) ^ (outLogit > -10))
                            << " bi: " << bi << " si: " << si << " vi: " << vi << " seed: " << seed;
                        if (refLogit > -10 && outLogit > -10)
                        {
                            if (!almostEqual(outLogit, refLogit, 1e-1, 1e-2))
                            {
                                std::cout << refLogit << " " << outLogit << std::endl;
                            }
                            ASSERT_TRUE(almostEqual(outLogit, refLogit, 1e-1, 1e-2))
                                << " bi: " << bi << " si: " << si << " vi: " << vi << " seed: " << seed;
                        }
                    }
                }
                else
                {
                    EXPECT_TRUE(outFinishedState.isSkipDecoding())
                        << " bi: " << bi << " si: " << si << " seed: " << seed;
                }
            }
        }
    }

    void verifyAcceptByIdsWithPathsResults(SizeType32 seed)
    {
        auto medusaLogitsPtrsPtr = BufferRange<T*>(*mMedusaLogitsPtrs);
        auto batchSlotsPtr = BufferRange<SizeType32>(*mBatchSlots);
        auto draftContextLengths = BufferRange<SizeType32>(*mSequenceLengths);
        auto draftContextLengthsInit = BufferRange<SizeType32>(*mSequenceLengthsCopy);
        auto acceptedLengths = BufferRange<SizeType32>(*mAcceptedLengths);
        auto outputIdsPtr = BufferRange<TokenIdType>(*mOutputTokens);
        auto bestPathIds = BufferRange<SizeType32>(*mBestPaths);
        auto finishedFinalPtr
            = reinterpret_cast<tk::FinishedState*>(bufferCast<tk::FinishedState::UnderlyingType>(*mFinishedFinal));

        for (SizeType32 bi = 0; bi < mBatchSize; ++bi)
        {
            auto const batchSlot = batchSlotsPtr[bi];
            auto const bestPathIdx = mAcceptedPathIdx[batchSlot];
            auto const lastTargetIdx = mLastTargetIdx[batchSlot];
            if (lastTargetIdx < 0)
            {
                continue;
            }

            auto const acceptedLen = mAcceptedLen[batchSlot];
            auto acceptedTokens = mRefAcceptedTokens[batchSlot];

            EXPECT_EQ(bestPathIds[batchSlot], bestPathIdx) << "bi: " << bi << " seed: " << seed;

            for (int32_t hi = 0; hi < mMaxNumHeads; ++hi)
            {
                auto refOffset
                    = tc::flat_index4(hi, batchSlot, lastTargetIdx, 0, mMaxBatchSize, mMaxDraftSeqPerStep, mVocabSize);
                auto outOffset
                    = static_cast<SizeType32>(medusaLogitsPtrsPtr[bi * mMaxNumHeads + hi] - static_cast<T*>(nullptr));
                EXPECT_EQ(outOffset, refOffset) << " bi: " << bi << " hi: " << hi << " seed: " << seed;
            }
            EXPECT_EQ(acceptedLengths[batchSlot], acceptedLen) << " bi: " << bi << " seed: " << seed;
            EXPECT_EQ(draftContextLengths[batchSlot], draftContextLengthsInit[batchSlot] + acceptedLen)
                << " bi: " << bi << " seed: " << seed << " out: " << draftContextLengths[batchSlot]
                << " ref: " << draftContextLengthsInit[batchSlot] + acceptedLen;

            for (SizeType32 ti = 0; ti < acceptedLen; ++ti)
            {
                ASSERT_EQ(mRefAcceptedTokens[batchSlot].size(), acceptedLen)
                    << " bi: " << bi << " ti: " << ti << " seed: " << seed;
                EXPECT_EQ(outputIdsPtr[batchSlot * mMaxSeqLen + draftContextLengthsInit[batchSlot] + ti],
                    mRefAcceptedTokens[batchSlot][ti])
                    << " bi: " << bi << " ti: " << ti << " seed: " << seed;
            }
            EXPECT_EQ(finishedFinalPtr[batchSlot].isFinished(), mFinishedByIdsPaths[batchSlot])
                << " bi: " << bi << " seed: " << seed;
        }
    }

    void verifyResult(SizeType32 seed)
    {
        switch (mAcceptMode)
        {
        case AcceptKernelMode::BY_IDS: verifyAcceptByIdsResults(seed); break;
        case AcceptKernelMode::BY_LOGITS: verifyAcceptByLogitsResults(seed); break;
        case AcceptKernelMode::BY_IDS_WITH_PATH: verifyAcceptByIdsWithPathsResults(seed); break;
        default: TLLM_CHECK(false); // Should never be here
        }
    }

    void runTest(DecodingKernelTestParam const& params)
    {
        mAcceptMode = params.mAcceptMode;

        mBatchSize = params.mBatchSize;
        mMaxBatchSize = params.mMaxBatchSize;
        mBeamWidth = params.mBeamWidth;
        mVocabSize = params.mVocabSize;
        mMaxDraftTokens = params.mMaxDraftTokens;
        mMaxSeqLen = params.mMaxSeqLen;

        mMaxNumHeads = params.mMaxNumHeads;
        if (mMaxNumHeads > 1 && mAcceptMode != AcceptKernelMode::BY_IDS_WITH_PATH)
        {
            GTEST_SKIP() << "MaxNumHeads > 1 is only supported for AcceptKernelMode::BY_IDS_WITH_PATH";
        }

        mMaxDraftSeqPerStep = params.mMaxDraftSeqPerStep;
        if (mMaxDraftSeqPerStep > 1 && mAcceptMode != AcceptKernelMode::BY_IDS_WITH_PATH)
        {
            GTEST_SKIP() << "MaxDraftSeqPerStep > 1 is only supported for AcceptKernelMode::BY_IDS_WITH_PATH";
        }

        mMaxTotalDraftTokens = mMaxDraftSeqPerStep * mMaxDraftTokens;
        mPadId = mVocabSize - 1;

        mMaxDraftSeqlen = mAcceptMode == AcceptKernelMode::BY_IDS_WITH_PATH ? mMaxDraftTokens - 1 : mMaxDraftTokens;
        mMaxTargetSeqlen = mAcceptMode == AcceptKernelMode::BY_IDS_WITH_PATH ? mMaxDraftTokens : mMaxSeqLen;

        createBuffers();

        for (SizeType32 seed = 0; seed < mSeeds; ++seed)
        {
            // if (seed != 145)
            // {
            //     continue;
            // }
            TLLM_LOG_DEBUG("Seed %d", seed);

            initData(seed);

            mStream->synchronize();

            callTestedKernel();

            mStream->synchronize();

            verifyResult(seed);
        }
    }

protected:
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;

    TensorPtr mDraftTokens;
    TensorPtr mTargetTokens;
    TensorPtr mOutputTokens;

    TensorPtr mDraftLogits;
    TensorPtr mTargetLogits;
    TensorPtr mTargetLogitsPtrs;
    TensorPtr mRefTargetLogits;

    TensorPtr mDraftProbs;
    TensorPtr mTargetProbs;

    TensorPtr mNumsDraftTokens;
    TensorPtr mSequenceLengths;
    TensorPtr mSequenceLengthsCopy;
    TensorPtr mAcceptedLengths;
    TensorPtr mContextLengths;
    TensorPtr mFinishedSteps;
    TensorPtr mFinishedFinal;
    TensorPtr mFinishedSum;
    TensorPtr mBatchSlots;

    TensorPtr mPaths;
    TensorPtr mEndIds;
    TensorPtr mMedusaLogitsPtrs;
    TensorPtr mMedusaInputLogitsPtrs;
    TensorPtr mTokensPerStep;
    TensorPtr mBestPaths;

    TensorPtr mCurandStates;

    std::vector<SizeType32> mAcceptedLen;
    std::vector<SizeType32> mOutputLen;
    std::vector<tk::FinishedState> mAcceptedFinished;
    std::vector<SizeType32> mAcceptedPathIdx;
    std::vector<SizeType32> mLastTargetIdx;
    std::vector<std::vector<SizeType32>> mRefAcceptedTokens;
    std::vector<bool> mFinishedByIdsPaths;

    SizeType32 mBatchSize;
    SizeType32 mMaxBatchSize;
    SizeType32 mBeamWidth;
    SizeType32 mMaxSeqLen;
    SizeType32 mVocabSize;
    SizeType32 mMaxDraftTokens;
    SizeType32 mMaxTotalDraftTokens;
    SizeType32 mMaxDraftSeqlen;
    SizeType32 mMaxTargetSeqlen;
    SizeType32 mMaxNumHeads;
    SizeType32 mMaxDraftSeqPerStep;
    AcceptKernelMode mAcceptMode;
    SizeType32 mPadId;
    static constexpr SizeType32 mSeeds = 64;
};

template class DecodingKernelsTest<float>;
template class DecodingKernelsTest<half>;

typedef testing::Types<float, half> FloatAndHalfTypes;

TYPED_TEST_SUITE(DecodingKernelsTest, FloatAndHalfTypes);

TYPED_TEST(DecodingKernelsTest, DISABLED_acceptDraftTokensByIdsKernelSmall)
{
    this->runTest(DecodingKernelTestParam()
                      .setBatchSize(1)
                      .setMaxSeqLen(16)
                      .setVocabSize(32)
                      .setMaxDraftTokens(8)
                      .setMaxDraftSeqPerStep(1)
                      .setAcceptMode(AcceptKernelMode::BY_IDS));
}

TYPED_TEST(DecodingKernelsTest, DISABLED_acceptDraftTokensByIdsKernelLarge)
{
    this->runTest(DecodingKernelTestParam()
                      .setBatchSize(128)
                      .setMaxSeqLen(128)
                      .setVocabSize(52000)
                      .setMaxDraftTokens(8)
                      .setMaxDraftSeqPerStep(1)
                      .setAcceptMode(AcceptKernelMode::BY_IDS));
}

TYPED_TEST(DecodingKernelsTest, DISABLED_acceptDraftTokensByLogitsKernelSmall)
{
    this->runTest(DecodingKernelTestParam()
                      .setBatchSize(1)
                      .setMaxSeqLen(16)
                      .setVocabSize(32)
                      .setMaxDraftTokens(8)
                      .setMaxDraftSeqPerStep(1)
                      .setAcceptMode(AcceptKernelMode::BY_LOGITS));
}

TYPED_TEST(DecodingKernelsTest, DISABLED_acceptDraftTokensByLogitsKernelLarge)
{
    this->runTest(DecodingKernelTestParam()
                      .setBatchSize(64)
                      .setMaxSeqLen(64)
                      .setVocabSize(4000)
                      .setMaxDraftTokens(8)
                      .setMaxDraftSeqPerStep(1)
                      .setAcceptMode(AcceptKernelMode::BY_LOGITS));
}

// FIXME: test is incorrect and too complicated.
TYPED_TEST(DecodingKernelsTest, DISABLED_acceptDraftTokensByIdsWithPathsKernelSmall)
{
    this->runTest(DecodingKernelTestParam()
                      .setBatchSize(1)
                      .setMaxSeqLen(128)
                      .setVocabSize(32)
                      .setMaxDraftTokens(3)
                      .setMaxDraftSeqPerStep(4)
                      .setMaxNumHeads(2)
                      .setAcceptMode(AcceptKernelMode::BY_IDS_WITH_PATH));
}

// FIXME: test is incorrect and too complicated.
TYPED_TEST(DecodingKernelsTest, DISABLED_acceptDraftTokensByIdsWithPathsKernelLarge)
{
    this->runTest(DecodingKernelTestParam()
                      .setBatchSize(128)
                      .setMaxSeqLen(1024)
                      .setVocabSize(4000)
                      .setMaxDraftTokens(8)
                      .setMaxDraftSeqPerStep(64)
                      .setMaxNumHeads(7)
                      .setAcceptMode(AcceptKernelMode::BY_IDS_WITH_PATH));
}
} // end of namespace
