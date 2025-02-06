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
#include <gtest/gtest.h>

#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/layers/lookaheadAlgorithm.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tests/unit_tests/layers/randomLlm.h"

namespace tensorrt_llm::tests::layers
{
namespace tk = tensorrt_llm::kernels;
namespace trk = tensorrt_llm::runtime::kernels;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::layers;

using TensorPtr = runtime::ITensor::SharedPtr;

TEST(LookaheadRandomllm, forward)
{
    auto ascii = std::make_shared<AsciiRandomTokenLogits>();
    EXPECT_EQ(ascii->getVocabSize(), 128);
    {
        auto tensor = ascii->tokenToLogits(static_cast<TokenIdType>('a'));
        auto token = ascii->logitsToToken(tensor);
        EXPECT_EQ(static_cast<char>(token), 'a');
    }
    {
        auto tensor = ascii->tokenToLogits(static_cast<TokenIdType>('W'));
        auto token = ascii->logitsToToken(tensor);
        EXPECT_EQ(static_cast<char>(token), 'W');
    }
    {
        std::string str("hello world!");
        TensorPtr logits
            = BufferManager::cpu(ITensor::makeShape({static_cast<SizeType32>(str.size()), ascii->getVocabSize()}),
                nvinfer1::DataType::kFLOAT);
        ascii->stringToLogits(logits, str);
        auto result = ascii->logitsToString(logits);
        EXPECT_EQ(result, str);
    }

    std::string oracle(
        "The following example uses a lambda-expression to increment all of the elements of a vector and "
        "then uses an overloaded operator() in a function object (a.k.a., \"functor\") to compute their sum. Note that "
        "to compute the sum, it is recommended to use the dedicated algorithm std::accumulate.");
    LookaheadRandomLlm llm(ascii, oracle);
    {
        TLLM_LOG_DEBUG("oracle[22]='%c'", oracle[22]);
        std::string input("ubcs23eess a la");
        auto len = static_cast<SizeType32>(input.size());
        TensorPtr inputTokens = initTensor(input);
        std::vector<TokenIdType> positionIdVec({22, 23, 24, 23, 24, 25, 24, 25, 26, 25, 26, 27, 26, 27, 28});
        TensorPtr positionIds = ITensor::wrap(positionIdVec, ITensor::makeShape({len}));
        TensorPtr outputLogits
            = BufferManager::cpu(ITensor::makeShape({len, ascii->getVocabSize()}), nvinfer1::DataType::kFLOAT);

        llm.forward(outputLogits, inputTokens, positionIds);

        auto result = ascii->logitsToString(outputLogits);
        auto invalid = ascii->getInvalidToken();
        TLLM_LOG_DEBUG("result=%s", result.c_str());
        for (SizeType32 i = 0; i < len; i++)
        {
            if (result[i] != invalid)
            {
                EXPECT_EQ(result[i], oracle[positionIdVec[i] + 1]);
            }
        }
    }
}

TEST(LookaheadRandomllm, gpuSampling)
{
    auto mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
    auto mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);

    int32_t device;
    struct cudaDeviceProp mDeviceProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&mDeviceProp, device);

    // auto mAscii = std::make_shared<RandomTokenLogits>();
    auto mAscii = std::make_shared<AsciiRandomTokenLogits>();

    std::vector<std::string> text({std::string("0123456789abcdef0123456789abcdef0123456&"),
        std::string("hello world, hello world, hello world!!&"),
        std::string("To be or not to be that is the question&"),
        std::string("To be or not to be that is the question&")});

    SizeType32 W = 5, N = 5, G = 5;
    SizeType32 maxBatchSize = 16;
    std::vector<SizeType32> batchSlotsVec({1, 4, 7, 11});
    SizeType32 batchSize = batchSlotsVec.size();
    SizeType32 vocabSizePadded = mAscii->getVocabSize();
    SizeType32 vocabSize = vocabSizePadded;
    SizeType32 maxTokensPerStep = (W + G) * (N - 1);
    SizeType32 maxNumHeads = 1;
    SizeType32 mRuntimeMaxTopK = 1;
    SizeType32 mMaxTopK = 1;
    SizeType32 mMaxTopP = 1.0;

    auto maxBatchShape1D = ITensor::makeShape({maxBatchSize});
    auto maxBatchShape3D = ITensor::makeShape({maxBatchSize, maxTokensPerStep, vocabSize});
    auto batchShape1D = ITensor::makeShape({batchSize});

    uint32_t mSeed = 0;
    SizeType32 mMaxSeqLen = 128;

    SizeType32 workspaceSize
        = tensorrt_llm::kernels::getTopKWorkspaceSize<float>(maxBatchSize, maxTokensPerStep, mMaxTopK, vocabSizePadded);
    TensorPtr workspaceDevice
        = mBufferManager->pinned(ITensor::makeShape({static_cast<int32_t>(workspaceSize)}), nvinfer1::DataType::kINT8);

    auto const dataType = TRTDataType<float>::value;
    auto const ptrType = TRTDataType<float*>::value;

    // Allocate GPU data
    TensorPtr mSeqLengths = BufferManager::pinned(maxBatchShape1D, nvinfer1::DataType::kINT32);
    TensorPtr mFinished = BufferManager::pinned(maxBatchShape1D, TRTDataType<tk::FinishedState::UnderlyingType>::value);
    TensorPtr mEndIds = BufferManager::pinned(maxBatchShape1D, nvinfer1::DataType::kINT32);
    TensorPtr mTopPs = BufferManager::pinned(maxBatchShape1D, nvinfer1::DataType::kFLOAT);
    TensorPtr mTopKs = BufferManager::pinned(maxBatchShape1D, nvinfer1::DataType::kINT32);
    TensorPtr mSkipDecode = BufferManager::pinned(maxBatchShape1D, nvinfer1::DataType::kBOOL);
    TensorPtr mTokensPerStep = BufferManager::pinned(maxBatchShape1D, nvinfer1::DataType::kINT32);

    TensorPtr mCurandStates
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize, sizeof(curandState_t)}), nvinfer1::DataType::kINT8);
    TensorPtr mOutputIds
        = BufferManager::pinned(ITensor::makeShape({maxBatchSize, mMaxSeqLen}), nvinfer1::DataType::kINT32);

    TensorPtr mProbs = BufferManager::pinned(maxBatchShape3D, dataType);

    TensorPtr mBatchSlots = BufferManager::pinned(batchShape1D, nvinfer1::DataType::kINT32);

    /////////////////////////////////////
    std::copy(batchSlotsVec.begin(), batchSlotsVec.end(), BufferRange<SizeType32>(*mBatchSlots).begin());

    auto batchSlotsPtr = bufferCast<int32_t>(*mBatchSlots);
    // Allocate and init curand states
    tk::invokeCurandInitialize(reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStates)), batchSlotsPtr,
        batchSize, mSeed, mStream->get());

    // Init by zero.
    trk::invokeFill(*mFinished, uint8_t{0}, *mStream);
    trk::invokeFill(*mOutputIds, int32_t{0}, *mStream);
    trk::invokeFill(*mSkipDecode, false, *mStream);
    trk::invokeFill(*mEndIds, mAscii->getEndToken(), *mStream);
    trk::invokeFill(*mTopPs, float{1.0}, *mStream);
    trk::invokeFill(*mTopKs, int32_t{1}, *mStream);
    trk::invokeFill(*mSeqLengths, int32_t{0}, *mStream);
    trk::invokeFill(*mTokensPerStep, maxTokensPerStep, *mStream);

    TLLM_CHECK(mMaxTopK * maxTokensPerStep <= mMaxSeqLen);

    // Init logits randomly
    for (SizeType32 bi = 0; bi < batchSize; bi++)
    {
        TensorPtr one = ITensor::at(mProbs, {bi});
        mAscii->stringToLogits(one, text[bi]);
        auto result = mAscii->logitsToString(one);
        EXPECT_EQ(result, text[bi]);
    }

    tensorrt_llm::kernels::TopKSamplingKernelParams<float> kernelParams;
    kernelParams.logProbs = bufferCast<float>(*mProbs);
    kernelParams.logProbsPtrs = nullptr;
    // kernelParams.outputIdsPtrs = bufferCast<int32_t*>(*mIdsPtrHost);
    // kernelParams.outputIds = nullptr;
    kernelParams.outputIdsPtrs = nullptr;
    kernelParams.outputIds = bufferCast<TokenIdType>(*mOutputIds);
    kernelParams.maxSeqLen = mMaxSeqLen;
    kernelParams.workspace = workspaceDevice->data();
    kernelParams.maxTopP = 1.0;
    kernelParams.topPs = bufferCast<float>(*mTopPs);
    kernelParams.maxTopK = mMaxTopK;
    kernelParams.topKs = bufferCast<int32_t>(*mTopKs);
    kernelParams.sequenceLengths = bufferCast<int32_t>(*mSeqLengths);
    kernelParams.endIds = bufferCast<int32_t>(*mEndIds);
    kernelParams.batchSlots = bufferCast<int32_t>(*mBatchSlots);
    kernelParams.finishedInput = reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
        bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(*mFinished));
    kernelParams.finishedOutput = reinterpret_cast<tensorrt_llm::kernels::FinishedState*>(
        bufferCast<tensorrt_llm::kernels::FinishedState::UnderlyingType>(*mFinished));
    kernelParams.skipDecode = bufferCast<bool>(*mSkipDecode);
    kernelParams.cumLogProbs = nullptr;
    kernelParams.outputLogProbs = nullptr;
    kernelParams.curandState = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStates));
    kernelParams.batchSize = batchSize;
    kernelParams.maxBatchSize = maxBatchSize;
    kernelParams.maxTokensPerStep = maxTokensPerStep;
    kernelParams.tokensPerStep = bufferCast<int32_t>(*mTokensPerStep);
    kernelParams.vocabSizePadded = vocabSize;
    kernelParams.normalizeLogProbs = false;
    kernelParams.logitsHasProbs = false;
    kernelParams.returnAllSelectedTokens = false;

    PRINT_TOKENS(mEndIds);
    PRINT_VALUES(mTokensPerStep);
    PRINT_VALUES(mBatchSlots);
    PRINT_VALUES(mTopKs);
    tensorrt_llm::kernels::invokeBatchTopKSampling(kernelParams, mStream->get());

    mStream->synchronize();

    std::ostringstream buf;
    buf << "finished states: ";
    for (SizeType32 bi = 0; bi < maxBatchSize; bi++)
    {
        buf << "[" << bi << "]=" << kernelParams.finishedOutput[bi].isFinished() << ", ";
    }
    TLLM_LOG_DEBUG(buf.str());

    for (SizeType32 bi = 0; bi < batchSize; bi++)
    {
        SizeType32 gbi = kernelParams.batchSlots[bi];
        bool finished = kernelParams.finishedOutput[bi].isFinished();
        TensorPtr one = ITensor::at(mOutputIds, {gbi});
        auto oneRange = BufferRange<TokenIdType>(*one);
        std::vector<char> result(mMaxSeqLen, '\0');
        std::copy(oneRange.begin(), oneRange.end(), result.begin());
        TLLM_LOG_DEBUG(result.data());
        EXPECT_EQ(text[bi], result.data());
    }
}

} // namespace tensorrt_llm::tests::layers
