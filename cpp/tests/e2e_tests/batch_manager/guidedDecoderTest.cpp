/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef TOP_LEVEL_DIR
#error "Define TOP_LEVEL_DIR"
#endif

#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/batch_manager/guidedDecoder.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/executor/executor.h"

using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::batch_manager;
namespace texec = tensorrt_llm::executor;

namespace
{
auto const TEST_RESOURCE_PATH = std::filesystem::path{TOP_LEVEL_DIR} / "cpp/tests/resources";
auto const DATA_PATH = TEST_RESOURCE_PATH / "data";
auto const GPT_XGRAMMAR_TOKENIZER_INFO_PATH = DATA_PATH / "gpt2" / "xgrammar_tokenizer_info.json";
auto const LLAMA_XGRAMMAR_TOKENIZER_INFO_PATH = DATA_PATH / "Llama-3.2-1B" / "xgrammar_tokenizer_info.json";
} // namespace

class GuidedDecoderTest : public ::testing::Test
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using VecTokens = std::vector<TokenIdType>;
    using RequestIdType = std::uint64_t;
    using RequestVector = std::vector<std::shared_ptr<LlmRequest>>;

    void SetUp() override
    {
        mStream = std::make_shared<CudaStream>();
        mRuntimeBufferManager = std::make_shared<BufferManager>(mStream);
    }

    void TearDown() override {}

    void initData(std::filesystem::path tokenizerInfoPath, SizeType32 vocabSizePadded, VecTokens outputIds,
        std::vector<int32_t> expectedNumRejected)
    {
        mLogitsDtype = nvinfer1::DataType::kFLOAT;
        mMaxNumRequests = 16;

        mVocabSizePadded = vocabSizePadded;
        auto const tokenizerInfo = nlohmann::json::parse(std::ifstream{tokenizerInfoPath});
        auto const encodedVocab = tokenizerInfo["encoded_vocab"].template get<std::vector<std::string>>();
        auto const tokenizerStr = tokenizerInfo["tokenizer_str"].template get<std::string>();
        auto const stopTokenIds = tokenizerInfo["stop_token_ids"].template get<std::vector<TokenIdType>>();
        texec::GuidedDecodingConfig guidedDecodingConfig(
            texec::GuidedDecodingConfig::GuidedDecodingBackend::kXGRAMMAR, encodedVocab, tokenizerStr, stopTokenIds);
        mGuidedDecoder = std::make_shared<GuidedDecoder>(
            guidedDecodingConfig, mMaxNumRequests, mVocabSizePadded, mLogitsDtype, *mRuntimeBufferManager);

        mLogits.resize(mMaxNumRequests);
        mLogitsHost.resize(mMaxNumRequests);
        for (int i = 0; i < mMaxNumRequests; i++)
        {
            mLogits[i] = mRuntimeBufferManager->gpu(ITensor::makeShape({mVocabSizePadded}), mLogitsDtype);
            mLogitsHost[i] = BufferManager::pinned(ITensor::makeShape({mVocabSizePadded}), mLogitsDtype);
        }

        mOutputIds = outputIds;
        mExpectedNumRejected = expectedNumRejected;
    }

    void resetLogits()
    {
        for (int i = 0; i < mMaxNumRequests; i++)
        {
            auto logitsHostData = bufferCast<float>(*mLogitsHost[i]);
            for (int j = 0; j < mVocabSizePadded; j++)
            {
                logitsHostData[j] = 0.0f;
            }
            mRuntimeBufferManager->copy(*(mLogitsHost[i]), *(mLogits[i]));
        }
    }

    void syncLogitsToHost()
    {
        for (int i = 0; i < mMaxNumRequests; i++)
        {
            mRuntimeBufferManager->copy(*(mLogits[i]), *(mLogitsHost[i]));
        }
    }

    int32_t countRejected(int i)
    {
        int32_t numRejected = 0;
        for (int j = 0; j < mVocabSizePadded; j++)
        {
            auto logitsHostData = bufferCast<float>(*mLogitsHost[i]);
            if (logitsHostData[j] < -1e6)
            {
                numRejected++;
            }
        }
        return numRejected;
    }

    void runTest()
    {
        auto llmReq1 = std::make_shared<LlmRequest>(1, 100, std::make_shared<VecTokens>(10), SamplingConfig(), false);
        texec::GuidedDecodingParams guidedDecodingParams(texec::GuidedDecodingParams::GuideType::kJSON);
        llmReq1->setGuidedDecodingParams(guidedDecodingParams);
        llmReq1->mSeqSlot = 1;

        auto llmReq2 = std::make_shared<LlmRequest>(1, 100, std::make_shared<VecTokens>(10), SamplingConfig(), false);
        llmReq2->mSeqSlot = 2;

        RequestVector contextRequests{llmReq1, llmReq2};
        RequestVector generationRequests{};
        ScheduledRequests scheduledRequests{contextRequests, generationRequests};
        DecoderInputBuffers decoderInputBuffers(mMaxNumRequests, 1, *mRuntimeBufferManager);

        for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
        {
            for (auto const& llmReq : requests)
            {
                decoderInputBuffers.decoderRequests.push_back(llmReq);
            }
        }
        decoderInputBuffers.logits = mLogits;

        // Context phase
        resetLogits();
        mGuidedDecoder->build(scheduledRequests);
        mGuidedDecoder->execute(decoderInputBuffers, *mRuntimeBufferManager);
        syncLogitsToHost();
        mRuntimeBufferManager->getStream().synchronize();

        // Move request to generation phase
        contextRequests.pop_back();
        contextRequests.pop_back();
        llmReq1->setState(LlmRequestState::kGENERATION_IN_PROGRESS);
        generationRequests.push_back(llmReq1);
        llmReq2->setState(LlmRequestState::kGENERATION_IN_PROGRESS);
        generationRequests.push_back(llmReq2);

        decoderInputBuffers.decoderRequests.clear();
        for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
        {
            for (auto const& llmReq : requests)
            {
                decoderInputBuffers.decoderRequests.push_back(llmReq);
            }
        }

        EXPECT_EQ(countRejected(0), mExpectedNumRejected[0]);
        EXPECT_EQ(countRejected(1), 0);

        // Generation phase
        for (int i = 0; i < mOutputIds.size(); i++)
        {
            llmReq1->addNewToken(mOutputIds[i], 0);
            llmReq2->addNewToken(mOutputIds[i], 0);

            resetLogits();
            mGuidedDecoder->build(scheduledRequests);
            mGuidedDecoder->execute(decoderInputBuffers, *mRuntimeBufferManager);
            syncLogitsToHost();
            mRuntimeBufferManager->getStream().synchronize();

            EXPECT_EQ(countRejected(0), mExpectedNumRejected[i + 1]);
            EXPECT_EQ(countRejected(1), 0);
        }
    }

private:
    SizeType32 mMaxNumRequests;
    SizeType32 mVocabSizePadded;
    nvinfer1::DataType mLogitsDtype;

    std::vector<TensorPtr> mLogits;     // [mBatchSize, mVocabSizePadded]
    std::vector<TensorPtr> mLogitsHost; // [mBatchSize, mVocabSizePadded]

    std::shared_ptr<BufferManager> mRuntimeBufferManager;
    std::shared_ptr<CudaStream> mStream;
    std::shared_ptr<GuidedDecoder> mGuidedDecoder;

    VecTokens mOutputIds;
    std::vector<int32_t> mExpectedNumRejected;
};

TEST_F(GuidedDecoderTest, GptTokenizer)
{
    VecTokens outputIds{4895, 824, 312, 1298, 366, 27743, 7934, 49793, 1600, 366, 12961, 19703, 4668, 1298, 366, 54,
        4537, 17, 12, 17469, 7919, 1600, 366, 3903, 10394, 1298, 366, 1485, 405, 41022, 20662};
    std::vector<int32_t> expectedNumRejected{50251, 219, 219, 219, 48558, 219, 219, 219, 219, 50191, 219, 219, 219, 219,
        48558, 219, 219, 219, 219, 219, 219, 219, 50191, 219, 219, 219, 48558, 219, 219, 219, 219, 50256};
    initData(GPT_XGRAMMAR_TOKENIZER_INFO_PATH, 50257, outputIds, expectedNumRejected);
    runTest();
}

TEST_F(GuidedDecoderTest, LlamaTokenizer)
{
    VecTokens outputIds{6377, 893, 333, 1115, 376, 27247, 6779, 7898, 545, 613, 376, 8926, 17830, 1115, 376, 29956,
        7228, 29906, 29899, 10399, 7734, 613, 376, 4980, 2103, 1115, 376, 29896, 29941, 29900, 29900, 341, 29890, 567,
        9092};
    std::vector<int32_t> expectedNumRejected{128235, 128235, 128235, 128235, 128235, 128235, 128235, 128235, 128235,
        128235, 128235, 128235, 128235, 128235, 128235, 128235, 128235, 128235, 128235, 128235, 128235, 128235, 128235,
        128235, 128235, 128235, 128235, 128235, 128235, 128235, 128235, 128235, 128235, 128235, 128235, 128235};
    initData(LLAMA_XGRAMMAR_TOKENIZER_INFO_PATH, 128256, outputIds, expectedNumRejected);
    runTest();
}
