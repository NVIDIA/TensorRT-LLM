/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <optional>
#include <queue>
#include <sstream>
#include <tuple>
#include <vector>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/layers/lookaheadDecodingLayer.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/lookaheadModule.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tests/unit_tests/layers/randomLlm.h"

namespace tensorrt_llm::tests::layers
{
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::layers;

namespace trk = tensorrt_llm::runtime::kernels;

using TensorPtr = runtime::ITensor::SharedPtr;
using TensorConstPtr = runtime::ITensor::SharedConstPtr;

struct TestParam
{
    SizeType32 maxBatchSize;

    enum BatchType
    {
        SINGLE_ONCE,
        SINGLE_TWICE,
        DYNAMIC
    } batchType;

    SizeType32 maxW;
    SizeType32 w;
    SizeType32 maxN;
    SizeType32 n;
    SizeType32 maxG;
    SizeType32 g;
};

class BatchSlotsManager
{
public:
    BatchSlotsManager(SizeType32 maxBatchSize, SizeType32 cases)
        : mMaxBatchSize(maxBatchSize)
        , mCases(cases)
    {
    }

    virtual std::vector<SizeType32> alloc(void) = 0;
    virtual void free(SizeType32 id) = 0;

    bool finished()
    {
        return mCases == 0;
    }

protected:
    SizeType32 quota(void)
    {
        return mCases - mRunning;
    }

    void consume(SizeType32 cases)
    {
        TLLM_CHECK(cases >= 0);
        TLLM_CHECK_DEBUG_WITH_INFO(cases <= mCases, "cases=%d, mCases=%d", cases, mCases);
        mRunning -= cases;
        mCases -= cases;
    }

protected:
    SizeType32 mMaxBatchSize{0};
    SizeType32 mCases{0};
    SizeType32 mRunning{0};
};

class SingleBatchSlotsManager : public BatchSlotsManager
{
public:
    SingleBatchSlotsManager(SizeType32 maxBatchSize, SizeType32 cases, SizeType32 id)
        : BatchSlotsManager(maxBatchSize, cases)
        , mId(id)
    {
        TLLM_CHECK(id < maxBatchSize);
    }

    virtual std::vector<SizeType32> alloc(void)
    {
        if (mState == FREE && quota() > 0)
        {
            mState = BUSY;
            mRunning += 1;
            return std::vector<SizeType32>({mId});
        }
        else
        {
            return std::vector<SizeType32>();
        }
    }

    virtual void free(SizeType32 id)
    {
        TLLM_CHECK(id == mId);
        mState = FREE;
        consume(1);
    }

private:
    enum
    {
        FREE,
        BUSY
    } mState{FREE};

    SizeType32 mId;
};

class DynamicBatchSlotsManager : public BatchSlotsManager
{
public:
    DynamicBatchSlotsManager(SizeType32 maxBatchSize, SizeType32 cases)
        : BatchSlotsManager(maxBatchSize, cases)
    {
        for (SizeType32 bi = 0; bi * 3 + 2 < maxBatchSize; bi++)
        {
            mFreeList.push(bi * 3 + 1);
            mFreeList.push(bi * 3 + 2);
            mFreeList.push(bi * 3);
        }
    }

    virtual std::vector<SizeType32> alloc()
    {
        SizeType32 waterline = mMaxBatchSize / 4;
        SizeType32 plan = mBusySet.size() < waterline ? rand() % (mMaxBatchSize / 4) : 0;
        SizeType32 num = std::min(plan, quota());
        std::vector<SizeType32> result;
        for (SizeType32 i = 0; i < num && !mFreeList.empty(); i++)
        {
            SizeType32 id = mFreeList.front();
            result.push_back(id);
            mBusySet.insert(id);
            mFreeList.pop();
        }
        mRunning += result.size();
        return result;
    }

    virtual void free(SizeType32 id)
    {
        auto search = mBusySet.find(id);
        TLLM_CHECK(search != mBusySet.end());
        mBusySet.erase(search);
        mFreeList.push(id);
        consume(1);
    }

private:
    std::queue<SizeType32> mFreeList;
    std::set<SizeType32> mBusySet;
};

class LookaheadDecodingLayerTest : public testing::Test
{
public:
    void SetUp() override;
    void TearDown() override;
    void runTest(TestParam const& param);

private:
    void allocateBuffers();

    void setupBuffers();

    void newRequests(std::vector<SizeType32> requestIds);

    void manageBatch();

    void llmForward();

    void decodeForward();

    void verifyDecode();

protected:
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> mBufferManager;
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> mStream;

    struct cudaDeviceProp mDeviceProp;

    TensorPtr mAlgoConfigBatch;

    TensorPtr mOutputIds;
    TensorPtr mSequenceLengths;
    TensorPtr mProbs;
    TensorPtr mEndIds;
    TensorPtr mTokensPerStep;
    TensorPtr mGoldenSampledTokens;
    TensorPtr mBatchSlots;
    TensorPtr mBatchSlotsMax;

    TensorPtr mNewTokens;
    TensorPtr mNumNewTokens;
    TensorPtr mNumNewTokensCumSum;
    TensorPtr mPathsOffsets;
    TensorPtr mDraftLengths;
    TensorPtr mPrevDraftLengths;
    TensorPtr mDraftTokens;
    TensorPtr mPackedMasks;
    TensorPtr mPackedMasksBool;
    TensorPtr mGenerationLengths;
    TensorPtr mPositionOffsets;
    TensorPtr mPositionIds;
    TensorPtr mAttentionPackedMask;

    TensorPtr mInputTokensBatch;
    TensorPtr mPositionIdsBatch;

    int32_t mMaxTopK = 1;
    static constexpr int32_t mMaxSeqLen = 512;
    float mMaxTopP = 1.0;
    std::shared_ptr<AsciiRandomTokenLogits> mAscii;
    std::vector<std::string> mOracle;
    std::vector<TensorPtr> mPrompt;
    std::vector<std::shared_ptr<RandomLlm>> mLlm;
    std::shared_ptr<LookaheadDecodingLayer<float>> mDecoder;
    std::shared_ptr<DecodingLayerWorkspace> mDecodingWorkspace;
    SizeType32 mVocabSize;
    SizeType32 mMaxTokensPerStep;
    TestParam mTestParam;
    std::shared_ptr<BatchSlotsManager> mBatchSlotsManager;
    std::vector<std::ostringstream> mScoreBoard;
    std::vector<TensorPtr> mHistogram;
    std::list<std::string> mReports;
};

void LookaheadDecodingLayerTest::SetUp()
{
    mStream = std::make_shared<tensorrt_llm::runtime::CudaStream>();
    mBufferManager = std::make_shared<tensorrt_llm::runtime::BufferManager>(mStream);

    int32_t device = 0;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&mDeviceProp, device);

    mAscii = std::make_shared<AsciiRandomTokenLogits>();
    mVocabSize = mAscii->getVocabSize();
}

void LookaheadDecodingLayerTest::TearDown() {}

void LookaheadDecodingLayerTest::allocateBuffers()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const maxBatchSize = mTestParam.maxBatchSize;
    auto const vocabSize = mAscii->getVocabSize();
    auto const maxBeamSize = 1;

    SizeType32 maxNumNewTokens, maxDraftLen, maxAcceptedDraftLen;
    std::tie(mMaxTokensPerStep, maxNumNewTokens, maxDraftLen, maxAcceptedDraftLen)
        = executor::LookaheadDecodingConfig(mTestParam.maxW, mTestParam.maxN, mTestParam.maxG)
              .calculateSpeculativeResource();
    //    mMaxTokensPerStep = maxTokensPerStep;

    auto const vocabSizePadded = vocabSize;
    auto const maxNumHeads = 1;
    std::ostringstream buf;

    std::vector<std::string> text({//
        std::string("To be, or not to be: that is the question. "
                    "To Be, Or Not To Be: That Is The Question.&"),
        std::string("Be not afraid of greatness. Some are born great, some achieve greatness, and others have "
                    "greatness thrust upon them. "
                    "Be Not Afraid Of Greatness. Some Are Born Great, Some Achieve Greatness, And Others Have "
                    "Greatness Thrust Upon Them.&"),
        std::string("Sweet are the uses of adversity which, like the toad, ugly and venomous, wears yet a precious "
                    "jewel in his head. "
                    "Sweet Are the Uses Of Adversity Which, Like The Toad, Ugly And Venomous, Wears Yet A Precious "
                    "Jewel In His Head.&"),
        std::string("Talking isn't doing. It is a kind of good deed to say well; and yet words are not deeds. "
                    "Talking Isn't Doing. It Is A Kind Of Good Deed To Say Well; And Yet Words Are Not Deeds.&"),
        std::string(
            "Reputation is an idle and most false imposition; oft got without merit, and lost without deserving. "
            "Reputation Is An Idle And Most False Imposition; Oft Got Without Merit, And Lost Without Deserving.&")});

    mOracle.resize(maxBatchSize);
    mLlm.resize(maxBatchSize);
    mPrompt.resize(maxBatchSize);
    mScoreBoard.resize(maxBatchSize);
    mHistogram.resize(maxBatchSize);
    for (SizeType32 gbi = 0; gbi < maxBatchSize; gbi++)
    {
        mOracle[gbi] = text[rand() % text.size()];
        mLlm[gbi] = std::make_shared<LookaheadRandomLlm>(mAscii, mOracle[gbi], gbi);

        mScoreBoard[gbi] = std::ostringstream();
        mHistogram[gbi] = BufferManager::cpu(ITensor::makeShape({mTestParam.n + 1}), nvinfer1::DataType::kINT32);
    }
    switch (mTestParam.batchType)
    {
    case TestParam::SINGLE_ONCE:
        mBatchSlotsManager = std::make_shared<SingleBatchSlotsManager>(maxBatchSize, 1, 1);
        break;
    case TestParam::SINGLE_TWICE:
        mBatchSlotsManager = std::make_shared<SingleBatchSlotsManager>(maxBatchSize, 2, 1);
        break;
    case TestParam::DYNAMIC:
        mBatchSlotsManager = std::make_shared<DynamicBatchSlotsManager>(maxBatchSize, maxBatchSize * 2);
        break;
    }

    auto lookaheadModule = std::make_shared<LookaheadModule>(mTestParam.maxN, mMaxTokensPerStep - 1);

    lookaheadModule->setExecutionConfig(
        executor::LookaheadDecodingConfig(mTestParam.maxW, mTestParam.maxN, mTestParam.maxG));
    auto const decodingDomain
        = tensorrt_llm::layers::DecoderDomain(maxBatchSize, 1, vocabSize, vocabSizePadded, lookaheadModule);

    mDecoder = std::make_shared<LookaheadDecodingLayer<float>>(decodingDomain, mBufferManager);

    TLLM_LOG_DEBUG("decoder ok");

    auto maxBatchShape1D = ITensor::makeShape({maxBatchSize});

    mAlgoConfigBatch = BufferManager::pinnedPool(ITensor::makeShape({maxBatchSize, 3}), nvinfer1::DataType::kINT32);

    mEndIds = BufferManager::pinnedPool(maxBatchShape1D, nvinfer1::DataType::kINT32);
    mTokensPerStep = BufferManager::pinnedPool(maxBatchShape1D, nvinfer1::DataType::kINT32);

    mOutputIds = BufferManager::pinnedPool(
        ITensor::makeShape({maxBatchSize, maxBeamSize, mMaxSeqLen + mMaxTokensPerStep}), nvinfer1::DataType::kINT32);
    mSequenceLengths = BufferManager::pinnedPool(maxBatchShape1D, nvinfer1::DataType::kINT32);

    mProbs = BufferManager::pinnedPool(
        ITensor::makeShape({maxBatchSize, mMaxTokensPerStep, vocabSize}), nvinfer1::DataType::kFLOAT);

    mGoldenSampledTokens
        = BufferManager::cpu(ITensor::makeShape({maxBatchSize, mMaxTokensPerStep}), nvinfer1::DataType::kINT32);
    mInputTokensBatch
        = BufferManager::pinnedPool(ITensor::makeShape({maxBatchSize, mMaxTokensPerStep}), nvinfer1::DataType::kINT32);
    mPositionIdsBatch
        = BufferManager::pinnedPool(ITensor::makeShape({maxBatchSize, mMaxTokensPerStep}), nvinfer1::DataType::kINT32);

    mNewTokens = BufferManager::pinnedPool(
        ITensor::makeShape({mMaxTokensPerStep, maxBatchSize, 1}), nvinfer1::DataType::kINT32);
    mNumNewTokens = BufferManager::pinnedPool(maxBatchShape1D, nvinfer1::DataType::kINT32);
    mDraftLengths = BufferManager::pinnedPool(maxBatchShape1D, nvinfer1::DataType::kINT32);
    mPrevDraftLengths = BufferManager::pinnedPool(maxBatchShape1D, nvinfer1::DataType::kINT32);
    mDraftTokens
        = BufferManager::pinnedPool(ITensor::makeShape({maxBatchSize, maxDraftLen}), nvinfer1::DataType::kINT32);
    auto packedMaskShape = ITensor::makeShape(
        {maxBatchSize, mMaxTokensPerStep, static_cast<ITensor::DimType64>(common::divUp(mMaxTokensPerStep, 32))});
    mPackedMasks = BufferManager::pinnedPool(packedMaskShape, nvinfer1::DataType::kINT32);
    mPackedMasksBool = BufferManager::pinnedPool(
        ITensor::makeShape({maxBatchSize, mMaxTokensPerStep, mMaxTokensPerStep}), nvinfer1::DataType::kBOOL);
    mNumNewTokensCumSum = BufferManager::pinnedPool(ITensor::makeShape({maxBatchSize + 1}), nvinfer1::DataType::kINT32);
    mPathsOffsets = BufferManager::pinnedPool(
        ITensor::makeShape({maxBatchSize, maxAcceptedDraftLen}), nvinfer1::DataType::kINT32);
    mGenerationLengths = BufferManager::pinnedPool(maxBatchShape1D, nvinfer1::DataType::kINT32);
    mPositionOffsets
        = BufferManager::pinnedPool(ITensor::makeShape({maxBatchSize, mMaxTokensPerStep}), nvinfer1::DataType::kINT32);
    mPositionIds
        = BufferManager::pinnedPool(ITensor::makeShape({maxBatchSize, mMaxTokensPerStep}), nvinfer1::DataType::kINT32);
    mAttentionPackedMask = BufferManager::pinnedPool(packedMaskShape, nvinfer1::DataType::kINT32);

    mBatchSlotsMax = BufferManager::pinnedPool(maxBatchShape1D, nvinfer1::DataType::kINT32);

    auto const batchSize = 0;
    auto batchShape1D = ITensor::makeShape({batchSize});
    auto batchShape2D = ITensor::makeShape({batchSize, mMaxTokensPerStep});

    mBatchSlots = ITensor::slice(mBatchSlotsMax, 0, batchSize);

    trk::invokeFill(*mEndIds, mAscii->getEndToken(), *mStream);
    trk::invokeFill(*mOutputIds, int32_t{0}, *mStream);
    trk::invokeFill(*mSequenceLengths, int32_t{0}, *mStream);
    trk::invokeFill(*mTokensPerStep, mMaxTokensPerStep, *mStream);
    mDecodingWorkspace = std::make_unique<tensorrt_llm::runtime::DecodingLayerWorkspace>(
        mBufferManager, decodingDomain, TRTDataType<float>::value, mDecoder->getWorkspaceSize());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadDecodingLayerTest::setupBuffers() {}

void LookaheadDecodingLayerTest::newRequests(std::vector<SizeType32> requestIds)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto const requestSize = requestIds.size();

    auto const beamSize = 1;
    SizeType32 vocabSize = mAscii->getVocabSize();

    ////////////////////////////////
    for (auto gbi : requestIds)
    {
        auto len = 5 + rand() % 10;
        auto prompt = mOracle[gbi].substr(0, len);

        TokenIdType contextToken = mOracle[gbi][len];
        SizeType32 contextLen = len + 1;

        BufferRange<TokenIdType> outputRange(*ITensor::at(mOutputIds, {gbi, 0}));
        for (auto& v : outputRange)
        {
            v = 0;
        }
        std::copy(prompt.begin(), prompt.end(), outputRange.begin());
        outputRange[len] = contextToken;
        BufferLocation<TokenIdType>(*mSequenceLengths).at(gbi) = len + 1;
        BufferLocation<TokenIdType>(*mDraftLengths).at(gbi) = 0;
        BufferLocation<SizeType32>(*mNumNewTokens).at(gbi) = 0;

        mPrompt[gbi] = ITensor::slice(mOutputIds, {gbi, 0, 0}, len + 1);

        for (auto& v : BufferRange<SizeType32>(*mHistogram[gbi]))
        {
            v = 0;
        }
        mScoreBoard[gbi] << "request id=[" << gbi << "] starts. prompt len=[" << len << "].";
    }

    TLLM_LOG_DEBUG("batch slots");
    ////////////////////////////////
    auto batchSize = ITensor::volume(mBatchSlots->getShape());
    BufferRange<SizeType32> batchSlotMaxRange(*mBatchSlotsMax);
    std::copy(requestIds.begin(), requestIds.end(), batchSlotMaxRange.begin() + batchSize);

    ////////////////////////////////
    auto setupParams = std::make_shared<LookaheadSetupParams>();
    setupParams->prompt.resize(0);
    setupParams->algoConfigs.resize(0);
    for (SizeType32 bi = 0; bi < requestSize; bi++)
    {
        SizeType32 gbi = requestIds[bi];
        setupParams->prompt.emplace_back(mPrompt[gbi]);
        setupParams->algoConfigs.emplace_back(mTestParam.w, mTestParam.n, mTestParam.g);
        PRINT_TOKENS(setupParams->prompt[bi]);
        setupParams->generationLengths = mGenerationLengths;
        setupParams->positionOffsets = mPositionOffsets;
        setupParams->attentionPackedMasks = mPackedMasks;
    }
    std::vector<uint64_t> seed(requestIds.begin(), requestIds.end());
    setupParams->randomSeed = std::make_optional(seed);
    TensorPtr newRequestSlots = ITensor::slice(mBatchSlotsMax, batchSize, requestSize);
    PRINT_VALUES(newRequestSlots);
    PRINT_VALUES(mBatchSlotsMax);
    mBatchSlots = ITensor::slice(mBatchSlotsMax, 0, batchSize);
    mDecodingWorkspace->setDeviceBatchSlots(newRequestSlots);
    mDecoder->setup(requestSize, beamSize, newRequestSlots, setupParams, mDecodingWorkspace);

    PRINT_VALUES(mPositionOffsets);

    batchSize += requestIds.size();
    mBatchSlots = ITensor::slice(mBatchSlotsMax, 0, batchSize);
    TLLM_LOG_DEBUG("new Requests mBatchSlots %s", D(mBatchSlots).values<int32_t>().c_str());
    PRINT_VALUES(mSequenceLengths);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadDecodingLayerTest::manageBatch()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const maxBatchSize = mTestParam.maxBatchSize;
    auto requests = mBatchSlotsManager->alloc();
    if (requests.size() > 0)
    {
        newRequests(requests);
    }
    PRINT_VALUES(mSequenceLengths);

    auto batchSize = ITensor::volume(mBatchSlots->getShape());
    BufferRange<SizeType32> batchSlotsRange(*mBatchSlots);
    auto batchShape1D = ITensor::makeShape({batchSize});
    auto batchShape2D = ITensor::makeShape({batchSize, mMaxTokensPerStep});
    auto newBatchSize = 0;
    PRINT_VALUES(mBatchSlots);
    for (SizeType32 bi = 0; bi < batchSize; bi++)
    {
        SizeType32 gbi = batchSlotsRange[bi];
        SizeType32 nbi = newBatchSize;

        TensorPtr theSequence = ITensor::at(mOutputIds, {gbi, 0});
        BufferRange<SizeType32> theSequenceRange(*theSequence);
        auto theSequenceLength = BufferRange<SizeType32>(*mSequenceLengths)[gbi];
        auto theNumNewTokens = BufferRange<SizeType32>(*mNumNewTokens)[gbi];

        TensorPtr generated = ITensor::slice(theSequence, 0, theSequenceLength);

        PRINT_TOKENS(generated);
        EXPECT_TRUE(mLlm[gbi]->verify(0, generated));

        BufferRange<SizeType32>(*mHistogram[gbi])[theNumNewTokens] += 1;

        if (BufferLocation<TokenIdType>(*theSequence).at(theSequenceLength - 1) == mAscii->getEndToken())
        {
            TLLM_LOG_DEBUG("request[%d] ends: '%s'", gbi, D(theSequence).string().c_str());
            mScoreBoard[gbi] << "[" << gbi << "] ends. " << D(mHistogram[gbi]).values<SizeType32>();
            mReports.push_back(mScoreBoard[gbi].str());
            mScoreBoard[gbi].str("");
            mScoreBoard[gbi].clear();
            mBatchSlotsManager->free(gbi);
        }
        else
        {
            batchSlotsRange[newBatchSize++] = gbi;
        }

        auto theDraftLen = BufferRange<SizeType32>(*mDraftLengths)[gbi];
        auto theGenerationLength = BufferRange<SizeType32>(*mGenerationLengths)[gbi];
        TLLM_CHECK_DEBUG_WITH_INFO(
            theDraftLen + 1 == theGenerationLength, "%d + 1 == %d", theDraftLen, theGenerationLength);
        BufferLocation<SizeType32>(*mTokensPerStep).at(gbi) = theGenerationLength;

        BufferLocation<TokenIdType>(*mInputTokensBatch).at(nbi, 0) = theSequenceRange[theSequenceLength - 1];
        mBufferManager->copy(*ITensor::slice(mDraftTokens, {gbi, 0}, theDraftLen),
            *ITensor::slice(mInputTokensBatch, {nbi, 1}, theDraftLen));
        mBufferManager->copy(*ITensor::slice(mPositionIds, {gbi, 0}), *ITensor::slice(mPositionIdsBatch, {nbi, 0}));
        BufferLocation<SizeType32>(*mPositionIdsBatch).at(nbi, 0) = theSequenceLength - 1;

        TLLM_LOG_DEBUG("W=%d, N=%d, G=%d, w=%d, n=%d, g=%d, draftLen = %d", mTestParam.maxW, mTestParam.maxN,
            mTestParam.maxG, mTestParam.w, mTestParam.n, mTestParam.g, theDraftLen);

        auto len = BufferRange<SizeType32>(*mTokensPerStep)[gbi];
        PRINT_TOKENS(ITensor::slice(mInputTokensBatch, {nbi, 0}, len));
        PRINT_VALUES(ITensor::slice(mPositionIdsBatch, {nbi, 0}, len));
    }
    mBatchSlots = ITensor::slice(mBatchSlotsMax, 0, newBatchSize);
    PRINT_VALUES(mBatchSlots);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void convertInt32ToBool(TensorPtr const& dst, TensorConstPtr const& src)
{
    auto dstShape = dst->getShape();
    auto srcShape = src->getShape();
    TLLM_CHECK(dstShape.d[0] == srcShape.d[0]);
    TLLM_CHECK(dstShape.d[1] <= srcShape.d[1] * 32);
    BufferLocation<bool> dstLocation(*dst);
    BufferLocation<SizeType32 const> srcLocation(*src);
    auto testBit = [](SizeType32 x, SizeType32 idx) { return x & (1 << idx); };
    for (auto i = 0; i < dstShape.d[0]; i++)
    {
        for (auto j = 0; j < dstShape.d[1]; j++)
        {
            dstLocation.at(i, j) = testBit(srcLocation.at(i, j / 32), j % 32);
        }
    }
}

void convertBoolToInt32(TensorPtr const& dst, TensorConstPtr const& src)
{
    auto dstShape = dst->getShape();
    auto srcShape = src->getShape();
    TLLM_CHECK(dstShape.d[0] == srcShape.d[0]);
    TLLM_CHECK(dstShape.d[1] * 32 >= srcShape.d[1]);
    BufferLocation<SizeType32> dstLocation(*dst);
    BufferLocation<bool const> srcLocation(*src);

    for (auto i = 0; i < dstLocation.size(); i++)
    {
        dstLocation[i] = 0;
    }

    auto setBit = [](SizeType32& x, SizeType32 idx, bool value) { x |= (value << idx); };
    for (auto i = 0; i < srcShape.d[0]; i++)
    {
        for (auto j = 0; j < srcShape.d[1]; j++)
        {
            setBit(dstLocation.at(i, j / 32), j % 32, srcLocation.at(i, j));
        }
    }
}

void LookaheadDecodingLayerTest::llmForward()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto batchSize = ITensor::volume(mBatchSlots->getShape());

    for (SizeType32 bi = 0; bi < batchSize; bi++)
    {
        auto gbi = BufferRange<SizeType32>(*mBatchSlots)[bi];
        auto start = BufferRange<SizeType32>(*mSequenceLengths)[gbi] - 1;
        auto len = BufferRange<SizeType32>(*mTokensPerStep)[gbi];
        TLLM_LOG_DEBUG("LookaheadDecodingLayerTest::llmForward input len=%d", len);
        TensorPtr output = ITensor::slice(mProbs, {bi, 0}, len);
        TensorPtr golden = ITensor::slice(mGoldenSampledTokens, {gbi, 0}, len);

        BufferRange<SizeType32> idRange(*ITensor::slice(mPositionIdsBatch, {bi, 0}, len));
        BufferRange<SizeType32> offsetRange(*ITensor::slice(mPositionOffsets, {gbi, 0}, len));
        PRINT_VALUES(ITensor::slice(mPositionIdsBatch, {bi, 0}));
        PRINT_VALUES(ITensor::slice(mPositionOffsets, {bi, 0}));
        for (auto i = 0; i < idRange.size(); i++)
        {
            TLLM_CHECK(idRange[i] == start + offsetRange[i]);
        }

        if (false)
        {
            convertInt32ToBool(ITensor::at(mPackedMasksBool, {gbi}), ITensor::at(mPackedMasks, {gbi}));
            mLlm[gbi]->forward(output,                           //
                ITensor::slice(mInputTokensBatch, {bi, 0}, len), //
                ITensor::slice(mPositionIdsBatch, {bi, 0}, len), //
                ITensor::at(mPackedMasksBool, {gbi}));
        }
        else
        {
            convertInt32ToBool(ITensor::at(mPackedMasksBool, {gbi}), ITensor::at(mPackedMasks, {gbi}));
            mLlm[gbi]->forward(output,                           //
                start,                                           //
                ITensor::slice(mInputTokensBatch, {bi, 0}, len), //
                ITensor::slice(mPositionOffsets, {gbi, 0}, len), //
                ITensor::at(mPackedMasksBool, {gbi}));
        }

        mAscii->logitsToTensor(golden, output);
        TLLM_LOG_DEBUG("batch[%d] LLM golden: '%s'", gbi, D(golden).tokens().c_str());
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadDecodingLayerTest::decodeForward()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto batchSize = ITensor::volume(mBatchSlots->getShape());
    PRINT_VALUES(mBatchSlots);

    auto inputParams = std::make_shared<LookaheadDecodingInputs>(mEndIds, mBatchSlots);
    inputParams->localBatchSize = batchSize;
    inputParams->logits = ITensor::slice(mProbs, 0, batchSize);
    inputParams->batchSlots = mBatchSlots;
    inputParams->curTokensPerStep = mTokensPerStep;

    auto outputParams = std::make_shared<LookaheadDecodingOutputs>(mOutputIds);

    PRINT_VALUES(mSequenceLengths);
    outputParams->sequenceLength = mSequenceLengths;
    outputParams->nextDraftLengths = mDraftLengths;
    outputParams->prevDraftLengths = mPrevDraftLengths;
    outputParams->nextDraftTokens = mDraftTokens;
    outputParams->packedMasks = mPackedMasks;
    outputParams->numNewTokens = mNumNewTokens;
    outputParams->newTokens = mNewTokens;
    outputParams->numNewTokensCumSum = mNumNewTokensCumSum;
    outputParams->pathsOffsets = mPathsOffsets;
    outputParams->generationLengths = mGenerationLengths;
    outputParams->positionOffsets = mPositionOffsets;
    outputParams->positionIds = mPositionIds;
    outputParams->packedMasks = mPackedMasks;

    PRINT_VALUES(mTokensPerStep);

    mDecodingWorkspace->setDeviceBatchSlots(mBatchSlots);
    mDecoder->forwardAsync(outputParams, inputParams, mDecodingWorkspace);

    mStream->synchronize();

    mDecodingWorkspace->setDeviceBatchSlots(mBatchSlots);
    mDecoder->forwardSync(outputParams, inputParams, mDecodingWorkspace);

    mStream->synchronize();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LookaheadDecodingLayerTest::verifyDecode()
{
    auto batchSize = ITensor::volume(mBatchSlots->getShape());
    for (SizeType32 bi = 0; bi < batchSize; bi++)
    {
        auto gbi = BufferRange<SizeType32>(*mBatchSlots)[bi];
        auto len = BufferRange<SizeType32>(*mTokensPerStep)[gbi];
        auto sequenceLength = BufferLocation<SizeType32>(*mSequenceLengths).at(gbi);

        auto draftLength = BufferLocation<SizeType32>(*mDraftLengths).at(gbi);
        auto generationLength = BufferLocation<SizeType32>(*mGenerationLengths).at(gbi);
        BufferRange<SizeType32> posOffsetRange(*ITensor::slice(mPositionOffsets, {gbi, 0}, generationLength));
        BufferRange<SizeType32> posIdRange(*ITensor::slice(mPositionIds, {gbi, 0}, generationLength));
        TLLM_LOG_DEBUG("generationLength = %d, draftLength = %d", generationLength, draftLength);
        TLLM_CHECK(draftLength + 1 == generationLength);
        TLLM_CHECK(posOffsetRange[0] == 0);
        TLLM_CHECK(posIdRange[0] == sequenceLength - 1);
        for (SizeType32 i = 0; i < posIdRange.size(); i++)
        {
            TLLM_CHECK(posIdRange[i] == posOffsetRange[i] + sequenceLength - 1);
        }
    }

    BufferRange<SizeType32> cumSumRange(*mNumNewTokensCumSum);
    BufferRange<SizeType32> pathOffsetsRange(*mPathsOffsets);
    PRINT_VALUES(mNumNewTokensCumSum);
    for (SizeType32 bi = 0; bi < batchSize; bi++)
    {
        auto gbi = BufferRange<SizeType32>(*mBatchSlots)[bi];
        SizeType32 pathOffsetBegin = cumSumRange[bi];
        SizeType32 pathOffsetEnd = cumSumRange[bi + 1];
        TensorPtr golden = ITensor::at(mGoldenSampledTokens, {gbi});
        auto sequenceLength = BufferLocation<SizeType32>(*mSequenceLengths).at(gbi);
        auto numNewTokens = BufferLocation<SizeType32>(*mNumNewTokens).at(gbi);
        TensorPtr newTokens = ITensor::slice(mOutputIds, {gbi, 0, sequenceLength - numNewTokens}, numNewTokens);
        BufferRange<SizeType32> goldenRange(*ITensor::at(mGoldenSampledTokens, {gbi}));
        BufferRange<TokenIdType> newTokensRange(*newTokens);

        SizeType32 ni = 1;
        for (SizeType32 poi = pathOffsetBegin; poi < pathOffsetEnd; poi++)
        {
            TLLM_CHECK(goldenRange[pathOffsetsRange[poi] + 1] == newTokensRange[ni++]);
        }
    }
}

void LookaheadDecodingLayerTest::runTest(TestParam const& param)
{
    TLLM_LOG_DEBUG("TEST BEGIN: maxBatchSize=%d, mode=%d, WNG=(%d, %d, %d), wng=(%d, %d, %d)", param.maxBatchSize,
        param.batchType, param.maxW, param.maxN, param.maxG, param.w, param.n, param.g);
    srand(42);

    mTestParam = param;
    allocateBuffers();

    int step = 0;
    for (; !mBatchSlotsManager->finished() && step < 3000; step++)
    {
        TLLM_LOG_DEBUG("!!!!!!!!!!!!!!!! < %d > !!!!!!!!!!!!!!!!", step);
        manageBatch();
        if (ITensor::volume(mBatchSlots->getShape()))
        {
            llmForward();
            mStream->synchronize();
            decodeForward();
            verifyDecode();
        }
    }

    for (auto& r : mReports)
    {
        TLLM_LOG_DEBUG(r);
    }
    if (!mBatchSlotsManager->finished())
    {
        TLLM_LOG_INFO("step=%d is not enough", step);
    }
}

TEST_F(LookaheadDecodingLayerTest, singleOnce)
{
    this->runTest(TestParam{16, TestParam::SINGLE_ONCE, 5, 3, 5, 3, 5, 3});
}

TEST_F(LookaheadDecodingLayerTest, singleTwice)
{
    this->runTest(TestParam{16, TestParam::SINGLE_TWICE, 7, 5, 7, 5, 7, 5});
}

TEST_F(LookaheadDecodingLayerTest, dynamic)
{
    this->runTest(TestParam{16, TestParam::DYNAMIC, 5, 5, 5, 5, 5, 5});
}

TEST_F(LookaheadDecodingLayerTest, dynamicLarge)
{
    this->runTest(TestParam{32, TestParam::DYNAMIC, 7, 6, 7, 6, 9, 8});
}

TEST_F(LookaheadDecodingLayerTest, dynamicSmall_110)
{
    this->runTest(TestParam{16, TestParam::SINGLE_TWICE, 1, 1, 2, 2, 0, 0});
}

TEST_F(LookaheadDecodingLayerTest, dynamicSmall_311)
{
    this->runTest(TestParam{32, TestParam::DYNAMIC, 3, 2, 2, 2, 1, 1});
}

TEST_F(LookaheadDecodingLayerTest, dynamicSmall_131)
{
    this->runTest(TestParam{32, TestParam::DYNAMIC, 1, 1, 3, 2, 1, 1});
}

TEST_F(LookaheadDecodingLayerTest, dynamicSmall_113)
{
    this->runTest(TestParam{32, TestParam::DYNAMIC, 1, 1, 2, 2, 3, 2});
}

TEST_F(LookaheadDecodingLayerTest, dynamicSmall_112110)
{
    this->runTest(TestParam{4, TestParam::SINGLE_TWICE, 1, 1, 2, 1, 1, 0});
}

using ParamType = std::tuple<SizeType32, TestParam::BatchType, std::tuple<SizeType32, SizeType32>,
    std::tuple<SizeType32, SizeType32>, std::tuple<SizeType32, SizeType32>>;

static int g_id = 0;

std::string generateTestName(testing::TestParamInfo<ParamType> const& info)
{
    auto [maxBatchSize, mode, Ww, Nn, Gg] = info.param;
    auto [W, w] = Ww;
    auto [N, n] = Nn;
    auto [G, g] = Gg;
    std::ostringstream buf;
    buf << (g_id++) << "maxBatchSize_" << maxBatchSize << "__mode_" << mode << '_' << '_' << W << '_' << w << '_' << '_'
        << N << '_' << n << '_' << '_' << G << '_' << g << '_';
    return buf.str();
}

class ParamTest : public LookaheadDecodingLayerTest, public ::testing::WithParamInterface<ParamType>
{
};

TEST_P(ParamTest, Test)
{
    srand(42);

    auto [maxBatchSize, mode, Ww, Nn, Gg] = GetParam();
    auto [W, w] = Ww;
    auto [N, n] = Nn;
    auto [G, g] = Gg;
    if (!executor::LookaheadDecodingConfig::isLegal(W, N, G) || !executor::LookaheadDecodingConfig::isLegal(w, n, g))
    {
        TLLM_LOG_DEBUG("Just Pass for illegal parameter combination");
        GTEST_SKIP() << "Algorithm does not support these parameters WNG=(" << W << ", " << N << ", " << G << "), wng=("
                     << w << ", " << n << ", " << g << ")";
    }
    runTest(TestParam{maxBatchSize, mode, W, w, N, n, G, g});
}

INSTANTIATE_TEST_SUITE_P(LookaheadDecodingLayerParamTest, ParamTest,
    testing::Combine( //
        testing::Values(4, 16), testing::Values(TestParam::DYNAMIC),
        testing::Values(std::make_tuple(1, 1), std::make_tuple(3, 3), std::make_tuple(5, 5), std::make_tuple(2, 1),
            std::make_tuple(3, 2), std::make_tuple(5, 3)),
        testing::Values(std::make_tuple(1, 1), std::make_tuple(3, 3), std::make_tuple(5, 5), std::make_tuple(2, 1),
            std::make_tuple(3, 2), std::make_tuple(5, 3)),
        testing::Values(std::make_tuple(0, 0), std::make_tuple(3, 3), std::make_tuple(5, 5), std::make_tuple(1, 0),
            std::make_tuple(3, 2), std::make_tuple(5, 3))),
    generateTestName);

} // namespace tensorrt_llm::tests::layers
