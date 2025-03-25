/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/dataTransceiverImpl.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/cache_transmission/mpi_utils/connection.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <tensorrt_llm/batch_manager/mlaCacheFormatter.h>
#include <tensorrt_llm/executor/cache_transmission/cacheConcatenate.h>

#include "gtest/gtest.h"
#include <gmock/gmock.h>

namespace tr = tensorrt_llm::runtime;
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using LlmRequest = tensorrt_llm::batch_manager::LlmRequest;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::batch_manager;
namespace texec = tensorrt_llm::executor;

using testing::Return;
using testing::ReturnRef;

// ---------------------------------------
//            RequestInfoTest
// ---------------------------------------

namespace
{

template <typename T>
T serializeDeserialize(T const& val)
{
    auto size = T::serializedSize(val);
    std::ostringstream oss;
    T::serialize(val, oss);
    EXPECT_EQ(oss.str().size(), size);

    std::istringstream iss(oss.str());
    return T::deserialize(iss);
}

} // namespace

class RequestInfoTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(RequestInfoTest, Basic)
{
    if (tensorrt_llm::mpi::MpiComm::world().getSize() > 2)
    {
        GTEST_SKIP() << "mpirun with procs<=2 is required to run this test.";
    }
    auto state = std::make_unique<texec::DataTransceiverState>();
    state->setCommState(texec::kv_cache::CommState{12, "127.0.0.1"});
    state->setCacheState(texec::kv_cache::CacheState{10, 12, 128, 128, 8, 8, nvinfer1::DataType::kFLOAT});
    RequestInfo info{1, *state};
    auto info2 = serializeDeserialize(info);
    EXPECT_EQ(info, info2);
}

// ---------------------------------------
//            CacheConfigTest
// ---------------------------------------

class CacheConfigTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(CacheConfigTest, EqualTo)
{
    if (tensorrt_llm::mpi::MpiComm::world().getSize() > 2)
    {
        GTEST_SKIP() << "mpirun with procs<=2 is required to run this test.";
    }
    using tensorrt_llm::executor::kv_cache::CacheState;
    constexpr SizeType32 vocabSize{25};
    constexpr SizeType32 nbAttentionLayers{10};
    constexpr SizeType32 nbRnnLayers{2};
    constexpr SizeType32 nbHeads{12};
    constexpr SizeType32 hiddenSize{768};
    constexpr nvinfer1::DataType dtype{nvinfer1::DataType::kFLOAT};
    constexpr SizeType32 tokensPerBlock{64};
    constexpr SizeType32 tensorParallelism{8};
    constexpr SizeType32 pipelineParallelism{2};
    constexpr SizeType32 contextParallelism{1};
    constexpr SizeType32 sizePerHead{hiddenSize / nbHeads};
    constexpr CacheState::AttentionType attentionType{CacheState::AttentionType::kDEFAULT};
    constexpr int kvFactor = 2;
    tr::ModelConfig modelConfig{
        vocabSize, nbAttentionLayers + nbRnnLayers, nbAttentionLayers, nbRnnLayers, nbHeads, hiddenSize, dtype};
    modelConfig.setTokensPerBlock(tokensPerBlock);
    tr::WorldConfig worldConfig{tensorParallelism, pipelineParallelism, contextParallelism};

    texec::kv_cache::CacheState::ModelConfig cacheStateCfg{
        modelConfig.getNumKvHeadsPerLayer(), modelConfig.getSizePerHead(), modelConfig.getTokensPerBlock()};

    texec::kv_cache::CacheState state0{
        cacheStateCfg, worldConfig, modelConfig.getKvDataType(), attentionType, kvFactor};
    texec::kv_cache::CacheState state1{nbAttentionLayers, nbHeads, sizePerHead, tokensPerBlock, tensorParallelism,
        pipelineParallelism, dtype, attentionType, kvFactor, false, 0, tensorParallelism};
    EXPECT_EQ(state0, state1);
}

// ---------------------------------------
//          MockTransceiverTest
// ---------------------------------------

class MockDataSender : public DataSender
{
public:
    MockDataSender()
    {
        ON_CALL(*this, getCommState).WillByDefault(ReturnRef(mState));
        ON_CALL(*this, recvRequestInfo)
            .WillByDefault(Return(RequestInfo{0,
                texec::DataTransceiverState{
                    texec::kv_cache::CacheState{10, 12, 128, 128, 8, 8, nvinfer1::DataType::kFLOAT},
                    texec::kv_cache::CommState{std::vector<SizeType32>{0}, 0}}}));
        ON_CALL(*this, getCounterpartsCount).WillByDefault(Return(1));
    }

    MOCK_METHOD(RequestInfo, recvRequestInfo, (), (override));
    MOCK_METHOD(void, sendSync, (LlmRequest const&), (override));
    MOCK_METHOD(texec::kv_cache::CommState const&, getCommState, (), (const));
    MOCK_METHOD(void, setCommState, (texec::kv_cache::CommState), (override));
    MOCK_METHOD(size_t, getCounterpartsCount, (LlmRequest::RequestIdType), (const));
    MOCK_METHOD(void, release, (LlmRequest::RequestIdType), (override));

private:
    static texec::kv_cache::CommState mState;
};

texec::kv_cache::CommState MockDataSender::mState;

class MockDataReceiver : public DataReceiver
{
public:
    MOCK_METHOD(void, sendRequestInfo, (LlmRequest const&), (override));
    MOCK_METHOD(void, receiveSync, (LlmRequest const&), (override));
};

class MockTransceiverTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    void SetUp() override {}

    void TearDown() override {}

    static auto makeLlmRequest(
        LlmRequest::RequestIdType requestId = 0, SizeType32 maxNewTokens = 1, VecTokens inputTokens = {-1})
    {
        texec::Request request{std::move(inputTokens), maxNewTokens};
        auto state = std::make_unique<texec::DataTransceiverState>();
        auto stats = texec::ContextPhaseParams({}, requestId, state.release(), std::nullopt);
        request.setContextPhaseParams(std::move(stats));
        return std::make_unique<LlmRequest>(requestId, std::move(request));
    }
};

TEST_F(MockTransceiverTest, MpiResponderBasic)
{
    if (tensorrt_llm::mpi::MpiComm::world().getSize() > 2)
    {
        GTEST_SKIP() << "mpirun with procs<=2 is required to run this test.";
    }
    auto sender = std::make_unique<MockDataSender>();
    EXPECT_CALL(*sender, recvRequestInfo)
        .WillOnce(Return(RequestInfo{0,
            texec::DataTransceiverState{texec::kv_cache::CacheState{10, 12, 128, 128, 8, 8, nvinfer1::DataType::kFLOAT},
                texec::kv_cache::CommState{std::vector<SizeType32>{0}, 0}}}));
    EXPECT_CALL(*sender, sendSync).WillOnce(Return());
    EXPECT_CALL(*sender, getCounterpartsCount).WillOnce(Return(1));
    EXPECT_CALL(*sender, release).WillOnce(Return());

    DataResponder responder{std::move(sender)};
    auto request = makeLlmRequest(0);
    auto future = responder.respondAndSendAsync(*request);
    future.get();
}

TEST_F(MockTransceiverTest, MpiRequesterBasic)
{

    if (tensorrt_llm::mpi::MpiComm::world().getSize() > 2)
    {
        GTEST_SKIP() << "mpirun with procs<=2 is required to run this test.";
    }
    auto receiver = std::make_unique<MockDataReceiver>();
    EXPECT_CALL(*receiver, sendRequestInfo).WillOnce(Return());
    EXPECT_CALL(*receiver, receiveSync).WillOnce(Return());
    DataRequester requester{std::move(receiver)};
    auto request = makeLlmRequest(0);
    auto state = std::make_unique<texec::DataTransceiverState>();
    state->setCommState(texec::kv_cache::CommState{std::vector<int>{0}});
    auto stats = texec::ContextPhaseParams({}, 0, state.release(), std::nullopt);
    request->setContextPhaseParams(std::move(stats));
    auto future = requester.requestAndReceiveAsync(*request);
    future.get();
}

// TODO: Restore multi-rank tests.

// ---------------------------------------
//          RealTransceiverTest
// ---------------------------------------

class SymmetricalCacheTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
protected:
    void SetUp() override {}

    void TearDown() override
    {
        for (auto& future : mFutures)
        {
            if (future.valid())
            {
                future.get();
            }
        }
    }

    SizeType32 setUpCommunicator()
    {
        tensorrt_llm::mpi::initialize(tensorrt_llm::mpi::MpiThreadSupport::THREAD_MULTIPLE);
        mComm = std::addressof(tensorrt_llm::mpi::MpiComm::world());
        mWorldSize = mComm->getSize();
        mlocalRank = mComm->getRank() / 2;
        isSender = mComm->getRank() % 2 == 0;
        tensorrt_llm::mpi::MpiComm::setSession(mComm->split(static_cast<int>(isSender), mlocalRank));
        return mWorldSize;
    }

    void setUpCacheManager()
    {
        auto constexpr numLayers = 4;
        auto constexpr numHeads = 2;
        auto constexpr sizePerHead = 64;
        auto constexpr hiddenSize = numHeads * sizePerHead;
        auto constexpr tokensPerBlock = 8;
        auto constexpr maxBlocksPerSeq = 10;
        auto constexpr maxBeamWidth = 4;
        auto constexpr sinkTokenLength = 0;
        mMaxNumSequences = 8;
        auto const stream = std::make_shared<tr::CudaStream>();

        auto constexpr maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
        auto constexpr maxAttentionWindow = maxNumTokens;
        auto constexpr temporaryAttentionWindow = 0;
        auto constexpr inputLength = maxNumTokens - tokensPerBlock - 1;
        auto constexpr numSharedBlocks = inputLength / tokensPerBlock;
        auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

        auto totalNumBlocks = mMaxNumSequences * numBlocksPerSeq;
        auto constexpr blocksInSecondaryPool = 0;

        auto constexpr enableBlockReuse = true;
        auto constexpr onboardBlocks = true;
        auto constexpr dataType = nvinfer1::DataType::kFLOAT;

        mManager = std::make_unique<KVCacheManager>(numLayers, numHeads, sizePerHead, tokensPerBlock, totalNumBlocks,
            blocksInSecondaryPool, mMaxNumSequences, maxBeamWidth, std::vector<SizeType32>{maxAttentionWindow},
            temporaryAttentionWindow, sinkTokenLength, stream, std::nullopt, enableBlockReuse, onboardBlocks,
            CacheType::kSELF, std::nullopt, nullptr, true);
        mCacheState = std::make_unique<texec::kv_cache::CacheState>(
            numLayers, numHeads, sizePerHead, tokensPerBlock, 1, 1, dataType);
        mConnectionManager = std::make_unique<texec::kv_cache::MpiConnectionManager>(mComm);

        // UVM seems to be incompatible with MPI, and it is continuing to investigate.
        bool constexpr useUvm = false;
        mManager->allocatePools(dataType, useUvm);
    }

    void setUpCacheTransceiver()
    {
        if (isSender)
        {
            mResponder = std::make_unique<DataResponder>(std::make_unique<DataSenderImpl>(
                mConnectionManager.get(), *mCacheState, mlocalRank, std::make_unique<CacheFormatter>(mManager.get())));
        }
        else
        {
            mRequester = std::make_unique<DataRequester>(std::make_unique<DataReceiverImpl>(
                mConnectionManager.get(), *mCacheState, mlocalRank, std::make_unique<CacheFormatter>(mManager.get())));
        }
    }

    auto makeLlmRequest(SizeType32 length)
    {
        constexpr SizeType32 maxNewTokens{1};
        // create request with tokens [length, ..., length] (<length> tokens)
        texec::Request request{VecTokens(length, length), maxNewTokens};
        auto state = std::make_unique<texec::DataTransceiverState>();
        state->setCommState(texec::kv_cache::CommState{std::vector<int>{0}});
        state->setCacheState(*mCacheState);
        auto stats = texec::ContextPhaseParams({}, mRequestId, state.release(), std::nullopt);
        request.setContextPhaseParams(std::move(stats));
        return std::make_unique<LlmRequest>(mRequestId++, std::move(request));
    }

    void addRequestAndTransportCache(std::shared_ptr<LlmRequest> const& llmRequest)
    {
        auto constexpr beamIdx{0};
        auto constexpr beamWidth{1};
        mManager->addSequence(llmRequest->mRequestId, llmRequest->getNumTokens(beamIdx), beamWidth, llmRequest);
        if (isSender)
        {
            auto blockRange = BlockRange(*mManager, llmRequest->mRequestId, beamIdx, 0);
            for (auto& block : blockRange)
            {
                // fill cache with tokens (= request length), for reuse test
                TLLM_CUDA_CHECK(cudaMemset(block.data(), llmRequest->getPromptLen(), block.getSizeInBytes()));
            }
            mFutures.emplace_back(mResponder->respondAndSendAsync(*llmRequest));
        }
        else
        {
            auto future = mRequester->requestAndReceiveAsync(*llmRequest);
            future.get();
            TLLM_CUDA_CHECK(cudaDeviceSynchronize());
            auto blockRange = BlockRange(*mManager, llmRequest->mRequestId, beamIdx, 0);
            for (auto& block : blockRange)
            {
                std::vector<uint8_t> bytes(block.getSizeInBytes());
                TLLM_CUDA_CHECK(cudaMemcpy(bytes.data(), block.data(), block.getSizeInBytes(), cudaMemcpyDeviceToHost));
                EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(),
                    [&llmRequest](uint8_t i) { return i == llmRequest->getPromptLen() & 0xff; }));
            }
        }
    }

    bool isSender{false};
    tensorrt_llm::mpi::MpiComm const* mComm;
    SizeType32 mWorldSize{0}, mlocalRank{0};
    LlmRequest::RequestIdType mRequestId{0};
    SizeType32 mMaxNumSequences{};
    std::unique_ptr<KVCacheManager> mManager;
    std::unique_ptr<DataResponder> mResponder;
    std::unique_ptr<DataRequester> mRequester;
    std::unique_ptr<texec::kv_cache::CacheState> mCacheState;
    std::vector<std::future<void>> mFutures;
    std::unique_ptr<texec::kv_cache::ConnectionManager> mConnectionManager;
};

TEST_F(SymmetricalCacheTest, SimpleTest)
{
    auto worldSize = setUpCommunicator();
    if (worldSize != 2)
    {
        GTEST_SKIP() << "mpirun 2 processes is required to run this test.";
    }
    setUpCacheManager();
    setUpCacheTransceiver();
    std::vector<std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest>> requests;

    for (auto len : {10, 20, 30})
    {
        requests.emplace_back(makeLlmRequest(len));
        addRequestAndTransportCache(requests.back());
    }
    for (auto& future : mFutures)
    {
        future.get();
    }
    mFutures.clear();
    for (auto& request : requests)
    {
        mManager->removeSequence(request->mRequestId, request);
    }
    requests.clear();

    // test reuse
    for (auto len : {10, 20, 30})
    {
        requests.emplace_back(makeLlmRequest(len));
        addRequestAndTransportCache(requests.back());
    }
    for (auto& future : mFutures)
    {
        future.get();
    }
}

#if ENABLE_MULTI_DEVICE

using AsymmetricTestParam
    = std::tuple<int, int, int, int, int, int, int, int, nvinfer1::DataType, int, bool, bool, bool>;

class AsymmetricalCacheTest : public ::testing::TestWithParam<AsymmetricTestParam>
{

protected:
    void SetUp() override {}

    void TearDown() override {}

    void setUpCommunicator(int contextTp, int contextPp, int genTp, int genPp, bool isMLA = false,
        bool contextDP = false, bool generationDP = false)
    {
#if ENABLE_MULTI_DEVICE
        tensorrt_llm::mpi::initialize(tensorrt_llm::mpi::MpiThreadSupport::THREAD_MULTIPLE);

        if (tensorrt_llm::mpi::MpiComm::world().getSize() != 8)
        {
            GTEST_SKIP() << "mpirun with procs=8  is required to run this test.";
        }
        int worldSize = tensorrt_llm::mpi::MpiComm::world().getSize();
        int worldRank = tensorrt_llm::mpi::MpiComm::world().getRank();
        tensorrt_llm::mpi::MpiComm::world().barrier();
        int contextRanks = contextTp * contextPp;
        int genRanks = genTp * genPp;
        int nprocs = (contextRanks + genRanks);

        mIsContext = false;
        mIsGeneration = false;
        mParticipatingComm = tensorrt_llm::mpi::MpiComm::world().split(static_cast<int>(worldRank < nprocs), worldRank);
        tensorrt_llm::mpi::MpiComm::setSession(
            tensorrt_llm::mpi::MpiComm::world().split(static_cast<int>(worldRank < nprocs), worldRank));

        mIsContext = worldRank < contextRanks;
        mIsGeneration = (worldRank >= contextRanks && worldRank < (contextRanks + genRanks));
        if (worldRank >= nprocs)
        {
            return;
        }
        TLLM_LOG_INFO("Run cacheTransceiverTest for ContextTp: %d, ContextPp: %d, GenTp: %d, GenPp:%d", contextTp,
            contextPp, genTp, genPp);
        mComm = std::addressof(mParticipatingComm);

        mWorldSize = mComm->getSize();
        mRank = mComm->getRank();

        {
            mIsContext = mRank < contextRanks;
            mIsGeneration = (mRank >= contextRanks && mRank < (contextRanks + genRanks));
            mRankInInstance = mIsContext ? mRank : (mRank - contextRanks);
            mSizeInInstance = mIsContext ? (contextTp * contextPp) : (genTp * genPp);
            int color = 0;
            if (mIsGeneration)
            {
                color = 1;
            }
            if (mIsContext)
            {
                color = 2;
            }
            auto sessionComm = mComm->split(static_cast<int>(color), mComm->getRank());

            if (mIsContext)
            {
                mTpSize = contextTp;
                mPpSize = contextPp;
            }
            if (mIsGeneration)
            {
                mTpSize = genTp;
                mPpSize = genPp;
            }

            mTpRank = mRankInInstance % mTpSize;
            mPpRank = mRankInInstance / mTpSize;
            mContextRankSize = contextRanks;
            mGenRankSize = genRanks;
            mContextTpSize = contextTp;
            mContextPpSize = contextPp;

            EXPECT_EQ((sessionComm.getRank()), mRankInInstance);
            EXPECT_EQ(sessionComm.getSize(), mSizeInInstance);
            mContextDP = contextDP;
            mGenerationDP = generationDP;
            mIsMLA = isMLA;
            tensorrt_llm::mpi::MpiComm::setSession(std::move(sessionComm));
        }
#else
        GTEST_SKIP() << "ENABLE_MULTI_DEVICE  is required to run this test.";

#endif
    }

    void setUpCacheManager(int numLayers, int numHeads, int sizePerHead, int tokensPerBlock,
        nvinfer1::DataType dataType, int kvFactor = 2, bool isMLA = false, bool enableDPAttention = false)
    {

        if (!(mIsContext || mIsGeneration))
        {
            return;
        }

        ASSERT_EQ(numLayers % mPpSize, 0);
        if (!isMLA)
        {
            ASSERT_EQ(numHeads % mTpSize, 0);
        }
        else
        {
            ASSERT_EQ(numHeads, 1);
        }
        int numHeadsPerRank = numHeads / mTpSize;
        if (isMLA || enableDPAttention)
        {
            numHeadsPerRank = numHeads;
        }
        auto hiddenSize = numHeadsPerRank * sizePerHead;
        auto maxBlocksPerSeq = 10;
        auto maxBeamWidth = 1;
        auto constexpr sinkTokenLength = 0;
        mMaxNumSequences = 8;
        auto const stream = std::make_shared<tr::CudaStream>();

        auto maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
        auto maxAttentionWindow = maxNumTokens;
        auto constexpr temporaryAttentionWindow = 0;
        auto inputLength = maxNumTokens - tokensPerBlock - 1;
        auto numSharedBlocks = inputLength / tokensPerBlock;
        auto numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

        auto totalNumBlocks = mMaxNumSequences * numBlocksPerSeq;
        auto constexpr blocksInSecondaryPool = 0;

        auto constexpr enableBlockReuse = true;
        auto constexpr onboardBlocks = true;
        CacheType cacheType = CacheType::kSELF;
        if (kvFactor == 1)
        {
            auto cacheType = CacheType::kSELFKONLY;
        }
        TLLM_CHECK(kvFactor == 2 || kvFactor == 1);
        int DPrank = 0;
        int DPsize = 0;
        if (mIsContext)
        {
            enableDPAttention = mContextDP;
            DPrank = mTpRank; // need to be changed in making the llmRequest
            DPsize = mTpSize;
        }
        if (mIsGeneration)
        {
            enableDPAttention = mGenerationDP;
            DPrank = mTpRank;
            DPsize = mTpSize;
        }

        int numHeadsPerRankForContext = numHeads / mContextTpSize;
        if (isMLA || mContextDP)
        {
            numHeadsPerRankForContext = numHeads;
        }
        mManager = std::make_unique<KVCacheManager>(numLayers / mPpSize, numHeadsPerRank, sizePerHead, tokensPerBlock,
            totalNumBlocks, blocksInSecondaryPool, mMaxNumSequences, maxBeamWidth,
            std::vector<SizeType32>{maxAttentionWindow}, temporaryAttentionWindow, sinkTokenLength, stream,
            std::nullopt, enableBlockReuse, onboardBlocks, cacheType, std::nullopt, nullptr, true);
        texec::kv_cache::CacheState::AttentionType attentionType = isMLA
            ? texec::kv_cache::CacheState::AttentionType::kMLA
            : texec::kv_cache::CacheState::AttentionType::kDEFAULT;
        mCacheState = std::make_unique<texec::kv_cache::CacheState>(numLayers, numHeadsPerRank, sizePerHead,
            tokensPerBlock, mTpSize, mPpSize, dataType, attentionType, kvFactor, enableDPAttention, DPrank, DPsize);
        mContextCacheState = std::make_unique<texec::kv_cache::CacheState>(numLayers, numHeadsPerRankForContext,
            sizePerHead, tokensPerBlock, mContextTpSize, mContextPpSize, dataType, attentionType, kvFactor, mContextDP,
            DPrank, mContextTpSize);

        // UVM seems to be incompatible with MPI, and it is continuing to investigate.
        bool constexpr useUvm = false;
        mManager->allocatePools(dataType, useUvm);
    }

    void setUpCacheTransceiver()
    {
        if (!(mIsContext || mIsGeneration))
        {
            return;
        }
        else if (tensorrt_llm::common::getEnvUseMPIKvCache())
        {
            TLLM_LOG_INFO("Enable MPI KV cache transport.");
            mConnectionManager = std::make_unique<texec::kv_cache::MpiConnectionManager>(mComm);

            if (mIsContext)
            {
                mResponder = mIsMLA
                    ? std::make_unique<DataResponder>(std::make_unique<DataSenderImpl>(mConnectionManager.get(),
                        *mCacheState, mRankInInstance, std::make_unique<MLACacheFormatter>(mManager.get())))
                    : std::make_unique<DataResponder>(std::make_unique<DataSenderImpl>(mConnectionManager.get(),
                        *mCacheState, mRankInInstance, std::make_unique<CacheFormatter>(mManager.get())));
            }
            else
            {
                mRequester = mIsMLA
                    ? std::make_unique<DataRequester>(std::make_unique<DataReceiverImpl>(mConnectionManager.get(),
                        *mCacheState, mRankInInstance, std::make_unique<MLACacheFormatter>(mManager.get())))
                    : std::make_unique<DataRequester>(std::make_unique<DataReceiverImpl>(mConnectionManager.get(),
                        *mCacheState, mRankInInstance, std::make_unique<CacheFormatter>(mManager.get())));
            }

            std::vector<int> contextRankVec(mContextRankSize);
            for (int i = 0; i < contextRankVec.size(); i++)
            {
                contextRankVec[i] = i;
            }
            mContextCommState = std::make_unique<tensorrt_llm::executor::kv_cache::CommState>(contextRankVec);
        }
        else
        {
            TLLM_CHECK(false);
        }
    }

    auto makeLlmRequest(SizeType32 length)
    {

        constexpr SizeType32 maxNewTokens{1};
        texec::Request request{VecTokens(length, length), maxNewTokens};

        auto state = std::make_unique<texec::DataTransceiverState>();

        TLLM_CHECK(mContextCommState);
        state->setCommState(texec::kv_cache::CommState{*mContextCommState});
        state->setCacheState(*mContextCacheState);
        auto stats = texec::ContextPhaseParams({}, mRequestId, state.release(), std::nullopt);
        request.setContextPhaseParams(std::move(stats));
        return std::make_unique<LlmRequest>(mRequestId++, std::move(request));
    }

    auto makeLlmRequestWithDP(SizeType32 length, LlmRequest::RequestIdType requestId, int contextDpRank)
    {
        constexpr SizeType32 maxNewTokens{1};
        texec::Request request{VecTokens(length), maxNewTokens};

        auto state = std::make_unique<texec::DataTransceiverState>();
        state->setCommState(texec::kv_cache::CommState{*mContextCommState});
        texec::kv_cache::CacheState cacheState{mContextCacheState->getModelConfig().mNbKvHeadsPerLayer,
            mContextCacheState->getModelConfig().mSizePerHead, mContextCacheState->getModelConfig().mTokensPerBlock,
            mContextCacheState->getParallelConfig().mTensorParallelism,
            mContextCacheState->getParallelConfig().mPipelineParallelism, mContextCacheState->getDataType(),
            mContextCacheState->getAttentionConfig().mAttentionType, mContextCacheState->getAttentionConfig().mKvFactor,
            mContextCacheState->getParallelConfig().mEnableAttenionDP, contextDpRank,
            mContextCacheState->getParallelConfig().mTensorParallelism};
        state->setCacheState(cacheState);
        auto stats = texec::ContextPhaseParams({}, requestId, state.release(), std::nullopt);
        request.setContextPhaseParams(std::move(stats));
        return std::make_unique<LlmRequest>(requestId, std::move(request));
    }

    std::future<void> addRequestAndTransportCacheForContext(std::shared_ptr<LlmRequest> const& llmRequest)
    {
        auto constexpr beamIdx{0};
        auto constexpr beamWidth{1};
        mManager->addSequence(llmRequest->mRequestId, llmRequest->getNumTokens(beamIdx), beamWidth, llmRequest);
        auto blockRange = BlockRange(*mManager, llmRequest->mRequestId, beamIdx, 0);
        int blockIdx = 0;
        for (auto& block : blockRange)
        {
            fillBlockData(block, blockIdx, llmRequest->getPromptLen());
            blockIdx++;
        }
        mManager->getBlockManager().getBufferManager().getStream().synchronize();
        auto future = mResponder->respondAndSendAsync(*llmRequest);
        return future;
    }

    std::future<void> addRequestAndTransportCacheForGeneration(std::shared_ptr<LlmRequest> const& llmRequest)
    {
        auto constexpr beamIdx{0};
        auto constexpr beamWidth{1};
        mManager->addSequence(llmRequest->mRequestId, llmRequest->getNumTokens(beamIdx), beamWidth, llmRequest);

        return mRequester->requestAndReceiveAsync(*llmRequest);
    }

    void generationVerifyKVCache(std::shared_ptr<LlmRequest> const& llmRequest)
    {
        auto constexpr beamIdx{0};
        auto constexpr beamWidth{1};
        int blockIdx = 0;

        TLLM_CUDA_CHECK(cudaDeviceSynchronize());

        auto blockRange = BlockRange(*mManager, llmRequest->mRequestId, beamIdx, 0);
        for (auto& block : blockRange)
        {
            verifyBlockData(block, blockIdx, llmRequest->getPromptLen());
            blockIdx++;
        }
    }

    void fillBlockData(tensorrt_llm::runtime::ITensor& blockData, int blockId, size_t initial)
    {
        auto hostTensor
            = mManager->getBlockManager().getBufferManager().cpu(blockData.getShape(), blockData.getDataType());
        int layerSizePerRank = mCacheState->getModelConfig().mNbKvHeadsPerLayer.size() / mPpSize;
        int startLayerId = layerSizePerRank * mPpRank;
        int headSizePerRank = mCacheState->getModelConfig().mNbKvHeadsPerLayer.at(0);
        int startHeadId = headSizePerRank * mTpRank;
        bool enableDP = mCacheState->getParallelConfig().mEnableAttenionDP;
        if (mIsMLA || enableDP)
        {
            startHeadId = 0;
        }
        int kvFactor = mCacheState->getAttentionConfig().mKvFactor;
        int tokensPerBlock = mCacheState->getModelConfig().mTokensPerBlock;
        int startTokenId = blockId * tokensPerBlock;
        int sizePerHead = mCacheState->getModelConfig().mSizePerHead;
        auto dataTypeSize = tensorrt_llm::common::getDTypeSize(blockData.getDataType());

        for (int layerId = 0; layerId < layerSizePerRank; layerId++)
        {
            for (int headId = 0; headId < headSizePerRank; headId++)
            {
                for (int tokenId = 0; tokenId < tokensPerBlock; tokenId++)
                {
                    for (int hiddenId = 0; hiddenId < sizePerHead; hiddenId++)
                    {
                        size_t keyIndex = layerId * (kvFactor * headSizePerRank * tokensPerBlock * sizePerHead)
                            + headId * (tokensPerBlock * sizePerHead) + tokenId * sizePerHead + hiddenId;
                        size_t valueIndex
                            = keyIndex + static_cast<size_t>(headSizePerRank * tokensPerBlock * sizePerHead);

                        std::visit(
                            [&](auto generateValue)
                            {
                                using ValueType = decltype(generateValue);
                                auto* dataPtr = static_cast<ValueType*>(hostTensor->data(keyIndex));
                                *dataPtr = generateValue;
                            },
                            generateExpectedValue(initial, tokenId + startTokenId, layerId + startLayerId,
                                headId + startHeadId, hiddenId, true, blockData.getDataType()));
                        if (kvFactor == 2)
                        {
                            std::visit(
                                [&](auto generateValue)
                                {
                                    using ValueType = decltype(generateValue);
                                    auto* dataPtr = static_cast<ValueType*>(hostTensor->data(valueIndex));
                                    *dataPtr = generateValue;
                                },
                                generateExpectedValue(initial, tokenId + startTokenId, layerId + startLayerId,
                                    headId + startHeadId, hiddenId, false, blockData.getDataType()));
                        }
                    }
                }
            }
        }
        mManager->getBlockManager().getBufferManager().copy(*hostTensor, blockData);
    }

    void verifyBlockData(tensorrt_llm::runtime::ITensor& blockData, int blockId, size_t initial)
    {
        auto hostTensor
            = mManager->getBlockManager().getBufferManager().cpu(blockData.getShape(), blockData.getDataType());
        int layerSizePerRank = mCacheState->getModelConfig().mNbKvHeadsPerLayer.size() / mPpSize;
        int startLayerId = layerSizePerRank * mPpRank;
        int headSizePerRank = mCacheState->getModelConfig().mNbKvHeadsPerLayer.at(0);
        int startHeadId = headSizePerRank * mTpRank;
        bool enableDP = mCacheState->getParallelConfig().mEnableAttenionDP;
        if (mIsMLA || enableDP)
        {
            startHeadId = 0;
        }
        int kvFactor = mCacheState->getAttentionConfig().mKvFactor;
        int tokensPerBlock = mCacheState->getModelConfig().mTokensPerBlock;
        int startTokenId = blockId * tokensPerBlock;
        int sizePerHead = mCacheState->getModelConfig().mSizePerHead;

        mManager->getBlockManager().getBufferManager().copy(blockData, *hostTensor);
        mManager->getBlockManager().getBufferManager().getStream().synchronize();

        for (int layerId = 0; layerId < layerSizePerRank; layerId++)
        {
            for (int headId = 0; headId < headSizePerRank; headId++)
            {
                for (int tokenId = 0; tokenId < tokensPerBlock; tokenId++)
                {
                    for (int hiddenId = 0; hiddenId < sizePerHead; hiddenId++)
                    {
                        size_t keyIndex = layerId * (kvFactor * headSizePerRank * tokensPerBlock * sizePerHead)
                            + headId * (tokensPerBlock * sizePerHead) + tokenId * sizePerHead + hiddenId;
                        size_t valueIndex
                            = keyIndex + static_cast<size_t>(headSizePerRank * tokensPerBlock * sizePerHead);

                        std::visit(
                            [&](auto generateValue)
                            {
                                using ValueType = decltype(generateValue);
                                auto* dataPtr = static_cast<ValueType*>(hostTensor->data(keyIndex));
                                EXPECT_EQ(*dataPtr, generateValue);
                            },
                            generateExpectedValue(initial, tokenId + startTokenId, layerId + startLayerId,
                                headId + startHeadId, hiddenId, true, blockData.getDataType()));
                        if (kvFactor == 2)
                        {
                            std::visit(
                                [&](auto generateValue)
                                {
                                    using ValueType = decltype(generateValue);
                                    auto* dataPtr = static_cast<ValueType*>(hostTensor->data(valueIndex));
                                    EXPECT_EQ(*dataPtr, generateValue);
                                },
                                generateExpectedValue(initial, tokenId + startTokenId, layerId + startLayerId,
                                    headId + startHeadId, hiddenId, false, blockData.getDataType()));
                        }
                    }
                }
            }
        }
    }

    std::variant<double, float, int16_t, int8_t> generateExpectedValue(
        size_t initial, int tokenId, int layerId, int headId, int hiddenId, bool key, nvinfer1::DataType dataType)
    {

        size_t seed = 0;
        std::size_t hashValue = std::hash<size_t>{}(initial);
        std::hash<int> hasher{};
        seed ^= hashValue + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hasher(tokenId) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hasher(layerId) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hasher(headId) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hasher(hiddenId) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed += key;
        generator.seed(seed);
        std::uniform_real_distribution<double> dis(-100.0f, 100.0f);
        double value = dis(generator);
        auto dataTypeSize = tensorrt_llm::common::getDTypeSize(dataType);
        switch (dataTypeSize)
        {
        case 8: return value; break;
        case 4: return static_cast<float>(value); break;
        case 2: return static_cast<int16_t>(value); break;
        case 1: return static_cast<int8_t>(value); break;
        default: TLLM_CHECK_WITH_INFO(false, "generateExpectedValue only support dataTypeSize in [8,4,2,1]"); break;
        };
        return 0.F;
    }

    bool mIsContext{false};
    bool mIsGeneration{false};
    tensorrt_llm::mpi::MpiComm const* mComm;
    tensorrt_llm::mpi::MpiComm mParticipatingComm{nullptr, false};
    SizeType32 mWorldSize{0}, mRank{0}, mRankInInstance{0};
    SizeType32 mSizeInInstance{0}, mTpRank{0}, mPpRank{0}, mTpSize{0}, mPpSize{0}, mContextRankSize{0}, mGenRankSize{0},
        mContextTpSize{0}, mContextPpSize{0};
    LlmRequest::RequestIdType mRequestId{0};
    bool mContextDP{false};
    bool mGenerationDP{false};
    bool mIsMLA{false};
    SizeType32 mMaxNumSequences{};
    std::unique_ptr<KVCacheManager> mManager;
    std::unique_ptr<DataResponder> mResponder;
    std::unique_ptr<DataRequester> mRequester;
    std::unique_ptr<texec::kv_cache::CacheState> mCacheState;
    std::unique_ptr<texec::kv_cache::CacheState> mContextCacheState;
    std::unique_ptr<texec::kv_cache::CommState> mContextCommState;
    std::unique_ptr<texec::kv_cache::ConnectionManager> mConnectionManager;
    std::mt19937 generator;
};

TEST_P(AsymmetricalCacheTest, TestCase)
{
    if (!(tensorrt_llm::common::getEnvUseUCXKvCache()))
    {
        setenv("UCX_TLS", "^cuda_ipc", 1); // disable cuda_ipc for testing for mpi
    }
    AsymmetricTestParam param = GetParam();
    int contextTp = std::get<0>(param);
    int contextPp = std::get<1>(param);
    int genTp = std::get<2>(param);
    int genPp = std::get<3>(param);
    int numLayers = std::get<4>(param);
    int numHeads = std::get<5>(param);
    int sizePerHead = std::get<6>(param);
    int tokensPerBlock = std::get<7>(param);
    nvinfer1::DataType dataType = std::get<8>(param);

    int kvFactor = std::get<9>(param);
    bool isMLA = std::get<10>(param);
    bool contextDP = std::get<11>(param);
    bool generationDP = std::get<12>(param);

    setUpCommunicator(contextTp, contextPp, genTp, genPp, isMLA, contextDP, generationDP);

    if (mIsContext || mIsGeneration)
    {
        setUpCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, dataType, kvFactor, isMLA);
        setUpCacheTransceiver();
        std::vector<std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest>> requests;

        // the second loop is for cache reuse
        for (int i = 0; i < 2; i++)
        {
            for (auto len : {30, 10, 60, 30, 60, 10})
            {
                requests.emplace_back(makeLlmRequest(len));
            }

            if (mIsContext)
            {
                std::vector<std::future<void>> contextFutures;
                for (auto&& request : requests)
                {
                    contextFutures.push_back(addRequestAndTransportCacheForContext(request));
                }
                mComm->barrier();
                for (auto&& cfuture : contextFutures)
                {
                    cfuture.get();
                }
            }
            else
            {
                std::vector<std::future<void>> generationFutures;
                mComm->barrier();
                for (auto&& request : requests)
                {
                    generationFutures.push_back(addRequestAndTransportCacheForGeneration(request));
                }

                for (auto&& gfuture : generationFutures)
                {
                    gfuture.get();
                }
                for (auto&& request : requests)
                {
                    generationVerifyKVCache(request);
                }
            }
            for (auto&& request : requests)
            {
                mManager->removeSequence(request->mRequestId, request);
            }
            requests.clear();
            mComm->barrier();
        }
    }
    tensorrt_llm::mpi::MpiComm::world().barrier();
}

class AsymmetricalCacheTestWithDP : public AsymmetricalCacheTest
{
};

TEST_P(AsymmetricalCacheTestWithDP, TestCase)
{
    if (!(tensorrt_llm::common::getEnvUseUCXKvCache()))
    {
        setenv("UCX_TLS", "^cuda_ipc", 1); // disable cuda_ipc for testing for mpi
    }
    AsymmetricTestParam param = GetParam();
    int contextTp = std::get<0>(param);
    int contextPp = std::get<1>(param);
    int genTp = std::get<2>(param);
    int genPp = std::get<3>(param);
    int numLayers = std::get<4>(param);
    int numHeads = std::get<5>(param);
    int sizePerHead = std::get<6>(param);
    int tokensPerBlock = std::get<7>(param);
    nvinfer1::DataType dataType = std::get<8>(param);

    int kvFactor = std::get<9>(param);
    bool isMLA = std::get<10>(param);
    bool contextDP = std::get<11>(param);
    bool generationDP = std::get<12>(param);

    setUpCommunicator(contextTp, contextPp, genTp, genPp, isMLA, contextDP, generationDP);

    if (mIsContext || mIsGeneration)
    {
        bool enableDP = mIsContext ? contextDP : generationDP;
        setUpCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, dataType, kvFactor, isMLA, enableDP);
        setUpCacheTransceiver();
        std::vector<std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest>> requests;
        int requestId = 0;
        for (auto len : {30, 10, 60, 30, 60, 10})
        {
            requests.emplace_back(makeLlmRequestWithDP(len, requestId++, requestId % contextTp));
        }
        std::vector<std::future<void>> contextFutures;
        std::vector<std::future<void>> generationFutures;
        std::vector<std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest>> generationRequests;

        if (mIsContext)
        {
            std::vector<std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest>> contextRequests;
            if (contextDP)
            {
                for (int i = 0; i < requests.size(); i++)
                {
                    if (i % mTpSize == mTpRank)
                    {
                        // round robin
                        contextRequests.push_back(requests[i]);
                    }
                }
            }
            else
            {
                contextRequests = requests;
            }
            for (auto&& request : contextRequests)
            {
                contextFutures.push_back(std::move(addRequestAndTransportCacheForContext(request)));
            }
            mComm->barrier();
        }
        else
        {
            if (generationDP)
            {
                for (int i = 0; i < requests.size(); i++)
                {
                    if (i % mTpSize == mTpRank)
                    {
                        generationRequests.push_back(requests[i]);
                    }
                }
            }
            else
            {
                generationRequests = requests;
            }
            mComm->barrier();
            for (auto&& request : generationRequests)
            {
                generationFutures.push_back(std::move(addRequestAndTransportCacheForGeneration(request)));
            }
        }
        if (mIsContext)
        {
            for (auto&& cfuture : contextFutures)
            {
                cfuture.get();
            }
        }
        else
        {
            for (auto&& gfuture : generationFutures)
            {
                gfuture.get();
            }
            for (auto&& request : generationRequests)
            {
                generationVerifyKVCache(request);
            }
        }
        mComm->barrier();
    }
    tensorrt_llm::mpi::MpiComm::world().barrier();
}

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTest0, AsymmetricalCacheTest,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2),
        testing::Values(4), testing::Values(4), testing::Values(4), testing::Values(16),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(false), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTest1, AsymmetricalCacheTest,
    testing::Combine(testing::Values(4), testing::Values(1), testing::Values(1), testing::Values(4), testing::Values(4),
        testing::Values(4), testing::Values(4), testing::Values(8),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(false), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTest2, AsymmetricalCacheTest,
    testing::Combine(testing::Values(1), testing::Values(2), testing::Values(1), testing::Values(1, 4),
        testing::Values(16), testing::Values(16), testing::Values(4), testing::Values(8),
        testing::Values(nvinfer1::DataType::kFLOAT), testing::Values(2), testing::Values(false), testing::Values(false),
        testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTest0ForMLA, AsymmetricalCacheTest,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2),
        testing::Values(4), testing::Values(1), testing::Values(4), testing::Values(16),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(1),
        testing::Values(true), testing::Values(false), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTest1ForMLA, AsymmetricalCacheTest,
    testing::Combine(testing::Values(4), testing::Values(1), testing::Values(1), testing::Values(4), testing::Values(4),
        testing::Values(1), testing::Values(4), testing::Values(8),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(1),
        testing::Values(true), testing::Values(false), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForMLA1, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2),
        testing::Values(4), testing::Values(1), testing::Values(4), testing::Values(16),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(1),
        testing::Values(true), testing::Values(true), testing::Values(true)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForMLA2, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2),
        testing::Values(4), testing::Values(1), testing::Values(4), testing::Values(16),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(1),
        testing::Values(true), testing::Values(true), testing::Values(false)));
INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForMLA3, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2),
        testing::Values(4), testing::Values(1), testing::Values(4), testing::Values(16),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(1),
        testing::Values(true), testing::Values(false), testing::Values(true)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForNoMLA, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2),
        testing::Values(4), testing::Values(4), testing::Values(4), testing::Values(16),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(true), testing::Values(true)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForNoMLA1, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2),
        testing::Values(4), testing::Values(4), testing::Values(4), testing::Values(16),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(true), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForNoMLA2, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2), testing::Values(1, 2),
        testing::Values(4), testing::Values(4), testing::Values(4), testing::Values(16),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(false), testing::Values(true)));
#endif

TEST(targetTest, CacheStateNODP)
{

    int const numLayers = 16;
    int const numHeads = 2;
    int const sizePerHead = 64;
    int const tokensPerBlock = 64;
    auto const dataType = nvinfer1::DataType::kFLOAT;
    bool const isMLA = true;
    int const kvFactor = 2;

    int contextPP = 2;
    int contextTP = 4;
    int genPP = 2;
    int genTP = 2;
    bool const contextEnableDP = false;
    bool const genEnableDP = false;

    auto const verifyContext = [&](int contextRank, std::vector<int> const& expectRanks, int expectPPDomain,
                                   int expectTPDomain, bool expectNeedSend)
    {
        auto attentionType = isMLA ? texec::kv_cache::CacheState::AttentionType::kMLA
                                   : texec::kv_cache::CacheState::AttentionType::kDEFAULT;
        auto const contextCache = tensorrt_llm::executor::kv_cache::CacheState{numLayers, numHeads, sizePerHead,
            tokensPerBlock, contextTP, contextPP, dataType, attentionType, kvFactor, contextEnableDP, 0, 0};

        auto const genCache = tensorrt_llm::executor::kv_cache::CacheState{numLayers, numHeads, sizePerHead,
            tokensPerBlock, genTP, genPP, dataType, attentionType, kvFactor, genEnableDP, 0, 0};

        auto const contextTragetInfo
            = tensorrt_llm::executor::kv_cache::TargetRanksInfoForDP(genCache, contextCache, contextRank);

        EXPECT_EQ(expectRanks, contextTragetInfo.mIRanks);
        EXPECT_EQ(expectPPDomain, contextTragetInfo.mDomainPPSize);
        EXPECT_EQ(expectTPDomain, contextTragetInfo.mDomainTPSize);
        EXPECT_EQ(expectNeedSend, MLACacheFormatter::needSendCache(contextCache, genCache, contextRank));
    };

    verifyContext(
        /*contextRank*/ 0, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1, /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 1, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1, /*expectNeedSend*/ false);
    verifyContext(
        /*contextRank*/ 2, /*expectRanks*/ {1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1, /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 3, /*expectRanks*/ {1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1, /*expectNeedSend*/ false);
    verifyContext(
        /*contextRank*/ 4, /*expectRanks*/ {2}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1, /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 5, /*expectRanks*/ {2}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1, /*expectNeedSend*/ false);
    verifyContext(
        /*contextRank*/ 6, /*expectRanks*/ {3}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1, /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 7, /*expectRanks*/ {3}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1, /*expectNeedSend*/ false);

    contextTP = 2;
    genTP = 4;

    verifyContext(
        /*contextRank*/ 0, /*expectRanks*/ {0, 1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 2, /*expectNeedSend*/ true);
    verifyContext(/*contextRank*/ 1, /*expectRanks*/ {2, 3}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 2,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 2, /*expectRanks*/ {4, 5}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 2, /*expectNeedSend*/ true);
    verifyContext(/*contextRank*/ 3, /*expectRanks*/ {6, 7}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 2,
        /*expectNeedSend*/ true);
    contextPP = 1;
    verifyContext(
        /*contextRank*/ 0, /*expectRanks*/ {0, 4, 1, 5}, /*expectPPDomain*/ 2, /*expectTPDomain*/ 2,
        /*expectNeedSend*/ true);
    verifyContext(/*contextRank*/ 1, /*expectRanks*/ {2, 6, 3, 7}, /*expectPPDomain*/ 2, /*expectTPDomain*/ 2,
        /*expectNeedSend*/ true);
}

TEST(targetTest, CacheStateContextDP)
{

    int const numLayers = 16;
    int const numHeads = 2;
    int const sizePerHead = 64;
    int const tokensPerBlock = 64;
    auto const dataType = nvinfer1::DataType::kFLOAT;
    bool const isMLA = true;
    int const kvFactor = 2;

    int contextPP = 1;
    int contextTP = 4;
    int genPP = 1;
    int genTP = 2;
    bool contextEnableDP = true;
    bool genEnableDP = true;

    auto const verifyContext = [&](int contextRank, int generationRank, std::vector<int> const& expectRanks,
                                   int expectPPDomain, int expectTPDomain, bool expectNeedSend)
    {
        int contextDPRank = contextRank % contextTP;
        int generationDPRank = generationRank % genTP;
        auto attentionType = isMLA ? texec::kv_cache::CacheState::AttentionType::kMLA
                                   : texec::kv_cache::CacheState::AttentionType::kDEFAULT;

        auto const contextCache
            = tensorrt_llm::executor::kv_cache::CacheState{numLayers, numHeads, sizePerHead, tokensPerBlock, contextTP,
                contextPP, dataType, attentionType, kvFactor, contextEnableDP, contextDPRank, contextTP};

        auto const genCache = tensorrt_llm::executor::kv_cache::CacheState{numLayers, numHeads, sizePerHead,
            tokensPerBlock, genTP, genPP, dataType, attentionType, kvFactor, genEnableDP, generationDPRank, genTP};

        auto const contextTragetInfo
            = tensorrt_llm::executor::kv_cache::TargetRanksInfoForDP(genCache, contextCache, contextRank);

        EXPECT_EQ(expectRanks, contextTragetInfo.mIRanks);
        EXPECT_EQ(expectPPDomain, contextTragetInfo.mDomainPPSize);
        EXPECT_EQ(expectTPDomain, contextTragetInfo.mDomainTPSize);
        EXPECT_EQ(expectNeedSend, MLACacheFormatter::needSendCache(contextCache, genCache, contextRank));
    };

    verifyContext(
        /*contextRank*/ 0, /*generationRank*/ 0, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 0, /*generationRank*/ 1, /*expectRanks*/ {1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 1, /*generationRank*/ 0, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 1, /*generationRank*/ 1, /*expectRanks*/ {1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 2, /*generationRank*/ 0, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 2, /*generationRank*/ 1, /*expectRanks*/ {1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 3, /*generationRank*/ 0, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 3, /*generationRank*/ 1, /*expectRanks*/ {1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ true);

    contextEnableDP = false;
    verifyContext(
        /*contextRank*/ 0, /*generationRank*/ 0, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 0, /*generationRank*/ 1, /*expectRanks*/ {1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 1, /*generationRank*/ 0, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ false);
    verifyContext(
        /*contextRank*/ 1, /*generationRank*/ 1, /*expectRanks*/ {1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ false);
    verifyContext(
        /*contextRank*/ 2, /*generationRank*/ 0, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ false);
    verifyContext(
        /*contextRank*/ 2, /*generationRank*/ 1, /*expectRanks*/ {1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ false);
    verifyContext(
        /*contextRank*/ 3, /*generationRank*/ 0, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ false);
    verifyContext(
        /*contextRank*/ 3, /*generationRank*/ 1, /*expectRanks*/ {1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ false);

    contextEnableDP = true;
    genEnableDP = false;

    verifyContext(
        /*contextRank*/ 0, /*generationRank*/ 0, /*expectRanks*/ {0, 1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 2,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 0, /*generationRank*/ 1, /*expectRanks*/ {0, 1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 2,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 1, /*generationRank*/ 0, /*expectRanks*/ {0, 1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 2,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 1, /*generationRank*/ 1, /*expectRanks*/ {0, 1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 2,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 2, /*generationRank*/ 0, /*expectRanks*/ {0, 1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 2,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 2, /*generationRank*/ 1, /*expectRanks*/ {0, 1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 2,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 3, /*generationRank*/ 0, /*expectRanks*/ {0, 1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 2,
        /*expectNeedSend*/ true);
    verifyContext(
        /*contextRank*/ 3, /*generationRank*/ 1, /*expectRanks*/ {0, 1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 2,
        /*expectNeedSend*/ true);

    contextTP = 1;
    genTP = 2;

    auto const verfiyGeneration = [&](int contextRank, int generationRank, std::vector<int> const& expectRanks,
                                      int expectPPDomain, int expectTPDomain)
    {
        int contextDPRank = contextRank % contextTP;
        int generationDPRank = generationRank % genTP;
        auto attentionType = isMLA ? texec::kv_cache::CacheState::AttentionType::kMLA
                                   : texec::kv_cache::CacheState::AttentionType::kDEFAULT;

        auto const contextCache
            = tensorrt_llm::executor::kv_cache::CacheState{numLayers, numHeads, sizePerHead, tokensPerBlock, contextTP,
                contextPP, dataType, attentionType, kvFactor, contextEnableDP, contextDPRank, contextTP};

        auto const genCache = tensorrt_llm::executor::kv_cache::CacheState{numLayers, numHeads, sizePerHead,
            tokensPerBlock, genTP, genPP, dataType, attentionType, kvFactor, genEnableDP, generationDPRank, genTP};

        auto const contextTragetInfo
            = tensorrt_llm::executor::kv_cache::TargetRanksInfoForDP(contextCache, genCache, generationRank);

        EXPECT_EQ(expectRanks, contextTragetInfo.mIRanks);
        EXPECT_EQ(expectPPDomain, contextTragetInfo.mDomainPPSize);
        EXPECT_EQ(expectTPDomain, contextTragetInfo.mDomainTPSize);
    };

    verfiyGeneration(
        /*contextRank*/ 0, /*generationRank*/ 0, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1);
    verfiyGeneration(
        /*contextRank*/ 0, /*generationRank*/ 1, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1);

    contextTP = 1;
    contextPP = 1;
    genTP = 1;
    genPP = 2;

    verfiyGeneration(
        /*contextRank*/ 0, /*generationRank*/ 0, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1);
    verfiyGeneration(
        /*contextRank*/ 0, /*generationRank*/ 1, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1);

    genEnableDP = false;
    contextEnableDP = true;

    contextTP = 2;
    contextPP = 1;
    genTP = 1;
    genPP = 1;

    verfiyGeneration(
        /*contextRank*/ 0, /*generationRank*/ 0, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1);
    verfiyGeneration(
        /*contextRank*/ 1, /*generationRank*/ 0, /*expectRanks*/ {1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1);
}
