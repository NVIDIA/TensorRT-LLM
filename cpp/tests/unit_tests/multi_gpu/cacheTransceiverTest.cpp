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

#define UCX_WRAPPER_LIB_NAME "tensorrt_llm_ucx_wrapper"

#if defined(_WIN32)
#include <windows.h>
#define dllOpen(name) LoadLibrary(name ".dll")
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) static_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name))
#else // For non-Windows platforms
#include <dlfcn.h>
#define dllOpen(name) dlopen("lib" name ".so", RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
#endif // defined(_WIN32)

#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
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
#include <tensorrt_llm/batch_manager/cacheTransBuffer.h>
#include <tensorrt_llm/batch_manager/mlaCacheFormatter.h>
#include <tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h>

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
std::mutex mDllMutex;

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
    state->setCacheState(texec::kv_cache::CacheState{10, 12, 128, 128, 8, 8, 8, {10}, nvinfer1::DataType::kFLOAT});
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
    constexpr SizeType32 contextParallelism{2};
    constexpr SizeType32 sizePerHead{hiddenSize / nbHeads};
    constexpr CacheState::AttentionType attentionType{CacheState::AttentionType::kDEFAULT};
    constexpr int kvFactor = 2;
    tr::ModelConfig modelConfig{
        vocabSize, nbAttentionLayers + nbRnnLayers, nbAttentionLayers, nbRnnLayers, nbHeads, hiddenSize, dtype};
    modelConfig.setTokensPerBlock(tokensPerBlock);
    tr::WorldConfig worldConfig{tensorParallelism, pipelineParallelism, contextParallelism};
    std::vector<SizeType32> attentionLayerNumPerPP(pipelineParallelism, nbAttentionLayers / pipelineParallelism);

    texec::kv_cache::CacheState::ModelConfig cacheStateCfg{
        modelConfig.getNumKvHeadsPerLayer(), modelConfig.getSizePerHead(), modelConfig.getTokensPerBlock()};

    texec::kv_cache::CacheState state0{
        cacheStateCfg, worldConfig, attentionLayerNumPerPP, modelConfig.getKvDataType(), attentionType, kvFactor};
    texec::kv_cache::CacheState state1{nbAttentionLayers, nbHeads, sizePerHead, tokensPerBlock, tensorParallelism,
        pipelineParallelism, contextParallelism, attentionLayerNumPerPP, dtype, attentionType, kvFactor, false, 0,
        tensorParallelism};
    EXPECT_EQ(state0, state1);
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
        auto constexpr inputLength = maxNumTokens - tokensPerBlock - 1;
        auto constexpr numSharedBlocks = inputLength / tokensPerBlock;
        auto constexpr numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

        auto totalNumBlocks = mMaxNumSequences * numBlocksPerSeq;
        auto constexpr blocksInSecondaryPool = 0;

        auto constexpr enableBlockReuse = false;
        auto constexpr onboardBlocks = true;
        auto constexpr dataType = nvinfer1::DataType::kFLOAT;

        using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;
        auto const blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};

        mManager = std::make_unique<KVCacheManager>(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow,
            mMaxNumSequences, maxBeamWidth, std::vector<BlockManager::SizeType32>{maxAttentionWindow}, std::nullopt,
            dataType, sinkTokenLength, stream, maxNumTokens, enableBlockReuse, onboardBlocks, CacheType::kSELF,
            std::nullopt, nullptr, true);
        auto attentionLayerNumPerPP = std::vector<SizeType32>{numLayers};
        mCacheState = std::make_unique<texec::kv_cache::CacheState>(
            numLayers, numHeads, sizePerHead, tokensPerBlock, 1, 1, 1, attentionLayerNumPerPP, dataType);

        if (tensorrt_llm::common::getEnvUseUCXKvCache())
        {
            std::lock_guard<std::mutex> lock(mDllMutex);
            void* WrapperLibHandle{nullptr};
            WrapperLibHandle = dllOpen(UCX_WRAPPER_LIB_NAME);
            TLLM_CHECK_WITH_INFO(WrapperLibHandle != nullptr, "UCX wrapper library is not open correctly.");
            auto load_sym = [](void* handle, char const* name)
            {
                void* ret = dllGetSym(handle, name);
                TLLM_CHECK_WITH_INFO(ret != nullptr,
                    "Unable to load UCX wrapper library symbol, possible cause is that TensorRT LLM library is not "
                    "built with UCX support, please rebuild in UCX-enabled environment.");
                return ret;
            };
            std::unique_ptr<tensorrt_llm::executor::kv_cache::ConnectionManager> (*makeUcxConnectionManager)();
            *(void**) (&makeUcxConnectionManager) = load_sym(WrapperLibHandle, "makeUcxConnectionManager");
            mConnectionManager = makeUcxConnectionManager();
            auto commState = mConnectionManager->getCommState();
            namespace su = tensorrt_llm::executor::serialize_utils;

            if (tensorrt_llm::mpi::MpiComm::world().getRank() == 0)
            {

                std::ostringstream oStream;
                su::serialize(commState, oStream);
                auto str = oStream.str();
                std::vector<char> buffer(str.begin(), str.end());
                int genRank = 1;
                int64_t bufferSize = buffer.size();
                TLLM_LOG_DEBUG(
                    tensorrt_llm::mpi::MpiComm::world().getRank(), "send bufferSize: %ld to %d", bufferSize, genRank);
                tensorrt_llm::mpi::MpiComm::world().sendRawTag(
                    &bufferSize, 1, tensorrt_llm::mpi::MpiType::kINT64, genRank, 0x1F);
                tensorrt_llm::mpi::MpiComm::world().sendRawTag(
                    buffer.data(), buffer.size(), tensorrt_llm::mpi::MpiType::kCHAR, genRank, 0x2F);
                TLLM_LOG_DEBUG(tensorrt_llm::mpi::MpiComm::world().getRank(), "send buffer to %d", genRank);
                mContextCommState = std::make_unique<tensorrt_llm::executor::kv_cache::CommState>(commState);
            }
            else
            {
                int64_t bufferSize;
                tensorrt_llm::mpi::MpiComm::world().recvRawTag(
                    &bufferSize, 1, tensorrt_llm::mpi::MpiType::kINT64, 0, 0x1F);
                TLLM_LOG_DEBUG(
                    tensorrt_llm::mpi::MpiComm::world().getRank(), "recv bufferSize: %ld from 0", bufferSize);
                std::vector<char> recvBuffer(bufferSize);
                tensorrt_llm::mpi::MpiComm::world().recvRawTag(
                    recvBuffer.data(), bufferSize, tensorrt_llm::mpi::MpiType::kCHAR, 0, 0x2F);
                TLLM_LOG_DEBUG(tensorrt_llm::mpi::MpiComm::world().getRank(), "recv buffer from 0", bufferSize);
                std::istringstream iStream(std::string(recvBuffer.begin(), recvBuffer.end()));
                su::VectorWrapBuf<char> strbuf(recvBuffer);
                std::istream is(&strbuf);
                mContextCommState = std::make_unique<tensorrt_llm::executor::kv_cache::CommState>(
                    su::deserialize<tensorrt_llm::executor::kv_cache::CommState>(is));
            }
        }
        else
        {
            mConnectionManager = std::make_unique<texec::kv_cache::MpiConnectionManager>(mComm);
            mContextCommState
                = std::make_unique<texec::kv_cache::CommState>(texec::kv_cache::CommState{std::vector<int>{0}});
        }
        // UVM seems to be incompatible with MPI, and it is continuing to investigate.
        bool constexpr useUvm = false;
        mManager->allocatePools(useUvm);
    }

    void setUpCacheTransceiver()
    {
        int maxNumTokens = 1024;
        mCacheTransBufferManager = std::make_unique<CacheTransBufferManager>(mManager.get(), maxNumTokens);
        if (isSender)
        {
            mSender = std::make_unique<CacheSender>(mConnectionManager.get(), *mCacheState, mlocalRank,
                std::make_unique<CacheFormatter>(mManager.get(), mCacheTransBufferManager.get()));
        }
        else
        {
            mRequester = std::make_unique<CacheReceiver>(mConnectionManager.get(), *mCacheState, mlocalRank,
                std::make_unique<CacheFormatter>(mManager.get(), mCacheTransBufferManager.get()));
        }
    }

    auto makeLlmRequest(SizeType32 length)
    {
        constexpr SizeType32 maxNewTokens{1};
        // create request with tokens [length, ..., length] (<length> tokens)
        texec::Request request{VecTokens(length, length), maxNewTokens};
        auto state = std::make_unique<texec::DataTransceiverState>();
        state->setCommState(*mContextCommState);
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
            auto blockRange = BlockRange::fromAllBlockIds(*mManager, llmRequest->mRequestId);
            auto const& windowSizes = blockRange.getWindowSizes();
            for (auto const& windowSize : windowSizes)
            {
                auto blockRangeForWindow = blockRange.getBlockRangeForWindow(windowSize);
                for (auto it = blockRangeForWindow.begin(); it != blockRangeForWindow.end(); ++it)
                {
                    // fill cache with tokens (= request length), for reuse test
                    TLLM_CUDA_CHECK(cudaMemset(it->data(), llmRequest->getPromptLen(), it->getSizeInBytes()));
                }
            }
            mFutures.emplace_back(mSender->sendAsync(*llmRequest));
        }
        else
        {
            auto future = mRequester->receiveAsync(*llmRequest);
            future.get();
            TLLM_CUDA_CHECK(cudaDeviceSynchronize());
            auto blockRange = BlockRange::fromAllBlockIds(*mManager, llmRequest->mRequestId);
            auto const& windowSizes = blockRange.getWindowSizes();
            for (auto const& windowSize : windowSizes)
            {
                auto blockRangeForWindow = blockRange.getBlockRangeForWindow(windowSize);
                for (auto it = blockRangeForWindow.begin(); it != blockRangeForWindow.end(); ++it)
                {
                    std::vector<uint8_t> bytes(it->getSizeInBytes());
                    TLLM_CUDA_CHECK(cudaMemcpy(bytes.data(), it->data(), it->getSizeInBytes(), cudaMemcpyDeviceToHost));
                    EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(),
                        [&llmRequest](uint8_t i) { return i == llmRequest->getPromptLen() & 0xff; }));
                }
            }
        }
    }

    bool isSender{false};
    tensorrt_llm::mpi::MpiComm const* mComm;
    SizeType32 mWorldSize{0}, mlocalRank{0};
    LlmRequest::RequestIdType mRequestId{0};
    SizeType32 mMaxNumSequences{};
    std::unique_ptr<KVCacheManager> mManager;
    std::unique_ptr<CacheTransBufferManager> mCacheTransBufferManager;
    std::unique_ptr<CacheSender> mSender;
    std::unique_ptr<CacheReceiver> mRequester;
    std::unique_ptr<texec::kv_cache::CacheState> mCacheState;
    std::unique_ptr<texec::kv_cache::CommState> mContextCommState;
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
    = std::tuple<int, int, int, int, int, int, int, int, int, int, nvinfer1::DataType, int, bool, bool, bool, bool>;

class AsymmetricalCacheTest : public ::testing::TestWithParam<AsymmetricTestParam>
{

protected:
    void SetUp() override {}

    void TearDown() override {}

    void setUpCommunicator(int contextTp, int contextPp, int contextCp, int genTp, int genPp, int genCp,
        bool isMLA = false, bool contextDP = false, bool generationDP = false)
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
        int contextRanks = contextTp * contextPp * contextCp;
        int genRanks = genTp * genPp * genCp;
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
        TLLM_LOG_INFO(
            "Run cacheTransceiverTest for ContextTp: %d, ContextPp: %d, ContextCp: %d, GenTp: %d, GenPp:%d, GenCp:%d",
            contextTp, contextPp, contextCp, genTp, genPp, genCp);
        mComm = std::addressof(mParticipatingComm);

        mWorldSize = mComm->getSize();
        mRank = mComm->getRank();

        {
            mIsContext = mRank < contextRanks;
            mIsGeneration = (mRank >= contextRanks && mRank < (contextRanks + genRanks));
            mRankInInstance = mIsContext ? mRank : (mRank - contextRanks);
            mSizeInInstance = mIsContext ? (contextTp * contextPp * contextCp) : (genTp * genPp * genCp);
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
                mCpSize = contextCp;
            }
            if (mIsGeneration)
            {
                mTpSize = genTp;
                mPpSize = genPp;
                mCpSize = genCp;
            }

            mTpRank = mRankInInstance % mTpSize;
            mPpRank = mRankInInstance / (mTpSize * mCpSize);
            mCpRank = (mRankInInstance % (mTpSize * mCpSize)) / mTpSize;
            mContextRankSize = contextRanks;
            mGenRankSize = genRanks;
            mContextTpSize = contextTp;
            mContextPpSize = contextPp;
            mContextCpSize = contextCp;

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
        nvinfer1::DataType dataType, int kvFactor = 2, bool isMLA = false, bool enableDPAttention = false,
        bool isWindow = false)
    {
        mIsWindowAttention = isWindow;

        if (!(mIsContext || mIsGeneration))
        {
            return;
        }

        // ASSERT_EQ(numLayers % mPpSize, 0);

        auto getLayerNumPPRank = [](int numLayers, int ppRank, int ppSize)
        {
            int layerNumPerPP = numLayers / ppSize;
            int layerNumExtraInPP = numLayers % ppSize;
            int layerNumInPPRank = layerNumPerPP + (ppRank < layerNumExtraInPP ? 1 : 0);
            return layerNumInPPRank;
        };

        mAttentionLayerNumPerPP = std::vector<SizeType32>(mPpSize, 0);
        for (int ppRank = 0; ppRank < mPpSize; ppRank++)
        {
            mAttentionLayerNumPerPP[ppRank] = getLayerNumPPRank(numLayers, ppRank, mPpSize);
        }
        int layerNumthisRank = getLayerNumPPRank(numLayers, mPpRank, mPpSize);

        auto contextAttentionLayerNumPerPP = std::vector<SizeType32>(mContextPpSize, 0);
        for (int ppRank = 0; ppRank < mContextPpSize; ppRank++)
        {
            contextAttentionLayerNumPerPP[ppRank] = getLayerNumPPRank(numLayers, ppRank, mContextPpSize);
        }

        if (!isMLA)
        {
            // ASSERT_EQ(numHeads % mTpSize , 0);
            ASSERT_TRUE(numHeads % mTpSize == 0 || mTpSize % numHeads == 0);
        }
        else
        {
            ASSERT_EQ(numHeads, 1);
        }
        int numHeadsPerRank = (numHeads + mTpSize - 1) / mTpSize;
        mDupHeadFactor = 1;
        if (mTpSize > numHeads)
        {
            mDupHeadFactor = mTpSize / numHeads;
            ASSERT_EQ(numHeadsPerRank, 1);
        }
        if (isMLA || enableDPAttention)
        {
            numHeadsPerRank = numHeads;
            mDupHeadFactor = 1;
        }
        auto hiddenSize = numHeadsPerRank * sizePerHead;
        auto maxBlocksPerSeq = 10;
        auto maxBeamWidth = 1;
        auto constexpr sinkTokenLength = 0;
        mMaxNumSequences = 16;
        auto const stream = std::make_shared<tr::CudaStream>();

        auto maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
        auto windowAttentionToken = 2 * tokensPerBlock;
        auto maxAttentionWindow = maxNumTokens;
        auto inputLength = maxNumTokens - tokensPerBlock - 1;
        auto numSharedBlocks = inputLength / tokensPerBlock;
        auto numBlocksPerSeq = numSharedBlocks + (maxBlocksPerSeq - numSharedBlocks) * maxBeamWidth;

        auto totalNumBlocks = mMaxNumSequences * numBlocksPerSeq;
        auto constexpr blocksInSecondaryPool = 0;

        auto constexpr enableBlockReuse = false;
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

        int numHeadsPerRankForContext = (numHeads + mContextTpSize - 1) / mContextTpSize;
        if (isMLA || mContextDP)
        {
            numHeadsPerRankForContext = numHeads;
        }

        using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;
        auto blocksPerWindow = BlocksPerWindow{{maxAttentionWindow, {totalNumBlocks, blocksInSecondaryPool}}};
        std::vector<SizeType32> maxAttentionWindowVec{};
        maxAttentionWindowVec.push_back(maxAttentionWindow);
        if (mIsWindowAttention)
        {
            auto attentionNumBlocks = 2 * mMaxNumSequences;
            blocksPerWindow[windowAttentionToken] = {attentionNumBlocks, blocksInSecondaryPool};
            maxAttentionWindowVec.push_back(windowAttentionToken);
        }
        TLLM_LOG_DEBUG(" cacheManager isWindowAttention: %d", mIsWindowAttention);
        mManager = std::make_unique<KVCacheManager>(layerNumthisRank, numHeadsPerRank, sizePerHead, tokensPerBlock,
            blocksPerWindow, mMaxNumSequences, maxBeamWidth, maxAttentionWindowVec, std::nullopt, dataType,
            sinkTokenLength, stream, maxNumTokens, enableBlockReuse, onboardBlocks, cacheType, std::nullopt, nullptr,
            true);
        texec::kv_cache::CacheState::AttentionType attentionType = isMLA
            ? texec::kv_cache::CacheState::AttentionType::kMLA
            : texec::kv_cache::CacheState::AttentionType::kDEFAULT;
        mCacheState = std::make_unique<texec::kv_cache::CacheState>(numLayers, numHeadsPerRank, sizePerHead,
            tokensPerBlock, mTpSize, mPpSize, mCpSize, mAttentionLayerNumPerPP, dataType, attentionType, kvFactor,
            enableDPAttention, DPrank, DPsize);
        mContextCacheState = std::make_unique<texec::kv_cache::CacheState>(numLayers, numHeadsPerRankForContext,
            sizePerHead, tokensPerBlock, mContextTpSize, mContextPpSize, mContextCpSize, contextAttentionLayerNumPerPP,
            dataType, attentionType, kvFactor, mContextDP, DPrank, mContextTpSize);

        // UVM seems to be incompatible with MPI, and it is continuing to investigate.
        bool constexpr useUvm = false;
        mManager->allocatePools(useUvm);
    }

    void setUpCacheTransceiver()
    {
        if (!(mIsContext || mIsGeneration))
        {
            return;
        }
        else if (tensorrt_llm::common::getEnvUseMPIKvCache() || tensorrt_llm::common::getEnvUseUCXKvCache()
            || tensorrt_llm::common::getEnvUseNixlKvCache())
        {
            int maxNumTokens = 2048;
            mCacheTransBufferManager = std::make_unique<CacheTransBufferManager>(mManager.get(), maxNumTokens);
            bool isUcx = tensorrt_llm::common::getEnvUseUCXKvCache();
            bool isNixl = tensorrt_llm::common::getEnvUseNixlKvCache();
            TLLM_LOG_INFO("Enable %s KV cache transport.", isUcx ? "UCX" : isNixl ? "NIXL" : "MPI");

            if (isUcx)
            {
                std::lock_guard<std::mutex> lock(mDllMutex);
                void* WrapperLibHandle = dllOpen(UCX_WRAPPER_LIB_NAME);
                TLLM_CHECK_WITH_INFO(
                    WrapperLibHandle != nullptr, "UCX wrapper library is not open correctly. dlerror: %s", dlerror());
                auto load_sym = [](void* handle, char const* name)
                {
                    void* ret = dllGetSym(handle, name);
                    TLLM_CHECK_WITH_INFO(ret != nullptr,
                        "Unable to load UCX wrapper library symbol, possible cause is that TensorRT LLM library is not "
                        "built with UCX support, please rebuild in UCX-enabled environment.");
                    return ret;
                };
                std::unique_ptr<tensorrt_llm::executor::kv_cache::ConnectionManager> (*makeUcxConnectionManager)();
                *(void**) (&makeUcxConnectionManager) = load_sym(WrapperLibHandle, "makeUcxConnectionManager");
                mConnectionManager = makeUcxConnectionManager();
            }
            else if (isNixl)
            {
                constexpr auto port = 22345;

                setenv("TRTLLM_NIXL_PORT", std::to_string(port).c_str(), 1);

                mConnectionManager = std::make_unique<texec::kv_cache::AgentConnectionManager>(
                    mCacheTransBufferManager.get(), *mCacheState);
            }
            else
            {
                mConnectionManager = std::make_unique<texec::kv_cache::MpiConnectionManager>(mComm);
            }

            auto makeFormatter
                = [this]() { return createCacheFormatter(mManager.get(), mCacheTransBufferManager.get(), mIsMLA); };

            if (mIsContext)
            {
                mSender = std::make_unique<CacheSender>(
                    mConnectionManager.get(), *mCacheState, mRankInInstance, makeFormatter());
            }
            else
            {
                mRequester = std::make_unique<CacheReceiver>(
                    mConnectionManager.get(), *mCacheState, mRankInInstance, makeFormatter());
            }

            std::vector<int> contextRankVec(mContextRankSize);
            std::iota(contextRankVec.begin(), contextRankVec.end(), 0);

            if (isUcx || isNixl)
            {
                auto commState = mConnectionManager->getCommState();
                namespace su = tensorrt_llm::executor::serialize_utils;

                if (tensorrt_llm::mpi::MpiComm::world().getRank() == 0)
                {
                    std::ostringstream oStream;
                    su::serialize(commState, oStream);
                    auto str = oStream.str();
                    std::vector<char> buffer(str.begin(), str.end());

                    for (int genRank = mContextRankSize; genRank < mContextRankSize + mGenRankSize; genRank++)
                    {
                        int64_t bufferSize = buffer.size();
                        TLLM_LOG_DEBUG(tensorrt_llm::mpi::MpiComm::world().getRank(), "send bufferSize: %ld to %d",
                            bufferSize, genRank);
                        tensorrt_llm::mpi::MpiComm::world().sendRawTag(
                            &bufferSize, 1, tensorrt_llm::mpi::MpiType::kINT64, genRank, 0x1F);
                        tensorrt_llm::mpi::MpiComm::world().sendRawTag(
                            buffer.data(), buffer.size(), tensorrt_llm::mpi::MpiType::kCHAR, genRank, 0x2F);
                        TLLM_LOG_DEBUG(tensorrt_llm::mpi::MpiComm::world().getRank(), "send buffer to %d", genRank);
                    }
                }

                if (mIsGeneration)
                {
                    int64_t bufferSize;
                    tensorrt_llm::mpi::MpiComm::world().recvRawTag(
                        &bufferSize, 1, tensorrt_llm::mpi::MpiType::kINT64, 0, 0x1F);
                    TLLM_LOG_DEBUG(
                        tensorrt_llm::mpi::MpiComm::world().getRank(), "recv bufferSize: %ld from 0", bufferSize);
                    std::vector<char> recvBuffer(bufferSize);
                    tensorrt_llm::mpi::MpiComm::world().recvRawTag(
                        recvBuffer.data(), bufferSize, tensorrt_llm::mpi::MpiType::kCHAR, 0, 0x2F);
                    TLLM_LOG_DEBUG(tensorrt_llm::mpi::MpiComm::world().getRank(), "recv buffer from 0", bufferSize);
                    std::istringstream iStream(std::string(recvBuffer.begin(), recvBuffer.end()));
                    su::VectorWrapBuf<char> strbuf(recvBuffer);
                    std::istream is(&strbuf);
                    mContextCommState = std::make_unique<tensorrt_llm::executor::kv_cache::CommState>(
                        su::deserialize<tensorrt_llm::executor::kv_cache::CommState>(is));
                }

                if (mIsContext)
                {
                    mContextCommState = std::make_unique<tensorrt_llm::executor::kv_cache::CommState>(commState);
                }

                TLLM_LOG_INFO(tensorrt_llm::mpi::MpiComm::world().getRank(), "mContextCommState: %s",
                    mContextCommState->toString().c_str());
            }
            else
            {
                mContextCommState = std::make_unique<tensorrt_llm::executor::kv_cache::CommState>(contextRankVec);
            }
        }
        else
        {
            TLLM_CHECK_WITH_INFO(false, "Please set at least one cache transfer backend");
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
            mContextCacheState->getParallelConfig().mPipelineParallelism,
            mContextCacheState->getParallelConfig().mContextParallelism,
            mContextCacheState->getParallelConfig().mAttentionLayerNumPerPP, mContextCacheState->getDataType(),
            mContextCacheState->getAttentionConfig().mAttentionType, mContextCacheState->getAttentionConfig().mKvFactor,
            mContextCacheState->getParallelConfig().mEnableAttentionDP, contextDpRank,
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
        auto blockRange = BlockRange::fromAllBlockIds(*mManager, llmRequest->mRequestId);

        int const numPools = mManager->getBlockManager().getNumPools();
        TLLM_LOG_DEBUG(" addRequestAndTransportCacheForContext mManager numPools: %d", numPools);
        auto const& windowSizes = blockRange.getWindowSizes();
        int blockIdx = 0;
        for (auto const& windowSize : windowSizes)
        {
            auto blockRangeForWindow = blockRange.getBlockRangeForWindow(windowSize);
            TLLM_LOG_DEBUG("update windowSize: %d", windowSize);
            for (auto it = blockRangeForWindow.begin(); it != blockRangeForWindow.end(); ++it)
            {
                fillBlockData(*it, blockIdx, llmRequest->getPromptLen(), windowSize);
                blockIdx++;
            }
            TLLM_LOG_DEBUG("windowSize: %d finish fill block data", windowSize);
        }

        TLLM_LOG_DEBUG(
            "addRequestAndTransportCacheForContext blockManager numPools: %d finish fill block data", numPools);
        auto const& blockManager = mManager->getBlockManager();

        auto const onlyWindowSize = blockManager.getPoolWindowSize(0);

        blockManager.getBufferManager(onlyWindowSize).getStream().synchronize();
        auto future = mSender->sendAsync(*llmRequest);
        return future;
    }

    std::future<void> addRequestAndTransportCacheForGeneration(std::shared_ptr<LlmRequest> const& llmRequest)
    {
        auto constexpr beamIdx{0};
        auto constexpr beamWidth{1};
        mManager->addSequence(llmRequest->mRequestId, llmRequest->getNumTokens(beamIdx), beamWidth, llmRequest);

        return mRequester->receiveAsync(*llmRequest);
    }

    // Called only by generationVerifyKVCache. Currently, generation ranks might over-allocate blocks when CP is
    // enabled.
    bool isBlockOverallocated(int blockIdx, int numTotalBlocks)
    {
        bool const generationHasCP = mCpSize > 1;
        if (!generationHasCP)
        {
            return false;
        }
        int const numValidBlocks = getBlockNumAccountingForCP(mCpRank, mCpSize, numTotalBlocks, /*strict=*/true);
        return blockIdx >= numValidBlocks;
    }

    void generationVerifyKVCache(std::shared_ptr<LlmRequest> const& llmRequest)
    {
        auto constexpr beamIdx{0};
        auto constexpr beamWidth{1};
        int blockIdx = 0;

        TLLM_CUDA_CHECK(cudaDeviceSynchronize());

        auto blockRange = BlockRange::fromAllBlockIds(*mManager, llmRequest->mRequestId);
        auto const& windowSizes = blockRange.getWindowSizes();
        for (auto const& windowSize : windowSizes)
        {
            auto blockRangeForWindow = blockRange.getBlockRangeForWindow(windowSize);
            int maxBlockInWindow = windowSize / mCacheState->getModelConfig().mTokensPerBlock;
            int startBlockId = std::max(0, static_cast<int>(blockRangeForWindow.size()) - (maxBlockInWindow + 1));
            int blockIdInWindow = 0;
            int const numTotalBlocks = blockRangeForWindow.size();
            for (auto it = blockRangeForWindow.begin(); it != blockRangeForWindow.end(); ++it)
            {
                if (isBlockOverallocated(blockIdx, numTotalBlocks))
                {
                    TLLM_LOG_INFO(
                        "[generationVerifyKVCache] Skipping over-allocated block for request id %d (rank %d, blockIdx "
                        "%d, numTotalBlocks %d)",
                        llmRequest->mRequestId, mRank, blockIdx, numTotalBlocks);
                    break;
                }
                if (blockIdInWindow >= startBlockId)
                {
                    verifyBlockData(*it, blockIdx, llmRequest->getPromptLen(), windowSize);
                }
                blockIdx++;
                blockIdInWindow++;
            }
        }
    }

    void fillBlockData(tensorrt_llm::runtime::ITensor& blockData, int blockId, size_t initial, int windowSize = 0)
    {
        auto const& blockManager = mManager->getBlockManager();
        auto const onlyWindowSize = windowSize == 0 ? blockManager.getPoolWindowSize(0) : windowSize;
        auto const& bufferManager = blockManager.getBufferManager(onlyWindowSize);
        auto hostTensor = tensorrt_llm::runtime::BufferManager::cpu(blockData.getShape(), blockData.getDataType());
        int layerSizeThisRank = blockData.getDimension<1>();
        int startLayerId = 0;
        if (mIsWindowAttention)
        {
            startLayerId = layerSizeThisRank * mPpRank;
        }
        else
        {
            for (int ppRank = 0; ppRank < mPpRank; ppRank++)
            {
                startLayerId += mAttentionLayerNumPerPP[ppRank];
            }
        }
        int headSizePerRank = mCacheState->getModelConfig().mNbKvHeadsPerLayer.at(0);
        int startHeadId = headSizePerRank * (mTpRank / mDupHeadFactor);
        bool enableDP = mCacheState->getParallelConfig().mEnableAttentionDP;
        if (mIsMLA || enableDP)
        {
            startHeadId = 0;
        }
        int kvFactor = mCacheState->getAttentionConfig().mKvFactor;
        int tokensPerBlock = mCacheState->getModelConfig().mTokensPerBlock;
        int startTokenId = blockId * tokensPerBlock;
        int sizePerHead = mCacheState->getModelConfig().mSizePerHead;
        auto dataTypeSize = tensorrt_llm::common::getDTypeSize(blockData.getDataType());
        for (int layerId = 0; layerId < layerSizeThisRank; layerId++)
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
                            generateExpectedValue(initial, windowSize, tokenId + startTokenId, layerId + startLayerId,
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
                                generateExpectedValue(initial, windowSize, tokenId + startTokenId,
                                    layerId + startLayerId, headId + startHeadId, hiddenId, false,
                                    blockData.getDataType()));
                        }
                    }
                }
            }
        }
        bufferManager.copy(*hostTensor, blockData);
        bufferManager.getStream().synchronize();
    }

    void verifyBlockData(tensorrt_llm::runtime::ITensor& blockData, int blockId, size_t initial, int windowSize = 0)
    {
        auto const& blockManager = mManager->getBlockManager();

        auto const onlyWindowSize = windowSize == 0 ? blockManager.getPoolWindowSize(0) : windowSize;
        auto const& bufferManager = blockManager.getBufferManager(onlyWindowSize);

        auto hostTensor = tensorrt_llm::runtime::BufferManager::cpu(blockData.getShape(), blockData.getDataType());
        int layerSizethisRank = blockData.getDimension<1>();
        int startLayerId = 0;
        if (mIsWindowAttention)
        {
            startLayerId = layerSizethisRank * mPpRank;
        }
        else
        {
            for (int ppRank = 0; ppRank < mPpRank; ppRank++)
            {
                startLayerId += mAttentionLayerNumPerPP[ppRank];
            }
        }

        int headSizePerRank = mCacheState->getModelConfig().mNbKvHeadsPerLayer.at(0);
        int startHeadId = headSizePerRank * (mTpRank / mDupHeadFactor);
        bool enableDP = mCacheState->getParallelConfig().mEnableAttentionDP;
        if (mIsMLA || enableDP)
        {
            startHeadId = 0;
        }
        int kvFactor = mCacheState->getAttentionConfig().mKvFactor;
        int tokensPerBlock = mCacheState->getModelConfig().mTokensPerBlock;
        int startTokenId = (blockId * mCpSize + mCpRank) * tokensPerBlock;
        int sizePerHead = mCacheState->getModelConfig().mSizePerHead;

        bufferManager.copy(blockData, *hostTensor);
        bufferManager.getStream().synchronize();

        for (int layerId = 0; layerId < layerSizethisRank; layerId++)
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
                            generateExpectedValue(initial, windowSize, tokenId + startTokenId, layerId + startLayerId,
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
                                generateExpectedValue(initial, windowSize, tokenId + startTokenId,
                                    layerId + startLayerId, headId + startHeadId, hiddenId, false,
                                    blockData.getDataType()));
                        }
                    }
                }
            }
        }
    }

    std::variant<double, float, int16_t, int8_t> generateExpectedValue(size_t initial, int windowSize, int tokenId,
        int layerId, int headId, int hiddenId, bool key, nvinfer1::DataType dataType)
    {

        size_t seed = 0;
        std::size_t hashValue = std::hash<size_t>{}(initial);
        std::hash<int> hasher{};
        seed ^= hashValue + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hasher(windowSize) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
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
    SizeType32 mSizeInInstance{0}, mTpRank{0}, mPpRank{0}, mCpRank{0}, mTpSize{0}, mPpSize{0}, mCpSize{0},
        mContextRankSize{0}, mGenRankSize{0}, mContextTpSize{0}, mContextPpSize{0}, mContextCpSize{0};
    LlmRequest::RequestIdType mRequestId{0};
    bool mContextDP{false};
    bool mGenerationDP{false};
    bool mIsMLA{false};
    bool mIsWindowAttention{false};
    int mDupHeadFactor{1};
    std::vector<SizeType32> mAttentionLayerNumPerPP;

    SizeType32 mMaxNumSequences{};
    std::unique_ptr<KVCacheManager> mManager;
    std::unique_ptr<CacheTransBufferManager> mCacheTransBufferManager;
    std::unique_ptr<CacheSender> mSender;
    std::unique_ptr<CacheReceiver> mRequester;
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
    else
    {
        setenv("UCX_TCP_CM_REUSEADDR", "y",
            1); // tests creates and destroies ucxCacheCommunicatoers frequently, so listener ports must be reused
    }
    AsymmetricTestParam param = GetParam();
    int contextTp = std::get<0>(param);
    int contextPp = std::get<1>(param);
    int contextCp = std::get<2>(param);
    int genTp = std::get<3>(param);
    int genPp = std::get<4>(param);
    int genCp = std::get<5>(param);
    int numLayers = std::get<6>(param);
    int numHeads = std::get<7>(param);
    int sizePerHead = std::get<8>(param);
    int tokensPerBlock = std::get<9>(param);
    nvinfer1::DataType dataType = std::get<10>(param);

    int kvFactor = std::get<11>(param);
    bool isMLA = std::get<12>(param);
    bool contextDP = std::get<13>(param);
    bool generationDP = std::get<14>(param);

    bool isWindow = std::get<15>(param);

    if (genCp > 1 && tensorrt_llm::common::getEnvUseNixlKvCache())
    {
        GTEST_SKIP() << "Temporarily skipping cache transceiver tests with NIXL backend for CP.";
    }

    setUpCommunicator(contextTp, contextPp, contextCp, genTp, genPp, genCp, isMLA, contextDP, generationDP);

    if (mIsContext || mIsGeneration)
    {
        setUpCacheManager(numLayers, numHeads, sizePerHead, tokensPerBlock, dataType, kvFactor, isMLA, false, isWindow);
        setUpCacheTransceiver();
        std::vector<std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest>> requests;

        // the second loop is for cache reuse
        for (int i = 0; i < 2; i++)
        {
            for (auto len : {30, 10, 60, 80})
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
    else
    {
        setenv("UCX_TCP_CM_REUSEADDR", "y",
            1); // tests creates and destroies ucxCacheCommunicatoers frequently, so listener ports must be reused
    }

    AsymmetricTestParam param = GetParam();
    int contextTp = std::get<0>(param);
    int contextPp = std::get<1>(param);
    int contextCp = std::get<2>(param);
    int genTp = std::get<3>(param);
    int genPp = std::get<4>(param);
    int genCp = std::get<5>(param);
    int numLayers = std::get<6>(param);
    int numHeads = std::get<7>(param);
    int sizePerHead = std::get<8>(param);
    int tokensPerBlock = std::get<9>(param);
    nvinfer1::DataType dataType = std::get<10>(param);

    int kvFactor = std::get<11>(param);
    bool isMLA = std::get<12>(param);
    bool contextDP = std::get<13>(param);
    bool generationDP = std::get<14>(param);
    bool isWindow = std::get<15>(param);

    if (genCp > 1 && tensorrt_llm::common::getEnvUseNixlKvCache())
    {
        GTEST_SKIP() << "Temporarily skipping cache transceiver tests with NIXL backend for CP.";
    }
    setUpCommunicator(contextTp, contextPp, contextCp, genTp, genPp, genCp, isMLA, contextDP, generationDP);

    if (mIsContext || mIsGeneration)
    {
        bool enableDP = mIsContext ? contextDP : generationDP;
        setUpCacheManager(
            numLayers, numHeads, sizePerHead, tokensPerBlock, dataType, kvFactor, isMLA, enableDP, isWindow);
        setUpCacheTransceiver();
        std::vector<std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest>> requests;
        int requestId = 0;
        for (auto len : {60, 30, 60, 10})
        {
            requests.emplace_back(makeLlmRequestWithDP(len, requestId, requestId % contextTp));
            requestId++;
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
                    if ((i) % mTpSize == mTpRank)
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
                    if ((i) % mTpSize == mTpRank)
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
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1), testing::Values(1, 2),
        testing::Values(1, 2), testing::Values(1), testing::Values(4), testing::Values(4), testing::Values(4),
        testing::Values(16), testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(false), testing::Values(false), testing::Values(true, false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithWindow, AsymmetricalCacheTest,
    testing::Combine(testing::Values(1), testing::Values(1), testing::Values(1), testing::Values(1), testing::Values(1),
        testing::Values(1), testing::Values(5), testing::Values(4), testing::Values(4), testing::Values(8),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(false), testing::Values(false), testing::Values(true)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTest1, AsymmetricalCacheTest,
    testing::Combine(testing::Values(4), testing::Values(1), testing::Values(1), testing::Values(1), testing::Values(4),
        testing::Values(1), testing::Values(8), testing::Values(4), testing::Values(4), testing::Values(8),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(false), testing::Values(false), testing::Values(false, true)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTest1EvenLayer, AsymmetricalCacheTest,
    testing::Combine(testing::Values(1), testing::Values(4), testing::Values(1), testing::Values(1), testing::Values(4),
        testing::Values(1), testing::Values(10), testing::Values(4), testing::Values(4), testing::Values(8),
        testing::Values(nvinfer1::DataType::kFLOAT), testing::Values(2), testing::Values(false), testing::Values(false),
        testing::Values(false), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTest2EvenLayer, AsymmetricalCacheTest,
    testing::Combine(testing::Values(4), testing::Values(1), testing::Values(1), testing::Values(1), testing::Values(4),
        testing::Values(1), testing::Values(10), testing::Values(4), testing::Values(4), testing::Values(8),
        testing::Values(nvinfer1::DataType::kFLOAT), testing::Values(2), testing::Values(false), testing::Values(false),
        testing::Values(false), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTest2, AsymmetricalCacheTest,
    testing::Combine(testing::Values(1), testing::Values(2), testing::Values(1), testing::Values(1),
        testing::Values(1, 4), testing::Values(1), testing::Values(16), testing::Values(16), testing::Values(4),
        testing::Values(8), testing::Values(nvinfer1::DataType::kFLOAT), testing::Values(2), testing::Values(false),
        testing::Values(false), testing::Values(false), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTest0ForMLA, AsymmetricalCacheTest,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1), testing::Values(1, 2),
        testing::Values(1, 2), testing::Values(1), testing::Values(4), testing::Values(1), testing::Values(4),
        testing::Values(16), testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(1),
        testing::Values(true), testing::Values(false), testing::Values(false), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTest1ForMLA, AsymmetricalCacheTest,
    testing::Combine(testing::Values(4), testing::Values(1), testing::Values(1), testing::Values(1), testing::Values(4),
        testing::Values(1), testing::Values(4), testing::Values(1), testing::Values(4), testing::Values(8),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(1),
        testing::Values(true), testing::Values(false), testing::Values(false), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTest1ForMLAEvenLayer, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(1), testing::Values(4), testing::Values(1), testing::Values(4), testing::Values(1),
        testing::Values(1), testing::Values(10), testing::Values(1), testing::Values(4), testing::Values(8),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(1),
        testing::Values(true), testing::Values(false), testing::Values(false, true), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTest2ForMLAEvenLayer, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(4), testing::Values(1), testing::Values(1), testing::Values(1), testing::Values(4),
        testing::Values(1), testing::Values(10), testing::Values(1), testing::Values(4), testing::Values(8),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(1),
        testing::Values(true), testing::Values(false), testing::Values(false, true), testing::Values(false)));

// Tests cases where there's non-trivial TP and PP on context side but only CP on gen side.
INSTANTIATE_TEST_CASE_P(AsymmetricCaseTest0WithCPForMLA, AsymmetricalCacheTest,
    testing::Combine(/*contextTp*/ testing::Values(1, 2),
        /*contextPp*/ testing::Values(1, 2),
        /*contextCp*/ testing::Values(1),
        /*genTp*/ testing::Values(1),
        /*genPp*/ testing::Values(1),
        /*genCp*/ testing::Values(2, 4),
        /*numLayers*/ testing::Values(4),
        /*numHeads*/ testing::Values(1),
        /*sizePerHead*/ testing::Values(4),
        /*tokensPerBlock*/ testing::Values(16),
        /*dataType*/ testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8),
        /*kvFactor*/ testing::Values(2),
        /*isMLA*/ testing::Values(true),
        /*contextDP*/ testing::Values(false),
        /*generationDP*/ testing::Values(false),
        /*isWindow*/ testing::Values(false)));

// Tests cases where there's non-trivial TP and PP on context side while non-trivial CP & PP on gen side.
INSTANTIATE_TEST_CASE_P(AsymmetricCaseTest1WithCPForMLA, AsymmetricalCacheTest,
    testing::Combine(/*contextTp*/ testing::Values(1, 2),
        /*contextPp*/ testing::Values(1, 2),
        /*contextCp*/ testing::Values(1),
        /*genTp*/ testing::Values(1),
        /*genPp*/ testing::Values(2),
        /*genCp*/ testing::Values(2),
        /*numLayers*/ testing::Values(4),
        /*numHeads*/ testing::Values(1),
        /*sizePerHead*/ testing::Values(4),
        /*tokensPerBlock*/ testing::Values(16),
        /*dataType*/ testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8),
        /*kvFactor*/ testing::Values(2),
        /*isMLA*/ testing::Values(true),
        /*contextDP*/ testing::Values(false),
        /*generationDP*/ testing::Values(false),
        /*isWindow*/ testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForMLA1, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1), testing::Values(1, 2),
        testing::Values(1, 2), testing::Values(1), testing::Values(4), testing::Values(1), testing::Values(4),
        testing::Values(16), testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(1),
        testing::Values(true), testing::Values(true), testing::Values(true), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForMLA2, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1), testing::Values(1, 2),
        testing::Values(1, 2), testing::Values(1), testing::Values(4), testing::Values(1), testing::Values(4),
        testing::Values(16), testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(1),
        testing::Values(true), testing::Values(true), testing::Values(false), testing::Values(false)));
INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForMLA3, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1), testing::Values(1, 2),
        testing::Values(1, 2), testing::Values(1), testing::Values(4), testing::Values(1), testing::Values(4),
        testing::Values(16), testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(1),
        testing::Values(true), testing::Values(false), testing::Values(true), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForMLA4, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(2), testing::Values(1), testing::Values(1), testing::Values(4), testing::Values(1),
        testing::Values(1), testing::Values(4), testing::Values(1), testing::Values(4), testing::Values(16),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(1),
        testing::Values(true), testing::Values(false), testing::Values(true), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForMLA5, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(4), testing::Values(1), testing::Values(1), testing::Values(2), testing::Values(1),
        testing::Values(1), testing::Values(4), testing::Values(1), testing::Values(4), testing::Values(16),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(1),
        testing::Values(true), testing::Values(false), testing::Values(true), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForNoMLA, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1), testing::Values(1, 2),
        testing::Values(1, 2), testing::Values(1), testing::Values(4), testing::Values(4), testing::Values(4),
        testing::Values(16), testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(true), testing::Values(true), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForNoMLA1, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1), testing::Values(1, 2),
        testing::Values(1, 2), testing::Values(1), testing::Values(4), testing::Values(4), testing::Values(4),
        testing::Values(16), testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(true), testing::Values(false), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForNoMLA2, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1), testing::Values(1, 2),
        testing::Values(1, 2), testing::Values(1), testing::Values(4), testing::Values(4), testing::Values(4),
        testing::Values(16), testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(false), testing::Values(true), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForNoMLADuplicate0, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1), testing::Values(4),
        testing::Values(1), testing::Values(1), testing::Values(4), testing::Values(2), testing::Values(4),
        testing::Values(16), testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(true, false), testing::Values(false), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForNoMLADuplicate0EvenLayer, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(1), testing::Values(4), testing::Values(1), testing::Values(4), testing::Values(1),
        testing::Values(1), testing::Values(5), testing::Values(2), testing::Values(4), testing::Values(16),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(true, false), testing::Values(false), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForNoMLADuplicate1, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(1, 2), testing::Values(1, 2), testing::Values(1), testing::Values(2),
        testing::Values(2), testing::Values(1), testing::Values(4), testing::Values(1), testing::Values(4),
        testing::Values(16), testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(true, false), testing::Values(false), testing::Values(false)));
INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForNoMLADuplicate2, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(4), testing::Values(1), testing::Values(1), testing::Values(4, 2),
        testing::Values(1), testing::Values(1), testing::Values(4), testing::Values(2), testing::Values(4),
        testing::Values(16), testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(false), testing::Values(false), testing::Values(false)));
INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForNoMLADuplicate3, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(2), testing::Values(1), testing::Values(1), testing::Values(4), testing::Values(1),
        testing::Values(1), testing::Values(4), testing::Values(2), testing::Values(4), testing::Values(16),
        testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(false), testing::Values(true), testing::Values(false)));

INSTANTIATE_TEST_CASE_P(AsymmetricCaseTestWithDPForNoMLADuplicate4, AsymmetricalCacheTestWithDP,
    testing::Combine(testing::Values(4), testing::Values(1), testing::Values(1), testing::Values(1, 2),
        testing::Values(2), testing::Values(1), testing::Values(4), testing::Values(1, 2), testing::Values(4),
        testing::Values(16), testing::Values(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kINT8), testing::Values(2),
        testing::Values(false), testing::Values(false), testing::Values(false), testing::Values(false)));

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

    auto const verifyContext = [&](int contextRank, tr::WorldConfig const& contextWC, tr::WorldConfig const& genWC,
                                   std::vector<int> const& expectRanks, int expectPPDomain, int expectTPDomain,
                                   int expectCPDomain, bool expectNeedSend)
    {
        auto attentionType = isMLA ? texec::kv_cache::CacheState::AttentionType::kMLA
                                   : texec::kv_cache::CacheState::AttentionType::kDEFAULT;
        std::vector<SizeType32> contextAttentionLayerNumPerPP(
            contextWC.getPipelineParallelism(), numLayers / contextWC.getPipelineParallelism());
        std::vector<SizeType32> genAttentionLayerNumPerPP(
            genWC.getPipelineParallelism(), numLayers / genWC.getPipelineParallelism());

        auto const sharedModelConfig
            = texec::kv_cache::CacheState::ModelConfig{std::vector(numLayers, numHeads), sizePerHead, tokensPerBlock};
        auto const contextCache = texec::kv_cache::CacheState(
            sharedModelConfig, contextWC, contextAttentionLayerNumPerPP, dataType, attentionType, kvFactor);
        auto const genCache = texec::kv_cache::CacheState(
            sharedModelConfig, genWC, genAttentionLayerNumPerPP, dataType, attentionType, kvFactor);

        auto const contextTargetInfo
            = tensorrt_llm::executor::kv_cache::TargetRanksInfoForDP(genCache, contextCache, contextRank);

        EXPECT_EQ(expectRanks, contextTargetInfo.mIRanks);
        EXPECT_EQ(expectPPDomain, contextTargetInfo.mDomainPPSize);
        EXPECT_EQ(expectTPDomain, contextTargetInfo.mDomainTPSize);
        EXPECT_EQ(expectCPDomain, contextTargetInfo.mDomainCPSize);
        EXPECT_EQ(expectNeedSend, MLACacheFormatter::needSendCache(contextCache, genCache, contextRank));
    };

    // TP shrinks from context to generation.
    {
        tr::WorldConfig const contextWC{/*tpSize*/ 4, /*ppSize*/ 2, /*cpSize*/ 1};
        tr::WorldConfig const genWC{/*tpSize*/ 2, /*ppSize*/ 2, /*cpSize*/ 1};
        verifyContext(
            /*contextRank*/ 0, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 1, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 1, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 1, /*expectNeedSend*/ false);
        verifyContext(
            /*contextRank*/ 2, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {1}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 1, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 3, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {1}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 1, /*expectNeedSend*/ false);
        verifyContext(
            /*contextRank*/ 4, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {2}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 1, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 5, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {2}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 1, /*expectNeedSend*/ false);
        verifyContext(
            /*contextRank*/ 6, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {3}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 1, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 7, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {3}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 1, /*expectNeedSend*/ false);
    }

    // TP grows from context to generation.
    {
        tr::WorldConfig const contextWC{/*tpSize*/ 2, /*ppSize*/ 2, /*cpSize*/ 1};
        tr::WorldConfig const genWC{/*tpSize*/ 4, /*ppSize*/ 2, /*cpSize*/ 1};
        verifyContext(
            /*contextRank*/ 0, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 1}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 2, /*expectCPDomain*/ 1, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 1, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {2, 3}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 2, /*expectCPDomain*/ 1, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 2, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {4, 5}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 2, /*expectCPDomain*/ 1, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 3, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {6, 7}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 2, /*expectCPDomain*/ 1, /*expectNeedSend*/ true);
    }

    // TP as well as PP grow from context to generation.
    {
        tr::WorldConfig const contextWC{/*tpSize*/ 2, /*ppSize*/ 1, /*cpSize*/ 1};
        tr::WorldConfig const genWC{/*tpSize*/ 4, /*ppSize*/ 2, /*cpSize*/ 1};
        verifyContext(
            /*contextRank*/ 0, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 4, 1, 5},
            /*expectPPDomain*/ 2, /*expectTPDomain*/ 2, /*expectCPDomain*/ 1, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 1, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {2, 6, 3, 7},
            /*expectPPDomain*/ 2, /*expectTPDomain*/ 2, /*expectCPDomain*/ 1, /*expectNeedSend*/ true);
    }

    // PP grows while TP shrinks from context to generation.
    {
        tr::WorldConfig const contextWC{/*tpSize*/ 2, /*ppSize*/ 1, /*cpSize*/ 1};
        tr::WorldConfig const genWC{/*tpSize*/ 1, /*ppSize*/ 2, /*cpSize*/ 1};
        verifyContext(
            /*contextRank*/ 0, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 1}, /*expectPPDomain*/
            2,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 1, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 1, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 1}, /*expectPPDomain*/
            2,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 1, /*expectNeedSend*/ false);
    }

    // CP grows from context to generation.
    {
        tr::WorldConfig const contextWC{/*tpSize*/ 2, /*ppSize*/ 2, /*cpSize*/ 1};
        tr::WorldConfig const genWC{/*tpSize*/ 2, /*ppSize*/ 2, /*cpSize*/ 2};
        verifyContext(
            /*contextRank*/ 0, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 2},
            /*expectPPDomain*/ 1, /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 1, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {1, 3},
            /*expectPPDomain*/ 1, /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 2, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {4, 6},
            /*expectPPDomain*/ 1, /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 3, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {5, 7},
            /*expectPPDomain*/ 1, /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
    }

    // TP as well as CP grow from context to generation.
    {
        tr::WorldConfig const contextWC{/*tpSize*/ 2, /*ppSize*/ 2, /*cpSize*/ 1};
        tr::WorldConfig const genWC{/*tpSize*/ 4, /*ppSize*/ 2, /*cpSize*/ 2};
        verifyContext(
            /*contextRank*/ 0, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 4, 1, 5},
            /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 2, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 1, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {2, 6, 3, 7},
            /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 2, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 2, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {8, 12, 9, 13},
            /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 2, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 3, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {10, 14, 11, 15},
            /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 2, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
    }

    // TP shrinks while CP grows from context to generation.
    {
        tr::WorldConfig const contextWC{/*tpSize*/ 4, /*ppSize*/ 1, /*cpSize*/ 1};
        tr::WorldConfig const genWC{/*tpSize*/ 2, /*ppSize*/ 1, /*cpSize*/ 2};
        verifyContext(
            /*contextRank*/ 0, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 2}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 1, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 2}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ false);
        verifyContext(
            /*contextRank*/ 2, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {1, 3}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 3, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {1, 3}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ false);
    }

    // PP as well as CP grow from context to generation.
    {
        tr::WorldConfig const contextWC{/*tpSize*/ 2, /*ppSize*/ 2, /*cpSize*/ 1};
        tr::WorldConfig const genWC{/*tpSize*/ 2, /*ppSize*/ 4, /*cpSize*/ 2};
        verifyContext(
            /*contextRank*/ 0, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 4, 2, 6},
            /*expectPPDomain*/ 2,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 1, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {1, 5, 3, 7},
            /*expectPPDomain*/ 2,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 2, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {8, 12, 10, 14},
            /*expectPPDomain*/ 2,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 3, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {9, 13, 11, 15},
            /*expectPPDomain*/ 2,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
    }

    // PP shrinks while CP grows from context to generation.
    {
        tr::WorldConfig const contextWC{/*tpSize*/ 2, /*ppSize*/ 4, /*cpSize*/ 1};
        tr::WorldConfig const genWC{/*tpSize*/ 2, /*ppSize*/ 2, /*cpSize*/ 2};
        verifyContext(
            /*contextRank*/ 0, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 2}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 1, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {1, 3}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 2, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 2}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 3, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {1, 3}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 4, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {4, 6}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 5, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {5, 7}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 6, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {4, 6}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 7, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {5, 7}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
    }

    // TP as well as PP shrink while CP grows from context to generation.
    {
        tr::WorldConfig const contextWC{/*tpSize*/ 4, /*ppSize*/ 2, /*cpSize*/ 1};
        tr::WorldConfig const genWC{/*tpSize*/ 2, /*ppSize*/ 1, /*cpSize*/ 2};
        verifyContext(
            /*contextRank*/ 0, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 2}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 1, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 2}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ false);
        verifyContext(
            /*contextRank*/ 2, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {1, 3}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 3, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {1, 3}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ false);
        verifyContext(
            /*contextRank*/ 4, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 2}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 5, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 2}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ false);
        verifyContext(
            /*contextRank*/ 6, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {1, 3}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 7, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {1, 3}, /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 2, /*expectNeedSend*/ false);
    }

    // TP, CP grow while PP shrinks from context to generation.
    {
        tr::WorldConfig const contextWC{/*tpSize*/ 2, /*ppSize*/ 2, /*cpSize*/ 1};
        tr::WorldConfig const genWC{/*tpSize*/ 4, /*ppSize*/ 1, /*cpSize*/ 2};
        verifyContext(
            /*contextRank*/ 0, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 4, 1, 5},
            /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 2, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 1, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {2, 6, 3, 7},
            /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 2, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 2, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 4, 1, 5},
            /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 2, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 3, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {2, 6, 3, 7},
            /*expectPPDomain*/ 1,
            /*expectTPDomain*/ 2, /*expectCPDomain*/ 2, /*expectNeedSend*/ true);
    }

    // PP, CP grow while TP shrinks from context to generation.
    {
        tr::WorldConfig const contextWC{/*tpSize*/ 2, /*ppSize*/ 1, /*cpSize*/ 1};
        tr::WorldConfig const genWC{/*tpSize*/ 1, /*ppSize*/ 2, /*cpSize*/ 4};
        verifyContext(
            /*contextRank*/ 0, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 4, 1, 5, 2, 6, 3, 7},
            /*expectPPDomain*/ 2,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 4, /*expectNeedSend*/ true);
        verifyContext(
            /*contextRank*/ 1, /*contextWC*/ contextWC, /*genWC*/ genWC, /*expectRanks*/ {0, 4, 1, 5, 2, 6, 3, 7},
            /*expectPPDomain*/ 2,
            /*expectTPDomain*/ 1, /*expectCPDomain*/ 4, /*expectNeedSend*/ false);
    }
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
    int contextCP = 1;
    int genPP = 1;
    int genTP = 2;
    int genCP = 1;
    bool contextEnableDP = true;
    bool genEnableDP = true;
    std::vector<SizeType32> contextAttentionLayerNumPerPP(contextPP, numLayers / contextPP);
    std::vector<SizeType32> genAttentionLayerNumPerPP(genPP, numLayers / genPP);

    auto const verifyContext = [&](int contextRank, int generationRank, std::vector<int> const& expectRanks,
                                   int expectPPDomain, int expectTPDomain, bool expectNeedSend)
    {
        int contextDPRank = contextRank % contextTP;
        int generationDPRank = generationRank % genTP;
        auto attentionType = isMLA ? texec::kv_cache::CacheState::AttentionType::kMLA
                                   : texec::kv_cache::CacheState::AttentionType::kDEFAULT;

        auto const contextCache = tensorrt_llm::executor::kv_cache::CacheState{numLayers, numHeads, sizePerHead,
            tokensPerBlock, contextTP, contextPP, contextCP, contextAttentionLayerNumPerPP, dataType, attentionType,
            kvFactor, contextEnableDP, contextDPRank, contextTP};

        auto const genCache = tensorrt_llm::executor::kv_cache::CacheState{numLayers, numHeads, sizePerHead,
            tokensPerBlock, genTP, genPP, genCP, genAttentionLayerNumPerPP, dataType, attentionType, kvFactor,
            genEnableDP, generationDPRank, genTP};

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
        /*expectNeedSend*/ false);
    verifyContext(
        /*contextRank*/ 1, /*generationRank*/ 0, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ false);
    verifyContext(
        /*contextRank*/ 1, /*generationRank*/ 1, /*expectRanks*/ {1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1,
        /*expectNeedSend*/ true);
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

        auto const contextCache = tensorrt_llm::executor::kv_cache::CacheState{numLayers, numHeads, sizePerHead,
            tokensPerBlock, contextTP, contextPP, contextCP, contextAttentionLayerNumPerPP, dataType, attentionType,
            kvFactor, contextEnableDP, contextDPRank, contextTP};

        auto const genCache = tensorrt_llm::executor::kv_cache::CacheState{numLayers, numHeads, sizePerHead,
            tokensPerBlock, genTP, genPP, genCP, genAttentionLayerNumPerPP, dataType, attentionType, kvFactor,
            genEnableDP, generationDPRank, genTP};

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
    contextAttentionLayerNumPerPP = std::vector<SizeType32>(contextPP, numLayers / contextPP);
    genAttentionLayerNumPerPP = std::vector<SizeType32>(genPP, numLayers / genPP);

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
    contextAttentionLayerNumPerPP = std::vector<SizeType32>(contextPP, numLayers / contextPP);
    genAttentionLayerNumPerPP = std::vector<SizeType32>(genPP, numLayers / genPP);

    verfiyGeneration(
        /*contextRank*/ 0, /*generationRank*/ 0, /*expectRanks*/ {0}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1);
    verfiyGeneration(
        /*contextRank*/ 1, /*generationRank*/ 0, /*expectRanks*/ {1}, /*expectPPDomain*/ 1, /*expectTPDomain*/ 1);
}
