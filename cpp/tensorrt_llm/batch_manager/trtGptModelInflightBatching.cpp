/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "trtGptModelInflightBatching.h"

#include "tensorrt_llm/batch_manager/allocateKvCache.h"
#include "tensorrt_llm/batch_manager/assignReqSeqSlots.h"
#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/capacityScheduler.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/contextProgress.h"
#include "tensorrt_llm/batch_manager/createNewDecoderRequests.h"
#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/batch_manager/guidedDecoder.h"
#include "tensorrt_llm/batch_manager/handleContextLogits.h"
#include "tensorrt_llm/batch_manager/handleGenerationLogits.h"
#include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/logitsPostProcessor.h"
#include "tensorrt_llm/batch_manager/makeDecodingBatchInputOutput.h"
#include "tensorrt_llm/batch_manager/microBatchScheduler.h"
#include "tensorrt_llm/batch_manager/pauseRequests.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/promptTuningBuffers.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/batch_manager/sequenceSlotManager.h"
#include "tensorrt_llm/batch_manager/transformerBuffers.h"
#include "tensorrt_llm/batch_manager/updateDecoderBuffers.h"
#include "tensorrt_llm/batch_manager/utils/debugUtils.h"
#include "tensorrt_llm/batch_manager/utils/inflightBatchingUtils.h"
#include "tensorrt_llm/batch_manager/utils/logitsThread.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/common/timestampUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/gptDecoderBatched.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/ipcUtils.h"
#include "tensorrt_llm/runtime/lookaheadModule.h"
#include "tensorrt_llm/runtime/memoryCounters.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/runtimeUtils.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;

using tensorrt_llm::batch_manager::CacheTransceiverFactory;

namespace tensorrt_llm::batch_manager
{

std::map<SizeType32, SizeType32> TrtGptModelInflightBatching::calculateCacheSizePerTokenForDisagg(
    ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    std::vector<SizeType32> const& maxAttentionWindowVec, bool isCrossAttention, SizeType32 kvFactor)
{
    // These are the number of attention layers on this PP rank.
    auto const numLocalAttnLayers
        = modelConfig.getNbAttentionLayers(worldConfig.getPipelineParallelism(), worldConfig.getPipelineParallelRank());
    // These are the number of attention layers on all previous PP ranks.
    auto const numLowerRankAttnLayers = modelConfig.countLowerRankLayers(ModelConfig::LayerType::kATTENTION,
        worldConfig.getPipelineParallelism(), worldConfig.getPipelineParallelRank());
    // Use global ranks of attention layers to lookup from maxAttentionWindowVec.
    auto const startAttnLayerId = numLowerRankAttnLayers;
    auto const endAttnLayerId = numLowerRankAttnLayers + numLocalAttnLayers;
    auto const numNonUniqueWindowSizes = static_cast<SizeType32>(maxAttentionWindowVec.size());
    std::map<SizeType32, std::vector<SizeType32>> uniqueWindowSizeToLayers;
    for (SizeType32 layerIdx = startAttnLayerId; layerIdx < endAttnLayerId; layerIdx++)
    {
        // maxAttentionWindowVec may or may not be stretched to the length of numLayers yet.
        // If not stretched yet, we cycle through the window sizes.
        auto const windowSize = maxAttentionWindowVec.at(layerIdx % numNonUniqueWindowSizes);
        uniqueWindowSizeToLayers[windowSize].push_back(layerIdx);
    }
    std::map<SizeType32, SizeType32> cacheSizeBytesPerTokenPerWindow;
    for (auto const& [windowSize, globalLayerIds] : uniqueWindowSizeToLayers)
    {
        auto const cacheSizePerToken = BaseKVCacheManager::calculateCacheSizePerTokenForSingleWindowSize(
            modelConfig, globalLayerIds, isCrossAttention, kvFactor);
        auto const cacheSizeBytesPerToken = cacheSizePerToken * BufferDataType(modelConfig.getKvDataType()).getSize();
        cacheSizeBytesPerTokenPerWindow[windowSize] = cacheSizeBytesPerToken;
    }

    return cacheSizeBytesPerTokenPerWindow;
};

bool TrtGptModelInflightBatching::executorConfigIsValid(
    ModelConfig const& modelConfig, executor::ExecutorConfig const& executorConfig)
{
    // Make sure logic in this function matches fixExecutorConfig
    if (executorConfig.getKvCacheConfig().getEnableBlockReuse())
    {
        if (!modelConfig.getPagedContextFMHA())
        {
            return false;
        }
        // Context logits cannot be returned for reused tokens, so disable reuse
        if (modelConfig.computeContextLogits())
        {
            return false;
        }
    }
    return true;
}

executor::ExecutorConfig TrtGptModelInflightBatching::fixExecutorConfig(
    ModelConfig const& modelConfig, executor::ExecutorConfig const& executorConfig)
{
    // Make sure logic in this function matches executorConfigIsValid
    if (executorConfig.getKvCacheConfig().getEnableBlockReuse())
    {
        auto kvCacheConfig = executorConfig.getKvCacheConfig();

        if (!modelConfig.getPagedContextFMHA())
        {
            TLLM_LOG_WARNING(
                "Fixing executorConfig: KV cache reuse disabled because model was not built with paged context FMHA "
                "support");
            kvCacheConfig.setEnableBlockReuse(false);
        }
        if (modelConfig.computeContextLogits())
        {
            TLLM_LOG_WARNING(
                "Fixing executorConfig: KV cache reuse disabled because model was built to return context logits");
            kvCacheConfig.setEnableBlockReuse(false);
        }

        auto fixedExecutorConfig = executor::ExecutorConfig(executorConfig);
        fixedExecutorConfig.setKvCacheConfig(kvCacheConfig);
        return fixedExecutorConfig;
    }
    return executorConfig;
}

TrtGptModelInflightBatching::TrtGptModelInflightBatching(std::shared_ptr<nvinfer1::ILogger> logger,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig, RawEngine const& rawEngine, bool ctxGenFusion,
    executor::ExecutorConfig const& executorConfig, bool isLeaderInOrchMode)
    : TrtGptModel(modelConfig, worldConfig, executorConfig)
    , mModelConfig(modelConfig)
    , mWorldConfig(worldConfig)
    , mDevice{runtime::utils::initDevice(worldConfig)}
    , mDecodingConfig{executorConfig.getDecodingConfig().value_or(executor::DecodingConfig{})}
    , mExtendedRuntimePerfKnobConfig{executorConfig.getExtendedRuntimePerfKnobConfig()}
    , mDebugConfig{executorConfig.getDebugConfig()}
    , mAdditionalModelOutputs{worldConfig.isLastPipelineParallelRank() ? executorConfig.getAdditionalModelOutputs()
                                                                       : std::nullopt}
    , mLogger{logger ? std::move(logger) : std::make_shared<TllmLogger>()}
    , mRuntime{std::make_unique<TllmRuntime>(rawEngine, mLogger.get(), executorConfig.getUseGpuDirectStorage(),
          executorConfig.getGpuWeightsPercent(), modelConfig.useShapeInference())}
    , mCopyBufferManager{std::make_shared<CudaStream>()}
    , mCtxGenFusion(ctxGenFusion)
    , mOperatingBeamWidth{getMaxBeamWidth()}
    , mGatherGenerationLogits{executorConfig.getGatherGenerationLogits()}
    , mPromptTableOffloading{executorConfig.getPromptTableOffloading()}
    , mIsLeaderInOrchMode{isLeaderInOrchMode}
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_LOG_INFO("gatherContextLogits: %d", mModelConfig.computeContextLogits());
    TLLM_LOG_INFO("gatherGenerationLogits: %d", getGatherGenerationLogits());

    if (!(mModelConfig.supportsInflightBatching()))
    {
        throw std::runtime_error(
            "TrtGptModelInflightBatching requires GPT attention/Mamba Conv 1d plugin with "
            "packed input and paged KV cache.");
    }
    if (mWorldConfig.isTensorParallel())
    {
        mRuntime->initializeUserBuffer(mWorldConfig, mModelConfig.getMaxBatchSize(), mModelConfig.getMaxBeamWidth(),
            mModelConfig.getMaxSequenceLen(), mModelConfig.getHiddenSize(), getMaxNumTokens());
    }
    if (mWorldConfig.isPipelineParallel())
    {
        mNumMicroBatches = mWorldConfig.getPipelineParallelism();
    }
    else
    {
        mNumMicroBatches = isTrtOverlap() ? 2 : 1;
    }

    mNumBuffers = (mCtxGenFusion ? 1 : 2) * mNumMicroBatches;

    auto const& kvCacheConfig = executorConfig.getKvCacheConfig();

    if (!kvCacheConfig.getOnboardBlocks())
    {
        TLLM_CHECK_WITH_INFO(
            !mModelConfig.getPagedContextFMHA(), "KV cache blocks need to be onboarded if context FMHA.");
    }

    if (mModelConfig.getSpeculativeDecodingMode().isDraftTokensExternal())
    {
        TLLM_CHECK_WITH_INFO(kvCacheConfig.getEnableBlockReuse(),
            "KV cache block reuse must be enabled for speculative decoding target model");
    }

    if (mCtxGenFusion)
    {
        TLLM_CHECK_WITH_INFO(!mModelConfig.isRnnBased(), "RNN based model doesn't support context generation fusion.");
        TLLM_CHECK_WITH_INFO(
            mModelConfig.isTransformerBased(), "Only transformer based model support context generation fusion now.");
    }

    if (mModelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
    {
        mSeamlessLADMaxDraftLen = modelConfig.getMaxDecodingDraftTokens();
        // TODO: enable it when speculativeDecodingMode is None and run with '--lookahead_config'
        mUseSeamlessLookahead = false;
    }

    setupSpeculativeDecodingModule(mDecodingConfig);

    if (mWorldConfig.isLastPipelineParallelRank() && executorConfig.getGuidedDecodingConfig())
    {
        mGuidedDecoder = std::make_unique<GuidedDecoder>(executorConfig.getGuidedDecodingConfig().value(),
            getMaxNumSequences(), mModelConfig.getVocabSizePadded(mWorldConfig.getSize()),
            mModelConfig.getLogitsDtype(), mRuntime->getBufferManager());
    }

    createRuntimeContexts();

    if (mWorldConfig.isTensorParallel())
    {
        createCustomAllReduceWorkspace();
    }

    if (mModelConfig.isTransformerBased())
    {
        createRuntimePerfKnobsTensor(mExtendedRuntimePerfKnobConfig);
    }

    auto& memCounter = MemoryCounters::getInstance();
    auto const gpuUsage1 = memCounter.getGpu();
    createBuffers(mDecodingConfig, mAdditionalModelOutputs);
    auto const gpuUsage2 = memCounter.getGpu();
    TLLM_LOG_INFO("[MemUsageChange] Allocated %s GPU memory for runtime buffers.",
        memCounter.bytesToString(gpuUsage2 - gpuUsage1).c_str());

    createDecoder(mDecodingConfig.getDecodingMode());
    auto const gpuUsage3 = memCounter.getGpu();
    TLLM_LOG_INFO("[MemUsageChange] Allocated %s GPU memory for decoder.",
        memCounter.bytesToString(gpuUsage3 - gpuUsage2).c_str());

    if (modelConfig.getManageWeightsType() != ModelConfig::ManageWeightsType::kDisabled)
    {
        mRuntime->loadManagedWeights(rawEngine, worldConfig.getLocalRank());
    }

    if (mModelConfig.useLoraPlugin())
    {
        auto const peftCacheManagerConfig
            = PeftCacheManagerConfig(executorConfig.getPeftCacheConfig().value_or(executor::PeftCacheConfig()));
        mPeftCacheManager = std::make_shared<PeftCacheManager>(
            peftCacheManagerConfig, mModelConfig, mWorldConfig, mRuntime->getBufferManager());
    }
    else
    {
        mPeftCacheManager = std::make_shared<NoOpPeftCacheManager>();
    }

    if (mModelConfig.isRnnBased())
    {
        createRnnStateManager();
    }
    if (mModelConfig.isTransformerBased() && modelConfig.isKVCacheEnabled())
    {
        auto cacheTransceiverConfig
            = executorConfig.getCacheTransceiverConfig().value_or(executor::CacheTransceiverConfig());

        auto const cacheSizeBytesPerTokenPerWindow = calculateCacheSizePerTokenForDisagg(
            mModelConfig, mWorldConfig, getMaxAttentionWindowVec(), mModelConfig.useCrossAttention(), 2);
        auto cacheTransPreAllocaSize = kv_cache_manager::CacheTransBufferManager::preAllocBufferSize(
            cacheSizeBytesPerTokenPerWindow, mModelConfig.getTokensPerBlock(), cacheTransceiverConfig);

        auto const [freePrimaryMemBytes, freeSecondaryMemBytes]
            = BaseKVCacheManager::calculateFreeMemBytes(mRuntime->getBufferManager(), kvCacheConfig);
        if (mModelConfig.useCrossAttention())
        {
            TLLM_CHECK_WITH_INFO(kvCacheConfig.getCrossKvCacheFraction().has_value(),
                "Must set crossKvCacheFraction for encoder-decoder model");
            auto const crossKvCacheFraction = kvCacheConfig.getCrossKvCacheFraction().value();
            mKvCacheManager = createKvCacheManager(kvCacheConfig, KvCacheType::kSELF,
                freePrimaryMemBytes * (1.0f - crossKvCacheFraction),
                freeSecondaryMemBytes * (1.0f - crossKvCacheFraction), cacheTransPreAllocaSize,
                executorConfig.getFailFastOnAttentionWindowTooLarge());
            mCrossKvCacheManager = createKvCacheManager(kvCacheConfig, KvCacheType::kCROSS,
                freePrimaryMemBytes * crossKvCacheFraction, freeSecondaryMemBytes * crossKvCacheFraction,
                cacheTransPreAllocaSize, executorConfig.getFailFastOnAttentionWindowTooLarge());
            TLLM_LOG_INFO("This is an Encoder-Decoder model, set %0.1f cross KV cache fraction based on the config.",
                crossKvCacheFraction);
        }
        else
        {
            TLLM_CHECK_WITH_INFO(!kvCacheConfig.getCrossKvCacheFraction().has_value(),
                "Do not set crossKvCacheFraction for decoder-only model");
            mKvCacheManager = createKvCacheManager(kvCacheConfig, KvCacheType::kSELF, freePrimaryMemBytes,
                freeSecondaryMemBytes, cacheTransPreAllocaSize, executorConfig.getFailFastOnAttentionWindowTooLarge());
        }

        mCacheTransceiver
            = CacheTransceiverFactory::createCacheTransceiver(mKvCacheManager.get(), mModelConfig, mWorldConfig,
                executor::kv_cache::CacheState::AttentionType::kDEFAULT, executorConfig.getCacheTransceiverConfig());
    }

    if (mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind())
    {
        TLLM_CHECK_WITH_INFO(
            mModelConfig.isKVCacheEnabled(), "When needsKVCacheRewind() returns true, KV cache needs to be enabled.");
        auto const& blockManager = mKvCacheManager->getBlockManager();

        TLLM_CHECK_WITH_INFO(blockManager.getNumPools() == 1,
            "Rewinding KV cache blocks for models with multiple pools is not supported");

        // Two "redundant" checks given the pool size check above, but those below don't rely on an implementation
        // detail I guess.
        TLLM_CHECK_WITH_INFO(
            !blockManager.isVariableWindow(), "Rewinding KV cache blocks for variable SWA models isn't supported");
        auto const maxBlocksPerSeq = blockManager.getMaxBlockPerSeqWhenSingleWindowSize();

        // TODO(oargov): VGQA is not supported, assume all layers have the same num_kv_heads
        TLLM_CHECK_WITH_INFO(
            !blockManager.isVariableGQA(), "Rewinding KV cache blocks for variable GQA models isn't supported");
        auto const numKvHeads = mModelConfig.getNbKvHeads(0);

        mRewindInputs = RewindInputs{maxBlocksPerSeq, /*isUseOneMoreBlock*/ false, numKvHeads};
    }

    if (mWorldConfig.isPipelineParallel())
    {
        mAsyncSendWaitThread = std::make_unique<tensorrt_llm::mpi::MpiWaitThread>(
            "asyncSendWaitThread",
            [this]()
            {
                mDecStepAsyncSndHdls.clear();
                mDecSlotAsyncSndHdls.clear();
            },
            [this]() { TLLM_CUDA_CHECK(cudaSetDevice(mWorldConfig.getDevice())); });

        auto const& commSession = COMM_SESSION;
        mMpiCommPipelinePara = std::make_unique<tensorrt_llm::mpi::MpiComm>(
            commSession.split(mWorldConfig.getTensorParallelRank(), mWorldConfig.getPipelineParallelRank()));
        mDecSlotAsyncSndHdls.reserve(getMaxBatchSize());
    }
    if (mWorldConfig.isTensorParallel())
    {
        auto const& commSession = COMM_SESSION;
        mMpiCommTensorPara = std::make_unique<tensorrt_llm::mpi::MpiComm>(
            commSession.split(mWorldConfig.getPipelineParallelRank(), mWorldConfig.getTensorParallelRank()));
    }

    mSeqSlotManager
        = std::make_shared<SequenceSlotManager>(getMaxNumSequences(), executorConfig.getMaxSeqIdleMicroseconds());

    mMicroBatchScheduledRequests.resize(mNumMicroBatches);
    mDecoderFinishedEvents.resize(mNumMicroBatches);
    mPeftTables.resize(mNumMicroBatches);

    if (modelConfig.isRnnBased())
    {
        TLLM_CHECK_WITH_INFO(modelConfig.getMaxBeamWidth() == 1, "RNN based model doesn't support beam search now.");
        TLLM_CHECK_WITH_INFO(
            !executorConfig.getEnableChunkedContext(), "RNN based model doesn't support Chunked Context now.");
        TLLM_CHECK_WITH_INFO(
            modelConfig.getSpeculativeDecodingMode().isNone(), "RNN based model doesn't support speculative decoding.");
    }

    std::optional<batch_scheduler::ContextChunkingConfig> ctxChunkConfig;
    if (executorConfig.getEnableChunkedContext())
    {
        TLLM_CHECK_WITH_INFO(modelConfig.isKVCacheEnabled() && mModelConfig.getPagedContextFMHA(),
            "Chunked context requires context FMHA, paged kv_cache and paged context FMHA all enabled at the same "
            "time.");
        SizeType32 chunkUnitSize = mKvCacheManager->getTokensPerBlock();
        // If sliding window attention is used, then make sure the unit size aligns with the paged context fmha's kv
        // step size.
        if (getMaxInputLen() > getMaxAttentionWindow()) // TODO(nhaber): minAttentionWindow
        {
            chunkUnitSize = std::max(/* maxKvStepSizeInFmha */ 256, chunkUnitSize);
            TLLM_LOG_INFO("ChunkUnitSize is set to %d as sliding window attention is used.", chunkUnitSize);
        }
        ctxChunkConfig = batch_scheduler::ContextChunkingConfig{
            executorConfig.getSchedulerConfig().getContextChunkingPolicy().value_or(
                executor::ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED),
            chunkUnitSize};
    }

    auto maxNumTokens = getMaxNumTokens();
    TLLM_CHECK_WITH_INFO(maxNumTokens, "Max number of tokens is not set in model config.");

    // Max context size is limited by `max_num_tokens` for chunked-context or context-FMHA,
    // or by `max_input_len` of the model.
    auto const maxContextLength = (executorConfig.getEnableChunkedContext() || mModelConfig.getContextFMHA())
        ? maxNumTokens
        : std::make_optional<SizeType32>(mModelConfig.getMaxInputLen());

    mMaxBatchSizeTunerRecommended = 0;
    mMaxBatchSizeRuntime = getMaxBatchSize();
    mMaxNumTokensStatic = maxNumTokens;
    mMaxNumTokensTunerRecommended = 0;
    mMaxNumTokensRuntime = maxNumTokens;

    if (mKvCacheManager && ctxChunkConfig)
    {
        TLLM_CHECK_WITH_INFO(ctxChunkConfig.value().chunkUnitSize % mKvCacheManager->getTokensPerBlock() == 0,
            "To prevent cache fragmentation, the context chunk unit size (%d) should be divisible by the number of "
            "tokens per kv-cache block (%d).",
            ctxChunkConfig.value().chunkUnitSize, mKvCacheManager->getTokensPerBlock());
    }

    mCapacityScheduler = std::make_unique<CapacityScheduler>(getMaxNumSequences(),
        executorConfig.getSchedulerConfig().getCapacitySchedulerPolicy(), mKvCacheManager != nullptr,
        mWorldConfig.isPipelineParallel());

    mMicroBatchScheduler = std::make_unique<MicroBatchScheduler>(ctxChunkConfig, maxContextLength);

    if (ctxChunkConfig)
    {
        if (maxContextLength)
        {
            ctxChunkConfig.value().chunkUnitSize
                = std::min(ctxChunkConfig.value().chunkUnitSize, maxContextLength.value());
        }
        TLLM_CHECK_WITH_INFO(ctxChunkConfig.value().chunkUnitSize > 0,
            "Context chunk size (%d) must be a positive integer.", maxContextLength.value());
    }
    else
    {
        if (maxContextLength && maxNumTokens)
        {
            TLLM_CHECK_WITH_INFO(maxContextLength.value() <= maxNumTokens.value(),
                "Without enabling chunked context, the max context length (%d) needs to be less than or equal to the "
                "max number of tokens (%d).",
                maxContextLength.value(), maxNumTokens.value());
        }
    }

    mPauseRequests = std::make_unique<PauseRequests>(getMaxInputLen());
    mAssignReqSeqSlots = std::make_unique<AssignReqSeqSlots>();
    mAllocateKvCache = std::make_unique<AllocateKvCache>();

    if (isCudaGraphMode())
    {
        // Limit cuda graph cache size. Depending on the model one graph is 4-10MB of GPU memory.
        SizeType32 cudaGraphCacheSize
            = std::min(getMaxBatchSize(), std::max(mExtendedRuntimePerfKnobConfig.getCudaGraphCacheSize(), 1));
        // We can't have common cache for all microbatches as cuda graph is tied to the memory pointers of the runtime
        // buffers.
        mCudaGraphExecutorCaches.resize(mNumBuffers, utils::CudaGraphExecutorCache(cudaGraphCacheSize));
    }

    mSpeculativeDecodingFastLogits
        = executorConfig.getSpecDecConfig().has_value() && executorConfig.getSpecDecConfig()->fastLogits;
    if (mSpeculativeDecodingFastLogits && modelConfig.getSpeculativeDecodingMode().isNone() && mIsLeaderInOrchMode)
    {
        mDraftModelSendLogitsThread = std::make_unique<std::thread>(&utils::draftModelSendLogitsThread, mDevice,
            &mDraftModelThreadShouldExit, &mDraftRequestsWaitingToSendLogits, mSeqSlotManager, getMaxInputLen(),
            mKvCacheManager, mCrossKvCacheManager, mPeftCacheManager);
    }

    mCreateNewDecoderRequests = std::make_unique<CreateNewDecoderRequests>(
        mSpeculativeDecodingFastLogits, mIsLeaderInOrchMode, isNormalizeLogProbs());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

TrtGptModelInflightBatching::~TrtGptModelInflightBatching()
{
    if (mCacheTransceiver)
    {
        mCacheTransceiver->checkContextTransferStatus(true);
        TLLM_CHECK_WITH_INFO(mCacheTransceiver->checkGenTransferComplete(), "Generation transfer not complete");
    }
    if (mAsyncSendWaitThread)
    {
        mAsyncSendWaitThread.reset(nullptr);
    }
    if (mDraftModelSendLogitsThread)
    {
        mDraftModelThreadShouldExit = true;
        mDraftModelSendLogitsThread->join();
        mDraftModelSendLogitsThread.reset(nullptr);
    }
}

void TrtGptModelInflightBatching::setupSpeculativeDecodingModule(executor::DecodingConfig const& decodingConfig)
{
    if (mModelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens()
        || mModelConfig.getSpeculativeDecodingMode().isEagle())
    {
        TLLM_CHECK_WITH_INFO(mCtxGenFusion, "Current speculative decoding mode requires context-gen fusion IFB");
    }

    if (mModelConfig.getSpeculativeDecodingMode().isLookaheadDecoding() && decodingConfig.getLookaheadDecodingConfig())
    {
        // FIXME choose defaults
        auto maxLookaheadConfig = decodingConfig.getLookaheadDecodingConfig().value();

        SizeType32 maxDraftTokens{0};
        SizeType32 maxDraftPathLen{0};
        std::tie(std::ignore, std::ignore, maxDraftTokens, maxDraftPathLen)
            = maxLookaheadConfig.calculateSpeculativeResource();
        TLLM_CHECK(maxDraftTokens <= mModelConfig.getMaxDecodingDraftTokens());
        mModelConfig.getSpeculativeDecodingModulePtr()->setMaxDraftTokens(maxDraftTokens);
        mModelConfig.getSpeculativeDecodingModulePtr()->setMaxDraftPathLen(maxDraftPathLen);

        auto lookaheadModulePtr
            = std::dynamic_pointer_cast<runtime::LookaheadModule>(mModelConfig.getSpeculativeDecodingModulePtr());
        lookaheadModulePtr->setExecutionConfig(maxLookaheadConfig);
    }
}

void TrtGptModelInflightBatching::reshapeKvTensors(OffsetTableDimensions const& dims)
{
    TLLM_CHECK(mBuffers.size() == static_cast<size_t>(mNumBuffers));
    auto const& manager = mRuntime->getBufferManager();
    for (auto& buffers : mBuffers)
    {
        TLLM_CHECK(buffers->transformerBuffers);
        // any method that operates on transformerBuffers must distinguish between self and cross cache, because
        // transformerBuffers is not managed by KVCacheManager same rule applies to kv pool pointers below
        buffers->transformerBuffers->reshapeKvTensors(
            getMaxBatchSize(), mOperatingBeamWidth, dims.maxBlocksPerSeq, dims.cacheType, dims.numPools, manager);
    }
}

using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;

std::pair<BlocksPerWindow, std::vector<SizeType32>>
TrtGptModelInflightBatching::clampWindowSizesToFitAtLeastOneSequence(
    BlocksPerWindow const& blocksPerWindow, bool const failFastOnAttentionWindowTooLarge)
{
    // At this point, we can only validate that the cheapest sequence in terms of kv-cache resources still fits. More
    // validation is needed on a per-request basis, once the prompt / output lengths and the actual beam width are
    // known.
    auto const promptLength = getMaxInputLen();
    auto const outputLength
        = getMaxSequenceLen() - promptLength; // This makes it the best case scenario, as context tokens are 'cheaper'
                                              // in terms of kv-cache resources on average.
    auto const sinkTokenLength = getSinkTokenLen();
    auto const maxBeamWidth = getMaxBeamWidth();
    auto const tokensPerBlock = mModelConfig.getTokensPerBlock();
    auto const& oldMaxAttentionWindowVec = getMaxAttentionWindowVec();
    std::vector<SizeType32> newMaxAttentionWindowVec;
    BlocksPerWindow newBlocksPerWindow;

    newMaxAttentionWindowVec.reserve(oldMaxAttentionWindowVec.size());
    for (auto const windowSize : oldMaxAttentionWindowVec)
    {
        auto const bestCaseBlockRequirements = kv_cache_manager::KVCacheManager::calculateMaxBlockRequirements(
            promptLength, outputLength, sinkTokenLength, windowSize, maxBeamWidth, tokensPerBlock);
        auto const [numPrimaryBlocks, numSecondaryBlocks] = blocksPerWindow.at(windowSize);
        if (bestCaseBlockRequirements > numPrimaryBlocks)
        {
            auto const newMaxAttentionWindow = KVCacheManager::calculateMaxAttentionWindow(
                promptLength, outputLength, sinkTokenLength, numPrimaryBlocks, maxBeamWidth, tokensPerBlock);
            newMaxAttentionWindowVec.push_back(newMaxAttentionWindow);
            newBlocksPerWindow[newMaxAttentionWindow] = std::make_tuple(numPrimaryBlocks, numSecondaryBlocks);
        }
        else
        {
            newMaxAttentionWindowVec.push_back(windowSize);
            newBlocksPerWindow[windowSize] = std::make_tuple(numPrimaryBlocks, numSecondaryBlocks);
        }
    }
    if (newMaxAttentionWindowVec == getMaxAttentionWindowVec())
    {
        return {blocksPerWindow, newMaxAttentionWindowVec};
    }
    TLLM_LOG_WARNING("maxAttentionWindowVec too large to fit at least one sequence in kvCache. Old: %s, New: %s",
        common::vec2str(getMaxAttentionWindowVec()).c_str(), common::vec2str(newMaxAttentionWindowVec).c_str());

    if (failFastOnAttentionWindowTooLarge)
    {
        throw std::runtime_error(
            "Attention window too large to fit even a single sequence in the KV cache. Failing fast rather than "
            "attempting an adjustment of the window sizes. "
            "Old: "
            + common::vec2str(getMaxAttentionWindowVec()) + ", New: " + common::vec2str(newMaxAttentionWindowVec));
    }

    setMaxAttentionWindowVec(newMaxAttentionWindowVec);
    if (getMaxSequenceLen() > getMaxAttentionWindow())
    {
        TLLM_LOG_WARNING("maxSequenceLen is reduced to maxAttentionWindow: %d", getMaxAttentionWindow());
        setMaxSequenceLen(getMaxAttentionWindow());
        if (getMaxInputLen() > getMaxSequenceLen() - 1)
        {
            setMaxInputLen(getMaxSequenceLen() - 1);
            TLLM_LOG_WARNING("maxInputLen is reduced to %d", getMaxInputLen());
        }
    }
    // createBuffers depends on:
    // maxAttentionWindow; maxAttentionWindowVec; maxSequenceLen;
    // TODO: This is problematic, as createBuffers edits the state of trtGptModelInflightBatching, but
    // what if there are different window values for cross+self etc. in encoder+decoder scenario...
    createBuffers(mDecodingConfig, mAdditionalModelOutputs);
    createDecoder(mDecodingConfig.getDecodingMode());
    return {newBlocksPerWindow, newMaxAttentionWindowVec};
}

std::unique_ptr<kv_cache_manager::KVCacheManager> TrtGptModelInflightBatching::createKvCacheManager(
    KvCacheConfig const& kvCacheConfig, KvCacheType kvCacheType, uint64_t freePrimaryMemBytes,
    uint64_t freeSecondaryMemBytes, size_t extraCostMemory, bool const failFastOnAttentionWindowTooLarge)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    bool isCrossAttention = kvCacheType == KvCacheType::kCROSS;
    TLLM_CHECK_WITH_INFO(
        mModelConfig.isTransformerBased(), "KvCacheManager is only needed by transformer based model.");

    auto const tokensPerBlock = mModelConfig.getTokensPerBlock();
    auto const kvDtype = mModelConfig.getKvDataType();

    // init KV cache block manager
    auto [numKvHeadsPerLayerBegin, numKvHeadsPerLayerEnd] = mModelConfig.getNumKvHeadsPerLayerLocalRange(
        mWorldConfig.getPipelineParallelism(), mWorldConfig.getPipelineParallelRank(), isCrossAttention);
    auto numKvHeadsPerLayer = std::vector<SizeType32>(numKvHeadsPerLayerBegin, numKvHeadsPerLayerEnd);

    auto maxAttentionWindowVec = getMaxAttentionWindowVec();
    if (kvCacheType != KvCacheType::kSELF) // TODO(nhaber): more foolproof way of initing cross-kvcache-manager
    {
        maxAttentionWindowVec = std::vector<SizeType32>{mModelConfig.getMaxEncoderLen()};
    }

    auto const numLayers = static_cast<SizeType32>(numKvHeadsPerLayer.size());
    auto const windowSizeToLayers = KVCacheManager::groupLayersByWindowSize(maxAttentionWindowVec, numLayers);
    auto blocksPerWindow = KVCacheManager::calculateMaxNumBlocks(kvCacheConfig, isCrossAttention, kvDtype, mModelConfig,
        mWorldConfig, windowSizeToLayers, freePrimaryMemBytes, freeSecondaryMemBytes, extraCostMemory, 2);

    // now we check if any of the window sizes is too large for at least one sequence to fit in kvCache
    // this can happen if e.g. maxSeqLen is deduced from the model and is too large
    // and user also didn't provide maxAttentionWindow, which leads it to be equal to maxSeqLen
    if (kvCacheType == KvCacheType::kSELF)
    {
        std::tie(blocksPerWindow, maxAttentionWindowVec)
            = clampWindowSizesToFitAtLeastOneSequence(blocksPerWindow, failFastOnAttentionWindowTooLarge);
    }

    kv_cache_manager::TempAttentionWindowInputs tempAttentionWindowInputs;
    tempAttentionWindowInputs.pagedContextFMHA = mModelConfig.getPagedContextFMHA();
    tempAttentionWindowInputs.maxInputLen = getMaxInputLen();
    tempAttentionWindowInputs.maxNumTokens = getMaxNumTokens().value();

    if (kvCacheType == KvCacheType::kCROSS && kvCacheConfig.getEnableBlockReuse())
    {
        TLLM_LOG_INFO(
            "Cross KV cache does not support reuse because cross attention depends on encoder and decoder input ids. "
            "Thus, KV cache reuse is disabled for cross KV cache.");
    }
    auto const enableBlockReuse = kvCacheType == KvCacheType::kSELF ? kvCacheConfig.getEnableBlockReuse() : false;
    auto const sizePerHead = mModelConfig.getSizePerHead();

    auto kvCacheManager = std::make_unique<KVCacheManager>(numKvHeadsPerLayer, sizePerHead, tokensPerBlock,
        blocksPerWindow, getMaxNumSequences(), getMaxBeamWidth(), maxAttentionWindowVec, tempAttentionWindowInputs,
        kvDtype, getSinkTokenLen(), mRuntime->getStreamPtr(),
        kvCacheType == KvCacheType::kCROSS ? mModelConfig.getMaxEncoderLen() : getMaxSequenceLen(), enableBlockReuse,
        kvCacheConfig.getOnboardBlocks(), kvCacheType, kvCacheConfig.getSecondaryOffloadMinPriority(),
        kvCacheConfig.getEventBufferMaxSize() > 0
            ? std::make_unique<kv_cache_manager::KVCacheEventManager>(kvCacheConfig.getEventBufferMaxSize())
            : nullptr,
        kvCacheConfig.getEnablePartialReuse(), kvCacheConfig.getCopyOnPartialReuse());

    reshapeKvTensors(kvCacheManager->getOffsetTableDimensions());

    kvCacheManager->allocatePools(kvCacheConfig.getUseUvm());

    TensorMap inputBuffers;
    TensorPtr poolPointers = kvCacheManager->getBlockPoolPointers();
    TensorPtr poolMapping = kvCacheManager->getLayerToPoolMapping();

    if (kvCacheType == KvCacheType::kSELF)
    {
        inputBuffers.insert_or_assign("host_kv_cache_pool_pointers", std::move(poolPointers));
        inputBuffers.insert_or_assign("host_kv_cache_pool_mapping", std::move(poolMapping));
    }
    else
    {
        inputBuffers.insert_or_assign("host_cross_kv_cache_pool_pointers", std::move(poolPointers));
        inputBuffers.insert_or_assign("host_cross_kv_cache_pool_mapping", std::move(poolMapping));
    }
    mRuntime->setStaticInputTensors(inputBuffers);

    // Emit the `created` event
    kvCacheManager->flushIterationEvents();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return kvCacheManager;
}

void TrtGptModelInflightBatching::createRnnStateManager()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(mModelConfig.isRnnBased(), "RnnStateManager is only needed by RNN based model.");

    mRnnStateManager = std::make_unique<RnnStateManager>(
        getMaxNumSequences(), mModelConfig, mWorldConfig, mRuntime->getBufferManager());

    TensorMap inputBuffers;
    mRnnStateManager->getPtrBuffers(inputBuffers, mModelConfig, mWorldConfig);
    mRuntime->setStaticInputTensors(inputBuffers);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::createCustomAllReduceWorkspace()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(mWorldConfig.isTensorParallel());

    auto const& manager = mRuntime->getBufferManager();
    auto const hiddenSize = mModelConfig.getHiddenSize();

    mAllReduceBuffers = std::make_unique<AllReduceBuffers>(getMaxBatchSize(), getMaxBeamWidth(), getMaxSequenceLen(),
        hiddenSize, manager, mWorldConfig, mRuntime->isUserBufferEnabled());

    TensorMap inputBuffers;
    inputBuffers.insert_or_assign("all_reduce_workspace", mAllReduceBuffers->mAllReduceCommPtrs);
    mRuntime->setStaticInputTensors(inputBuffers);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::createRuntimePerfKnobsTensor(
    executor::ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    SizeType32 constexpr perfKnobSize{16};
    mExtendedRuntimePerfKnobsHost = BufferManager::cpu(ITensor::makeShape({perfKnobSize}), nvinfer1::DataType::kINT64);
    auto* runtimePerfKnobsHostPtr = bufferCast<int64_t>(*mExtendedRuntimePerfKnobsHost);
    std::fill_n(runtimePerfKnobsHostPtr, perfKnobSize, -1);
    SizeType32 multiBlockModeVal = extendedRuntimePerfKnobConfig.getMultiBlockMode() ? 1 : 0;
    SizeType32 enableContextFMHAFP32AccVal = extendedRuntimePerfKnobConfig.getEnableContextFMHAFP32Acc() ? 1 : 0;
    runtimePerfKnobsHostPtr[0] = multiBlockModeVal;
    runtimePerfKnobsHostPtr[1] = enableContextFMHAFP32AccVal;

    TensorMap inputBuffers;
    inputBuffers.insert_or_assign("host_runtime_perf_knobs", mExtendedRuntimePerfKnobsHost);
    mRuntime->setStaticInputTensors(inputBuffers);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::terminateRequest(LlmRequestPtr const& llmReq, bool pause)
{
    utils::terminateRequest(
        *mSeqSlotManager, *llmReq, getMaxInputLen(), mKvCacheManager, mCrossKvCacheManager, mPeftCacheManager, pause);
}

void TrtGptModelInflightBatching::terminateRequestSync(
    LlmRequestPtr const& llmRequest, executor::FinishReason finishReason)
{
    TLLM_LOG_DEBUG("Registering termination for request %lu with finish reason %d", llmRequest->mRequestId,
        static_cast<int>(finishReason));
    mReqIdsToTerminate.try_emplace(llmRequest->mRequestId, finishReason);
}

TrtGptModelInflightBatching::IterationStatsIFB TrtGptModelInflightBatching::fillIterationStats(
    ScheduledRequests const& scheduledRequests, RequestVector const& requestsToPause)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(fillIterationStats);

    IterationStatsIFB iterationStatsIfb{mMicroBatchId};
    iterationStatsIfb.numCtxRequests = scheduledRequests.contextRequests.size();
    iterationStatsIfb.numGenRequests = scheduledRequests.generationRequests.size();
    iterationStatsIfb.avgNumDecodedTokensPerIter = 0;

    auto const contextBufferId = mCtxGenFusion ? getFusedBufferId() : getContextBufferId();
    auto const& buffers = mBuffers.at(contextBufferId);
    iterationStatsIfb.numCtxTokens = buffers->getNumContextTokens();

    for (auto const& llmReq : scheduledRequests.contextRequests)
    {
        iterationStatsIfb.scheduledRequests.insert(llmReq->mRequestId);
    }
    for (auto const& llmReq : scheduledRequests.generationRequests)
    {
        iterationStatsIfb.scheduledRequests.insert(llmReq->mRequestId);
        iterationStatsIfb.avgNumDecodedTokensPerIter += llmReq->getAvgDecodedTokensPerIter();
    }
    if (iterationStatsIfb.numGenRequests > 0)
    {
        iterationStatsIfb.avgNumDecodedTokensPerIter /= iterationStatsIfb.numGenRequests;
        TLLM_LOG_DEBUG(
            "iterationStatsIfb.avgNumDecodedTokensPerIter = %.2f", iterationStatsIfb.avgNumDecodedTokensPerIter);
    }
    for (auto const& llmReq : requestsToPause)
    {
        iterationStatsIfb.pausedRequests.insert(llmReq->mRequestId);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return iterationStatsIfb;
}

void TrtGptModelInflightBatching::forwardSync()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE_WITH_NAME(range, "TrtGptModelInflightBatching::forwardSync");

    TLLM_CUDA_CHECK(cudaSetDevice(mWorldConfig.getDevice()));

    if (!mWorldConfig.isLastPipelineParallelRank())
    {
        mAsyncSendWaitThread->waitStop();
    }

    auto& currRequests = mMicroBatchScheduledRequests.at(mMicroBatchId);

    if (!currRequests.empty())
    {
        if (!mWorldConfig.isPipelineParallel() || !mWorldConfig.isLastPipelineParallelRank())
        {
            for (auto& hdl : mDecStepAsyncSndHdls)
            {
                TLLM_CHECK_WITH_INFO(hdl.get() == nullptr, "decoderSync handle must be nullptr.");
            }
            // Wait for decoding for requests in flight for the current micro batch
            auto& decoderWaitEvent = mDecoderFinishedEvents.at(mMicroBatchId);
            mDecStepAsyncSndHdls = decoderSync(currRequests, decoderWaitEvent);
            decoderWaitEvent.reset();

            if (!mWorldConfig.isLastPipelineParallelRank())
            {
                mAsyncSendWaitThread->notifyStart();
            }
        }
        else
        {
            for (auto const& requests : {currRequests.contextRequests, currRequests.generationRequests})
            {
                for (auto const& llmReq : requests)
                {
                    for (SizeType32 beam = 0; beam < llmReq->mSamplingConfig.beamWidth; ++beam)
                    {
                        llmReq->setNumPreDecodedTokens(0, beam);
                    }
                    if (llmReq->isGenerationToCompleteState())
                    {
                        llmReq->setState(LlmRequestState::kGENERATION_COMPLETE);
                        terminateRequest(llmReq);
                    }
                }
            }
        }

        (*mPauseRequests)(currRequests.contextRequests, mInflightReqIds, mReqIdsToPause, true, *mSeqSlotManager,
            mKvCacheManager, mCrossKvCacheManager, mPeftCacheManager);
        (*mPauseRequests)(currRequests.generationRequests, mInflightReqIds, mReqIdsToPause, true, *mSeqSlotManager,
            mKvCacheManager, mCrossKvCacheManager, mPeftCacheManager);

        if (!mReqIdsToTerminate.empty())
        {
            for (auto const& requests : {currRequests.contextRequests, currRequests.generationRequests})
            {
                for (auto const& llmReq : requests)
                {
                    if (mReqIdsToTerminate.count(llmReq->mRequestId) != 0U)
                    {
                        if (!llmReq->isGenerationCompleteState())
                        {
                            TLLM_LOG_DEBUG("Terminating request %lu with finish reason %d", llmReq->mRequestId,
                                static_cast<int>(mReqIdsToTerminate[llmReq->mRequestId]));
                            terminateRequest(llmReq);
                            llmReq->finishByReason(mReqIdsToTerminate[llmReq->mRequestId]);
                            llmReq->clearGeneratedTokens();
                        }
                        mReqIdsToTerminate.erase(llmReq->mRequestId);
                    }
                }
            }
        }

        // Finished context requests have been moved to generationRequests by moveFinishedContextRequestsToGeneration
        for (auto const& llmReq : currRequests.generationRequests)
        {
            // If a context-only request is finished, send its KV cache and mark it.
            if (llmReq->isContextOnlyRequest() && llmReq->isContextFinished())
            {
                // TODO: skip if sending layer-wise
                {
                    TLLM_CHECK_WITH_INFO(mCacheTransceiver,
                        "Disaggregated serving is not enabled, please check the configuration of "
                        "cacheTransceiverConfig.");
                    mCacheTransceiver->respondAndSendAsync(llmReq.get());
                }
                mSeqSlotManager->freeSequenceSlot(llmReq->mRequestId);
            }
        }
    }
    // report profile data
    auto const bufferId = getFusedBufferId();
    auto const contextId = mBuffers[bufferId]->getContextIndex();
    if (mRuntime->hasLayerProfiler(contextId))
    {
        mRuntime->reportToProfiler(contextId);
    }
    if (mCacheTransceiver)
    {
        mCacheTransceiver->checkContextTransferStatus(0);
    }
    ++mIterCounter;

    if (mKvCacheManager)
    {
        mKvCacheManager->flushIterationEvents();
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::storeContextBlocks(std::shared_ptr<LlmRequest> const& llmReq)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // TMJ - Note
    // Make context blocks reusable immediately after context phase finishes.
    // For chunked contexts, this occurs in step that processes last context chunk.
    // isLastContextChunk() is always true for non-chunked contexts.
    // This check is made in code that calls storeContextBlocks, so omitted here.
    if (mKvCacheManager)
    {
        mKvCacheManager->storeContextBlocks(*llmReq);
    }
    if (mCrossKvCacheManager)
    {
        mCrossKvCacheManager->storeContextBlocks(*llmReq);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::storeNewBlock(std::shared_ptr<LlmRequest> const& llmReq)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // TMJ - Note
    // Make context blocks reusable immediately after each generation step.

    if (mKvCacheManager)
    {
        mKvCacheManager->storeNewBlock(*llmReq);
    }
    if (mCrossKvCacheManager)
    {
        mCrossKvCacheManager->storeNewBlock(*llmReq);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::resetIterationStats()
{
    mLastIterationStatsIFB = IterationStatsIFB{mMicroBatchId};
}

void TrtGptModelInflightBatching::forwardAsync(RequestList const& activeRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE_WITH_NAME(range, "TrtGptModelInflightBatching::forwardAsync");

    TLLM_CUDA_CHECK(cudaSetDevice(mWorldConfig.getDevice()));

    try
    {
        verifyRequests(activeRequests);
        if (mModelConfig.isTransformerBased() && getKVCacheManager() && mCacheTransceiver)
        {
            checkDisaggGenTransferStatus(activeRequests);
        }
        auto& currRequests = mMicroBatchScheduledRequests.at(mMicroBatchId);

        // Get a new set of requests for that context
        // The scheduler will not include any requests that are (i) still in encoder state if encoder-decoder models OR
        // (ii) already in flight for decoder models
        TLLM_LOG_DEBUG("Running DECODER request scheduler");
        auto [fittingRequests, fittingDisaggGenInitRequests, requestsToPause]
            = (*mCapacityScheduler)(activeRequests, mKvCacheManager, mPeftCacheManager, mCrossKvCacheManager);
        // Remove from fitting requests the requests that cannot be scheduled due to disagg KV cache transfer
        if (mModelConfig.isTransformerBased() && getKVCacheManager() && mCacheTransceiver)
        {
            prepareDisaggGenInitRequests(activeRequests, fittingDisaggGenInitRequests);
        }
        if (fittingRequests.empty() && fittingDisaggGenInitRequests.empty())
        {
            TLLM_LOG_WARNING(
                "CapacityScheduler didn't schedule any requests in iteration %lu, "
                "probably because of insufficient resources such as KV cache, "
                "will try wait for KV cache transfer to complete",
                mIterCounter);
            if (mCacheTransceiver)
            {
                mCacheTransceiver->checkContextTransferStatus(1);
                // will free kvCache in next iteration.
            }
        }
        std::tie(currRequests.contextRequests, currRequests.generationRequests)
            = (*mMicroBatchScheduler)(fittingRequests, mInflightReqIds, mMaxBatchSizeRuntime, mMaxNumTokensRuntime);
        TLLM_CHECK(currRequests.size() <= static_cast<size_t>(getMaxBatchSize()));

        (*mPauseRequests)(requestsToPause, mInflightReqIds, mReqIdsToPause, false, *mSeqSlotManager, mKvCacheManager,
            mCrossKvCacheManager, mPeftCacheManager);

        if (mUseSeamlessLookahead)
        {
            changeSpecDecMode(currRequests);
        }

        if (!currRequests.empty())
        {
            TLLM_LOG_DEBUG("Running DECODER model with batch size: %lu", currRequests.size());
            // For overlap don't store inflight requests, so they are not skipped in scheduler
            if (!isTrtOverlap())
            {
                NVTX3_SCOPED_RANGE(updateInflightReqIds);
                // Add requests to in-flight set, so they can be skipped in other micro batches
                for (auto const& requests : {currRequests.contextRequests, currRequests.generationRequests})
                {
                    for (auto const& llmReq : requests)
                    {
                        TLLM_LOG_DEBUG("request with ID %lu added to DECODER model inflight set", llmReq->mRequestId);
                        mInflightReqIds.insert(llmReq->mRequestId);
                    }
                }
            }

            (*mAssignReqSeqSlots)(*mSeqSlotManager, currRequests.contextRequests, currRequests.generationRequests);

            if (mKvCacheManager)
            {
                (*mAllocateKvCache)(*mKvCacheManager, currRequests.contextRequests, currRequests.generationRequests,
                    mModelConfig, mCrossKvCacheManager);
            }

            mPeftTables.at(mMicroBatchId)
                = mPeftCacheManager->ensureBatch(currRequests.contextRequests, currRequests.generationRequests, true);

            // Do decoder setup before context phase if model needs to setup buffers for the context phase.
            if (mModelConfig.getSpeculativeDecodingMode().needsDecoderPrologue())
            {
                auto const contextBufferId = mCtxGenFusion ? getFusedBufferId() : getContextBufferId();
                setupDecoderStep(currRequests.contextRequests, *mBuffers.at(contextBufferId),
                    mDecoderInputBuffers.at(getFusedBufferId()));
                // WAR: Sync to ensure that the decoder setup is complete before the context phase starts.
                // Without this, there may be a race condition between the decoder setup and the context phase
                // which also leads to spurious test failure in trtGptModelRealDecoderTest.
                mRuntime->getStream().synchronize();
            }
            else
            {
                prepareDistGenBufferAndDecoder(currRequests.generationRequests);
            }
            sync_check_cuda_error(mRuntime->getStream().get());

            executeBatch(currRequests);
            if (mWorldConfig.isLastPipelineParallelRank() && mGuidedDecoder)
            {
                // XGrammar: build maskcache for context requests and perform maskgen for all requests
                // These need to be overlapped with the kernel execution of forward step
                mGuidedDecoder->build(currRequests);
            }

            sync_check_cuda_error(mRuntime->getStream().get());

            // Postpone decoder setup if model does not need to setup buffers for the context phase.
            if (!mModelConfig.getSpeculativeDecodingMode().needsDecoderPrologue())
            {
                auto const contextBufferId = mCtxGenFusion ? getFusedBufferId() : getContextBufferId();
                setupDecoderStep(currRequests.contextRequests, *mBuffers.at(contextBufferId),
                    mDecoderInputBuffers.at(getFusedBufferId()));
            }

            sync_check_cuda_error(mRuntime->getStream().get());

            if (isTrtOverlap())
            {
                // WAR: Because the decoder is not stateless (yet) a sync is needed between
                // decoder execution and next decoder step preparation.
                auto const prevMicroBatchId = getPrevMicroBatchId(mMicroBatchId);
                auto& prevDecoderFinishedEvent = mDecoderFinishedEvents.at(prevMicroBatchId);
                if (prevDecoderFinishedEvent)
                {
                    prevDecoderFinishedEvent->synchronize();
                }
            }

            auto& decoderFinishedEvent = mDecoderFinishedEvents.at(mMicroBatchId);
            TLLM_CHECK_WITH_INFO(!decoderFinishedEvent.has_value(), "decoderFinishedEvent must be nullopt.");
            decoderFinishedEvent = mWorldConfig.isLastPipelineParallelRank()
                ? std::make_optional(decoderStepAsync(currRequests))
                : std::nullopt;

            sync_check_cuda_error(mRuntime->getStream().get());

            mLastIterationStatsIFB = fillIterationStats(currRequests, requestsToPause);
            for (auto const& requests : {currRequests.contextRequests, currRequests.generationRequests})
            {
                for (auto const& llmReq : requests)
                {
                    if (llmReq->isContextInitState())
                    {
                        llmReq->moveToNextContextChunk();
                        if (llmReq->getContextRemainingLength() == 0)
                        {
                            TLLM_LOG_DEBUG("[RANK %d] request with ID %lu finishes decoder ctx phase",
                                COMM_SESSION.getRank(), llmReq->mRequestId);

                            llmReq->setState(LlmRequestState::kGENERATION_IN_PROGRESS);

                            // for encoder-decoder models, free encoder output buffers after decoder context phase is
                            // completed
                            if (llmReq->getEncoderTokens().has_value())
                            {
                                llmReq->freeEncoderOutputBuffers();
                            }
                            storeContextBlocks(llmReq);

                            if (isTrtOverlap() && llmReq->willCompleteNextIteration())
                            {
                                // This prohibits the request from being scheduled for another iteration if only one
                                // iteration is expected.
                                llmReq->setState(LlmRequestState::kGENERATION_TO_COMPLETE);
                            }
                        }
                    }
                    else if (llmReq->isGenerationInProgressState())
                    {
                        storeNewBlock(llmReq);
                        TLLM_LOG_DEBUG("request with ID %lu forwards a step in decoder gen phase", llmReq->mRequestId);
                    }
                }
            }

            utils::moveFinishedContextRequestsToGeneration(currRequests);
        }
        else
        {
            mLastIterationStatsIFB = IterationStatsIFB{mMicroBatchId};
        }

        if (mWorldConfig.isPipelineParallel() && mWorldConfig.isLastPipelineParallelRank())
        {
            mAsyncSendWaitThread->waitStop();
            if (!currRequests.empty())
            {
                for (auto& hdl : mDecStepAsyncSndHdls)
                {
                    TLLM_CHECK_WITH_INFO(hdl.get() == nullptr, "decoderSync handle must be nullptr.");
                }
                // Wait for decoding for requests in flight for the current micro batch
                auto& decoderFinishedEvent = mDecoderFinishedEvents.at(mMicroBatchId);
                mDecStepAsyncSndHdls = decoderSync(currRequests, decoderFinishedEvent);
                decoderFinishedEvent.reset();

                mAsyncSendWaitThread->notifyStart();
            }
        }

        // Update the micro batch ID
        mMicroBatchId = getNextMicroBatchId(mMicroBatchId);
    }
    // In case of error, we need to free the batch slot associated with those requests
    catch (std::exception const&)
    {
        try
        {
            for (auto const& llmReq : activeRequests)
            {
                terminateRequest(llmReq);
            }
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR("forwardAsync catch-all catch block that runs `terminateRequest` has failed with:");
            TLLM_LOG_EXCEPTION(e);
            TLLM_LOG_ERROR("Rethrowing *outer* exception:");
        }
        throw;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::setRuntimeBatchSize(SizeType32 runtimeMaxBatchSize)
{
    mMaxBatchSizeTunerRecommended = runtimeMaxBatchSize;
    mMaxBatchSizeRuntime = std::min(getMaxBatchSize(), runtimeMaxBatchSize);
}

SizeType32 TrtGptModelInflightBatching::getRuntimeBatchSize() const
{
    return mMaxBatchSizeRuntime;
}

void TrtGptModelInflightBatching::setRuntimeMaxNumTokens(SizeType32 runtimeMaxNumTokens)
{
    mMaxNumTokensTunerRecommended = runtimeMaxNumTokens;
    mMaxNumTokensRuntime
        = (mMaxNumTokensStatic) ? std::min(mMaxNumTokensStatic.value(), runtimeMaxNumTokens) : runtimeMaxNumTokens;
}

void TrtGptModelInflightBatching::updatePeftCache(std::shared_ptr<LlmRequest> const& llmRequest)
{
    mPeftCacheManager->addRequestPeft(llmRequest, true);
}

runtime::BufferManager const& TrtGptModelInflightBatching::getBufferManager() const
{
    return mRuntime->getBufferManager();
}

BufferManager::CudaStreamPtr TrtGptModelInflightBatching::getRuntimeStreamPtr() const
{
    return mRuntime->getStreamPtr();
}

void TrtGptModelInflightBatching::executeContext(SizeType32 runtimeContextId, SizeType32 bufferId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(executeContext);

    auto const& currBatchState = mBuffers[bufferId]->getBatchState();

    bool hasCudaGraph = false;
    // If batch state is context only, do not capture/launch graph and execute the engine as is.
    if (isCudaGraphMode() && !currBatchState.isAnyContext())
    {
        auto cudaGraphOpt = mCudaGraphExecutorCaches[bufferId].get(currBatchState);
        // If graph exists for current batch state, launch it.
        if (cudaGraphOpt.has_value())
        {
            hasCudaGraph = true;
        }
    }

    // If there is no graph for current state, execute the engine.
    if (!hasCudaGraph)
    {
        auto enqueueSuccessful = mRuntime->executeContext(runtimeContextId);
        if (!enqueueSuccessful)
        {
            throw std::runtime_error("Executing TRT engine failed!");
        }
    }
    else
    {
        // Launch graph.
        auto cudaGraphOpt = mCudaGraphExecutorCaches[bufferId].get(currBatchState);
        cudaGraphOpt.value()->launch(mRuntime->getStream());
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::setLayerProfiler()
{
    mRuntime->setLayerProfiler();
}

std::string TrtGptModelInflightBatching::getLayerProfileInfo() const
{
    return mRuntime->getLayerProfileInfo();
}

void TrtGptModelInflightBatching::verifyRequests(RequestList const& activeRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(verifyRequests);

    if (activeRequests.empty())
    {
        return;
    }

    auto const& firstRequest = activeRequests.front();
    auto const firstRequestId = firstRequest->mRequestId;
    auto const firstBeamWidth = firstRequest->mSamplingConfig.beamWidth;

    for (auto const& llmReq : activeRequests)
    {
        auto const beamWidth = llmReq->mSamplingConfig.beamWidth;
        auto const draftLength = llmReq->getNumDraftTokens();
        auto const maxDraftLength = mModelConfig.getMaxDecodingDraftTokens();

        TLLM_CHECK_WITH_INFO(beamWidth == 1 || draftLength == 0, "Can't use speculative decoding with beam search.");
        TLLM_CHECK_WITH_INFO(draftLength <= maxDraftLength,
            "Number of draft tokens (%d) is larger than maximum number of draft tokens (%d)", draftLength,
            maxDraftLength);

        // FIXME: Remove this check when varying beam width is supported
        {
            TLLM_CHECK_WITH_INFO(beamWidth == firstBeamWidth,
                "All active requests must have same beam width, "
                "but request %lu with beam width %d differs from first request %lu with beam width %d",
                llmReq->mRequestId, beamWidth, firstRequestId, firstBeamWidth);
        }
    }

    if (firstBeamWidth != mOperatingBeamWidth)
    {
        changeBeamWidth(firstBeamWidth);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::executeBatch(ScheduledRequests const& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(executeBatch);

    if (!mCtxGenFusion)
    {
        if (!scheduledRequests.contextRequests.empty())
        {
            auto const bufferId = getContextBufferId();
            executeStep(scheduledRequests.contextRequests, {}, bufferId);
        }
        if (!scheduledRequests.generationRequests.empty())
        {
            auto const bufferId = getGenerationBufferId();
            executeStep({}, scheduledRequests.generationRequests, bufferId);
        }
    }
    else
    {
        auto const bufferId = getFusedBufferId();
        executeStep(scheduledRequests.contextRequests, scheduledRequests.generationRequests, bufferId);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::createRuntimeContexts()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    mRuntime->clearContexts();
    auto const numProfiles = mRuntime->getNbProfiles();
    for (auto i = 0; i < numProfiles; ++i)
    {
        mRuntime->addContext(i);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

namespace
{
// TODO: move this somewhere else?
/**
 * This function logic is also implemented in tensorrt_llm/_torch/pyexecutor/_util.py get_decoding_mode().
 */
executor::DecodingMode getDecodingMode(SpeculativeDecodingMode specDecodingMode,
    std::optional<executor::DecodingMode> const& decodingModeOpt, runtime::SizeType32 const beamWidth)
{
    auto getDefaultDecodingMode = [beamWidth](std::optional<executor::DecodingMode> const& decodingModeOpt)
    {
        if (decodingModeOpt.has_value() && !decodingModeOpt->isAuto())
        {
            return decodingModeOpt.value();
        }
        return (beamWidth == 1) ? executor::DecodingMode::TopKTopP() : executor::DecodingMode::BeamSearch();
    };

    auto decodingMode = getDefaultDecodingMode(decodingModeOpt);
    // Variable-Beam-Width-Search (special mode of Beam-Search) is enabled.
    if (decodingMode.isBeamSearch() && decodingMode.isUseVariableBeamWidthSearch())
    {
        TLLM_LOG_INFO("Variable-Beam-Width-Search is enabled");
    }
    // Overwrite decoding mode when beam width is one.
    if (beamWidth == 1 && decodingMode.isBeamSearch())
    {
        TLLM_LOG_WARNING(
            "Beam width is set to 1, but decoding mode is BeamSearch. Overwriting decoding mode to TopKTopP.");
        decodingMode = executor::DecodingMode::TopKTopP();
    }
    // Overwrite decoding mode when Medusa is used.
    if (specDecodingMode.isMedusa() && !decodingMode.isMedusa())
    {
        TLLM_LOG_WARNING("Model is Medusa, but decoding mode is not Medusa. Overwriting decoding mode to Medusa.");
        decodingMode = executor::DecodingMode::Medusa();
    }
    // Overwrite decoding mode when Medusa is not used.
    if (!specDecodingMode.isMedusa() && decodingMode.isMedusa())
    {
        TLLM_LOG_WARNING("Model is not Medusa, but decoding mode is Medusa. Overwriting decoding mode.");
        decodingMode = getDefaultDecodingMode(decodingModeOpt);
    }
    // Overwrite decoding mode when lookahead decoding is used.
    if (specDecodingMode.isLookaheadDecoding() && !decodingMode.isLookahead())
    {
        TLLM_LOG_WARNING(
            "Model is Lookahead, but decoding mode is not Lookahead. Overwriting decoding mode to Lookahead.");
        decodingMode = executor::DecodingMode::Lookahead();
    }
    // Overwrite decoding mode when lookahead decoding is not used.
    if (!specDecodingMode.isLookaheadDecoding() && decodingMode.isLookahead())
    {
        TLLM_LOG_WARNING(
            "Model is not built with Lookahead decoding, but decoding mode is Lookahead. Overwriting decoding "
            "mode.");
        decodingMode = getDefaultDecodingMode(decodingModeOpt);
    }
    // Overwrite decoding mode when 'explicit draft tokens' is used.
    if (specDecodingMode.isExplicitDraftTokens() && !decodingMode.isExplicitDraftTokens())
    {
        TLLM_LOG_WARNING(
            "Model is built with 'explicit draft tokens' decoding, but decoding mode is something else. Overwriting "
            "decoding mode.");
        decodingMode = executor::DecodingMode::ExplicitDraftTokens();
    }
    // Overwrite decoding mode when 'explicit draft tokens' is not used.
    if (!specDecodingMode.isExplicitDraftTokens() && decodingMode.isExplicitDraftTokens())
    {
        TLLM_LOG_WARNING(
            "Model is not built with 'explicit draft tokens' decoding, but decoding mode is set to it. Overwriting "
            "decoding "
            "mode to default.");
        decodingMode = getDefaultDecodingMode(decodingModeOpt);
    }
    // Overwrite decoding mode when EAGLE is used.
    if (specDecodingMode.isEagle() && !decodingMode.isEagle())
    {
        TLLM_LOG_WARNING("Model is Eagle, but decoding mode is not Eagle. Overwriting decoding mode to Eagle.");
        decodingMode = executor::DecodingMode::Eagle();
    }
    // Overwrite decoding mode when Eagle is not used.
    if (!specDecodingMode.isEagle() && decodingMode.isEagle())
    {
        TLLM_LOG_WARNING("Model is not Eagle, but decoding mode is Eagle. Overwriting decoding mode.");
        decodingMode = getDefaultDecodingMode(decodingModeOpt);
    }
    if (specDecodingMode.isDraftTokensExternal())
    {
        TLLM_LOG_WARNING("Overwriting decoding mode to external draft token");
        decodingMode = executor::DecodingMode::ExternalDraftTokens();
    }
    TLLM_LOG_DEBUG("DecodingMode: %s", decodingMode.getName());
    return decodingMode;
}
} // namespace

void TrtGptModelInflightBatching::createDecoder(std::optional<executor::DecodingMode> const& decodingModeOpt)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mDecoderState = std::make_unique<runtime::decoder::DecoderState>();

    if (mWorldConfig.isLastPipelineParallelRank())
    {
        auto decoderType = mRuntime->getEngine().getTensorDataType("logits");

        auto const decodingMode
            = getDecodingMode(mModelConfig.getSpeculativeDecodingMode(), decodingModeOpt, mOperatingBeamWidth);

        if (decodingMode.isExplicitDraftTokens())
        {
            // There are no logits in Explicit draft tokens model.
            decoderType = mModelConfig.getDataType();
            // Decoder is not instantiated for bf16. We use half to get the same data size
            // and explicitly pass dtype to redrafter that has bf16 kernels.
            if (decoderType == nvinfer1::DataType::kBF16)
            {
                decoderType = nvinfer1::DataType::kHALF;
            }
        }

        mDecoder = std::make_unique<runtime::GptDecoderBatched>(mRuntime->getStreamPtr());
        mDecoder->setup(
            decodingMode, getMaxNumSequences(), mOperatingBeamWidth, decoderType, mModelConfig, mWorldConfig);

        mDecoderState->setup(getMaxNumSequences(), mOperatingBeamWidth, getMaxAttentionWindow(), getSinkTokenLen(),
            getMaxSequenceLen(), decoderType, mModelConfig, mWorldConfig, mRuntime->getBufferManager());

        if (!mModelConfig.getSpeculativeDecodingMode().isNone())
        {
            mDecoderState->setupSpeculativeDecoding(mModelConfig.getSpeculativeDecodingMode(),
                mModelConfig.getMaxDecodingTokens(), decoderType, mModelConfig, mWorldConfig,
                mRuntime->getBufferManager());
        }
    }
    else
    {
        mDecoderState->setupCacheIndirection(
            getMaxNumSequences(), mOperatingBeamWidth, getMaxAttentionWindow(), mRuntime->getBufferManager());
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::createBuffers(executor::DecodingConfig const& decodingConfig,
    std::optional<std::vector<executor::AdditionalModelOutput>> const& additionalModelOutputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mBuffers.clear();
    for (SizeType32 i = 0; i < mNumBuffers; ++i)
    {
        mBuffers.emplace_back(
            std::make_unique<RuntimeBuffers>(getMaxBatchSize(), mOperatingBeamWidth, getMaxAttentionWindowVec(),
                getMaxAttentionWindow(), getSinkTokenLen(), *mRuntime, mModelConfig, mWorldConfig, decodingConfig,
                getGatherGenerationLogits(), getMaxNumTokens(), additionalModelOutputs, mPromptTableOffloading));
    }

    mDecoderInputBuffers.clear();
    mDecoderOutputBuffers.clear();
    for (SizeType32 i = 0; i < mNumMicroBatches; ++i)
    {
        mDecoderInputBuffers.emplace_back(
            getMaxBatchSize(), mModelConfig.getMaxDecodingTokens(), mRuntime->getBufferManager());
        mDecoderInputBuffers.back().setupMedusaLogits(getMaxNumSequences(), mModelConfig);
        mDecoderOutputBuffers.emplace_back(getMaxNumSequences(), mOperatingBeamWidth, getMaxSequenceLen(),
            mModelConfig.getMaxDecodingTokens(), mRuntime->getBufferManager());
        mDecoderOutputBuffers.back().setupSpeculativeDecoding(
            getMaxNumSequences(), mModelConfig.getMaxDecodingTokens(), mModelConfig);
    }

    mSlotDecoderBuffers.clear();
    for (SizeType32 i = 0; i < getMaxNumSequences(); ++i)
    {
        mSlotDecoderBuffers.emplace_back(std::make_unique<SlotDecoderBuffers>(
            mOperatingBeamWidth, getMaxSequenceLen(), mRuntime->getBufferManager()));
    }

    mDecodingInputs.resize(mNumMicroBatches);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::prepareDisaggGenInitRequests(
    RequestList const& activeRequests, RequestVector& newGenReqs)
{
    NVTX3_SCOPED_RANGE(prepareDisaggGenInitRequests);

    // Allocate KV cache by treating them as context requests
    (*mAllocateKvCache)(*mKvCacheManager, newGenReqs, {}, mModelConfig, mCrossKvCacheManager);

    // Initiate KV cache transfer
    auto timeStart = std::chrono::steady_clock::now();

    if (tc::getEnvDisaggBenchmarkGenOnly())
    {
        TLLM_LOG_DEBUG("Disaggregated generation only benchmark mode is enabled");
        for (auto& req : newGenReqs)
        {
            req->setState(LlmRequestState::kDISAGG_GENERATION_TRANS_COMPLETE);
        }
        return;
    }

    auto const genInitReqNum = std::count_if(activeRequests.begin(), activeRequests.end(),
        [](auto const& req) { return req->isDisaggGenerationInitState(); });

    // Loop over the new disagg gen requests and trigger receive of KV cache
    for (auto& newGenReq : newGenReqs)
    {
        TLLM_CHECK_WITH_INFO(
            mCacheTransceiver, "Disaggregated serving is not enabled, please check the configuration.");
        if (common::getEnvDisableKVCacheTransferOverlap())
        {
            mCacheTransceiver->requestAndReceiveSync(newGenReq.get());
        }
        else
        {
            mCacheTransceiver->requestAndReceiveAsync(newGenReq.get());
        }
    }
    if (!common::getEnvDisableKVCacheTransferOverlap())
    {
        auto const blockTransfer = std::all_of(activeRequests.begin(), activeRequests.end(),
            [](auto const& req) { return req->isDisaggGenerationTransmissionInProgress(); });
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            "newGenReqs.size():%ld requests, activeRequests.size():%ld checkGenTransferStatus :%d original "
            "gen_only_requests_num:%ld",
            newGenReqs.size(), activeRequests.size(), blockTransfer, genInitReqNum);
        mCacheTransceiver->checkGenTransferStatus(blockTransfer ? 1 : 0);
        auto timeEnd = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<float, std::milli>(timeEnd - timeStart).count();
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            "receiveDisaggGenCache time:%f ms, "
            "blockTransfer:%d,genInitReqNum:%ld,newGenReqs.size():%ld,activeRequests.size():%ld",
            duration, blockTransfer, genInitReqNum, newGenReqs.size(), activeRequests.size());
    }

    return;
}

void TrtGptModelInflightBatching::checkDisaggGenTransferStatus(RequestList const& activeRequests)
{
    NVTX3_SCOPED_RANGE(checkDisaggGenTransferStatus);

    if (common::getEnvDisableKVCacheTransferOverlap())
    {
        return;
    }

    auto timeStart = std::chrono::steady_clock::now();

    // TODO:
    auto const needCheck = std::any_of(activeRequests.begin(), activeRequests.end(),
        [](auto const& req) { return req->isDisaggGenerationTransmissionInProgress(); });

    if (needCheck)
    {
        auto const needCheckOne = std::all_of(activeRequests.begin(), activeRequests.end(),
            [](auto const& req) { return req->isDisaggGenerationTransmissionInProgress(); });

        int atLeastNum = needCheckOne ? 1 : 0;
        TLLM_LOG_DEBUG(
            mpi::MpiComm::world().getRank(), "noPreppared requests, checkGenTransferStatus atLeastNum:%d", atLeastNum);

        mCacheTransceiver->checkGenTransferStatus(atLeastNum);

        auto timeEnd = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<float, std::milli>(timeEnd - timeStart).count();
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            "no Prepare checkDisaggGenTransferStatus time:%f ms, "
            "needCheckOne:%d,needCheck:%ld,activeRequests.size():%ld",
            duration, needCheckOne, needCheck, activeRequests.size());
    }
}

void TrtGptModelInflightBatching::prepareDistGenBufferAndDecoder(RequestVector const& generationRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // set decoderStep for disagg_generation
    RequestVector cacheTransCompleteRequests;
    for (auto const& request : generationRequests)
    {
        if (request->isDisaggGenerationTransmissionComplete())
        {
            cacheTransCompleteRequests.push_back((request));
        }
    }
    if (!cacheTransCompleteRequests.empty())
    {
        auto timeStart = std::chrono::steady_clock::now();
        auto const bufferId = getFusedBufferId();
        auto& runtimeBuffers = *mBuffers[bufferId];
        runtimeBuffers.prepareStep(cacheTransCompleteRequests, {}, getMaxBeamWidth(), getMaxAttentionWindow(),
            *mDecoderState, mKvCacheManager.get(), mCrossKvCacheManager.get(), mRnnStateManager.get(),
            mPeftTables[mMicroBatchId], *mRuntime, mModelConfig, mWorldConfig, getGatherGenerationLogits(),
            isTrtOverlap());
        auto const contextBufferId = mCtxGenFusion ? getFusedBufferId() : getContextBufferId();
        setupDecoderStep(
            cacheTransCompleteRequests, *mBuffers.at(contextBufferId), mDecoderInputBuffers.at(getFusedBufferId()));
        sync_check_cuda_error(mRuntime->getStream().get());
        auto timeEnd = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<float, std::milli>(timeEnd - timeStart).count();
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            "prepareDistGenBufferAndDecoder time:%f ms , cacheTransCompleteRequests.size():%ld", duration,
            cacheTransCompleteRequests.size());
    }
    for (auto& request : cacheTransCompleteRequests)
    {
        request->setState(LlmRequestState::kGENERATION_IN_PROGRESS);
        request->setContextCurrentPosition(request->mPromptLen);
        request->setDecodingIter(1);
        auto const reqBeamWidth = request->mSamplingConfig.beamWidth;
        auto firstGenTokens = request->getContextPhaseParams().value().getFirstGenTokens();
        for (SizeType32 beam = 0; beam < reqBeamWidth; ++beam)
        {
            request->addNewToken(firstGenTokens.at(beam), beam);
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::debugIOTensors(RequestVector const& contextRequests,
    RequestVector const& generationRequests, TensorMap const& inputMap, TensorMap const& outputMap)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(mDebugConfig);

    auto const& manager = mRuntime->getBufferManager();
    auto requestIds = utils::collectRequestIds(contextRequests, generationRequests);

    if (mDebugConfig->getDebugTensorsMaxIterations() > 0)
    {
        mLastIterationDebugTensors.clear();
        mLastIterationDebugTensors = utils::storeIOTensors(*mDebugConfig, requestIds, inputMap, outputMap, manager);
    }
    else
    {
        utils::dumpIOTensors(*mDebugConfig, mIterCounter, requestIds, inputMap, outputMap, mWorldConfig, manager);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::tuple<SizeType32, runtime::StringPtrMap<runtime::ITensor> const&, runtime::StringPtrMap<runtime::ITensor>&>
TrtGptModelInflightBatching::prepareBuffers(
    RequestVector const& contextRequests, RequestVector const& generationRequests, SizeType32 bufferId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(prepareBuffers);

    auto& runtimeBuffers = *mBuffers.at(bufferId);

    auto allNewTokens = mWorldConfig.isLastPipelineParallelRank()
        ? RuntimeBuffers::OptionalRef<runtime::ITensor const>(mDecoderState->getAllNewTokens())
        : std::nullopt;

    auto [optProfileId, inputMap, outputMap] = runtimeBuffers.prepareStep(contextRequests, generationRequests,
        mOperatingBeamWidth, getMaxAttentionWindow(), *mDecoderState, mKvCacheManager.get(), mCrossKvCacheManager.get(),
        mRnnStateManager.get(), mPeftTables[bufferId], *mRuntime, mModelConfig, mWorldConfig,
        getGatherGenerationLogits(), isTrtOverlap(), allNewTokens);

    // For Variable-Beam-Width-Search
    mRuntime->setCurrentBeamWidths(
        tensorrt_llm::batch_manager::utils::getRequestBeamWidths(contextRequests, generationRequests));

    mRuntime->setInputTensors(optProfileId, inputMap);
    mRuntime->setOutputTensors(optProfileId, outputMap);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return {optProfileId, inputMap, outputMap};
}

void TrtGptModelInflightBatching::prepareGraph(SizeType32 bufferId, SizeType32 optProfileId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(prepareGraph);

    auto const nextBatchState = mBuffers[bufferId]->getBatchState();
    auto cudaGraphOpt = mCudaGraphExecutorCaches[bufferId].get(nextBatchState);
    // If graph is not found in the cache, capture it.
    if (!cudaGraphOpt.has_value())
    {
        // We need to prepare some tensors once again to properly set values for graph capture.
        // Graph capture requires setting some tensors (e.g. past_kv_len)
        // to the round_up(max_kv_cache_len, kKV_CACHE_LEN_CUDA_GRAPH_ROUND_SIZE)
        // in order to capture the kernels with the large enough grid.
        mBuffers[bufferId]->prepareBuffersForCudaGraph(getMaxSequenceLen());

        auto cudaGraph = std::make_shared<utils::CudaGraphExecutor>();
        cudaGraph->prepareNextGraph(mRuntime, optProfileId);
        mCudaGraphExecutorCaches[bufferId].put(nextBatchState, cudaGraph);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::executeStep(
    RequestVector const& contextRequests, RequestVector const& generationRequests, SizeType32 bufferId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE_WITH_NAME(range,
        "executeStep: " + std::to_string(contextRequests.size()) + " ctx reqs, "
            + std::to_string(generationRequests.size()) + " gen reqs");

    if (mPromptTableOffloading)
    {
        prefetchNextPromptTableChunk(contextRequests, /* isFirstChunk */ true, bufferId);
    }

    auto [optProfileId, inputMap, outputMap] = prepareBuffers(contextRequests, generationRequests, bufferId);

    if (mBuffers[bufferId]->transformerBuffers)
    {
        // Creation of context progress, or remains nullptr if not needed
        std::shared_ptr<ContextProgress> progress = nullptr;
        RequestVector layerWiseRequests;
        if (common::getEnvDisaggLayerwise())
        {
            for (auto const& request : contextRequests)
            {
                bool const enableLayerWise = request->isContextOnlyRequest() && request->isLastContextChunk();
                if (enableLayerWise)
                {
                    layerWiseRequests.push_back(request);
                }
            }
        }
        // TODO: support layer-wise cross kv cache in encoder-decoder models
        if (!layerWiseRequests.empty() && !mModelConfig.useCrossAttention())
        {
            int const numLayers = mModelConfig.getNbAttentionLayers(
                mWorldConfig.getPipelineParallelism(), mWorldConfig.getPipelineParallelRank());
            progress = std::make_shared<ContextProgress>(numLayers);
        }
        bufferCast<void*>(*mBuffers[bufferId]->transformerBuffers->contextProgressHost)[0] = progress.get();
        if (progress)
        {
            TLLM_CHECK_WITH_INFO(mCacheTransceiver,
                "Disaggregated serving is not enabled, please check the configuration of cacheTransceiverConfig.");
            mCacheTransceiver->respondAndSendLayerWise(layerWiseRequests, progress);
        }
    }

    if (mPromptTableOffloading)
    {
        prefetchNextPromptTableChunk(contextRequests, /* isFirstChunk */ false, bufferId);
    }

    executeContext(optProfileId, bufferId);

    // If batch state has any context request, do not capture this graph.
    if (isCudaGraphMode() && contextRequests.empty())
    {
        // Capture graph of current batch state during engine execution.
        // This is based on the assumptions that
        // a) We can hide CPU graph capture behind the GPU engine execution.
        // b) Batch size in the next iterations won't change and we can reuse the graph multiple times.
        prepareGraph(bufferId, optProfileId);
    }

    if (mDebugConfig)
    {
        debugIOTensors(contextRequests, generationRequests, inputMap, outputMap);
    }

    if (mAdditionalModelOutputs.has_value() && !mAdditionalModelOutputs.value().empty())
    {
        utils::copyAdditionalOutputs(
            mAdditionalModelOutputs.value(), contextRequests, generationRequests, outputMap, getBufferManager());
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::setupDecoderStep(
    RequestVector const& contextRequests, RuntimeBuffers const& buffers, DecoderInputBuffers& inputBuffers)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(setupDecoderStep);

    if (mWorldConfig.isLastPipelineParallelRank() && !contextRequests.empty())
    {
        auto const logitsType = mRuntime->getEngine().getTensorDataType("logits");

        auto [batchSlots, samplingConfigs, lookaheadPrompt, lookaheadAlgoConfigs]
            = (*mCreateNewDecoderRequests)(mModelConfig, mWorldConfig, mDecodingConfig, contextRequests, logitsType,
                inputBuffers, *mDecoderState, mRuntime->getStream(), *mDecoder->getDecoderStream(), getMaxSequenceLen(),
                mOperatingBeamWidth, buffers.mMedusaBuffers);

        auto const localBatchSize = batchSlots->getSize();
        if (localBatchSize > 0)
        {
            auto samplingConfig = SamplingConfig(samplingConfigs);
            mDecoder->getUnderlyingDecoder().setup(samplingConfig, localBatchSize, batchSlots,
                {mDecoderState->getJointDecodingOutput()}, mModelConfig.getDataType(), lookaheadPrompt,
                lookaheadAlgoConfigs);

            auto const& stream = mDecoder->getDecoderStream();
            CudaEvent event{};
            stream->record(event);
            mRuntime->getStreamPtr()->wait(event);
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::postProcessRequest(
    LlmRequest& llmReq, std::vector<SizeType32> const& numDroppedTokens)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const seqSlot = llmReq.mSeqSlot.value();
    auto const reqBeamWidth = llmReq.getBeamWidthByIter(true);
    auto const& bufferManager = getBufferManager();

    if (llmReq.getReturnGenerationLogits() && !llmReq.getGenerationLogitsFragments().empty())
    {
        TLLM_CHECK(!llmReq.isStreaming());
        auto const genBufferId = mCtxGenFusion ? getFusedBufferId() : getGenerationBufferId();
        auto& genRuntimeBuffers = *mBuffers.at(genBufferId);

        auto constexpr beforeDecoder = false;
        utils::copyGenerationLogits(
            genRuntimeBuffers.generationLogitsCache, bufferManager, llmReq, beforeDecoder, numDroppedTokens);

        bufferManager.getStream().synchronize();
    }

    if (reqBeamWidth == 1)
    {
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return;
    }

    // Update mDecoderBuffers->slotOutputIdsHost and synchronize
    getDecoderSlotHostOutputs(seqSlot, llmReq.returnLogProbs(), llmReq.mSamplingConfig, llmReq.isStreaming());

    auto const* outputIdsHostData = bufferCast<TokenIdType>(*mSlotDecoderBuffers[seqSlot]->outputIdsHost);
    auto const* sequenceLengthsHostData = bufferCast<SizeType32>(*mSlotDecoderBuffers[seqSlot]->sequenceLengthsHost);
    auto const* cumLogProbsHostData = bufferCast<float>(*mSlotDecoderBuffers[seqSlot]->cumLogProbsHost);
    auto logProbsHost = mSlotDecoderBuffers[seqSlot]->logProbsHost;
    auto const* logProbsHostData = bufferCast<float>(*logProbsHost);

    auto const& outputIdsShape = mSlotDecoderBuffers[seqSlot]->outputIdsHost->getShape();
    auto const maxSeqLength = outputIdsShape.d[1];

    std::vector<std::vector<TokenIdType>> generatedTokens(reqBeamWidth);
    for (SizeType32 beam = 0; beam < reqBeamWidth; ++beam)
    {
        auto const* const begin = outputIdsHostData + tc::flat_index2(beam, llmReq.mPromptLen, maxSeqLength);
        auto const generatedLength = sequenceLengthsHostData[beam] - llmReq.mPromptLen;
        auto const* const end = begin + generatedLength;
        generatedTokens[beam].assign(begin, end);

        if (llmReq.returnLogProbs())
        {
            llmReq.setCumLogProb(cumLogProbsHostData[beam], beam);

            auto const beginLogProbsOffset = reqBeamWidth == 1 ? llmReq.mPromptLen : 0;
            auto const* const begin = logProbsHostData + beam * logProbsHost->getShape().d[1] + beginLogProbsOffset;
            auto const* const end = begin + generatedLength;
            LlmRequest::VecLogProbs logProbs(begin, end);
            llmReq.setLogProbs(logProbs, beam);
        }
    }

    // store the generated tokens into the mTokensGathered buffer
    llmReq.setGeneratedTokens(generatedTokens);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::getDecoderSlotHostOutputs(
    SizeType32 seqSlot, bool returnLogProbs, SamplingConfig const& samplingConfig, bool streaming)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mWorldConfig.isLastPipelineParallelRank())
    {
        auto event = mDecoder->finalize(*mDecoderState, seqSlot, samplingConfig, streaming);
        // Make sure that postprocessing is done before copying outputIds
        mCopyBufferManager.getStream().wait(event.get());

        auto sequenceLengths = mDecoderState->getSequenceLengths(seqSlot);
        auto outputIds = mDecoderState->getGatheredIds(seqSlot);
        auto cumLogProbs = mDecoderState->getCumLogProbs(seqSlot);
        auto logProbs = mDecoderState->getLogProbs(seqSlot);

        mCopyBufferManager.copy(*sequenceLengths, *mSlotDecoderBuffers[seqSlot]->sequenceLengths);
        mCopyBufferManager.copy(*outputIds, *mSlotDecoderBuffers[seqSlot]->outputIds);
        if (returnLogProbs)
        {
            mCopyBufferManager.copy(*cumLogProbs, *mSlotDecoderBuffers[seqSlot]->cumLogProbs);
            mCopyBufferManager.copy(*logProbs, *mSlotDecoderBuffers[seqSlot]->logProbs);
        }

        if (mWorldConfig.isPipelineParallel())
        {
            // Make sure that postprocessing is done before sending outputIds
            event.synchronize();

            auto const peerSend = 0;
            mDecSlotAsyncSndHdls.emplace_back(std::make_unique<DecoderSlotAsyncSend>(
                outputIds, sequenceLengths, cumLogProbs, logProbs, returnLogProbs, *mMpiCommPipelinePara, peerSend));
        }
    }
    else
    {
        auto const peerRecv = mWorldConfig.getPipelineParallelRank() == 0 ? mWorldConfig.getPipelineParallelism() - 1
                                                                          : mWorldConfig.getPipelineParallelRank() - 1;
        DecoderSlotAsyncSend::recv(*mSlotDecoderBuffers[seqSlot], returnLogProbs, *mMpiCommPipelinePara, peerRecv);

        auto const peerSend = mWorldConfig.getPipelineParallelRank() + 1;
        if (peerSend != mWorldConfig.getPipelineParallelism() - 1)
        {
            mDecSlotAsyncSndHdls.emplace_back(std::make_unique<DecoderSlotAsyncSend>(
                *mSlotDecoderBuffers[seqSlot], returnLogProbs, *mMpiCommPipelinePara, peerSend));
        }
    }
    sync_check_cuda_error(mRuntime->getStream().get());

    // Here copy stream is synchronized after receiving decoderSlotOutputIdsView either by copy or by receive
    // before copying to host on copy stream
    runtime::CudaEvent beforeEvent{};
    mRuntime->getStreamPtr()->record(beforeEvent);
    mCopyBufferManager.getStream().wait(beforeEvent);
    mCopyBufferManager.copy(*mSlotDecoderBuffers[seqSlot]->outputIds, *mSlotDecoderBuffers[seqSlot]->outputIdsHost);
    mCopyBufferManager.copy(
        *mSlotDecoderBuffers[seqSlot]->sequenceLengths, *mSlotDecoderBuffers[seqSlot]->sequenceLengthsHost);

    if (returnLogProbs)
    {
        mCopyBufferManager.copy(
            *mSlotDecoderBuffers[seqSlot]->cumLogProbs, *mSlotDecoderBuffers[seqSlot]->cumLogProbsHost);
        mCopyBufferManager.copy(*mSlotDecoderBuffers[seqSlot]->logProbs, *mSlotDecoderBuffers[seqSlot]->logProbsHost);
    }

    // Make sure copy is done before continuing on host
    mCopyBufferManager.getStream().synchronize();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

namespace
{
// Check if one of the request needs log probs, need to get from decoder and communicate
bool batchReturnLogProbs(ScheduledRequests const& scheduledRequests)
{
    auto pred = [](auto const& llmReq) { return llmReq->returnLogProbs(); };
    return std::any_of(scheduledRequests.contextRequests.begin(), scheduledRequests.contextRequests.end(), pred)
        || std::any_of(scheduledRequests.generationRequests.begin(), scheduledRequests.generationRequests.end(), pred);
}
} // namespace

runtime::CudaEvent TrtGptModelInflightBatching::decoderStepAsync(ScheduledRequests const& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(decoderStepAsync);

    auto& decoderInputBuffers = mDecoderInputBuffers.at(getFusedBufferId());

    auto const contextBufferId = mCtxGenFusion ? getFusedBufferId() : getContextBufferId();
    auto& contextRuntimeBuffers = mBuffers.at(contextBufferId);
    auto const logitsIndex = (*mHandleContextLogits)(decoderInputBuffers, scheduledRequests.contextRequests,
        contextRuntimeBuffers->logits, contextRuntimeBuffers->numContextLogits, mModelConfig,
        mRuntime->getBufferManager(), contextRuntimeBuffers->mMedusaBuffers);

    auto const genLogitsIndex = mCtxGenFusion ? logitsIndex : 0;
    auto const genBufferId = mCtxGenFusion ? getFusedBufferId() : getGenerationBufferId();
    auto& genRuntimeBuffers = mBuffers.at(genBufferId);
    (*mHandleGenerationLogits)(decoderInputBuffers, scheduledRequests.generationRequests, genRuntimeBuffers->logits,
        genLogitsIndex, mModelConfig, mRuntime->getBufferManager(), *genRuntimeBuffers,
        genRuntimeBuffers->mMedusaBuffers);

    if (mOperatingBeamWidth > 1)
    {
        copyCacheIndirectionFromOutputsToInputs(scheduledRequests, genBufferId);
    }

    mLogitsPostProcessorIsApplied = (*mLogitsPostProcessor)(decoderInputBuffers, mReplicateLogitsPostProcessor,
        mWorldConfig, mRuntime->getStreamPtr(), mLogitsPostProcessorBatched);

    if (mGuidedDecoder)
    {
        mGuidedDecoder->execute(decoderInputBuffers, mRuntime->getBufferManager());
    }

    auto const fusedBufferId = getFusedBufferId();
    auto& fusedRuntimeBuffers = mBuffers.at(fusedBufferId);

    auto& decodingInput = mDecodingInputs.at(mMicroBatchId);
    decodingInput = (*mMakeDecodingBatchInputOutput)(mDecoderInputBuffers.at(fusedBufferId), *mDecoderState,
        mModelConfig, getMaxNumSequences(), *fusedRuntimeBuffers);

    auto decoderFinishEvent = mDecoder->forwardAsync(*mDecoderState, *decodingInput);

    auto const returnLogProbs = batchReturnLogProbs(scheduledRequests);
    auto updateDecoderBuffersEvent = (*mUpdateDecoderBuffers)(mModelConfig, mDecoderOutputBuffers.at(fusedBufferId),
        mRuntime->getBufferManager(), *mDecoderState, returnLogProbs, decoderFinishEvent);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return updateDecoderBuffersEvent;
}

void TrtGptModelInflightBatching::copyCacheIndirectionFromOutputsToInputs(
    ScheduledRequests const& scheduledRequests, SizeType32 genBufferId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(copyCacheIndirectionFromOutputsToInputs);

    auto& genRuntimeBuffers = *mBuffers.at(genBufferId);
    auto* srcOffsetsPtr = bufferCast<SizeType64>(*genRuntimeBuffers.cacheIndirDecoderIOBatchedCopySrcOffsets);
    auto* dstOffsetsPtr = bufferCast<SizeType64>(*genRuntimeBuffers.cacheIndirDecoderIOBatchedCopyDstOffsets);
    auto* copySizesPtr = bufferCast<SizeType64>(*genRuntimeBuffers.cacheIndirDecoderIOBatchedCopySizes);

    // Only `cacheIndirShape.d[2]` is used
    auto const& cacheIndirShape = mDecoderState->getCacheIndirectionOutput()->getShape();
    auto const maxBeamWidth = cacheIndirShape.d[1];
    auto const maxAttentionWindow = cacheIndirShape.d[2];
    auto const slotOffset = maxBeamWidth * maxAttentionWindow;

    SizeType32 batchIdx{0};
    SizeType64 maxCopySize{0};
    auto& manager = mRuntime->getBufferManager();
    for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            auto const reqBeamWidth = llmReq->getBeamWidthByIter();
            auto const seqSlot = llmReq->mSeqSlot.value();
            auto const copySize = reqBeamWidth * maxAttentionWindow;
            srcOffsetsPtr[batchIdx] = seqSlot * slotOffset;
            dstOffsetsPtr[batchIdx] = seqSlot * slotOffset;
            copySizesPtr[batchIdx] = copySize;
            maxCopySize = std::max(maxCopySize, copySize);
            batchIdx++;
        }
    }
    if (batchIdx != 0)
    {
        auto const srcOffsetsSlice
            = ITensor::slice(genRuntimeBuffers.cacheIndirDecoderIOBatchedCopySrcOffsets, 0, batchIdx);
        auto const srcOffsetsSliceDeviceSlice
            = ITensor::slice(genRuntimeBuffers.mCacheIndirDecoderIOBatchedCopySrcOffsetsSliceDevice, 0, batchIdx);
        manager.copy(srcOffsetsSlice->data(), *srcOffsetsSliceDeviceSlice,
            runtime::MemoryType::kGPU); // Explicitly move to device for faster access.
        auto const dstOffsetsSlice
            = ITensor::slice(genRuntimeBuffers.cacheIndirDecoderIOBatchedCopyDstOffsets, 0, batchIdx);
        auto const dstOffsetsSliceDeviceSlice
            = ITensor::slice(genRuntimeBuffers.mCacheIndirDecoderIOBatchedCopyDstOffsetsSliceDevice, 0, batchIdx);
        manager.copy(dstOffsetsSlice->data(), *dstOffsetsSliceDeviceSlice,
            runtime::MemoryType::kGPU); // Explicitly move to device for faster access.
        auto const sizesSlice = ITensor::slice(genRuntimeBuffers.cacheIndirDecoderIOBatchedCopySizes, 0, batchIdx);
        auto const copySizesDeviceSlice
            = ITensor::slice(genRuntimeBuffers.mCacheIndirDecoderIOBatchedCopyCopySizesDevice, 0, batchIdx);
        manager.copy(sizesSlice->data(), *copySizesDeviceSlice); // Explicitly move to device for faster access.
        runtime::kernels::invokeCopyBatch(*mDecoderState->getCacheIndirectionOutput(),
            *mDecoderState->getCacheIndirectionInput(), *srcOffsetsSliceDeviceSlice, *dstOffsetsSliceDeviceSlice,
            *copySizesDeviceSlice, maxCopySize, manager.getStream());
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::vector<std::unique_ptr<DecoderStepAsyncSend>> TrtGptModelInflightBatching::communicateDecoderBuffers(
    bool returnLogProbs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(communicateDecoderBuffers);

    auto& decoderOutputBuffers = mDecoderOutputBuffers.at(getFusedBufferId());

    std::vector<std::unique_ptr<DecoderStepAsyncSend>> asyncHandles;
    if (mWorldConfig.isLastPipelineParallelRank())
    {
        if (broadcastPostDecoder())
        {
            DecoderStepAsyncSend::bcast(decoderOutputBuffers, *mDecoderState, returnLogProbs, mOperatingBeamWidth,
                mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind(), *mMpiCommTensorPara, 0);
        }

        if (mWorldConfig.isPipelineParallel())
        {
            auto const peerSend = 0;
            asyncHandles.emplace_back(std::make_unique<DecoderStepAsyncSend>(decoderOutputBuffers, *mDecoderState,
                returnLogProbs, mOperatingBeamWidth, mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind(),
                *mMpiCommPipelinePara, peerSend));
        }
    }
    else
    {
        auto const peerRecv = mWorldConfig.isFirstPipelineParallelRank() ? mWorldConfig.getPipelineParallelism() - 1
                                                                         : mWorldConfig.getPipelineParallelRank() - 1;
        DecoderStepAsyncSend::recv(decoderOutputBuffers, *mDecoderState, returnLogProbs, mOperatingBeamWidth,
            mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind(), *mMpiCommPipelinePara, peerRecv);
        auto const peerSend = mWorldConfig.getPipelineParallelRank() + 1;
        if (peerSend != mWorldConfig.getPipelineParallelism() - 1)
        {
            asyncHandles.emplace_back(std::make_unique<DecoderStepAsyncSend>(decoderOutputBuffers, *mDecoderState,
                returnLogProbs, mOperatingBeamWidth, mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind(),
                *mMpiCommPipelinePara, peerSend));
        }
    }
    TLLM_CHECK_WITH_INFO(asyncHandles.size() <= 2, "Up to two decoder step async handles expected");

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return asyncHandles;
}

void TrtGptModelInflightBatching::updateRequests(ScheduledRequests const& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(updateRequests);

    auto const& decoderOutputBuffers = mDecoderOutputBuffers.at(getFusedBufferId());

    auto const hostNewOutputTokensShape = decoderOutputBuffers.newOutputTokensHost->getShape();
    auto const* const hostNewOutputTokensData
        = bufferCast<TokenIdType const>(*decoderOutputBuffers.newOutputTokensHost);
    auto const* const sequenceLengthsHostData = bufferCast<SizeType32 const>(*decoderOutputBuffers.sequenceLengthsHost);
    auto const* const decoderFinishedSumPtr = bufferCast<SizeType32 const>(*decoderOutputBuffers.finishedSumHost);
    auto const* const cumLogProbsPtr = bufferCast<float const>(*decoderOutputBuffers.cumLogProbsHost);
    auto const* const logProbsPtr = bufferCast<float const>(*decoderOutputBuffers.logProbsHost);
    auto const* const finishReasonsHostData
        = bufferCast<kernels::FinishedState>(*decoderOutputBuffers.finishReasonsHost);

    // Update only requests that ran through the decoder
    for (auto const& llmReq : scheduledRequests.generationRequests)
    {
        if (llmReq->isGenerationCompleteState())
        {
            continue;
        }
        auto const reqBeamWidth = llmReq->getBeamWidthByIter(true);
        auto const seqSlot = llmReq->mSeqSlot.value();
        auto const currentNumOfTokens = llmReq->getMaxBeamNumTokens();

        // Save the accepted token logits from target model
        if (mModelConfig.getSpeculativeDecodingMode().isDraftTokensExternal() && llmReq->getReturnGenerationLogits()
            && llmReq->hasDraftTokens())
        {
            TLLM_CHECK_WITH_INFO(reqBeamWidth == 1, "Speculative decoding only works for beam width == 1");

            SizeType32 numAcceptedTokens
                = sequenceLengthsHostData[seqSlot * mOperatingBeamWidth + 0] - llmReq->getMaxBeamNumTokens();

            auto const& generationLogitsHost = llmReq->getGenerationLogitsHost();
            auto shape = generationLogitsHost->getShape();
            shape.d[1] = numAcceptedTokens;
            generationLogitsHost->reshape(shape);
        }

        std::vector<SizeType32> numNewTokens(reqBeamWidth);
        std::vector<SizeType32> numDroppedTokens(reqBeamWidth);

        // numGeneratedTokens is the number of tokens generated by the decoder.
        // Some tokens might be dropped due to end token or rejected draft tokens.
        auto const numGeneratedTokens = llmReq->getNumDraftTokens() + 1;

        for (SizeType32 beam = 0; beam < reqBeamWidth; ++beam)
        {
            // Sequence length is only advanced for accepted tokens.
            auto const seqLen = sequenceLengthsHostData[seqSlot * mOperatingBeamWidth + beam];
            // Actual number of tokens that should be added to the request.
            auto const numNewOutputTokens = seqLen - llmReq->getNumTokens(beam);
            if (reqBeamWidth == 1)
            {
                TLLM_CHECK_WITH_INFO(numGeneratedTokens >= numNewOutputTokens,
                    "numNewOutputTokens must not be greater than numGeneratedTokens: "
                    "numGeneratedTokens %d < numNewOutputTokens %d",
                    numGeneratedTokens, numNewOutputTokens);
            }
            numNewTokens[beam] = std::min(numGeneratedTokens, numNewOutputTokens);
            numDroppedTokens[beam] = numGeneratedTokens - numNewTokens[beam];
            for (SizeType32 step = 0; step < numNewTokens[beam]; ++step)
            {
                auto const newTokenIdx = tc::flat_index(hostNewOutputTokensShape.d, step, seqSlot, beam);
                auto const newToken = hostNewOutputTokensData[newTokenIdx];
                llmReq->addNewToken(newToken, beam);
                TLLM_LOG_DEBUG("request ID %ld beam %d newToken %d", llmReq->mRequestId, beam, newToken);

                if (llmReq->returnLogProbs())
                {
                    auto const cumLogProb = cumLogProbsPtr[seqSlot * mOperatingBeamWidth + beam];
                    llmReq->setCumLogProb(cumLogProb, beam);

                    auto const beginLogProbsOffset = reqBeamWidth == 1 ? llmReq->mPromptLen : 0;
                    SizeType32 offset
                        = (seqSlot * mOperatingBeamWidth + beam) * getMaxSequenceLen() + beginLogProbsOffset;
                    auto const generatedLength = seqLen - llmReq->mPromptLen;
                    std::vector<float> logProbs(logProbsPtr + offset, logProbsPtr + offset + generatedLength);
                    llmReq->setLogProbs(logProbs, beam);
                }
            }

            auto const finishReason = finishReasonsHostData[seqSlot * mOperatingBeamWidth + beam];
            llmReq->setFinishedReason(finishReason.toFinishReason(), beam);

            TLLM_LOG_DEBUG("[RANK %d] decoderSync: request ID %lu beam %d tokens %s finished %d",
                COMM_SESSION.getRank(), llmReq->mRequestId, beam, common::vec2str(llmReq->getTokens(beam)).c_str(),
                static_cast<int>(finishReason.toFinishReason()));
        }

        // Set number of tokens predicted per runtime iteration. Will be > 1 for speculative decoding.
        llmReq->updateNumTokensPerIteration(llmReq->getMaxBeamNumTokens() - currentNumOfTokens, mModelConfig);

        // Fill new draft tokens for the next step
        if (decoderFinishedSumPtr[seqSlot] != reqBeamWidth
            && (mModelConfig.getSpeculativeDecodingMode().predictsDraftTokens()
                || mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind()))
        {
            auto const maxDraftTokensLen = mModelConfig.getMaxDecodingDraftTokens();
            auto prevDraftTokensLen = llmReq->getNumDraftTokens();

            // We overallocate KV cache for EAGLE to the maxDecodingTokens + maxPathLen in order to fit both
            // Base model verification (needs up to maxDecodingTokens) and
            // Drafter (needs up to maxPathLen of accepted tokens and maxDecodingDraftTokens for new draft tokens).
            if (mModelConfig.getSpeculativeDecodingMode().isEagle())
            {
                prevDraftTokensLen = mModelConfig.getSpeculativeDecodingModule().getMaxDecodingTokens()
                    + mModelConfig.getSpeculativeDecodingModule().getMaxPathLen() - 1;
            }

            auto nextDraftTokensLen = mModelConfig.getSpeculativeDecodingModule().getMaxDecodingDraftTokens();
            if (mModelConfig.getSpeculativeDecodingMode().variableDraftLength())
            {
                auto const* const nextDraftTokensLengthsHostData
                    = bufferCast<SizeType32 const>(*decoderOutputBuffers.nextDraftTokensLengthsHost);
                nextDraftTokensLen = nextDraftTokensLengthsHostData[seqSlot];
            }
            TLLM_CHECK(nextDraftTokensLen <= maxDraftTokensLen);

            auto const* const nextDraftTokensHostData
                = bufferCast<TokenIdType const>(*decoderOutputBuffers.nextDraftTokensHost);
            auto draftTokensShared
                = std::make_shared<std::vector<TokenIdType>>(nextDraftTokensHostData + seqSlot * maxDraftTokensLen,
                    nextDraftTokensHostData + seqSlot * maxDraftTokensLen + nextDraftTokensLen);

            llmReq->setDraftTokens(draftTokensShared);

            // For all phases except context that does not have draft tokens
            if (!llmReq->isGenerationCompleteState() && prevDraftTokensLen != 0
                && mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind())
            {
                // -1 here is for current 'main' token
                auto const acceptedTokensLen = llmReq->getMaxBeamNumTokens() - currentNumOfTokens - 1;
                auto const rewindLength = prevDraftTokensLen - acceptedTokensLen;

                TLLM_LOG_DEBUG("request ID %lu (seqSlot %d): accepted %d of %d draft tokens, rewind %d tokens",
                    llmReq->mRequestId, seqSlot, acceptedTokensLen, prevDraftTokensLen, rewindLength);
                TLLM_CHECK(0 <= acceptedTokensLen && acceptedTokensLen <= prevDraftTokensLen);

                // At this point, KV cache rows are already gathered and moved to the right location.
                // We can safely rewind (draft - accepted) tokens
                mKvCacheManager->rewindKVCache(llmReq->mRequestId, rewindLength);
            }
        }

        // Terminate if request has finished or if it is speculative decoding target model
        if (decoderFinishedSumPtr[seqSlot] == reqBeamWidth
            || (mModelConfig.getSpeculativeDecodingMode().isDraftTokensExternal() && llmReq->hasDraftTokens()))
        {
            postProcessRequest(*llmReq, numDroppedTokens);

            if (!mWorldConfig.isPipelineParallel() || !mWorldConfig.isLastPipelineParallelRank())
            {
                if (llmReq->getReturnGenerationLogits() && mSpeculativeDecodingFastLogits && mIsLeaderInOrchMode)
                {
                    mDraftRequestsWaitingToSendLogits.push_back(llmReq);
                }
                else
                {
                    terminateRequest(llmReq);
                }
                llmReq->setState(LlmRequestState::kGENERATION_COMPLETE);
            }
            else
            {
                llmReq->setState(LlmRequestState::kGENERATION_TO_COMPLETE);
            }
        }
        else
        {
            // gather tokens in the case of streaming and beam search
            if (llmReq->isStreaming() && llmReq->mSamplingConfig.beamWidth > 1)
            {
                postProcessRequest(*llmReq, numDroppedTokens);
            }
            if (llmReq->isContextInitState())
            {
                llmReq->setState(LlmRequestState::kGENERATION_IN_PROGRESS);
            }

            if (isTrtOverlap() && llmReq->willCompleteNextIteration())
            {
                // This state prohibits the request from being scheduled for another iteration. It assumes that the next
                // iteration has already been scheduled and the request can finish in the next call to updateRequests().
                llmReq->setState(LlmRequestState::kGENERATION_TO_COMPLETE);
            }
        }

        if (llmReq->getReturnPerfMetrics())
        {
            llmReq->updatePerfMetrics(mIterCounter);
        }

        llmReq->advanceDecodingIter();

        if (mWorldConfig.isPipelineParallel() && mWorldConfig.isLastPipelineParallelRank())
        {
            for (SizeType32 beam = 0; beam < reqBeamWidth; ++beam)
            {
                llmReq->setNumPreDecodedTokens(numNewTokens[beam], beam);
            }
        }
    }

    if (mModelConfig.getSpeculativeDecodingMode().needsKVCacheRewind())
    {
        SizeType32 numSequences{0};
        for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
        {
            for (auto const& llmReq : requests)
            {
                auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
                numSequences += reqBeamWidth;
            }
        }

        TLLM_CHECK_WITH_INFO(mCtxGenFusion, "Current speculative decoding mode requires context-gen fusion IFB");
        rewindKVCacheBlocks(numSequences);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::vector<std::unique_ptr<DecoderStepAsyncSend>> TrtGptModelInflightBatching::decoderSync(
    ScheduledRequests const& scheduledRequests, std::optional<runtime::CudaEvent> const& decoderFinishEvent)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(decoderSync);

    if (mWorldConfig.isLastPipelineParallelRank())
    {
        decoderFinishEvent->synchronize();
    }

    auto const returnLogProbs = batchReturnLogProbs(scheduledRequests);
    auto asyncHandles = communicateDecoderBuffers(returnLogProbs);

    updateRequests(scheduledRequests);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return asyncHandles;
}

void TrtGptModelInflightBatching::rewindKVCacheBlocks(SizeType32 numSequences)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const bufferId = getFusedBufferId();
    auto& runtimeBuffers = *mBuffers.at(bufferId);
    auto& decoderOutputBuffers = mDecoderOutputBuffers.at(bufferId);

    auto localNbLayers = mModelConfig.getNbAttentionLayers(
        mWorldConfig.getPipelineParallelism(), mWorldConfig.getPipelineParallelRank());
    if (mWorldConfig.isLastPipelineParallelRank() && mModelConfig.getSpeculativeDecodingMode().isEagle())
    {
        // Do not correct the last kv caches, which are for EagleNet drafter. Those KV caches are managed separately.
        auto eagleModulePtr
            = std::dynamic_pointer_cast<runtime::EagleModule>(mModelConfig.getSpeculativeDecodingModulePtr());
        localNbLayers -= eagleModulePtr->getNumTransformerLayers();
    }

    auto const tokensPerBlock = mModelConfig.getTokensPerBlock();
    auto const elemSize = BufferDataType(mModelConfig.getKvDataType()).getSize();
    auto const sizeInBytesPerKVHead = mModelConfig.getSizePerHead() * elemSize;

    auto const poolPointers = mKvCacheManager->getBlockPoolPointers();
    auto* const* pointerArrayPtr = bufferCast<void*>(*poolPointers);
    auto const* offsetArrayPtr
        = bufferCast<tk::KVCacheIndex>(*runtimeBuffers.transformerBuffers->kvCacheBlockOffsetsDevice);

    auto commonRewindLen = mModelConfig.getSpeculativeDecodingModule().getMaxDecodingDraftTokens();
    SizeType32 const* rewindLens = nullptr;
    if (mModelConfig.getSpeculativeDecodingMode().variableDraftLength())
    {
        commonRewindLen = 0;
        rewindLens = bufferCast<SizeType32 const>(*decoderOutputBuffers.prevDraftTokensLengthsHost);
    }

    tensorrt_llm::runtime::kernels::invokeUpdateKVBlockArrayDraftTokenLocation(
        *mDecoderState->getAcceptedLengthsCumSum(), *mDecoderState->getAcceptedPackedPaths(),
        *runtimeBuffers.sequenceLengthsDevice, pointerArrayPtr, offsetArrayPtr, localNbLayers, numSequences,
        mRewindInputs.numKvHeads, sizeInBytesPerKVHead, commonRewindLen, rewindLens, *runtimeBuffers.seqSlots,
        getMaxAttentionWindow(), mRewindInputs.maxBlocksPerSeq, tokensPerBlock, mRewindInputs.isUseOneMoreBlock,
        mRuntime->getStreamPtr()->get());

    sync_check_cuda_error(mRuntime->getStream().get());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

nvinfer1::DataType TrtGptModelInflightBatching::getLogitDataType() const
{
    return mModelConfig.getLogitsDtype();
}

void TrtGptModelInflightBatching::changeBeamWidth(SizeType32 beamWidth)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(mInflightReqIds.empty());

    TLLM_CHECK_WITH_INFO(beamWidth <= getMaxBeamWidth(),
        "Requested beam width %d is larger than configured max beam width %d", beamWidth, getMaxBeamWidth());
    TLLM_LOG_DEBUG("Changing operating beam width from %d to %d", mOperatingBeamWidth, beamWidth);
    mOperatingBeamWidth = beamWidth;

    createBuffers(mDecodingConfig, mAdditionalModelOutputs);
    createDecoder(mDecodingConfig.getDecodingMode());

    if (static_cast<bool>(mKvCacheManager))
    {
        auto const dims = mKvCacheManager->getOffsetTableDimensions();
        reshapeKvTensors(dims);
    }
    if (static_cast<bool>(mCrossKvCacheManager))
    {
        auto const dims = mCrossKvCacheManager->getOffsetTableDimensions();
        reshapeKvTensors(dims);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::changeSpecDecMode(ScheduledRequests const& scheduledRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if ((!mModelConfig.getSpeculativeDecodingMode().isLookaheadDecoding()
            && !mModelConfig.getSpeculativeDecodingMode().isNone())
        || scheduledRequests.empty() || mSeamlessLADMaxDraftLen == 0 || getGatherGenerationLogits()
        || mModelConfig.isRnnBased())
    {
        return;
    }

    bool canUseLookahead = false;
    auto maxNumRequestForLad = mDecodingConfig.getLookaheadDecodingMaxNumRequest();
    SizeType32 numRequests = scheduledRequests.contextRequests.size() + scheduledRequests.generationRequests.size();
    if (numRequests > maxNumRequestForLad)
    {
        if (mModelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
        {
            canUseLookahead = false;
        }
        else
        {
            return;
        }
    }
    {
        bool useTopKTopP = false;
        bool useBanWords = false;
        bool useTempAccVocabPenalties = false; // use temperature and penalties that need to accumulate #vocab.
        SizeType32 beamWidth = 1;
        for (auto const& requests : {scheduledRequests.contextRequests, scheduledRequests.generationRequests})
        {
            for (auto const& llmReq : requests)
            {
                useTopKTopP |= !(llmReq->mSamplingConfig.useDefaultValues(
                                     llmReq->mSamplingConfig.topK, layers::DefaultDecodingParams::getTopK())
                    || llmReq->mSamplingConfig.useDefaultValues(llmReq->mSamplingConfig.topK, 1));
                useTopKTopP |= !llmReq->mSamplingConfig.useDefaultValues(
                    llmReq->mSamplingConfig.topP, layers::DefaultDecodingParams::getTopP());
                useBanWords |= llmReq->getBadWordsList().has_value();
                useBanWords |= !llmReq->mSamplingConfig.useDefaultValues(
                    llmReq->mSamplingConfig.noRepeatNgramSize, layers::DefaultDecodingParams::getNoRepeatNgramSize());
                useTempAccVocabPenalties |= !llmReq->mSamplingConfig.useDefaultValues(
                    llmReq->mSamplingConfig.temperature, layers::DefaultDecodingParams::getTemperature());
                useTempAccVocabPenalties |= !llmReq->mSamplingConfig.useDefaultValues(
                    llmReq->mSamplingConfig.repetitionPenalty, layers::DefaultDecodingParams::getRepetitionPenalty());
                useTempAccVocabPenalties |= !llmReq->mSamplingConfig.useDefaultValues(
                    llmReq->mSamplingConfig.presencePenalty, layers::DefaultDecodingParams::getPresencePenalty());
                useTempAccVocabPenalties |= !llmReq->mSamplingConfig.useDefaultValues(
                    llmReq->mSamplingConfig.frequencyPenalty, layers::DefaultDecodingParams::getFrequencyPenalty());
                beamWidth = llmReq->mSamplingConfig.beamWidth;
                if (useTopKTopP || useBanWords || useTempAccVocabPenalties || beamWidth > 1)
                {
                    break;
                }
            }
            canUseLookahead = !(useTopKTopP || useBanWords || useTempAccVocabPenalties || beamWidth > 1);
        }
    }

    // Change speculative decoding mode
    auto const bufferId = mCtxGenFusion
        ? getFusedBufferId()
        : (!scheduledRequests.contextRequests.empty() ? getContextBufferId() : getGenerationBufferId());
    // TODO: enable lookahead for generation requests.
    bool canChangeToLookahead = scheduledRequests.generationRequests.empty();
    if (mModelConfig.getSpeculativeDecodingMode().isNone() && canUseLookahead && canChangeToLookahead)
    {
        // None -> Lookahead
        mModelConfig.enableSeamlessLookaheadDecoding(mSeamlessLADMaxDraftLen);
        mDecodingConfig.enableSeamlessLookaheadDecoding();
        setupSpeculativeDecodingModule(mDecodingConfig);
        mBuffers.at(bufferId)->mLookaheadBuffers->enableLookaheadDecoding(
            getMaxBatchSize(), mModelConfig.getMaxDecodingTokens());
        mDecoderOutputBuffers.at(getFusedBufferId())
            .enableLookaheadDecoding(getMaxNumSequences(), mModelConfig.getMaxDecodingTokens());
        createDecoder(mDecodingConfig.getDecodingMode());
    }
    else if (mModelConfig.getSpeculativeDecodingMode().isLookaheadDecoding()
        && (!canUseLookahead || numRequests > maxNumRequestForLad))
    {
        // Lookahead -> None
        mModelConfig.disableSeamlessLookaheadDecoding();
        mDecodingConfig.setDecodingMode(executor::DecodingMode::Auto());
        mBuffers.at(bufferId)->mLookaheadBuffers->disableLookaheadDecoding();
        mDecoderOutputBuffers.at(getFusedBufferId()).disableLookaheadDecoding(getMaxNumSequences());
        mDecoder->disableLookahead(
            scheduledRequests.generationRequests, mDecoderInputBuffers.at(getFusedBufferId()).setupBatchSlots);
        mDecoderState->disableLookahead(scheduledRequests.generationRequests);
        for (auto const& llmReq : scheduledRequests.generationRequests)
        {
            if (llmReq->getNumDraftTokens() > 0)
            {
                llmReq->discardDraftTokens(llmReq->getNumDraftTokens());
            }
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TrtGptModelInflightBatching::getCurrentIterationStats(executor::IterationStats& stats) const
{
    stats.iter = mIterCounter;

    // Max batch size and max num tokens can be tuned at runtime
    stats.maxBatchSizeStatic = getMaxBatchSize();
    stats.maxBatchSizeTunerRecommended = mMaxBatchSizeTunerRecommended;
    stats.maxBatchSizeRuntime = mMaxBatchSizeRuntime;
    stats.maxNumTokensStatic = mMaxNumTokensStatic.value_or(0);
    stats.maxNumTokensTunerRecommended = mMaxNumTokensTunerRecommended;
    stats.maxNumTokensRuntime = mMaxNumTokensRuntime.value_or(0);

    // KVCacheManager statistics
    auto const& kvCacheManager = getKVCacheManager();
    if (kvCacheManager)
    {
        executor::KvCacheStats kvStats{};
        auto kvCacheStats = kvCacheManager->getKvCacheStats();
        kvStats.maxNumBlocks = kvCacheStats.maxNumBlocks;
        kvStats.freeNumBlocks = kvCacheStats.freeNumBlocks;
        kvStats.usedNumBlocks = kvCacheStats.usedNumBlocks;
        kvStats.tokensPerBlock = kvCacheStats.toksPerBlock;
        kvStats.allocTotalBlocks = kvCacheStats.allocTotalBlocks;
        kvStats.allocNewBlocks = kvCacheStats.allocNewBlocks;
        kvStats.reusedBlocks = kvCacheStats.reusedBlocks;
        kvStats.missedBlocks = kvCacheStats.missedBlocks;
        kvStats.cacheHitRate = kvCacheStats.cacheHitRate;
        stats.kvCacheStats = kvStats;
    }
    auto const& crossKvCacheManager = getCrossKVCacheManager();
    if (crossKvCacheManager)
    {
        executor::KvCacheStats kvStats{};
        auto kvCacheStats = crossKvCacheManager->getKvCacheStats();
        kvStats.maxNumBlocks = kvCacheStats.maxNumBlocks;
        kvStats.freeNumBlocks = kvCacheStats.freeNumBlocks;
        kvStats.usedNumBlocks = kvCacheStats.usedNumBlocks;
        kvStats.tokensPerBlock = kvCacheStats.toksPerBlock;
        kvStats.allocTotalBlocks = kvCacheStats.allocTotalBlocks;
        kvStats.allocNewBlocks = kvCacheStats.allocNewBlocks;
        kvStats.reusedBlocks = kvCacheStats.reusedBlocks;
        kvStats.missedBlocks = kvCacheStats.missedBlocks;
        kvStats.cacheHitRate = kvCacheStats.cacheHitRate;
        stats.crossKvCacheStats = kvStats;
    }
    executor::InflightBatchingStats modelStats{};
    modelStats.numScheduledRequests = mLastIterationStatsIFB.scheduledRequests.size();
    modelStats.numContextRequests = mLastIterationStatsIFB.numCtxRequests;
    modelStats.numGenRequests = mLastIterationStatsIFB.numGenRequests;
    modelStats.numPausedRequests = mLastIterationStatsIFB.pausedRequests.size();
    modelStats.avgNumDecodedTokensPerIter = mLastIterationStatsIFB.avgNumDecodedTokensPerIter;
    modelStats.numCtxTokens = mLastIterationStatsIFB.numCtxTokens;
    modelStats.microBatchId = mLastIterationStatsIFB.microBatchId;
    stats.inflightBatchingStats = modelStats;
}

void TrtGptModelInflightBatching::getCurrentRequestStats(executor::RequestStatsPerIteration& stats) const
{
    stats.iter = mIterCounter;
    for (auto& requestStat : stats.requestStats)
    {
        requestStat.scheduled
            = mLastIterationStatsIFB.scheduledRequests.count(static_cast<RequestIdType>(requestStat.id));
        requestStat.paused = mLastIterationStatsIFB.pausedRequests.count(static_cast<RequestIdType>(requestStat.id));
    }
}

executor::DebugTensorsPerIteration TrtGptModelInflightBatching::getCurrentDebugTensors() const
{
    executor::DebugTensorsPerIteration debugTensors;
    debugTensors.iter = mIterCounter;

    for (auto const& [name, tensor] : mLastIterationDebugTensors)
    {
        debugTensors.debugTensors.emplace(name, executor::detail::ofITensor(tensor));
    }

    return debugTensors;
}

nvinfer1::DataType TrtGptModelInflightBatching::getTensorDataType(std::string const& name) const
{
    auto const& engine = mRuntime->getEngine();
    return engine.getTensorDataType(name.c_str());
}

nvinfer1::Dims TrtGptModelInflightBatching::getTensorShape(std::string const& name) const
{
    auto const& engine = mRuntime->getEngine();
    return engine.getTensorShape(name.c_str());
}

SizeType32 TrtGptModelInflightBatching::getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const
{
    return mKvCacheManager->getMaxCapacityBatchSize(inputLength, outputLength);
}

/*
 * Manages prefetching of prompt table chunks using a double-buffer strategy
 *
 * Function Flow:
 * 1. First Chunk Processing (isFirstChunk == true):
 *    - Uses blocking prefetch on main runtime stream
 *    - Ensures initial data is ready before computation starts
 *
 * 2. Subsequent Chunks (isFirstChunk == false):
 *    - Uses non-blocking prefetch on separate copy stream
 *    - Overlaps data transfer with computation
 *
 * Synchronization:
 * - First prefetch: No wait needed (fresh start)
 * - Later prefetches: Wait for previous copy to complete
 * - Uses mPtableCopyDoneEvent to track completion
 *
 * Key Functions:
 * 1. prefetchNextPromptTableChunk:
 *    - Calls the correct function based on position in code (before or after prepareBuffers())
 *    - Waits for previous copy to complete if not the first chunk
 *
 * 2. remapInputTokensForPromptTable:
 *    - Identifies tokens that need prompt table embeddings (tokens that are greater than vocabSize)
 *    - Remaps IDs to match chunked prompt table layout
 *
 * 3. copyPromptTableToGpuInChunk:
 *    - Handles actual transfer from CPU pinned memory to GPU
 *    - Uses appropriate buffer manager based on isFirstChunk
 */
void TrtGptModelInflightBatching::prefetchNextPromptTableChunk(
    RequestVector const& contextRequests, bool isFirstChunk, SizeType32 bufferId)
{
    auto& promptTuningBuffers = mBuffers[bufferId]->promptTuningBuffers;

    if (!isFirstChunk)
    {
        // Only switch buffer after prepareBuffer()
        promptTuningBuffers->switchChunkPtableBuffer();
    }

    SizeType32 contextId = 0;
    for (auto const& llmReq : contextRequests)
    {
        if (llmReq->isFirstContextChunk() && isFirstChunk)
        {
            // For first chunk: Blocking prefetch on runtime stream to ensure data is ready
            remapInputTokensForPromptTable(llmReq, true, bufferId, contextId);
        }
        else if (!isFirstChunk) // prefetching for subsequent chunks
        {
            // For the first prefetch chunk, don't need to wait for previous prefetch to complete
            // For subsequent chunks: Need to wait for previous prefetch to complete
            if (!llmReq->isFirstContextChunk())
            {
                mRuntime->getBufferManager().getStream().wait(mPtableCopyDoneEvent);
            }

            // Non-blocking prefetch on copy stream to prepare next chunk in pong buffer
            if (llmReq->getContextRemainingLength() > 0)
            {
                remapInputTokensForPromptTable(llmReq, false, bufferId, contextId);
            }
        }

        ++contextId;
    }
}

void TrtGptModelInflightBatching::remapInputTokensForPromptTable(
    std::shared_ptr<LlmRequest> const& llmReq, bool isFirstChunk, SizeType32 bufferId, SizeType32 contextId)
{
    NVTX3_SCOPED_RANGE_WITH_NAME(range, "remapInputTokensForPromptTable");
    auto& promptTuningBuffers = mBuffers[bufferId]->promptTuningBuffers;
    auto const chunkSize = llmReq->getContextChunkSize();
    auto& inputTokensMutable = llmReq->getTokensMutable(0);
    auto vocabSize = mModelConfig.getVocabSize();

    if (isFirstChunk)
    {
        promptTuningBuffers->initializeChunkPtableBuffers(
            mRuntime->getBufferManager(), mModelConfig, chunkSize, llmReq);
    }

    size_t processChunkSize;
    size_t beginPos;

    if (!isFirstChunk)
    {
        processChunkSize = std::min(chunkSize, llmReq->getContextRemainingLength() - chunkSize);
    }
    else
    {
        processChunkSize = std::min(chunkSize, llmReq->getContextRemainingLength());
    }

    if (!isFirstChunk)
    {
        // For prefetching next chunk
        if (llmReq->getContextRemainingLength() - chunkSize <= 0)
        {
            promptTuningBuffers->updateBufferStartPosition(promptTuningBuffers->getChunkPtableCurrentIndex(), 0);
            return; // No more chunks to prefetch
        }
        beginPos = llmReq->getContextCurrentPosition() + chunkSize;
    }
    else
    {
        // For current chunk
        beginPos = llmReq->getContextCurrentPosition();
    }

    TLLM_CHECK_WITH_INFO(beginPos + processChunkSize <= inputTokensMutable.size(),
        "Invalid chunk access: beginPos(%zu) + processChunkSize(%zu) > totalSize(%zu)", beginPos, processChunkSize,
        inputTokensMutable.size());

    auto inputTokensChunk = inputTokensMutable.begin() + beginPos;
    std::vector<SizeType32> outOfVocabTokens;
    SizeType32 ptableTokenId = vocabSize;
    for (size_t i = 0; i < processChunkSize; i++)
    {
        if (inputTokensChunk[i] >= vocabSize)
        {
            outOfVocabTokens.push_back(inputTokensChunk[i]);
            inputTokensChunk[i] = ptableTokenId++;
        }
    }

    copyPromptTableToGpuInChunk(llmReq, outOfVocabTokens, isFirstChunk, bufferId, contextId);
}

void TrtGptModelInflightBatching::copyPromptTableToGpuInChunk(std::shared_ptr<LlmRequest> const& llmReq,
    std::vector<int32_t> const& outOfVocabTokens, bool isFirstChunk, SizeType32 bufferId, SizeType32 contextId)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE_WITH_NAME(range, "copyPromptTableToGpuInChunk");
    auto& promptTuningBuffers = mBuffers[bufferId]->promptTuningBuffers;

    if (outOfVocabTokens.empty())
    {
        return;
    }

    auto const& promptTable = llmReq->getPromptEmbeddingTable();
    TLLM_CHECK_WITH_INFO(promptTable.has_value(), "promptTable is empty but there's fake_prompt");
    TLLM_CHECK_WITH_INFO(promptTable.value() != nullptr, "promptTable value is null but there's fake_prompt");

    auto currentBufferManager = isFirstChunk ? mRuntime->getBufferManager() : mCopyBufferManager;
    auto const hiddenSize = mModelConfig.getHiddenSize();
    auto numRows = outOfVocabTokens.size();
    std::size_t sliceSize = static_cast<size_t>(numRows * hiddenSize);
    auto currentIndex = promptTuningBuffers->getChunkPtableCurrentIndex();

    // Calculate the offset based on current position
    size_t srcOffset = llmReq->mPtableCurrentPosition * hiddenSize;
    size_t dstOffset = promptTuningBuffers->getChunkPtableBufferStartPosition(currentIndex, contextId);

    auto gpuBuffer = promptTuningBuffers->getChunkPtableBuffer(currentIndex);

    // First view as 1D tensor of elements
    auto totalElements = promptTable.value()->getSize();
    auto table1D = runtime::ITensor::view(
        promptTable.value(), runtime::ITensor::makeShape({static_cast<int64_t>(totalElements)}));

    TLLM_CHECK_WITH_INFO(srcOffset + sliceSize <= totalElements,
        "Buffer bounds violation: Trying to access up to %zu elements but buffer only has %zu elements (offset: %zu, "
        "slice size: %zu)",
        srcOffset + sliceSize, totalElements, srcOffset, sliceSize);

    auto table1DShared = runtime::ITensor::SharedPtr(table1D.release());
    auto pTableView = runtime::ITensor::slice(table1DShared, srcOffset, sliceSize);

    auto gpuBufferSlice = runtime::ITensor::slice(gpuBuffer, dstOffset, numRows);

    currentBufferManager.copy(*pTableView, *gpuBufferSlice);

    promptTuningBuffers->updateBufferStartPosition(currentIndex, outOfVocabTokens.size());

    llmReq->mPtableCurrentPosition += outOfVocabTokens.size();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::batch_manager
