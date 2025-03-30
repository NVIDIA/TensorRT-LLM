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

#pragma once

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/sequenceSlotManager.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/gptDecoderBatched.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include "trtGptModel.h"

#include <NvInferRuntime.h>

namespace tensorrt_llm::runtime
{
class TllmRuntime;
class GptDecoderBatched;
class AllReduceBuffers;
class NcclCommunicator;
class SpeculativeDecodingMode;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::mpi
{
class MpiWaitThread;
} // namespace tensorrt_llm::mpi

namespace tensorrt_llm::batch_manager
{

namespace kv_cache_manager
{
class KVCacheManager;
} // namespace kv_cache_manager

namespace rnn_state_manager
{
class RnnStateManager;
} // namespace rnn_state_manager
class SequenceSlotManager;
class DecoderStepAsyncSend;
class DecoderSlotAsyncSend;
class DecoderInputBuffers;
class DecoderBuffers;
class SlotDecoderBuffers;
class LlmRequest;
class RuntimeBuffers;
class BasePeftCacheManager;
class GuidedDecoder;
class BaseCacheTransceiver;

// Algorithms
class CapacityScheduler;
class MicroBatchScheduler;
class PauseRequests;
class AssignReqSeqSlots;
class AllocateKvCache;
class HandleContextLogits;
class HandleGenerationLogits;
class GenerateRequestOptions;
class LogitsPostProcessor;
class MakeDecodingBatchInputOutput;
class CreateNewDecoderRequests;

namespace utils
{
class CudaGraphExecutorCache;
} // namespace utils

class TrtGptModelInflightBatching : public TrtGptModel
{
    using BaseKVCacheManager = kv_cache_manager::BaseKVCacheManager;
    using KVCacheManager = kv_cache_manager::KVCacheManager;
    using KvCacheType = kv_cache_manager::CacheType;
    using KvCacheConfig = kv_cache_manager::KvCacheConfig;
    using RnnStateManager = rnn_state_manager::RnnStateManager;
    using LlmRequestPtr = std::shared_ptr<batch_manager::LlmRequest>;

public:
    class IterationStatsIFB
    {
    public:
        explicit IterationStatsIFB(SizeType32 microBatchId)
            : microBatchId{microBatchId}
        {
        }

        SizeType32 microBatchId;
        SizeType32 numCtxRequests{};
        SizeType32 numGenRequests{};
        SizeType32 numCtxTokens{};
        float avgNumDecodedTokensPerIter{};
        ReqIdsSet scheduledRequests;
        ReqIdsSet pausedRequests;
    };

    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TokenIdType = tensorrt_llm::runtime::TokenIdType;
    using BufferManager = tensorrt_llm::runtime::BufferManager;
    using PeftTable = PeftCacheManager::PeftTable;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;
    using TensorPtr = runtime::ITensor::SharedPtr;

    TrtGptModelInflightBatching(std::shared_ptr<nvinfer1::ILogger> logger, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, runtime::RawEngine const& rawEngine, bool ctxGenFusion,
        TrtGptModelOptionalParams const& optionalParams = TrtGptModelOptionalParams());

    ~TrtGptModelInflightBatching() override;

    void terminateRequest(LlmRequestPtr const& llmRequest, bool pause = false) override;

    /// @brief Function that waits for the decoding of requests in flight.
    ///        When the requests have finished or using speculative decoding, the state of requests
    ///        will become LlmRequestState::kGENERATION_COMPLETE. Else, it will be set to
    ///        LlmRequestState::kGENERATION_IN_PROGRESS.
    void forwardSync() override;

    /// @brief Function that tries to advance the active requests.
    ///        Depending on resources available, it's possible that not all requests will get advanced.
    ///        Requests that may be in state LlmRequestState::kCONTEXT_INIT become
    ///        LlmRequestState::kGENERATION_IN_PROGRESS or LlmRequestState::kGENERATION_TO_COMPLETE.
    /// @param activeRequests The list of request to try to advance.
    void forwardAsync(RequestList const& activeRequests) override;

    /// @brief Override the runtime batch size for the model
    void setRuntimeBatchSize(SizeType32 runtimeMaxBatchSize) override;

    /// @brief Get the runtime batch size for the model
    [[nodiscard]] SizeType32 getRuntimeBatchSize() const override;

    /// @brief Override the runtime max num tokens for the model
    void setRuntimeMaxNumTokens(SizeType32 runtimeMaxNumTokens) override;

    void updatePeftCache(std::shared_ptr<LlmRequest> const& llmRequest) override;

    [[nodiscard]] IterationStatsIFB getLastIterationStats() const
    {
        return mLastIterationStatsIFB;
    }

    [[nodiscard]] TrtGptModelType getModelType() const override
    {
        return mCtxGenFusion ? TrtGptModelType::InflightFusedBatching : TrtGptModelType::InflightBatching;
    };

    [[nodiscard]] runtime::BufferManager const& getBufferManager() const override;
    [[nodiscard]] runtime::BufferManager::CudaStreamPtr getRuntimeStreamPtr() const override;

    void getCurrentIterationStats(executor::IterationStats& stats) const override;
    void getCurrentRequestStats(executor::RequestStatsPerIteration& stats) const override;
    [[nodiscard]] executor::DebugTensorsPerIteration getCurrentDebugTensors() const override;

    [[nodiscard]] executor::IterationType getIterCounter() const noexcept override
    {
        return mIterCounter;
    }

    [[nodiscard]] static bool optionalParamsAreValid(
        runtime::ModelConfig const& modelConfig, TrtGptModelOptionalParams const& optionalParams);
    [[nodiscard]] static TrtGptModelOptionalParams fixOptionalParams(
        runtime::ModelConfig const& modelConfig, TrtGptModelOptionalParams const& optionalParams);

    void prepareDisaggGenInitRequests(RequestList const& activeRequests, RequestVector& newGenReques);
    void checkDisaggGenTransferStatus(RequestList const& activeRequests);
    void prepareDistGenBufferAndDecoder(RequestVector const& generationRequests);

    void resetIterationStats() override;

    runtime::SpeculativeDecodingMode getSpeculativeDecodingMode() const noexcept
    {
        return mModelConfig.getSpeculativeDecodingMode();
    }

private:
    [[nodiscard]] SizeType32 getContextBufferId() const
    {
        return mMicroBatchId;
    }

    [[nodiscard]] SizeType32 getGenerationBufferId() const
    {
        return mNumMicroBatches + mMicroBatchId;
    }

    [[nodiscard]] SizeType32 getFusedBufferId() const
    {
        return mMicroBatchId;
    }

    [[nodiscard]] SizeType32 getNextMicroBatchId(SizeType32 bufferId) const
    {
        return (bufferId + 1) % mNumMicroBatches;
    }

    [[nodiscard]] SizeType32 getPrevMicroBatchId(SizeType32 bufferId) const
    {
        return (bufferId + mNumMicroBatches - 1) % mNumMicroBatches;
    }

    //! @brief Store full kv cache blocks contributed by req.
    //! These blocks become reusable from next step.
    void storeContextBlocks(std::shared_ptr<LlmRequest> const& req);

    //! @brief Set LayerProfiler to collect performance per layer.
    void setLayerProfiler() override;

    //! @brief Print profile information per layer.
    std::string getLayerProfileInfo() const override;

    std::tuple<SizeType32, TensorMap const&, TensorMap&> prepareBuffers(
        RequestVector const& contextRequests, RequestVector const& generationRequests, SizeType32 bufferId);

    //! @brief Capture graph of current batch state during engine execution.
    //! This is based on the assumptions that
    //! a) We can hide CPU graph capture behind the GPU engine execution.
    //! b) Batch size in the next iterations won't change and we can reuse the graph multiple times.
    void prepareGraph(SizeType32 bufferId, SizeType32 optProfileId);

    void executeContext(SizeType32 runtimeContextId, SizeType32 bufferId);
    void executeBatch(ScheduledRequests const& scheduledRequests);
    void executeStep(
        RequestVector const& contextRequests, RequestVector const& generationRequests, SizeType32 bufferId);

    void debugIOTensors(RequestVector const& contextRequests, RequestVector const& generationRequests,
        TensorMap const& inputMap, TensorMap const& outputMap);

    void createRuntimeContexts();
    void createDecoder(std::optional<executor::DecodingMode> const& decodingModeOpt);
    void createBuffers(executor::DecodingConfig const& decodingConfig,
        std::optional<std::vector<std::string>> const& additionalOutputNames);
    std::shared_ptr<KVCacheManager> createKvCacheManager(KvCacheConfig const& kvCacheConfig,
        SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool, KvCacheType kvCacheType = KvCacheType::kSELF);
    void createRnnStateManager();
    void createCustomAllReduceWorkspace();
    void createRuntimePerfKnobsTensor(executor::ExtendedRuntimePerfKnobConfig const& extendedRuntimePerfKnobConfig);

    /// @brief Verify draft token length and beam width of all active requests.
    ///        May change operating beam width if all requests agree on same beam width.
    void verifyRequests(RequestList const& activeRequests);

    /// @brief Change the operating beam width.
    ///        Only possible if no requests are currently in-flight.
    /// @param beamWidth New operating beam width. Must be smaller than initial maxBeamWidth.
    void changeBeamWidth(SizeType32 beamWidth);

    SizeType32 getOperatingBeamWidth() const override
    {
        return mOperatingBeamWidth;
    }

    /// @details Should be called after setting up the current batch in executeBatch to get the correct number of
    /// context tokens.
    IterationStatsIFB fillIterationStats(
        ScheduledRequests const& scheduledRequests, RequestVector const& requestsToPause);

    /// @brief Function that sets up the TensorRT execution context that is going to be used for execution. If multiple
    /// TensorRT optimization profiles are built in the engine, it selects the corresponding context that is going to be
    /// used, and prepares the input and output tensors so that both buffers and the context is ready for the execution.
    /// @return The TensorRT execution context index that has been setup.
    void setupContext(
        RequestVector const& contextRequests, RequestVector const& generationRequests, SizeType32 bufferId);

    void setupDecoderStep(
        RequestVector const& contextRequests, RuntimeBuffers const& buffers, DecoderInputBuffers const& inputBuffers);
    runtime::CudaEvent decoderStepAsync(ScheduledRequests const& scheduledRequests);
    std::vector<std::unique_ptr<DecoderStepAsyncSend>> decoderSync(
        ScheduledRequests const& scheduledRequests, runtime::CudaEvent decoderFinishEvent);

    runtime::CudaEvent updateDecoderBuffers(bool returnLogProbs, runtime::CudaEvent decoderFinishEvent);
    std::vector<std::unique_ptr<DecoderStepAsyncSend>> communicateDecoderBuffers(bool returnLogProbs);
    void updateRequests(ScheduledRequests const& scheduledRequests);

    /// @brief It gathers the logits if they need to be returned, calls getDecoderSlotHostOutputs,
    /// and overwrites the llmRequest tokens buffer.
    /// Called either on request finishing, or at every step when doing beam search and streaming.
    void postProcessRequest(LlmRequest& llmReq, std::vector<SizeType32> const& numDroppedTokens);
    /// @brief Calls gatherTree (via finalize) and transmits the received data across ranks if PP>1
    void getDecoderSlotHostOutputs(
        SizeType32 seqSlot, bool returnLogProbs, runtime::SamplingConfig const& samplingConfig, bool streaming);
    void rewindKVCacheBlocks(SizeType32 numSequences);
    void setupSpeculativeDecodingModule(executor::DecodingConfig const& decodingConfig);

    /// @brief Copies the content of the cache indirection outputs to the cache indirection inputs.
    /// @param[in] scheduledRequests The requests to copy the cache indirections for.
    /// @param[in] genBufferId The id of the generation buffers for those requests.
    void copyCacheIndirectionFromOutputsToInputs(ScheduledRequests const& scheduledRequests, SizeType32 genBufferId);

    [[nodiscard]] bool getGatherGenerationLogits() const override
    {
        return getModelConfig().computeGenerationLogits() || mGatherGenerationLogits;
    }

    [[nodiscard]] runtime::ModelConfig const& getModelConfig() const override
    {
        return mModelConfig;
    }

    [[nodiscard]] runtime::WorldConfig const& getWorldConfig() const override
    {
        return mWorldConfig;
    }

    [[nodiscard]] SizeType32 getNumMicroBatches() const override
    {
        return mNumMicroBatches;
    }

    [[nodiscard]] nvinfer1::DataType getLogitDataType() const override;

    [[nodiscard]] nvinfer1::DataType getTensorDataType(std::string const& name) const override;

    [[nodiscard]] nvinfer1::Dims getTensorShape(std::string const& name) const override;

    void reshapeKvTensors(SizeType32 maxBlocksPerSeq, kv_cache_manager::CacheType kvCacheType, SizeType32 numPools);

    [[nodiscard]] bool hasSpeculativeDecodingFastLogits() const noexcept override
    {
        return mSpeculativeDecodingFastLogits;
    }

    [[nodiscard]] bool hasGuidedDecoder() const noexcept override
    {
        return static_cast<bool>(mGuidedDecoder);
    }

    /// @brief Based on the KV-cache manager's capacity and configuration, we adjust the maximum supported attention
    /// window.
    ///
    /// @param numPrimaryBlocks The number of blocks in the kv-cache's primary pool.
    /// @param numTokensPerBlock The number of tokens per kv-cache block.
    void adjustMaxAttentionWindow(SizeType32 numPrimaryBlocks, SizeType32 numTokensPerBlock);

    /// @brief Change the speculative decoding mode.
    void changeSpecDecMode(ScheduledRequests const& scheduledRequests);

protected:
    std::shared_ptr<BaseKVCacheManager> getKVCacheManager() override
    {
        return mKvCacheManager;
    }

    [[nodiscard]] std::shared_ptr<BaseKVCacheManager const> getKVCacheManager() const override
    {
        return mKvCacheManager;
    }

    std::shared_ptr<BaseKVCacheManager> getCrossKVCacheManager()
    {
        return mCrossKvCacheManager;
    }

    [[nodiscard]] std::shared_ptr<BaseKVCacheManager const> getCrossKVCacheManager() const
    {
        return mCrossKvCacheManager;
    }

    [[nodiscard]] std::shared_ptr<BasePeftCacheManager> getPeftCacheManager() override
    {
        return mPeftCacheManager;
    }

    [[nodiscard]] std::shared_ptr<BasePeftCacheManager const> getPeftCacheManager() const override
    {
        return mPeftCacheManager;
    }

    void setLogitsPostProcessorBatched(std::optional<LogitsPostProcessorBatched> logitsPostProcessorBatched) override
    {
        mLogitsPostProcessorBatched = logitsPostProcessorBatched;
    }

    void setReplicateLogitsPostProcessor(bool replicateLogitsPostProcessor) override
    {
        mReplicateLogitsPostProcessor = replicateLogitsPostProcessor;
    }

    [[nodiscard]] bool getReplicateLogitsPostProcessor() const override
    {
        return mReplicateLogitsPostProcessor;
    }

    SizeType32 getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const override;

private:
    /******************** Configs ********************/
    // Parameters of the model (TRT engine)
    runtime::ModelConfig mModelConfig;
    // Parameters of the execution environment
    runtime::WorldConfig mWorldConfig;
    // Device ID of this instance
    int mDevice{-1};
    // Config for (speculative) decoding
    executor::DecodingConfig mDecodingConfig;
    // Performance knobs for the engine.
    executor::ExtendedRuntimePerfKnobConfig mExtendedRuntimePerfKnobConfig;
    TensorPtr mExtendedRuntimePerfKnobsHost;
    // Config for debugging output
    std::optional<executor::DebugConfig> mDebugConfig;
    // List of additional outputs for each request
    std::optional<std::vector<std::string>> mAdditionalOutputNames;

    /******************** Components ********************/
    std::shared_ptr<nvinfer1::ILogger> mLogger;
    // Runner for the TRT engine. The engine produces logits.
    std::shared_ptr<runtime::TllmRuntime> mRuntime;
    // Decoder that generates new tokens from the logits.
    std::shared_ptr<runtime::GptDecoderBatched> mDecoder;
    // Synchronization handles for decoder
    std::vector<std::optional<runtime::CudaEvent>> mDecoderFinishedEvents;

    // Manager that maps requests to slots
    std::shared_ptr<SequenceSlotManager> mSeqSlotManager;
    // KV cache manager for attention layers (optional)
    std::shared_ptr<BaseKVCacheManager> mKvCacheManager;
    // KV cache manager for cross attention in enc-dec models (optional)
    std::shared_ptr<BaseKVCacheManager> mCrossKvCacheManager = nullptr;
    // RNN state manager for recurrent layers (optional)
    std::shared_ptr<RnnStateManager> mRnnStateManager;
    // PEFT cache manager for LoRA tasks (optional)
    std::shared_ptr<BasePeftCacheManager> mPeftCacheManager;
    // BufferManager using a separate stream for async copy operations.
    runtime::BufferManager mCopyBufferManager;

    /******************** Logits Post-Processor ********************/
    std::optional<LogitsPostProcessorBatched> mLogitsPostProcessorBatched;
    bool mReplicateLogitsPostProcessor{true};
    // Set if any request invoked a logits processor in current step
    bool mLogitsPostProcessorIsApplied{false};

    constexpr bool broadcastPostDecoder()
    {
        return mWorldConfig.isTensorParallel() && !mReplicateLogitsPostProcessor && mLogitsPostProcessorIsApplied;
    }

    std::unique_ptr<tensorrt_llm::batch_manager::GuidedDecoder> mGuidedDecoder;

    /******************** Pipeline parallelism ********************/
    std::shared_ptr<tensorrt_llm::mpi::MpiComm> mMpiCommPipelinePara;
    std::vector<std::unique_ptr<DecoderStepAsyncSend>> mDecStepAsyncSndHdls;
    std::vector<std::unique_ptr<DecoderSlotAsyncSend>> mDecSlotAsyncSndHdls;
    std::unique_ptr<tensorrt_llm::mpi::MpiWaitThread> mAsyncSendWaitThread;

    /******************** Tensor parallelism ********************/
    std::shared_ptr<tensorrt_llm::mpi::MpiComm> mMpiCommTensorPara;
    std::shared_ptr<runtime::AllReduceBuffers> mAllReduceBuffers;

    /******************** Runtime parameters ********************/
    // Flag to select fused or unfused context+generation execution
    bool mCtxGenFusion;
    // ID of current micro batch, changes after each iteration
    SizeType32 mMicroBatchId{0};
    // Number of micro batches. Multiple batches are used for overlapping setup and execution,
    // and in pipeline parallelism.
    SizeType32 mNumMicroBatches;
    // Number of buffers to be added to mBuffers.
    SizeType32 mNumBuffers;
    // Current operating beam width. Can be changed with changeBeamWidth function.
    SizeType32 mOperatingBeamWidth;
    // Runtime batch size optimized during execution for microBatchScheduler:
    /// The max batch size recommended by the dynamic tuner
    SizeType32 mMaxBatchSizeTunerRecommended;
    /// The min of mMaxBatchSize and mMaxBatchSizeTunerRecommended
    SizeType32 mMaxBatchSizeRuntime;
    // Runtime max num tokens optimized during execution for microBatchScheduler:
    /// Build time max num tokens
    std::optional<SizeType32> mMaxNumTokensStatic;
    /// The max num tokens recommended by the dynamic tuner
    SizeType32 mMaxNumTokensTunerRecommended;
    /// The min of mMaxNumTokens and mMaxNumTokensTunerRecommended
    std::optional<SizeType32> mMaxNumTokensRuntime;
    // Controls if generation logits should be gathered, so that returnGenerationLogits can be requested.
    bool mGatherGenerationLogits{false};

    /******************** Buffers ********************/
    // Buffers for each micro batch. Unfused path (mCtxGenFusion==false) uses two times the buffers.
    std::vector<std::shared_ptr<RuntimeBuffers>> mBuffers;
    // Decoder buffers for each micro batch.
    std::vector<DecoderInputBuffers> mDecoderInputBuffers;
    // Global buffer to interface with decoder. Slots in this buffer are selected by mSeqSlotManager.
    std::shared_ptr<DecoderBuffers> mDecoderBuffers;
    // Buffers for each slot in the decoder
    std::vector<std::shared_ptr<SlotDecoderBuffers>> mSlotDecoderBuffers;
    // PEFT table for each micro batch
    std::vector<PeftTable> mPeftTables;
    // Decoder input for each micro batch.
    std::vector<std::unique_ptr<runtime::decoder_batch::Input>> mDecodingInputs;
    std::unique_ptr<runtime::decoder_batch::Output> mDecodingOutput;

    /******************** Book keeping ********************/
    // List of requests in each micro batch
    std::vector<ScheduledRequests> mMicroBatchScheduledRequests;
    // Set of in-flight requests of *all* micro batches
    ReqIdsSet mInflightReqIds;
    // Requests that the scheduler selected to be paused
    ReqIdsSet mReqIdsToPause;
    // Stats collected in last iteration
    IterationStatsIFB mLastIterationStatsIFB{-1};
    // Iteration counter used to distinguish debug output
    executor::IterationType mIterCounter{0};
    // Debug tensors of last itreation
    TensorMap mLastIterationDebugTensors;
    // Cuda graph instances for each microbatch.
    std::vector<utils::CudaGraphExecutorCache> mCudaGraphExecutorCaches;

    /******************** Cache transceiver ********************/
    std::unique_ptr<BaseCacheTransceiver> mCacheTransceiver;

    /******************** Spec dec ***********************/
    std::unique_ptr<std::thread> mDraftModelSendLogitsThread;
    bool mSpeculativeDecodingFastLogits;
    std::atomic<bool> mDraftModelThreadShouldExit{false};
    bool mIsLeaderInOrchMode{false};
    // List of completed draft requests which logits will need to be sent to the target model.
    RequestVector mDraftRequestsWaitingToSendLogits;
    SizeType32 mSeamlessLADMaxDraftLen{0};
    bool mUseSeamlessLookahead{false};

    /******************** Algorithms ********************/
    // Algorithms are reentrant, they are assigned a state at
    // construction time and it is not modified through execution, hence they are const.
    // Schedulers that select which requests to run in each iteration
    std::unique_ptr<tensorrt_llm::batch_manager::CapacityScheduler const> mCapacityScheduler;
    std::unique_ptr<tensorrt_llm::batch_manager::MicroBatchScheduler const> mMicroBatchScheduler;
    std::unique_ptr<tensorrt_llm::batch_manager::PauseRequests const> mPauseRequests;
    std::unique_ptr<tensorrt_llm::batch_manager::AssignReqSeqSlots const> mAssignReqSeqSlots;
    std::unique_ptr<tensorrt_llm::batch_manager::AllocateKvCache const> mAllocateKvCache;
    std::unique_ptr<tensorrt_llm::batch_manager::HandleContextLogits const> mHandleContextLogits;
    std::unique_ptr<tensorrt_llm::batch_manager::HandleGenerationLogits const> mHandleGenerationLogits;
    std::unique_ptr<tensorrt_llm::batch_manager::GenerateRequestOptions const> mGenerateRequestOptions;
    std::unique_ptr<tensorrt_llm::batch_manager::LogitsPostProcessor const> mLogitsPostProcessor;
    std::unique_ptr<tensorrt_llm::batch_manager::MakeDecodingBatchInputOutput const> mMakeDecodingBatchInputOutput;
    std::unique_ptr<tensorrt_llm::batch_manager::CreateNewDecoderRequests const> mCreateNewDecoderRequests;
};

} // namespace tensorrt_llm::batch_manager
