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

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/arrayView.h"
#include "tensorrt_llm/executor/dynamicBatchTuner.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/intervalSet.h"
#include "tensorrt_llm/executor/model.h"
#include "tensorrt_llm/executor/orchestratorUtils.h"
#include "tensorrt_llm/executor/requestWithId.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <atomic>
#include <condition_variable>
#include <list>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace tensorrt_llm::executor
{

class RequestWithIdAsyncSend;
class CancelledRequestsAsyncSend;

class MpiMessageQueue
{
public:
    void push(MpiMessage&& message)
    {
        std::lock_guard<std::mutex> const lock(mMutex);
        mQueue.push(std::move(message));
        mCv.notify_one();
    }

    MpiMessage pop()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mCv.wait(lock, [this] { return !mQueue.empty(); });
        MpiMessage message = std::move(mQueue.front());
        mQueue.pop();
        return message;
    }

private:
    std::queue<MpiMessage> mQueue;
    std::mutex mMutex;
    std::condition_variable mCv;
};

class Executor::Impl

{
    using LlmRequestPtr = std::shared_ptr<batch_manager::LlmRequest>;
    using RequestList = std::list<LlmRequestPtr>;

    // When block reuse is enabled for context worker for disaggregated serving,
    // we need to store the last block id so that we can unpin the block when
    // the request is finished.
    struct InTransmissionItem
    {
        LlmRequestPtr request;
        std::optional<SizeType32> lastBlockId;
    };

    using InTransList = std::list<InTransmissionItem>;

public:
    Impl(std::filesystem::path const& modelPath, std::optional<std::filesystem::path> const& encoderModelPath,
        [[maybe_unused]] ModelType modelType, ExecutorConfig const& executorConfig);

    Impl(BufferView const& engineBufferView, std::string const& jsonConfigStr,
        std::optional<BufferView> const& encoderEngineBufferView,
        std::optional<std::string> const& encoderJsonConfigStr, [[maybe_unused]] ModelType modelType,
        ExecutorConfig const& executorConfig, std::optional<std::map<std::string, Tensor>> const& managedWeightsOpt);

    Impl(std::shared_ptr<Model> model, std::optional<std::shared_ptr<Model>> encoderModel,
        ExecutorConfig const& executorConfig);

    ~Impl();

    Impl(Impl const& executor) = delete;
    Impl& operator=(Impl const& executor) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

    IdType enqueueRequest(Request const& request);

    std::vector<IdType> enqueueRequests(std::vector<Request> const& requests);

    std::vector<IdType> enqueueRequests(common::ArrayView<Request const> const& requests);

    std::vector<Response> awaitResponses(std::optional<std::chrono::milliseconds> const& timeout = std::nullopt);

    std::vector<Response> awaitResponses(
        IdType const& reqId, std::optional<std::chrono::milliseconds> const& optTimeout = std::nullopt);

    std::vector<std::vector<Response>> awaitResponses(
        std::vector<IdType> const& requestIds, std::optional<std::chrono::milliseconds> const& timeout);

    SizeType32 getNumResponsesReady(std::optional<IdType> const& optId = std::nullopt) const;

    void cancelRequest(IdType requestId);

    void shutdown();

    std::deque<IterationStats> getLatestIterationStats();
    std::deque<RequestStatsPerIteration> getLatestRequestStats();
    std::deque<DebugTensorsPerIteration> getLatestDebugTensors();

    bool canEnqueueRequests() const;

    bool isParticipant() const;

    std::optional<std::shared_ptr<KVCacheEventManager>> getKVCacheEventManager() const;

private:
    using RtTensorPtr = runtime::ITensor::SharedPtr;
    using CudaStreamPtr = runtime::BufferManager::CudaStreamPtr;
    using LlmRequestLogitsPostProcessor
        = std::function<void(IdType, RtTensorPtr&, BeamTokens const&, CudaStreamPtr, std::optional<IdType>)>;

    void initialize(ExecutorConfig const& executorConfig);

    void loadModel(std::optional<std::filesystem::path> const& modelPath, std::optional<BufferView> const& engineBuffer,
        runtime::GptJsonConfig const& jsonConfig, ExecutorConfig const& executorConfig, bool isEncoder,
        std::optional<std::map<std::string, Tensor>> const& managedWeightsOpt);

    std::shared_ptr<Model> createModel(runtime::RawEngine const& rawEngine, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, ExecutorConfig const& executorConfig);

    std::shared_ptr<Model> createEncoderModel(runtime::RawEngine const& rawEngine,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        ExecutorConfig const& executorConfig);

    void setOrchLeaderComm(SizeType32 tp, SizeType32 pp, SizeType32 cp, ParallelConfig const& parallelConfig);

    void initializeCommAndWorkers(SizeType32 tp, SizeType32 pp, SizeType32 cp, ExecutorConfig const& executorConfig,
        std::optional<ModelType> modelType = std::nullopt,
        std::optional<std::filesystem::path> const& modelPath = std::nullopt,
        std::optional<runtime::WorldConfig> const& worldConfig = std::nullopt,
        std::optional<runtime::GptJsonConfig> const& decoderGptJsonConfig = std::nullopt);

    static void validateParallelConfig(ParallelConfig const& parallelConfig, std::optional<ModelType> modelType,
        std::optional<std::filesystem::path> const& modelPath);

    void initializeOrchestrator(SizeType32 tp, SizeType32 pp, SizeType32 cp, ExecutorConfig const& executorConfig,
        ParallelConfig parallelConfig, ModelType modelType, std::filesystem::path const& modelPath);

    void initializeWorkers(SizeType32 tp, SizeType32 pp, SizeType32 cp, ParallelConfig& parallelConfig,
        std::optional<runtime::WorldConfig> const& worldConfig = std::nullopt,
        std::optional<runtime::GptJsonConfig> const& decoderGptJsonConfig = std::nullopt);

    void initializeLogitsPostProcessorBatched(LogitsPostProcessorConfig const& logitsProcConfig);

    IdType generateReqId()
    {
        return (mLastReqId++ % UINT64_MAX);
    }

    std::vector<RequestWithId> getLeaderNewReqWithIds(
        SizeType32 numActiveRequests, std::optional<PriorityType> lowestPriorityActive);
    std::vector<RequestWithId> getNewReqWithIds(
        SizeType32 numActiveRequests, std::optional<PriorityType> lowestPriorityActive);

    std::tuple<Executor::Impl::RequestList, double> fetchNewRequests(
        SizeType32 numActiveRequests, std::optional<PriorityType> lowestPriorityActive);

    void forwardSync(RequestList& activeRequests);

    void forwardAsync(RequestList& activeRequests);

    void prepRequestsForEncoderSkip(RequestList& activeRequests);

    void terminateActiveRequests(RequestList& activeRequests, std::string const& err);

    IterationStats getCurrentIterationStats(RequestList const& activeRequests, double iterLatencyMS,
        SizeType32 numNewActiveRequests, double newActiveRequestsQueueLatencyMS, SizeType32 numCompletedRequests);

    void appendCurrentIterStats(IterationStats&& currentIterStats);
    void appendMultipleIterStats(std::vector<IterationStats>&& currentIterStatsVec);
    void updateIterationStats(RequestList const& activeRequests, double iterLatencyMS, SizeType32 numNewActiveRequests,
        double newActiveRequestsQueueLatencyMS, SizeType32 numCompletedRequests, bool flushToOrchestrator);
    void appendCurrentRequestStats(RequestStatsPerIteration&& currentRequestStats);
    void appendMultipleRequestStats(std::vector<RequestStatsPerIteration>&& currentRequestStatsVec);
    RequestStatsPerIteration getCurrentRequestStats(
        RequestList const& activeRequests, RequestList const& finishedRequests);
    void updateRequestStats(
        RequestList const& activeRequests, RequestList const& finishedRequests, bool flushToOrchestrator);

    void appendCurrentDebugTensors();

    void terminateCancelledRequests(RequestList& activeRequests);

    void terminateContextFinishedRequests(InTransList& inTransmissionRequests);

    void appendNewResponses(std::vector<Response>&& newResponses);

    /// @brief Populates new responses from active requests.
    ///        Active requests that have completed are erased from activeRequests
    ///        and returned for bookkeeping.
    /// @return A list of requests that have completed.
    RequestList populateNewResponses(
        RequestList& activeRequests, InTransList& inTransmissionRequests, std::vector<Response>& newResponses);

    void executionLoop();

    void enqueueTerminateRequest();
    void enqueueNewResponses(std::vector<Response>&& newResponses);

    LlmRequestLogitsPostProcessor getLogitsPostProcessor(std::string const& name);
    void setupDynamicLogitsPostProcessors(std::vector<RequestWithId>& newReqWithIds);
    void cleanupDynamicLogitsPostProcessors(RequestList const& finishedRequests);

    void orchSendReqThread();
    void orchRecvThread(mpi::MpiTag idTag, mpi::MpiTag dataTag);
    void leaderRecvReqThread();
    void leaderSendThread(MpiMessageQueue& sendQueue, mpi::MpiTag idTag, mpi::MpiTag dataTag);

    void addTerminatedReqId(std::vector<Response> const& responses, IdType const& reqId);

    // Check that the current process is the leader or orchestrator
    void checkParallelApiUsage(std::string const& methodName) const;

    // These functions wait for MPI async sends on separate threads
    void requestWithIdWaitThread();
    void cancelledRequestsWaitThread();
    // These functions send data from leader to pipeline leader on separate threads
    void requestWithIdLeaderThread();
    void cancelledRequestsLeaderThread();

    /// @brief mark requests that have timed out before ever being executed as finished.
    ///        uses cancellation based on communication mode.
    ///
    /// @param activeRequests [in] List of active requests to check for timeouts
    void finishTimedOutRequests(RequestList const& activeRequests);

    // The model to execute
    std::shared_ptr<Model> mModel = nullptr;
    std::shared_ptr<Model> mEncoderModel = nullptr;

    // The maximum number of activeRequests
    SizeType32 mMaxNumActiveRequests;

    // Thread the executes the main loop
    std::thread mExecutionThread;

    // Atomic that indicates threads should shutdown
    std::atomic<bool> mShutdown;

    // Atomic that indicates if shutdown method has been called
    std::atomic<bool> mShutdownCalled = false;

    // Queued requests
    std::mutex mQueuedReqMtx;
    std::condition_variable mQueuedReqCv;
    std::deque<RequestWithId> mQueuedRequests;
    std::optional<SizeType32> mMaxQueueSize;

    // Cancelled requests
    std::mutex mCancelReqMtx;
    std::unordered_set<IdType> mCancelledReqIds;
    std::unordered_set<IdType> mPipelineCancelledReqIds;

    // Ready responses
    std::unordered_map<IdType, std::vector<Response>> mResponses;
    mutable std::mutex mResponsesMtx;
    std::condition_variable mResponsesCv;

    // Since the request IDs are generated sequentially, IntervalSet is preferred over unordered_set for its efficient
    // memory usage to stores request ID intervals rather than individual request ID numbers.
    IntervalSet<IdType> mTerminatedReqIds;

    std::unordered_map<IdType, std::vector<IdType>> mChildReqIdsMap;

    // Iteration stats
    IterationType mIterStatsMaxIterations;
    std::mutex mIterStatsMtx;
    std::deque<IterationStats> mIterationStats;

    // Request stats
    IterationType mRequestStatsMaxIterations;
    std::mutex mRequestStatsMtx;
    std::deque<RequestStatsPerIteration> mRequestStats;

    // Debug
    IterationType mDebugTensorsMaxIterations;
    std::mutex mDebugTensorsMtx;
    std::deque<DebugTensorsPerIteration> mDebugTensors;

    IdType mLastReqId = 1;

    static constexpr IdType mTerminateReqId = 0;

    BatchingType mBatchingType;
    bool mIsSchedulerMaxUtilization;
    bool mIsSchedulerGuaranteedNoEvict;
    bool mIsChunkedContext;
    bool mPromptTableOffloading;

    CommunicationMode mCommMode;
    bool mIsWorker = false;
    bool mIsLeader = false;
    bool mIsPipelineLeader = false;
    bool mUsePipelineParallel = false;

    std::unordered_map<std::string, LogitsPostProcessor> mLogitsPostProcessorMap;
    std::optional<Model::LogitsPostProcessorBatched> mLogitsPostProcessorBatched;

    bool mIsOrchestrator = false;
    std::shared_ptr<tensorrt_llm::mpi::MpiComm> mOrchLeaderComm;

    std::thread mOrchSendReqThread;
    std::thread mOrchRecvThread;
    std::thread mLeaderRecvReqThread;
    std::thread mLeaderSendThread;

    int32_t mRecvPollPeriodMs = 0;

    int32_t mLeaderRank = -1;
    int32_t mOrchRank = 0;
    int32_t mWorldRank = -1;
    int32_t mDeviceId = 0;

    MpiMessageQueue mSendQueue;

    std::shared_ptr<tensorrt_llm::mpi::MpiComm> mCommTensorParallel;
    std::shared_ptr<tensorrt_llm::mpi::MpiComm> mCommPipelineParallel;
    std::shared_ptr<tensorrt_llm::mpi::MpiComm> mCommContextParallel;
    std::unique_ptr<RequestWithIdAsyncSend> mRequestWithIdAsyncSndHdl;
    std::unique_ptr<CancelledRequestsAsyncSend> mCancelledRequestsAsyncSndHdl;
    std::unique_ptr<std::thread> mRequestWithIdLeaderThread;
    std::unique_ptr<std::thread> mCancelledRequestsLeaderThread;
    std::unique_ptr<tensorrt_llm::mpi::MpiWaitThread> mRequestWithIdWaitThread;
    std::unique_ptr<tensorrt_llm::mpi::MpiWaitThread> mCancelledRequestsWaitThread;

    // for validating requests
    bool mEnableBlockReuse;

    inline static std::string const kPROFILE_START_STOP_ENV_VAR_NAME = "TLLM_PROFILE_START_STOP";
    inline static std::string const kLEGACY_PROFILE_START_STOP_ENV_VAR_NAME = "TLLM_GPTM_PROFILE_START_STOP";

    std::shared_ptr<DynamicBatchTuner> mDynamicBatchTuner;
};

} // namespace tensorrt_llm::executor
