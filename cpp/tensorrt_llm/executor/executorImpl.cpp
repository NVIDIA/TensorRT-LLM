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

#include "tensorrt_llm/executor/executorImpl.h"
#include "tensorrt_llm/batch_manager/trtEncoderModel.h"
#include "tensorrt_llm/batch_manager/trtGptModelFactory.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaProfilerUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/common/timestampUtils.h"
#include "tensorrt_llm/common/utils.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/orchestratorUtils.h"
#include "tensorrt_llm/executor/requestUtils.h"
#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/executor/version.h"
#include "tensorrt_llm/runtime/loraCache.h"
#include "tensorrt_llm/runtime/memoryCounters.h"
#include "tensorrt_llm/runtime/utils/mpiTags.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuda_profiler_api.h>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>

namespace tensorrt_llm::executor
{

namespace
{

[[nodiscard]] bool executorConfigIsValid(ExecutorConfig const& executorConfig, runtime::ModelConfig const& modelConfig)
{
    // Make sure logic in this function matches fixExecutorConfig
    if (executorConfig.getEnableChunkedContext())
    {
        if (modelConfig.isRnnBased() || !modelConfig.isKVCacheEnabled() || !modelConfig.getPagedContextFMHA())
        {
            return false;
        }
    }
    return true;
}

[[nodiscard]] ExecutorConfig fixExecutorConfig(
    ExecutorConfig const& executorConfig, runtime::ModelConfig const& modelConfig)
{
    // Make sure logic in this function matches executorConfigIsValid
    auto fixedExecutorConfig = executorConfig;
    // Disable chunked context when not supported
    if (executorConfig.getEnableChunkedContext())
    {
        if (modelConfig.isRnnBased() || !modelConfig.isKVCacheEnabled() || !modelConfig.getPagedContextFMHA())
        {
            fixedExecutorConfig.setEnableChunkedContext(false);
            TLLM_LOG_WARNING(
                "Chunked context is not supported for this configuration and will be disabled. "
                "Related configs: RNNBased: %d, KVCacheEnabled: %d, PagedContextFMHA: %d",
                modelConfig.isRnnBased(), modelConfig.isKVCacheEnabled(), modelConfig.getPagedContextFMHA());
        }
    }
    return fixedExecutorConfig;
}

SizeType32 getNumChildRequests(Request const& request)
{
    auto samplingConfig = request.getSamplingConfig();
    return samplingConfig.getBeamWidth() > 1 ? 0 : samplingConfig.getNumReturnSequences().value_or(1) - 1;
}

} // namespace

/// @brief Version of TRT-LLM as defined in tensorrt_llm/version.py
char const* version() noexcept
{
    return kTensorRtLlmVersion;
}

class CancelledRequestsAsyncSend
{
public:
    CancelledRequestsAsyncSend(std::shared_ptr<tensorrt_llm::mpi::MpiComm> const& commSession,
        std::unordered_set<IdType> const& cancelledReqIds, int peer)
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        mNumReq = static_cast<int64_t>(cancelledReqIds.size());
        TLLM_LOG_DEBUG("start send %ld cancelled requests to rank %d", mNumReq, peer);
        mRequest1
            = commSession->sendAsync(&mNumReq, 1, mpi::MpiType::kINT64, peer, mpi::MpiTag::kCancelledRequestsNumReq);
        if (mNumReq > 0)
        {
            mIds.assign(cancelledReqIds.begin(), cancelledReqIds.end());
            mRequest2 = commSession->sendAsync(
                mIds.data(), mIds.size(), mpi::MpiType::kUINT64, peer, mpi::MpiTag::kCancelledRequestsIds);
        }
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    }

    ~CancelledRequestsAsyncSend()
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        mRequest1->wait();
        if (mRequest2)
        {
            mRequest2->wait();
        }
        TLLM_LOG_DEBUG("end send cancelled requests");
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    }

    CancelledRequestsAsyncSend(CancelledRequestsAsyncSend const& executor) = delete;
    CancelledRequestsAsyncSend& operator=(CancelledRequestsAsyncSend const& executor) = delete;
    CancelledRequestsAsyncSend(CancelledRequestsAsyncSend&&) = delete;
    CancelledRequestsAsyncSend& operator=(CancelledRequestsAsyncSend&&) = delete;

    static std::unordered_set<IdType> cancelledRequestsRecv(
        std::shared_ptr<tensorrt_llm::mpi::MpiComm> const& commSession, int peer)
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        TLLM_LOG_DEBUG("start recv cancelled requests from rank %d", peer);
        std::unordered_set<IdType> cancelledReqIds;
        int64_t numReq{0};
        commSession->recv(&numReq, 1, mpi::MpiType::kINT64, peer, mpi::MpiTag::kCancelledRequestsNumReq);
        TLLM_LOG_DEBUG("recv %ld cancelled requests", numReq);
        if (numReq > 0)
        {
            std::vector<IdType> buffer(numReq);
            commSession->recv(
                buffer.data(), buffer.size(), mpi::MpiType::kUINT64, peer, mpi::MpiTag::kCancelledRequestsIds);
            cancelledReqIds = std::unordered_set<IdType>(buffer.begin(), buffer.end());
        }
        TLLM_LOG_DEBUG("end recv cancelled requests from rank %d", peer);
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return cancelledReqIds;
    }

private:
    int64_t mNumReq;
    std::vector<IdType> mIds;
    std::shared_ptr<tensorrt_llm::mpi::MpiRequest> mRequest1;
    std::shared_ptr<tensorrt_llm::mpi::MpiRequest> mRequest2;
};

class RequestWithIdAsyncSend
{
public:
    RequestWithIdAsyncSend(std::shared_ptr<tensorrt_llm::mpi::MpiComm> const& commSession,
        std::vector<RequestWithId> const& reqWithIds, int peer)
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        TLLM_LOG_DEBUG("start send requests to rank %d", peer);
        mNumReq = static_cast<int64_t>(reqWithIds.size());
        mRequest1 = commSession->sendAsync(&mNumReq, 1, mpi::MpiType::kINT64, peer, mpi::MpiTag::kRequestWithIdNumReq);
        if (mNumReq > 0)
        {
            mPacked = RequestWithId::serializeReqWithIds(reqWithIds);
            mVecSize = static_cast<int64_t>(mPacked.size());
            mRequest2
                = commSession->sendAsync(&mVecSize, 1, mpi::MpiType::kINT64, peer, mpi::MpiTag::kRequestWithIdVecSize);
            mRequest3 = commSession->sendAsync(
                mPacked.data(), mPacked.size(), mpi::MpiType::kCHAR, peer, mpi::MpiTag::kRequestWithIdPacked);
        }
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    }

    ~RequestWithIdAsyncSend()
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        mRequest1->wait();
        if (mRequest2)
        {
            mRequest2->wait();
        }
        if (mRequest3)
        {
            mRequest3->wait();
        }
        TLLM_LOG_DEBUG("end send requests");
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    }

    RequestWithIdAsyncSend(RequestWithIdAsyncSend const& executor) = delete;
    RequestWithIdAsyncSend& operator=(RequestWithIdAsyncSend const& executor) = delete;
    RequestWithIdAsyncSend(RequestWithIdAsyncSend&&) = delete;
    RequestWithIdAsyncSend& operator=(RequestWithIdAsyncSend&&) = delete;

    static std::vector<RequestWithId> requestWithIdRecv(
        std::shared_ptr<tensorrt_llm::mpi::MpiComm> const& commSession, int peer)
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        TLLM_LOG_DEBUG("start recv requests from rank %d", peer);
        std::vector<RequestWithId> reqWithIds;
        int64_t numReq{0};
        commSession->recv(&numReq, 1, mpi::MpiType::kINT64, peer, mpi::MpiTag::kRequestWithIdNumReq);
        if (numReq > 0)
        {
            std::vector<char> buffer;
            int64_t vecSize = 0;
            commSession->recv(&vecSize, 1, mpi::MpiType::kINT64, peer, mpi::MpiTag::kRequestWithIdVecSize);
            buffer.resize(vecSize);
            commSession->recv(
                buffer.data(), buffer.size(), mpi::MpiType::kCHAR, peer, mpi::MpiTag::kRequestWithIdPacked);
            reqWithIds = RequestWithId::deserializeReqWithIds(buffer);
        }
        TLLM_LOG_DEBUG("end recv requests from rank %d", peer);
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return reqWithIds;
    }

private:
    int64_t mNumReq;
    int64_t mVecSize;
    std::vector<char> mPacked;
    std::shared_ptr<tensorrt_llm::mpi::MpiRequest> mRequest1;
    std::shared_ptr<tensorrt_llm::mpi::MpiRequest> mRequest2;
    std::shared_ptr<tensorrt_llm::mpi::MpiRequest> mRequest3;
};

void Executor::Impl::loadModel(std::optional<std::filesystem::path> const& modelPathOpt,
    std::optional<BufferView> const& engineBufferOpt, runtime::GptJsonConfig const& jsonConfig,
    ExecutorConfig const& executorConfig, bool isEncoder,
    std::optional<std::map<std::string, Tensor>> const& managedWeightsOpt)
{
    auto const gpusPerNode = jsonConfig.getGpusPerNode();
    auto const tp = jsonConfig.getTensorParallelism();
    auto const pp = jsonConfig.getPipelineParallelism();
    auto const cp = jsonConfig.getContextParallelism();
    auto parallelConfig = executorConfig.getParallelConfig().value_or(ParallelConfig());
    auto worldConfig = runtime::WorldConfig::mpi(gpusPerNode, tp, pp, cp, parallelConfig.getDeviceIds());

    TLLM_CHECK_WITH_INFO(modelPathOpt.has_value() || engineBufferOpt.has_value(),
        "Either engine path or deserialized engine buffer should be given to load the model properly.");
    auto rawEngine = engineBufferOpt.has_value()
        ? runtime::RawEngine(engineBufferOpt.value().data(), engineBufferOpt.value().size())
        : runtime::RawEngine(modelPathOpt.value() / jsonConfig.engineFilename(worldConfig));

    if (rawEngine.getType() != tensorrt_llm::runtime::RawEngine::FilePath)
    {
        if (modelPathOpt.has_value())
        {
            rawEngine.setPath(modelPathOpt.value() / jsonConfig.engineFilename(worldConfig));
            if (managedWeightsOpt.has_value())
            {
                TLLM_LOG_WARNING(
                    "Executor::Impl::loadModel: managedWeightsOpt argument is ignored when loading engine from file.");
            }
        }
        else if (managedWeightsOpt.has_value())
        {
            rawEngine.setManagedWeightsMap(managedWeightsOpt.value());
        }
    }

    auto const& modelConfig = jsonConfig.getModelConfig();

    if (isEncoder)
    {
        mEncoderModel = createEncoderModel(rawEngine, modelConfig, worldConfig, executorConfig);
    }
    else
    {
        mModel = createModel(rawEngine, modelConfig, worldConfig, executorConfig);
    }
};

Executor::Impl::Impl(std::filesystem::path const& modelPath,
    std::optional<std::filesystem::path> const& encoderModelPath, ModelType const modelType,
    ExecutorConfig const& executorConfig)
{
    auto decoderJsonConfig = runtime::GptJsonConfig::parse(modelPath / "config.json");

    // for now, assume encoder & decoder models share the same MPI config
    auto const tp = decoderJsonConfig.getTensorParallelism();
    auto const pp = decoderJsonConfig.getPipelineParallelism();
    auto const cp = decoderJsonConfig.getContextParallelism();
    initializeCommAndWorkers(tp, pp, cp, executorConfig, modelType, modelPath, std::nullopt, decoderJsonConfig);

    if (mIsWorker)
    {
        if (modelType == ModelType::kENCODER_DECODER)
        {
            if (encoderModelPath.has_value())
            {
                auto const encoderJsonConfig = runtime::GptJsonConfig::parse(encoderModelPath.value() / "config.json");

                auto const encoderMaxInputLen = encoderJsonConfig.getModelConfig().getMaxInputLen();
                auto const encoderHiddenSize = encoderJsonConfig.getModelConfig().getHiddenSize()
                    * encoderJsonConfig.getTensorParallelism(); // recover full hidden size
                // add encoder info to decoder for encoder-decoder models
                // note: GptJsonConfig can no longer have modelConfig as const member since it must be mutable here
                decoderJsonConfig.getModelConfigMutable().setMaxEncoderLen(encoderMaxInputLen);
                decoderJsonConfig.getModelConfigMutable().setEncoderHiddenSize(encoderHiddenSize);

                loadModel(
                    encoderModelPath.value(), std::nullopt, encoderJsonConfig, executorConfig, true, std::nullopt);
            }
            else
            {
                TLLM_LOG_WARNING("Encoder model path not provided. Skipping Encoder Run.");
            }
        }
        loadModel(modelPath, std::nullopt, decoderJsonConfig, executorConfig, false, std::nullopt);
    }
    initialize(executorConfig);
}

Executor::Impl::Impl(BufferView const& engineBufferView, std::string const& jsonConfigStr,
    std::optional<BufferView> const& encoderEngineBufferView, std::optional<std::string> const& encoderJsonConfigStr,
    ModelType const modelType, ExecutorConfig const& executorConfig,
    std::optional<std::map<std::string, Tensor>> const& managedWeightsOpt)
{
    auto decoderJsonConfig = runtime::GptJsonConfig::parse(jsonConfigStr);

    // for now, assume encoder & decoder models share the same MPI config
    auto const tp = decoderJsonConfig.getTensorParallelism();
    auto const pp = decoderJsonConfig.getPipelineParallelism();
    auto const cp = decoderJsonConfig.getContextParallelism();
    initializeCommAndWorkers(tp, pp, cp, executorConfig, modelType, std::nullopt, std::nullopt, decoderJsonConfig);

    if (mIsWorker)
    {
        if (modelType == ModelType::kENCODER_DECODER)
        {
            TLLM_CHECK(encoderEngineBufferView.has_value() && encoderJsonConfigStr.has_value());
            TLLM_CHECK_WITH_INFO(
                !managedWeightsOpt.has_value(), "Managed weights are not supported for enc-dec models");

            auto const encoderJsonConfig = runtime::GptJsonConfig::parse(encoderJsonConfigStr.value());

            auto const encoderMaxInputLen = encoderJsonConfig.getModelConfig().getMaxInputLen();
            auto const encoderHiddenSize = encoderJsonConfig.getModelConfig().getHiddenSize()
                * encoderJsonConfig.getTensorParallelism(); // recover full hidden size
            // add encoder info to decoder for encoder-decoder models
            // note: GptJsonConfig can no longer have modelConfig as const member since it must be mutable here
            decoderJsonConfig.getModelConfigMutable().setMaxEncoderLen(encoderMaxInputLen);
            decoderJsonConfig.getModelConfigMutable().setEncoderHiddenSize(encoderHiddenSize);

            loadModel(
                std::nullopt, encoderEngineBufferView.value(), encoderJsonConfig, executorConfig, true, std::nullopt);
        }
        loadModel(std::nullopt, engineBufferView, decoderJsonConfig, executorConfig, false, managedWeightsOpt);
    }
    initialize(executorConfig);
}

Executor::Impl::Impl(std::shared_ptr<Model> model, std::optional<std::shared_ptr<Model>> encoderModel,
    ExecutorConfig const& executorConfig)
{
    auto const& worldConfig = model->getWorldConfig();
    auto const tp = worldConfig.getTensorParallelism();
    auto const pp = worldConfig.getPipelineParallelism();
    auto const cp = worldConfig.getContextParallelism();
    auto const modelType = encoderModel.has_value() ? ModelType::kENCODER_DECODER : ModelType::kDECODER_ONLY;
    initializeCommAndWorkers(tp, pp, cp, executorConfig, modelType, std::nullopt, worldConfig);
    if (modelType == ModelType::kENCODER_DECODER)
    {
        mEncoderModel = encoderModel.value();
    }
    mModel = std::move(model);
    initialize(executorConfig);
}

Executor::Impl::~Impl()
{
    shutdown();
}

void Executor::Impl::initialize(ExecutorConfig const& executorConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mShutdown = false;
    mShutdownCalled = false;
    mIterStatsMaxIterations = executorConfig.getIterStatsMaxIterations();
    mRequestStatsMaxIterations = executorConfig.getRequestStatsMaxIterations();
    mDebugTensorsMaxIterations
        = executorConfig.getDebugConfig() ? executorConfig.getDebugConfig()->getDebugTensorsMaxIterations() : 0;
    TLLM_CHECK_WITH_INFO(mDebugTensorsMaxIterations == 0 || mCommMode == CommunicationMode::kLEADER,
        "debugTensorsMaxIterations > 0 is only allowed in leader mode.");
    mBatchingType = executorConfig.getBatchingType();
    mIsSchedulerMaxUtilization = (executorConfig.getSchedulerConfig().getCapacitySchedulerPolicy()
        == CapacitySchedulerPolicy::kMAX_UTILIZATION);
    mIsSchedulerGuaranteedNoEvict = (executorConfig.getSchedulerConfig().getCapacitySchedulerPolicy()
        == CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT);
    mIsChunkedContext = executorConfig.getEnableChunkedContext();
    mPromptTableOffloading = executorConfig.getPromptTableOffloading();
    mMaxQueueSize = executorConfig.getMaxQueueSize();

    mLastReqId = 1;

    auto const& logitsProcConfig = executorConfig.getLogitsPostProcessorConfig();
    if (logitsProcConfig.has_value())
    {
        mLogitsPostProcessorMap = logitsProcConfig.value().getProcessorMap().value_or(LogitsPostProcessorMap{});
        initializeLogitsPostProcessorBatched(logitsProcConfig.value());
        if (!logitsProcConfig.value().getReplicate())
        {
            mModel->setReplicateLogitsPostProcessor(false);
        }
    }

    auto const& commComm = COMM_SESSION;
    int32_t const commSize = commComm.getSize();
    if (mIsWorker)
    {
        if (commSize > 1)
        {
            auto const& worldConfig = mModel->getWorldConfig();
            auto const& commSession = COMM_SESSION;
            auto const& rank = commSession.getRank();
            auto const& tp = worldConfig.getTensorParallelism();
            auto const& cp = worldConfig.getContextParallelism();

            mCommTensorParallel = std::make_shared<tensorrt_llm::mpi::MpiComm>(
                commSession.split(rank / tp, worldConfig.getTensorParallelRank()));
            mCommContextParallel = std::make_shared<tensorrt_llm::mpi::MpiComm>(
                commSession.split(rank / (tp * cp) * tp + rank % tp, worldConfig.getContextParallelRank()));
            mCommPipelineParallel = std::make_shared<tensorrt_llm::mpi::MpiComm>(
                commSession.split(rank % (tp * cp), worldConfig.getPipelineParallelRank()));

            if (worldConfig.isPipelineParallel())
            {
                mRequestWithIdWaitThread = std::make_unique<tensorrt_llm::mpi::MpiWaitThread>(
                    "requestWithIdWaitThread", [this]() { mRequestWithIdAsyncSndHdl.reset(nullptr); });
                mCancelledRequestsWaitThread = std::make_unique<tensorrt_llm::mpi::MpiWaitThread>(
                    "cancelledRequestsWaitThread", [this]() { mCancelledRequestsAsyncSndHdl.reset(nullptr); });
                if (mIsLeader)
                {
                    mRequestWithIdLeaderThread
                        = std::make_unique<std::thread>(&Executor::Impl::requestWithIdLeaderThread, this);
                    mCancelledRequestsLeaderThread
                        = std::make_unique<std::thread>(&Executor::Impl::cancelledRequestsLeaderThread, this);
                }
            }
        }
        // Launch the execution thread
        mMaxNumActiveRequests = mModel->getMaxNumSequences();
        mExecutionThread = std::thread(&Impl::executionLoop, this);
    }

    mEnableBlockReuse = executorConfig.getKvCacheConfig().getEnableBlockReuse();

    auto const& dynamicBatchConfig = executorConfig.getSchedulerConfig().getDynamicBatchConfig();
    if (dynamicBatchConfig)
    {
        if (mIsWorker)
        {
            if (mModel->getModelConfig().isTransformerBased() && mModel->getModelConfig().isKVCacheEnabled())
            {
                mDynamicBatchTuner = std::make_shared<DynamicBatchTuner>(dynamicBatchConfig.value());
            }
            else
            {
                TLLM_LOG_WARNING("Dynamic batch tuner can only support transformer models that use KV cache.");
            }
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::shared_ptr<Model> Executor::Impl::createModel(runtime::RawEngine const& rawEngine,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
    ExecutorConfig const& executorConfig)
{
    auto const gptModelType = [&executorConfig, &modelConfig]()
    {
        switch (executorConfig.getBatchingType())
        {
        case BatchingType::kSTATIC:
            TLLM_THROW(
                "Static batching type is deprecated. Please use in-flight batching with "
                "CapacitySchedulerPolicy::kSTATIC_BATCH instead.");
        case BatchingType::kINFLIGHT:
            return modelConfig.isRnnBased() ? batch_manager::TrtGptModelType::InflightBatching
                                            : batch_manager::TrtGptModelType::InflightFusedBatching;
        default: TLLM_THROW("Invalid batching strategy");
        }
    }();

    bool const isLeaderInOrchMode = (mCommMode == CommunicationMode::kORCHESTRATOR) && mIsLeader;
    auto const& fixedExecutorConfig = executorConfigIsValid(executorConfig, modelConfig)
        ? executorConfig
        : fixExecutorConfig(executorConfig, modelConfig);

    return batch_manager::TrtGptModelFactory::create(
        rawEngine, modelConfig, worldConfig, gptModelType, fixedExecutorConfig, isLeaderInOrchMode);
}

std::shared_ptr<Model> Executor::Impl::createEncoderModel(runtime::RawEngine const& rawEngine,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
    ExecutorConfig const& executorConfig)
{
    auto fixedExecutorConfig = ExecutorConfig{};
    fixedExecutorConfig.setSchedulerConfig(executorConfig.getSchedulerConfig());
    return std::make_shared<batch_manager::TrtEncoderModel>(
        modelConfig, worldConfig, rawEngine, std::make_shared<runtime::TllmLogger>(), fixedExecutorConfig);
}

void Executor::Impl::setOrchLeaderComm(
    SizeType32 tp, SizeType32 pp, SizeType32 cp, ParallelConfig const& parallelConfig)
{
#if ENABLE_MULTI_DEVICE
    auto optOrchestratorConfig = parallelConfig.getOrchestratorConfig();
    if (optOrchestratorConfig.value().getIsOrchestrator())
    {
        TLLM_CHECK_WITH_INFO(mWorldRank == 0, "Rank 0 must be orchestrator");
    }

    TLLM_CHECK_WITH_INFO(parallelConfig.getParticipantIds(),
        "When not spawning processes in orchestrator mode, participant IDs must be provided");
    auto participantIds = parallelConfig.getParticipantIds().value();

    TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(participantIds.size()) == tp * pp * cp,
        "When specifying participantIds, participantIds size must be equal to tp*pp*cp");

    bool isLeader = (mWorldRank == participantIds.front());
    bool isOrchestrator = (mWorldRank == 0);

    // OrchLeaderComm rank 0 is orchestrator, rank 1 is leader
    mOrchRank = 0;
    mLeaderRank = 1;

    // Create a leaderOrch comm
    std::vector<int32_t> leaderOrchRanks{0, participantIds.front()};

    MPI_Group worldGroup = nullptr;
    MPICHECK(MPI_Comm_group(MPI_COMM_WORLD, &worldGroup)); // NOLINT
    int worldGroupRank = 0;
    MPI_Group_rank(worldGroup, &worldGroupRank);

    int worldSize = 0;
    MPICHECK(MPI_Group_size(worldGroup, &worldSize)); // NOLINT
    TLLM_CHECK_WITH_INFO(participantIds.front() < worldSize, "Not enough ranks in world");

    MPI_Group leaderOrchCommGroup = nullptr;
    MPICHECK(
        MPI_Group_incl(worldGroup, leaderOrchRanks.size(), leaderOrchRanks.data(), &leaderOrchCommGroup)); // NOLINT
    int leaderOrchGroupRank = 0;
    int leaderOrchGroupSize = 0;
    MPI_Group_rank(leaderOrchCommGroup, &leaderOrchGroupRank);
    MPI_Group_size(leaderOrchCommGroup, &leaderOrchGroupSize);

    if (isOrchestrator || isLeader)
    {
        MPI_Comm leaderOrchComm = nullptr;
        MPICHECK(MPI_Comm_create_group(
            MPI_COMM_WORLD, leaderOrchCommGroup, participantIds.front(), &leaderOrchComm)); // NOLINT
        mOrchLeaderComm = std::make_shared<tensorrt_llm::mpi::MpiComm>(leaderOrchComm, false);
    }
    else
    {
        mOrchLeaderComm = nullptr;
    }
#endif // ENABLE_MULTI_DEVICE
}

void Executor::Impl::initializeCommAndWorkers(SizeType32 tp, SizeType32 pp, SizeType32 cp,
    ExecutorConfig const& executorConfig, std::optional<ModelType> modelType,
    std::optional<std::filesystem::path> const& modelPath, std::optional<runtime::WorldConfig> const& worldConfig,
    std::optional<runtime::GptJsonConfig> const& decoderGptJsonConfig)
{
    if (modelType.has_value() && modelType.value() == ModelType::kENCODER_DECODER)
    {
        TLLM_CHECK_WITH_INFO(pp == 1,
            "Encoder-Decoder C++ runtime doesn't support Pipeline Parallelism currently. Please switch to Python "
            "runtime for PP mode, if necessary.");
    }

    tensorrt_llm::mpi::initialize(tensorrt_llm::mpi::MpiThreadSupport::THREAD_MULTIPLE);
    mWorldRank = tensorrt_llm::mpi::MpiComm::world().getRank();
    mUsePipelineParallel = pp > 1;

    auto parallelConfig = executorConfig.getParallelConfig().value_or(ParallelConfig());
    validateParallelConfig(parallelConfig, modelType, modelPath);

    mCommMode = parallelConfig.getCommunicationMode();
    auto optOrchestratorConfig = parallelConfig.getOrchestratorConfig();

    mRecvPollPeriodMs = executorConfig.getRecvPollPeriodMs();

    // Need to create communicator between orchestrator and leader if not spawning processes in orchestrator mode
    if (mCommMode == CommunicationMode::kORCHESTRATOR && !optOrchestratorConfig.value().getSpawnProcesses())
    {
        setOrchLeaderComm(tp, pp, cp, parallelConfig);
    }

    if (mCommMode == CommunicationMode::kORCHESTRATOR && optOrchestratorConfig.value().getIsOrchestrator())
    {
        initializeOrchestrator(tp, pp, cp, executorConfig, parallelConfig, modelType.value(), modelPath.value());
    }
    else
    {
        initializeWorkers(tp, pp, cp, parallelConfig, worldConfig, decoderGptJsonConfig);
    }
}

void Executor::Impl::validateParallelConfig(ParallelConfig const& parallelConfig, std::optional<ModelType> modelType,
    std::optional<std::filesystem::path> const& modelPath)
{
    TLLM_CHECK_WITH_INFO(parallelConfig.getCommunicationType() == CommunicationType::kMPI,
        "Only CommunicationType kMPI is supported for now.");

    auto optOrchestratorConfig = parallelConfig.getOrchestratorConfig();

    if (parallelConfig.getCommunicationMode() == CommunicationMode::kORCHESTRATOR)
    {
        TLLM_CHECK_WITH_INFO(
            optOrchestratorConfig, "OrchestratorConfig must be set when using ORCHESTRATOR communication mode.");

        TLLM_CHECK_WITH_INFO(modelPath, "OrchestratorMode only supports reading model weight from disk currently.");

        TLLM_CHECK_WITH_INFO(modelType, "OrchestratorMode requires modelType to be specified.");
    }
}

void Executor::Impl::initializeOrchestrator(SizeType32 tp, SizeType32 pp, SizeType32 cp,
    ExecutorConfig const& executorConfig, ParallelConfig parallelConfig, ModelType modelType,
    std::filesystem::path const& modelPath)
{
#if ENABLE_MULTI_DEVICE
    namespace su = tensorrt_llm::executor::serialize_utils;

    auto const& worldComm = tensorrt_llm::mpi::MpiComm::world();
    int32_t const worldSize = worldComm.getSize();

    auto orchestratorConfig = parallelConfig.getOrchestratorConfig().value();

    mIsWorker = false;
    mIsLeader = false;
    mIsPipelineLeader = false;
    mIsOrchestrator = true;

    // Verify that worldSize is 1
    if (orchestratorConfig.getSpawnProcesses())
    {
        TLLM_CHECK_WITH_INFO(worldSize == 1,
            "When using the orchestrator mode and isOrchestrator is true, expect MPI worldSize to be 1.");

        // Spawn the worker threads
        auto workerExecPath = orchestratorConfig.getWorkerExecutablePath();
        MPI_Comm intercomm = nullptr;
        MPI_Info mpiInfo = nullptr;
        MPICHECK(MPI_Info_create(&mpiInfo));
        MPICHECK(MPI_Info_set(mpiInfo, "env", "FORCE_NCCL_ALL_REDUCE_STRATEGY"));

        // Binding policy is not inherited for dynamically spawned jobs, resulting in the worker being bound
        // to a single core. Override the setting to avoid perf issue - see https://nvbugs/4574329
        MPICHECK(MPI_Info_set(mpiInfo, "bind_to", "none"));

        MPICHECK(MPI_Comm_spawn(workerExecPath.c_str(), MPI_ARGV_NULL, tp * pp * cp, mpiInfo, 0, MPI_COMM_SELF,
            &intercomm, MPI_ERRCODES_IGNORE));

        mOrchLeaderComm = std::make_shared<tensorrt_llm::mpi::MpiComm>(intercomm, true);
        // With intercomm, leader is rank 0 in the local group
        mLeaderRank = 0;
        mOrchRank = 0;

        // Copy the executor config, but set the orchestrator flag to false
        auto newOrchConfig = OrchestratorConfig(false, orchestratorConfig.getWorkerExecutablePath());
        parallelConfig.setOrchestratorConfig(newOrchConfig);
        auto execConfig = executorConfig;
        execConfig.setParallelConfig(parallelConfig);

        // Serialize and send the executorConfig, the modelType and the modelPath
        std::ostringstream oStream;
        su::serialize(modelPath.string(), oStream);
        su::serialize(modelType, oStream);
        su::serialize(execConfig, oStream);

        auto str = oStream.str();
        std::vector<char> buffer(str.begin(), str.end());
        auto bufferSize = static_cast<int64_t>(buffer.size());
        mOrchLeaderComm->bcast(&bufferSize, 1, mpi::MpiType::kINT64, MPI_ROOT);
        mOrchLeaderComm->bcast(buffer.data(), buffer.size(), mpi::MpiType::kCHAR, MPI_ROOT);

        // Wait for workers to have created their executor instance
        MPICHECK(MPI_Barrier(intercomm));
    }

    // Spawn the thread responsible for sending new requests to the leader of the model
    mOrchSendReqThread = std::thread(&Impl::orchSendReqThread, this);

    // Spawn the thread responsible for receiving new responses from the leader of the model
    mOrchRecvThread
        = std::thread([&]() { this->orchRecvThread(mpi::MpiTag::kOrchestratorId, mpi::MpiTag::kOrchestratorData); });

#endif // ENABLE_MULTI_DEVICE
}

void Executor::Impl::initializeWorkers(SizeType32 tp, SizeType32 pp, SizeType32 cp, ParallelConfig& parallelConfig,
    std::optional<runtime::WorldConfig> const& worldConfig,
    std::optional<runtime::GptJsonConfig> const& decoderGptJsonConfig)
{
    auto const& worldComm = tensorrt_llm::mpi::MpiComm::world();
    int32_t const worldSize = worldComm.getSize();

    auto const& orchestratorConfig = parallelConfig.getOrchestratorConfig();
    mIsOrchestrator = mCommMode == CommunicationMode::kORCHESTRATOR && orchestratorConfig.value().getIsOrchestrator();

    TLLM_CHECK_WITH_INFO(mCommMode != CommunicationMode::kORCHESTRATOR || orchestratorConfig.has_value(),
        "When using ORCHESTRATOR mode, orchestrator config must be set");

    if (mCommMode == CommunicationMode::kORCHESTRATOR && !orchestratorConfig.value().getSpawnProcesses())
    {
        TLLM_CHECK_WITH_INFO(parallelConfig.getParticipantIds(),
            "When not spawning processes in orchestrator mode, participant IDs must be provided");

        // Check that rank 0 is reserved for the orchestrator
        auto const participantIds = parallelConfig.getParticipantIds().value();
        for (auto const& participantId : participantIds)
        {
            TLLM_CHECK_WITH_INFO(participantId != 0, "Rank 0 is reserved for the orchestrator");
        }
    }

    // Participant ids
    std::vector<SizeType32> participantIds;
    if (!parallelConfig.getParticipantIds())
    {
        TLLM_CHECK_WITH_INFO(worldSize == tp * pp * cp,
            "With communicationMode kLEADER, MPI worldSize is expected to be equal to tp*pp*cp when "
            "participantIds are not specified");

        participantIds.resize(tp * pp * cp);
        std::iota(participantIds.begin(), participantIds.end(), 0);
    }
    else
    {
        if (mCommMode == CommunicationMode::kORCHESTRATOR && orchestratorConfig.value().getSpawnProcesses())
        {
            TLLM_THROW(
                "Participant ids should not be set when using CommunicationMode::kORCHESTRATOR with "
                "spawnProcesses=true");
        }
        participantIds = parallelConfig.getParticipantIds().value();
        TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(participantIds.size()) == tp * pp * cp,
            tensorrt_llm::common::fmtstr("When specifying participantIds, participantIds size (%lu) must be equal to "
                                         "tp*pp*cp (tp is %u, pp is %u, cp is %u)",
                participantIds.size(), tp, pp, cp));
    }

    // If deviceIds are specified, check that they match tp*pp*cp
    if (parallelConfig.getDeviceIds())
    {
        auto deviceIds = parallelConfig.getDeviceIds().value();
        auto const hasNumNodes = parallelConfig.getNumNodes().has_value();
        if (hasNumNodes || static_cast<SizeType32>(deviceIds.size()) != tp * pp * cp)
        {
            auto const numNodes = hasNumNodes ? parallelConfig.getNumNodes().value() : tensorrt_llm::mpi::getNumNodes();
            TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(deviceIds.size() * numNodes) == tp * pp * cp,
                tensorrt_llm::common::fmtstr("When specifying deviceIds, deviceIds (%lu) * numNodes (%u) must be equal "
                                             "to tp*pp*cp (tp is %u, pp is %u, cp is %u)",
                    deviceIds.size(), numNodes, tp, pp, cp));
        }
    }

    // Bool that indicates if current process is worker for this model or not
    auto participantIt = std::find(participantIds.begin(), participantIds.end(), mWorldRank);
    mIsWorker = participantIt != participantIds.end();
    // Bool that indicates if current ranks is leader for this model
    mIsLeader = (mWorldRank == participantIds.front());
    mIsPipelineLeader = (mWorldRank == participantIds[tp * (pp - 1)]);

#if ENABLE_MULTI_DEVICE
    if (mIsWorker)
    {
        // Create a session, but only assign to COMM_SESSION for ranks participating in this model
        MPI_Group worldGroup = MPI_GROUP_NULL;
        MPICHECK(MPI_Comm_group(MPI_COMM_WORLD, &worldGroup)); // NOLINT
        MPI_Group sessionGroup = MPI_GROUP_NULL;
        if (pp > 1)
        {
            // reverse participantIds to move leader to last pp rank. retain order in each tp group
            std::reverse(participantIds.begin(), participantIds.end());
            if (tp > 1)
            {
                for (SizeType32 ppRank = 0; ppRank < pp; ppRank++)
                {
                    std::reverse(participantIds.begin() + ppRank * tp, participantIds.begin() + (ppRank + 1) * tp);
                }
            }
        }
        MPICHECK(MPI_Group_incl(worldGroup, participantIds.size(), participantIds.data(), &sessionGroup)); // NOLINT
        MPI_Comm sessionComm = MPI_COMM_NULL;
        MPICHECK(
            MPI_Comm_create_group(MPI_COMM_WORLD, sessionGroup, 1000 + participantIds.front(), &sessionComm)); // NOLINT

        tensorrt_llm::mpi::MpiComm::setSession(tensorrt_llm::mpi::MpiComm(sessionComm, false));
    }

    if (mIsLeader && mCommMode == CommunicationMode::kORCHESTRATOR)
    {
        auto optOrchestratorConfig = parallelConfig.getOrchestratorConfig();
        if (orchestratorConfig.has_value() && orchestratorConfig.value().getSpawnProcesses())
        {
            mOrchLeaderComm = optOrchestratorConfig.value().getOrchLeaderComm();
        }
        else
        {
            // mOrchLeaderComm has already been created
        }
        TLLM_CHECK(mOrchLeaderComm.get() != nullptr);

        TLLM_CHECK(worldConfig.has_value() || decoderGptJsonConfig.has_value());
        if (worldConfig.has_value())
        {
            mDeviceId = worldConfig->getDevice();
        }
        else
        {
            auto gpusPerNode = decoderGptJsonConfig->getGpusPerNode();
            auto worldConfig = runtime::WorldConfig::mpi(gpusPerNode, tp, pp, cp, parallelConfig.getDeviceIds());
            mDeviceId = worldConfig.getDevice();
        }
        // Spawn the thread responsible for receiving new requests from the orchestrator
        mLeaderRecvReqThread = std::thread(&Impl::leaderRecvReqThread, this);

        // Spawn the thread responsible for sending new responses to the orchestrator
        mLeaderSendThread = std::thread([&]()
            { this->leaderSendThread(mSendQueue, mpi::MpiTag::kOrchestratorId, mpi::MpiTag::kOrchestratorData); });
    }
#endif // ENABLE_MULTI_DEVICE
}

void Executor::Impl::initializeLogitsPostProcessorBatched(LogitsPostProcessorConfig const& logitsProcConfig)
{
    if (logitsProcConfig.getProcessorBatched().has_value())
    {
        mLogitsPostProcessorBatched
            = [cb = logitsProcConfig.getProcessorBatched().value()](
                  std::vector<batch_manager::LlmRequest::RequestIdType> const& reqIdsVec,
                  std::vector<batch_manager::LlmRequest::TensorPtr>& logitsVec,
                  std::vector<std::reference_wrapper<batch_manager::LlmRequest::BeamTokens const>> const& beamTokensVec,
                  CudaStreamPtr const& cudaStreamPtr,
                  std::vector<std::optional<batch_manager::LlmRequest::RequestIdType>> const& clientIdsVec)
        {
            std::vector<Tensor> cbLogitsVec;
            cbLogitsVec.reserve(logitsVec.size());
            for (auto& logits : logitsVec)
            {
                cbLogitsVec.emplace_back(executor::detail::ofITensor(logits));
            }

            cb(reqIdsVec, cbLogitsVec, beamTokensVec, cudaStreamPtr, clientIdsVec);
        };

        mModel->setLogitsPostProcessorBatched(mLogitsPostProcessorBatched);
    }
}

IdType Executor::Impl::enqueueRequest(Request const& request)
{
    return enqueueRequests({&request, 1}).at(0);
}

std::vector<IdType> Executor::Impl::enqueueRequests(std::vector<Request> const& requests)
{
    return enqueueRequests({requests.data(), requests.size()});
}

std::vector<IdType> Executor::Impl::enqueueRequests(common::ArrayView<Request const> const& requests)
{
    TLLM_CHECK_WITH_INFO(!mShutdownCalled, "Shutdown called, cannot enqueue requests");
    checkParallelApiUsage(__func__);

    TLLM_LOG_DEBUG("Enqueuing %lu requests", requests.size());
    std::vector<RequestWithId> requestWithIds;
    requestWithIds.reserve(requests.size());

    // First check valid of request in enqueue thread, so Exceptions can be thrown to user.
    for (auto const& req : requests)
    {
        auto logitsPostProcessorName = req.getLogitsPostProcessorName();
        if (logitsPostProcessorName && logitsPostProcessorName.value() != Request::kBatchedPostProcessorName)
        {
            getLogitsPostProcessor(*logitsPostProcessorName);
        }
    }

    std::vector<IdType> ids;
    {
        auto now = std::chrono::steady_clock::now();
        for (auto const& req : requests)
        {
            ids.emplace_back(generateReqId());
            TLLM_LOG_DEBUG("Enqueue new request with id %d", ids.back());

            std::vector<IdType> childReqIds;
            auto numChildRequests = getNumChildRequests(req);
            if (numChildRequests > 0)
            {
                childReqIds.reserve(numChildRequests);
                for (int childId = 0; childId < numChildRequests; childId++)
                {
                    childReqIds.emplace_back(generateReqId());
                    TLLM_LOG_DEBUG("Add new child request with id %d", childReqIds.back());
                }
            }
            requestWithIds.emplace_back(RequestWithId{req, ids.back(), std::move(childReqIds), now});
        }
    }

    if (mCommMode == CommunicationMode::kLEADER)
    {
        {
            std::scoped_lock<std::mutex> const lck(mQueuedReqMtx);
            if (mMaxQueueSize)
            {
                auto const maxQueueSize = mMaxQueueSize.value();

                auto totalRequestSize = 0;
                for (auto&& reqWithId : requestWithIds)
                {
                    totalRequestSize += (getNumChildRequests(reqWithId.req) + 1);
                }

                if (maxQueueSize > 0 && mQueuedRequests.size() + totalRequestSize > static_cast<size_t>(maxQueueSize))
                {
                    TLLM_THROW("Maximum queue size of %d has been reached, please try again later", maxQueueSize);
                }
            }

            for (auto&& req : requestWithIds)
            {
                insertRequestInOrder(mQueuedRequests, std::move(req));
            }
        }
        mQueuedReqCv.notify_one();
    }
    else if (mCommMode == CommunicationMode::kORCHESTRATOR)
    {
        MpiMessage message(MpiId::PENDING_REQUEST);
        message.data = PendingRequestData{std::move(requestWithIds)};
        mSendQueue.push(std::move(message));
    }
    return ids;
}

std::vector<Response> Executor::Impl::awaitResponses(std::optional<std::chrono::milliseconds> const& timeout)
{
    TLLM_CHECK_WITH_INFO(!mShutdownCalled, "Shutdown called");
    checkParallelApiUsage(__func__);
    std::unique_lock<std::mutex> lck(mResponsesMtx);
    auto pred = [this]() -> bool { return !mResponses.empty() || mShutdown; };
    auto storeResponses = [this]()
    {
        std::vector<Response> responses;
        for (auto it = mResponses.begin(); it != mResponses.end();)
        {
            responses.insert(responses.end(), it->second.begin(), it->second.end());
            addTerminatedReqId(it->second, it->first);
            it = mResponses.erase(it);
        }
        return responses;
    };

    std::vector<Response> responses;
    if (timeout)
    {
        if (mResponsesCv.wait_for(lck, timeout.value(), pred))
        {
            responses = storeResponses();
        }
    }
    else
    {
        mResponsesCv.wait(lck, pred);
        responses = storeResponses();
    }
    return responses;
}

std::vector<Response> Executor::Impl::awaitResponses(
    IdType const& reqId, std::optional<std::chrono::milliseconds> const& timeout)
{
    TLLM_CHECK_WITH_INFO(!mShutdownCalled, "Shutdown called");
    checkParallelApiUsage(__func__);
    std::unique_lock<std::mutex> lck(mResponsesMtx);
    auto pred = [this, reqId]() -> bool
    { return (mResponses.find(reqId) != mResponses.end() && !mResponses.at(reqId).empty()) || mShutdown; };
    auto storeIdResponse = [this, reqId]()
    {
        std::vector<Response> responses;
        responses.swap(mResponses.at(reqId));
        mResponses.erase(reqId);
        addTerminatedReqId(responses, reqId);
        return responses;
    };

    // We don't process a terminated request again. Terminated request is defined as a response
    // with isFinal = true for a given requestId.
    if (mTerminatedReqIds.contains(reqId))
    {
        if (mResponses.find(reqId) != mResponses.end())
        {
            TLLM_THROW("ReqId should already be removed from responses!");
        }
        std::string const err = "ReqId " + std::to_string(reqId) + " has already been processed and was terminated.";
        TLLM_LOG_ERROR("%s", err.c_str());

        return {Response(reqId, err)};
    }

    std::vector<Response> responses;
    if (timeout)
    {
        if (mResponsesCv.wait_for(lck, timeout.value(), pred))
        {
            responses = storeIdResponse();
        }
    }
    else
    {
        mResponsesCv.wait(lck, pred);
        responses = storeIdResponse();
    }
    return responses;
}

std::vector<std::vector<Response>> Executor::Impl::awaitResponses(
    std::vector<IdType> const& requestIds, std::optional<std::chrono::milliseconds> const& timeout)
{
    TLLM_CHECK_WITH_INFO(!mShutdownCalled, "Shutdown called");
    checkParallelApiUsage(__func__);
    std::vector<std::vector<Response>> responses;
    responses.reserve(requestIds.size());
    if (timeout)
    {
        auto const start_time = std::chrono::high_resolution_clock::now();
        for (auto const requestId : requestIds)
        {
            auto const elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time);
            responses.emplace_back(awaitResponses(
                requestId, timeout.value() > elapsed_ms ? timeout.value() - elapsed_ms : std::chrono::milliseconds{0}));
        }
    }
    else
    {
        for (auto const requestId : requestIds)
        {
            responses.emplace_back(awaitResponses(requestId));
        }
    }
    return responses;
}

SizeType32 Executor::Impl::getNumResponsesReady(std::optional<IdType> const& optId) const
{
    TLLM_CHECK_WITH_INFO(!mShutdownCalled, "Shutdown called");
    checkParallelApiUsage(__func__);
    std::scoped_lock<std::mutex> lck(mResponsesMtx);
    SizeType32 numResponsesReady = 0;
    if (optId)
    {
        auto const reqId = optId.value();
        auto const respIt = mResponses.find(reqId);
        if (respIt != mResponses.end())
        {
            numResponsesReady = static_cast<SizeType32>(respIt->second.size());
        }
    }
    else
    {
        for (auto const& [id, responses] : mResponses)
        {
            numResponsesReady += static_cast<SizeType32>(responses.size());
        }
    }
    return numResponsesReady;
}

void Executor::Impl::shutdown()
{
    // Cannot call shutdown multiple times
    if (mShutdownCalled)
    {
        return;
    }
    mShutdownCalled = true;

    if (!mShutdown)
    {
        if (mCommMode == CommunicationMode::kLEADER && mIsLeader)
        {
            //  Enqueue a request to indicate to other ranks to terminate
            enqueueTerminateRequest();
        }
        else if (mCommMode == CommunicationMode::kORCHESTRATOR)
        {
            if (mIsOrchestrator)
            {
                // Send to the leader the termination signal
                mShutdown = true;
                mResponsesCv.notify_all();

                mSendQueue.push(MpiMessage(MpiId::TERMINATION));

                // Wait for sender thread to exit
                if (mOrchSendReqThread.joinable())
                {
                    mOrchSendReqThread.join();
                }
                // Wait for recv response thread to exit
                if (mOrchRecvThread.joinable())
                {
                    mOrchRecvThread.join();
                }
            }
            else if (mIsLeader)
            {
                // Wait for sender thread to exit
                if (mLeaderRecvReqThread.joinable())
                {
                    mLeaderRecvReqThread.join();
                }
                // Wait for send response thread to exit
                if (mLeaderSendThread.joinable())
                {
                    mLeaderSendThread.join();
                }
            }
        }
    }

    // Wait for execution thread to terminate
    if (mExecutionThread.joinable())
    {
        mExecutionThread.join();
    }

    // If we overwrote COMM_SESSION with split, free it now. Otherwise, since
    // COMM_SESSION is a global static object, it will be destroyed in an
    // undefined order and can cause crashes on program exit.
    if (mIsWorker)
    {
        tensorrt_llm::mpi::MpiComm::setSession(tensorrt_llm::mpi::MpiComm(MPI_COMM_WORLD, false));
    }
}

void Executor::Impl::cancelRequest(IdType requestId)
{
    TLLM_CHECK_WITH_INFO(!mShutdownCalled, "Shutdown called");
    checkParallelApiUsage(__func__);

    // Check if the request is terminated already. If so, return
    {
        std::scoped_lock<std::mutex> lckResp(mResponsesMtx);
        if (mTerminatedReqIds.contains(requestId))
        {
            TLLM_LOG_INFO("Ignoring already terminated request %lu", requestId);
            return;
        }
    }

    if (mCommMode == CommunicationMode::kLEADER)
    {
        std::scoped_lock<std::mutex> lck(mCancelReqMtx);
        auto& selCancelledReqIds = mUsePipelineParallel ? mPipelineCancelledReqIds : mCancelledReqIds;
        selCancelledReqIds.insert(requestId);
    }
    else if (mCommMode == CommunicationMode::kORCHESTRATOR)
    {
        MpiMessage message(MpiId::CANCEL_REQUEST);
        std::vector<IdType> cancelledReqIds{requestId};
        message.data = RequestIdsData{std::move(cancelledReqIds)};
        mSendQueue.push(std::move(message));
    }
}

std::deque<IterationStats> Executor::Impl::getLatestIterationStats()
{
    TLLM_CHECK_WITH_INFO(!mShutdownCalled, "Shutdown called");
    checkParallelApiUsage(__func__);
    std::scoped_lock<std::mutex> lck(mIterStatsMtx);
    return std::exchange(mIterationStats, {});
}

std::deque<RequestStatsPerIteration> Executor::Impl::getLatestRequestStats()
{
    TLLM_CHECK_WITH_INFO(!mShutdownCalled, "Shutdown called");
    checkParallelApiUsage(__func__);

    std::scoped_lock<std::mutex> lck(mRequestStatsMtx);
    return std::exchange(mRequestStats, {});
}

std::deque<DebugTensorsPerIteration> Executor::Impl::getLatestDebugTensors()
{
    TLLM_CHECK_WITH_INFO(!mShutdownCalled, "Shutdown called");
    if (mCommMode == CommunicationMode::kORCHESTRATOR)
    {
        TLLM_LOG_WARNING("getLatestDebugTensors is not supported in ORCHESTRATOR mode yet");
        return {};
    }
    if (mEncoderModel)
    {
        TLLM_LOG_WARNING("getLatestDebugTensors is not supported for encoder model yet");
    }
    std::scoped_lock<std::mutex> lck(mDebugTensorsMtx);
    return std::exchange(mDebugTensors, {});
}

bool Executor::Impl::canEnqueueRequests() const
{
    return !mShutdownCalled
        && ((mCommMode == CommunicationMode::kLEADER && mIsLeader)
            || (mCommMode == CommunicationMode::kORCHESTRATOR && mIsOrchestrator));
}

bool Executor::Impl::isParticipant() const
{
    return mIsWorker;
}

std::optional<std::shared_ptr<KVCacheEventManager>> Executor::Impl::getKVCacheEventManager() const
{
    if (!mModel)
    {
        return std::nullopt;
    }
    auto cacheEventManager = mModel->getKVCacheManager();
    return cacheEventManager ? std::optional(std::make_shared<KVCacheEventManager>(cacheEventManager)) : std::nullopt;
}

void Executor::Impl::requestWithIdLeaderThread()
{
    TLLM_CUDA_CHECK(cudaSetDevice(mModel->getWorldConfig().getDevice()));
    auto constexpr peer = 0;
    while (true)
    {
        int64_t numActiveRequests;
        mCommPipelineParallel->recv(
            &numActiveRequests, 1, mpi::MpiType::kINT64, peer, mpi::MpiTag::kExecutorNumActiveRequests);
        if (numActiveRequests < 0)
        {
            break;
        }

        bool lowestPriorityActiveHasValue;
        std::optional<PriorityType> lowestPriorityActive;
        mCommPipelineParallel->recv(&lowestPriorityActiveHasValue, 1, mpi::MpiType::kBOOL, peer,
            mpi::MpiTag::kExecutorLowestPriorityActiveHasValue);
        if (lowestPriorityActiveHasValue)
        {
            PriorityType lowestPriorityActiveValue;
            mCommPipelineParallel->recv(
                &lowestPriorityActiveValue, 1, mpi::MpiType::kFLOAT, peer, mpi::MpiTag::kExecutorLowestPriorityActive);
            lowestPriorityActive = lowestPriorityActiveValue;
        }

        auto reqWithIds = getLeaderNewReqWithIds(numActiveRequests, lowestPriorityActive);
        setupDynamicLogitsPostProcessors(reqWithIds);
        auto requestWithIdAsyncSndHdl
            = std::make_unique<RequestWithIdAsyncSend>(mCommPipelineParallel, reqWithIds, peer);
        requestWithIdAsyncSndHdl.reset(nullptr);
    }
}

void Executor::Impl::cancelledRequestsLeaderThread()
{
    TLLM_CUDA_CHECK(cudaSetDevice(mModel->getWorldConfig().getDevice()));
    auto constexpr peer = 0;
    while (true)
    {
        bool shouldExit;
        mCommPipelineParallel->recv(&shouldExit, 1, mpi::MpiType::kBOOL, peer, mpi::MpiTag::kExecutorShouldExit);
        if (shouldExit)
        {
            break;
        }

        std::unique_ptr<CancelledRequestsAsyncSend> cancelledRequestsAsyncSndHdl;
        {
            std::scoped_lock<std::mutex> lck(mCancelReqMtx);
            cancelledRequestsAsyncSndHdl
                = std::make_unique<CancelledRequestsAsyncSend>(mCommPipelineParallel, mPipelineCancelledReqIds, peer);
            mPipelineCancelledReqIds.clear();
        }
        cancelledRequestsAsyncSndHdl.reset(nullptr);
    }
}

std::vector<RequestWithId> Executor::Impl::getLeaderNewReqWithIds(
    SizeType32 numActiveRequests, std::optional<PriorityType> lowestPriorityActive)
{
    std::unique_lock<std::mutex> lck(mQueuedReqMtx);
    mQueuedReqCv.wait(lck, [&]() { return (!mQueuedRequests.empty() || numActiveRequests > 0 || mShutdown); });

    std::vector<RequestWithId> reqWithIds;

    if (mQueuedRequests.empty() || mShutdown)
    {
        return reqWithIds;
    }

    if (mQueuedRequests.front().id == mTerminateReqId)
    {
        reqWithIds.emplace_back(std::move(mQueuedRequests.front()));
        mQueuedRequests.pop_front();
        return reqWithIds;
    }

    auto const& firstRequest = mQueuedRequests.front();
    auto const firstBeamWidth = firstRequest.req.getSamplingConfig().getBeamWidth();
    auto const operatingBeamWidth = numActiveRequests > 0 ? mModel->getOperatingBeamWidth() : firstBeamWidth;

    auto const tryInsertQueuedRequestIntoReqWithIds = [this, &reqWithIds, operatingBeamWidth]() -> bool
    {
        auto& nextRequest = mQueuedRequests.front();
        auto const beamWidth = nextRequest.req.getSamplingConfig().getBeamWidth();
        if (beamWidth != operatingBeamWidth)
        {
            TLLM_LOG_INFO(
                "Can't dequeue request with ID %ld because beam width %d differs from operating beam width %d.",
                nextRequest.id, beamWidth, operatingBeamWidth);
            return false;
        }

        TLLM_LOG_DEBUG("Dequeue request with ID %ld", nextRequest.id);
        reqWithIds.emplace_back(std::move(nextRequest));
        mQueuedRequests.pop_front();
        return true;
    };

    auto const maxNewRequests = static_cast<size_t>(std::max(mMaxNumActiveRequests - numActiveRequests, 0));
    for (size_t req = 0; !mQueuedRequests.empty() && req < maxNewRequests;)
    {
        req += (getNumChildRequests(mQueuedRequests.front().req) + 1);
        if (req > maxNewRequests)
        {
            break;
        }
        if (!tryInsertQueuedRequestIntoReqWithIds())
        {
            break;
        }
    }

    if (lowestPriorityActive)
    {
        while (!mQueuedRequests.empty() && mQueuedRequests.front().req.getPriority() > (*lowestPriorityActive))
        {
            if (!tryInsertQueuedRequestIntoReqWithIds())
            {
                break;
            }
        }
    }
    return reqWithIds;
}

std::vector<RequestWithId> Executor::Impl::getNewReqWithIds(
    SizeType32 numActiveRequests, std::optional<PriorityType> lowestPriorityActive)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const& worldConfig = mModel->getWorldConfig();

    if (worldConfig.isPipelineParallel())
    {
        mRequestWithIdWaitThread->waitStop();
    }

    TLLM_CUDA_CHECK(cudaSetDevice(mModel->getWorldConfig().getDevice()));
    std::vector<RequestWithId> reqWithIds;
    if (mIsPipelineLeader)
    {
        if (!worldConfig.isPipelineParallel())
        {
            reqWithIds = getLeaderNewReqWithIds(numActiveRequests, lowestPriorityActive);
            setupDynamicLogitsPostProcessors(reqWithIds);
        }
        else
        {
            auto const peer = worldConfig.getPipelineParallelism() - 1;
            auto numActiveRequestsValue = static_cast<int64_t>(numActiveRequests);
            auto request1 = mCommPipelineParallel->sendAsync(
                &numActiveRequestsValue, 1, mpi::MpiType::kINT64, peer, mpi::MpiTag::kExecutorNumActiveRequests);
            bool lowestPriorityActiveHasValue = lowestPriorityActive.has_value();
            auto request2 = mCommPipelineParallel->sendAsync(&lowestPriorityActiveHasValue, 1, mpi::MpiType::kBOOL,
                peer, mpi::MpiTag::kExecutorLowestPriorityActiveHasValue);
            auto request3 = lowestPriorityActiveHasValue
                ? mCommPipelineParallel->sendAsync(&lowestPriorityActive.value(), 1, mpi::MpiType::kFLOAT, peer,
                    mpi::MpiTag::kExecutorLowestPriorityActive)
                : nullptr;
            request1->wait();
            request2->wait();
            if (request3)
            {
                request3->wait();
            }
            reqWithIds = RequestWithIdAsyncSend::requestWithIdRecv(mCommPipelineParallel, peer);
        }
        if (worldConfig.isTensorParallel() || worldConfig.isContextParallel())
        {
            auto packed = RequestWithId::serializeReqWithIds(reqWithIds);
            if (worldConfig.isTensorParallel())
            {
                mCommTensorParallel->bcast(packed, 0);
            }
            if (worldConfig.isContextParallel())
            {
                mCommContextParallel->bcast(packed, 0);
            }
        }
    }
    else
    {
        if (worldConfig.isFirstPipelineParallelRank())
        {
            std::vector<char> buffer;
            mCommTensorParallel->bcast(buffer, 0);
            mCommContextParallel->bcast(buffer, 0);
            reqWithIds = RequestWithId::deserializeReqWithIds(buffer);
        }
        else
        {
            auto const peer = worldConfig.getPipelineParallelRank() - 1;
            reqWithIds = RequestWithIdAsyncSend::requestWithIdRecv(mCommPipelineParallel, peer);
        }
    }
    if (!worldConfig.isLastPipelineParallelRank())
    {
        auto const peer = worldConfig.getPipelineParallelRank() + 1;
        mRequestWithIdAsyncSndHdl = std::make_unique<RequestWithIdAsyncSend>(mCommPipelineParallel, reqWithIds, peer);
        mRequestWithIdWaitThread->notifyStart();
    }
    TLLM_CUDA_CHECK(cudaSetDevice(mModel->getWorldConfig().getDevice()));

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return reqWithIds;
}

std::tuple<Executor::Impl::RequestList, double> Executor::Impl::fetchNewRequests(
    SizeType32 numActiveRequests, std::optional<PriorityType> lowestPriorityActive)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(fetchNewRequests);

    // If grab requests from queue, do exchange between ranks
    auto reqWithIds = getNewReqWithIds(numActiveRequests, lowestPriorityActive);
    RequestList newRequests;
    double newActiveRequestsQueueLatencyMS{0.};
    for (auto& reqWithId : reqWithIds)
    {
        if (reqWithId.id == mTerminateReqId)
        {
            mShutdown = true;
            mResponsesCv.notify_all();
            return {};
        }

        try
        {
            std::optional<LlmRequestLogitsPostProcessor> llmRequestLogitsPostProcessor;
            bool applyLogitsPostProcessorBatched{false};
            if (mModel->getWorldConfig().isLastPipelineParallelRank())
            {
                auto logitsPostProcessorName = reqWithId.req.getLogitsPostProcessorName();
                if (logitsPostProcessorName)
                {
                    if (logitsPostProcessorName.value() == Request::kBatchedPostProcessorName)
                    {
                        TLLM_CHECK_WITH_INFO(
                            mLogitsPostProcessorBatched, "Batched logits post processor is not defined.");
                        applyLogitsPostProcessorBatched = true;
                    }
                    else
                    {
                        if (logitsPostProcessorName->compare(0,
                                std::char_traits<char>::length(Request::kDynamicPostProcessorNamePrefix),
                                Request::kDynamicPostProcessorNamePrefix)
                            == 0)
                        {
                            TLLM_CHECK_WITH_INFO(!mModel->getReplicateLogitsPostProcessor()
                                    || mModel->getWorldConfig().getTensorParallelism() == 1,
                                "Dynamic logits postprocessor must be used with replicate=false or no tensor "
                                "parallelism.");
                        }
                        if (mModel->getWorldConfig().isFirstTensorParallelRank()
                            || mModel->getReplicateLogitsPostProcessor())
                        {
                            llmRequestLogitsPostProcessor = getLogitsPostProcessor(logitsPostProcessorName.value());
                        }
                        else
                        {
                            llmRequestLogitsPostProcessor
                                = [](IdType reqId, RtTensorPtr& logits, BeamTokens const& beamTokens,
                                      CudaStreamPtr const& cudaStreamPtr, std::optional<IdType> clientId) {};
                        }
                    }
                }
            }
            auto newLlmReq = std::make_shared<batch_manager::LlmRequest>(
                reqWithId.id, reqWithId.req, llmRequestLogitsPostProcessor, applyLogitsPostProcessorBatched);

            auto numReturnSequences = newLlmReq->getNumSubRequests();
            if (numReturnSequences > 1)
            {
                TLLM_CHECK(reqWithId.childReqIds.size() == static_cast<size_t>(numReturnSequences - 1));
                mChildReqIdsMap[reqWithId.id] = reqWithId.childReqIds;
            }

            for (auto seqIdx = 0; seqIdx < numReturnSequences; seqIdx++)
            {
                auto newReq
                    = seqIdx == 0 ? newLlmReq : newLlmReq->createChildRequest(reqWithId.childReqIds.at(seqIdx - 1));

                // If static batching and streaming, disable streaming and exclude input
                if (mBatchingType == BatchingType::kSTATIC && newReq->isStreaming())
                {
                    newReq->setStreaming(false);
                    newReq->setExcludeInputFromOutput(true);
                }

                // Validate the request parameters
                newReq->validate(mModel->getMaxInputLen(), mModel->getMaxSequenceLen(), mModel->getMaxDraftLen(),
                    mModel->getVocabSizePadded(),
                    mEncoderModel ? std::optional<SizeType32>(mEncoderModel->getMaxInputLen()) : std::nullopt,
                    mEnableBlockReuse);

                TLLM_CHECK_WITH_INFO(!mEncoderModel || !mIsSchedulerMaxUtilization,
                    "Encoder or Encoder-Decoder model don't support max utilization scheduler yet. Only max requests "
                    "or guaranteed no evict.");

                // When streaming is enabled and scheduling policy permits evict/restart, need to guard against the case
                // where the sequence is truncated on eviction (to respect maxInputLen limits), resulting in loss of
                // some tokens that have been streamed out. In this case, resuming generation may result in different
                // completion for locations whose tokens have already been returned. There is no way to protect against
                // this, so disallowing.
                if (newReq->isStreaming() && !mIsSchedulerGuaranteedNoEvict && !mIsChunkedContext)
                {
                    auto const maxReqSeqLen = newReq->mPromptLen + newReq->mMaxNewTokens;
                    auto const maxRestartLen = maxReqSeqLen - 1;
                    TLLM_CHECK_WITH_INFO(maxRestartLen <= mModel->getMaxInputLen(),
                        "Request sequence length is potentially greater than max input length. This cannot be run "
                        "unless streaming is disabled, context chunking is enabled or the GUARANTEED_NO_EVICT "
                        "scheduling policy is used");
                }

                // Create the encoder output tensor
                if (mEncoderModel)
                {
                    TLLM_CHECK_WITH_INFO(mModel || (!mModel && newReq->getReturnEncoderOutput()),
                        "Encoder-Decoder models allow optionally returning encoder output. But if it is Encoder-only "
                        "models, please make sure returnEncoderOutput is always true.");

                    // gpu buffers for passing to the next phase
                    newReq->allocEncoderOutput(mEncoderModel->getBufferManager(), mEncoderModel->getLogitDataType());
                    newReq->allocEncoderHiddenStates(
                        mEncoderModel->getBufferManager(), mEncoderModel->getLogitDataType());
                    // pinned buffers for returning results to host
                    if (newReq->getReturnEncoderOutput())
                    {
                        newReq->allocEncoderOutputHost(
                            mEncoderModel->getHiddenSize() * mEncoderModel->getWorldConfig().getTensorParallelism(),
                            mEncoderModel->getLogitDataType());
                    }
                }

                if (!mEncoderModel && newReq->getEncoderInputFeatures())
                {
                    TLLM_LOG_INFO("Allocating buffers for encoder output");
                    // gpu buffers for passing to the next phase
                    newReq->allocEncoderOutput(mModel->getBufferManager(), mModel->getLogitDataType());
                    newReq->allocEncoderHiddenStates(mModel->getBufferManager(), mModel->getLogitDataType());
                }

                // Create the context logits tensor
                if (newReq->getReturnContextLogits())
                {
                    TLLM_CHECK_WITH_INFO(mModel->getModelConfig().computeContextLogits(),
                        "Return context logit need to build engine with gather_context_logits");
                    newReq->allocContextLogitsHost(mModel->getVocabSizePadded(), mModel->getLogitDataType());
                }

                // Create the generation logits tensor
                if (newReq->getReturnGenerationLogits())
                {
                    TLLM_CHECK_WITH_INFO(mModel->getGatherGenerationLogits(),
                        "To return generation logits, gather_generation_logits must be enabled in ExecutorConfig");

                    if (mModel->getModelConfig().getSpeculativeDecodingMode().isDraftTokensExternal()
                        && newReq->hasDraftTokens())
                    {
                        newReq->allocTargetModelAcceptedTokenLogitsHost(
                            mModel->getVocabSizePadded(), mModel->getLogitDataType());
                    }
                    else
                    {
                        newReq->allocGenerationLogitsHost(mModel->getVocabSizePadded(), mModel->getLogitDataType());
                    }
                }

                if (mModel->getWorldConfig().isLastPipelineParallelRank() && newReq->getGuidedDecodingParams())
                {
                    TLLM_CHECK_WITH_INFO(mModel->hasGuidedDecoder(),
                        "Request is specified with GuidedDecodingParams, but GuidedDecoder is not setup. Please "
                        "provide a valid GuidedDecodingConfig to setup GuidedDecoder.");
                }

                if (mModel->getWorldConfig().isLastPipelineParallelRank() && newReq->hasAdditionalOutputs())
                {
                    newReq->allocAdditionalOutputs([this](std::string const& name)
                        { return mModel->getTensorDataType(name); },
                        [this](std::string const& name) { return mModel->getTensorShape(name); });
                }

                mModel->updatePeftCache(newReq);

                newRequests.emplace_back(std::move(newReq));
            }

            auto queuedEnd = std::chrono::steady_clock::now();
            auto reqQueueLatencyMS
                = std::chrono::duration<double, std::milli>(queuedEnd - reqWithId.queuedStart).count();
            newActiveRequestsQueueLatencyMS += reqQueueLatencyMS;
        }
        catch (runtime::LoraExpectedException const& e)
        {
            if (mIsLeader)
            {
                // In case  of an expected LoRA exception (e.g. cache full, cache miss), log a warning and enqueue
                // response
                TLLM_LOG_WARNING("%s", e.what());
                enqueueNewResponses({{reqWithId.id, e.what(), reqWithId.req.getClientId()}});
            }
        }
        catch (std::exception const& e)
        {
            if (mIsLeader)
            {
                // In case of error, create a response with error for this request
                auto err = std::string("Encountered an error when fetching new request: ") + e.what();
                TLLM_LOG_ERROR("%s", err.c_str());
                enqueueNewResponses({{reqWithId.id, err, reqWithId.req.getClientId()}});
            }
        }
    }
    TLLM_LOG_DEBUG("[RANK %d] num new requests fetched from queue: %d", COMM_SESSION.getRank(), newRequests.size());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return {newRequests, newActiveRequestsQueueLatencyMS};
}

void Executor::Impl::terminateActiveRequests(RequestList& activeRequests, std::string const& err)
{
    TLLM_LOG_ERROR("%s", err.c_str());

    // Create a response for all requests and add to queue
    for (auto it = activeRequests.cbegin(); it != activeRequests.cend();)
    {
        auto llmReq = (*it);

        llmReq->setState(batch_manager::LlmRequestState::kGENERATION_COMPLETE);
        mModel->terminateRequest(llmReq);

        if (mIsLeader)
        {
            enqueueNewResponses({{llmReq->mRequestId, err, llmReq->mClientId}});
        }

        // Remove from the requestList
        it = activeRequests.erase(it);
    }
}

void Executor::Impl::forwardSync(RequestList& activeRequests)
{
    TLLM_LOG_TRACE("[RANK %d] %s start", COMM_SESSION.getRank(), __PRETTY_FUNCTION__);
    try
    {
        if (mEncoderModel)
        {
            mEncoderModel->forwardSync();
        }
        mModel->forwardSync();
    }
    catch (std::exception const& e)
    {
        std::string const err = std::string("Encountered an error in forwardSync function: ") + e.what();
        terminateActiveRequests(activeRequests, err);
    }
    TLLM_LOG_TRACE("[RANK %d] %s stop", COMM_SESSION.getRank(), __PRETTY_FUNCTION__);
}

// The function is used to change the state of a request to context_init from encoder_init for enc-dec model whose
// encoder is skipped. The encoder output is populated accordingly with input features given through model executor of
// decoder.
void Executor::Impl::prepRequestsForEncoderSkip(RequestList& activeRequests)
{

    for (auto& req : activeRequests)
    {

        if (req->isEncoderInitState() && req->getEncoderInputFeatures())
        {
            TLLM_LOG_INFO("Changing state of request and setting encoder output to skip encoder run");
            req->setState(batch_manager::LlmRequestState::kCONTEXT_INIT);
            req->setEncoderOutput(req->getEncoderInputFeatures());
        }
    }
}

void Executor::Impl::finishTimedOutRequests(RequestList const& activeRequests)
{
    if (mIsLeader)
    {
        for (auto const& request : activeRequests)
        {
            if (request->isTimedOut() && !request->isFinished())
            {
                // workaround to cancelRequest since it throws an error if
                // mCommMode == CommunicationMode::kORCHESTRATOR && !mIsOrchestrator
                {
                    std::scoped_lock<std::mutex> lck(mCancelReqMtx);
                    auto& selCancelledReqIds = mUsePipelineParallel ? mPipelineCancelledReqIds : mCancelledReqIds;
                    selCancelledReqIds.insert(request->mRequestId);
                }
            }
        }
    }
}

void Executor::Impl::forwardAsync(RequestList& activeRequests)
{
    try
    {
        TLLM_LOG_DEBUG("num active requests in scope: %d", activeRequests.size());

        if (mDynamicBatchTuner)
        {
            auto const averageInputLength = static_cast<SizeType32>(mDynamicBatchTuner->getAverageInputLength());
            auto const averageOutputLength = static_cast<SizeType32>(mDynamicBatchTuner->getAverageOutputLength());
            auto const maxCapacityBatchSize = mModel->getMaxCapacityBatchSize(averageInputLength, averageOutputLength);

            if (mDynamicBatchTuner->isBatchSizeTuningEnabled())
            {
                auto runtimeBatchSize = mDynamicBatchTuner->getRuntimeBatchSize(maxCapacityBatchSize);
                mModel->setRuntimeBatchSize(runtimeBatchSize);
            }

            if (mDynamicBatchTuner->isMaxNumTokensTuningEnabled())
            {
                auto runtimeBatchSize = mModel->getRuntimeBatchSize();
                auto runtimeMaxNumTokens = mDynamicBatchTuner->getRuntimeMaxNumTokens(runtimeBatchSize);
                mModel->setRuntimeMaxNumTokens(runtimeMaxNumTokens);
            }
        }

        if (mEncoderModel)
        {
            mEncoderModel->forwardAsync(activeRequests);
            auto const& encoderStream = *(mEncoderModel->getRuntimeStreamPtr());
            auto const& decoderStream = *(mModel->getRuntimeStreamPtr());
            runtime::CudaEvent encoderFinished;
            encoderStream.record(encoderFinished);
            decoderStream.wait(encoderFinished);
        }

        if (!mEncoderModel)
        {
            prepRequestsForEncoderSkip(activeRequests);
        }

        mModel->forwardAsync(activeRequests);
    }
    catch (std::exception const& e)
    {
        std::string err = std::string("Encountered an error in forwardAsync function: ") + e.what();
        terminateActiveRequests(activeRequests, err);
    }
}

IterationStats Executor::Impl::getCurrentIterationStats(RequestList const& activeRequests, double iterLatencyMS,
    SizeType32 numNewActiveRequests, double newActiveRequestsQueueLatencyMS, SizeType32 numCompletedRequests)
{
    IterationStats stats;
    // Timestamp
    stats.timestamp = tensorrt_llm::common::getCurrentTimestamp();
    stats.numNewActiveRequests = numNewActiveRequests;
    stats.iterLatencyMS = iterLatencyMS;
    stats.newActiveRequestsQueueLatencyMS = newActiveRequestsQueueLatencyMS;
    // Active request count
    stats.numActiveRequests = static_cast<SizeType32>(activeRequests.size());
    // Queued request count
    {
        std::scoped_lock<std::mutex> lck(mQueuedReqMtx);
        stats.numQueuedRequests = static_cast<SizeType32>(mQueuedRequests.size());
    }
    stats.numCompletedRequests = numCompletedRequests;
    // Max number of requests
    stats.maxNumActiveRequests = mMaxNumActiveRequests;
    // Runtime memory allocation statistics
    auto const& memoryCounters = runtime::MemoryCounters::getInstance();
    stats.gpuMemUsage = memoryCounters.getGpu();
    stats.cpuMemUsage = memoryCounters.getCpu();
    stats.pinnedMemUsage = memoryCounters.getPinned();

    // Model specific stats
    mModel->getCurrentIterationStats(stats);
    return stats;
}

RequestStatsPerIteration Executor::Impl::getCurrentRequestStats(
    RequestList const& activeRequests, RequestList const& finishedRequests)
{
    std::vector<RequestStats> requestStatsVec;

    auto includeDisServingStats = [](LlmRequestPtr const& request, tensorrt_llm::executor::RequestStats& requestStats)
    {
        auto requestType = request->getLlmRequestType();
        if (requestType == batch_manager::LlmRequestType::LLMREQUEST_TYPE_CONTEXT_ONLY
            || requestType == batch_manager::LlmRequestType::LLMREQUEST_TYPE_GENERATION_ONLY)
        {
            requestStats.disServingStats
                = executor::DisServingRequestStats{request->getKvCacheTransferTimeMS(), request->getKvCacheSize()};
        }
    };

    for (auto const& request : activeRequests)
    {
        RequestStats requestStats;
        requestStats.id = request->mRequestId;
        requestStats.stage = request->getRequestStage();
        requestStats.contextPrefillPosition = request->getContextCurrentPosition();
        requestStats.numGeneratedTokens = request->getMaxBeamNumTokens() - request->getOrigPromptLen();
        requestStats.avgNumDecodedTokensPerIter = request->getAvgDecodedTokensPerIter();
        includeDisServingStats(request, requestStats);
        requestStats.allocTotalBlocksPerRequest = request->getAllocTotalBlocksPerRequest();
        requestStats.allocNewBlocksPerRequest = request->getAllocNewBlocksPerRequest();
        requestStats.reusedBlocksPerRequest = request->getReusedBlocksPerRequest();
        requestStats.missedBlocksPerRequest = request->getMissedBlocksPerRequest();
        requestStats.kvCacheHitRatePerRequest = request->getKVCacheHitRatePerRequest();
        requestStatsVec.emplace_back(requestStats);
    }

    {
        std::unique_lock<std::mutex> lck(mQueuedReqMtx);
        for (auto const& request : mQueuedRequests)
        {
            // Still waiting for the first scheduling
            RequestStats requestStats;
            requestStats.id = static_cast<executor::IdType>(request.id);
            requestStats.stage = executor::RequestStage::kQUEUED;
            requestStats.contextPrefillPosition = 0;
            requestStats.numGeneratedTokens = 0;
            requestStats.avgNumDecodedTokensPerIter = 0;
            requestStats.allocTotalBlocksPerRequest = 0;
            requestStats.allocNewBlocksPerRequest = 0;
            requestStats.reusedBlocksPerRequest = 0;
            requestStats.missedBlocksPerRequest = 0;
            requestStats.kvCacheHitRatePerRequest = 0;
            requestStatsVec.emplace_back(requestStats);
        }
    }

    for (auto const& request : finishedRequests)
    {
        // Still waiting for the first scheduling
        RequestStats requestStats;
        requestStats.id = static_cast<executor::IdType>(request->mRequestId);
        requestStats.stage = executor::RequestStage::kGENERATION_COMPLETE;
        requestStats.contextPrefillPosition = request->getContextCurrentPosition();
        requestStats.numGeneratedTokens = request->getMaxBeamNumTokens() - request->getOrigPromptLen();
        requestStats.avgNumDecodedTokensPerIter = request->getAvgDecodedTokensPerIter();
        includeDisServingStats(request, requestStats);
        requestStats.allocTotalBlocksPerRequest = request->getAllocTotalBlocksPerRequest();
        requestStats.allocNewBlocksPerRequest = request->getAllocNewBlocksPerRequest();
        requestStats.reusedBlocksPerRequest = request->getReusedBlocksPerRequest();
        requestStats.missedBlocksPerRequest = request->getMissedBlocksPerRequest();
        requestStats.kvCacheHitRatePerRequest = request->getKVCacheHitRatePerRequest();
        requestStatsVec.emplace_back(requestStats);
    }

    RequestStatsPerIteration stats{0, std::move(requestStatsVec)};

    // Model specific stats
    mModel->getCurrentRequestStats(stats);
    return stats;
}

void Executor::Impl::appendCurrentIterStats(IterationStats&& currentIterStats)
{
    std::scoped_lock<std::mutex> lck(mIterStatsMtx);
    if (mIterationStats.size() >= mIterStatsMaxIterations)
    {
        mIterationStats.pop_front();
    }
    mIterationStats.emplace_back(std::move(currentIterStats));
}

void Executor::Impl::appendMultipleIterStats(std::vector<IterationStats>&& currentIterStatsVec)
{
    std::scoped_lock<std::mutex> lck(mIterStatsMtx);
    if (mIterationStats.size() + currentIterStatsVec.size() > mIterStatsMaxIterations)
    {
        size_t removeCount = mIterationStats.size() + currentIterStatsVec.size() - mIterStatsMaxIterations;
        for (size_t i = 0; i < removeCount; i++)
        {
            mIterationStats.pop_front();
        }
    }
    mIterationStats.insert(mIterationStats.end(), std::make_move_iterator(currentIterStatsVec.begin()),
        std::make_move_iterator(currentIterStatsVec.end()));
}

void Executor::Impl::updateIterationStats(RequestList const& activeRequests, double iterLatencyMS,
    SizeType32 numNewActiveRequests, double newActiveRequestsQueueLatencyMS, SizeType32 numCompletedRequests,
    bool flushToOrchestrator)
{
    NVTX3_SCOPED_RANGE(updateIterationStats);
    if (mIterStatsMaxIterations > 0 && mIsLeader)
    {
        auto currentIterStats = getCurrentIterationStats(
            activeRequests, iterLatencyMS, numNewActiveRequests, newActiveRequestsQueueLatencyMS, numCompletedRequests);
        // Send the stats to the orchestrator
        if (mCommMode == CommunicationMode::kORCHESTRATOR)
        {
            bool hasSchedThisIter = (currentIterStats.inflightBatchingStats
                                        && currentIterStats.inflightBatchingStats->numScheduledRequests > 0)
                || (currentIterStats.staticBatchingStats
                    && currentIterStats.staticBatchingStats->numScheduledRequests > 0);
            appendCurrentIterStats(std::move(currentIterStats));
            if (hasSchedThisIter || flushToOrchestrator)
            {
                std::deque<IterationStats> iterStatsQueue;
                {
                    std::scoped_lock<std::mutex> lck(mIterStatsMtx);
                    iterStatsQueue = std::exchange(mIterationStats, {});
                }
                MpiMessage message(MpiId::ITER_STATS);
                std::vector<IterationStats> iterStates(
                    std::make_move_iterator(iterStatsQueue.begin()), std::make_move_iterator(iterStatsQueue.end()));
                message.data = IterStatsData{std::move(iterStates)};
                mSendQueue.push(std::move(message));
            }
        }
        else
        {
            // Add current iteration stats
            appendCurrentIterStats(std::move(currentIterStats));
        }
    }
}

void Executor::Impl::appendCurrentRequestStats(RequestStatsPerIteration&& currentRequestStats)
{
    std::scoped_lock<std::mutex> lck(mRequestStatsMtx);
    if (mRequestStats.size() >= mRequestStatsMaxIterations)
    {
        mRequestStats.pop_front();
    }
    mRequestStats.emplace_back(std::move(currentRequestStats));
}

void Executor::Impl::appendMultipleRequestStats(std::vector<RequestStatsPerIteration>&& currentRequestStatsVec)
{
    std::scoped_lock<std::mutex> lck(mRequestStatsMtx);
    if (mRequestStats.size() + currentRequestStatsVec.size() > mRequestStatsMaxIterations)
    {
        size_t removeCount = mRequestStats.size() + currentRequestStatsVec.size() - mRequestStatsMaxIterations;
        for (size_t i = 0; i < removeCount; i++)
        {
            mRequestStats.pop_front();
        }
    }
    mRequestStats.insert(mRequestStats.end(), std::make_move_iterator(currentRequestStatsVec.begin()),
        std::make_move_iterator(currentRequestStatsVec.end()));
}

void Executor::Impl::updateRequestStats(
    RequestList const& activeRequests, RequestList const& finishedRequests, bool flushToOrchestrator)
{
    NVTX3_SCOPED_RANGE(updateRequestStats);
    if (mRequestStatsMaxIterations > 0 && mIsLeader)
    {
        // Add current iteration request stats
        auto currentRequestStats = getCurrentRequestStats(activeRequests, finishedRequests);
        // Send the stats to the orchestrator
        if (mCommMode == CommunicationMode::kORCHESTRATOR)
        {
            bool hasScheduledReqs = false;
            if (!flushToOrchestrator)
            {
                size_t activeSize = activeRequests.size();
                TLLM_CHECK_WITH_INFO(currentRequestStats.requestStats.size() >= activeSize,
                    "currentRequestStats num is %ld should >= activeRequest num:%zu",
                    currentRequestStats.requestStats.size(), activeSize);
                hasScheduledReqs = std::any_of(currentRequestStats.requestStats.begin(),
                    currentRequestStats.requestStats.begin() + static_cast<int64_t>(activeSize),
                    [](RequestStats const& requestStat) { return requestStat.scheduled; });
            }
            appendCurrentRequestStats(std::move(currentRequestStats));
            if (hasScheduledReqs || flushToOrchestrator)
            {
                std::deque<RequestStatsPerIteration> requestStatsQueue;
                {
                    std::scoped_lock<std::mutex> lck(mRequestStatsMtx);
                    requestStatsQueue = std::exchange(mRequestStats, {});
                }
                std::vector<RequestStatsPerIteration> requestIterStates(
                    std::make_move_iterator(requestStatsQueue.begin()),
                    std::make_move_iterator(requestStatsQueue.end()));
                MpiMessage message(MpiId::REQUEST_ITER_STATS);
                message.data = RequestStatsPerIterationData{std::move(requestIterStates)};
                mSendQueue.push(std::move(message));
            }
        }
        else
        {
            // Add current iteration stats
            appendCurrentRequestStats(std::move(currentRequestStats));
        }
    }
}

void Executor::Impl::appendCurrentDebugTensors()
{
    if (mDebugTensorsMaxIterations > 0)
    {
        std::scoped_lock<std::mutex> lck(mDebugTensorsMtx);
        if (mDebugTensors.size() >= mDebugTensorsMaxIterations)
        {
            mDebugTensors.pop_front();
        }
        mDebugTensors.emplace_back(mModel->getCurrentDebugTensors());
    }
}

void Executor::Impl::terminateCancelledRequests(RequestList& activeRequests)
{
    NVTX3_SCOPED_RANGE(terminateCancelledRequests);
    auto const& worldConfig = mModel->getWorldConfig();
    auto const broadcastCancelledRequests = [this, &activeRequests, &worldConfig]
    {
        auto const& commSession = COMM_SESSION;

        if (worldConfig.isPipelineParallel())
        {
            mCancelledRequestsWaitThread->waitStop();
        }

        if (commSession.getSize() > 1 && !activeRequests.empty())
        {
            if (mIsPipelineLeader)
            {
                if (worldConfig.isPipelineParallel())
                {
                    auto const peer = worldConfig.getPipelineParallelism() - 1;
                    bool shouldExit = false;
                    mCommPipelineParallel->send(
                        &shouldExit, 1, mpi::MpiType::kBOOL, peer, mpi::MpiTag::kExecutorShouldExit);
                    auto pipelineCancelledReqIds
                        = CancelledRequestsAsyncSend::cancelledRequestsRecv(mCommPipelineParallel, peer);
                    mCancelledReqIds.insert(pipelineCancelledReqIds.begin(), pipelineCancelledReqIds.end());
                }

                auto numCancelledRequests = static_cast<int64_t>(mCancelledReqIds.size());
                if (worldConfig.isTensorParallel())
                {
                    mCommTensorParallel->bcastValue(numCancelledRequests, 0);
                    if (numCancelledRequests > 0)
                    {
                        std::vector<IdType> cancelledReqIdsVec(mCancelledReqIds.begin(), mCancelledReqIds.end());
                        mCommTensorParallel->bcast(
                            cancelledReqIdsVec.data(), cancelledReqIdsVec.size(), mpi::MpiType::kUINT64, 0);
                    }
                }
                if (worldConfig.isContextParallel())
                {
                    mCommContextParallel->bcastValue(numCancelledRequests, 0);
                    if (numCancelledRequests > 0)
                    {
                        std::vector<IdType> cancelledReqIdsVec(mCancelledReqIds.begin(), mCancelledReqIds.end());
                        mCommContextParallel->bcast(
                            cancelledReqIdsVec.data(), cancelledReqIdsVec.size(), mpi::MpiType::kUINT64, 0);
                    }
                }
            }
            // If not leader
            else
            {
                if (worldConfig.isFirstPipelineParallelRank())
                {
                    int64_t numCancelledRequests = 0;
                    mCommTensorParallel->bcastValue(numCancelledRequests, 0);
                    mCommContextParallel->bcastValue(numCancelledRequests, 0);
                    if (numCancelledRequests > 0)
                    {
                        std::vector<IdType> cancelledReqIdsVec(numCancelledRequests);
                        mCommTensorParallel->bcast(
                            cancelledReqIdsVec.data(), cancelledReqIdsVec.size(), mpi::MpiType::kUINT64, 0);
                        mCommContextParallel->bcast(
                            cancelledReqIdsVec.data(), cancelledReqIdsVec.size(), mpi::MpiType::kUINT64, 0);
                        mCancelledReqIds
                            = std::unordered_set<IdType>(cancelledReqIdsVec.begin(), cancelledReqIdsVec.end());
                    }
                }
                else
                {
                    auto const peer = worldConfig.getPipelineParallelRank() - 1;
                    mCancelledReqIds = CancelledRequestsAsyncSend::cancelledRequestsRecv(mCommPipelineParallel, peer);
                }
            }
            if (!worldConfig.isLastPipelineParallelRank())
            {
                auto const peer = worldConfig.getPipelineParallelRank() + 1;
                mCancelledRequestsAsyncSndHdl
                    = std::make_unique<CancelledRequestsAsyncSend>(mCommPipelineParallel, mCancelledReqIds, peer);
                mCancelledRequestsWaitThread->notifyStart();
            }
        }
    };

    std::unique_lock<std::mutex> lck{mCancelReqMtx, std::defer_lock};
    if (!worldConfig.isPipelineParallel())
    {
        lck.lock();
    }

    broadcastCancelledRequests();

    if (!mCancelledReqIds.empty())
    {
        // Loop over active requests and terminate those that have been cancelled
        std::unordered_set<IdType> terminatedReqIds;
        for (auto& req : activeRequests)
        {
            auto reqId = req->isChild() ? req->getParentRequestId() : req->mRequestId;
            if (mCancelledReqIds.find(reqId) != mCancelledReqIds.end())
            {
                auto finishReason = req->isTimedOut() ? FinishReason::kTIMED_OUT : FinishReason::kCANCELLED;
                mModel->terminateRequestSync(req, finishReason);
                // Parent and child requests share the same request id.
                // Mark it terminated first and remove from the set later.
                terminatedReqIds.insert(reqId);
            }
        }

        for (auto const& reqId : terminatedReqIds)
        {
            mCancelledReqIds.erase(reqId);
        }
    }
}

void Executor::Impl::terminateContextFinishedRequests(InTransList& inTransmissionRequests)
{
    NVTX3_SCOPED_RANGE(terminateContextFinishedRequests);
    for (auto it = inTransmissionRequests.begin(); it != inTransmissionRequests.end();)
    {
        auto& item = *it;
        auto req = item.request;
        if (req->isDisaggContextCompleteState())
        {
            // If lastBlockId was tracked, unpin it. Otherwise, just terminate.
            auto kvMgr = mModel->getKVCacheManager();
            if (kvMgr && item.lastBlockId.has_value())
            {
                kvMgr->unpinBlocksById(item.lastBlockId.value());
            }
            else
            {
                mModel->terminateRequest(req);
            }
            it = inTransmissionRequests.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void Executor::Impl::appendNewResponses(std::vector<Response>&& newResponses)
{
    {
        std::scoped_lock<std::mutex> lck(mResponsesMtx);
        for (auto& response : newResponses)
        {
            mResponses[response.getRequestId()].emplace_back(std::move(response));
        }
    }
    mResponsesCv.notify_all();
}

Executor::Impl::RequestList Executor::Impl::populateNewResponses(
    RequestList& activeRequests, InTransList& inTransmissionRequests, std::vector<Response>& newResponses)
{
    NVTX3_SCOPED_RANGE(populateNewResponses);
    RequestList finishedRequests;
    for (auto it = activeRequests.begin(); it != activeRequests.end();)
    {
        auto const& llmReq = (*it);
        bool const requestDone = llmReq->isFinished();
        // Only leader should store responses
        if (mIsLeader)
        {
            auto response = llmReq->createResponse(mModel->hasSpeculativeDecodingFastLogits(), mWorldRank);
            if (response)
            {
                newResponses.emplace_back(std::move(response.value()));
            }
        }
        // Remove from active requests if last response has been generated
        if (requestDone)
        {
            // move the in transmission requests to another tracker
            if (llmReq->isDisaggContextTransmissionState())
            {
                std::optional<SizeType32> lastBlockId{};
                auto kvMgr = mModel->getKVCacheManager();
                if (kvMgr && kvMgr->isEnableBlockReuse() && !kvMgr->getBlockManager().isVariableWindow())
                {
                    lastBlockId = kvMgr->storeBlocksForReuse(llmReq->mRequestId, llmReq, /*pinBlocks=*/true);
                    mModel->terminateRequest(llmReq);
                }
                inTransmissionRequests.push_back(InTransmissionItem{*it, lastBlockId});
            }
            finishedRequests.push_back(*it);
            it = activeRequests.erase(it);
        }
        else
        {
            ++it;
        }
    }
    return finishedRequests;
}

void Executor::Impl::executionLoop()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    tensorrt_llm::common::setThreadName("executionLoop");

    auto const& worldConfig = mModel->getWorldConfig();
    TLLM_CUDA_CHECK(cudaSetDevice(worldConfig.getDevice()));

    auto const [profileIterIdxs, stopIterIdxs] = tensorrt_llm::common::populateIterationIndexes(
        kPROFILE_START_STOP_ENV_VAR_NAME, kLEGACY_PROFILE_START_STOP_ENV_VAR_NAME);

    SizeType32 numNewActiveRequests{0};
    std::chrono::time_point<std::chrono::steady_clock> iterStart;
    std::chrono::time_point<std::chrono::steady_clock> iterEnd;
    bool firstIteration{true};
    RequestList activeRequests;
    InTransList inTransmissionRequests;
    std::vector<Response> newResponses;
    while (!mShutdown || !activeRequests.empty())
    {
        double iterLatencyMS{0.0};
        double newActiveRequestsQueueLatencyMS{0.0};
        bool reportFinishedRequests = true;
        RequestList finishedRequests;
        if (!activeRequests.empty())
        {
            finishTimedOutRequests(activeRequests);
            terminateCancelledRequests(activeRequests);
            forwardSync(activeRequests);
            finishedRequests = populateNewResponses(activeRequests, inTransmissionRequests, newResponses);
            cleanupDynamicLogitsPostProcessors(finishedRequests);
            auto const iterCounter = mModel->getIterCounter();
            auto const stopIter = !stopIterIdxs.empty() && (stopIterIdxs.count(iterCounter - 1) > 0);
            if (stopIter)
            {
                cudaProfilerStop();
            }

            // When there are no active or inflight requests, we need to update the stats before calling
            // fetchNewRequests to make sure that the stats are reported accurately.
            if (activeRequests.empty() && (!firstIteration))
            {
                mModel->resetIterationStats();
                updateIterationStats(activeRequests, iterLatencyMS, numNewActiveRequests,
                    newActiveRequestsQueueLatencyMS, static_cast<SizeType32>(finishedRequests.size()), true);
                updateRequestStats(activeRequests, finishedRequests, true);
                reportFinishedRequests = false;
            }
            if (!newResponses.empty())
            {
                enqueueNewResponses(std::move(newResponses));
                newResponses.clear();
            }
            iterEnd = std::chrono::steady_clock::now();
            iterLatencyMS = std::chrono::duration<double, std::milli>(iterEnd - iterStart).count();
        }

        if (!inTransmissionRequests.empty())
        {
            terminateContextFinishedRequests(inTransmissionRequests);
        }

        if (!mShutdown)
        {
            auto const iterCounter = mModel->getIterCounter();
            auto const profileIter = !profileIterIdxs.empty() && (profileIterIdxs.count(iterCounter) > 0);
            if (profileIter)
            {
                cudaProfilerStart();
            }
            iterStart = std::chrono::steady_clock::now();
            std::optional<PriorityType> lowestPriority = std::nullopt;
            if (!activeRequests.empty())
            {
                lowestPriority = activeRequests.back()->priority();
            }

            auto [newRequests, newActiveRequestsQueueLatency]
                = fetchNewRequests(static_cast<SizeType32>(activeRequests.size()), lowestPriority);
            newActiveRequestsQueueLatencyMS = newActiveRequestsQueueLatency;
            numNewActiveRequests = newRequests.size();

            if (firstIteration)
            {
                firstIteration = false;
            }

            for (auto const& newRequest : newRequests)
            {
                insertRequestInOrder(activeRequests, newRequest);
            }

            // Update dynamic tuning stats
            if (mDynamicBatchTuner)
            {
                for (auto const& req : activeRequests)
                {
                    auto const inputLength = req->mPromptLen;
                    auto const outputLength = req->mMaxNewTokens;
                    mDynamicBatchTuner->updateStats(inputLength, outputLength);
                }
            }
        }

        if (!activeRequests.empty())
        {
            forwardAsync(activeRequests);
            updateIterationStats(activeRequests, iterLatencyMS, numNewActiveRequests, newActiveRequestsQueueLatencyMS,
                static_cast<SizeType32>(finishedRequests.size()), false);
            // Finished requests were reported once. Avoid reporting it twice.
            if (reportFinishedRequests)
            {
                updateRequestStats(activeRequests, finishedRequests, false);
            }
            else
            {
                updateRequestStats(activeRequests, {}, false);
            }
            appendCurrentDebugTensors();
        }
    }

    if (mCancelledRequestsWaitThread)
    {
        mCancelledRequestsWaitThread.reset(nullptr);
    }
    if (mRequestWithIdWaitThread)
    {
        mRequestWithIdWaitThread.reset(nullptr);
    }
    if (worldConfig.isPipelineParallel() && mIsPipelineLeader)
    {
        auto const peer = worldConfig.getPipelineParallelism() - 1;
        int64_t numActiveRequests = -1;
        mCommPipelineParallel->send(
            &numActiveRequests, 1, mpi::MpiType::kINT64, peer, mpi::MpiTag::kExecutorNumActiveRequests);
        bool shouldExit = true;
        mCommPipelineParallel->send(&shouldExit, 1, mpi::MpiType::kBOOL, peer, mpi::MpiTag::kExecutorShouldExit);
    }
    if (mRequestWithIdLeaderThread)
    {
        mRequestWithIdLeaderThread->join();
        mRequestWithIdLeaderThread.reset(nullptr);
    }
    if (mCancelledRequestsLeaderThread)
    {
        mCancelledRequestsLeaderThread->join();
        mCancelledRequestsLeaderThread.reset(nullptr);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void Executor::Impl::enqueueTerminateRequest()
{
    {
        std::scoped_lock<std::mutex> lck(mQueuedReqMtx);
        Request dummyReq({1}, 1);
        RequestWithId reqWithId{std::move(dummyReq), mTerminateReqId};
        mQueuedRequests.emplace_back(reqWithId);
    }
    mQueuedReqCv.notify_one();
}

void Executor::Impl::enqueueNewResponses(std::vector<Response>&& newResponses)
{
    TLLM_CHECK_WITH_INFO(mIsLeader, "Only leader should store responses");

    if (mCommMode == CommunicationMode::kLEADER)
    {
        appendNewResponses(std::move(newResponses));
    }
    else if (mCommMode == CommunicationMode::kORCHESTRATOR)
    {
        MpiMessage message(MpiId::RESPONSE);
        message.data = ResponseData{std::move(newResponses)};
        mSendQueue.push(std::move(message));
    }
}

// Orchestrator thread sending new requests to leader of the model
void Executor::Impl::orchSendReqThread()
{
    tensorrt_llm::common::setThreadName("orchSendReq");

    while (true)
    {
        auto message = mSendQueue.pop();

        if (message.id == MpiId::TERMINATION)
        {
            mOrchLeaderComm->send(&message.id, 1, mpi::MpiType::kUINT64, mLeaderRank, mpi::MpiTag::kOrchestratorId);
            TLLM_LOG_INFO("Orchestrator sendReq thread exiting");
            break;
        }
        if (message.id == MpiId::PENDING_REQUEST)
        {
            auto& reqWithIds = std::get<PendingRequestData>(message.data);
            auto packed = RequestWithId::serializeReqWithIds(reqWithIds.requests);

            TLLM_LOG_DEBUG("Orchestrator sendReq thread sending %d pending requests", reqWithIds.requests.size());
            // Temporary WAR to indicate to client that we cannot send the serialized request
            // because it exceeds int32_t size limit.
            // TODO: Should fix as part of https://jirasw.nvidia.com/browse/TRTLLM-708
            if (packed.size() > std::numeric_limits<int32_t>::max())
            {
                for (auto const& reqWithId : reqWithIds.requests)
                {
                    {
                        std::scoped_lock<std::mutex> lck(mResponsesMtx);
                        mResponses[reqWithId.id].emplace_back(reqWithId.id,
                            "Request is too large, or you are enqueuing too many requests at once "
                            "to be sent via MPI_Send, please try to enqueue the request(s) again. "
                            "This issue will be resolved in a future version of TRT-LLM.");
                    }
                    mResponsesCv.notify_all();
                }
            }
            else
            {
                mOrchLeaderComm->send(&message.id, 1, mpi::MpiType::kUINT64, mLeaderRank, mpi::MpiTag::kOrchestratorId);
                mOrchLeaderComm->send(
                    packed.data(), packed.size(), mpi::MpiType::kCHAR, mLeaderRank, mpi::MpiTag::kOrchestratorData);
            }
        }
        else if (message.id == MpiId::CANCEL_REQUEST)
        {
            auto& data = std::get<RequestIdsData>(message.data);

            mOrchLeaderComm->send(&message.id, 1, mpi::MpiType::kUINT64, mLeaderRank, mpi::MpiTag::kOrchestratorId);
            mOrchLeaderComm->send(
                data.ids.data(), data.ids.size(), mpi::MpiType::kUINT64, mLeaderRank, mpi::MpiTag::kOrchestratorData);
        }
        else
        {
            TLLM_THROW("Invalid message id");
        }
    }
}

// Leader thread receiving new requests from orchestrator
void Executor::Impl::leaderRecvReqThread()
{
    tensorrt_llm::common::setThreadName("leaderRecvReq");
    TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
#if ENABLE_MULTI_DEVICE
    auto& selCancelledReqIds = mUsePipelineParallel ? mPipelineCancelledReqIds : mCancelledReqIds;
    while (true)
    {
        if (mRecvPollPeriodMs > 0)
        {
            mOrchLeaderComm->recvPoll(mOrchRank, mpi::MpiTag::kOrchestratorId, mRecvPollPeriodMs);
        }

        // Blocking is okay: terminate message is expected to arrive here
        MPI_Message msg = nullptr;
        MPI_Status status;
        mOrchLeaderComm->mprobe(mOrchRank, mpi::MpiTag::kOrchestratorId, &msg, &status);

        int32_t count = 0;
        MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count)); // NOLINT
        TLLM_CHECK(count == 1);

        MpiId mpiId{};
        MPICHECK(MPI_Mrecv(&mpiId, count, MPI_UINT64_T, &msg, &status)); // NOLINT

        // EXIT condition from receiving TERMINATE msg
        if (mpiId == MpiId::TERMINATION)
        {
            // Enqueue a request to indicate to other ranks to terminate
            enqueueTerminateRequest();

            // Send message to orchestrator to indicate to terminate orch recv thread
            mSendQueue.push(MpiMessage(mpiId));
            TLLM_LOG_INFO("Leader recvReq thread exiting");
            break;
        }
        if (mpiId == MpiId::PENDING_REQUEST)
        {
            mOrchLeaderComm->mprobe(mOrchRank, mpi::MpiTag::kOrchestratorData, &msg, &status);
            MPICHECK(MPI_Get_count(&status, MPI_CHAR, &count));                 // NOLINT
            std::vector<char> buffer(count);
            MPICHECK(MPI_Mrecv(buffer.data(), count, MPI_CHAR, &msg, &status)); // NOLINT

            auto requestWithIds = RequestWithId::deserializeReqWithIds(buffer);
            TLLM_LOG_DEBUG("Leader recvReq thread receiving %d pending requests", requestWithIds.size());
            {
                std::scoped_lock<std::mutex> lck(mQueuedReqMtx);
                if (mMaxQueueSize)
                {
                    auto const maxQueueSize = mMaxQueueSize.value();
                    if (maxQueueSize > 0 && mQueuedRequests.size() >= static_cast<size_t>(maxQueueSize))
                    {
                        auto err = tensorrt_llm::common::fmtstr(
                            "Maximum queue size of %d has been reached, please try again later", maxQueueSize);
                        TLLM_LOG_ERROR("%s", err.c_str());
                        std::vector<Response> responses;
                        responses.reserve(requestWithIds.size());
                        for (auto const& reqWithId : requestWithIds)
                        {
                            responses.emplace_back(reqWithId.id, err);
                        }
                        enqueueNewResponses(std::move(responses));
                        continue;
                    }
                }
                for (auto&& req : requestWithIds)
                {
                    req.queuedStart = std::chrono::steady_clock::now();
                    insertRequestInOrder(mQueuedRequests, std::move(req));
                }
            }
            mQueuedReqCv.notify_one();
        }
        else if (mpiId == MpiId::CANCEL_REQUEST)
        {
            // Prepare receiving data
            mOrchLeaderComm->mprobe(mOrchRank, mpi::MpiTag::kOrchestratorData, &msg, &status);
            MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));                          // NOLINT
            std::vector<uint64_t> cancelledReqIds(count);
            MPICHECK(MPI_Mrecv(cancelledReqIds.data(), count, MPI_UINT64_T, &msg, &status)); // NOLINT

            std::scoped_lock<std::mutex> lck(mCancelReqMtx);
            selCancelledReqIds.insert(cancelledReqIds.begin(), cancelledReqIds.end());
        }
        else
        {
            TLLM_THROW("Invalid message id");
        }
    }
#endif // ENABLE_MULTI_DEVICE
}

// Leader thread sending responses to orchestrator
void Executor::Impl::leaderSendThread(MpiMessageQueue& sendQueue, mpi::MpiTag idTag, mpi::MpiTag dataTag)
{
    tensorrt_llm::common::setThreadName("leaderSend");
    TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));

#if ENABLE_MULTI_DEVICE
    while (true)
    {
        auto message = sendQueue.pop();

        if (message.id == MpiId::TERMINATION)
        {
            mOrchLeaderComm->send(&message.id, 1, mpi::MpiType::kUINT64, mOrchRank, idTag);
            TLLM_LOG_INFO("Leader sendThread exiting");
            break;
        }
        if (message.id == MpiId::RESPONSE || message.id == MpiId::ITER_STATS
            || message.id == MpiId ::REQUEST_ITER_STATS)
        {
            std::vector<char> buffer;
            if (message.id == MpiId::RESPONSE)
            {
                auto& responseData = std::get<ResponseData>(message.data);
                TLLM_LOG_DEBUG("Leader sendResp thread sending %d responses", responseData.responses.size());
                buffer = Serialization::serialize(responseData.responses);
            }
            else if (message.id == MpiId::ITER_STATS)
            {
                auto& iterStatsData = std::get<IterStatsData>(message.data);
                TLLM_LOG_DEBUG("Leader sendResp thread sending iter stats");
                buffer = Serialization::serialize(iterStatsData.iterStatsVec);
            }
            else if (message.id == MpiId::REQUEST_ITER_STATS)
            {
                auto& requestIterStatsData = std::get<RequestStatsPerIterationData>(message.data);
                TLLM_LOG_DEBUG("Leader sendResp thread sending iter request stats");
                buffer = Serialization::serialize(requestIterStatsData.requestStatsPerIterationVec);
            }
            mOrchLeaderComm->send(&message.id, 1, mpi::MpiType::kUINT64, mOrchRank, idTag);
            mOrchLeaderComm->send(buffer.data(), buffer.size(), mpi::MpiType::kCHAR, mOrchRank, dataTag);
        }
        else
        {
            TLLM_THROW("Invalid message id");
        }
    }
#endif // ENABLE_MULTI_DEVICE
}

void Executor::Impl::orchRecvThread(mpi::MpiTag idTag, mpi::MpiTag dataTag)
{
    tensorrt_llm::common::setThreadName("orchRecv");

#if ENABLE_MULTI_DEVICE
    while (true)
    {
        if (mRecvPollPeriodMs > 0)
        {
            mOrchLeaderComm->recvPoll(mOrchRank, mpi::MpiTag::kOrchestratorId, mRecvPollPeriodMs);
        }

        MPI_Message msg = nullptr;
        MPI_Status status;
        mOrchLeaderComm->mprobe(mLeaderRank, idTag, &msg, &status);

        int32_t count = 0;
        MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count)); // NOLINT
        TLLM_CHECK(count == 1);

        MpiId mpiId{};
        MPICHECK(MPI_Mrecv(&mpiId, count, MPI_UINT64_T, &msg, &status)); // NOLINT

        if (mpiId == MpiId::TERMINATION)
        {
            TLLM_LOG_INFO("Orchestrator recv thread exiting");
            break;
        }
        if (mpiId == MpiId::RESPONSE || mpiId == MpiId::ITER_STATS || mpiId == MpiId::REQUEST_ITER_STATS)
        {
            mOrchLeaderComm->mprobe(mLeaderRank, dataTag, &msg, &status);
            MPICHECK(MPI_Get_count(&status, MPI_CHAR, &count)); // NOLINT

            std::vector<char> buffer(count);
            MPICHECK(MPI_Mrecv(buffer.data(), count, MPI_CHAR, &msg, &status)); // NOLINT

            if (mpiId == MpiId::RESPONSE)
            {
                auto newResponses = Serialization::deserializeResponses(buffer);
                TLLM_LOG_DEBUG("Orchestrator recv thread receiving %d responses", newResponses.size());
                appendNewResponses(std::move(newResponses));
            }
            else if (mpiId == MpiId::ITER_STATS)
            {
                appendMultipleIterStats(Serialization::deserializeIterationStatsVec(buffer));
            }
            else if (mpiId == MpiId::REQUEST_ITER_STATS)
            {
                appendMultipleRequestStats(Serialization::deserializeRequestStatsPerIterationVec(buffer));
            }
        }
        else
        {
            TLLM_THROW("Invalid message id");
        }
    }
#endif // ENABLE_MULTI_DEVICE
}

Executor::Impl::LlmRequestLogitsPostProcessor Executor::Impl::getLogitsPostProcessor(std::string const& name)
{
    auto const postProcIt = mLogitsPostProcessorMap.find(name);
    TLLM_CHECK_WITH_INFO(
        postProcIt != mLogitsPostProcessorMap.end(), "LogitsPostProcessor %s not found.", name.c_str());
    auto executorLogitsPostProcessor = postProcIt->second;
    return [executorLogitsPostProcessor](IdType reqId, RtTensorPtr& logits, BeamTokens const& beamTokens,
               CudaStreamPtr const& cudaStreamPtr, std::optional<IdType> clientId)
    {
        auto logitsTensor = executor::detail::ofITensor(logits);
        executorLogitsPostProcessor(reqId, logitsTensor, beamTokens, cudaStreamPtr, clientId);
    };
}

void Executor::Impl::setupDynamicLogitsPostProcessors(std::vector<RequestWithId>& newReqWithIds)
{
    for (auto& reqWithId : newReqWithIds)
    {
        auto logitsPostProcessor = reqWithId.req.getLogitsPostProcessor();
        if (logitsPostProcessor)
        {
            std::string const name = Request::kDynamicPostProcessorNamePrefix + std::to_string(reqWithId.id);
            mLogitsPostProcessorMap[name] = logitsPostProcessor.value();
            reqWithId.req.setLogitsPostProcessor(std::nullopt);
            reqWithId.req.setLogitsPostProcessorName(name);
        }
    }
}

void Executor::Impl::cleanupDynamicLogitsPostProcessors(RequestList const& finishedRequests)
{
    for (auto& req : finishedRequests)
    {
        std::string const name = Request::kDynamicPostProcessorNamePrefix + std::to_string(req->mRequestId);
        auto const postProcIt = mLogitsPostProcessorMap.find(name);
        if (postProcIt != mLogitsPostProcessorMap.end())
        {
            mLogitsPostProcessorMap.erase(name);
        }
    }
}

void Executor::Impl::addTerminatedReqId(std::vector<Response> const& responses, IdType const& reqId)
{
    for (auto const& response : responses)
    {
        if (response.hasError() || (!response.hasError() && response.getResult().isFinal))
        {
            mTerminatedReqIds.insert(reqId);
            if (mChildReqIdsMap.find(reqId) != mChildReqIdsMap.end())
            {
                for (auto childReqId : mChildReqIdsMap.at(reqId))
                {
                    mTerminatedReqIds.insert(childReqId);
                }
                mChildReqIdsMap.erase(reqId);
            }
        }
    }
}

void Executor::Impl::checkParallelApiUsage(std::string const& methodName) const
{
    // If leader mode, and not leader, throw error
    if (mCommMode == CommunicationMode::kLEADER && !mIsLeader)
    {
        // Non-leader are not expected to call cancelRequest
        TLLM_THROW("With LEADER communication mode, only leader rank is expected to call %s", methodName.c_str());
    }
    if (mCommMode == CommunicationMode::kORCHESTRATOR && !mIsOrchestrator)
    {
        TLLM_THROW(
            "With ORCHESTRATOR communication mode, only orchestrator rank is expected to call %s", methodName.c_str());
    }
}

} // namespace tensorrt_llm::executor
