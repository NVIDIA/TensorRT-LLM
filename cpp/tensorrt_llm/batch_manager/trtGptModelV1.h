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
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include "trtGptModel.h"

#include <NvInferRuntime.h>

#include <tuple>

namespace tensorrt_llm::runtime
{
class GptSession;
}

namespace tensorrt_llm::batch_manager
{
class CapacityScheduler;
class MicroBatchScheduler;
class LlmRequest;

class [[deprecated("Use the InflightBatching model instead.")]] TrtGptModelV1 : public TrtGptModel
{
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TokenIdType = tensorrt_llm::runtime::TokenIdType;
    using TensorPtr = runtime::ITensor::SharedPtr;

public:
    struct IterationStatsV1
    {
        SizeType32 numScheduledRequests;
        SizeType32 numCtxTokensInBatch;
        SizeType32 numGenTokensInBatch;
        SizeType32 emptyGenSlots;
        ReqIdsSet scheduledRequests;
        ReqIdsSet pausedRequests;
    };

    TrtGptModelV1(std::shared_ptr<nvinfer1::ILogger> logger, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, runtime::RawEngine const& rawEngine,
        TrtGptModelOptionalParams const& optionalParams = TrtGptModelOptionalParams());

    ~TrtGptModelV1();

    // V1 model is stateless, so nothing to do here
    void terminateRequest(std::shared_ptr<LlmRequest> const& llmRequest, bool pause = false) override{};

    /// @brief This override is empty and solely exists to adhere to the interface
    void forwardSync() override;

    /// @brief Function that tries to advance the active requests
    ///        Depending on resources available, it's possible that not all requests will get advanced
    /// @param activeRequests The list of request to try to advance
    void forwardAsync(RequestList const& activeRequests) override;

    //! @brief Set LayerProfiler to collect performance per layer.
    void setLayerProfiler() override;

    //! @brief Print profile information per layer.
    std::string getLayerProfileInfo() const override;

    void updatePeftCache(std::shared_ptr<LlmRequest> const& llmRequest) override {}

    [[nodiscard]] runtime::ModelConfig const& getModelConfig() const override;

    [[nodiscard]] bool getGatherGenerationLogits() const override;

    [[nodiscard]] TrtGptModelType getModelType() const override
    {
        return TrtGptModelType::V1;
    };

    [[nodiscard]] SizeType32 getNumMicroBatches() const override;
    [[nodiscard]] runtime::WorldConfig const& getWorldConfig() const override;
    [[nodiscard]] IterationStatsV1 getLastIterationStats() const;
    [[nodiscard]] runtime::BufferManager const& getBufferManager() const override;
    [[nodiscard]] runtime::BufferManager::CudaStreamPtr getRuntimeStreamPtr() const override;
    [[nodiscard]] nvinfer1::DataType getLogitDataType() const override;
    [[nodiscard]] nvinfer1::DataType getTensorDataType(std::string const& name) const override;
    [[nodiscard]] nvinfer1::Dims getTensorShape(std::string const& name) const override;

    [[nodiscard]] executor::IterationType getIterCounter() const noexcept override
    {
        return mIterCounter;
    }

    void getCurrentIterationStats(executor::IterationStats& stats) const override;
    void getCurrentRequestStats(executor::RequestStatsPerIteration& stats) const override;
    [[nodiscard]] executor::DebugTensorsPerIteration getCurrentDebugTensors() const override;

    void setLogitsPostProcessorBatched(std::optional<LogitsPostProcessorBatched> logitsPostProcessorBatched) override;
    void setReplicateLogitsPostProcessor(bool replicateLogitsPostProcessor) override;
    bool getReplicateLogitsPostProcessor() const override;

    [[nodiscard]] static bool optionalParamsAreValid(
        runtime::ModelConfig const& modelConfig, TrtGptModelOptionalParams const& optionalParams);
    [[nodiscard]] static TrtGptModelOptionalParams fixOptionalParams(
        runtime::ModelConfig const& modelConfig, TrtGptModelOptionalParams const& optionalParams);

    void resetIterationStats() override;

    [[nodiscard]] SizeType32 getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const override
    {
        return 0;
    };

protected:
    [[nodiscard]] std::shared_ptr<kv_cache_manager::BaseKVCacheManager> getKVCacheManager() override;
    [[nodiscard]] std::shared_ptr<kv_cache_manager::BaseKVCacheManager const> getKVCacheManager() const override;

    [[nodiscard]] std::shared_ptr<BasePeftCacheManager> getPeftCacheManager() override
    {
        return mPeftCacheManager;
    }

    [[nodiscard]] std::shared_ptr<BasePeftCacheManager const> getPeftCacheManager() const override
    {
        return mPeftCacheManager;
    }

private:
    // callback stats
    static IterationStatsV1 fillIterationStats(RequestVector const& scheduledRequests, SizeType32 cappedMaxNewTokens,
        RequestVector const& requestsToTerminate);

    // Helper function to fill the generation table and batch sampling config from scheduled requests
    static std::tuple<runtime::GenerationInput, runtime::SamplingConfig> fillGenInputAndSamplingConfig(
        RequestVector const& scheduledRequests, runtime::BufferManager const& bufferManager,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig, SizeType32 maxSeqLen,
        SizeType32 maxBatchSize, bool normalizeLogProbs);

    std::shared_ptr<runtime::GptSession> mSession;
    std::unique_ptr<tensorrt_llm::batch_manager::CapacityScheduler const> mCapacityScheduler;
    std::unique_ptr<tensorrt_llm::batch_manager::MicroBatchScheduler const> mMicroBatchScheduler;
    std::shared_ptr<BasePeftCacheManager> mPeftCacheManager;
    IterationStatsV1 mLastIterationStatsV1;
    // Iteration counter used to distinguish debug output
    executor::IterationType mIterCounter{0};
    SizeType32 mPpTimesMaxBatchSize;
};

} // namespace tensorrt_llm::batch_manager
