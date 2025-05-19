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
#include "tensorrt_llm/batch_manager/logitsPostProcessor.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <nlohmann/json.hpp>

namespace tensorrt_llm::executor
{

class Model
{
    using LlmRequestPtr = std::shared_ptr<batch_manager::LlmRequest>;

public:
    Model() = default;

    virtual ~Model() = default;

    /// @brief Function that marks a request Id as complete and cleans up associated state
    virtual void terminateRequest(LlmRequestPtr const& llmRequest, bool pause) = 0;

    void terminateRequest(LlmRequestPtr const& llmRequest)
    {
        terminateRequest(llmRequest, false);
    }

    /// @brief Terminate request in the next forwardSync call that includes the request.
    virtual void terminateRequestSync(LlmRequestPtr const& llmRequest, FinishReason finishReason) = 0;

    /// @brief Function that synchronizes the decoder
    virtual void forwardSync() = 0;

    /// @brief Function that tries to advance the active requests
    ///        Depending on resources available, it's possible that not all requests will get advanced
    /// @param activeRequests The list of request to try to advance
    virtual void forwardAsync(batch_manager::RequestList const& activeRequests) = 0;

    /// @brief Override the runtime batch size for the model
    virtual void setRuntimeBatchSize(SizeType32 runtimeBatchSize)
    {
        // By default, we ignore the runtimeBatchSize unless the model actively supports it
    }

    /// @brief Get the runtime batch size for the model
    [[nodiscard]] virtual SizeType32 getRuntimeBatchSize() const
    {
        TLLM_CHECK_WITH_INFO(false, "getRuntimeBatchSize is not implemented");
    }

    /// @brieft Override the runtime max num tokens for the model
    virtual void setRuntimeMaxNumTokens(SizeType32 runtimeMaxNumTokens)
    {
        // By default, we ignore the runtimeMaxNumTokens unless the model actively supports it
    }

    virtual void updatePeftCache(LlmRequestPtr const& llmRequest) = 0;

    /// @brief Reset the iteration stats when there are no inflight requests
    virtual void resetIterationStats() = 0;

    [[nodiscard]] virtual SizeType32 getMaxNumSequences() const = 0;
    [[nodiscard]] virtual SizeType32 getMaxInputLen() const = 0;
    [[nodiscard]] virtual SizeType32 getHiddenSize() const = 0;
    [[nodiscard]] virtual SizeType32 getMaxSequenceLen() const = 0;
    [[nodiscard]] virtual SizeType32 getVocabSizePadded() const = 0;
    [[nodiscard]] virtual SizeType32 getMaxDraftLen() const = 0;
    [[nodiscard]] virtual SizeType32 getNumMicroBatches() const = 0;
    [[nodiscard]] virtual SizeType32 getOperatingBeamWidth() const = 0;
    [[nodiscard]] virtual nvinfer1::DataType getLogitDataType() const = 0;
    [[nodiscard]] virtual runtime::WorldConfig const& getWorldConfig() const = 0;
    [[nodiscard]] virtual runtime::ModelConfig const& getModelConfig() const = 0;
    [[nodiscard]] virtual runtime::BufferManager const& getBufferManager() const = 0;
    [[nodiscard]] virtual runtime::BufferManager::CudaStreamPtr getRuntimeStreamPtr() const = 0;
    [[nodiscard]] virtual IterationType getIterCounter() const noexcept = 0;
    [[nodiscard]] virtual bool hasSpeculativeDecodingFastLogits() const noexcept = 0;
    [[nodiscard]] virtual bool getGatherGenerationLogits() const = 0;
    [[nodiscard]] virtual nvinfer1::DataType getTensorDataType(std::string const& name) const = 0;
    [[nodiscard]] virtual nvinfer1::Dims getTensorShape(std::string const& name) const = 0;

    /// @brief Function that provides per iteration stats specific to a certain model
    /// @param stats The json object to write stats to
    virtual void getCurrentIterationStats(IterationStats& stats) const = 0;

    /// @brief Function that provides per request stats specific to a certain model
    /// @param stats The request stats to be updated
    virtual void getCurrentRequestStats(RequestStatsPerIteration& stats) const = 0;

    [[nodiscard]] virtual DebugTensorsPerIteration getCurrentDebugTensors() const = 0;

    using LogitsPostProcessorBatched = tensorrt_llm::batch_manager::LogitsPostProcessor::LogitsPostProcessorBatched;

    virtual void setLogitsPostProcessorBatched(std::optional<LogitsPostProcessorBatched> logitsPostProcessorBatched)
        = 0;
    virtual void setReplicateLogitsPostProcessor(bool replicateLogitsPostProcessor) = 0;
    [[nodiscard]] virtual bool getReplicateLogitsPostProcessor() const = 0;

    [[nodiscard]] virtual bool hasGuidedDecoder() const noexcept = 0;

    [[nodiscard]] virtual std::shared_ptr<tensorrt_llm::batch_manager::kv_cache_manager::BaseKVCacheManager>
    getKVCacheManager() = 0;
    [[nodiscard]] virtual std::shared_ptr<tensorrt_llm::batch_manager::kv_cache_manager::BaseKVCacheManager const>
    getKVCacheManager() const = 0;

    //! \brief Get the batch size that can fill the kv cache to the maximum capacity give the sequence length
    //! \param seqLen The sequence length
    //! \return The batch size that can fill the kv cache to the maximum capacity. If unsuporrted, return 0.
    [[nodiscard]] virtual SizeType32 getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const = 0;
};

} // namespace tensorrt_llm::executor
