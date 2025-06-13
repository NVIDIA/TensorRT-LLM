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

#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "trtGptModel.h"

#include <NvInferRuntime.h>

namespace tensorrt_llm::runtime
{
class TllmRuntime;
class NcclCommunicator;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::batch_manager
{
class CapacityScheduler;
class MicroBatchScheduler;
class EncoderBuffers;

class TrtEncoderModel : public TrtGptModel
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TokenIdType = tensorrt_llm::runtime::TokenIdType;
    using BufferManager = tensorrt_llm::runtime::BufferManager;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;
    using TensorPtr = runtime::ITensor::SharedPtr;

    TrtEncoderModel(runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        runtime::RawEngine const& rawEngine, std::shared_ptr<nvinfer1::ILogger> logger,
        executor::ExecutorConfig const& executorConfig);

    ~TrtEncoderModel() override;

    void terminateRequest(std::shared_ptr<LlmRequest> const& llmRequest, bool pause = false) override;
    void terminateRequestSync(
        std::shared_ptr<LlmRequest> const& llmRequest, executor::FinishReason finishReason) override;

    void forward(RequestVector& activeRequests);

    void forwardSync() override;

    void forwardAsync(RequestList const& activeRequests) override;

    [[nodiscard]] runtime::BufferManager const& getBufferManager() const override;
    [[nodiscard]] runtime::BufferManager::CudaStreamPtr getRuntimeStreamPtr() const override;

    runtime::ModelConfig const& getModelConfig() const override
    {
        return mModelConfig;
    }

    [[nodiscard]] bool getGatherGenerationLogits() const override
    {
        return getModelConfig().computeGenerationLogits();
    }

    runtime::WorldConfig const& getWorldConfig() const override
    {
        return mWorldConfig;
    }

    [[nodiscard]] SizeType32 getHiddenSize() const override
    {
        return mHiddenSize;
    }

    [[nodiscard]] SizeType32 getMaxInputLen() const override
    {
        return mMaxInputLen;
    }

    [[nodiscard]] SizeType32 getNumMicroBatches() const override
    {
        return mNumMicroBatches;
    }

    [[nodiscard]] nvinfer1::DataType getLogitDataType() const override
    {
        return getModelConfig().getDataType();
    }

    nvinfer1::DataType getTensorDataType(std::string const& name) const override;
    nvinfer1::Dims getTensorShape(std::string const& name) const override;

    [[nodiscard]] TrtGptModelType getModelType() const override
    {
        throw std::runtime_error("TrtEncoderModel does not have model type."); // FIXME:
    }

    [[nodiscard]] executor::IterationType getIterCounter() const noexcept override
    {
        return mIterCounter;
    }

    void updatePeftCache(std::shared_ptr<LlmRequest> const& /*llmRequest*/) override
    {
        throw std::runtime_error("TrtEncoderModel does not have Peft Cache.");
    }

    void getCurrentIterationStats(executor::IterationStats& stats) const override;
    void getCurrentRequestStats(executor::RequestStatsPerIteration& stats) const override;
    [[nodiscard]] executor::DebugTensorsPerIteration getCurrentDebugTensors() const override;

    void setLayerProfiler() override;
    std::string getLayerProfileInfo() const override;

    void setLogitsPostProcessorBatched(std::optional<LogitsPostProcessorBatched> logitsPostProcessorBatched) override;
    void setReplicateLogitsPostProcessor(bool replicateLogitsPostProcessor) override;
    [[nodiscard]] bool getReplicateLogitsPostProcessor() const override;

    void resetIterationStats() override {}

    [[nodiscard]] SizeType32 getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const override
    {
        return 0;
    };

protected:
    std::shared_ptr<kv_cache_manager::BaseKVCacheManager> getKVCacheManager() override
    {
        throw std::runtime_error("TrtEncoderModel does not have KVCache.");
    }

    [[nodiscard]] std::shared_ptr<kv_cache_manager::BaseKVCacheManager const> getKVCacheManager() const override
    {
        throw std::runtime_error("TrtEncoderModel does not have KVCache.");
    }

    [[nodiscard]] std::shared_ptr<BasePeftCacheManager> getPeftCacheManager() override
    {
        throw std::runtime_error("TrtEncoderModel does not use PEFT.");
    }

    [[nodiscard]] std::shared_ptr<BasePeftCacheManager const> getPeftCacheManager() const override
    {
        throw std::runtime_error("TrtEncoderModel does not use PEFT.");
    }

private:
    [[nodiscard]] SizeType32 getBufferId() const
    {
        return mMicroBatchId;
    }

    void createRuntimeContexts();
    void executeContext(SizeType32 runtimeContextId);
    void createBuffers();
    void executeBatch(RequestVector const& requestList);
    void executeBatch(ScheduledRequests const& scheduledRequests);
    void rearrangeOutputs(ScheduledRequests const& scheduledRequests);
    void createCustomAllReduceWorkspace();
    void fillEncoderOutputSync(RequestVector const& requestList, TensorMap outputTensors);

    runtime::ModelConfig const mModelConfig;
    runtime::WorldConfig const mWorldConfig;
    int mDevice{-1};
    std::shared_ptr<tensorrt_llm::mpi::MpiComm> mMpiCommPipelinePara;

    std::shared_ptr<nvinfer1::ILogger> mLogger;
    std::shared_ptr<runtime::TllmRuntime> mRuntime;

    SizeType32 mMicroBatchId{0};

    // TODO: Add runtime buffers for async PP
    std::vector<std::shared_ptr<EncoderBuffers>> mBuffers;

    SizeType32 mNumMicroBatches;
    SizeType32 mNumBuffers;

    std::vector<ScheduledRequests> mMicroBatchScheduledRequests;
    ReqIdsSet mInflightReqIds;
    ReqIdsSet mReqIdsToPause;

    std::unique_ptr<tensorrt_llm::batch_manager::CapacityScheduler const> mCapacityScheduler;
    std::unique_ptr<tensorrt_llm::batch_manager::MicroBatchScheduler const> mMicroBatchScheduler;

    SizeType32 mHiddenSize;  // already divided by Tensor Parallelism
    SizeType32 mMaxInputLen; // WAR for max_input_len == max_seq_len at all circumstances

    runtime::BufferManager mCopyBufferManager;

    // Iteration counter used to distinguish debug output
    executor::IterationType mIterCounter{0};
};

} // namespace tensorrt_llm::batch_manager
