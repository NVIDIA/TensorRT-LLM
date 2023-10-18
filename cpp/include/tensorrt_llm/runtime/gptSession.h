/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/generationOutput.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/ipcUtils.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <NvInferRuntime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class KVCacheManager;
}

namespace tensorrt_llm::runtime
{

namespace utils
{
std::vector<uint8_t> loadEngine(std::string const& enginePath);
}

class TllmRuntime;
class IStatefulGptDecoder;
class NcclCommunicator;
class RuntimeBuffers;

class GptSession
{
public:
    using LoggerPtr = std::shared_ptr<nvinfer1::ILogger>;

    GptSession(GptModelConfig const& modelConfig, WorldConfig const& worldConfig, void const* engineBuffer,
        std::size_t engineSize, LoggerPtr logger = nullptr);

    GptSession(GptModelConfig const& modelConfig, WorldConfig const& worldConfig,
        std::vector<uint8_t> const& engineBuffer, LoggerPtr logger = nullptr)
        : GptSession(modelConfig, worldConfig, engineBuffer.data(), engineBuffer.size(), logger)
    {
    }

    GptSession(GptModelConfig const& modelConfig, WorldConfig const& worldConfig, std::string const& engineFile,
        LoggerPtr logger = nullptr)
        : GptSession(modelConfig, worldConfig, utils::loadEngine(engineFile), logger)
    {
    }

    [[nodiscard]] nvinfer1::ILogger& getLogger() const;

    [[nodiscard]] BufferManager& getBufferManager() const;

    [[nodiscard]] GptModelConfig const& getModelConfig() const
    {
        return mModelConfig;
    }

    [[nodiscard]] WorldConfig const& getWorldConfig() const
    {
        return mWorldConfig;
    }

    [[nodiscard]] int getDevice() const noexcept
    {
        return mDevice;
    }

    [[nodiscard]] bool isCudaGraphMode() const noexcept
    {
        return mCudaGraphMode;
    }

    void setCudaGraphMode(bool value)
    {
        mCudaGraphMode = value;
    }

    //! @brief   Initialize buffers for the given sizes.
    //!          `generate` may be called with batch size and beam width smaller than the setup parameters.
    //! @details `maxBatchSize` will be devided by the number of micro batches to initialize each batch buffer.
    void setup(SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxSequenceLength, bool decoderPerRequest,
        std::optional<SizeType> maxTokensInPagedKvCache = std::nullopt,
        std::optional<SizeType> numMicroBatches = std::nullopt);

    void generate(GenerationOutput& outputs, GenerationInput const& inputs, SamplingConfig const& samplingConfig)
    {
        if (mNumMicroBatches == 1)
            generateSingleBatch(outputs, inputs, samplingConfig);
        else
            generateMultiBatch(outputs, inputs, samplingConfig);
    }

private:
    void generateSingleBatch(
        GenerationOutput& outputs, GenerationInput const& inputs, SamplingConfig const& samplingConfig);

    void generateMultiBatch(
        GenerationOutput& outputs, GenerationInput const& inputs, SamplingConfig const& samplingConfig);

    using KvCacheManager = batch_manager::kv_cache_manager::KVCacheManager;

    void createContexts(SizeType numMicroBatches);
    void createBuffers(SizeType numMicroBatches);
    void createDecoders(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength,
        nvinfer1::DataType logitsType, bool decoderPerRequest, SizeType numMicroBatches);
    void createKvCacheManagers(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength,
        SizeType numMicroBatches, std::optional<SizeType> maxTokensInPagedKvCache);
    void createCustomAllReduceWorkspace(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength);

    //! @brief Execute decoder on last PP rank, receive decoder output on other PP ranks.
    void decoderStepAsync(
        ITensor::SharedPtr& outputIds, ITensor::SharedPtr& newTokens, SizeType decoderStep, SizeType microBatchId);

    //! @brief Synchronize with the decoder and return the `shouldStop` flag.
    bool shouldStopSync(SizeType batchSize, SizeType beamWidth, SizeType microBatchId);

    //! @brief Collect final output ids on last PP rank and send them to first PP rank.
    //! @details Receives are asynchronous on host, so synchronization is required before access.
    void finalizeOutputIds(ITensor& outputIds, SizeType microBatchId);

    void kvCacheAddSequences(SizeType beamWidth, SizeType microBatchId);

    ITensor::SharedPtr initNewTokens(
        GenerationInput const& inputs, SamplingConfig const& samplingConfig, SizeType microBatchId);

    class CudaGraphExecutor
    {
    public:
        CudaGraphExecutor() = default;

        ~CudaGraphExecutor()
        {
            try
            {
                clear();
            }
            catch (std::exception& e)
            {
                TLLM_LOG_EXCEPTION(e);
            }
        }

        bool hasInstance()
        {
            return mInstance != nullptr;
        }

        void clear();
        void prepareNextGraph(TllmRuntime const& runtime, SizeType nextContextId);
        void launch(CudaStream const& stream);

    private:
        void create(cudaGraph_t const& graph);
        bool update(cudaGraph_t const& graph);
        void uploadToStream(CudaStream const& stream);

        using cudaGraphExecPtr = cudaGraphExec_t;
        cudaGraphExecPtr mInstance;
    };

private:
    GptModelConfig const mModelConfig;
    WorldConfig const mWorldConfig;
    int mDevice{-1};
    std::shared_ptr<NcclCommunicator> mPipelineComm;
    std::shared_ptr<CudaStream> mCommStream;
    CudaEvent mCommEvent{};

    SizeType mDecoderMaxSequenceLength{};

    LoggerPtr mLogger;
    std::shared_ptr<TllmRuntime> mRuntime;

    SizeType mNumMicroBatches;
    // for each micro batch
    std::vector<std::shared_ptr<IStatefulGptDecoder>> mDecoders;
    std::vector<std::shared_ptr<RuntimeBuffers>> mBuffers;
    std::vector<std::shared_ptr<KvCacheManager>> mKvCacheManagers;
    std::vector<CudaEvent> mReceivedEvents;

    bool mCudaGraphMode{false};
    // ping-pong instances
    std::array<CudaGraphExecutor, 2> mCudaGraphInstances;
};

} // namespace tensorrt_llm::runtime
