/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/generationOutput.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <NvInferRuntime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tensorrt_llm::batch_manager
{
class TrtGptModelV1;
}

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

class IpcMemory;
class IStatefulGptDecoder;
class NcclCommunicator;
class RuntimeBuffers;
class TllmRuntime;

class GptSession
{
    using KvCacheManager = batch_manager::kv_cache_manager::KVCacheManager;
    using KvCacheConfig = batch_manager::kv_cache_manager::KvCacheConfig;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TokenGeneratedCallback = std::function<void(SizeType step, bool finished)>;

public:
    using LoggerPtr = std::shared_ptr<nvinfer1::ILogger>;

    //! @brief   Configuration for session execution and buffer sizes.
    //!          `generate` may be called with batch size and beam width smaller than the configured parameters.
    //! @details `maxBatchSize` will be divided by the number of micro batches to initialize each batch buffer.
    class Config
    {
    public:
        Config(SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxSequenceLength)
            : maxBatchSize{maxBatchSize}
            , maxBeamWidth{maxBeamWidth}
            , maxSequenceLength{maxSequenceLength}
        {
        }

        SizeType maxBatchSize;
        SizeType maxBeamWidth;
        SizeType maxSequenceLength;
        bool decoderPerRequest{false};
        bool cudaGraphMode{false};
        KvCacheConfig kvCacheConfig{};
        std::optional<SizeType> ctxMicroBatchSize = std::nullopt;
        std::optional<SizeType> genMicroBatchSize = std::nullopt;
    };

    GptSession(Config const& sessionConfig, GptModelConfig const& modelConfig, WorldConfig const& worldConfig,
        void const* engineBuffer, std::size_t engineSize, LoggerPtr logger = nullptr);

    GptSession(Config const& sessionConfig, GptModelConfig const& modelConfig, WorldConfig const& worldConfig,
        std::vector<uint8_t> const& engineBuffer, LoggerPtr logger = nullptr)
        : GptSession(
            sessionConfig, modelConfig, worldConfig, engineBuffer.data(), engineBuffer.size(), std::move(logger))
    {
    }

    GptSession(Config const& sessionConfig, GptModelConfig const& modelConfig, WorldConfig const& worldConfig,
        std::string const& engineFile, LoggerPtr logger = nullptr)
        : GptSession(sessionConfig, modelConfig, worldConfig, utils::loadEngine(engineFile), std::move(logger))
    {
    }

    [[nodiscard]] nvinfer1::ILogger& getLogger() const;

    [[nodiscard]] BufferManager const& getBufferManager() const;

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

    [[nodiscard]] nvinfer1::DataType getLogitDataType() const;

    void generate(GenerationOutput& outputs, GenerationInput const& inputs, SamplingConfig const& samplingConfig);

private:
    [[nodiscard]] bool useCudaGraphs()
    {
        return !mCudaGraphInstances.empty();
    }

    void generateBatched(std::vector<GenerationOutput>& microBatchesOutputs,
        std::vector<GenerationInput> const& microBatchesInputs, SamplingConfig const& samplingConfig,
        TokenGeneratedCallback const& onTokenGenerated);

    void setup(Config const& sessionConfig);

    void createContexts();
    void createBuffers(SizeType numMicroBatches);
    void createDecoders(SizeType batchSize, SizeType beamWidth, SizeType maxAttentionWindow, SizeType sinkTokenLength,
        SizeType maxSequenceLength, nvinfer1::DataType logitsType, bool decoderPerRequest, SizeType numMicroBatches);
    void createKvCacheManager(SizeType batchSize, SizeType beamWidth, SizeType maxAttentionWindow,
        SizeType sinkTokenLength, SizeType maxSequenceLength, KvCacheConfig const& config);
    void createCustomAllReduceWorkspace(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength);

    void executeContextStep(std::vector<GenerationInput> const& microBatchesInputs,
        std::vector<GenerationOutput>& microBatchesOutputs, std::vector<SizeType> const& microBatchOffsets,
        KvCacheManager const* kvCacheManager);
    SizeType executeGenerationStep(SizeType step, std::vector<GenerationInput> const& microBatchesInputs,
        std::vector<GenerationOutput>& microBatchesOutputs, std::vector<SizeType> const& microBatchOffsets,
        KvCacheManager* kvCacheManager, std::vector<bool>& microBatchesFinished);

    //! @brief Execute decoder on last PP rank, receive decoder output on other PP ranks.
    void decoderStepAsync(SizeType decoderStep, SizeType microBatchId);

    //! @brief Synchronize with the decoder and return the `shouldStop` flag.
    bool shouldStopSync(SizeType batchSize, SizeType beamWidth, SizeType microBatchId);

    //! @brief Collect final output ids and log probs on last PP rank and send them to first PP rank.
    //! @details Receives are asynchronous on host, so synchronization is required before access.
    void finalize(SizeType microBatchId);

    void kvCacheAddSequences(SizeType beamWidth, SizeType microBatchId, SizeType firstBatchIdx);

    //! @brief Populate outputIds and return reference to newTokens tensor
    ITensor::SharedPtr initDecoder(ITensor& outputIds, GenerationInput const& inputs, GenerationOutput const& outputs,
        SamplingConfig const& samplingConfig, SizeType microBatchId) const;

    TokenGeneratedCallback createOnTokenGeneratedCallback(GenerationOutput& outputs);

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

        cudaGraphExec_t mInstance;
    };

    class MicroBatchConfig
    {
    public:
        MicroBatchConfig()
            : numCtxBatches{1}
            , numGenBatches{1}
            , ctxBatchSize{0}
            , genBatchSize{0}
        {
        }

        explicit MicroBatchConfig(SizeType maxBatchSize, SizeType pipelineParallelism,
            std::optional<SizeType> genMicroBatchSize, std::optional<SizeType> ctxMicroBatchSize);

        constexpr SizeType numCtxPerGen() const
        {
            return numCtxBatches / numGenBatches;
        }

        //! @details flip-flop between 2 graph instances for each generation batch.
        constexpr SizeType getGenGraphId(SizeType flipFlopId, SizeType generationBatchId) const
        {
            return flipFlopId * numGenBatches + generationBatchId;
        }

        SizeType numCtxBatches;
        SizeType numGenBatches;
        SizeType ctxBatchSize;
        SizeType genBatchSize;
    };

    friend class batch_manager::TrtGptModelV1;

private:
    GptModelConfig const mModelConfig;
    WorldConfig const mWorldConfig;
    int mDevice{-1};
    std::shared_ptr<NcclCommunicator> mPipelineComm;
    std::shared_ptr<CudaStream> mCommStream;
    CudaEvent mCommEvent{};

    // tensor parallelism with custom allreduce plugin
    ITensor::SharedPtr mCommPtrs;
    std::vector<std::shared_ptr<IpcMemory>> mIpcMemoryHandles;

    SizeType mDecoderMaxSequenceLength{};
    SizeType mDecoderMaxAttentionWindow{};
    SizeType mDecoderSinkTokenLength{};

    LoggerPtr mLogger;
    std::shared_ptr<TllmRuntime> mRuntime;
    std::shared_ptr<KvCacheManager> mKvCacheManager;

    MicroBatchConfig mMicroBatchConfig;
    // for each micro batch
    std::vector<std::shared_ptr<IStatefulGptDecoder>> mDecoders;
    std::vector<std::shared_ptr<RuntimeBuffers>> mBuffers;
    std::vector<CudaEvent> mReceivedEvents;

    bool mCudaGraphMode{false};
    // ping-pong instances
    std::vector<CudaGraphExecutor> mCudaGraphInstances;
};

} // namespace tensorrt_llm::runtime
