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

#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/decodingInput.h"
#include "tensorrt_llm/runtime/decodingOutput.h"
#include "tensorrt_llm/runtime/samplingConfig.h"

#include <NvInferRuntime.h>
#include <curand_kernel.h>

#include <memory>

namespace tensorrt_llm
{

namespace layers
{
// Forward declaration
template <typename T>
class DynamicDecodeLayer;
} // namespace layers

namespace runtime
{

class SpeculativeDecodingModule;

class DecodingLayerWorkspace;

class IGptDecoder
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorConstPtr = runtime::ITensor::SharedConstPtr;

    virtual ~IGptDecoder() = default;

    /// @param explicitDraftTokensDType is only used by ExplicitDraftTokens model to WAR the lack of bf16 decoder.
    virtual void setup(SamplingConfig const& samplingConfig, size_t batchSize, TensorConstPtr const& batchSlots,
        std::optional<DecodingOutput> const& output = std::nullopt,
        std::optional<nvinfer1::DataType> explicitDraftTokensDType = std::nullopt,
        std::optional<std::vector<TensorConstPtr>> const& lookaheadPrompt = std::nullopt,
        std::optional<std::vector<executor::LookaheadDecodingConfig>> const& lookaheadAlgoConfigs = std::nullopt)
        = 0;

    virtual void forwardAsync(DecodingOutput& output, DecodingInput const& input) = 0;

    virtual void forwardSync(DecodingOutput& output, DecodingInput const& input) = 0;

    virtual SamplingConfig const& getSamplingConfig() = 0;

    virtual void disableLookahead(
        std::optional<SamplingConfig> const& samplingConfig, SizeType32 batchSize, TensorConstPtr batchSlots)
        = 0;

    static std::unique_ptr<IGptDecoder> create(executor::DecodingMode const& mode, nvinfer1::DataType dtype,
        size_t maxNumSequences, size_t maxBeamWidth, size_t vocabSize, size_t vocabSizePadded,
        BufferManager::CudaStreamPtr const& stream,
        std::shared_ptr<SpeculativeDecodingModule const> const& speculativeDecodingModule = nullptr);
};

template <typename T>
class GptDecoder : public virtual IGptDecoder
{

public:
    using CudaStreamPtr = BufferManager::CudaStreamPtr;
    using TensorPtr = std::shared_ptr<ITensor>;

    GptDecoder(executor::DecodingMode const& mode, size_t maxNumSequences, size_t maxBeamWidth, size_t vocabSize,
        size_t vocabSizePadded, CudaStreamPtr const& stream,
        std::shared_ptr<SpeculativeDecodingModule const> speculativeDecodingModule = nullptr);

    void setup(SamplingConfig const& samplingConfig, size_t batchSize, TensorConstPtr const& batchSlots,
        std::optional<DecodingOutput> const& output = std::nullopt,
        std::optional<nvinfer1::DataType> explicitDraftTokensDType = std::nullopt,
        std::optional<std::vector<TensorConstPtr>> const& lookaheadPrompt = std::nullopt,
        std::optional<std::vector<executor::LookaheadDecodingConfig>> const& lookaheadAlgoConfigs
        = std::nullopt) override;

    void forwardAsync(DecodingOutput& output, DecodingInput const& input) override;

    void forwardSync(DecodingOutput& output, DecodingInput const& input) override;

    SamplingConfig const& getSamplingConfig() override
    {
        return mSamplingConfig;
    }

    void disableLookahead(
        std::optional<SamplingConfig> const& samplingConfig, SizeType32 batchSize, TensorConstPtr batchSlots) override;

private:
    std::shared_ptr<BufferManager> mManager;
    std::shared_ptr<tensorrt_llm::layers::DynamicDecodeLayer<T>> mDynamicDecodeLayer;
    std::shared_ptr<tensorrt_llm::runtime::DecodingLayerWorkspace> mDecodingLayerWorkspace;

    SamplingConfig mSamplingConfig;

    size_t mMaxNumSequences;
    size_t mVocabSize;
    size_t mVocabSizePadded;

    executor::DecodingMode mDecodingMode;
};

inline std::unique_ptr<IGptDecoder> IGptDecoder::create(executor::DecodingMode const& mode, nvinfer1::DataType dtype,
    size_t maxNumSequences, size_t maxBeamWidth, size_t vocabSize, size_t vocabSizePadded,
    BufferManager::CudaStreamPtr const& stream,
    std::shared_ptr<SpeculativeDecodingModule const> const& speculativeDecodingModule)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        return std::make_unique<GptDecoder<float>>(
            mode, maxNumSequences, maxBeamWidth, vocabSize, vocabSizePadded, stream, speculativeDecodingModule);
    case nvinfer1::DataType::kHALF:
        return std::make_unique<GptDecoder<half>>(
            mode, maxNumSequences, maxBeamWidth, vocabSize, vocabSizePadded, stream, speculativeDecodingModule);
    default:
        TLLM_THROW("Unsupported decoder data type: %d. Use either kFLOAT or kHALF.", static_cast<int>(dtype));
        return nullptr;
    }
}

/// @brief Helper function to produce batch slots [0, 1, ..., batchSize - 1] for paths that do not explicitly provide
/// batch slots to the decoder.
inline runtime::ITensor::SharedConstPtr getDefaultBatchSlots(runtime::SizeType32 batchSize)
{
    auto defaultBatchSlots = runtime::BufferManager::pinnedPool(
        runtime::ITensor::makeShape({batchSize}), runtime::TRTDataType<runtime::SizeType32>::value);
    auto range = runtime::BufferRange<runtime::SizeType32>(*defaultBatchSlots);
    std::iota(range.begin(), range.end(), 0);
    return defaultBatchSlots;
}
} // namespace runtime
} // namespace tensorrt_llm
