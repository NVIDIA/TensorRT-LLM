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

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/decodingInput.h"
#include "tensorrt_llm/runtime/decodingMode.h"
#include "tensorrt_llm/runtime/decodingOutput.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include <curand_kernel.h>

#include <memory>

#include <NvInferRuntime.h>

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

class IGptDecoder
{
public:
    using TensorPtr = std::shared_ptr<ITensor>;

    virtual ~IGptDecoder() = default;

    virtual void setup(SamplingConfig const& samplingConfig, size_t batchSize, SizeType maxSequenceLength,
        std::optional<TensorPtr> const& batchSlots = std::nullopt)
        = 0;

    virtual bool forward(DecodingOutput& output, DecodingInput const& input) = 0;

    virtual void forwardAsync(DecodingOutput& output, DecodingInput const& input) = 0;

    virtual void gatherTree(ITensor& finalOutputIds, DecodingOutput const& decodingOutput,
        DecodingInput const& decodingInput, BufferManager const& manager)
        = 0;

    virtual const SamplingConfig& getSamplingConfig() = 0;

    static void acceptDraftTokensByIds(ITensor const& targetTokenIds, ITensor const& draftTokenIds,
        ITensor const& contextLengths, ITensor const& numDraftTokens, ITensor& sequenceLengths,
        ITensor const& finishedVec, ITensor& finishedFinal, ITensor& finishedSum, ITensor const& batchSlots,
        BufferManager::CudaStreamPtr const& stream);

    static void acceptDraftTokensByLogits(ITensor& draftLogits, ITensor const& targetLogits, ITensor& draftProbs,
        ITensor& targetProbs, ITensor const& numDraftTokens, ITensor& finished, ITensor const& batchSlots,
        SizeType vocabSize, SizeType vocabSizePadded, bool useRandomAcceptThreshold, float randomAcceptThreshold,
        curandState_t* curandState, BufferManager::CudaStreamPtr const& stream);

    static std::unique_ptr<IGptDecoder> create(DecodingMode const& mode, nvinfer1::DataType dtype, size_t maxBatchSize,
        size_t maxBeamWidth, size_t vocabSize, size_t vocabSizePadded, size_t maxSequenceLength,
        BufferManager::CudaStreamPtr const& stream);
};

template <typename T>
class GptDecoder : public virtual IGptDecoder
{

public:
    using CudaStreamPtr = BufferManager::CudaStreamPtr;
    using TensorPtr = std::shared_ptr<ITensor>;

    GptDecoder(DecodingMode const& mode, size_t maxBatchSize, size_t maxBeamWidth, size_t vocabSize,
        size_t vocabSizePadded, size_t maxSequenceLength, CudaStreamPtr const& stream);

    void setup(SamplingConfig const& samplingConfig, size_t batchSize, SizeType maxSequenceLength,
        std::optional<TensorPtr> const& batchSlots = std::nullopt) override;

    bool forward(DecodingOutput& output, DecodingInput const& input) override;

    void forwardAsync(DecodingOutput& output, DecodingInput const& input) override;

    void gatherTree(ITensor& finalOutputIds, DecodingOutput const& decodingOutput, DecodingInput const& decodingInput,
        BufferManager const& manager) override;

    const SamplingConfig& getSamplingConfig() override
    {
        return mSamplingConfig;
    }

private:
    BufferManager mManager;
    std::shared_ptr<tensorrt_llm::layers::DynamicDecodeLayer<T>> mDynamicDecodeLayer;

    TensorPtr mLogProbsTiled; // Buffer used to store the transpose of the logProbs. Needed because the kernels have
                              // been written to use that shape.
    SamplingConfig mSamplingConfig;
};

inline std::unique_ptr<IGptDecoder> IGptDecoder::create(DecodingMode const& mode, nvinfer1::DataType dtype,
    size_t maxBatchSize, size_t maxBeamWidth, size_t vocabSize, size_t vocabSizePadded, size_t maxSequenceLength,
    BufferManager::CudaStreamPtr const& stream)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT:
        return std::make_unique<GptDecoder<float>>(
            mode, maxBatchSize, maxBeamWidth, vocabSize, vocabSizePadded, maxSequenceLength, stream);
    case nvinfer1::DataType::kHALF:
        return std::make_unique<GptDecoder<half>>(
            mode, maxBatchSize, maxBeamWidth, vocabSize, vocabSizePadded, maxSequenceLength, stream);
    default: return nullptr;
    }
}
} // namespace runtime
} // namespace tensorrt_llm
