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

#include "tensorrt_llm/common/cudaAllocator.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/decodingInput.h"
#include "tensorrt_llm/runtime/decodingOutput.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include <curand_kernel.h>

#include <cstdint>
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
    virtual ~IGptDecoder() = default;

    virtual void setup(SamplingConfig const& samplingConfig, size_t batchSize, SizeType maxSequenceLength) = 0;

    virtual bool forward(DecodingOutput& output, DecodingInput const& input) = 0;

    virtual void forwardAsync(DecodingOutput& output, DecodingInput const& input) = 0;

    virtual void gatherTree(ITensor& finalOutputIds, DecodingOutput const& decodingOutput,
        DecodingInput const& decodingInput, BufferManager const& manager)
        = 0;

    virtual const SamplingConfig& getSamplingConfig() = 0;

    static void acceptDraftTokensByIds(const ITensor& targetTokenIds, const ITensor& draftTokenIds,
        const ITensor& contextLengths, const ITensor& numDraftTokens, ITensor& sequenceLengths,
        const ITensor& finishedVec, ITensor& finishedFinal, ITensor& finishedSum,
        BufferManager::CudaStreamPtr const& stream);

    static void acceptDraftTokensByLogits(ITensor& draftLogits, const ITensor& targetLogits, ITensor& draftProbs,
        ITensor& targetProbs, const ITensor& numDraftTokens, ITensor& finished, SizeType vocabSize,
        SizeType vocabSizePadded, bool useRandomAcceptThreshold, float randomAcceptThreshold,
        curandState_t* curandState, BufferManager::CudaStreamPtr const& stream);

    static std::unique_ptr<IGptDecoder> create(
        nvinfer1::DataType dtype, size_t vocabSize, size_t vocabSizePadded, BufferManager::CudaStreamPtr const& stream);
};

template <typename T>
class GptDecoder : public virtual IGptDecoder
{

public:
    using CudaStreamPtr = BufferManager::CudaStreamPtr;
    using TensorPtr = std::shared_ptr<ITensor>;

    GptDecoder(size_t vocabSize, size_t vocabSizePadded, CudaStreamPtr const& stream);

    void setup(SamplingConfig const& samplingConfig, size_t batchSize, SizeType maxSequenceLength) override;

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

inline std::unique_ptr<IGptDecoder> IGptDecoder::create(
    nvinfer1::DataType dtype, size_t vocabSize, size_t vocabSizePadded, BufferManager::CudaStreamPtr const& stream)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT: return std::make_unique<GptDecoder<float>>(vocabSize, vocabSizePadded, stream);
    case nvinfer1::DataType::kHALF: return std::make_unique<GptDecoder<half>>(vocabSize, vocabSizePadded, stream);
    default: return nullptr;
    }
}
} // namespace runtime
} // namespace tensorrt_llm
