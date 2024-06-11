/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <curand_kernel.h>

#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace layers
{

class ExplicitDraftTokensSetupParams : public BaseSetupParams
{
public:
    std::optional<std::vector<float>> temperature;   // [setupBatchSize] on cpu
    std::optional<std::vector<uint64_t>> randomSeed; // [1] or [setupBatchSize] on cpu
};

class ExplicitDraftTokensInputParams : public BaseInputParams
{
public:
    explicit ExplicitDraftTokensInputParams()
        : BaseInputParams{0, 0, tc::Tensor()}
    {
    }

    //! Draft tokens for the next iteration. The first token in each path is the last accepted token at current
    //! iteration. E.g. if forwardBatchSize == 1, maxNumPaths == 2, maxPathLen== 3, [[[0, 1, 2], [0, 1, 10]]]
    tc::Tensor nextDraftTokens; // [forwardBatchSize, maxNumPaths, maxPathLen], gpu
    //! Compressed form of `nextDraftTokens`, where common prefixes and collapsed.
    //! Using example above [0, 1, 2, 10]
    tc::Tensor nextFlatTokens; // [forwardBatchSize * maxDecodingTokens], gpu
    //! Indices of draft tokens in the compressed `nextFlatTokens` for the next iteration.
    //! Using example above, [[[0, 1, 2], [0, 1, 3]]]
    tc::Tensor nextDraftIndices; // [forwardBatchSize, maxNumPaths, maxPathLen], gpu
    //! Probabilities of the next draft tokens.
    tc::Tensor nextDraftProbs; // [forwardBatchSize, maxNumPaths, maxDraftPathLen, vocabSize], gpu
    //! Same as `nextDraftTokens`, but for current iteration.
    //! Current accepted tokens obtained as `lastDraftTokens[bi][bestPathIndices[bi]][1:bestPathLengths[bi]]`.
    tc::Tensor lastDraftTokens; // [forwardBatchSize, maxNumPaths, maxPathLen], gpu
    //! Same as `nextDraftIndices`, but for current iteration.
    tc::Tensor lastDraftIndices; // [forwardBatchSize, maxNumPaths, maxPathLen], gpu
    //! Boolean attention masks.
    //! maxDecodingTokens' = specDecodingGenerationLengths.max()
    tc::Tensor masks; // [forwardBatchSize, maxDecodingTokens', maxDecodingTokens'], gpu
    //! Relative to `positionIdsBase` position ids. Same as `nextFlatTokens` for next draft indices.
    //! Using example above, [0, 1, 2, 3]
    tc::Tensor packedPosIds; // [forwardBatchSize * maxDecodingTokens], gpu
    //! Lengths of the accepted paths for each request. It is 1 for context phase (Only 1 primary tokens is accepted).
    tc::Tensor bestPathLengths; // [forwardBatchSize], gpu
    //! Indices of the accepted paths for each request. It is 0 for context phase.
    tc::Tensor bestPathIndices; // [forwardBatchSize], gpu
    //! Number of the draft tokens for the next iteration.
    tc::Tensor specDecodingGenerationLengths; // [forwardBatchSize], gpu
    //! Baseline for the position ids.
    tc::Tensor positionIdsBase; // [forwardBatchSize], gpu
};

//! \brief Decoding layer for speculative decoding technique, when all tokens are generated, decoded and accepted in the
//! engine.
template <typename T>
class ExplicitDraftTokensLayer : public BaseLayer
{
public:
    using Base = BaseLayer;
    using PathsVec = std::vector<std::vector<std::vector<runtime::SizeType32>>>;

    ExplicitDraftTokensLayer(DecoderDomain const& decoderDomain, cudaStream_t stream,
        std::shared_ptr<tensorrt_llm::common::IAllocator> allocator);

    ~ExplicitDraftTokensLayer() override;

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, runtime::SizeType32 const* batchSlots,
        std::shared_ptr<BaseSetupParams> setupParams) override;

    void forwardAsync(std::shared_ptr<BaseOutputParams> outputs, std::shared_ptr<BaseInputParams> inputs) override;

private:
    void allocateBuffer();
    void freeBuffer();

    void convertPackedMask(DynamicDecodeOutputParams const& outputs, ExplicitDraftTokensInputParams const& inputs);

    void splitInputDataToBatchSlots(
        DynamicDecodeOutputParams const& outputs, ExplicitDraftTokensInputParams const& inputs);

    void packAcceptedPaths(DynamicDecodeOutputParams const& outputs, ExplicitDraftTokensInputParams const& inputs);

private:
    using Base::mStream;
    using Base::mAllocator;
    using Base::mWorkspaceSize;

    using Base::mDecoderDomain;

    SizeType32 mNumPaths;
    SizeType32 mMaxPathLength;

    size_t mWorkspaceSizeInBytes{0};
    size_t mScanWorkspaceSizeInBytes{0};
    size_t mReduceWorkspaceSizeInBytes{0};

    uint64_t* mRandomSeedsDevice{nullptr};
    curandState_t* mCurandStatesDevice{nullptr};
    void* mWorkspaceDevice{nullptr};
    SizeType32* mGenerationLengthInclusiveSum{nullptr};
    SizeType32* mMaxGenerationLength{nullptr};
    float* mTemperatureDevice{nullptr};

    std::vector<float> mTemperature;
};

} // namespace layers
} // namespace tensorrt_llm
