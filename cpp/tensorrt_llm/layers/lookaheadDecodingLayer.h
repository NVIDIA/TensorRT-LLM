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

#include "lookaheadAlgorithm.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/lookaheadAlgorithm.h"
#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/lookaheadModule.h"

#include <cstddef>
#include <memory>
#include <tuple>

namespace tensorrt_llm::layers
{

using SizeType32 = tensorrt_llm::runtime::SizeType32;

//! \brief LookaheadDecodingLayer
template <typename T>
class LookaheadDecodingLayer : public BaseLayer
{
public:
    using Base = BaseLayer;
    using Base::mBufferManager;

    LookaheadDecodingLayer(DecoderDomain const& decoderDomain, std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& baseSetupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputParams,
        std::shared_ptr<BaseDecodingInputs> const& inputParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    //! @returns workspace needed for this layer in bytes
    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

    void print(char const* functionName, int const lineNumber) const noexcept;

private:
    void forwardSyncCPU(std::shared_ptr<LookaheadDecodingOutputs> const& outputs,
        std::shared_ptr<LookaheadDecodingInputs> const& inputs);

private:
    using Base::mDecoderDomain;

    size_t mWorkspaceSize{};
    size_t mSetupWorkspaceSize{};
    TensorPtr mCurandStatesDevice; // [maxBatchSize, sizeof(curandState_t)], on gpu, int8
    TensorPtr mTargetTokensDevice; // [maxBatchSize, maxTokensPerStep], on gpu, int32

    struct CpuAlgorithmResources
    {
        explicit CpuAlgorithmResources(DecoderDomain const& decoderDomain);

        std::vector<LookaheadAlgorithm> mAlgos; // [maxBatchSize]
        std::vector<TensorPtr> mPrompts;        // [maxBatchSize]

        // maxNumNewTokens = mNgramSize
        // maxAcceptedDraftLen = mNgramSize - 1
        // "on cpu, int32" for no-extra comment
        TensorPtr mBatchSlots;         // [maxBatchSize]
        TensorPtr mTargetTokens;       // [maxBatchSize, maxTokensPerStep]
        TensorPtr mTokensPerStep;      // [maxBatchSize]
        TensorPtr mEndIds;             // [maxBatchSize]
        TensorPtr mOutputIds;          // [maxBatchSize, maxNumNewTokens]
        TensorPtr mPathsOffsets;       // [maxBatchSize, maxAcceptedDraftLen]
        TensorPtr mPathsOffsetsBatch;  // [maxBatchSize, maxAcceptedDraftLen]
        TensorPtr mNumNewTokens;       // [maxBatchSize]
        TensorPtr mNumNewTokensCumSum; // [maxBatchSize + 1]
        TensorPtr mNewTokens;          // [maxTokensPerStep, maxBatchSize, beamWidth]
        TensorPtr mNextDraftTokens;    // [maxBatchSize, maxDraftLen]
        TensorPtr mNextDraftPosIds;    // [maxBatchSize, maxDraftLen]
        TensorPtr mNextDraftLengths;   // [maxBatchSize]
        TensorPtr mSequenceLengths;    // [maxBatchSize]
        TensorPtr mGenerationLengths;  // [maxBatchSize]
        TensorPtr mAttentionMask;      // [maxTokensPerStep, maxTokensPerStep], bool
        TensorPtr mPackedMask;         // [maxBatchSize, maxTokensPerStep, maxTokensPerStep//32]
        TensorPtr mPositionOffsets;    // [maxBatchSize, maxTokensPerStep]
        TensorPtr mPositionIds;        // [maxBatchSize, maxTokensPerStep]
    };

    std::optional<CpuAlgorithmResources> mCpuAlgo;

    SizeType32 mGlobalSteps{0};
};

inline void initAttentionMask(TensorPtr const& mask, std::shared_ptr<runtime::BufferManager>& bufferManager)
{
    bufferManager->setZero(*mask);
    BufferLocation<bool> maskLocation(*mask);
    auto maskShape = mask->getShape();
    for (auto i = 0; i < maskShape.d[0]; i++)
    {
        maskLocation.at(i, 0) = true;
    }
}

inline void convertBoolToInt32(TensorPtr const& dst, TensorConstPtr const& src)
{
    auto dstShape = dst->getShape();
    auto srcShape = src->getShape();
    TLLM_CHECK(dstShape.d[0] == srcShape.d[0]);
    TLLM_CHECK(dstShape.d[1] * 32 >= srcShape.d[1]);
    BufferLocation<SizeType32> dstLocation(*dst);
    BufferLocation<bool const> srcLocation(*src);

    auto setBit = [](SizeType32& x, SizeType32 idx, bool value) { x |= (value << idx); };
    for (auto i = 0; i < srcShape.d[0]; i++)
    {
        for (auto j = 0; j < srcShape.d[1]; j++)
        {
            setBit(dstLocation.at(i, j / 32), j % 32, srcLocation.at(i, j));
        }
    }
}

} // namespace tensorrt_llm::layers
