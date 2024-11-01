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
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/common.h"

namespace tensorrt_llm::layers
{

//! \brief LookaheadDecodingLayer
template <typename T>
class LookaheadDecodingLayer : public BaseLayer
{
public:
    using Base = BaseLayer;
    using Base::mBufferManager;

    LookaheadDecodingLayer(DecoderDomain const& decoderDomain, std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& baseSetupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputParams,
        std::shared_ptr<BaseDecodingInputs> const& inputParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    //! @returns workspace needed for this layer in bytes
    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

private:
    void forwardSyncCPU(std::shared_ptr<LookaheadDecodingOutputs> const& outputs,
        std::shared_ptr<LookaheadDecodingInputs> const& inputs);

private:
    using Base::mDecoderDomain;

    size_t mWorkspaceSize{};
    size_t mSetupWorkspaceSize{};
    TensorPtr mCurandStatesDevice;
    TensorPtr mTargetTokensDevice;

    struct CpuAlgorithmResources
    {
        explicit CpuAlgorithmResources(DecoderDomain const& decoderDomain);

        std::vector<LookaheadAlgorithm> mAlgos;
        std::vector<TensorPtr> mPrompts;
        TensorPtr mBatchSlots;
        TensorPtr mTargetTokens;
        TensorPtr mTokensPerStep;
        TensorPtr mEndIds;

        TensorPtr mOutputIds;
        TensorPtr mPathsOffsets;
        TensorPtr mPathsOffsetsBatch;
        TensorPtr mNumNewTokens;
        TensorPtr mNumNewTokensCumSum;
        TensorPtr mNewTokens;

        TensorPtr mNextDraftTokens;
        TensorPtr mNextDraftPosIds;
        TensorPtr mNextDraftLengths;
        TensorPtr mSequenceLengths;
        TensorPtr mGenerationLengths;
        TensorPtr mAttentionMask;
        TensorPtr mPackedMask;
        TensorPtr mPositionOffsets;
        TensorPtr mPositionIds;
    };

    std::optional<CpuAlgorithmResources> mCpuAlgo;

    runtime::SizeType32 mGlobalSteps{0};
};

} // namespace tensorrt_llm::layers
