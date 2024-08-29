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
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/lookaheadModule.h"

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

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, BufferConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& baseSetupParams) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputParams,
        std::shared_ptr<BaseDecodingInputs> const& inputParams) override;

    void forwardSync(std::shared_ptr<BaseDecodingOutputs> const& outputParams,
        std::shared_ptr<BaseDecodingInputs> const& inputParams) override;

    //! @returns workspace needed for this layer in bytes
    [[nodiscard]] size_t getWorkspaceSize() const noexcept;

private:
    void forwardSyncCPU(std::shared_ptr<BaseDecodingOutputs> const& outputParams,
        std::shared_ptr<BaseDecodingInputs> const& inputParams);
    void posIdsToMask(TensorPtr mask, TensorConstPtr posIds);

private:
    using Base::mDecoderDomain;

    TensorPtr mCurandStatesDevice;
    TensorPtr mSamplingWorkspaceDevice;
    TensorPtr mTargetTokensDevice;
    TensorPtr mRandomSeedsDevice;
    TensorPtr mSamplingMaskDevice;

    struct CpuAlgorithmResources
    {
        explicit CpuAlgorithmResources(DecoderDomain const& decoderDomain);

        std::vector<LookaheadAlgorithm> mAlgos;
        TensorPtr mBatchSlots;
        TensorPtr mTargetTokens;
        TensorPtr mTokensPerStep;
        TensorPtr mEndIds;

        TensorPtr mOutputIds;
        TensorPtr mPathsOffsets;
        TensorPtr mNumNewTokens;
        TensorPtr mNumNewTokensCumSum;

        TensorPtr mNextDraftTokens;
        TensorPtr mNextDraftPosIds;
        TensorPtr mPackedMasks;
        TensorPtr mSamplingMask;
        TensorPtr mNextDraftLengths;
        TensorPtr mSequenceLengths;
    };

    std::optional<CpuAlgorithmResources> mCpuAlgo;
};

} // namespace tensorrt_llm::layers
