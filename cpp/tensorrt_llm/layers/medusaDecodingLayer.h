/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/common.h"

namespace tensorrt_llm::layers
{

//! \brief
template <typename T>
class MedusaDecodingLayer : public BaseLayer
{
public:
    using Base = BaseLayer;
    using PathsVec = std::vector<std::vector<std::vector<runtime::SizeType32>>>;

    MedusaDecodingLayer(DecoderDomain const& decoderDomain, std::shared_ptr<runtime::BufferManager> bufferManager);

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, TensorConstPtr batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace) override;

    //! @returns workspace needed for this layer in bytes
    [[nodiscard]] size_t getWorkspaceSize() const noexcept override;

private:
    void allocateBuffer();

    void samplePrimeHeadTokens(SpeculativeDecodingOutputs const& outputs, MedusaDecodingInputs const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
    void acceptDraftTokens(SpeculativeDecodingOutputs const& outputs, MedusaDecodingInputs const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
    void sampleNewDraftTokens(SpeculativeDecodingOutputs const& outputs, MedusaDecodingInputs const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);
    void scatterNewDraftTokens(SpeculativeDecodingOutputs const& outputs, MedusaDecodingInputs const& inputs);
    void packAcceptedPaths(SpeculativeDecodingOutputs const& outputs, MedusaDecodingInputs const& inputs,
        std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace);

private:
    using Base::mDecoderDomain;

    size_t mWorkspaceSize{0};
    size_t mSetupWorkspaceSize{0};
    runtime::SizeType32 mRuntimeMaxTopK{0};
    runtime::SizeType32 mRuntimeMaxTopKPerRequestPerMedusaHead{0};

    TensorPtr mCurandStatesDevice;
    TensorPtr mRuntimeTopKDevice;
    TensorPtr mTargetTokensDevice;
    TensorPtr mRandomSeedsDevice;
    TensorPtr mMedusaSelectedLogitsPtrsDevice;
    TensorPtr mCurandStatesMedusaLogitsDevice;
    TensorPtr mRuntimeTopKPerRequestPerMedusaHeadDevice;
    TensorPtr mNewDraftTokensDevice;
    TensorPtr mBestPathIdsDevice;

    TensorPtr mTiledBatchSlotsSetup;
    TensorPtr mTiledBatchSlotsForward;
    TensorPtr mDraftIdsPtrHost;
    TensorPtr mMedusaInputLogitsPtrs;

    std::vector<runtime::SizeType32> mCummulativeTopK;
};

} // namespace tensorrt_llm::layers
