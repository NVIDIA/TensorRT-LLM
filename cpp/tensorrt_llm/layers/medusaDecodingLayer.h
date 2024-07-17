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
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::layers
{

//! \brief
template <typename T>
class MedusaDecodingLayer : public BaseLayer
{
public:
    using Base = BaseLayer;
    using PathsVec = std::vector<std::vector<std::vector<runtime::SizeType32>>>;

    MedusaDecodingLayer(DecoderDomain const& decoderDomain, cudaStream_t stream,
        std::shared_ptr<tensorrt_llm::common::IAllocator> allocator);

    ~MedusaDecodingLayer() override;

    void setup(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, runtime::SizeType32 const* batchSlots,
        std::shared_ptr<BaseSetupParams> const& setupParams) override;

    void forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
        std::shared_ptr<BaseDecodingInputs> const& inputs) override;

private:
    void allocateBuffer();
    void freeBuffer();

    void samplePrimeHeadTokens(SpeculativeDecodingOutputs const& outputs, MedusaDecodingInputs const& inputs);
    void acceptDraftTokens(SpeculativeDecodingOutputs const& outputs, MedusaDecodingInputs const& inputs);
    void sampleNewDraftTokens(SpeculativeDecodingOutputs const& outputs, MedusaDecodingInputs const& inputs);
    void scatterNewDraftTokens(SpeculativeDecodingOutputs const& outputs, MedusaDecodingInputs const& inputs);
    void packAcceptedPaths(SpeculativeDecodingOutputs const& outputs, MedusaDecodingInputs const& inputs);

private:
    using Base::mStream;
    using Base::mAllocator;
    using Base::mWorkspaceSize;

    using Base::mDecoderDomain;

    runtime::SizeType32 mRuntimeMaxTopK{0};
    runtime::SizeType32 mRuntimeMaxTopKPerRequestPerMedusaHead{0};

    curandState_t* mCurandStatesDevice{nullptr};
    void* mSetupWorkspaceDevice{nullptr};
    void* mSamplingWorkspaceDevice{nullptr};
    runtime::SizeType32* mRuntimeTopKDevice{nullptr};
    runtime::TokenIdType* mTargetTokensDevice{nullptr};
    uint64_t* mRandomSeedsDevice{nullptr};
    T** mMedusaSelectedLogitsPtrsDevice{nullptr};
    curandState_t* mCurandStatesMedusaLogitsDevice{nullptr};
    runtime::SizeType32* mRuntimeTopKPerRequestPerMedusaHeadDevice{nullptr};
    runtime::TokenIdType* mNewDraftTokensDevice{nullptr};
    runtime::SizeType32* mBestPathIdsDevice{nullptr};

    runtime::ITensor::UniquePtr mTiledBatchSlotsSetup;
    runtime::ITensor::UniquePtr mTiledBatchSlotsForward;
    runtime::ITensor::UniquePtr mDraftIdsPtrHost;
    runtime::ITensor::UniquePtr mMedusaInputLogitsPtrs;

    std::vector<runtime::SizeType32> mCummulativeTopK;
};

} // namespace tensorrt_llm::layers
