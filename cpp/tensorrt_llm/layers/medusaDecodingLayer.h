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

#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/decodingMode.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace layers
{

class MedusaSetupParams : public BaseSetupParams
{
public:
    std::optional<std::vector<int32_t>> runtimeTopK;                   // [1] or [batchSize] on cpu
    std::optional<std::vector<std::vector<int32_t>>> runtimeHeadsTopK; // [batchSize, maxMedusaHeads] on cpu
    std::optional<std::vector<uint64_t>> randomSeed;                   // [1] or [batchSize] on cpu
};

class MedusaInputParams : public BaseInputParams
{
public:
    explicit MedusaInputParams(tc::Tensor logits, tc::Tensor endIds)
        : BaseInputParams{0, 0, std::move(endIds)}
        , logits{std::move(logits)}
    {
    }

    tc::Tensor logits;                                 // [maxBatchSize, beamWidth, vocabSizePadded]

    tc::Tensor paths;                                  // [maxBatchSize, maxTokensPerStep, maxNumHeads + 1] on gpu
    std::vector<std::vector<tc::Tensor>> medusaLogits; // [maxBatchSize][maxNumHeads][tokensPerStep, vocabSize] on gpu
    tc::Tensor medusaCurTokensPerStep;                 // [maxBatchSize] on gpu
    tc::Tensor medusaTargetTokensPerStep;              // [maxBatchSize] on gpu
    tc::Tensor treeIds;                                // [maxBatchSize, maxTokensPerStep] on gpu
};

class MedusaOutputParams : public BaseOutputParams
{
public:
    explicit MedusaOutputParams(tc::Tensor outputIds)
        : BaseOutputParams{std::move(outputIds)}
    {
    }
};

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

    void setup(runtime::SizeType batchSize, runtime::SizeType beamWidth, runtime::SizeType const* batchSlots,
        std::shared_ptr<BaseSetupParams> setupParams) override;

    void forward(std::shared_ptr<BaseOutputParams> outputs, std::shared_ptr<BaseInputParams> inputs) override;

private:
    void allocateBuffer();
    void freeBuffer();

    void samplePrimeHeadTokens(
        std::shared_ptr<MedusaOutputParams> const& outputs, std::shared_ptr<MedusaInputParams> const& inputs);
    void acceptDraftTokens(
        std::shared_ptr<MedusaOutputParams> const& outputs, std::shared_ptr<MedusaInputParams> const& inputs);
    void sampleNewDraftTokens(
        std::shared_ptr<MedusaOutputParams> const& outputs, std::shared_ptr<MedusaInputParams> const& inputs);
    void scatterNewDraftTokens(
        std::shared_ptr<MedusaOutputParams> const& outputs, std::shared_ptr<MedusaInputParams> const& inputs);
    void packAcceptedPaths(
        std::shared_ptr<MedusaOutputParams> const& outputs, std::shared_ptr<MedusaInputParams> const& inputs);

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

} // namespace layers
} // namespace tensorrt_llm
