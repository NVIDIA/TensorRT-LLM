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

//! \brief
template <typename T>
class MedusaDecodingLayer : public BaseLayer
{
public:
    using Base = BaseLayer;
    using PathsVec = std::vector<std::vector<std::vector<runtime::SizeType>>>;

    class MedusaSetupParams : public DecodingSetupParams
    {
    public:
        std::optional<std::vector<runtime::SizeType>> runtimeTopK; // [1] or [batchSize] on cpu
        std::optional<std::vector<std::vector<runtime::SizeType>>>
            runtimeHeadsTopK;                                      // [batchSize, maxMedusaHeads] on cpu
        std::optional<std::vector<uint64_t>> randomSeed;           // [1] or [batchSize] on cpu
    };

    class MedusaForwardParams : public DecodingParams
    {
    public:
        MedusaForwardParams(tc::Tensor logits, tc::Tensor endIds)
            : DecodingParams{0, 0, std::move(logits), std::move(endIds)}
        {
        }

        tc::Tensor paths;                     // [maxBatchSize, maxTokensPerStep, maxNumHeads + 1] on gpu
        std::vector<std::vector<tc::Tensor>>
            medusaLogits;                     // [maxBatchSize][maxNumHeads][tokensPerStep, vocabSize] on gpu
        tc::Tensor medusaCurTokensPerStep;    // [maxBatchSize] on gpu
        tc::Tensor medusaTargetTokensPerStep; // [maxBatchSize] on gpu
        tc::Tensor treeIds;                   // [maxBatchSize, maxTokensPerStep] on gpu
    };

    MedusaDecodingLayer(runtime::SizeType maxBatchSize, runtime::SizeType vocabSize, runtime::SizeType vocabSizePadded,
        runtime::SizeType maxTokensPerStep, runtime::SizeType maxNumHeads, cudaStream_t stream,
        std::shared_ptr<tensorrt_llm::common::IAllocator> allocator);

    ~MedusaDecodingLayer() override;

    void setup(runtime::SizeType batchSize, runtime::SizeType const* batchSlots, MedusaSetupParams const& setupParams);

    void forward(DecodingOutputParams& outputs, MedusaForwardParams& inputs);

private:
    void allocateBuffer();
    void freeBuffer();

    void samplePrimeHeadTokens(DecodingOutputParams& outputs, MedusaForwardParams& inputs);
    void acceptDraftTokens(DecodingOutputParams& outputs, MedusaForwardParams& inputs);
    void sampleNewDraftTokens(DecodingOutputParams& outputs, MedusaForwardParams& inputs);
    void scatterNewDraftTokens(DecodingOutputParams& outputs, MedusaForwardParams& inputs);
    void packAcceptedPaths(DecodingOutputParams& outputs, MedusaForwardParams& inputs);

private:
    using Base::mStream;
    using Base::mAllocator;

    runtime::SizeType mMaxBatchSize;
    runtime::SizeType mVocabSize;
    runtime::SizeType mVocabSizePadded;

    runtime::SizeType mMaxTokensPerStep;
    runtime::SizeType mMaxNumHeads;

    size_t mSamplingWorkspaceSize;
    runtime::SizeType mRuntimeMaxTopK{0};
    runtime::SizeType mRuntimeMaxTopKPerRequestPerMedusaHead{0};

    curandState_t* mCurandStatesDevice{nullptr};
    void* mSetupWorkspaceDevice{nullptr};
    void* mSamplingWorkspaceDevice{nullptr};
    runtime::SizeType* mRuntimeTopKDevice{nullptr};
    runtime::TokenIdType* mTargetTokensDevice{nullptr};
    uint64_t* mRandomSeedsDevice{nullptr};
    T** mMedusaSelectedLogitsPtrsDevice{nullptr};
    curandState_t* mCurandStatesMedusaLogitsDevice{nullptr};
    runtime::SizeType* mRuntimeTopKPerRequestPerMedusaHeadDevice{nullptr};
    runtime::TokenIdType* mNewDraftTokensDevice{nullptr};
    runtime::SizeType* mBestPathIdsDevice{nullptr};

    runtime::ITensor::UniquePtr mTiledBatchSlotsSetup;
    runtime::ITensor::UniquePtr mTiledBatchSlotsForward;
    runtime::ITensor::UniquePtr mDraftIdsPtrHost;
    runtime::ITensor::UniquePtr mMedusaInputLogitsPtrs;

    std::vector<runtime::SizeType> mCummulativeTopK;
};

} // namespace layers
} // namespace tensorrt_llm
