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

#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/layers/baseSamplingLayer.h"

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace layers
{

//! \brief Layer to randomly sample tokens from TopP logits.
//! Layer expects probs precomputed in "logits" tensor
template <typename T>
class TopPSamplingLayer : public BaseSamplingLayer<T>
{
public:
    using Base = BaseSamplingLayer<T>;
    using SetupParams = typename Base::SetupParams;
    using ForwardParams = typename Base::ForwardParams;

    TopPSamplingLayer(runtime::SizeType maxBatchSize, runtime::SizeType vocabSize, runtime::SizeType vocabSizePadded,
        cudaStream_t stream, std::shared_ptr<tensorrt_llm::common::IAllocator> allocator, cudaDeviceProp* prop,
        bool isDeterministic = true, bool isAirTopP = true);
    ~TopPSamplingLayer();

    void setup(
        runtime::SizeType batchSize, runtime::SizeType const* batchSlots, SetupParams const& setupParams) override;
    void forward(DecodingOutputParams& outputs, ForwardParams& inputs) override;

    bool const* getSkipDecodeHost() const
    {
        return mSkipDecodeHost;
    }

protected:
    runtime::SizeType* mRuntimeTopKDevice = nullptr;
    float* mRuntimeTopPDevice = nullptr;
    float mRuntimeMaxTopP{0.f};
    float* mInitialTopPDevice = nullptr;
    float* mTopPDecayDevice = nullptr;
    float* mTopPMinDevice = nullptr;
    runtime::TokenIdType* mTopPResetIdsDevice = nullptr;
    void* mSetupWorkspaceDevice = nullptr;

    runtime::TokenIdType* mTopPIdValsDevice = nullptr;
    runtime::SizeType* mTopPOffsetDevice = nullptr;
    runtime::SizeType* mBeginTopPOffsetDevice = nullptr;
    bool* mSkipDecodeDevice = nullptr;
    bool* mSkipDecodeHost = nullptr;
    bool mIsDeterministic = true;
    runtime::SizeType mAirTopPBlockNum;
    bool mIsAirTopP = false;

    using Base::mMaxBatchSize;
    using Base::mVocabSize;
    using Base::mVocabSizePadded;

    using Base::mSamplingWorkspaceSize;
    using Base::mAllocatedSize;

    using Base::mStream;
    using Base::mAllocator;
    using Base::mCudaDeviceProp;

private:
    void allocateBuffer(runtime::SizeType batchSize);
    void freeBuffer();
};

} // namespace layers
} // namespace tensorrt_llm
