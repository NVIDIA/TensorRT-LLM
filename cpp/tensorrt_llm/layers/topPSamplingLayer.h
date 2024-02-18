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

    TopPSamplingLayer(std::size_t maxBatchSize, std::size_t vocabSize, std::size_t vocabSizePadded, cudaStream_t stream,
        std::shared_ptr<tensorrt_llm::common::IAllocator> allocator, cudaDeviceProp* prop, bool isDeterministic = true);
    ~TopPSamplingLayer();

    void setup(std::size_t batchSize, int32_t const* batchSlots, SetupParams const& setupParams) override;
    void forward(DecodingOutputParams& outputs, ForwardParams& inputs) override;

    const bool* getSkipDecodeHost() const
    {
        return mSkipDecodeHost;
    }

protected:
    uint32_t* mRuntimeTopKDevice = nullptr;
    float* mRuntimeTopPDevice = nullptr;
    float mRuntimeMaxTopP{0.f};
    float* mInitialTopPDevice = nullptr;
    float* mTopPDecayDevice = nullptr;
    float* mTopPMinDevice = nullptr;
    int32_t* mTopPResetIdsDevice = nullptr;
    void* mSetupWorkspaceDevice = nullptr;

    int32_t* mTopPIdValsDevice = nullptr;
    int32_t* mTopPOffsetDevice = nullptr;
    int32_t* mBeginTopPOffsetDevice = nullptr;
    bool* mSkipDecodeDevice = nullptr;
    bool* mSkipDecodeHost = nullptr;
    size_t mCubTempStorageSize;
    bool mIsDeterministic = true;
    int mAirTopPBlockNum;

    using Base::mMaxBatchSize;
    using Base::mVocabSize;
    using Base::mVocabSizePadded;

    using Base::mSamplingWorkspaceSize;
    using Base::mAllocatedSize;

    using Base::mStream;
    using Base::mAllocator;
    using Base::mCudaDeviceProp;

private:
    void allocateBuffer(std::size_t batchSize);
    void freeBuffer();
};

} // namespace layers
} // namespace tensorrt_llm
