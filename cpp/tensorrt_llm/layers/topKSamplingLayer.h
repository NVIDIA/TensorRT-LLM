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

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/layers/baseSamplingLayer.h"

namespace tensorrt_llm
{
namespace layers
{

//! \brief Layer to randomly sample tokens from TopK logits.
//! When both TopK and TopP are specified, layer jointly samples using TopK and TopP.
//! When no TopK param is specified, sampling is skipped for particular request.
template <typename T>
class TopKSamplingLayer : public BaseSamplingLayer<T>
{
public:
    using Base = BaseSamplingLayer<T>;
    using SetupParams = typename Base::SetupParams;
    using ForwardParams = typename Base::ForwardParams;

    TopKSamplingLayer(size_t maxBatchSize, size_t vocabSize, size_t vocabSizePadded, cudaStream_t stream,
        std::shared_ptr<tensorrt_llm::common::IAllocator> allocator);
    ~TopKSamplingLayer();

    void setup(size_t batchSize, int32_t const* batchSlots, SetupParams const& setupParams) override;
    void forward(DecodingOutputParams& outputs, ForwardParams& inputs) override;

    const bool* getSkipDecodeHost() const
    {
        return mSkipDecodeHost;
    }

protected:
    bool mNormalizeLogProbs = true;
    uint32_t mRuntimeMaxTopK = 0;
    uint32_t* mRuntimeTopKDevice = nullptr;
    float* mRuntimeTopPDevice = nullptr;
    void* mSetupWorkspaceDevice = nullptr;
    bool* mSkipDecodeDevice = nullptr;
    bool* mSkipDecodeHost = nullptr;

    using Base::mMaxBatchSize;
    using Base::mVocabSize;
    using Base::mVocabSizePadded;

    using Base::mSamplingWorkspaceSize;
    using Base::mAllocatedSize;

    using Base::mStream;
    using Base::mAllocator;

    static constexpr uint32_t TOP_K_MAX = 1024;

private:
    void allocateBuffer(size_t batchSize);
    void freeBuffer();
};

} // namespace layers
} // namespace tensorrt_llm
