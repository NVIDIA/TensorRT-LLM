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
#include "tensorrt_llm/layers/baseSamplingLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/layers/topKSamplingLayer.h"
#include "tensorrt_llm/layers/topPSamplingLayer.h"
#include "tensorrt_llm/runtime/decodingMode.h"

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace layers
{

template <typename T>
inline bool allOfBatchSlots(int32_t const* batchSlotsHost, T const* data, size_t batchSize, T value)
{
    return std::all_of(batchSlotsHost, batchSlotsHost + batchSize, [&](int32_t b) { return data[b] == value; });
};

//! \brief Top class for sampling layers.
//! It sets up and executes TopKSamplingLayer and TopPSamplingLayer samplings
template <typename T>
class SamplingLayer : public BaseSamplingLayer<T>
{
public:
    using Base = BaseSamplingLayer<T>;
    using SetupParams = typename Base::SetupParams;
    using ForwardParams = typename Base::ForwardParams;

    SamplingLayer(runtime::DecodingMode const& mode, size_t maxBatchSize, size_t vocabSize, size_t vocabSizePadded,
        cudaStream_t stream, std::shared_ptr<tensorrt_llm::common::IAllocator> allocator, cudaDeviceProp* prop);

    ~SamplingLayer() override = default;

    void forward(DecodingOutputParams& outputs, ForwardParams& inputs) override;

    void setup(size_t batchSize, int32_t const* batchSlots, SetupParams const& setupParams) override;

private:
    using Base::mMaxBatchSize;
    using Base::mVocabSize;
    using Base::mVocabSizePadded;
    using Base::mSamplingWorkspaceSize;
    using Base::mAllocatedSize;

    using Base::mStream;
    using Base::mAllocator;

    runtime::DecodingMode mDecodingMode;

    void* mSamplingWorkspaceDevice = nullptr;
    curandState_t* mCurandStatesDevice = nullptr;
    uint64_t* mRandomSeedsDevice = nullptr;

    bool* mSkipDecodeDevice = nullptr;

    bool* mSkipDecodeHost = nullptr;
    bool mSkipAny = false;

    std::unique_ptr<TopKSamplingLayer<T>> mTopKDecode;
    std::unique_ptr<TopPSamplingLayer<T>> mTopPDecode;

private:
    void allocateBuffer(size_t batchSize);
    void freeBuffer();
};

} // namespace layers
} // namespace tensorrt_llm
