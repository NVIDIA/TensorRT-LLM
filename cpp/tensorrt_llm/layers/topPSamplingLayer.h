/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
template <typename T>
class TopPSamplingLayer : public BaseSamplingLayer<T>
{
public:
    using Base = BaseSamplingLayer<T>;
    using SetupParams = typename Base::SetupParams;

    TopPSamplingLayer(std::size_t maxBatchSize, std::size_t vocabSize, std::size_t vocabSizePadded, cudaStream_t stream,
        std::shared_ptr<tensorrt_llm::common::IAllocator> allocator, cudaDeviceProp* prop, bool isDeterministic = true);
    TopPSamplingLayer(TopPSamplingLayer<T> const& top_p_sampling_layer);
    ~TopPSamplingLayer();

    void setup(std::size_t batchSize, int const* batch_slots, SetupParams const& setupParams) override;

protected:
    void runSampling(DecodingOutputParams& outputs, DecodingParams const& inputs) override;
    void freeBuffer() override;

protected:
    uint32_t* runtime_top_k_buf_ = nullptr;
    float* runtime_top_p_buf_ = nullptr;
    float mRuntimeMaxTopP;
    float* initial_top_p_buf_ = nullptr;
    float* top_p_decay_buf_ = nullptr;
    float* top_p_min_buf_ = nullptr;
    int32_t* top_p_reset_ids_buf_ = nullptr;
    void* setup_workspace_buf_ = nullptr;

    int32_t* topp_id_vals_buf_ = nullptr;
    int32_t* topp_offset_buf_ = nullptr;
    int32_t* begin_topp_offset_buf_ = nullptr;
    std::size_t cub_temp_storage_size_;
    bool is_deterministic_ = true;
    int air_topp_block_num_;

    using Base::mMaxBatchSize;
    using Base::mVocabSize;
    using Base::mVocabSizePadded;

    using Base::mSamplingWorkspaceSize;
    using Base::mSamplingWorkspaceDevice;
    using Base::mCurandStatesDevice;
    using Base::mSkipDecodeDevice;
    using Base::mSkipDecodeHost;
    using Base::mSkipAny;
    using Base::mRuntimeLogitsDevice;

    using Base::mStream;
    using Base::mAllocator;
    using Base::mIsAllocateBuffer;
    using Base::mCudaDeviceProp;

private:
    void allocateBuffer(std::size_t batchSize);
};

} // namespace layers
} // namespace tensorrt_llm
