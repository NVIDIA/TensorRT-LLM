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

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/layers/baseSamplingLayer.h"

namespace tensorrt_llm
{
namespace layers
{

template <typename T>
class TopKSamplingLayer : public BaseSamplingLayer<T>
{
public:
    static constexpr uint32_t TOP_K_MAX = 1024;
    using Base = BaseSamplingLayer<T>;
    using SetupParams = typename Base::SetupParams;

    TopKSamplingLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
        std::shared_ptr<tensorrt_llm::common::IAllocator> allocator, bool is_free_buffer_after_forward);
    TopKSamplingLayer(TopKSamplingLayer<T> const& top_k_sampling_layer);
    ~TopKSamplingLayer();

    void setup(size_t batch_size, SetupParams const& setupParams) override;

protected:
    void runSampling(DecodingOutputParams& outputs, DecodingParams const& params) override;

    void freeBuffer() override;

    bool normalize_log_probs = true;
    uint32_t runtime_max_top_k_ = 1;
    uint32_t* runtime_top_k_buf_ = nullptr;
    float* runtime_top_p_buf_ = nullptr;
    using Base::vocab_size_;
    using Base::vocab_size_padded_;

    using Base::sampling_workspace_size_;
    using Base::sampling_workspace_;
    using Base::curandstate_buf_;
    using Base::random_seeds_buf_;
    using Base::skip_decode_buf_;
    using Base::skip_decode_;
    using Base::skip_any_;
    using Base::runtime_logits_buf_;

    using Base::stream_;
    using Base::allocator_;
    using Base::is_allocate_buffer_;

private:
    void allocateBuffer(size_t batch_size, std::vector<uint32_t> const& top_k);
};

} // namespace layers
} // namespace tensorrt_llm
