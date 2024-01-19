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

template <typename T>
class TopPSamplingLayer : public BaseSamplingLayer<T>
{
public:
    using Base = BaseSamplingLayer<T>;
    using SetupParams = typename Base::SetupParams;

    TopPSamplingLayer(std::size_t vocab_size, std::size_t vocab_size_padded, cudaStream_t stream,
        std::shared_ptr<tensorrt_llm::common::IAllocator> allocator, bool is_free_buffer_after_forward,
        cudaDeviceProp* cuda_device_prop, bool is_deterministic = true);
    TopPSamplingLayer(TopPSamplingLayer<T> const& top_p_sampling_layer);
    ~TopPSamplingLayer();

    void setup(std::size_t batch_size, SetupParams const& setupParams) override;

protected:
    void runSampling(DecodingOutputParams& outputs, DecodingParams const& params) override;
    void freeBuffer() override;

    std::uint32_t* runtime_top_k_buf_ = nullptr;
    float* runtime_top_p_buf_ = nullptr;
    float runtime_max_top_p_;
    float* initial_top_p_buf_ = nullptr;
    float* top_p_decay_buf_ = nullptr;
    float* top_p_min_buf_ = nullptr;
    std::int32_t* top_p_reset_ids_buf_ = nullptr;

    std::int32_t* topp_id_vals_buf_ = nullptr;
    std::int32_t* topp_offset_buf_ = nullptr;
    std::int32_t* begin_topp_offset_buf_ = nullptr;
    std::size_t cub_temp_storage_size_;
    bool is_deterministic_ = true;
    int air_topp_block_num_;

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
    using Base::cuda_device_prop_;

private:
    void allocateBuffer(std::size_t batch_size, std::vector<float> const& top_k);
};

} // namespace layers
} // namespace tensorrt_llm
