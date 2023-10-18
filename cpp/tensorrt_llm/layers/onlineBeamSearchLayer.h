/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/onlineSoftmaxBeamsearchKernels.h"
#include "tensorrt_llm/layers/baseBeamSearchLayer.h"

#include <optional>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace layers
{

// TODO  - merge this class with BaseBeamSearchLayer
template <typename T>
class OnlineBeamSearchLayer : public BaseBeamSearchLayer<T>
{
public:
    using Base = BaseBeamSearchLayer<T>;

    class SetupParams : public Base::SetupParams
    {
    public:
        std::optional<float> beam_search_diversity_rate;
        std::optional<float> length_penalty;
    };

    OnlineBeamSearchLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream, tc::IAllocator* allocator,
        bool is_free_buffer_after_forward);

    OnlineBeamSearchLayer(OnlineBeamSearchLayer<T> const& beam_search_layer);

    ~OnlineBeamSearchLayer() override;

    void setup(SetupParams const& setupParams);

protected:
    // meta data
    using Base::vocab_size_;
    using Base::vocab_size_padded_;

    using Base::topk_softmax_workspace_size_;
    using Base::topk_softmax_workspace_;

    void allocateBuffer(size_t batch_size, size_t beam_width) override;

    using typename Base::BeamSearchOutputParams;
    using typename Base::SoftmaxParams;

    void invokeSoftMax(BeamSearchOutputParams& outputs, SoftmaxParams const& params) override;

    using Base::stream_;
    using Base::is_allocate_buffer_;
    using Base::allocator_;

    float mDiversityRate;
    float mLengthPenalty;
};

} // namespace layers
} // namespace tensorrt_llm
