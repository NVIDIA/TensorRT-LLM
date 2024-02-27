/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/decodingCommon.h"
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
        std::optional<std::vector<float>> beam_search_diversity_rate; // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> length_penalty;             // [1] or [batch_size] on cpu
        std::optional<std::vector<int>> early_stopping;               // [1] or [batch_size] on cpu
    };

    OnlineBeamSearchLayer(
        size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream, std::shared_ptr<tc::IAllocator> allocator);

    OnlineBeamSearchLayer(OnlineBeamSearchLayer<T> const& beam_search_layer);

    ~OnlineBeamSearchLayer() override;

    void setup(size_t batch_size, SetupParams const& setupParams);

protected:
    // meta data
    using Base::vocab_size_;
    using Base::vocab_size_padded_;

    using Base::topk_softmax_workspace_size_;
    using Base::topk_softmax_workspace_;

    using typename Base::BeamSearchOutputParams;
    using typename Base::SoftmaxParams;

    void invokeSoftMax(BeamSearchOutputParams& outputs, SoftmaxParams const& params) override;

    using Base::mStream;
    using Base::mIsAllocateBuffer;
    using Base::mAllocator;

    std::vector<float> mDiversityRate;
    std::vector<float> mLengthPenalty;
    std::vector<int> mEarlyStopping;
    float* diversity_rates_buf_;
    float* length_penalties_buf_;
    int* early_stoppings_buf_;

private:
    void allocateBuffer(size_t batch_size);
    void freeBuffer();
};

} // namespace layers
} // namespace tensorrt_llm
