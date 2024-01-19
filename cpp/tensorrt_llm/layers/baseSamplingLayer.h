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

#include <curand_kernel.h>

#include "tensorrt_llm/common/tensor.h"
#include "tensorrt_llm/kernels/penaltyTypes.h"
#include "tensorrt_llm/layers/baseLayer.h"
#include "tensorrt_llm/layers/decodingParams.h"

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace layers
{

template <typename T>
class BaseSamplingLayer : public BaseLayer
{
public:
    BaseSamplingLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
        std::shared_ptr<tensorrt_llm::common::IAllocator> allocator, bool is_free_buffer_after_forward,
        cudaDeviceProp* cuda_device_prop);

    BaseSamplingLayer(BaseSamplingLayer const& sampling_layer);

    ~BaseSamplingLayer() override;

    class SetupParams : public DecodingSetupParams
    {
    public:
        std::optional<std::vector<std::uint32_t>> runtime_top_k;  // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> runtime_top_p;          // [1] or [batch_size] on cpu
        std::optional<std::vector<uint64_t>> randomSeed;          // [1] or [batch_size] on cpu
        std::optional<std::vector<float>> top_p_decay;            // [batch_size], must between [0, 1]
        std::optional<std::vector<float>> top_p_min;              // [batch_size], must between [0, 1]
        std::optional<std::vector<std::int32_t>> top_p_reset_ids; // [batch_size]
        std::optional<bool> normalize_log_probs;
    };

    class ForwardParams : public DecodingParams
    {
    public:
        ForwardParams(int step, int ite, tc::Tensor logits, tc::Tensor end_ids, int max_seq_len)
            : DecodingParams{step, ite, std::move(logits), std::move(end_ids)}
            , max_seq_len{max_seq_len}
        {
        }

        // mandatory parameters
        int max_seq_len;

        // optional parameters
        std::optional<tc::Tensor> embedding_bias; // [vocab_size_padded]
        std::optional<tc::Tensor> input_lengths;  // [local_batch_size * beam_width]
    };

    void forward(DecodingOutputParams& outputs, ForwardParams const& params, int* penalty_workspace);

    virtual void setup(size_t batch_size, SetupParams const& setupParams) = 0;

protected:
    size_t vocab_size_;
    size_t vocab_size_padded_;

    size_t sampling_workspace_size_;
    void* sampling_workspace_ = nullptr;
    curandState_t* curandstate_buf_ = nullptr;
    uint64_t* random_seeds_buf_ = nullptr;

    float* temperature_buf_ = nullptr;
    float* repetition_penalty_buf_ = nullptr;
    float* presence_penalty_buf_ = nullptr;
    float* frequency_penalty_buf_ = nullptr;
    int* min_lengths_buf_ = nullptr;
    bool* skip_decode_buf_ = nullptr;
    T* runtime_logits_buf_ = nullptr;

    std::vector<float> mTemperature;
    std::vector<float> mRepetitionPenalty;
    std::vector<float> mPresencePenalty;
    std::vector<float> mFrequencyPenalty;
    std::vector<int> mMinLengths;
    bool* skip_decode_ = nullptr;
    bool skip_any_ = false;

    bool use_temperature_ = false;
    bool use_repetition_penalty_ = false;
    bool use_presence_penalty_ = false;
    bool use_frequency_penalty_ = false;
    bool use_min_lengths_ = false;

    virtual void runSampling(DecodingOutputParams& outputs, DecodingParams const& params) = 0;

    virtual void freeBuffer();
    void setupBase(size_t batch_size, SetupParams const& setupParams);

private:
    void allocateBuffer(size_t batch_size);
    bool isValidBatchSize(size_t batch_size);
};

} // namespace layers
} // namespace tensorrt_llm
