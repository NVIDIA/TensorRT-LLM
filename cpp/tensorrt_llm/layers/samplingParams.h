/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/layers/decodingParams.h"
#include <tensorrt_llm/common/tensor.h>
#include <tensorrt_llm/runtime/common.h>

#include <optional>
#include <vector>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::layers
{

class SamplingSetupParams : public BaseSetupParams
{
public:
    std::optional<std::vector<runtime::SizeType32>> runtime_top_k;    // [1] or [batchSize] on cpu
    std::optional<std::vector<float>> runtime_top_p;                  // [1] or [batchSize] on cpu
    std::optional<std::vector<uint64_t>> randomSeed;                  // [1] or [batchSize] on cpu
    std::optional<std::vector<float>> top_p_decay;                    // [batchSize], must between [0, 1]
    std::optional<std::vector<float>> top_p_min;                      // [batchSize], must between [0, 1]
    std::optional<std::vector<runtime::TokenIdType>> top_p_reset_ids; // [batchSize]
    std::optional<bool> normalize_log_probs;
};

class SamplingInputParams : public BaseInputParams
{
public:
    explicit SamplingInputParams(runtime::SizeType32 step, runtime::SizeType32 ite, tc::Tensor logits,
        tc::Tensor end_ids, runtime::SizeType32 max_seq_len)
        : BaseInputParams{step, ite, std::move(end_ids)}
        , logits{std::move(logits)}
        , max_seq_len{max_seq_len}
    {
    }

    // mandatory parameters
    tc::Tensor logits; // [local_batch_size, beam_width, vocab_size_padded]
    runtime::SizeType32 max_seq_len;

    // optional parameters
    std::optional<tc::Tensor> input_lengths; // [localBatchSize]
    curandState_t* curand_states;            // [localBatchSize]
    // Pointer to the workspace for sampling computation
    void* sampling_workspace;
    // Flag to mark that logits tensor contains probabilities
    bool probs_computed;
};

class SamplingOutputParams : public BaseOutputParams
{
public:
    explicit SamplingOutputParams(tc::Tensor outputIds)
        : BaseOutputParams{std::move(outputIds)}
    {
    }
};

} // namespace tensorrt_llm::layers
