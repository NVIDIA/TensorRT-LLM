/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/config.h"
#include <cstdint>
#include <cuda_runtime.h>
TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{
// IPC pointer table and its real mutable state. This is not the MNNVL ABI.
struct IntraNodeReduceParams
{
    void* const* workspace{};
    int32_t* flags{};
    int64_t* layout{};
    void const* allreduce_in{};
    void const* residual_in{};
    void const* rms_gamma{};
    void* residual_out{};
    void* norm_out{};
    void const* expert_scale_factor{};
    int32_t const* expanded_idx_to_permuted_idx{};
    void const* shared_expert_output{};
    int rank{};
    int nranks{};
    int size{};
    int hidden_dim{};
    int top_k{};
    float rms_eps{};
    bool trigger_completion_at_end{true};
    cudaStream_t stream{};
};

bool intraNodeReduceSupported(int rows, int hidden, int nranks, bool finalize);
void intraNodeReduce(IntraNodeReduceParams const& params);
void intraNodeMoeFinalize(IntraNodeReduceParams const& params);
} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
