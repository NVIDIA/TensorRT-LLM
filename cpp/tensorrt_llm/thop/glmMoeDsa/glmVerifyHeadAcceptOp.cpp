/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/glmMoeDsa/glmVerifyHeadAccept.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Accepted-token counts of the strict acceptance of one-model speculative decoding, with the forced count of
// TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS applied (force_mode 0: none, 1: force_base, 2: force_base plus a pool
// draw below force_frac, capped at force_max, force_counter advanced by one). Writes the accepted-token rows into
// accepted_tokens when given.
at::Tensor glmVerifyHeadAccept(at::Tensor const& target_tokens, at::Tensor const& draft_tokens, int64_t num_contexts,
    std::optional<at::Tensor> accepted_tokens, std::optional<at::Tensor> const& force_pool,
    std::optional<at::Tensor> force_counter, int64_t force_mode, int64_t force_base, int64_t force_max,
    double force_frac, int64_t counter_stride, int64_t slot_stride, int64_t pool_size)
{
    CHECK_TH_CUDA(target_tokens);
    CHECK_CONTIGUOUS(target_tokens);
    auto const targetType = target_tokens.scalar_type();
    TORCH_CHECK(targetType == at::ScalarType::Int || targetType == at::ScalarType::Long,
        "glm_verify_head_accept expects int32 or int64 target tokens, got ", targetType);
    CHECK_INPUT(draft_tokens, at::ScalarType::Int);
    TORCH_CHECK(draft_tokens.dim() == 2, "glm_verify_head_accept expects draft tokens [num_gens, draft_len]");
    int64_t const numGens = draft_tokens.size(0);
    int64_t const draftLen = draft_tokens.size(1);
    int64_t const batchSize = num_contexts + numGens;
    TORCH_CHECK(target_tokens.numel() >= num_contexts + numGens * (draftLen + 1),
        "glm_verify_head_accept: target tokens hold one row per context and draft_len + 1 per generation request");
    CHECK_OPTIONAL_INPUT(accepted_tokens, at::ScalarType::Int);
    TORCH_CHECK(!accepted_tokens.has_value() || accepted_tokens->numel() >= batchSize * (draftLen + 1),
        "glm_verify_head_accept: accepted_tokens must be [batch, draft_len + 1]");
    TORCH_CHECK(force_mode >= 0 && force_mode <= 2, "glm_verify_head_accept: unknown force_mode ", force_mode);
    bool const randomForce = force_mode == 2;
    if (randomForce)
    {
        TORCH_CHECK(force_pool.has_value() && force_counter.has_value(),
            "glm_verify_head_accept: force_mode 2 needs force_pool and force_counter");
        CHECK_INPUT(force_pool.value(), at::ScalarType::Float);
        CHECK_INPUT(force_counter.value(), at::ScalarType::Long);
        TORCH_CHECK(pool_size > 0 && (pool_size & (pool_size - 1)) == 0 && force_pool->numel() >= pool_size,
            "glm_verify_head_accept: the pool size must be a power of two within force_pool");
    }

    at::Tensor numAccepted = at::empty({batchSize}, draft_tokens.options());

    namespace tk = tensorrt_llm::kernels::glm_moe_dsa;
    tk::VerifyHeadAcceptParams params;
    params.targetTokens = target_tokens.data_ptr();
    params.targetTokensInt64 = targetType == at::ScalarType::Long;
    params.draftTokens = draft_tokens.data_ptr<int32_t>();
    params.numAcceptedTokens = numAccepted.data_ptr<int32_t>();
    params.acceptedTokens = accepted_tokens.has_value() ? accepted_tokens->data_ptr<int32_t>() : nullptr;
    params.numContexts = static_cast<int>(num_contexts);
    params.numGens = static_cast<int>(numGens);
    params.draftLen = static_cast<int>(draftLen);
    params.forceMode = static_cast<tk::VerifyHeadForceMode>(force_mode);
    params.forceBase = static_cast<int>(force_base);
    params.forceMax = static_cast<int>(force_max);
    params.forceFrac = static_cast<float>(force_frac);
    params.forcePool = randomForce ? force_pool->data_ptr<float>() : nullptr;
    params.forceCounter = randomForce ? force_counter->data_ptr<int64_t>() : nullptr;
    params.counterStride = counter_stride;
    params.slotStride = slot_stride;
    params.poolMask = pool_size - 1;
    params.stream = at::cuda::getCurrentCUDAStream(draft_tokens.get_device());
    tk::invokeVerifyHeadAccept(params);
    return numAccepted;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "glm_verify_head_accept(Tensor target_tokens, Tensor draft_tokens, int num_contexts, "
        "Tensor(a!)? accepted_tokens, Tensor? force_pool, Tensor(b!)? force_counter, int force_mode, int force_base, "
        "int force_max, float force_frac, int counter_stride, int slot_stride, int pool_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm_verify_head_accept", &tensorrt_llm::torch_ext::glmVerifyHeadAccept);
}
