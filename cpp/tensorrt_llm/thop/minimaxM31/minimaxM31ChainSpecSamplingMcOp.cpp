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

#include "tensorrt_llm/kernels/minimaxM31/chainSpecSamplingMcSm103.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! ST-01-REJ's multi-CTA chain speculative sampling (see kernels/minimaxM31/chainSpecSamplingMcSm103.h): the inputs
//! and outputs of trtllm::chain_speculative_sampling (output_token_ids [batch, numDraft + 1], num_tokens [batch]),
//! bitwise its results, on batch x ctas_per_row CTAs. workspace: a zero-initialized byte buffer of at least
//! batch x minimax_m31_chain_speculative_sampling_mc_ws_bytes(), reused across calls; ctas_per_row divides 1024 and
//! batch x ctas_per_row must not exceed the SM count when ctas_per_row > 1.
std::tuple<torch::Tensor, torch::Tensor> minimaxM31ChainSpecSamplingMc(torch::Tensor const& draftProbs,
    torch::Tensor const& draftTokenIds, torch::Tensor const& targetProbs, torch::Tensor const& seed,
    torch::Tensor const& offset, torch::Tensor& workspace, int64_t ctasPerRow)
{
    TORCH_CHECK(draftProbs.dim() == 3,
        "minimax_m31_chain_speculative_sampling_mc: draft_probs must be [batch, "
        "numDraft, vocab]");
    TORCH_CHECK(draftTokenIds.dim() == 2,
        "minimax_m31_chain_speculative_sampling_mc: draft_token_ids must be [batch, "
        "numDraft]");
    TORCH_CHECK(targetProbs.dim() == 3,
        "minimax_m31_chain_speculative_sampling_mc: target_probs must be [batch, numDraft + 1, vocab]");
    auto const batchSize = draftProbs.size(0);
    auto const numDraft = draftProbs.size(1);
    auto const vocabSize = draftProbs.size(2);
    TORCH_CHECK(draftTokenIds.size(0) == batchSize && draftTokenIds.size(1) == numDraft,
        "minimax_m31_chain_speculative_sampling_mc: draft_token_ids must be [batch, numDraft]");
    TORCH_CHECK(
        targetProbs.size(0) == batchSize && targetProbs.size(1) == numDraft + 1 && targetProbs.size(2) == vocabSize,
        "minimax_m31_chain_speculative_sampling_mc: target_probs must be [batch, numDraft + 1, vocab]");
    for (auto const* t : {&draftProbs, &draftTokenIds, &targetProbs, &seed, &offset})
    {
        TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->get_device() == draftProbs.get_device(),
            "minimax_m31_chain_speculative_sampling_mc: inputs must be contiguous CUDA tensors on one device");
    }
    TORCH_CHECK(draftProbs.scalar_type() == torch::kFloat32 && targetProbs.scalar_type() == torch::kFloat32,
        "minimax_m31_chain_speculative_sampling_mc: draft_probs and target_probs must be float32");
    TORCH_CHECK(draftTokenIds.scalar_type() == torch::kInt32,
        "minimax_m31_chain_speculative_sampling_mc: draft_token_ids must be int32");
    TORCH_CHECK(seed.scalar_type() == torch::kInt64 && offset.scalar_type() == torch::kInt64,
        "minimax_m31_chain_speculative_sampling_mc: seed and offset must be int64 tensors");
    TORCH_CHECK(seed.numel() == batchSize && offset.numel() == batchSize,
        "minimax_m31_chain_speculative_sampling_mc: seed and offset must hold one entry per request");
    TORCH_CHECK(vocabSize % 4 == 0 && vocabSize <= tensorrt_llm::kernels::minimaxM31::kChainSpecMcMaxVocab
            && reinterpret_cast<uintptr_t>(draftProbs.data_ptr()) % 16 == 0
            && reinterpret_cast<uintptr_t>(targetProbs.data_ptr()) % 16 == 0,
        "minimax_m31_chain_speculative_sampling_mc: the probs must be 16-byte aligned with a vocab divisible by 4 and "
        "at most ",
        tensorrt_llm::kernels::minimaxM31::kChainSpecMcMaxVocab);
    TORCH_CHECK(batchSize <= 65535, "minimax_m31_chain_speculative_sampling_mc: batch must be <= 65535");
    TORCH_CHECK(ctasPerRow >= 1 && ctasPerRow <= 1024 && 1024 % ctasPerRow == 0,
        "minimax_m31_chain_speculative_sampling_mc: ctas_per_row must divide 1024");
    auto const wsBytesPerRow
        = static_cast<int64_t>(tensorrt_llm::kernels::minimaxM31::chainSpecSamplingMcWorkspaceBytesPerRow());
    TORCH_CHECK(workspace.is_cuda() && workspace.is_contiguous() && workspace.get_device() == draftProbs.get_device()
            && reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0
            && workspace.numel() * workspace.element_size() >= wsBytesPerRow * batchSize,
        "minimax_m31_chain_speculative_sampling_mc: workspace must be a contiguous, 16-byte aligned CUDA buffer of at "
        "least ",
        wsBytesPerRow * batchSize, " bytes on draft_probs' device");

    c10::cuda::CUDAGuard const deviceGuard{draftProbs.device()};
    TORCH_CHECK(
        ctasPerRow == 1 || batchSize * ctasPerRow <= at::cuda::getCurrentDeviceProperties()->multiProcessorCount,
        "minimax_m31_chain_speculative_sampling_mc: batch x ctas_per_row must not exceed the SM count");
    auto outputTokenIds = torch::empty({batchSize, numDraft + 1}, draftTokenIds.options());
    auto outputNumTokens = torch::empty({batchSize}, draftTokenIds.options());
    if (batchSize == 0)
    {
        return {outputTokenIds, outputNumTokens};
    }

    tensorrt_llm::kernels::speculative_decoding::ChainSpeculativeSamplingParams params;
    params.draftProbs = draftProbs.const_data_ptr<float>();
    params.draftTokenIds = draftTokenIds.const_data_ptr<int32_t>();
    params.targetProbs = targetProbs.const_data_ptr<float>();
    params.seed = seed.const_data_ptr<int64_t>();
    params.offset = offset.const_data_ptr<int64_t>();
    params.outputTokenIds = outputTokenIds.data_ptr<int32_t>();
    params.outputNumTokens = outputNumTokens.data_ptr<int32_t>();
    params.batchSize = static_cast<int32_t>(batchSize);
    params.numDraftTokens = static_cast<int32_t>(numDraft);
    params.vocabSize = static_cast<int32_t>(vocabSize);
    auto const err = tensorrt_llm::kernels::minimaxM31::invokeChainSpecSamplingMc(params, static_cast<int>(ctasPerRow),
        workspace.data_ptr(), at::cuda::getCurrentCUDAStream(draftProbs.get_device()));
    TORCH_CHECK(
        err == cudaSuccess, "minimax_m31_chain_speculative_sampling_mc launch failed: ", cudaGetErrorString(err));
    return {outputTokenIds, outputNumTokens};
}

//! Workspace bytes per request of minimax_m31_chain_speculative_sampling_mc.
int64_t minimaxM31ChainSpecSamplingMcWsBytes()
{
    return static_cast<int64_t>(tensorrt_llm::kernels::minimaxM31::chainSpecSamplingMcWorkspaceBytesPerRow());
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m31_chain_speculative_sampling_mc(Tensor draft_probs, Tensor draft_token_ids, Tensor target_probs, "
        "Tensor seed, Tensor offset, Tensor(a!) workspace, int ctas_per_row) -> (Tensor, Tensor)");
    // No tensor arguments: a catch-all kernel (as moe_a2a_get_timeout), callable without a device.
    m.def("minimax_m31_chain_speculative_sampling_mc_ws_bytes() -> int",
        &tensorrt_llm::torch_ext::minimaxM31ChainSpecSamplingMcWsBytes);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_chain_speculative_sampling_mc", &tensorrt_llm::torch_ext::minimaxM31ChainSpecSamplingMc);
}
