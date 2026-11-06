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

#include "tensorrt_llm/kernels/speculativeDecoding/chainSpeculativeSamplingKernels.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <tuple>

namespace torch_ext
{

namespace tk = tensorrt_llm::kernels::speculative_decoding;

//! Returns (output_token_ids [batch, numDraft + 1], num_tokens [batch]); see
//! ChainSpeculativeSamplingParams for the semantics.
std::tuple<torch::Tensor, torch::Tensor> chain_speculative_sampling(torch::Tensor const& draftProbs,
    torch::Tensor const& draftTokenIds, torch::Tensor const& targetProbs, torch::Tensor const& seed,
    torch::Tensor const& offset)
{
    TORCH_CHECK(draftProbs.dim() == 3, "draft_probs must be [batch, numDraft, vocab]");
    TORCH_CHECK(draftTokenIds.dim() == 2, "draft_token_ids must be [batch, numDraft]");
    TORCH_CHECK(targetProbs.dim() == 3, "target_probs must be [batch, numDraft + 1, vocab]");

    auto const batchSize = draftProbs.size(0);
    auto const numDraft = draftProbs.size(1);
    auto const vocabSize = draftProbs.size(2);
    TORCH_CHECK(draftTokenIds.size(0) == batchSize && draftTokenIds.size(1) == numDraft,
        "draft_token_ids must be [batch, numDraft]");
    TORCH_CHECK(
        targetProbs.size(0) == batchSize && targetProbs.size(1) == numDraft + 1 && targetProbs.size(2) == vocabSize,
        "target_probs must be [batch, numDraft + 1, vocab]");

    for (auto const* t : {&draftProbs, &draftTokenIds, &targetProbs, &seed, &offset})
    {
        TORCH_CHECK(t->is_cuda() && t->is_contiguous(), "chain_speculative_sampling inputs must be contiguous CUDA");
    }
    TORCH_CHECK(draftProbs.scalar_type() == torch::kFloat32 && targetProbs.scalar_type() == torch::kFloat32,
        "draft_probs and target_probs must be float32");
    TORCH_CHECK(draftTokenIds.scalar_type() == torch::kInt32, "draft_token_ids must be int32");
    TORCH_CHECK(seed.scalar_type() == torch::kInt64 && offset.scalar_type() == torch::kInt64,
        "seed and offset must be int64 tensors");
    TORCH_CHECK(
        seed.numel() == batchSize && offset.numel() == batchSize, "seed and offset must hold one entry per request");

    auto outputTokenIds = torch::empty({batchSize, numDraft + 1}, draftTokenIds.options());
    auto outputNumTokens = torch::empty({batchSize}, draftTokenIds.options());

    tk::ChainSpeculativeSamplingParams params;
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
    tk::invokeChainSpeculativeSampling(params, at::cuda::getCurrentCUDAStream());
    return {outputTokenIds, outputNumTokens};
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "chain_speculative_sampling(Tensor draft_probs, Tensor draft_token_ids, Tensor target_probs, "
        "Tensor seed, Tensor offset) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("chain_speculative_sampling", &torch_ext::chain_speculative_sampling);
}
