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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/speculativeDecoding/ngramKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"

namespace th = torch;
namespace tkn = tensorrt_llm::kernels::speculative_decoding::ngram;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

void checkInt32Cuda(th::Tensor const& tensor, char const* name, int64_t dims)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == th::kInt32, name, " must be int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.dim() == dims, name, " must be ", dims, "-D");
}

} // namespace

//! Append this step's accepted tokens to each row's token history and look up the next draft tokens.
//! See ``tkn::NGramDraftParams`` for the buffer layouts. ``draft_tokens_out`` may have zero columns, in
//! which case only the histories are extended.
void ngram_extend_and_draft_op(th::Tensor history_tokens, th::Tensor history_lens, th::Tensor const& slot_ids,
    th::Tensor const& row_mask, th::Tensor const& accepted_tokens, th::Tensor const& num_accepted_tokens,
    th::Tensor draft_tokens_out, th::Tensor match_len_out, int64_t max_ngram_size, bool use_oldest, bool public_pool)
{
    checkInt32Cuda(history_tokens, "history_tokens", 2);
    checkInt32Cuda(history_lens, "history_lens", 1);
    checkInt32Cuda(slot_ids, "slot_ids", 1);
    checkInt32Cuda(row_mask, "row_mask", 1);
    checkInt32Cuda(accepted_tokens, "accepted_tokens", 2);
    checkInt32Cuda(num_accepted_tokens, "num_accepted_tokens", 1);
    checkInt32Cuda(draft_tokens_out, "draft_tokens_out", 2);
    checkInt32Cuda(match_len_out, "match_len_out", 1);

    int64_t const batchSize = slot_ids.size(0);
    int64_t const numSlots = history_tokens.size(0);
    TORCH_CHECK(history_lens.size(0) == numSlots, "history_lens must have one entry per history slot");
    TORCH_CHECK(row_mask.size(0) == batchSize, "row_mask must have one entry per batch row");
    TORCH_CHECK(accepted_tokens.size(0) == batchSize, "accepted_tokens must have one row per batch row");
    TORCH_CHECK(num_accepted_tokens.size(0) == batchSize, "num_accepted_tokens must have one entry per batch row");
    TORCH_CHECK(draft_tokens_out.size(0) == batchSize, "draft_tokens_out must have one row per batch row");
    TORCH_CHECK(match_len_out.size(0) == batchSize, "match_len_out must have one entry per batch row");

    tkn::NGramDraftParams params;
    params.batchSize = static_cast<int>(batchSize);
    params.draftLen = static_cast<int>(draft_tokens_out.size(1));
    params.acceptedStride = static_cast<int>(accepted_tokens.size(1));
    params.maxNgramSize = static_cast<int>(max_ngram_size);
    params.numSlots = static_cast<int>(numSlots);
    params.maxSeqLen = static_cast<int>(history_tokens.size(1));
    params.useOldest = use_oldest;
    params.publicPool = public_pool;
    params.slotIds = slot_ids.data_ptr<int>();
    params.rowMask = row_mask.data_ptr<int>();
    params.acceptedTokens = accepted_tokens.data_ptr<int>();
    params.numAcceptedTokens = num_accepted_tokens.data_ptr<int>();
    params.historyTokens = history_tokens.data_ptr<int>();
    params.historyLens = history_lens.data_ptr<int>();
    params.draftTokensOut = params.draftLen > 0 ? draft_tokens_out.data_ptr<int>() : nullptr;
    params.matchLenOut = match_len_out.data_ptr<int>();

    auto stream = at::cuda::getCurrentCUDAStream(history_tokens.get_device());
    tkn::invokeNGramExtendAndDraft(params, stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "ngram_extend_and_draft_op(Tensor(a!) history_tokens, Tensor(b!) history_lens, Tensor slot_ids, "
        "Tensor row_mask, Tensor accepted_tokens, Tensor num_accepted_tokens, Tensor(c!) draft_tokens_out, "
        "Tensor(d!) match_len_out, int max_ngram_size, bool use_oldest, bool public_pool) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("ngram_extend_and_draft_op", &tensorrt_llm::torch_ext::ngram_extend_and_draft_op);
}
