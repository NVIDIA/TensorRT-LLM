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

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/csrc/foldTail.h"
#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8/csrc/foldTailParams.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

//! FOLD_F2: the tail (one CTA per row) writing the step's next_new_tokens (src_next, [rows, D + 1]) itself from
//! spec_next_new_tokens' operands before it stores and stages them: one launch instead of two.
void minimaxM3Sm107Tp8FoldTailNext(torch::Tensor const& ctrl, torch::Tensor const& slots, int64_t rows,
    torch::Tensor const& srcNew, torch::Tensor& dstNew, torch::Tensor& srcNext, torch::Tensor& dstNext,
    torch::Tensor const& srcDraft, torch::Tensor& dstDraft, torch::Tensor const& srcLens, torch::Tensor& dstLens,
    torch::Tensor& ids, torch::Tensor& draftIds, torch::Tensor& posOff, torch::Tensor& kvOff, int64_t tokensPerStep,
    torch::Tensor const& accepted, torch::Tensor const& numAccepted, torch::Tensor const& batchIdx,
    torch::Tensor const& nextDraft)
{
    c10::cuda::CUDAGuard const deviceGuard{ctrl.device()};
    auto const p = minimaxM3Sm107Tp8FoldTailParams::buildParams(ctrl, slots, rows, srcNew, dstNew, srcNext, dstNext,
        srcDraft, dstDraft, srcLens, dstLens, ids, draftIds, posOff, kvOff, tokensPerStep);
    auto const is2d = [](torch::Tensor const& t)
    { return t.is_cuda() && t.scalar_type() == torch::kInt32 && t.dim() == 2 && t.stride(1) == 1; };
    auto const is1d = [rows](torch::Tensor const& t) {
        return t.is_cuda() && t.scalar_type() == torch::kInt32 && t.dim() == 1 && t.stride(0) == 1 && t.numel() >= rows;
    };
    TORCH_CHECK(is2d(accepted) && is2d(nextDraft) && is2d(srcNext) && is1d(numAccepted) && is1d(batchIdx),
        "minimax_m3_sm107_tp8_fold_tail_next: accepted / next_draft / src_next must be int32 CUDA views with unit "
        "column stride, num_accepted / batch_idx contiguous 1-D int32 CUDA tensors of at least rows elements");
    for (torch::Tensor const* t : {&accepted, &numAccepted, &batchIdx, &nextDraft})
    {
        TORCH_CHECK(t->device() == ctrl.device(),
            "minimax_m3_sm107_tp8_fold_tail_next: the next-token operands must be on ctrl's device (", ctrl.device(),
            ")");
    }
    TORCH_CHECK(nextDraft.size(0) >= rows && srcNext.size(0) >= rows && srcNext.size(1) == nextDraft.size(1) + 1,
        "minimax_m3_sm107_tp8_fold_tail_next: src_next must be [rows, D + 1] for next_draft [rows, D]");
    TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp8::FoldTailNextParams q{};
    q.accepted = accepted.data_ptr<int32_t>();
    q.acc0 = accepted.stride(0);
    q.numAccepted = numAccepted.data_ptr<int32_t>();
    q.batchIdx = batchIdx.data_ptr<int32_t>();
    q.nextDraft = nextDraft.data_ptr<int32_t>();
    q.nd0 = nextDraft.stride(0);
    q.draftLen = static_cast<int32_t>(nextDraft.size(1));
    q.next = srcNext.data_ptr<int32_t>();
    q.nx0 = srcNext.stride(0);
    auto const stream = at::cuda::getCurrentCUDAStream(ctrl.get_device());
    auto const err
        = TRTLLM_MV2_NS::kernels::minimaxM3Sm107Tp8::invokeFoldTailNext(p, q, static_cast<int32_t>(rows), stream);
    TORCH_CHECK(err == cudaSuccess, "minimax_m3_sm107_tp8_fold_tail_next launch failed: ", cudaGetErrorString(err));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m3_sm107_tp8_fold_tail_next(Tensor ctrl, Tensor slots, int rows, Tensor src_new, Tensor(a!) dst_new, "
        "Tensor(i!) src_next, Tensor(b!) dst_next, Tensor src_draft, Tensor(c!) dst_draft, Tensor src_lens, "
        "Tensor(d!) dst_lens, Tensor(e!) ids, Tensor(f!) draft_ids, Tensor(g!) pos_off, Tensor(h!) kv_off, "
        "int tokens_per_step, Tensor accepted, Tensor num_accepted, Tensor batch_idx, Tensor next_draft) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m3_sm107_tp8_fold_tail_next", &TRTLLM_MV2_NS::torch_ext::minimaxM3Sm107Tp8FoldTailNext);
}
