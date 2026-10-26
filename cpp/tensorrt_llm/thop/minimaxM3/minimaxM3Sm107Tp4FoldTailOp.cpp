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

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp4/foldTail.h"
#include "tensorrt_llm/thop/minimaxM3/minimaxM3Sm107Tp4FoldTailParams.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! The speculative step's forward-tail store and next-step staging in one launch (the C++ form of the Triton
//! kernels/m3_fold_tail.py, the same writes). The caller checks the row counts against the buffers (fold_tail).
static void foldTailImpl(bool oneWarp, torch::Tensor const& ctrl, torch::Tensor const& slots, int64_t rows,
    torch::Tensor const& srcNew, torch::Tensor& dstNew, torch::Tensor const& srcNext, torch::Tensor& dstNext,
    torch::Tensor const& srcDraft, torch::Tensor& dstDraft, torch::Tensor const& srcLens, torch::Tensor& dstLens,
    torch::Tensor& ids, torch::Tensor& draftIds, torch::Tensor& posOff, torch::Tensor& kvOff, int64_t tokensPerStep)
{
    c10::cuda::CUDAGuard const deviceGuard{ctrl.device()};
    auto const p = minimaxM3Sm107Tp4FoldTailParams::buildParams(ctrl, slots, rows, srcNew, dstNew, srcNext, dstNext,
        srcDraft, dstDraft, srcLens, dstLens, ids, draftIds, posOff, kvOff, tokensPerStep);
    auto const stream = at::cuda::getCurrentCUDAStream(ctrl.get_device());
    auto const err = oneWarp
        ? tensorrt_llm::kernels::minimaxM3Sm107Tp4::invokeFoldTailOneWarp(p, stream)
        : tensorrt_llm::kernels::minimaxM3Sm107Tp4::invokeFoldTail(p, static_cast<int32_t>(rows), stream);
    TORCH_CHECK(err == cudaSuccess, "minimax_m3_sm107_tp4_fold_tail launch failed: ", cudaGetErrorString(err));
}

//! The tail as its standalone kernel (one CTA per row).
void minimaxM3Sm107Tp4FoldTail(torch::Tensor const& ctrl, torch::Tensor const& slots, int64_t rows,
    torch::Tensor const& srcNew, torch::Tensor& dstNew, torch::Tensor const& srcNext, torch::Tensor& dstNext,
    torch::Tensor const& srcDraft, torch::Tensor& dstDraft, torch::Tensor const& srcLens, torch::Tensor& dstLens,
    torch::Tensor& ids, torch::Tensor& draftIds, torch::Tensor& posOff, torch::Tensor& kvOff, int64_t tokensPerStep)
{
    foldTailImpl(false, ctrl, slots, rows, srcNew, dstNew, srcNext, dstNext, srcDraft, dstDraft, srcLens, dstLens, ids,
        draftIds, posOff, kvOff, tokensPerStep);
}

//! The tail's in-kernel form (one warp over the rows, foldTailWarp), the oracle of a caller that folds it in.
void minimaxM3Sm107Tp4FoldTailOneWarp(torch::Tensor const& ctrl, torch::Tensor const& slots, int64_t rows,
    torch::Tensor const& srcNew, torch::Tensor& dstNew, torch::Tensor const& srcNext, torch::Tensor& dstNext,
    torch::Tensor const& srcDraft, torch::Tensor& dstDraft, torch::Tensor const& srcLens, torch::Tensor& dstLens,
    torch::Tensor& ids, torch::Tensor& draftIds, torch::Tensor& posOff, torch::Tensor& kvOff, int64_t tokensPerStep)
{
    foldTailImpl(true, ctrl, slots, rows, srcNew, dstNew, srcNext, dstNext, srcDraft, dstDraft, srcLens, dstLens, ids,
        draftIds, posOff, kvOff, tokensPerStep);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "minimax_m3_sm107_tp4_fold_tail(Tensor ctrl, Tensor slots, int rows, Tensor src_new, Tensor(a!) dst_new, "
        "Tensor src_next, Tensor(b!) dst_next, Tensor src_draft, Tensor(c!) dst_draft, Tensor src_lens, "
        "Tensor(d!) dst_lens, Tensor(e!) ids, Tensor(f!) draft_ids, Tensor(g!) pos_off, Tensor(h!) kv_off, "
        "int tokens_per_step) -> ()");
    m.def(
        "minimax_m3_sm107_tp4_fold_tail_one_warp(Tensor ctrl, Tensor slots, int rows, Tensor src_new, Tensor(a!) "
        "dst_new, Tensor src_next, Tensor(b!) dst_next, Tensor src_draft, Tensor(c!) dst_draft, Tensor src_lens, "
        "Tensor(d!) dst_lens, Tensor(e!) ids, Tensor(f!) draft_ids, Tensor(g!) pos_off, Tensor(h!) kv_off, "
        "int tokens_per_step) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m3_sm107_tp4_fold_tail", &tensorrt_llm::torch_ext::minimaxM3Sm107Tp4FoldTail);
    m.impl("minimax_m3_sm107_tp4_fold_tail_one_warp", &tensorrt_llm::torch_ext::minimaxM3Sm107Tp4FoldTailOneWarp);
}
