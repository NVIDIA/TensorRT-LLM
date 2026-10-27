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

#pragma once

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp4/foldTail.h"

#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace minimaxM3Sm107Tp4FoldTailParams
{

constexpr int64_t kMaxWidth = 32;

inline void check2d(torch::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kInt32 && t.dim() == 2 && t.size(1) <= kMaxWidth,
        "minimax_m3_sm107_tp4_fold_tail: ", name, " must be a 2-D int32 CUDA view at most 32 wide");
}

inline void check1d(torch::Tensor const& t, char const* name)
{
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kInt32 && t.dim() == 1 && t.stride(0) == 1,
        "minimax_m3_sm107_tp4_fold_tail: ", name, " must be a contiguous 1-D int32 CUDA tensor");
}

//! The fold tail's operands checked and packed (the standalone fold tail op and any op that runs the tail in-kernel,
//! e.g. the drafter's last-pass draft_step, build them identically).
inline tensorrt_llm::kernels::minimaxM3Sm107Tp4::FoldTailParams buildParams(torch::Tensor const& ctrl,
    torch::Tensor const& slots, int64_t rows, torch::Tensor const& srcNew, torch::Tensor& dstNew,
    torch::Tensor const& srcNext, torch::Tensor& dstNext, torch::Tensor const& srcDraft, torch::Tensor& dstDraft,
    torch::Tensor const& srcLens, torch::Tensor& dstLens, torch::Tensor& ids, torch::Tensor& draftIds,
    torch::Tensor& posOff, torch::Tensor& kvOff, int64_t tokensPerStep)
{
    TORCH_CHECK(ctrl.is_cuda() && ctrl.scalar_type() == torch::kInt32 && ctrl.numel() >= 1,
        "minimax_m3_sm107_tp4_fold_tail: ctrl must be an int32 CUDA tensor");
    TORCH_CHECK(slots.is_cuda() && slots.scalar_type() == torch::kInt64 && slots.dim() == 1 && slots.stride(0) == 1,
        "minimax_m3_sm107_tp4_fold_tail: slots must be a contiguous 1-D int64 CUDA tensor");
    TORCH_CHECK(rows > 0 && tokensPerStep > 0 && tokensPerStep <= kMaxWidth,
        "minimax_m3_sm107_tp4_fold_tail: rows must be positive and tokens_per_step in [1, 32]");
    check2d(srcNew, "src_new");
    check2d(dstNew, "dst_new");
    check2d(srcNext, "src_next");
    check2d(dstNext, "dst_next");
    check2d(srcDraft, "src_draft");
    check2d(dstDraft, "dst_draft");
    check1d(srcLens, "src_lens");
    check1d(dstLens, "dst_lens");
    check1d(ids, "ids");
    check1d(draftIds, "draft_ids");
    check1d(posOff, "pos_off");
    check1d(kvOff, "kv_off");
    tensorrt_llm::kernels::minimaxM3Sm107Tp4::FoldTailParams p{};
    p.ctrl = ctrl.data_ptr<int32_t>();
    p.slots = slots.data_ptr<int64_t>();
    p.srcNew = srcNew.data_ptr<int32_t>();
    p.sn0 = srcNew.stride(0);
    p.sn1 = srcNew.stride(1);
    p.snW = static_cast<int32_t>(srcNew.size(1));
    p.dstNew = dstNew.data_ptr<int32_t>();
    p.dn0 = dstNew.stride(0);
    p.dn1 = dstNew.stride(1);
    p.dnW = static_cast<int32_t>(dstNew.size(1));
    p.srcNext = srcNext.data_ptr<int32_t>();
    p.sx0 = srcNext.stride(0);
    p.sx1 = srcNext.stride(1);
    p.sxW = static_cast<int32_t>(srcNext.size(1));
    p.dstNext = dstNext.data_ptr<int32_t>();
    p.dx0 = dstNext.stride(0);
    p.dx1 = dstNext.stride(1);
    p.dxW = static_cast<int32_t>(dstNext.size(1));
    p.srcDraft = srcDraft.data_ptr<int32_t>();
    p.sd0 = srcDraft.stride(0);
    p.sd1 = srcDraft.stride(1);
    p.sdW = static_cast<int32_t>(srcDraft.size(1));
    p.dstDraft = dstDraft.data_ptr<int32_t>();
    p.dd0 = dstDraft.stride(0);
    p.dd1 = dstDraft.stride(1);
    p.ddW = static_cast<int32_t>(dstDraft.size(1));
    p.srcLens = srcLens.data_ptr<int32_t>();
    p.dstLens = dstLens.data_ptr<int32_t>();
    p.ids = ids.data_ptr<int32_t>();
    p.draftIds = draftIds.data_ptr<int32_t>();
    p.posOff = posOff.data_ptr<int32_t>();
    p.kvOff = kvOff.data_ptr<int32_t>();
    p.tokensPerStep = static_cast<int32_t>(tokensPerStep);
    return p;
}

} // namespace minimaxM3Sm107Tp4FoldTailParams
} // namespace torch_ext

TRTLLM_NAMESPACE_END
