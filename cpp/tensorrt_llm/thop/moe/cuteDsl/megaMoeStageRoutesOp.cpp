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

// The route staging of a MegaMoE CuteDSL launch in one op: the kernel reads every row of its bucket, so the rows past
// the launch's tokens must hold id -1 and weight 0 on every launch, also when a CUDA graph replays it. With the
// forward's live-row word, the rows of its padding requests are staged the same way, read on the device so that a
// replayed graph follows the word of each forward.

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/moe/cuteDsl/megaMoeStageRoutes.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cstdint>
#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

template <typename IdT, typename OutT>
void launchStageRoutes(at::Tensor const& ids, at::Tensor const& weights, int32_t const* liveRowWord, at::Tensor& idsOut,
    at::Tensor& weightsOut)
{
    kernels::megaMoeStageRoutes<IdT, OutT>(ids.data_ptr<IdT>(), weights.data_ptr<float>(), ids.size(0), liveRowWord,
        idsOut.data_ptr<OutT>(), weightsOut.data_ptr<float>(), idsOut.size(0), static_cast<int>(idsOut.size(1)),
        at::cuda::getCurrentCUDAStream(idsOut.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

// ids: [n, top_k] int32 or int64 expert ids; weights: [n, top_k] float32. ids_out: [rows, top_k] int32 or int64 with
// rows >= n; weights_out: [rows, top_k] float32. All contiguous on one device. Rows [0, live) of the outputs get the
// ids and weights, rows [live, rows) id -1 and weight 0, where live is n, or with live_row_word (one int32 on the same
// device) kernels::megaMoeLiveRows of it and n.
void megaMoeStageRoutes(at::Tensor const& ids, at::Tensor const& weights, at::Tensor& idsOut, at::Tensor& weightsOut,
    std::optional<at::Tensor> const& liveRowWord)
{
    auto const isId = [](at::Tensor const& t) { return t.scalar_type() == at::kInt || t.scalar_type() == at::kLong; };
    TORCH_CHECK(isId(ids) && isId(idsOut), "ids and ids_out must be int32 or int64");
    TORCH_CHECK(weights.scalar_type() == at::kFloat && weightsOut.scalar_type() == at::kFloat,
        "weights and weights_out must be float32");
    TORCH_CHECK(idsOut.dim() == 2 && idsOut.is_contiguous() && weightsOut.sizes() == idsOut.sizes()
            && weightsOut.is_contiguous(),
        "ids_out and weights_out must be contiguous [rows, top_k] buffers");
    TORCH_CHECK(ids.dim() == 2 && ids.size(1) == idsOut.size(1) && ids.size(0) <= idsOut.size(0)
            && weights.sizes() == ids.sizes() && ids.is_contiguous() && weights.is_contiguous(),
        "ids and weights must be contiguous [n, top_k] with n at most the rows of ids_out");
    TORCH_CHECK(ids.device() == idsOut.device() && weights.device() == idsOut.device()
            && weightsOut.device() == idsOut.device(),
        "all tensors must be on one device");
    TORCH_CHECK(!liveRowWord.has_value()
            || (liveRowWord->scalar_type() == at::kInt && liveRowWord->numel() == 1
                && liveRowWord->device() == idsOut.device()),
        "live_row_word must be one int32 on the outputs' device");
    auto const n = ids.size(0);
    auto const rows = idsOut.size(0);
    if (!idsOut.is_cuda())
    {
        // The reference staging, in torch ops: copy_ widens the ids.
        auto const live = liveRowWord.has_value() ? kernels::megaMoeLiveRows(liveRowWord->item<int32_t>(), n) : n;
        idsOut.narrow(0, 0, live).copy_(ids.narrow(0, 0, live));
        idsOut.narrow(0, live, rows - live).fill_(-1);
        weightsOut.narrow(0, 0, live).copy_(weights.narrow(0, 0, live));
        weightsOut.narrow(0, live, rows - live).zero_();
        return;
    }
    c10::cuda::CUDAGuard const deviceGuard(idsOut.device());
    int32_t const* word = liveRowWord.has_value() ? liveRowWord->data_ptr<int32_t>() : nullptr;
    bool const wideIds = ids.scalar_type() == at::kLong;
    if (idsOut.scalar_type() == at::kLong)
    {
        wideIds ? launchStageRoutes<int64_t, int64_t>(ids, weights, word, idsOut, weightsOut)
                : launchStageRoutes<int32_t, int64_t>(ids, weights, word, idsOut, weightsOut);
    }
    else
    {
        wideIds ? launchStageRoutes<int64_t, int32_t>(ids, weights, word, idsOut, weightsOut)
                : launchStageRoutes<int32_t, int32_t>(ids, weights, word, idsOut, weightsOut);
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "megamoe_stage_routes(Tensor ids, Tensor weights, Tensor(a!) ids_out, Tensor(b!) weights_out, "
        "Tensor? live_row_word=None) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CPU, m)
{
    m.impl("megamoe_stage_routes", &tensorrt_llm::torch_ext::megaMoeStageRoutes);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("megamoe_stage_routes", &tensorrt_llm::torch_ext::megaMoeStageRoutes);
}
