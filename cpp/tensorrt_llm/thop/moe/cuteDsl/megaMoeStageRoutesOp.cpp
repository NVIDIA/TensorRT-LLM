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
// the launch's tokens must hold id -1 and weight 0 on every launch, also when a CUDA graph replays it.

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/moe/cuteDsl/megaMoeStageRoutes.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

template <typename IdT, typename OutT>
void launchStageRoutes(at::Tensor const& ids, at::Tensor const& weights, at::Tensor& idsOut, at::Tensor& weightsOut)
{
    kernels::megaMoeStageRoutes<IdT, OutT>(ids.data_ptr<IdT>(), weights.data_ptr<float>(), ids.size(0),
        idsOut.data_ptr<OutT>(), weightsOut.data_ptr<float>(), idsOut.size(0), static_cast<int>(idsOut.size(1)),
        at::cuda::getCurrentCUDAStream(idsOut.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

// ids: [n, top_k] int32 or int64 expert ids; weights: [n, top_k] float32. ids_out: [rows, top_k] int32 or int64 with
// rows >= n; weights_out: [rows, top_k] float32. All contiguous on one device. Rows [0, n) of the outputs get the ids
// and weights, rows [n, rows) id -1 and weight 0.
void megaMoeStageRoutes(at::Tensor const& ids, at::Tensor const& weights, at::Tensor& idsOut, at::Tensor& weightsOut)
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
    auto const n = ids.size(0);
    auto const rows = idsOut.size(0);
    if (!idsOut.is_cuda())
    {
        // The reference staging, in torch ops: copy_ widens the ids.
        idsOut.narrow(0, 0, n).copy_(ids);
        idsOut.narrow(0, n, rows - n).fill_(-1);
        weightsOut.narrow(0, 0, n).copy_(weights);
        weightsOut.narrow(0, n, rows - n).zero_();
        return;
    }
    c10::cuda::CUDAGuard const deviceGuard(idsOut.device());
    bool const wideIds = ids.scalar_type() == at::kLong;
    if (idsOut.scalar_type() == at::kLong)
    {
        wideIds ? launchStageRoutes<int64_t, int64_t>(ids, weights, idsOut, weightsOut)
                : launchStageRoutes<int32_t, int64_t>(ids, weights, idsOut, weightsOut);
    }
    else
    {
        wideIds ? launchStageRoutes<int64_t, int32_t>(ids, weights, idsOut, weightsOut)
                : launchStageRoutes<int32_t, int32_t>(ids, weights, idsOut, weightsOut);
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("megamoe_stage_routes(Tensor ids, Tensor weights, Tensor(a!) ids_out, Tensor(b!) weights_out) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CPU, m)
{
    m.impl("megamoe_stage_routes", &tensorrt_llm::torch_ext::megaMoeStageRoutes);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("megamoe_stage_routes", &tensorrt_llm::torch_ext::megaMoeStageRoutes);
}
