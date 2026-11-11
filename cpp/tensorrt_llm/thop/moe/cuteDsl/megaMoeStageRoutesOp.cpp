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
#include <torch/library.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

template <typename IdT, typename OutT>
void stageRoutes(at::Tensor const& ids, at::Tensor const& weights, at::Tensor& idsOut, at::Tensor& weightsOut)
{
    auto const rows = idsOut.size(0);
    auto const topK = idsOut.size(1);
    auto const live = ids.size(0) * topK;
    auto const* id = ids.data_ptr<IdT>();
    auto const* weight = weights.data_ptr<float>();
    auto* idOut = idsOut.data_ptr<OutT>();
    auto* weightOut = weightsOut.data_ptr<float>();
    if (idsOut.is_cuda())
    {
        kernels::megaMoeStageRoutes<IdT, OutT>(id, weight, ids.size(0), idOut, weightOut, weightsOut.stride(0), rows,
            static_cast<int>(topK), at::cuda::getCurrentCUDAStream(idsOut.get_device()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    for (int64_t i = 0; i < rows * topK; ++i)
    {
        idOut[i] = i < live ? static_cast<OutT>(id[i]) : OutT(-1);
        weightOut[i / topK * weightsOut.stride(0) + i % topK] = i < live ? weight[i] : 0.F;
    }
}

} // namespace

// ids: [n, top_k] int32 or int64 expert ids; weights: [n, top_k] float32. ids_out: [rows, top_k] int32 or int64 with
// rows >= n; weights_out: [rows, top_k] float32 with a unit column stride. All contiguous on one device. Rows [0, n)
// of the outputs get the ids and weights, rows [n, rows) id -1 and weight 0.
void megaMoeStageRoutes(at::Tensor const& ids, at::Tensor const& weights, at::Tensor& idsOut, at::Tensor& weightsOut)
{
    TORCH_CHECK(ids.defined() && weights.defined(), "ids and weights are required ([0, top_k] for no rows)");
    auto const isId = [](at::Tensor const& t) { return t.scalar_type() == at::kInt || t.scalar_type() == at::kLong; };
    TORCH_CHECK(isId(ids) && isId(idsOut), "ids and ids_out must be int32 or int64");
    TORCH_CHECK(weights.scalar_type() == at::kFloat && weightsOut.scalar_type() == at::kFloat,
        "weights and weights_out must be float32");
    TORCH_CHECK(idsOut.dim() == 2 && idsOut.is_contiguous() && weightsOut.dim() == 2
            && weightsOut.sizes() == idsOut.sizes() && weightsOut.stride(1) == 1,
        "ids_out must be a contiguous [rows, top_k] buffer and weights_out [rows, top_k] with a unit column stride");
    TORCH_CHECK(ids.dim() == 2 && ids.size(1) == idsOut.size(1) && ids.size(0) <= idsOut.size(0)
            && weights.sizes() == ids.sizes() && ids.is_contiguous() && weights.is_contiguous(),
        "ids and weights must be contiguous [n, top_k] with n at most the rows of ids_out");
    TORCH_CHECK(ids.device() == idsOut.device() && weights.device() == idsOut.device()
            && weightsOut.device() == idsOut.device(),
        "all tensors must be on one device");
    bool const wideIds = ids.scalar_type() == at::kLong;
    if (idsOut.scalar_type() == at::kLong)
    {
        wideIds ? stageRoutes<int64_t, int64_t>(ids, weights, idsOut, weightsOut)
                : stageRoutes<int32_t, int64_t>(ids, weights, idsOut, weightsOut);
    }
    else
    {
        wideIds ? stageRoutes<int64_t, int32_t>(ids, weights, idsOut, weightsOut)
                : stageRoutes<int32_t, int32_t>(ids, weights, idsOut, weightsOut);
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
