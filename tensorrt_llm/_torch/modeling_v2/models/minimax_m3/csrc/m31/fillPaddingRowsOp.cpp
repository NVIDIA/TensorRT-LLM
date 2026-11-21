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

#include "minimax_m3/csrc/m31/fillPaddingRowsSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

//! DEP_PAD_FILL: in place, every row r of rows with is_padding[r] set to value (column 0 to col0 when given); the real
//! rows are neither read nor written. rows: contiguous 2-D float32 / bf16 / fp16 CUDA; is_padding: contiguous bool
//! [rows] on its device.
void minimaxM31FillPaddingRows(
    torch::Tensor& rows, torch::Tensor const& isPadding, double value, std::optional<double> col0, bool pdl)
{
    TORCH_CHECK(rows.is_cuda() && rows.dim() == 2 && rows.is_contiguous(),
        "minimax_m31_fill_padding_rows: contiguous 2-D CUDA rows");
    auto const dtype = rows.scalar_type();
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kBFloat16 || dtype == at::kHalf,
        "minimax_m31_fill_padding_rows: float32 / bf16 / fp16 rows");
    TORCH_CHECK(isPadding.scalar_type() == at::kBool && isPadding.dim() == 1 && isPadding.is_contiguous()
            && isPadding.numel() == rows.size(0) && isPadding.device() == rows.device(),
        "minimax_m31_fill_padding_rows: is_padding must be a contiguous bool [rows] on the rows' device");
    at::cuda::CUDAGuard const guard(rows.device());
    cudaError_t const e = TRTLLM_MV2_NS::kernels::minimaxM31::invokeFillPaddingRows(rows.data_ptr(),
        static_cast<int>(rows.element_size()), dtype == at::kBFloat16, isPadding.data_ptr<bool>(), rows.size(0),
        rows.size(1), static_cast<float>(value), static_cast<float>(col0.value_or(0.0)), col0.has_value(), pdl,
        at::cuda::getCurrentCUDAStream(rows.get_device()));
    TORCH_CHECK(e == cudaSuccess, "minimax_m31_fill_padding_rows launch: ", cudaGetErrorString(e));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_fill_padding_rows(Tensor(a!) rows, Tensor is_padding, float value, float? col0=None, bool "
        "pdl=True) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_fill_padding_rows", &TRTLLM_MV2_NS::torch_ext::minimaxM31FillPaddingRows);
}
