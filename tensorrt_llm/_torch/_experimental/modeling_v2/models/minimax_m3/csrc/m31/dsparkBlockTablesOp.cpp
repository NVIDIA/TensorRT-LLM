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

#include "minimax_m3/csrc/m31/dsparkBlockTables.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

//! ops/dspark/dspark_block_tables.refresh_block_tables: src = offsets[pool_idx, :num_seqs, 0] (int32 / int64
//! [rows, width], any strides) decoded into tables[:rows] (int32, unit column stride) and counted into counts[:rows]
//! (int64, contiguous), in place.
void minimaxM31DsparkRefreshBlockTables(
    torch::Tensor const& src, int64_t divisor, torch::Tensor& tables, torch::Tensor& counts)
{
    char const* op = "minimax_m31_dspark_refresh_block_tables";
    TORCH_CHECK(
        src.is_cuda() && (src.scalar_type() == torch::kInt32 || src.scalar_type() == torch::kInt64) && src.dim() == 2,
        op, ": offsets rows must be an int32 / int64 [rows, width] CUDA tensor");
    int64_t const rows = src.size(0);
    int64_t const width = src.size(1);
    TORCH_CHECK(tables.is_cuda() && tables.scalar_type() == torch::kInt32 && tables.dim() == 2 && tables.stride(1) == 1
            && tables.size(0) >= rows && tables.size(1) == width && tables.get_device() == src.get_device(),
        op, ": tables must be int32 [>= rows, width] with unit column stride on the offsets' device");
    TORCH_CHECK(counts.is_cuda() && counts.scalar_type() == torch::kInt64 && counts.dim() == 1 && counts.is_contiguous()
            && counts.size(0) >= rows && counts.get_device() == src.get_device(),
        op, ": counts must be a contiguous int64 [>= rows] tensor on the offsets' device");
    TORCH_CHECK(divisor >= 1 && rows <= std::numeric_limits<int32_t>::max()
            && width <= TRTLLM_MV2_NS::kernels::minimaxM31::kDsparkRefreshMaxWidth,
        op, ": invalid divisor / shape");
    if (rows == 0)
        return;
    at::cuda::CUDAGuard const guard(src.device());
    TRTLLM_MV2_NS::kernels::minimaxM31::invokeDsparkRefreshBlockTables(src.data_ptr(),
        src.scalar_type() == torch::kInt64, src.stride(0), src.stride(1), tables.data_ptr<int32_t>(), tables.stride(0),
        counts.data_ptr<int64_t>(), static_cast<int32_t>(rows), static_cast<int32_t>(width), divisor,
        at::cuda::getCurrentCUDAStream(src.get_device()));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_dspark_refresh_block_tables(Tensor offsets, int divisor, Tensor(a!) tables, "
        "Tensor(b!) counts) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_dspark_refresh_block_tables", &TRTLLM_MV2_NS::torch_ext::minimaxM31DsparkRefreshBlockTables);
}
