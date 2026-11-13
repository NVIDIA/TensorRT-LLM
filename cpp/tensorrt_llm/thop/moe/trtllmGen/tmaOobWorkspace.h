/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/config.h"

#include <ATen/ATen.h>
#include <ATen/cuda/EmptyTensor.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

//! \brief Allocates a [rows, cols] GEMM workspace for the TMA-OOB ("rgTma") batchedGemm cubins.
//!
//! Their TMA-OOB descriptors run in OOB_ADDR_GEN_MODE_BASE_128kB on Blackwell, so the TMA unit may probe up to a
//! 128 KiB-aligned boundary past the base. The returned tensor therefore starts on a 128 KiB boundary and has at
//! least 128 KiB of mapped memory past its end; empty_cuda alone gives 512 B alignment, and a pool slice near a block
//! end then faults sporadically. The tensor is a view of its backing allocation and keeps it alive.
inline at::Tensor allocTmaOobWorkspace(int64_t rows, int64_t cols, at::ScalarType dtype, c10::Device device)
{
    constexpr int64_t kTmaOobAlignBytes = 128 * 1024;
    int64_t const dataBytes = rows * cols * static_cast<int64_t>(c10::elementSize(dtype));
    // One alignment of slack to round the base up, one of headroom past the end.
    at::Tensor const backing = at::detail::empty_cuda(
        {dataBytes + 2 * kTmaOobAlignBytes}, at::ScalarType::Byte, device, /*memory_format_opt=*/std::nullopt);
    auto const base = reinterpret_cast<uintptr_t>(backing.data_ptr());
    auto const aligned = (base + kTmaOobAlignBytes - 1) & ~static_cast<uintptr_t>(kTmaOobAlignBytes - 1);
    return backing.narrow(0, static_cast<int64_t>(aligned - base), dataBytes).view(dtype).view({rows, cols});
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
