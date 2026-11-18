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

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>
#include <cuda_runtime_api.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

//! Rearranges MXFP4 block scales, `[experts, rows, cols]` E8M0 bytes, into the 128x4 swizzled layout the MoE
//! kernel reads: within an expert, the byte at `(m, c)` moves to offset
//! `(m / 128) * 512 * (cols / 4) + (c / 4) * 512 + (m % 32) * 16 + ((m % 128) / 32) * 4 + c % 4`.
//! Requires `rows % 128 == 0` and `cols % 4 == 0`.
cudaError_t invokeSwizzleBlockScales(
    uint8_t const* in, uint8_t* out, int64_t experts, int64_t rows, int64_t cols, cudaStream_t stream);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
