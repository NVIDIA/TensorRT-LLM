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
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

//! SWIGLU_MXFP8 decode (the SWQ producer, SWQ_CPP TP2 / SWQ16 TP8): the swigluoai activation of x = [gate | up]
//! (bit for bit the Triton silu_and_mul at limit 7, alpha 1.702, beta 1, rounded to bf16) quantized in registers to
//! E4M3 [rows, d] with UE8M0 K32 scales in mxfp8_quantize(h, True)'s 128x4-swizzled layout; the scale bytes of the
//! row padding to a multiple of 128 and of the K32-block padding to a multiple of 4 are zeroed. Each K32 block's code
//! rounds amax * rcp.approx(448) toward +inf (capped at 254); its values are scaled by rcp.approx(2^(code - 127)).
//! Scale code 0 (0 < amax <= 448 * 2^-127):
//!   finiteCode0 = true:  scaled by the exact 2^127 (the Triton m3_swq rule; every byte is m3_swq's);
//!   finiteCode0 = false: quantized to zero like an all-zero block (every byte is the in-tree silu_and_mul +
//!                        mxfp8_quantize pair's, which publishes code 0 with zero data).
//! Shaped for decode rows: one CTA per padded row, ceil32(d / 8) threads (one 8-column chunk per live thread; the
//! lanes past d / 8 only zero the block padding of their row), no trailing fence (completion is implicit at exit),
//! PDL wait only. d must be a multiple of 32 and at most 8192; xRowStride (elements, a multiple of 8, >= 2d) is x's
//! row pitch; x 16-byte aligned; qOut [rows, d] contiguous (8-byte aligned); sfOut pad128(rows) * pad4(d / 32) bytes.
cudaError_t invokeSwigluOaiMxfp8Decode(void const* x, int64_t xRowStride, void* qOut, uint8_t* sfOut, int64_t rows,
    int64_t d, bool finiteCode0, bool pdl, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
