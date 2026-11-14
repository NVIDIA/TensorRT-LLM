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
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

constexpr int kDflashMergeHeadDim = 128;
constexpr int kDflashMergeMaxBlock = 16;

//! Operands of invokeDflashMergeLaterKeys; strides in elements, every last dim unit-stride.
struct DflashMergeParams
{
    __nv_bfloat16* out;         // [B * blk, Hq, D]: strides outRow, outHead
    float const* lseLog2;       // [B * blk, Hq]: strides lseRow, lseHead
    __nv_bfloat16 const* query; // [B, blk, Hq, D]: strides qB, qT, qH
    __nv_bfloat16 const* key;   // [B, blk, Hkv, D]: strides kB, kT, kH
    __nv_bfloat16 const* value; // [B, blk, Hkv, D]: strides vB, vT, vH
    int64_t outRow, outHead, lseRow, lseHead, qB, qT, qH, kB, kT, kH, vB, vT, vH;
    int B, blk, numHeads, group, windowRight;
    float scale;
};

//! The DFlash / DSpark non-causal block merge (modeling_dflash.dflash_merge_later_block_keys_), in place on out: each
//! block query j of request b adds its later block keys j < k <= j + windowRight (all later keys when windowRight < 0)
//! to its causal attention output, given the base-2 log-sum-exp of that output:
//!   logit[k] = scale * sum_d float(q[b, j, h, d]) * float(key[b, k, h / group, d])
//!   m = max_k logit[k]; s = sum_k exp(logit[k] - m) (k ascending); w = sigmoid(m + log(s) - lseLog2[r, h] * ln 2)
//!   out[r, h, :] = bf16(float(out) * (1 - w) + sum_k (w / s * exp(logit[k] - m)) * float(value[b, k, h / group, :]))
//! A row without a later key keeps its output. One warp per (row r = b * blk + j, head h), each lane holding 4 of the
//! 128 head_dim values (8-byte loads and stores: every base pointer 8-byte aligned and every outer stride a multiple of
//! 4); precise expf / logf. Not bitwise the eager form (fp32 reduction order): <= 1 bf16 ULP on |out| >= 1/16 at the
//! DEP8 drafter shapes. PDL: waits on its predecessor before the first load, releases its dependents after the last
//! store. Requires head_dim 128 and blk <= 16.
cudaError_t invokeDflashMergeLaterKeys(DflashMergeParams const& params, bool pdl, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
