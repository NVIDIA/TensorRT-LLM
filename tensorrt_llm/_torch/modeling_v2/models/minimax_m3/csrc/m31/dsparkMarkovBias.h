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

//! DSPARK_MARKOV_BIAS_CPP (replaces the Triton _prev_embeddings_kernel of
//! ops/dspark/dspark_markov_bias.py, bit for bit): out[b * K + k, :] = w1[prev, :] (bf16, R columns), with
//! prev = firstPrev[b] (k = 0) or tokens[b * K + k - 1] (int32 / int64, the *Is64 flags), and a +0.0 row when prev is
//! outside [0, V). A bf16 bit copy. R is a power of two in [16, 1024]; w1 / out 16-byte aligned. PDL-aware.
void invokeDsparkPrevEmbeddings(void const* firstPrev, bool firstPrevIs64, void const* tokens, bool tokensIs64,
    void const* w1, void* out, int32_t B, int32_t K, int64_t V, int32_t R, cudaStream_t stream);

//! DSPARK_MARKOV_BIAS_CPP (replaces the Triton _widen_add_kernel, bit for bit): out[r, c] = float(logits[r *
//! logitsRowStride + c]) + float(bias[r * N + c]) (fp32 add.rn of two exactly widened bf16), fp32 [M, N], M = B * K.
//! 16-byte vectors when every operand allows it, then 8-byte, then scalar. PDL-aware.
void invokeDsparkWidenAdd(void const* logits, int64_t logitsRowStride, void const* bias, float* out, int64_t M,
    int64_t N, cudaStream_t stream);

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
