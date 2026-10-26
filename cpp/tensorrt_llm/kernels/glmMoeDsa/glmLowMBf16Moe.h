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

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{

//! Routed experts and experts per token the low-M BF16 MoE kernels are built for.
constexpr int kLowMBf16MoeNumExperts = 256;
constexpr int kLowMBf16MoeTopK = 8;

//! Whether invokeLowMBf16Moe supports a layer: hiddenSize a multiple of 2048 and at most 8192, interSize a multiple
//! of 64 and at most 512.
bool lowMBf16MoeSupportsShape(int64_t hiddenSize, int64_t interSize);

//! One token through an MoE layer with unquantized BF16 SwiGLU experts held as tensor-parallel slices (no expert
//! parallelism) and DeepSeek-V3 routing with one expert group, in two kernels:
//!   1. route, then for each routed expert e_k: act[k, :] = bf16(silu(bf16(W1[e_k] x)) * bf16(W3[e_k] x));
//!   2. route again, then out = bf16(sum_k bf16(w_k * (W2[e_k] act[k, :]))).
//! Routing is the noaux_tc_op computation for n_group == 1 (deepseek_v3_topk_kernel): the same experts, in the same
//! order, and bit-identical weights w_k (normalized sigmoid scores times routedScalingFactor). The BF16 rounding points
//! are those of the CUTLASS MoE path: the FC1 output, the SwiGLU output and each expert's scaled FC2 output. The sum
//! over the experts is in FP32 in a fixed order, where the CUTLASS finalize adds BF16 terms with atomics.
//!
//! The FC1 kernel runs one CTA per SM: a producer warp selects the experts and streams whole weight rows through a
//! shared-memory ring, which 8 consumer warps read two rows at a time. It releases the FC2 kernel right after its
//! grid-dependency wait, so that FC2 CTAs fit beside it and stage their weights in shared memory while FC1 streams.
//! Neither kernel runs FP64 instructions on its common path (FP64 lowers the SM clock): the FC2 kernel forms the
//! weights in FP32 with the result of noaux_tc_op's FP64 expression, which it runs only where that is not certain.
//!
//! Under PDL the FC1 kernel reads x and the logits after its grid-dependency wait; the FC2 kernel reads only the
//! logits, the bias and the FC2 weights before its own wait, which is safe because the FC1 kernel releases it after
//! its wait has returned.
//!
//! \param out           Output [hiddenSize].
//! \param act           Scratch [kLowMBf16MoeTopK, interSize] for the SwiGLU outputs, 16-byte aligned.
//! \param x             Input token [hiddenSize], 16-byte aligned.
//! \param routerLogits  Router logits [kLowMBf16MoeNumExperts], FP32.
//! \param bias          Routing correction bias [kLowMBf16MoeNumExperts].
//! \param w3w1          FC1 weights [kLowMBf16MoeNumExperts, 2 * interSize, hiddenSize], 16-byte aligned: rows
//!                      [0, interSize) of an expert are W3 (up), rows [interSize, 2 * interSize) are W1 (gate), the
//!                      layout CutlassFusedMoE stores.
//! \param w2            FC2 weights [kLowMBf16MoeNumExperts, hiddenSize, interSize], 16-byte aligned.
template <typename BiasT>
void invokeLowMBf16Moe(__nv_bfloat16* out, __nv_bfloat16* act, __nv_bfloat16 const* x, float const* routerLogits,
    BiasT const* bias, __nv_bfloat16 const* w3w1, __nv_bfloat16 const* w2, int hiddenSize, int interSize,
    double routedScalingFactor, cudaStream_t stream);

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
