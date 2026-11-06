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

//! Most tokens invokeLowMBf16MoeMulti takes.
constexpr int kLowMBf16MoeMaxTokens = 6;

//! Whether invokeLowMBf16MoeMulti supports a layer and token count: 2 to kLowMBf16MoeMaxTokens tokens, hiddenSize a
//! multiple of 2048 and at most 6144, interSize a multiple of 64 and at most 512.
bool lowMBf16MoeMultiSupportsShape(int64_t numTokens, int64_t hiddenSize, int64_t interSize);

//! invokeLowMBf16Moe for 2 to kLowMBf16MoeMaxTokens tokens, in two kernels:
//!   1. route every token, then stream the FC1 rows of the union of the routed experts once, and for each token that
//!      routed to the expert: act[m, k, :] = bf16(silu(bf16(W1[e] x_m)) * bf16(W3[e] x_m)), k the expert's slot in
//!      token m's list;
//!   2. route again, stream the FC2 rows of the union, and out[m] = bf16(sum_k bf16(w_mk * (W2[e_mk] act[m, k, :]))).
//! Each output row is bit-identical to invokeLowMBf16Moe on that token alone: the same routing, the same per-row
//! reductions in the same order and the same rounding points. The union is in ascending expert order in both kernels.
//!
//! The FC1 kernel runs one CTA per SM, as invokeLowMBf16Moe's does, with x of every token in shared memory; its units
//! are (union expert, intermediate index) pairs, and an epilogue warp forms the SwiGLU outputs of each chunk of units
//! while the consumer warps go on. The FC2 kernel takes 16 output rows per CTA: one CTA fits beside each FC1 CTA,
//! routes and stages the FC2 weights of the first union experts in a ring of shared-memory stages before its
//! grid-dependency wait, and two share an SM once the FC1 CTAs exit; the ring streams the rest of the union after the
//! wait.
//!
//! Under PDL the FC1 kernel reads x and the logits after its grid-dependency wait and releases the FC2 kernel right
//! after it; the FC2 kernel reads only the logits, the bias and the FC2 weights before its own wait.
//!
//! \param out           Output [numTokens, hiddenSize].
//! \param act           Scratch [numTokens, kLowMBf16MoeTopK, interSize] for the SwiGLU outputs, 16-byte aligned.
//! \param x             Input tokens [numTokens, hiddenSize], 16-byte aligned.
//! \param routerLogits  Router logits [numTokens, kLowMBf16MoeNumExperts], FP32.
template <typename BiasT>
void invokeLowMBf16MoeMulti(__nv_bfloat16* out, __nv_bfloat16* act, __nv_bfloat16 const* x, float const* routerLogits,
    BiasT const* bias, __nv_bfloat16 const* w3w1, __nv_bfloat16 const* w2, int numTokens, int hiddenSize, int interSize,
    double routedScalingFactor, cudaStream_t stream);

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
