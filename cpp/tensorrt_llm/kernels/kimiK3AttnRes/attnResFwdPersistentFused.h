/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "attnResFwd.h"

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kimiK3AttnRes
{

//! Persistent-grid counterpart to invokeAttnResAddRmsNormFwd.
//!
//! Same AttnResFwdParams, different topology: one CTA per SM looping over
//! tokens, instead of one CTA (or one cluster) per token. That makes it the
//! shape to reach for when there are far more tokens than SMs -- prefill --
//! where the per-token kernels stop being able to amortize the score
//! projection load. The shape contract is below.
//!
//! Requires B == 1, H == 7168, N in [2, 9] and bf16 contiguous tensors. Unlike
//! the per-token kernels there is no N == 1 case: the persistent kernel always
//! has at least one snapshot plus the layer residual, and it produces no
//! rsigma/probs/logits. T is bounded only by the int token index: offsets that
//! grow with T are formed in 64 bits wherever they can pass INT_MAX, and the
//! token index needs T - 1 + grid <= INT_MAX, where grid is the persistent grid
//! size; the launch checks that bound.
//!
//! outputRmsWeight is required -- this entry point exists for the fully fused
//! form. layerResidualAdd is optional: shapes with no residual add to fold
//! would otherwise be pushed back onto the per-token path. When it is non-null
//! updatedLayerResidual must be non-null too (see
//! attnResPersistentFusedSupported): the kernel writes the updated prefix sum
//! there rather than folding the add into layerResidual in place.
void invokeAttnResPersistentFusedFwd(AttnResFwdParams const& params, cudaStream_t stream);

//! True when B, H, N, T >= 1 and the required tensors above all hold, so
//! callers can route without duplicating those checks.
//! invokeAttnResPersistentFusedFwd also rejects non-null rsigma/probs/logits
//! and, at launch, a T past the token index bound, which depends on the grid
//! size.
bool attnResPersistentFusedSupported(AttnResFwdParams const& params);

//! invokeAttnResPersistentFusedFwd with two optional extensions (TRTLLM_K3_ATTN_RES_PERSISTENT_FUSE),
//! each bit-identical to the kernel launch it replaces:
//!
//! - layerResidualAdd2 (needs base.layerResidualAdd): the updated prefix sum is
//!   layerResidual + bf16(layerResidualAdd + layerResidualAdd2). With the routed and the shared
//!   expert outputs as the two addends this is the MoE residual update of kimi_k3_moe_residual
//!   (residual + bf16(routed + shared)), folded into the next layer's pre-attention mix.
//! - fp8Output / fp8Scale (both or neither): the normalized output also as E4M3 codes [T, H] and
//!   UE8M0 1x128 scales in the 128x4-swizzled R128c4 layout of the Rubin CuTe DSL MXFP8 GEMM
//!   ([roundUp(T, 128) * H / 32] bytes, padded rows zero), byte-identical to
//!   fp8_quantize_1x128_cutedsl_ue8m0 of base.output.
//!
//! The candidate stride is formed in int: T * H * (N - 2) must fit (T <= 42799 at N = 9).
struct AttnResPersistentExtParams
{
    AttnResFwdParams base;
    __nv_bfloat16 const* layerResidualAdd2; //!< [T, B, H], optional
    void* fp8Output;                        //!< [T, H] E4M3, optional
    uint8_t* fp8Scale;                      //!< R128c4 UE8M0, optional (with fp8Output)
};

//! attnResPersistentFusedSupported plus the extension contract above.
bool attnResPersistentFusedExtSupported(AttnResPersistentExtParams const& params);

void invokeAttnResPersistentFusedExtFwd(AttnResPersistentExtParams const& params, cudaStream_t stream);

} // namespace kernels::kimiK3AttnRes

TRTLLM_NAMESPACE_END
