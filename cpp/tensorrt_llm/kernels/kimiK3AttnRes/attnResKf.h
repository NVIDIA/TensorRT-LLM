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

#include "tensorrt_llm/common/config.h"

#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kimiK3AttnRes
{

//! Kimi K3 attention-residual mix + optional residual add + output RMSNorm, H = 7168, BF16.
//! With N = numSnapshots + 1 candidates v_0..v_{N-2} = block[k] and v_{N-1} = updated,
//! where updated = bf16(layer + add) (hasAdd) or a copy of layer: rs_i = rsqrt(mean(v_i^2) + rmsEps),
//! logit_i = sum_h v_ih * rs_i * rmsW_h * resW_h, p = softmax(logit), mixed = sum_i p_i v_i in FP32, and
//! out = bf16(mixed * rsqrt(mean(mixed^2) + outputRmsEps) * outW), rounded once, as in the persistent kernel
//! (invokeAttnResPersistentFusedFwd). The two reduce in a different order, so out is not bit-identical to it: on
//! a small fraction of the elements differ, none by more than one BF16 ulp or 2^-8 of the row RMS (relative
//! L2 < 1e-4); updated is bit-identical.
//! layer, add, updated and out are [M, 7168], block [numSnapshots, M, 7168], the weights [7168]; every pointer
//! 32-byte aligned; 1 <= numSnapshots <= 8. SM10x only (thread-block clusters, TMA bulk copies). The decode kernel
//! (M <= 100) splits each row over a 7-CTA cluster.
void invokeAttnResKfFwd(void const* layer, void const* add, void const* block, void const* resW, void const* rmsW,
    void const* outW, void* updated, void* out, int M, int numSnapshots, bool hasAdd, float rmsEps, float outputRmsEps,
    cudaStream_t stream);

} // namespace kernels::kimiK3AttnRes

TRTLLM_NAMESPACE_END
