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

namespace kernels::minimaxM31Sm107
{

//! TP2CTX_SP_ADDNORM (spAddNormqKernel): each owned row is the two-rank sum of this rank's slot row (local) and the
//! peer's slot row (peer, read over NVLink), rounded to BF16 as the peer add rounds it; interOut gets that sum plus the
//! residual, and the rest is the CTXNORMQ kernel's arithmetic (residual RMSNorm -> MXFP8 qOut / sfOut; normOut
//! optional, required with f32Out; NVFP4 fp4Out / fp4SfOut iff fp4GlobalScale). Persistent grid of
//! min(rows, numSms * ctasPerSm) CTAs of hidden / 8 threads; every grid gives the same bits. hidden % 256 == 0,
//! hidden / 8 <= 1024.
cudaError_t invokeSpAddNormq(__nv_bfloat16 const* local, __nv_bfloat16 const* peer, __nv_bfloat16* interOut,
    __nv_bfloat16 const* residual, __nv_bfloat16 const* weight, __nv_bfloat16* normOut, uint64_t* qOut, uint32_t* sfOut,
    uint32_t* fp4Out, uint32_t* fp4SfOut, float const* fp4GlobalScale, float* f32Out, int rows, int hidden, float eps,
    int numSms, int ctasPerSm, bool enablePdl, cudaStream_t stream);

} // namespace kernels::minimaxM31Sm107

TRTLLM_NAMESPACE_END
