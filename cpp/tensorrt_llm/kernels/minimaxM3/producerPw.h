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
#include <cuda_fp8.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3
{
//! The MiniMax-M3 NVFP4 QKV + indexer RMSNorm / RoPE / KV-insert producer with its constant loads before the PDL
//! grid-dependency wait and an optional early dependents trigger (sm_103a). Same outputs, bit for bit, as
//! minimaxM3Nvfp4QKVIndexerNormRopeKVInsert. See producerPw.cu.
namespace producerPw
{

struct Params
{
    __nv_bfloat16 const* packed;
    __nv_fp8_e4m3* qOut;
    __nv_fp8_e4m3* indexQOut;
    uint8_t* kvData;
    uint8_t* kvScale;
    __nv_fp8_e4m3* indexK;
    int const* outCacheLoc;
    float const* kvQuantScale;
    int64_t dataPageStride, dataPlaneStride, dataHeadStride, dataTokenStride;
    int64_t scalePageStride, scalePlaneStride, scaleHeadStride;
    int64_t indexPageStride, indexTokenStride, numPages;
    int numTokens;
    float eps;
    __nv_bfloat16 const* qWeight;
    __nv_bfloat16 const* kWeight;
    __nv_bfloat16 const* indexQWeight;
    __nv_bfloat16 const* indexKWeight;
    float const* rotaryCosSin;
    int const* positionIds;
};

//! Launch over `numTokens` tokens, one warp per (token, packed head); headsQ is 8 or 16.
cudaError_t invokeProducerPw(Params const& p, int numTokens, int headsQ, bool pdl, bool early, cudaStream_t stream);

} // namespace producerPw
} // namespace minimaxM3
} // namespace kernels

TRTLLM_NAMESPACE_END
