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
namespace minimaxM3Sm107Tp8
{
//! The MiniMax-M3 NVFP4 QKV + indexer RMSNorm / RoPE / KV-insert producer at the SM107 TP8 heads (8 query, 1 KV and
//! 1 index head per rank) with its constant loads before the PDL grid-dependency wait and an optional early dependents
//! trigger. Same outputs, bit for bit, as minimaxM3Nvfp4QKVIndexerNormRopeKVInsert. See producerPw8.cu.
namespace producerPw8
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

//! The heads per rank this producer is built for.
constexpr int kHeadsQ = 8;
constexpr int kHeadsKV = 1;
constexpr int kHeadsIndex = 1;

//! Launch over `numTokens` tokens, one warp per (token, packed head). `prewaitStepInputs` reads the position ids, the
//! cache slots and the RoPE rows before the grid-dependency wait (they must not be written by a PDL predecessor).
cudaError_t invokeProducerPw8(
    Params const& p, int numTokens, bool pdl, bool early, bool prewaitStepInputs, cudaStream_t stream);

} // namespace producerPw8
} // namespace minimaxM3Sm107Tp8
} // namespace kernels

TRTLLM_NAMESPACE_END
