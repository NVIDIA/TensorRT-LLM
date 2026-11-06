/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

//! Publish each top-k-group member's owner list and gather member zero. Caller-owned storage must
//! outlive captured users. Initialize ownerTable to -1 and ownerState (two int32 values) to zero;
//! the leader restores both after each call. memberStride counts int32 pairs, not bytes.
void invokeGlmRubinNvfp4WorklistPipelined(uint8_t const* dataPool, __nv_fp8_e4m3 const* scalePool,
    int32_t const* memberIndices, int32_t numMembers, int32_t const* queryReqIndices, __nv_fp8_e4m3* output,
    int32_t* compactIndices, int64_t* ownerTable, int32_t* memberOwners, int64_t memberStride, int32_t* ownerCount,
    int32_t* ownerState, float const* globalDequantScale, int32_t numRows, int32_t topK, int32_t numRequests,
    int32_t tableSlots, int64_t numPoolTokens, cudaStream_t stream = 0);

//! Gather a follower's rows using the matching leader's owner list and count on the same stream.
void invokeGlmRubinNvfp4WorklistMember(uint8_t const* dataPool, __nv_fp8_e4m3 const* scalePool,
    int32_t const* memberOwners, int32_t const* ownerCount, __nv_fp8_e4m3* output, int32_t* compactIndices,
    float const* globalDequantScale, int32_t numRows, int32_t topK, cudaStream_t stream = 0);

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
