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
#include "tensorrt_llm/kernels/kvCacheUtils.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{

// The MLA latent row of GLM-5.2: kv_lora_rank 512 latent columns followed by qk_rope_head_dim 64 K-RoPE columns.
constexpr int32_t kGlmKvHalfLatentDim = 512;
constexpr int32_t kGlmKvHalfRopeDim = 64;

// The KV half of MLA generation for the latent layout above, in one kernel: RMSNorm of the 512 latent columns of the
// raw kv_a_proj output, GPT-J RoPE of the 64 K-RoPE columns, E4M3 quantization at unit scale straight from FP32, and
// the append to the paged FP8 cache through kvCache, at the slot and RoPE position of the generation RoPE kernel
// (position kvLens[s] - seqLenQ + t for token t of sequence s; a negative position is skipped). Block 0 also writes
// the generation attention's scales: bmm1Scale = {softmaxScale, softmaxScale * log2(e)}, bmm2Scale = {1}.
// - latent: [numTokens] rows of at least 576 BF16 values, latentRowStride elements apart (a multiple of 8, rows
//   16-byte aligned);
// - cosSinCache: per position, 64 (cos, sin) float pairs (the duplicated-data table of the MLA rope), of which pair
//   p < 32 rotates K-RoPE columns 2p and 2p + 1;
// - kvNormWeight: the 512 kv_a_layernorm weights.
// PDL: the kernel waits on its grid dependency before any global read. With triggerAtEntry its dependents may launch
// at its entry, otherwise after its stores; either way they read its outputs only after their own grid-dependency
// wait.
void invokeGlmMlaKvHalfAppend(__nv_bfloat16 const* latent, int32_t latentRowStride, KVBlockArray const& kvCache,
    float2 const* cosSinCache, __nv_bfloat16 const* kvNormWeight, float kvNormEps, int32_t numTokens, int32_t seqLenQ,
    int32_t const* kvLens, float* bmm1Scale, float* bmm2Scale, float softmaxScale, bool triggerAtEntry,
    cudaStream_t stream);

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
