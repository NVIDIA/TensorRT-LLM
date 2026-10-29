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

#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmFusedEmbeddingLookup.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"

#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{
namespace
{

constexpr int kThreadsPerBlock = 128;
// BF16 elements per 16-byte vector.
constexpr int kElemsPerVec = 8;

// blockIdx.x selects the token, blockIdx.y the 128-vector slice of its row; each thread copies one vector.
template <typename IdT>
__global__ void __launch_bounds__(kThreadsPerBlock) fusedEmbeddingLookupKernel(uint4* __restrict__ out,
    IdT const* __restrict__ ids, uint4 const* __restrict__ table, int64_t vocabSize, int32_t vecsPerRow)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // The ids may be written by the preceding kernel. Dependents read the output only after their own wait.
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    int32_t const vec = static_cast<int32_t>(blockIdx.y * kThreadsPerBlock + threadIdx.x);
    if (vec >= vecsPerRow)
    {
        return;
    }
    int64_t const token = blockIdx.x;
    int64_t const id = static_cast<int64_t>(ids[token]);

    // Ids outside [0, vocabSize), such as the -1 padding of rejected draft tokens, give a zero row.
    uint4 value = make_uint4(0U, 0U, 0U, 0U);
    if (id >= 0 && id < vocabSize)
    {
        value = table[id * vecsPerRow + vec];
    }
    out[token * vecsPerRow + vec] = value;
}

} // namespace

void invokeFusedEmbeddingLookup(__nv_bfloat16* out, void const* ids, bool idsAreInt64, __nv_bfloat16 const* table,
    int64_t numTokens, int64_t vocabSize, int64_t hiddenSize, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(hiddenSize % kElemsPerVec == 0, "fusedEmbeddingLookup needs a hidden size divisible by 8");
    if (numTokens == 0)
    {
        return;
    }

    int32_t const vecsPerRow = static_cast<int32_t>(hiddenSize / kElemsPerVec);
    int32_t const blocksPerRow = (vecsPerRow + kThreadsPerBlock - 1) / kThreadsPerBlock;
    dim3 const grid(static_cast<uint32_t>(numTokens), static_cast<uint32_t>(blocksPerRow));
    dim3 const block(kThreadsPerBlock);
    auto* outVec = reinterpret_cast<uint4*>(out);
    auto const* tableVec = reinterpret_cast<uint4 const*>(table);
    if (idsAreInt64)
    {
        tensorrt_llm::common::launchWithPdlWhenEnabled("fusedEmbeddingLookup", fusedEmbeddingLookupKernel<int64_t>,
            grid, block, 0, stream, outVec, static_cast<int64_t const*>(ids), tableVec, vocabSize, vecsPerRow);
    }
    else
    {
        tensorrt_llm::common::launchWithPdlWhenEnabled("fusedEmbeddingLookup", fusedEmbeddingLookupKernel<int32_t>,
            grid, block, 0, stream, outVec, static_cast<int32_t const*>(ids), tableVec, vocabSize, vecsPerRow);
    }
}

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
