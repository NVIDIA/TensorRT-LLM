/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/heuristicTopKDecode.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/envUtils.h"

// Import gvrTopKJob (__device__ __noinline__, the GVR micro-kernel) and
// all helpers. gvrTopKJob is independently optimized by ptxas, matching standalone
// SASS quality regardless of the caller's prologue code.
#include "tensorrt_llm/kernels/heuristic_topk.cuh"

#include <cfloat>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

using heuristic_topk::BLOCK_SIZE;
using heuristic_topk::gvrTopKJob;
using heuristic_topk::KernelSmem;
using heuristic_topk::TOP_K;

// heuristicTopKMultiRowKernel — outer multi-row launch wrapper (1 CTA per row).
// Computes per-row parameters, then dispatches to the GVR micro-kernel
// (gvrTopKJob, single-CTA single-row, independently optimized device function).
__global__ void __launch_bounds__(BLOCK_SIZE) heuristicTopKMultiRowKernel(float const* __restrict__ logits,
    int const* __restrict__ seqLens, int const* __restrict__ preIdx, float* __restrict__ scratchValues,
    int* __restrict__ outIndices, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount)
{
    int const rowIdx = blockIdx.x;
    int const seq_len = seqLens[rowIdx / next_n];
    int const N = seq_len - next_n + (rowIdx % next_n) + 1;

    float const* __restrict__ input = logits + static_cast<int64_t>(rowIdx) * stride0;
    int const* __restrict__ rowPreIdx = preIdx + static_cast<int64_t>(rowIdx / next_n) * preIdxStride;
    float* __restrict__ outputValues = scratchValues + static_cast<int64_t>(rowIdx) * topK;
    int* __restrict__ outputIndices = outIndices + static_cast<int64_t>(rowIdx) * topK;

    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<KernelSmem*>(smem_raw);

    if (N <= topK)
    {
        int const tid = threadIdx.x;
        for (int i = tid; i < N; i += BLOCK_SIZE)
        {
            outputValues[i] = input[i];
            outputIndices[i] = i;
        }
        for (int i = N + tid; i < topK; i += BLOCK_SIZE)
        {
            outputValues[i] = -FLT_MAX;
            outputIndices[i] = -1;
        }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        return;
    }

    // +1 accounts for the temporal shift: prev_topk indices were computed at
    // seq_len-1, but the current step has one additional KV token appended.
    int const preIdxOffset = (rowIdx % next_n) + 1;
    gvrTopKJob(input, N, rowPreIdx, preIdxCount, topK, outputValues, outputIndices, smem, preIdxOffset);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

} // anonymous namespace

void launchHeuristicTopKDecode(float const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    float* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(topK == TOP_K, "heuristicTopKDecode requires topK == 2048 (compile-time constant)");

    size_t const smemSize = sizeof(KernelSmem);

    // Opt-in to extended shared memory. cudaFuncSetAttribute is device-scoped
    // and cheap — call unconditionally to be safe across multi-GPU processes.
    if (smemSize > 48u * 1024u)
    {
        cudaFuncSetAttribute(
            heuristicTopKMultiRowKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemSize));
    }

    // float4 loads require 16-byte-aligned (4-float) row pointers.
    // In TRT-LLM the logits stride is always a multiple of tokens_per_block (≥64),
    // so this condition is never hit. Assert rather than silently allocating.
    TLLM_CHECK_WITH_INFO(stride0 % 4 == 0 || numRows <= 1,
        "heuristicTopKDecode requires logits stride0 divisible by 4 for multi-row launch");

    cudaLaunchConfig_t config;
    config.gridDim = numRows;
    config.blockDim = BLOCK_SIZE;
    config.dynamicSmemBytes = smemSize;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;

    cudaLaunchKernelEx(&config, heuristicTopKMultiRowKernel, logits, seqLens, preIdx, scratchValues, outIndices,
        stride0, next_n, topK, preIdxStride, preIdxCount);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
