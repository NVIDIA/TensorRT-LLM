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
#include <cuda_bf16.h>
#include <cuda_fp16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

using heuristic_topk::BLOCK_SIZE;
using heuristic_topk::GvrDtypeTraits;
using heuristic_topk::GvrParams;
using heuristic_topk::gvrTopKJob;
using heuristic_topk::gvrTopKJobDtype;
using heuristic_topk::KernelSmemTplK;

// Templated on TopK so the launcher can dispatch K=512/1024/2048 to the
// same kernel template. Smem layout is derived from GvrParams<float, TopK>
// at compile time.
template <int TopK>
__global__ void __launch_bounds__(BLOCK_SIZE) heuristicTopKMultiRowKernel(float const* __restrict__ logits,
    int const* __restrict__ seqLens, int const* __restrict__ preIdx, float* __restrict__ scratchValues,
    int* __restrict__ outIndices, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount)
{
    using SmemT = KernelSmemTplK<float, GvrParams<float, TopK>::kC, GvrParams<float, TopK>::kNumBins>;

    int const rowIdx = blockIdx.x;
    int const seq_len = seqLens[rowIdx / next_n];
    int const N = seq_len - next_n + (rowIdx % next_n) + 1;

    float const* __restrict__ input = logits + static_cast<int64_t>(rowIdx) * stride0;
    int const* __restrict__ rowPreIdx = preIdx + static_cast<int64_t>(rowIdx / next_n) * preIdxStride;
    float* __restrict__ outputValues = scratchValues + static_cast<int64_t>(rowIdx) * topK;
    int* __restrict__ outputIndices = outIndices + static_cast<int64_t>(rowIdx) * topK;

    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<SmemT*>(smem_raw);

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
    gvrTopKJob<TopK>(input, N, rowPreIdx, preIdxCount, topK, outputValues, outputIndices, smem, preIdxOffset);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ============================================================================
// Multi-dtype path (bf16 / fp16)
// ============================================================================
// Mirrors heuristicTopKMultiRowKernel for bf16/fp16 inputs. The kernel body
// is structurally identical; only the input/output dtype, the smem-key
// dtype, and the GVR job (gvrTopKJobDtype<InputT>) differ.

// Templated on (InputT, TopK). Smem layout is derived from
// GvrParams<InputT, TopK>.
template <typename InputT, int TopK>
__global__ void __launch_bounds__(BLOCK_SIZE) heuristicTopKMultiRowKernelDtype(InputT const* __restrict__ logits,
    int const* __restrict__ seqLens, int const* __restrict__ preIdx, InputT* __restrict__ scratchValues,
    int* __restrict__ outIndices, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount)
{
    // dtype path uses fp32 keys[] in smem (down-conversion deferred to writeback).
    using SmemT = KernelSmemTplK<float, GvrParams<InputT, TopK>::kC, GvrParams<InputT, TopK>::kNumBins>;

    int const rowIdx = blockIdx.x;
    int const seq_len = seqLens[rowIdx / next_n];
    int const N = seq_len - next_n + (rowIdx % next_n) + 1;

    InputT const* __restrict__ input = logits + static_cast<int64_t>(rowIdx) * stride0;
    int const* __restrict__ rowPreIdx = preIdx + static_cast<int64_t>(rowIdx / next_n) * preIdxStride;
    InputT* __restrict__ outputValues = scratchValues + static_cast<int64_t>(rowIdx) * topK;
    int* __restrict__ outputIndices = outIndices + static_cast<int64_t>(rowIdx) * topK;

    extern __shared__ unsigned char smem_raw[];
    auto* smem = reinterpret_cast<SmemT*>(smem_raw);

    if (N <= topK)
    {
        int const tid = threadIdx.x;
        for (int i = tid; i < N; i += BLOCK_SIZE)
        {
            outputValues[i] = input[i];
            outputIndices[i] = i;
        }
        InputT const neg_max = GvrDtypeTraits<InputT>::from_fp32(-FLT_MAX);
        for (int i = N + tid; i < topK; i += BLOCK_SIZE)
        {
            outputValues[i] = neg_max;
            outputIndices[i] = -1;
        }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaTriggerProgrammaticLaunchCompletion();
#endif
        return;
    }

    int const preIdxOffset = (rowIdx % next_n) + 1;
    gvrTopKJobDtype<InputT, TopK>(
        input, N, rowPreIdx, preIdxCount, topK, outputValues, outputIndices, smem, preIdxOffset);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Explicit instantiations — 6 (dtype × K) combos. Launchers dispatch on
// runtime topK via switch, so all 6 must be available at link time.
template __global__ void heuristicTopKMultiRowKernelDtype<__nv_bfloat16, 512>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernelDtype<__nv_bfloat16, 1024>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernelDtype<__nv_bfloat16, 2048>(
    __nv_bfloat16 const*, int const*, int const*, __nv_bfloat16*, int*, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernelDtype<__half, 512>(
    __half const*, int const*, int const*, __half*, int*, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernelDtype<__half, 1024>(
    __half const*, int const*, int const*, __half*, int*, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernelDtype<__half, 2048>(
    __half const*, int const*, int const*, __half*, int*, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernel<512>(
    float const*, int const*, int const*, float*, int*, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernel<1024>(
    float const*, int const*, int const*, float*, int*, int, int, int, int, int);
template __global__ void heuristicTopKMultiRowKernel<2048>(
    float const*, int const*, int const*, float*, int*, int, int, int, int, int);

// Dispatch on topK at runtime — each TopK-instantiation gets its own smem
// size (driven by GvrParams<InputT, TopK>::kC/kNumBins) and own kfn pointer
// (cudaFuncSetAttribute / cudaLaunchKernelEx target the right kernel).
//
// fp32 routes to heuristicTopKMultiRowKernel<TopK>; bf16/fp16 route to
// heuristicTopKMultiRowKernelDtype<InputT, TopK>. Vector-load alignment
// requirement is 4 elements for fp32 (float4) and 8 elements for bf16/fp16
// (int4 of 16-bit). In TRT-LLM the logits stride is always a multiple of
// tokens_per_block (≥64), so the alignment check is never hit at runtime
// — it's an assert against caller misuse.
template <typename InputT>
void launchHeuristicTopKDecodeImpl(InputT const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    InputT* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(
        topK == 512 || topK == 1024 || topK == 2048, "heuristicTopKDecode requires topK ∈ {512, 1024, 2048}");

    constexpr int kAlign = std::is_same_v<InputT, float> ? 4 : 8;
    TLLM_CHECK_WITH_INFO(stride0 % kAlign == 0 || numRows <= 1,
        "heuristicTopKDecode requires logits stride0 divisible by %d for multi-row launch", kAlign);

    auto launchOne = [&]<int TopK>()
    {
        // bf16/fp16 path also uses fp32 keys[] in smem (down-conversion deferred).
        using SmemT = KernelSmemTplK<float, GvrParams<InputT, TopK>::kC, GvrParams<InputT, TopK>::kNumBins>;
        size_t const smemSize = sizeof(SmemT);

        auto kfn = []()
        {
            if constexpr (std::is_same_v<InputT, float>)
                return heuristicTopKMultiRowKernel<TopK>;
            else
                return heuristicTopKMultiRowKernelDtype<InputT, TopK>;
        }();

        if (smemSize > 48u * 1024u)
        {
            cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemSize));
        }

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

        cudaLaunchKernelEx(&config, kfn, logits, seqLens, preIdx, scratchValues, outIndices, stride0, next_n, topK,
            preIdxStride, preIdxCount);
    };

    switch (topK)
    {
    case 512: launchOne.template operator()<512>(); break;
    case 1024: launchOne.template operator()<1024>(); break;
    case 2048: launchOne.template operator()<2048>(); break;
    default: TLLM_THROW("heuristicTopKDecode: topK validated above; unreachable");
    }
}

} // anonymous namespace

void launchHeuristicTopKDecode(float const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    float* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    cudaStream_t stream)
{
    launchHeuristicTopKDecodeImpl<float>(logits, seqLens, preIdx, outIndices, scratchValues, stride0, next_n, topK,
        preIdxStride, preIdxCount, numRows, stream);
}

void launchHeuristicTopKDecode(__nv_bfloat16 const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    __nv_bfloat16* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    cudaStream_t stream)
{
    launchHeuristicTopKDecodeImpl<__nv_bfloat16>(logits, seqLens, preIdx, outIndices, scratchValues, stride0, next_n,
        topK, preIdxStride, preIdxCount, numRows, stream);
}

void launchHeuristicTopKDecode(__half const* logits, int const* seqLens, int const* preIdx, int* outIndices,
    __half* scratchValues, int stride0, int next_n, int topK, int preIdxStride, int preIdxCount, int numRows,
    cudaStream_t stream)
{
    launchHeuristicTopKDecodeImpl<__half>(logits, seqLens, preIdx, outIndices, scratchValues, stride0, next_n, topK,
        preIdxStride, preIdxCount, numRows, stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
