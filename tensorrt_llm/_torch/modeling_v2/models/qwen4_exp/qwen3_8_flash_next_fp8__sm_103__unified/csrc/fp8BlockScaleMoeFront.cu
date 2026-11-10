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

#include "qwen4_exp/qwen3_8_flash_next_fp8__sm_103__unified/csrc/fp8BlockScaleMoeFront.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/moe/trtllmGen/routing/RoutingKernel.cuh"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <array>
#include <mutex>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::fp8_block_scale_moe_front
{

namespace
{

namespace cg = cooperative_groups;
namespace routing = ::moe::dev::routing;
using routing::WarpSize;

constexpr int NumThreads = 256;
constexpr int NumWarps = NumThreads / WarpSize;
constexpr int VecSize = MaxNumExperts / WarpSize;
constexpr int ScanItemsPerThread = MaxNumExperts / NumThreads;
static_assert(MaxNumExperts % NumThreads == 0);

// Per-device state, zeroed at allocation: two banks of per-local-expert counters (a launch counts in bank
// generation % 2 and zeroes the other), then the grid barrier's arrival count and generation flag.
constexpr int CounterInts = 2 * MaxNumExperts + 64;
constexpr int BarrierCount = 2 * MaxNumExperts;
constexpr int BarrierFlag = 2 * MaxNumExperts + 32;
constexpr int QuantIters = QuantHidden / 256; // two 1x128 groups a warp iteration

__device__ __forceinline__ int32_t ldAcquire(int32_t const* p)
{
    int32_t v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void stRelease(int32_t* p, int32_t v)
{
    asm volatile("st.release.gpu.global.b32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ int32_t atomAddAcqRel(int32_t* p, int32_t v)
{
    int32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

// Bitwise `quantize_1x128_lane` (fp8_blockscale_gemm_kernel.cuh) for one 1x128 group over 16 lanes of 8 bf16 each.
__device__ __forceinline__ void quantizeGroup(uint4 in, __nv_fp8_e4m3* out, float* scale, int laneInGroup)
{
    union
    {
        uint4 pack;
        __nv_bfloat16 v[8];
    } x;

    x.pack = in;
    __nv_bfloat16 laneAmax = __nv_bfloat16(0);
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        laneAmax = __hmax(laneAmax, __habs(x.v[i]));
    }
    float groupAmax = float(laneAmax);
#pragma unroll
    for (int mask = 8; mask > 0; mask /= 2)
    {
        groupAmax = std::max(groupAmax, __shfl_xor_sync(0xFFFFFFFF, groupAmax, mask));
    }
    __nv_bfloat16 amax = ::tensorrt_llm::common::cuda_max(__nv_bfloat16(groupAmax), __nv_bfloat16(1e-10f));
    float quantScale = 448.f / float(amax);
    float dequantScale = 1.f / quantScale;
    if (laneInGroup == 0)
    {
        *scale = dequantScale;
    }

    union
    {
        uint2 pack;
        __nv_fp8_e4m3 v[8];
    } y;

#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        y.v[i] = __nv_fp8_e4m3(float(x.v[i]) * quantScale);
    }
    *reinterpret_cast<uint2*>(out) = y.pack;
}

// One warp a token selects its experts with the trtllm-gen routing device functions and takes each slot's offset in
// its local expert from a global atomic counter. After a grid barrier of the routing CTAs, every CTA scans the counts
// into expert offsets and writes its own slots' permutation; CTA 0 writes the grouped-GEMM launch tables.
__global__ void __launch_bounds__(NumThreads) moeFrontKernel(Data params, int32_t* state)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    using InputT = __nv_bfloat16;
    using BaseType = __nv_bfloat16;
    using TypePacked = routing::PackedScoreIdx<BaseType>;
    using Scan = cub::BlockScan<int32_t, NumThreads, cub::BLOCK_SCAN_WARP_SCANS>;

    __shared__ typename Scan::TempStorage tempStorage;
    __shared__ int32_t smemExpertOffset[MaxNumExperts];

    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    int32_t const laneIdx = threadIdx.x % WarpSize;
    int32_t const tokenIdx = blockIdx.x * NumWarps + warpIdx;
    bool const validToken = tokenIdx < params.numTokens;
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    int32_t* barrierCount = state + BarrierCount;
    int32_t* barrierFlag = state + BarrierFlag;

    if (params.usePdl)
    {
        cudaGridDependencySynchronize();
    }

    int32_t const routeCtas = (params.numTokens + NumWarps - 1) / NumWarps;
    if (static_cast<int32_t>(blockIdx.x) >= routeCtas)
    {
        // Quantization CTAs, outside the routing CTAs' barrier: half a hidden row a warp.
        int32_t const half = (blockIdx.x - routeCtas) * NumWarps + warpIdx;
        int32_t const tokenQ = half / 2;
        if (tokenQ < params.numTokens)
        {
            int32_t const it0 = (half % 2) * (QuantIters / 2);
            auto const* in
                = static_cast<__nv_bfloat16 const*>(params.ptrHiddenBf16) + static_cast<int64_t>(tokenQ) * QuantHidden;
            uint4 part[QuantIters / 2];
#pragma unroll
            for (int it = 0; it < QuantIters / 2; it++)
            {
                part[it] = *reinterpret_cast<uint4 const*>(in + (it0 + it) * 256 + laneIdx * 8);
            }
            auto* out = static_cast<__nv_fp8_e4m3*>(params.ptrHiddenFp8) + static_cast<int64_t>(tokenQ) * QuantHidden;
#pragma unroll
            for (int it = 0; it < QuantIters / 2; it++)
            {
                int32_t const group = 2 * (it0 + it) + laneIdx / 16;
                quantizeGroup(part[it], out + (it0 + it) * 256 + laneIdx * 8,
                    params.ptrHiddenScale + group * params.numTokens + tokenQ, laneIdx % 16);
            }
        }
        __syncthreads();
        if (params.usePdl)
        {
            cudaTriggerProgrammaticLaunchCompletion();
        }
        return;
    }

    BaseType const minScore = BaseType{-INFINITY};
    BaseType score[VecSize];
    int32_t idx[VecSize];
    if (validToken)
    {
        InputT const* scores = static_cast<InputT const*>(params.ptrScores) + tokenIdx * params.numExperts;
#pragma unroll
        for (int i = 0; i < VecSize; i++)
        {
            auto expertIdx = i * WarpSize + laneIdx;
            score[i] = expertIdx < params.numExperts ? static_cast<BaseType>(scores[expertIdx]) : minScore;
            idx[i] = expertIdx;
        }
    }

    // No CTA flips the generation before every CTA, this one included, has arrived; the previous launch flipped it
    // before it completed.
    int32_t const generation = *reinterpret_cast<int32_t volatile*>(barrierFlag);
    int32_t* counters = state + (generation & 1) * MaxNumExperts;
    if (blockIdx.x == 0)
    {
        // The other bank is the previous launch's, which is complete (this grid depends on it transitively).
        int32_t* other = state + ((generation + 1) & 1) * MaxNumExperts;
        for (int e = threadIdx.x; e < MaxNumExperts; e += NumThreads)
        {
            other[e] = 0;
        }
    }

    int32_t slotExpert = -1; // this lane's slot (lane < topK): local expert, or -1
    int32_t slotOffset = 0;
    if (validToken)
    {
        BaseType warpTopKScore[MaxTopK];
        int32_t warpTopKExpertIdx[MaxTopK];
        routing::topk::reduceTopK(warp, warpTopKScore, warpTopKExpertIdx, score, idx, minScore, params.topK);
        BaseType const softmaxScore = routing::calcSoftmax(
            warp, laneIdx < params.topK ? warpTopKScore[laneIdx] : minScore, laneIdx, params.topK);
        if (laneIdx < params.topK)
        {
            // Round through the packed (score, idx) as the runner's cluster kernel does.
            TypePacked const packed{softmaxScore, static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
            int32_t const expandedIdx = tokenIdx * params.topK + laneIdx;
            static_cast<__nv_bfloat16*>(params.ptrTopKWeights)[expandedIdx] = __nv_bfloat16{packed.score};
            int32_t const localExpertIdx = static_cast<int32_t>(packed.idx) - params.localExpertsStartIdx;
            if (localExpertIdx >= 0 && localExpertIdx < params.numLocalExperts)
            {
                slotExpert = localExpertIdx;
                slotOffset = atomicAdd(counters + localExpertIdx, 1);
            }
        }
    }

    // Grid barrier of the routing CTAs, all resident (a few dozen CTAs, and no awaited kernel holds the SMs): one
    // acq_rel arrival releases this CTA's counter atomics and acquires the others', so the counts are final after it.
    __syncthreads();
    if (threadIdx.x == 0)
    {
        if (atomAddAcqRel(barrierCount, 1) == routeCtas - 1)
        {
            *barrierCount = 0;
            stRelease(barrierFlag, generation + 1);
        }
        else
        {
            while (ldAcquire(barrierFlag) == generation)
            {
            }
        }
    }
    __syncthreads();

    int32_t const tileN = params.tileTokensDim;
    int32_t count[ScanItemsPerThread];
    int32_t numCta[ScanItemsPerThread];
#pragma unroll
    for (int e = 0; e < ScanItemsPerThread; e++)
    {
        int32_t const expert = threadIdx.x * ScanItemsPerThread + e;
        count[e] = expert < params.numLocalExperts ? __ldcg(counters + expert) : 0;
    }
#pragma unroll
    for (int e = 0; e < ScanItemsPerThread; e++)
    {
        numCta[e] = (count[e] + tileN - 1) / tileN;
    }
    int32_t ctaOffset[ScanItemsPerThread];
    int32_t numNonExitingCtas;
    Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);
#pragma unroll
    for (int e = 0; e < ScanItemsPerThread; e++)
    {
        int32_t const expert = threadIdx.x * ScanItemsPerThread + e;
        smemExpertOffset[expert] = ctaOffset[e] * tileN;
        if (blockIdx.x == 0 && expert < params.numLocalExperts)
        {
            for (int32_t cta = 0; cta < numCta[e]; cta++)
            {
                params.ptrCtaIdxXyToBatchIdx[ctaOffset[e] + cta] = expert;
                int32_t const mnLimit1 = (ctaOffset[e] + cta + 1) * tileN;
                int32_t const mnLimit2 = ctaOffset[e] * tileN + count[e];
                params.ptrCtaIdxXyToMnLimit[ctaOffset[e] + cta] = min(mnLimit1, mnLimit2);
            }
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        params.ptrPermutedIdxSize[0] = numNonExitingCtas * tileN;
        params.ptrNumNonExitingCtas[0] = numNonExitingCtas;
    }
    __syncthreads();

    if (validToken && laneIdx < params.topK)
    {
        int32_t const expandedIdx = tokenIdx * params.topK + laneIdx;
        if (slotExpert >= 0)
        {
            int32_t const permutedIdx = smemExpertOffset[slotExpert] + slotOffset;
            params.ptrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
            params.ptrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
        }
        else
        {
            params.ptrExpandedIdxToPermutedIdx[expandedIdx] = -1;
        }
    }

    // The consumers (activation, both grouped GEMMs, finalize) read every table written above.
    __syncthreads();
    if (params.usePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
}

int32_t* deviceCounters(int device)
{
    static std::mutex mutex;
    static std::array<int32_t*, 64> buffers{};
    std::lock_guard<std::mutex> lock(mutex);
    TLLM_CHECK_WITH_INFO(device >= 0 && device < static_cast<int>(buffers.size()), "device %d out of range", device);
    if (buffers[device] == nullptr)
    {
        // May run during stream capture: allocate in relaxed mode and zero on a private stream outside the capture.
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        TLLM_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        int32_t* ptr = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&ptr, CounterInts * sizeof(int32_t)));
        cudaStream_t side;
        TLLM_CUDA_CHECK(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
        TLLM_CUDA_CHECK(cudaMemsetAsync(ptr, 0, CounterInts * sizeof(int32_t), side));
        TLLM_CUDA_CHECK(cudaStreamSynchronize(side));
        TLLM_CUDA_CHECK(cudaStreamDestroy(side));
        TLLM_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        buffers[device] = ptr;
    }
    return buffers[device];
}

} // namespace

bool supports(int32_t numTokens, int32_t numExperts, int32_t topK, int32_t numLocalExperts, int32_t hiddenDim)
{
    return numTokens > 0 && numTokens <= MaxNumTokens && numExperts > 0 && numExperts <= MaxNumExperts && topK > 0
        && topK <= MaxTopK && numLocalExperts > 0 && numLocalExperts <= MaxNumExperts && hiddenDim % 128 == 0;
}

void run(Data const& data, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(supports(data.numTokens, data.numExperts, data.topK, data.numLocalExperts, data.hiddenDim),
        "fp8_block_scale_moe_front: unsupported problem (tokens %d, experts %d, topK %d, local %d, hidden %d)",
        data.numTokens, data.numExperts, data.topK, data.numLocalExperts, data.hiddenDim);
    TLLM_CHECK_WITH_INFO(data.ptrHiddenBf16 == nullptr
            || (data.ptrHiddenFp8 != nullptr && data.ptrHiddenScale != nullptr && data.hiddenDim == QuantHidden),
        "fp8_block_scale_moe_front: quantization needs both outputs and hidden %d", QuantHidden);
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    int32_t* counters = deviceCounters(device);

    cudaLaunchConfig_t config{};
    int32_t const routeCtas = (data.numTokens + NumWarps - 1) / NumWarps;
    int32_t const quantCtas = data.ptrHiddenBf16 != nullptr ? (2 * data.numTokens + NumWarps - 1) / NumWarps : 0;
    config.gridDim = routeCtas + quantCtas;
    config.blockDim = NumThreads;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attributes[1] = {};
    attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[0].val.programmaticStreamSerializationAllowed = int(data.usePdl);
    config.attrs = attributes;
    config.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, moeFrontKernel, data, counters));
}

} // namespace kernels::fp8_block_scale_moe_front

TRTLLM_MV2_NAMESPACE_END
