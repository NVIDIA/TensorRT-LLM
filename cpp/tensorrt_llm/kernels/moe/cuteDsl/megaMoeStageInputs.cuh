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

// One launch stages a MegaMoE launch's inputs. Its first blocks run the route staging and the others quantize the
// activation into the staging. Both jobs read only
// the stream's earlier work, so their latencies overlap instead of adding up as two small kernels in series.

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/moe/cuteDsl/megaMoeStageRoutes.h"
#include "tensorrt_llm/kernels/quantization.cuh"

#include <algorithm>
#include <mutex>
#include <unordered_map>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

constexpr int kMegaMoeStageThreads = 256;
// Vectors of eight elements a thread quantizes per item (a row's run of columns): a 256-thread block covers as many
// columns per item as fp4_quantize's 512-thread block covers per row, at most two vectors per thread.
constexpr int kMegaMoeQuantizeVecs = 2;
constexpr int kMegaMoeQuantizeItemVecs = kMegaMoeStageThreads * kMegaMoeQuantizeVecs;

//! The route staging of megaMoeStageRoutes: the writer's blocks stride over the bucket's elements.
template <typename IdT, typename OutT>
struct MegaMoeStageWriter
{
    IdT const* ids;
    float const* weights;
    int64_t rows;
    int32_t const* liveRowWord;
    OutT* idsOut;
    float* weightsOut;
    int64_t weightsRowStride;
    int64_t elements;
    int topK;
    int blocks;

    __device__ void operator()(int block) const
    {
        int64_t const liveElements = (liveRowWord == nullptr ? rows : megaMoeLiveRows(*liveRowWord, rows)) * topK;
        for (int64_t i = block * static_cast<int64_t>(kMegaMoeStageThreads) + threadIdx.x; i < elements;
             i += static_cast<int64_t>(blocks) * kMegaMoeStageThreads)
        {
            bool const live = i < liveElements;
            int64_t const row = i / topK;
            idsOut[i] = live ? static_cast<OutT>(ids[i]) : static_cast<OutT>(-1);
            weightsOut[row * weightsRowStride + (i - row * topK)] = live ? weights[i] : 0.F;
        }
    }
};

inline int64_t megaMoeQuantizeItems(MegaMoeQuantize const& q)
{
    int const vecs = q.cols / CVT_ELTS_PER_THREAD;
    return static_cast<int64_t>(q.rows) * ((vecs + kMegaMoeQuantizeItemVecs - 1) / kMegaMoeQuantizeItemVecs);
}

// Items [first, ...) in steps of stride, each with quantize_with_block_size's per-element math for bfloat16 input,
// 16-element UE4M3 scale factors and the linear layout: the same bytes, in a different order of threads.
__device__ __forceinline__ void megaMoeQuantize(MegaMoeQuantize const& q, int64_t first, int64_t stride)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    using Vec = PackedVec<__nv_bfloat16>;
    constexpr int kThreadsPerSf = 16 / CVT_ELTS_PER_THREAD;
    int const vecs = q.cols / CVT_ELTS_PER_THREAD;
    int const parts = (vecs + kMegaMoeQuantizeItemVecs - 1) / kMegaMoeQuantizeItemVecs;
    float const scale = q.globalScale == nullptr ? 1.0F : q.globalScale[0];
    auto const* in = reinterpret_cast<Vec const*>(q.in);
    for (int64_t item = first; item < static_cast<int64_t>(q.rows) * parts; item += stride)
    {
        int const row = static_cast<int>(item / parts);
        int const vec0 = static_cast<int>(item % parts) * kMegaMoeQuantizeItemVecs + static_cast<int>(threadIdx.x);
        int64_t const rowBase = static_cast<int64_t>(row) * vecs;
        // Both loads are issued before either vector is converted, so their memory round trips overlap.
        Vec vec[kMegaMoeQuantizeVecs];
#pragma unroll
        for (int v = 0; v < kMegaMoeQuantizeVecs; ++v)
        {
            if (vec0 + v * kMegaMoeStageThreads < vecs)
            {
                vec[v] = in[rowBase + vec0 + v * kMegaMoeStageThreads];
            }
        }
#pragma unroll
        for (int v = 0; v < kMegaMoeQuantizeVecs; ++v)
        {
            int const column = vec0 + v * kMegaMoeStageThreads;
            if (column < vecs)
            {
                // The linear layout's scale factor of the column's 16 elements, written by the first of its threads.
                uint8_t* sf = column % kThreadsPerSf == 0
                    ? q.sfOut + static_cast<int64_t>(row) * (q.cols / 16) + column / kThreadsPerSf
                    : nullptr;
                q.out[rowBase + column] = cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, false>(vec[v], scale, sf);
            }
        }
    }
#endif
}

template <typename IdT, typename OutT>
__global__ void __launch_bounds__(kMegaMoeStageThreads)
    megaMoeStageInputsKernel(MegaMoeQuantize quantize, MegaMoeStageWriter<IdT, OutT> writer)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize(); // launched early under PDL: the routes and the activation are earlier work
#endif
    if (static_cast<int>(blockIdx.x) < writer.blocks)
    {
        writer(static_cast<int>(blockIdx.x));
        return;
    }
    megaMoeQuantize(quantize, blockIdx.x - writer.blocks, gridDim.x - writer.blocks);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // As in fp4_quantize: every thread's stores are visible before a thread of the block lets dependents launch.
    __threadfence();
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

//! Launch writer.blocks route staging blocks and as many quantization blocks as stay resident on the device, at
//! most one per item. The quantization grid does not shrink by the writer's blocks: those finish almost at once, and a
//! large bucket would otherwise leave the quantization a few blocks.
template <typename IdT, typename OutT>
void launchMegaMoeStageInputs(
    MegaMoeQuantize const& quantize, MegaMoeStageWriter<IdT, OutT> const& writer, cudaStream_t stream)
{
    auto const kernel = megaMoeStageInputsKernel<IdT, OutT>;
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    int resident = 0;
    {
        static std::mutex mutex;
        static std::unordered_map<int, int> residentOf; // by device
        std::lock_guard<std::mutex> lock(mutex);
        auto it = residentOf.find(device);
        if (it == residentOf.end())
        {
            int perSm = 0;
            int sms = 0;
            TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &perSm, kernel, kMegaMoeStageThreads, /*dynamicSMemSize=*/0));
            TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
            it = residentOf.emplace(device, perSm * sms).first;
        }
        resident = it->second;
    }
    auto const quantizeBlocks
        = static_cast<int>(std::min<int64_t>(megaMoeQuantizeItems(quantize), std::max(1, resident)));
    if (writer.blocks + quantizeBlocks > 0)
    {
        common::launchWithPdlWhenEnabled("megaMoeStageInputs", kernel, dim3(writer.blocks + quantizeBlocks),
            dim3(kMegaMoeStageThreads), /*dynamicShmSize=*/0, stream, quantize, writer);
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
