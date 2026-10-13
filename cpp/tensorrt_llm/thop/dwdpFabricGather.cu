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

#include "dwdpFabricGather.h"
#include "dwdpFabricGatherTma.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

namespace tensorrt_llm::torch_ext
{
namespace
{

//! Unroll depth: 16-byte fabric loads kept in flight per thread.  A fabric read is long-latency, so
//! the rate comes from loads in flight per thread rather than from more threads; 8 measured ~3x the
//! rate of 4 from the same blocks and still lands in registers without spills.
constexpr int kUnroll = 8;

//! Largest block the op accepts.  1024 threads leave 64 registers per thread; the launch bound makes
//! ptxas fit the unroll-8 body into that instead of letting a 1024-thread launch fail at runtime.
constexpr int kMaxThreads = 1024;

//! One grid row per group (in DWDP, per peer).  Blocks are what the gather charges the resident GEMM,
//! so a group's descriptors are walked inside its row instead of spending more rows on them.
template <int UNROLL>
__global__ void __launch_bounds__(kMaxThreads) dwdpFabricGatherKernel(uint4* const* __restrict__ dsts,
    uint4 const* const* __restrict__ srcs, int64_t const* __restrict__ lines, int64_t const* __restrict__ rowStarts)
{
    int64_t const rowEnd = rowStarts[blockIdx.y + 1];
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t desc = rowStarts[blockIdx.y]; desc < rowEnd; ++desc)
    {
        uint4 const* __restrict__ src = srcs[desc];
        uint4* __restrict__ dst = dsts[desc];
        int64_t const n = lines[desc];

        int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        for (; i + (UNROLL - 1) * stride < n; i += UNROLL * stride)
        {
            uint4 v[UNROLL];
#pragma unroll
            for (int u = 0; u < UNROLL; ++u)
            {
                v[u] = src[i + u * stride];
            }
#pragma unroll
            for (int u = 0; u < UNROLL; ++u)
            {
                dst[i + u * stride] = v[u];
            }
        }
        for (; i < n; i += stride)
        {
            dst[i] = src[i];
        }
    }
}

} // namespace

//! Cluster launch: co-locate each cluster of DWDP_GATHER_CLUSTER gather blocks on one TPC, so the
//! 2-CTA-cluster persistent CuTeDSL GEMMs keep whole TPCs; a plain launch scatters the blocks over
//! up to one TPC each.  TRTLLM_DWDP_GATHER_CLUSTER=<n> overrides the cluster size at runtime (1 = off).
#ifndef DWDP_GATHER_CLUSTER
#define DWDP_GATHER_CLUSTER 2
#endif

namespace
{
int gatherClusterSize()
{
    static int const size = []
    {
        char const* env = std::getenv("TRTLLM_DWDP_GATHER_CLUSTER");
        int const value = env ? std::atoi(env) : DWDP_GATHER_CLUSTER;
        return value > 0 ? value : 1;
    }();
    return size;
}

//! TRTLLM_DWDP_GATHER_KERNEL=tma selects the shared-memory-staged bulk-copy kernel (dwdpFabricGatherTma.cu).
bool gatherUsesTma()
{
    static bool const tma = []
    {
        char const* env = std::getenv("TRTLLM_DWDP_GATHER_KERNEL");
        return env != nullptr && std::strcmp(env, "tma") == 0;
    }();
    return tma;
}
} // namespace

void launchDwdpFabricGather(void* const* dsts, void const* const* srcs, int64_t const* lines, int64_t const* rowStarts,
    int64_t rows, int64_t blocksPerRow, int64_t threads, cudaStream_t stream)
{
    if (gatherUsesTma())
    {
        launchDwdpFabricGatherTma(dsts, srcs, lines, rowStarts, rows, blocksPerRow, threads, stream);
        return;
    }
    TLLM_CHECK_WITH_INFO(threads > 0 && threads <= kMaxThreads,
        "dwdp_fabric_gather: threads must be in [1, %d], got %lld", kMaxThreads, static_cast<long long>(threads));
    dim3 const grid(static_cast<unsigned int>(blocksPerRow), static_cast<unsigned int>(rows));
    auto const* typedDsts = reinterpret_cast<uint4* const*>(dsts);
    auto const* typedSrcs = reinterpret_cast<uint4 const* const*>(srcs);
    int const cluster = gatherClusterSize();
    cudaError_t err = cudaSuccess;
    if (cluster > 1 && blocksPerRow % cluster == 0)
    {
        cudaLaunchConfig_t config = {};
        config.gridDim = grid;
        config.blockDim = dim3(static_cast<unsigned int>(threads));
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = static_cast<unsigned int>(cluster);
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        err = cudaLaunchKernelEx(&config, dwdpFabricGatherKernel<kUnroll>, typedDsts, typedSrcs, lines, rowStarts);
    }
    else
    {
        dwdpFabricGatherKernel<kUnroll>
            <<<grid, static_cast<unsigned int>(threads), 0, stream>>>(typedDsts, typedSrcs, lines, rowStarts);
        err = cudaGetLastError();
    }
    // A launch that fails silently leaves the double buffer holding the previous layer's experts while
    // the prefetch event is still recorded, so compute would run on stale weights without a trace.
    TLLM_CHECK_WITH_INFO(err == cudaSuccess,
        "dwdp_fabric_gather launch failed: %s (grid %lld x %lld, threads %lld, cluster %d)", cudaGetErrorString(err),
        static_cast<long long>(blocksPerRow), static_cast<long long>(rows), static_cast<long long>(threads), cluster);
}

} // namespace tensorrt_llm::torch_ext
