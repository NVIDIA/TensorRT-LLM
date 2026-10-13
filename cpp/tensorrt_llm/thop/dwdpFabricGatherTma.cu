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

// DWDP fabric gather, TMA bulk-copy variant: the bytes in flight live in shared memory.
//
// A block's rate is its bytes in flight over the fabric latency, and the SM holds them.  One elected
// thread streams `cp.async.bulk` copies from the fabric-mapped source into a ring of kStages x
// kStageBytes shared-memory stages (one mbarrier each) and drains landed stages to local HBM with
// bulk shared->global copies, keeping more bytes in flight per SM than the register kernel can.
// Same descriptor contract as dwdpFabricGather.cu; a row's blocks split each descriptor's chunks
// round-robin.  No register loads or L2 prefetch here: hybrids and L2 prefetch measured slower; keep the ring alone.

#include "dwdpFabricGatherTma.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdlib>
#include <cuda_runtime.h>

namespace tensorrt_llm::torch_ext
{
namespace
{

//! Stage ring geometry: kStages x kStageBytes = 224 KB of the SM's 256 KB unified L1/shared memory,
//! one mbarrier per stage; lazy refill keeps kStages - 1 of them in flight.
constexpr int kStages = 14;
constexpr int kStageBytes = 16384;
//! The kernel is one elected thread; the block only has to be launchable.
constexpr int kMaxThreads = 128;
constexpr int kLineBytes = 16;
constexpr int kSmemAlign = 128;
//! Dynamic shared memory: the stage ring plus slack so the ring can be re-aligned to kSmemAlign.
constexpr int kDynamicSmemBytes = kStages * kStageBytes + kSmemAlign;
constexpr int kMaxDevices = 64;

static_assert(kStages >= 2, "lazy refill keeps one stage out of flight, so it needs at least 2");
static_assert(kStageBytes % kSmemAlign == 0, "stage size must be a multiple of the ring alignment");
static_assert(kStageBytes < (1 << 20), "mbarrier tx-count is 20 bits");
static_assert(sizeof(uint4) == kLineBytes, "descriptor lines are one uint4 wide");

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

__device__ __forceinline__ uint32_t smemAddr(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarrierInit(uint32_t bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void fenceMbarrierInit()
{
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void fenceProxyAsyncShared()
{
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void mbarrierArriveExpectTx(uint32_t bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ bool mbarrierTryWait(uint32_t bar, uint32_t parity)
{
    uint32_t done;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, P1;\n\t"
        "}"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
    return done != 0;
}

__device__ __forceinline__ void mbarrierWait(uint32_t bar, uint32_t parity)
{
    while (!mbarrierTryWait(bar, parity))
    {
    }
}

//! Global -> shared bulk copy; completion is signalled on `bar` as `bytes` of transaction count.
__device__ __forceinline__ void bulkLoadG2S(uint32_t smemDst, void const* gsrc, uint32_t bytes, uint32_t bar)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smemDst),
        "l"(gsrc), "r"(bytes), "r"(bar)
        : "memory");
}

//! Shared -> global bulk copy, tracked by the bulk async-group mechanism.
__device__ __forceinline__ void bulkStoreS2G(void* gdst, uint32_t smemSrc, uint32_t bytes)
{
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(gdst), "r"(smemSrc), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void bulkCommitGroup()
{
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

//! Wait until at most N of the committed groups are still reading their shared-memory source.
template <int N>
__device__ __forceinline__ void bulkWaitGroupRead()
{
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

//! Wait until at most N of the committed groups are still pending (reads and writes).
template <int N>
__device__ __forceinline__ void bulkWaitGroup()
{
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
}

//! Walks the kStageBytes chunks this block owns across the row's descriptors: descriptor d is cut into
//! ceil(bytes / kStageBytes) chunks, block b takes chunks b, b + gridDim.x, ...; the last chunk is the
//! remaining bytes.  Producer and consumer run two copies of the cursor, so no metadata goes through
//! shared memory; chunk j of the block's sequence lands in stage j % kStages with parity (j / kStages) & 1.
struct ChunkCursor
{
    int64_t desc;      //!< current descriptor
    int64_t rowEnd;    //!< one past the row's last descriptor
    int64_t chunk;     //!< current chunk index within `desc`
    int64_t numChunks; //!< chunks in `desc`
    int64_t descBytes; //!< bytes in `desc`
    int64_t const* lines;

    __device__ __forceinline__ void loadDesc()
    {
        descBytes = lines[desc] * kLineBytes;
        numChunks = (descBytes + kStageBytes - 1) / kStageBytes;
        chunk = blockIdx.x;
    }

    //! Skip descriptors in which this block has no chunk left (short descriptors included).
    __device__ __forceinline__ void settle()
    {
        while (desc < rowEnd && chunk >= numChunks)
        {
            ++desc;
            if (desc < rowEnd)
            {
                loadDesc();
            }
        }
    }

    __device__ __forceinline__ void init(int64_t rowBegin, int64_t rowEnd_, int64_t const* lines_)
    {
        desc = rowBegin;
        rowEnd = rowEnd_;
        lines = lines_;
        chunk = 0;
        numChunks = 0;
        descBytes = 0;
        if (desc < rowEnd)
        {
            loadDesc();
        }
        settle();
    }

    __device__ __forceinline__ bool valid() const
    {
        return desc < rowEnd;
    }

    __device__ __forceinline__ void advance()
    {
        chunk += gridDim.x;
        settle();
    }

    __device__ __forceinline__ int64_t offsetBytes() const
    {
        return chunk * kStageBytes;
    }

    __device__ __forceinline__ uint32_t bytes() const
    {
        int64_t const rem = descBytes - offsetBytes();
        return static_cast<uint32_t>(rem < kStageBytes ? rem : kStageBytes);
    }
};

#endif // __CUDA_ARCH__ >= 900

__global__ void __launch_bounds__(kMaxThreads) dwdpFabricGatherTmaKernel(uint4* const* __restrict__ dsts,
    uint4 const* const* __restrict__ srcs, int64_t const* __restrict__ lines, int64_t const* __restrict__ rowStarts)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __align__(16) unsigned char smemRaw[];
    __shared__ __align__(8) uint64_t mbar[kStages];

    // Re-align the stage ring to kSmemAlign; kDynamicSmemBytes reserves the slack.
    uint32_t const rawBase = smemAddr(smemRaw);
    uint32_t const smemBase = (rawBase + (kSmemAlign - 1)) & ~static_cast<uint32_t>(kSmemAlign - 1);
    uint32_t const mbarBase = smemAddr(mbar);

    if (threadIdx.x == 0)
    {
        for (int s = 0; s < kStages; ++s)
        {
            mbarrierInit(mbarBase + 8u * s, 1u);
        }
        fenceMbarrierInit();
    }
    __syncthreads();
    if (threadIdx.x != 0)
    {
        return; // the elected thread issues both the loads and the stores; the rest of the block is done
    }

    ChunkCursor prod;
    prod.init(rowStarts[blockIdx.y], rowStarts[blockIdx.y + 1], lines);
    ChunkCursor cons = prod;
    int64_t produced = 0;
    int64_t consumed = 0;

    // Load the producer's current chunk into `stage` and step the producer.
    auto issueLoad = [&](int stage)
    {
        uint32_t const bytes = prod.bytes();
        unsigned char const* gsrc = reinterpret_cast<unsigned char const*>(srcs[prod.desc]) + prod.offsetBytes();
        uint32_t const bar = mbarBase + 8u * stage;
        mbarrierArriveExpectTx(bar, bytes);
        bulkLoadG2S(smemBase + static_cast<uint32_t>(stage) * kStageBytes, gsrc, bytes, bar);
        prod.advance();
        ++produced;
    };

    // Prologue: fill every stage this block has a chunk for.
    for (int s = 0; s < kStages && prod.valid(); ++s)
    {
        issueLoad(s);
    }

    int stage = 0;
    int prevStage = -1;
    uint32_t parity = 0;
    while (consumed < produced)
    {
        mbarrierWait(mbarBase + 8u * stage, parity);

        uint32_t const bytes = cons.bytes();
        unsigned char* gdst = reinterpret_cast<unsigned char*>(dsts[cons.desc]) + cons.offsetBytes();
        // The stage was written by the async proxy and observed through the mbarrier; order that
        // ahead of the async-proxy read the store is about to make.
        fenceProxyAsyncShared();
        bulkStoreS2G(gdst, smemBase + static_cast<uint32_t>(stage) * kStageBytes, bytes);
        bulkCommitGroup();

        // Lazy refill: every store but the one just committed has finished reading its stage, so
        // the previous stage is free.  Refilling this stage would need wait_group.read 0 and stall
        // here on the store's shared-memory read.
        bulkWaitGroupRead<1>();
        if (prevStage >= 0 && prod.valid())
        {
            issueLoad(prevStage);
        }
        prevStage = stage;

        cons.advance();
        ++consumed;
        if (++stage == kStages)
        {
            stage = 0;
            parity ^= 1u;
        }
    }
    // The last stores are still in flight; their global writes must land before the block exits.
    bulkWaitGroup<0>();
#else
    // Pre-sm_90 pass (no bulk copies): plain strided copy so the binary loads and stays correct anywhere.
    int64_t const rowEnd = rowStarts[blockIdx.y + 1];
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t desc = rowStarts[blockIdx.y]; desc < rowEnd; ++desc)
    {
        uint4 const* __restrict__ src = srcs[desc];
        uint4* __restrict__ dst = dsts[desc];
        int64_t const n = lines[desc];
        for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        {
            dst[i] = src[i];
        }
    }
#endif
}

} // namespace

//! Cluster launch, same as the register kernel: pin the blocks of one grid row (one peer) onto a
//! TPC pair so the 2-CTA-cluster persistent GEMMs keep whole TPCs.  TRTLLM_DWDP_GATHER_CLUSTER=<n>
//! overrides at runtime (1 = off).
#ifndef DWDP_GATHER_CLUSTER
#define DWDP_GATHER_CLUSTER 2
#endif

namespace
{
int gatherTmaClusterSize()
{
    static int const size = []
    {
        char const* env = std::getenv("TRTLLM_DWDP_GATHER_CLUSTER");
        int const value = env ? std::atoi(env) : DWDP_GATHER_CLUSTER;
        return value > 0 ? value : 1;
    }();
    return size;
}
} // namespace

void launchDwdpFabricGatherTma(void* const* dsts, void const* const* srcs, int64_t const* lines,
    int64_t const* rowStarts, int64_t rows, int64_t blocksPerRow, int64_t threads, cudaStream_t stream)
{
    auto* const kernel = &dwdpFabricGatherTmaKernel;

    // Opt in to more than 48 KB of dynamic shared memory, once per device.
    static bool attrSet[kMaxDevices] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < kMaxDevices && !attrSet[dev])
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmemBytes));
        attrSet[dev] = true;
    }

    // One elected thread drives the ring, so anything above kMaxThreads only costs launch slots.
    int64_t const blockThreads = threads > kMaxThreads ? kMaxThreads : threads;
    dim3 const grid(static_cast<unsigned int>(blocksPerRow), static_cast<unsigned int>(rows));
    auto const* typedDsts = reinterpret_cast<uint4* const*>(dsts);
    auto const* typedSrcs = reinterpret_cast<uint4 const* const*>(srcs);

    int const cluster = gatherTmaClusterSize();
    cudaError_t err = cudaSuccess;
    if (cluster > 1 && blocksPerRow % cluster == 0)
    {
        cudaLaunchConfig_t config = {};
        config.gridDim = grid;
        config.blockDim = dim3(static_cast<unsigned int>(blockThreads));
        config.dynamicSmemBytes = kDynamicSmemBytes;
        config.stream = stream;
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = static_cast<unsigned int>(cluster);
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;
        err = cudaLaunchKernelEx(&config, kernel, typedDsts, typedSrcs, lines, rowStarts);
    }
    else
    {
        kernel<<<grid, static_cast<unsigned int>(blockThreads), kDynamicSmemBytes, stream>>>(
            typedDsts, typedSrcs, lines, rowStarts);
        err = cudaGetLastError();
    }
    // Same reason as the register kernel: a silent launch failure leaves stale experts in the buffer.
    TLLM_CHECK_WITH_INFO(err == cudaSuccess,
        "dwdp_fabric_gather (tma) launch failed: %s (grid %lld x %lld, threads %lld, cluster %d, dyn_smem %d)",
        cudaGetErrorString(err), static_cast<long long>(blocksPerRow), static_cast<long long>(rows),
        static_cast<long long>(blockThreads), cluster, kDynamicSmemBytes);
}

} // namespace tensorrt_llm::torch_ext
