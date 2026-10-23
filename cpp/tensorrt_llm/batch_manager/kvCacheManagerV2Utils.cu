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

#include "kvCacheManagerV2Utils.h"
#include "kv_cache_manager_v2/utils/hostMem.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cuda_runtime.h>
#include <limits>
#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
using Grain = uint4;
constexpr uint32_t ctaSize = 128;
constexpr uint32_t copyBlockCtaSize = 128;
constexpr uint32_t copyBlocknbBufs = 2;
constexpr uint32_t nbBufs = 4;
constexpr uint32_t grainBytes = sizeof(Grain);

using MMTask = Task<MemAddress, MemAddress>;

__device__ __host__ inline uint32_t divUp(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

template <uint32_t N>
__global__ void batchedCopy(std::array<MMTask, N> const __grid_constant__ tasks, uint32_t nbBytes)
{
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;\n");
#endif
    assert(nbBytes % sizeof(Grain) == 0);
    __shared__ Grain data[nbBufs][ctaSize];

    uint32_t const nbTasks = gridDim.y;
    assert(nbTasks <= N);
    auto const& task = tasks[blockIdx.y];
    uint32_t const nbSplits = gridDim.x;
    uint32_t const idxSplit = blockIdx.x;
    uint32_t const tid = threadIdx.x;

    constexpr uint32_t bytesPerIter = grainBytes * ctaSize;

    uint32_t const totalIters = divUp(nbBytes, bytesPerIter);
    uint32_t const maxItersPerCta = divUp(totalIters, nbSplits);
    uint32_t const idxGrainBeg = ctaSize * maxItersPerCta * idxSplit + tid;
    uint32_t const idxGrainEnd = std::min(idxGrainBeg + ctaSize * maxItersPerCta, nbBytes / grainBytes);

#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;\n");
#endif
    for (uint32_t i = 0; i < maxItersPerCta + nbBufs; i++)
    {
        uint32_t const idxBuf = i % nbBufs;
        if (i >= nbBufs)
        {
            uint32_t const stIter = i - nbBufs;
            assert(idxBuf == (stIter % nbBufs));
            Grain const& src = data[idxBuf][tid];
            uint32_t const idxGrain = idxGrainBeg + ctaSize * stIter;
            Grain& dst = reinterpret_cast<Grain*>(task.dst)[idxGrain];
            asm volatile("cp.async.wait_group %0;\n" ::"n"(nbBufs - 1) : "memory");
            if (idxGrain < idxGrainEnd)
            {
                dst = src;
            }
        }
        uint32_t const ldIter = i;
        Grain* const dst = &data[idxBuf][tid];
        uint32_t const idxGrain = idxGrainBeg + ctaSize * ldIter;
        Grain const* const src = &reinterpret_cast<Grain const*>(task.src)[idxGrain];
        if (idxGrain < idxGrainEnd)
        {
            uint32_t const size = grainBytes;
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"l"(__cvta_generic_to_shared(dst)),
                         "l"(src), "n"(grainBytes), "r"(size)
                         : "memory");
        }
        asm volatile("cp.async.commit_group;\n" : : : "memory");
    }
}

template <uint32_t N>
CUresult launchBatchedCopyImpl(
    bool lowBandwidth, MMTask const* tasks, uint32_t nbTasks, uint32_t nbBytes, cudaStream_t stream)
{
    TLLM_CHECK(nbTasks <= N);
    TLLM_CHECK_WITH_INFO(
        nbBytes % sizeof(Grain) == 0, "Not implemented case: nbBytes = %d must be a multiple of 16.", nbBytes);
    std::array<MMTask, N> const* pTasks;
    std::array<MMTask, N> tmp;
    if (nbTasks < N)
    {
        std::copy_n(tasks, nbTasks, tmp.begin());
        pTasks = &tmp;
    }
    else
    {
        pTasks = reinterpret_cast<std::array<MMTask, N> const*>(tasks);
    }
    uint32_t const nbSplits = lowBandwidth ? 1 : divUp(nbBytes, grainBytes * ctaSize * 2);
    void* args[] = {(void*) pTasks, (void*) &nbBytes};
    static CUkernel const kernel = []() -> CUkernel
    {
        cudaKernel_t kernel = nullptr;
        TLLM_CUDA_CHECK(cudaGetKernel(&kernel, reinterpret_cast<void const*>(&batchedCopy<N>)));
        return kernel;
    }();
    return common::CUDADriverWrapper::getInstance()->cuLaunchKernel(reinterpret_cast<CUfunction>(kernel), nbSplits,
        nbTasks, 1,    // gridDimX, gridDimY, gridDimZ
        ctaSize, 1, 1, // blockDimX, blockDimY, blockDimZ
        0,             // sharedMemBytes
        stream, args, nullptr);
}

// lowBandwidth gives each task one CTA instead of splitting a task over several CTAs. It is meant for copies that
// involve host memory, where a small SM footprint matters. The price is throughput per task: a 128-thread CTA with a
// 4-deep ring of 16 B grains has at most 4 x 128 x 16 B = 8 KiB of loads in flight, and bandwidth is bytes in flight
// divided by the latency of those loads, so one task is capped at a few to low tens of GB/s however fast the link is
// (measured 1.8 to 17.5 GB/s per CTA; the stores are posted, so device->host is bounded by device-memory load latency
// and only host->device by the link round trip). A batch of small pages is still fast because its N tasks run as N
// concurrent CTAs from one launch. A single 217 MB recurrent-state page is not: it takes tens of milliseconds where
// the copy engines need about a millisecond.
//
// copyHostToDevice / copyDeviceToHost therefore send calls whose tasks are large to the copy engines (one
// cuMemcpyBatchAsync per call on a real stream, one cuMemcpy{HtoD,DtoH}Async per task on the legacy NULL stream, which
// the batched call rejects) and keep this kernel for everything else; useCopyEngineForHostCopy below holds the rule.
// Reach of that rule: StorageManager moves pages between the hot GPU level and a GPU-accessible cold level (HOST_MEM,
// or a cold GPU_MEM level) through IKvCacheColdPageCodec, whose default implementation runs the multi-CTA
// BatchedPageCopier in kv_cache_manager_v2/batchedPageCopy.cu (see StorageManager::submitMigrationBatch).
// CopyEngine::transfer, and so these two functions, serves migrations between two cold levels (a cold GPU_MEM level
// next to HOST_MEM or DISK) and the nanobind bindings; its task vector holds one slot per page of the migration batch.
// BatchedPageCopier is not reused here because it works on pool descriptions (base, stride, index pairs), not on a
// plain task vector.
CUresult launchBatchedCopy(bool lowBandwidth, std::vector<MMTask> const& tasks, uint32_t nbBytes, cudaStream_t stream)
{
    constexpr uint32_t maxN = 256;
    uint32_t const nbWholeBatches = tasks.size() / maxN;
    for (uint32_t i = 0; i < nbWholeBatches; i++)
    {
        CUresult const err = launchBatchedCopyImpl<maxN>(lowBandwidth, tasks.data() + maxN * i, maxN, nbBytes, stream);
        if (err != CUDA_SUCCESS)
        {
            return err;
        }
    }
    {
        auto const* const pTasks = tasks.data() + maxN * nbWholeBatches;
        auto const batchSize = tasks.size() % maxN;
        if (batchSize == 0)
        {
            return CUDA_SUCCESS;
        }
        if (batchSize > maxN / 2)
        {
            return launchBatchedCopyImpl<maxN>(lowBandwidth, pTasks, batchSize, nbBytes, stream);
        }
        if (batchSize > maxN / 4)
        {
            return launchBatchedCopyImpl<maxN / 2>(lowBandwidth, pTasks, batchSize, nbBytes, stream);
        }
        if (batchSize > maxN / 8)
        {
            return launchBatchedCopyImpl<maxN / 4>(lowBandwidth, pTasks, batchSize, nbBytes, stream);
        }
        return launchBatchedCopyImpl<maxN / 8>(lowBandwidth, pTasks, batchSize, nbBytes, stream);
    }
}

namespace
{

constexpr size_t kDefaultHostCopyDmaMinBytes = size_t{4} << 20;

// Strict decimal integer from the raw value of the environment variable `name` (nullptr = unset): nullopt when unset,
// empty, not a decimal integer or out of range. A value that does not parse is reported and ignored rather than read
// as 0, because 0 has a meaning of its own for the knob below.
std::optional<long long> integerEnv(char const* name, char const* value) noexcept
{
    if (value == nullptr || *value == '\0')
    {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    long long const parsed = std::strtoll(value, &end, 10);
    char const* reason = nullptr;
    if (end == value || *end != '\0')
    {
        reason = "not an integer";
    }
    else if (errno == ERANGE)
    {
        reason = "out of range for a 64-bit integer";
    }
    else if (errno != 0)
    {
        reason = "not an integer";
    }
    if (reason == nullptr)
    {
        return parsed;
    }
    try
    {
        TLLM_LOG_WARNING("Ignoring %s=\"%s\": %s, using the default", name, value, reason);
    }
    catch (...)
    {
        // Logging allocates; a bad knob value must not turn into std::terminate from this noexcept function.
    }
    return std::nullopt;
}

// All tasks of one call go to the driver in a single cuMemcpyBatchAsync, so the CPU cost is one call however many
// pages there are. The batch is stream-ordered as a whole, but copies inside it may run in any order and dependent
// copies in one batch are undefined; the tasks of one call read and write disjoint slots, as the kernel launch they
// replace already requires, so the result is the same. The batched call does not accept the legacy NULL stream, as 0
// or as the CU_STREAM_LEGACY handle; those take one cuMemcpy{HtoD,DtoH}Async per task. Streams under graph capture
// are not supported either; no caller of these helpers captures. The attributes are the ones
// BatchedPageCopier::launchCopyEngine (kv_cache_manager_v2/batchedPageCopy.cu) uses; keep the two in sync.
template <bool HostToDevice>
CUresult copyWithCopyEngine(std::vector<MMTask> const& tasks, size_t numBytes, CUstream stream)
{
    if (stream == nullptr || stream == CU_STREAM_LEGACY)
    {
        for (auto const& task : tasks)
        {
            CUresult err = CUDA_SUCCESS;
            if constexpr (HostToDevice)
            {
                err = cuMemcpyHtoDAsync(
                    static_cast<CUdeviceptr>(task.dst), reinterpret_cast<void const*>(task.src), numBytes, stream);
            }
            else
            {
                err = cuMemcpyDtoHAsync(
                    reinterpret_cast<void*>(task.dst), static_cast<CUdeviceptr>(task.src), numBytes, stream);
            }
            if (err != CUDA_SUCCESS)
            {
                return err;
            }
        }
        return CUDA_SUCCESS;
    }

    // Descriptor scratch, reused across calls; thread-local so that two threads submitting migrations do not share it.
    thread_local std::vector<CUdeviceptr> dsts;
    thread_local std::vector<CUdeviceptr> srcs;
    thread_local std::vector<size_t> sizes;
    dsts.resize(tasks.size());
    srcs.resize(tasks.size());
    sizes.assign(tasks.size(), numBytes);
    for (size_t i = 0; i < tasks.size(); ++i)
    {
        dsts[i] = static_cast<CUdeviceptr>(tasks[i].dst);
        srcs[i] = static_cast<CUdeviceptr>(tasks[i].src);
    }
    CUmemcpyAttributes attributes{};
    attributes.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
    attributes.flags = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
    size_t attributeIndex = 0;
    return cuMemcpyBatchAsync(
        dsts.data(), srcs.data(), sizes.data(), tasks.size(), &attributes, &attributeIndex, 1, stream);
}

} // namespace

// Policy for host<->device copies.
//
// A call whose tasks are large goes to the copy engines through one cuMemcpyBatchAsync; every other call takes the
// batched copy kernel above. One environment variable sets the boundary and is read once per process:
//
//   TLLM_KV_CACHE_MANAGER_V2_HOST_COPY_DMA_MIN_BYTES   Smallest per-task size that uses the copy engines.
//                                                      Default 4 MiB: an 884,736 B attention page stays on the kernel,
//                                                      a 15 MB conv-state or 217 MB SSM-state page does not.
//                                                      0 = any size, negative = never (the previous behaviour).
//
// Small pages stay on the kernel because that is the configuration the A/B measured: a call of many small pages (one
// request's attention pages arrive about 140 at a time) runs as one launch with one CTA per page and already reaches
// the link rate (209 GB/s measured for 140 x 884,736 B on an NVLink-C2C system), so the copy engines could save SMs
// there but not time; MIN_BYTES=0 is the switch for that experiment. The copies are enqueued on the stream the caller
// passes, after its prior-event waits and before its finish event, so the events the storage manager records still
// cover them.
size_t hostCopyDmaMinBytesFor(char const* rawValue) noexcept
{
    auto const parsed = integerEnv(kHostCopyDmaMinBytesEnv, rawValue);
    if (!parsed.has_value())
    {
        return kDefaultHostCopyDmaMinBytes;
    }
    return *parsed < 0 ? std::numeric_limits<size_t>::max() : static_cast<size_t>(*parsed);
}

size_t hostCopyDmaMinBytes() noexcept
{
    static size_t const value = hostCopyDmaMinBytesFor(std::getenv(kHostCopyDmaMinBytesEnv));
    return value;
}

bool useCopyEngineForHostCopy(size_t numBytes) noexcept
{
    // A single memcpy may not span two cuMemHostRegister ranges. On kernels where HostMem registers the host tier in
    // 2 GiB chunks (see shouldUseChunkedRegistration) every call keeps the copy kernel, which reads pinned memory
    // through ordinary loads and stores and does not care how it was registered.
    static bool const chunked = HostMem::shouldUseChunkedRegistration();
    return !chunked && numBytes >= hostCopyDmaMinBytes();
}

namespace
{

// One dispatch for both directions: nothing for an empty call, the copy engines when the policy selects them, the
// batched copy kernel otherwise. The kernel's launcher takes a 32-bit byte count and rounds it up to whole 2 KiB
// iterations in 32-bit arithmetic, so it is limited to tasks below 4 GiB; the copy engines take a size_t.
template <bool HostToDevice>
CUresult copyAcrossHostLink(std::vector<MMTask> const& tasks, ssize_t numBytes, CUstream stream)
{
    TLLM_CHECK_WITH_INFO(numBytes >= 0, "numBytes = %zd must be non-negative", numBytes);
    if (tasks.empty())
    {
        return CUDA_SUCCESS;
    }
    if (numBytes > 0 && useCopyEngineForHostCopy(static_cast<size_t>(numBytes)))
    {
        return copyWithCopyEngine<HostToDevice>(tasks, static_cast<size_t>(numBytes), stream);
    }
    constexpr size_t kMaxKernelBytes = std::numeric_limits<uint32_t>::max() - (grainBytes * ctaSize - 1);
    TLLM_CHECK_WITH_INFO(static_cast<size_t>(numBytes) <= kMaxKernelBytes,
        "Not implemented case: the batched copy kernel takes less than 4 GiB per task, numBytes = %zd", numBytes);
    return launchBatchedCopy(/*lowBandwidth=*/true, tasks, static_cast<uint32_t>(numBytes), stream);
}

} // namespace

CUresult copyHostToDevice(std::vector<MMTask> const& tasks, ssize_t numBytes, CUstream stream)
{
    return copyAcrossHostLink<true>(tasks, numBytes, stream);
}

CUresult copyDeviceToHost(std::vector<MMTask> const& tasks, ssize_t numBytes, CUstream stream)
{
    return copyAcrossHostLink<false>(tasks, numBytes, stream);
}

CUresult copyDeviceToDevice(std::vector<MMTask> const& tasks, ssize_t numBytes, CUstream stream)
{
    return launchBatchedCopy(false, tasks, numBytes, stream);
}

// dst_tensor[:, :num_seqs, 0] = src_tensor[:, copy_idx]
// dst_tensor[:, :num_seqs, 1] = dst_tensor[:, :num_seqs, 0] + 1
__global__ void copyBatchBlockOffsetsToDeviceKernel(SizeType32 const* __restrict__ srcPtr,
    SizeType32* __restrict__ dstPtr, SizeType32 const srcMaxNumSequences, SizeType32 const dstMaxNumSequences,
    SizeType32 numBlocksPerSeq, SizeType32 const* __restrict__ copyIndex, SizeType32 const* __restrict__ indexScales,
    SizeType32 const* __restrict__ kvOffset)
{
    constexpr uint32_t kvFactor = 2;
    constexpr auto elemPerAccess = sizeof(PackedInt) / sizeof(SizeType32);

    __shared__ PackedInt data[copyBlocknbBufs][copyBlockCtaSize];

    auto const iterPerSeq = divUp(numBlocksPerSeq * sizeof(SizeType32), sizeof(PackedInt) * copyBlockCtaSize);
    auto const tid = threadIdx.x;
    auto const poolIdx = blockIdx.x;
    auto const seqIdx = blockIdx.y;
    auto const seqDimStride = kvFactor * numBlocksPerSeq;
    uint32_t const srcIdxBeg = tid * elemPerAccess + (poolIdx * srcMaxNumSequences + copyIndex[seqIdx]) * seqDimStride;
    uint32_t const dstIdxKBeg = tid * elemPerAccess + (poolIdx * dstMaxNumSequences + seqIdx) * seqDimStride;
    uint32_t const dstIdxVBeg = dstIdxKBeg + numBlocksPerSeq;

    uint32_t const srcIdxEnd = (poolIdx * srcMaxNumSequences + copyIndex[seqIdx]) * seqDimStride + numBlocksPerSeq;

    for (uint32_t i = 0; i < iterPerSeq + copyBlocknbBufs; i++)
    {
        uint32_t const idxBuf = i % copyBlocknbBufs;
        if (i >= copyBlocknbBufs)
        {
            uint32_t const stIter = i - copyBlocknbBufs;
            assert(idxBuf == (stIter % copyBlocknbBufs));
            auto const offset = copyBlockCtaSize * stIter * elemPerAccess;
            SizeType32 const srcIdx = srcIdxBeg + offset;
            SizeType32 const dstIdxK = dstIdxKBeg + offset;
            SizeType32 const dstIdxV = dstIdxVBeg + offset;
            PackedInt const& src = data[idxBuf][tid];
            PackedInt& dstK = *reinterpret_cast<PackedInt*>(dstPtr + dstIdxK);
            PackedInt& dstV = *reinterpret_cast<PackedInt*>(dstPtr + dstIdxV);
            asm volatile("cp.async.wait_group %0;\n" ::"n"(copyBlocknbBufs - 1) : "memory");
            if (srcIdx < srcIdxEnd)
            {
#pragma unroll
                for (uint32_t j = 0; j < elemPerAccess; j++)
                {
                    auto const val = src.unpacked[j];
                    dstK.unpacked[j] = (val == BAD_PAGE_INDEX) ? 0 : (indexScales[poolIdx] * val);
                    dstV.unpacked[j] = (val == BAD_PAGE_INDEX) ? 0 : (indexScales[poolIdx] * val + kvOffset[poolIdx]);
                }
            }
        }
        uint32_t const ldIter = i;
        PackedInt* const dst = &data[idxBuf][tid];
        uint32_t const srcIdx = srcIdxBeg + copyBlockCtaSize * ldIter * elemPerAccess;
        PackedInt const* const src = reinterpret_cast<PackedInt const*>(srcPtr + srcIdx);
        if (srcIdx < srcIdxEnd)
        {
            uint32_t const size = sizeof(PackedInt);
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"l"(__cvta_generic_to_shared(dst)),
                         "l"(src), "n"(size), "r"(size)
                         : "memory");
        }
        asm volatile("cp.async.commit_group;\n" : : : "memory");
    }
}

// Host-side launcher
void copyBatchBlockOffsetsToDevice(ITensor const& input, ITensor& output, ITensor const& copyIndex,
    ITensor const& indexScales, ITensor const& kvOffset, CUstream stream) noexcept
{
    using namespace tensorrt_llm::runtime;

    auto const* srcPtr = input.getBufferPtr<tk::KVCacheIndex::UnderlyingType const>();
    // [numPools, maxNumSequences, kvFactor, numBlocksPerSeq]
    auto* dstPtr = output.getBufferPtr<tk::KVCacheIndex::UnderlyingType>();
    auto const* copyIndexPtr = copyIndex.getBufferPtr<SizeType32 const>();
    auto const* indexScalesPtr = indexScales.getBufferPtr<SizeType32 const>();
    auto const* kvOffsetPtr = kvOffset.getBufferPtr<SizeType32 const>();
    auto const& srcShape = input.getShape();
    auto const& dstShape = output.getShape();
    auto const& copyIndexShape = copyIndex.getShape();

    TLLM_CHECK(srcShape.nbDims == 4); // [numPools, srcMaxNumSequences, kvFactor, numBlocksPerSeq]
    TLLM_CHECK(dstShape.nbDims == 4); // [numPools, dstMaxNumSequences, kvFactor, numBlocksPerSeq]

    SizeType32 numPools = srcShape.d[0];
    SizeType32 srcMaxNumSequences = srcShape.d[1];
    SizeType32 dstMaxNumSequences = dstShape.d[1];
    SizeType32 numBlocksPerSeq = srcShape.d[3];
    SizeType32 numSeqs = copyIndexShape.d[0];

    if (numSeqs == 0)
    {
        return;
    }

    TLLM_CHECK_WITH_INFO((numBlocksPerSeq * sizeof(SizeType32)) % sizeof(PackedInt) == 0,
        "Not implemented case: numBlocksPerSeq * sizeof(SizeType32) = %zu must be a multiple of %zu.",
        static_cast<size_t>(numBlocksPerSeq * sizeof(SizeType32)), static_cast<size_t>(sizeof(PackedInt)));

    dim3 gridDim(numPools, numSeqs, 1);
    dim3 blockDim(copyBlockCtaSize);

    copyBatchBlockOffsetsToDeviceKernel<<<gridDim, blockDim, 0, stream>>>(srcPtr, dstPtr, srcMaxNumSequences,
        dstMaxNumSequences, numBlocksPerSeq, copyIndexPtr, indexScalesPtr, kvOffsetPtr);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
