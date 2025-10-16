/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <atomic>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/moeLoadBalance/moeLoadBalanceKernels.h"

namespace cg = cooperative_groups;

namespace tensorrt_llm
{
namespace kernels
{

using tensorrt_llm::common::launchWithPdlWhenEnabled;

int getOwnerDevice(unsigned long long int stepAndOwner)
{
    return static_cast<int>(stepAndOwner & MoeLoadBalanceSingleLayerSignal::kDevice);
}

__device__ int getOwnerDeviceGpu(unsigned long long int stepAndOwner)
{
    return static_cast<int>(stepAndOwner & MoeLoadBalanceSingleLayerSignal::kDevice);
}

__device__ unsigned long long int getCurrentStep(unsigned long long int stepAndOwner)
{
    return stepAndOwner >> 2U;
}

__device__ bool isDisabled(unsigned long long int stepAndOwner)
{
    return stepAndOwner >= MoeLoadBalanceSingleLayerSignal::kDisabled;
}

__device__ __forceinline__ void moeWaitSignalForGpuStageFunc(MoeLoadBalanceSingleLayerSignal* signal, int* enabled)
{
    bool ready = false;
    int stepEnable = 0;
    do
    {
        unsigned long long int loaded = signal->stepAndOwner;
        int owner = getOwnerDeviceGpu(loaded);
        bool disabled = isDisabled(loaded);
        if (owner == MoeLoadBalanceSingleLayerSignal::kGPU || disabled)
        {
            ready = true;
            if (!disabled && !(loaded & MoeLoadBalanceSingleLayerSignal::kSkipStep))
            {
                stepEnable = 1;
            }
        }
    } while (!ready);
    *enabled = stepEnable;
}

__global__ void moeWaitSignalForGpuStageKernel(MoeLoadBalanceSingleLayerSignal* signal, int* enabled)
{

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    if (threadIdx.x == 0 and blockIdx.x == 0)
    {
        moeWaitSignalForGpuStageFunc(signal, enabled);
    }
}

__global__ void moeSetSignalForCpuStageKernel(MoeLoadBalanceSingleLayerSignal* signal)
{

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    if (threadIdx.x == 0 and blockIdx.x == 0)
    {
        unsigned long long int loaded = signal->stepAndOwner;
        if (!isDisabled(loaded))
        {
            signal->stepAndOwner |= MoeLoadBalanceSingleLayerSignal::kCPU;
        }
    }
}

void moeWaitSignalForGpuStageDevice(MoeLoadBalanceSingleLayerSignal* signal, int* enabled, cudaStream_t stream)
{
    launchWithPdlWhenEnabled(
        "moeWaitSignalForGpuStage", moeWaitSignalForGpuStageKernel, 1, 1, 0, stream, signal, enabled);
}

void moeWaitSignalForGpuStageForTest(MoeLoadBalanceSingleLayerSignal* signal, int* enabled)
{
    bool ready = false;
    do
    {
        auto loaded = signal->stepAndOwner;
        ready = getOwnerDevice(loaded) == MoeLoadBalanceSingleLayerSignal::kGPU;
        if (ready)
        {
            if (loaded >= MoeLoadBalanceSingleLayerSignal::kDisabled
                || (loaded & MoeLoadBalanceSingleLayerSignal::kSkipStep))
            {
                *enabled = 0;
            }
            else
            {
                *enabled = 1;
            }
        }
    } while (!ready);
    std::atomic_thread_fence(std::memory_order_acquire);
}

void moeSetSignalForCpuStageDevice(MoeLoadBalanceSingleLayerSignal* signal, cudaStream_t stream)
{
    launchWithPdlWhenEnabled("moeSetSignalForCpuStage", moeSetSignalForCpuStageKernel, 1, 1, 0, stream, signal);
}

void moeSetSignalForCpuStageForTest(MoeLoadBalanceSingleLayerSignal* signal)
{
    std::atomic_thread_fence(std::memory_order_release);
    signal->stepAndOwner += MoeLoadBalanceSingleLayerSignal::kCPU;
}

template <typename TYPE>
__global__ void zeroExpertTokenCountKernel(MoeLoadBalanceMetaInfo metaInfo, int* const enabled, int* expertTokenCount)
{
    if (*enabled == 0)
    {
        return;
    }
    TYPE oldExpertTokenCount = {0};
    int* expertTokenCountPtr = expertTokenCount + metaInfo.expertCount * blockIdx.x;
    TYPE* typedExpertTokenCountPtr = reinterpret_cast<TYPE*>(expertTokenCountPtr);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    typedExpertTokenCountPtr[threadIdx.x] = oldExpertTokenCount;
}

template <typename TYPE>
__global__ void shiftWindowKernel(MoeLoadBalanceMetaInfo metaInfo, int* const enabled, int* expertTokenCount)
{
    if (*enabled == 0)
    {
        return;
    }
    TYPE oldExpertTokenCount = {0};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    if (blockIdx.x > 0)
    {
        int* oldExpertTokenCountPtr = expertTokenCount + metaInfo.expertCount * (blockIdx.x - 1);
        TYPE* typedOldExpertTokenCountPtr = reinterpret_cast<TYPE*>(oldExpertTokenCountPtr);
        oldExpertTokenCount = typedOldExpertTokenCountPtr[threadIdx.x];
    }
    if (gridDim.x > 1)
    {
        cg::this_grid().sync();
    }
    int* expertTokenCountPtr = expertTokenCount + metaInfo.expertCount * blockIdx.x;
    TYPE* typedExpertTokenCountPtr = reinterpret_cast<TYPE*>(expertTokenCountPtr);
    typedExpertTokenCountPtr[threadIdx.x] = oldExpertTokenCount;
}

__global__ void statisticKernel(MoeLoadBalanceMetaInfo metaInfo, int* expertTokenCount, int totalEltCount,
    int* const enabled, int* const gatheredRawExpertIds)
{
    extern __shared__ int sharedExpertCount[];
    if (*enabled == 0)
    {
        return;
    }
    for (int i = threadIdx.x; i < metaInfo.expertCount; i += blockDim.x)
    {
        sharedExpertCount[i] = 0;
    }
    __syncthreads();
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < totalEltCount; idx += gridDim.x * blockDim.x)
    {
        int expertId = gatheredRawExpertIds[idx];
        if (expertId >= 0 && expertId < metaInfo.expertCount)
        {
            atomicAdd_block(&sharedExpertCount[expertId], 1);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < metaInfo.expertCount; i += blockDim.x)
    {
        atomicAdd_system(&expertTokenCount[i], sharedExpertCount[i]);
    }
}

__global__ void updateLoadFactorKernel(MoeLoadBalanceMetaInfo metaInfo, MoeLoadBalanceStatisticInfo statisticInfo,
    int* expertTokenCountPtr, int* const enabled)
{
    if (*enabled == 0)
    {
        return;
    }
    int expertIdx = blockIdx.x * blockDim.x + threadIdx.x;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    int expertTokenCount = expertTokenCountPtr[expertIdx];
    float* loadFactor = statisticInfo.expertLoadFactor;
    loadFactor[expertIdx] = loadFactor[expertIdx] * statisticInfo.decayFactor + expertTokenCount;
}

void moeStatisticDevice(MoeLoadBalanceMetaInfo metaInfo, MoeLoadBalanceStatisticInfo statisticInfo, int numTotalTokens,
    int* const enabled, bool isFirstStage, bool isLastStage, int* const gatheredRawExpertIds, cudaStream_t stream)
{
    static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
    if (isFirstStage)
    {
        // shift window and zero expertTokenCount
        // only first stage need shift window.
        int threadCount = metaInfo.expertCount;
        auto* kernelFunc = shiftWindowKernel<int>;
        if (threadCount % 4 == 0)
        {
            threadCount /= 4;
            kernelFunc = shiftWindowKernel<int4>;
        }
        else if (threadCount % 2 == 0)
        {
            threadCount /= 2;
            kernelFunc = shiftWindowKernel<int2>;
        }
        dim3 gridDim(statisticInfo.rawDataWindowSize + 1);
        dim3 blockDim(threadCount);
        int* expertTokenCount = statisticInfo.expertTokenCount;
        void* args[]
            = {&metaInfo, static_cast<void*>(const_cast<int**>(&enabled)), static_cast<void*>(&expertTokenCount)};
        TLLM_CHECK_WITH_INFO(
            threadCount <= 1024, "expertCount=%d is too large and not supported now.", metaInfo.expertCount);
        // TODO: add PDL support with cooperative launch
        TLLM_CUDA_CHECK(cudaLaunchCooperativeKernel(kernelFunc, gridDim, blockDim, &args[0], 0, stream));
    }

    {
        // do the statistic into expertTokenCount and maybe also expertLoadFactor;
        int threadCount = 1024;
        int totalEltCount = numTotalTokens * metaInfo.topK;
        int blockCount = (totalEltCount + threadCount - 1) / threadCount;
        if (blockCount > smCount)
        {
            blockCount = smCount;
        }
        int sharedMemorySize = metaInfo.expertCount * sizeof(int);
        launchWithPdlWhenEnabled("statisticKernel", statisticKernel, blockCount, threadCount, sharedMemorySize, stream,
            metaInfo, statisticInfo.expertTokenCount, totalEltCount, enabled, gatheredRawExpertIds);
    }

    if (isLastStage)
    {
        // only last stage need update load factor.
        int threadCount = 128;
        int blockCount = (metaInfo.expertCount + threadCount - 1) / threadCount;
        launchWithPdlWhenEnabled("updateLoadFactor", updateLoadFactorKernel, blockCount, threadCount, 0, stream,
            metaInfo, statisticInfo, statisticInfo.expertTokenCount, enabled);
    }
}

void moeHierarchicalStatisticLocalDevice(MoeLoadBalanceMetaInfo metaInfo, int numTotalTokens,
    int* localExpertTokenCount, int* const enabled, bool isFirstStage, bool isLastStage, int* const localRawExpertIds,
    cudaStream_t stream)
{
    static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
    if (isFirstStage)
    {
        // shift window and zero expertTokenCount
        // only first stage need shift window.
        int threadCount = metaInfo.expertCount;
        auto* kernelFunc = zeroExpertTokenCountKernel<int>;
        if (threadCount % 4 == 0)
        {
            threadCount /= 4;
            kernelFunc = zeroExpertTokenCountKernel<int4>;
        }
        else if (threadCount % 2 == 0)
        {
            threadCount /= 2;
            kernelFunc = zeroExpertTokenCountKernel<int2>;
        }
        dim3 gridDim(1);
        dim3 blockDim(threadCount);
        TLLM_CHECK_WITH_INFO(
            threadCount <= 1024, "expertCount=%d is too large and not supported now.", metaInfo.expertCount);
        launchWithPdlWhenEnabled(
            "zeroExpertTokenCount", kernelFunc, gridDim, blockDim, 0, stream, metaInfo, enabled, localExpertTokenCount);
    }

    {
        // do the statistic into expertTokenCount and maybe also expertLoadFactor;
        int threadCount = 1024;
        int totalEltCount = numTotalTokens * metaInfo.topK;
        int blockCount = (totalEltCount + threadCount - 1) / threadCount;
        if (blockCount > smCount)
        {
            blockCount = smCount;
        }
        int sharedMemorySize = metaInfo.expertCount * sizeof(int);
        launchWithPdlWhenEnabled("statisticKernel", statisticKernel, blockCount, threadCount, sharedMemorySize, stream,
            metaInfo, localExpertTokenCount, totalEltCount, enabled, localRawExpertIds);
    }
}

void moeHierarchicalStatisticUpdate(MoeLoadBalanceMetaInfo metaInfo, MoeLoadBalanceStatisticInfo statisticInfo,
    int* globalExpertTokenCount, int* const enabled, cudaStream_t stream)
{
    int threadCount = 128;
    int blockCount = (metaInfo.expertCount + threadCount - 1) / threadCount;
    launchWithPdlWhenEnabled("updateLoadFactor", updateLoadFactorKernel, blockCount, threadCount, 0, stream, metaInfo,
        statisticInfo, globalExpertTokenCount, enabled);
}

template <int MAX_EXPERT_COUNT = 1024, int THREAD_COUNT = 256, int ITEM_PER_THREAD = 4>
__global__ void moeComputeRouteNoRedundantKernel(MoeLoadBalanceMetaInfo metaInfo, MoePlacementInfo placementInfo,
    int* const tokenSelectedExperts, int* tokenRoutedSlotIds, int tokenCount)
{
    extern __shared__ int16_t sharedGlobalSlotIdsInfo[];
    int expertIds[ITEM_PER_THREAD];
    int slotIds[ITEM_PER_THREAD];

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    for (int slotId = threadIdx.x; slotId < metaInfo.epSize * metaInfo.slotCountPerRank; slotId += THREAD_COUNT)
    {
        sharedGlobalSlotIdsInfo[slotId] = placementInfo.globalSlotIds[slotId];
    }

    int blockOffset = blockIdx.x * THREAD_COUNT * ITEM_PER_THREAD;
    for (; blockOffset < tokenCount * metaInfo.topK; blockOffset += gridDim.x * THREAD_COUNT * ITEM_PER_THREAD)
    {
        int tokenIdxBase = blockOffset + threadIdx.x;
#pragma unroll
        for (int i = 0; i < ITEM_PER_THREAD; i++)
        {
            int tokenIdx = tokenIdxBase + i * THREAD_COUNT;
            expertIds[i]
                = tokenIdx < tokenCount * metaInfo.topK ? tokenSelectedExperts[tokenIdx] : metaInfo.expertCount;
        }
#pragma unroll
        for (int i = 0; i < ITEM_PER_THREAD; i++)
        {
            if (expertIds[i] < 0 || expertIds[i] >= metaInfo.expertCount)
            {
                expertIds[i] = metaInfo.expertCount;
            }
        }
        if (blockOffset == blockIdx.x * THREAD_COUNT * ITEM_PER_THREAD)
        {
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < ITEM_PER_THREAD; i++)
        {
            slotIds[i] = expertIds[i] < metaInfo.expertCount ? sharedGlobalSlotIdsInfo[expertIds[i]]
                                                             : metaInfo.epSize * metaInfo.slotCountPerRank;
        }
#pragma unroll
        for (int i = 0; i < ITEM_PER_THREAD; i++)
        {
            int tokenIdx = tokenIdxBase + i * THREAD_COUNT;
            if (tokenIdx < tokenCount * metaInfo.topK)
            {
                tokenRoutedSlotIds[tokenIdx] = slotIds[i];
            }
        }
    }
}

template <int MAX_EXPERT_COUNT = 1024, int THREAD_COUNT = 256, int ITEM_PER_THREAD = 4>
__global__ void moeComputeRouteKernel(MoeLoadBalanceMetaInfo metaInfo, MoePlacementInfo placementInfo,
    int* const tokenSelectedExperts, int* tokenRoutedSlotIds, int tokenCount, bool offsetByEpRank)
{
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    static int const kWarpCount = THREAD_COUNT / 32;
    extern __shared__ int16_t sharedGlobalSlotIdsInfo[];
    __shared__ int sharedExpertReplicaCountAndStartOffset[MAX_EXPERT_COUNT];

    __shared__ int sharedArbitrateExpertId[THREAD_COUNT * ITEM_PER_THREAD];
    __shared__ int sharedExpertCount[MAX_EXPERT_COUNT];

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    for (int expertIdx = threadIdx.x; expertIdx < metaInfo.expertCount; expertIdx += THREAD_COUNT)
    {
        int replicaCount = placementInfo.expertReplicaCount[expertIdx];
        int replicaStartOffset = placementInfo.expertReplicaStartOffset[expertIdx];
        sharedExpertReplicaCountAndStartOffset[expertIdx] = (replicaCount << 16) | replicaStartOffset;
        sharedExpertCount[expertIdx] = 0;
    }
    for (int slotId = threadIdx.x; slotId < metaInfo.epSize * metaInfo.slotCountPerRank; slotId += THREAD_COUNT)
    {
        sharedGlobalSlotIdsInfo[slotId] = placementInfo.globalSlotIds[slotId];
    }

    int expertIds[ITEM_PER_THREAD];
    int tokenIdxBase = blockIdx.x * THREAD_COUNT * ITEM_PER_THREAD + threadIdx.x;
#pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; i++)
    {
        int tokenIdx = tokenIdxBase + i * THREAD_COUNT;
        expertIds[i] = tokenIdx < tokenCount * metaInfo.topK ? tokenSelectedExperts[tokenIdx] : metaInfo.expertCount;
    }
#pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; i++)
    {
        if (expertIds[i] < 0 || expertIds[i] >= metaInfo.expertCount)
        {
            expertIds[i] = metaInfo.expertCount;
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; i++)
    {
        int countAndStart
            = expertIds[i] < metaInfo.expertCount ? sharedExpertReplicaCountAndStartOffset[expertIds[i]] : (1 << 16);
        int arbitrateExpertId = (countAndStart >> 16) > 1 ? expertIds[i] : metaInfo.expertCount;
        sharedArbitrateExpertId[threadIdx.x + i * THREAD_COUNT] = arbitrateExpertId;
    }
    __syncthreads();
    int baseOffset = blockIdx.x + (offsetByEpRank ? metaInfo.epRank : 0);
    if (warpId == 0)
    {
#pragma unroll
        for (int i = 0; i < kWarpCount * ITEM_PER_THREAD; ++i)
        {
            int expertId = sharedArbitrateExpertId[laneId + i * 32];
            __syncwarp();
            unsigned match = __match_any_sync(0xFFFFFFFF, expertId);
            int leader = __ffs(match) - 1;
            int matchCount = __popc(match);
            int oldVal = 0;
            if (laneId == leader && expertId < metaInfo.expertCount)
            {
                oldVal = atomicAdd_block(&sharedExpertCount[expertId], matchCount);
            }
            __syncwarp();
            oldVal = __shfl_sync(0XFFFFFFFF, oldVal, leader);
            unsigned lowerMask = match & ((1u << laneId) - 1);
            int rankInGroup = __popc(lowerMask);
            int offset = oldVal + rankInGroup;
            offset += baseOffset;
            sharedArbitrateExpertId[laneId + i * 32] = offset;
        }
    }
    __syncthreads();
    int targetGlobalSlotId[ITEM_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; i++)
    {
        int countAndStart
            = expertIds[i] < metaInfo.expertCount ? sharedExpertReplicaCountAndStartOffset[expertIds[i]] : (1 << 16);
        int count = countAndStart >> 16;
        int offset = countAndStart & 0xFFFF;
        int arbitratedIndex = sharedArbitrateExpertId[threadIdx.x + i * THREAD_COUNT];
        offset += arbitratedIndex % count;
        targetGlobalSlotId[i] = expertIds[i] < metaInfo.expertCount ? sharedGlobalSlotIdsInfo[offset]
                                                                    : metaInfo.epSize * metaInfo.slotCountPerRank;
    }
#pragma unroll
    for (int i = 0; i < ITEM_PER_THREAD; i++)
    {
        int tokenIdx = tokenIdxBase + i * THREAD_COUNT;
        if (tokenIdx < tokenCount * metaInfo.topK)
        {
            tokenRoutedSlotIds[tokenIdx] = targetGlobalSlotId[i];
        }
    }
}

template <int MAX_EXPERT_COUNT = 1024, int THREAD_COUNT = 256, int ITEM_PER_THREAD = 4>
__global__ void moeComputeRouteSortKernel(MoeLoadBalanceMetaInfo metaInfo, MoePlacementInfo placementInfo,
    int* const tokenSelectedExperts, int* tokenRoutedSlotIds, int tokenCount, bool offsetByEpRank)
{
    using BlockSort = cub::BlockRadixSort<int, THREAD_COUNT, 1>;
    extern __shared__ int16_t sharedGlobalSlotIdsInfo[];

    __shared__ typename BlockSort::TempStorage tempStorage;

    __shared__ int sharedExpertReplicaCount[MAX_EXPERT_COUNT];
    __shared__ int sharedExpertReplicaStartOffset[MAX_EXPERT_COUNT];

    __shared__ int sharedExpertTokenCount[MAX_EXPERT_COUNT];

    __shared__ int sharedSortedExpertId[THREAD_COUNT * ITEM_PER_THREAD];
    __shared__ int sharedExpertStartThread[MAX_EXPERT_COUNT];

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif

    for (int expertIdx = threadIdx.x; expertIdx < metaInfo.expertCount; expertIdx += THREAD_COUNT)
    {
        sharedExpertTokenCount[expertIdx] = 0;
        sharedExpertStartThread[expertIdx] = -1;

        int replicaCount = placementInfo.expertReplicaCount[expertIdx];
        sharedExpertReplicaCount[expertIdx] = replicaCount;
        sharedExpertReplicaStartOffset[expertIdx] = placementInfo.expertReplicaStartOffset[expertIdx];
    }
    for (int slotId = threadIdx.x; slotId < metaInfo.epSize * metaInfo.slotCountPerRank; slotId += THREAD_COUNT)
    {
        sharedGlobalSlotIdsInfo[slotId] = placementInfo.globalSlotIds[slotId];
    }
    __syncthreads();

    int expertIds[ITEM_PER_THREAD];
    for (int blockOffset = blockIdx.x * THREAD_COUNT * ITEM_PER_THREAD; blockOffset < tokenCount * metaInfo.topK;
         blockOffset += gridDim.x * THREAD_COUNT * ITEM_PER_THREAD)
    {
        int tokenIdxBase = blockOffset + threadIdx.x;
#pragma unroll
        for (int i = 0; i < ITEM_PER_THREAD; i++)
        {
            int tokenIdx = tokenIdxBase + i * THREAD_COUNT;
            expertIds[i]
                = tokenIdx < tokenCount * metaInfo.topK ? tokenSelectedExperts[tokenIdx] : metaInfo.expertCount;
        }
#pragma unroll
        for (int i = 0; i < ITEM_PER_THREAD; i++)
        {
            if (expertIds[i] < 0 || expertIds[i] >= metaInfo.expertCount)
            {
                expertIds[i] = metaInfo.expertCount;
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < ITEM_PER_THREAD; i++)
        {
            constexpr int kMaxThreadBits = 10;
            constexpr int kMasThreadMask = (1 << kMaxThreadBits) - 1;
            int expertId = expertIds[i];
            int tokenIdx = tokenIdxBase + i * THREAD_COUNT;
            bool needCompute = expertId >= 0 && expertId < metaInfo.expertCount;
            int expertIdForSort[1];
            // make lower thread first when have equal key.
            expertIdForSort[0] = (expertId << kMaxThreadBits) | threadIdx.x;

            BlockSort(tempStorage).Sort(expertIdForSort);
            int sortedExpertId = expertIdForSort[0] >> kMaxThreadBits;
            sharedSortedExpertId[threadIdx.x] = sortedExpertId;
            int originalThreadId = (expertIdForSort[0] & kMasThreadMask);
            __syncthreads();
            int sortedExpertIdLast = threadIdx.x > 0 ? sharedSortedExpertId[threadIdx.x - 1] : -1;
            if (sortedExpertIdLast != sortedExpertId && sortedExpertId < metaInfo.expertCount)
            {
                sharedExpertStartThread[sortedExpertId] = threadIdx.x;
            }
            __syncthreads();
            int sortedThreadId[1];
            sortedThreadId[0] = (originalThreadId << kMaxThreadBits) | threadIdx.x;
            BlockSort(tempStorage).Sort(sortedThreadId);

            int idxInBlock = needCompute ? (sortedThreadId[0] & kMasThreadMask) - sharedExpertStartThread[expertId]
                    + sharedExpertTokenCount[expertId]
                                         : 0;

            __syncthreads();

            int targetGlobalSlotId = metaInfo.epSize * metaInfo.slotCountPerRank;
            if (needCompute)
            {
                int replicaCount = sharedExpertReplicaCount[expertId];
                int replicaStartOffset = sharedExpertReplicaStartOffset[expertId];
                int key = blockIdx.x + idxInBlock; // using local round robin here, do we need global round robin?
                if (offsetByEpRank)
                {
                    key += metaInfo.epRank;
                }
                int replicaId = key % replicaCount;
                targetGlobalSlotId = sharedGlobalSlotIdsInfo[replicaStartOffset + replicaId];
                atomicAdd_block(&sharedExpertTokenCount[expertId], 1);
            }
            if (tokenIdx < tokenCount * metaInfo.topK)
            {
                tokenRoutedSlotIds[tokenIdx] = targetGlobalSlotId;
            }
            __syncthreads();
        }
    }
}

void moeComputeRouteDevice(MoeLoadBalanceMetaInfo metaInfo, MoePlacementInfo placementInfo,
    int* const tokenSelectedExperts, int* tokenRoutedSlotIds, int tokenCount, bool offsetByEpRank, cudaStream_t stream)
{
    constexpr int kThreadCount = 256;
    constexpr int kEltPerThread = 4;
    int blockCount = (tokenCount * metaInfo.topK + kThreadCount * kEltPerThread - 1) / (kThreadCount * kEltPerThread);
    int dynamicShmSize = sizeof(int16_t) * metaInfo.epSize * metaInfo.slotCountPerRank;
    if (metaInfo.expertCount == metaInfo.epSize * metaInfo.slotCountPerRank)
    {
        auto* kernelFn = moeComputeRouteNoRedundantKernel<1024, kThreadCount, kEltPerThread>;
        // no redundant expert, so we don't need complex routing, but just assign to the correct solt.
        launchWithPdlWhenEnabled("moeComputeRouteNoRedundant", kernelFn, blockCount, kThreadCount, dynamicShmSize,
            stream, metaInfo, placementInfo, tokenSelectedExperts, tokenRoutedSlotIds, tokenCount);
    }
    else
    {
        auto* kernelFn = moeComputeRouteKernel<1024, kThreadCount, kEltPerThread>;
        launchWithPdlWhenEnabled("moeComputeRoute", kernelFn, blockCount, kThreadCount, dynamicShmSize, stream,
            metaInfo, placementInfo, tokenSelectedExperts, tokenRoutedSlotIds, tokenCount, offsetByEpRank);
    }
}

void moeWaitSignalForCpuStageHost(MoeLoadBalanceSingleLayerSignal* signal)
{
    bool ready = false;
    do
    {
        auto loaded = __atomic_load_n(&signal->stepAndOwner, __ATOMIC_ACQUIRE);
        ready = getOwnerDevice(loaded) == MoeLoadBalanceSingleLayerSignal::kCPU;
    } while (!ready);
    std::atomic_thread_fence(std::memory_order_acquire);
}

void moeSetSignalForGpuStageHost(MoeLoadBalanceSingleLayerSignal* signal, int64_t iterId, bool enableStatistic)
{
    std::atomic_thread_fence(std::memory_order_release);
    bool skipStep = !enableStatistic;
    unsigned long long value = iterId << 2U;
    value += MoeLoadBalanceSingleLayerSignal::kGPU;
    if (skipStep)
    {
        value |= MoeLoadBalanceSingleLayerSignal::kSkipStep;
    }
    __atomic_store_n(&signal->stepAndOwner, value, __ATOMIC_RELEASE);
}

} // namespace kernels
} // namespace tensorrt_llm
