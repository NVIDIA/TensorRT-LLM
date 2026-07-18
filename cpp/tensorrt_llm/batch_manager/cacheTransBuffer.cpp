/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cacheTransBuffer.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/executor/executor.h"

#include "tensorrt_llm/common/tllmDataType.h"
#include <mutex>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

namespace
{

bool isCachePool(BlockManager const& blockManager, SizeType32 poolIdx)
{
    auto const& pool = blockManager.getPool(poolIdx);
    return !pool.containsBlockScales && !pool.containsIndexerKCache;
}

bool isAttentionCachePool(BlockManager const& blockManager, SizeType32 poolIdx)
{
    return isCachePool(blockManager, poolIdx)
        && !LinearAttentionMetadata::hasLinearCache(blockManager.getPoolWindowSize(poolIdx));
}

tensorrt_llm::DataType getTransferDataType(KVCacheManager::BaseKVCacheManager* cacheManager, bool transferIndexerKCache)
{
    TLLM_CHECK(cacheManager);
    if (transferIndexerKCache)
    {
        auto const indexerKCachePool = cacheManager->getIndexerKCachePool();
        TLLM_CHECK(indexerKCachePool);
        return indexerKCachePool->getDataType();
    }

    auto const& blockManager = cacheManager->getBlockManager();
    std::optional<tensorrt_llm::DataType> cacheDataType;
    std::optional<tensorrt_llm::DataType> attentionDataType;
    SizeType32 firstPoolIdx = -1;
    // Recurrent-state pools have a separate transfer manager and formatter. Only
    // attention pools determine the KV transfer-buffer dtype.
    for (SizeType32 poolIdx = 0; poolIdx < blockManager.getNumPools(); ++poolIdx)
    {
        if (!isCachePool(blockManager, poolIdx))
        {
            continue;
        }

        auto const poolDataType = blockManager.getPrimaryPool(poolIdx)->getDataType();
        if (!cacheDataType.has_value())
        {
            cacheDataType = poolDataType;
        }
        if (!isAttentionCachePool(blockManager, poolIdx))
        {
            continue;
        }
        if (!attentionDataType.has_value())
        {
            attentionDataType = poolDataType;
            firstPoolIdx = poolIdx;
            continue;
        }

        TLLM_CHECK_WITH_INFO(poolDataType == attentionDataType.value(),
            "Disaggregated KV cache transfer does not yet support attention pools with differing dtypes "
            "(pool %d dtype=%d, pool %d dtype=%d). TODO(disagg-multi-dtype): per-pool dtype dispatch in formatter.",
            firstPoolIdx, static_cast<int>(attentionDataType.value()), poolIdx, static_cast<int>(poolDataType));
    }

    TLLM_CHECK_WITH_INFO(cacheDataType.has_value(), "Disaggregated KV cache transfer requires a cache pool");
    return attentionDataType.value_or(cacheDataType.value());
}

} // namespace

// ============================================================================
// FabricMemory Implementation
// ============================================================================

class FabricMemory::Impl
{
public:
    Impl(size_t size)
        : mSize(size)
    {
        TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceIdx));
        CUmemAllocationHandleType const handle_type = CU_MEM_HANDLE_TYPE_FABRIC;
        CUmemAllocationProp prop = {};
        prop.requestedHandleTypes = handle_type;
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = mDeviceIdx;
        prop.allocFlags.gpuDirectRDMACapable = 1;

        size_t granularity{0};
        TLLM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        mGranularity = granularity;
        mAllocSize = (size + granularity - 1) / granularity * granularity;
        TLLM_CU_CHECK(cuMemCreate(&mHandle, mAllocSize, &prop, 0));
        TLLM_CU_CHECK(cuMemAddressReserve(&mDevicePtr, mAllocSize, mGranularity, 0, 0));
        mPtr = reinterpret_cast<void*>(mDevicePtr);
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        accessDesc.location.id = mDeviceIdx;
        TLLM_CU_CHECK(cuMemMap(mDevicePtr, mAllocSize, 0, mHandle, 0));
        TLLM_CU_CHECK(cuMemSetAccess(mDevicePtr, mAllocSize, &accessDesc, 1));
        TLLM_LOG_DEBUG("FabricMemory::Impl::Impl mAllocSize:%ld", mAllocSize);
    }

    ~Impl()
    {
        TLLM_LOG_DEBUG("FabricMemory::Impl::~Impl mAllocSize:%ld", mAllocSize);
        TLLM_CU_CHECK(cuMemUnmap(mDevicePtr, mAllocSize));
        TLLM_CU_CHECK(cuMemRelease(mHandle));
        TLLM_CU_CHECK(cuMemAddressFree(mDevicePtr, mAllocSize));
    }

    [[nodiscard]] void* getPtr() const
    {
        return mPtr;
    }

    [[nodiscard]] size_t getSize() const
    {
        return mSize;
    }

private:
    size_t mSize;
    size_t mAllocSize;
    size_t mGranularity;
    void* mPtr;
    CUdeviceptr mDevicePtr;
    CUmemGenericAllocationHandle mHandle;
    int mDeviceIdx;
};

FabricMemory::FabricMemory(size_t size)
    : pImpl(std::make_unique<Impl>(size))
{
}

FabricMemory::~FabricMemory() = default;

FabricMemory::FabricMemory(FabricMemory&&) noexcept = default;
FabricMemory& FabricMemory::operator=(FabricMemory&&) noexcept = default;

void* FabricMemory::getPtr() const
{
    return pImpl->getPtr();
}

size_t FabricMemory::getSize() const
{
    return pImpl->getSize();
}

size_t FabricMemory::getAlignedSize(size_t size)
{
    int deviceIdx = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&deviceIdx));
    CUmemAllocationHandleType const handle_type = CU_MEM_HANDLE_TYPE_FABRIC;
    CUmemAllocationProp prop = {};
    prop.requestedHandleTypes = handle_type;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = deviceIdx;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    size_t granularity{0};
    TLLM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    return (size + granularity - 1) / granularity * granularity;
}

bool FabricMemory::supportFabricMemory()
{
#ifdef __aarch64__
    auto support_fun = []()
    {
        int fabric_handle_supported{0};
        int gpu_direct_rdma_with_cuda_vmm_supported{0};
        int deviceIdx = 0;
        TLLM_CUDA_CHECK(cudaGetDevice(&deviceIdx));
        CUresult ret0 = cuDeviceGetAttribute(
            &fabric_handle_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, deviceIdx);

        CUresult ret1 = cuDeviceGetAttribute(&gpu_direct_rdma_with_cuda_vmm_supported,
            CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, deviceIdx);
        TLLM_LOG_DEBUG("FabricMemory::supportFabricMemory fabric_handle_supported:%d", fabric_handle_supported);
        TLLM_LOG_DEBUG("FabricMemory::supportFabricMemory gpu_direct_rdma_with_cuda_vmm_supported:%d",
            gpu_direct_rdma_with_cuda_vmm_supported);
        if (ret0 != CUresult::CUDA_SUCCESS || ret1 != CUresult::CUDA_SUCCESS || fabric_handle_supported == 0
            || gpu_direct_rdma_with_cuda_vmm_supported == 0)
        {
            return false;
        }

        CUmemAllocationHandleType const handle_type = CU_MEM_HANDLE_TYPE_FABRIC;
        CUmemAllocationProp prop = {};
        prop.requestedHandleTypes = handle_type;
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = deviceIdx;
        prop.allocFlags.gpuDirectRDMACapable = 1;

        size_t granularity{0};
        TLLM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        CUmemGenericAllocationHandle handle;

        auto cuRet = cuMemCreate(&handle, granularity, &prop, 0);

        if (cuRet == CUresult::CUDA_SUCCESS)
        {
            TLLM_CU_CHECK(cuMemRelease(handle));
            return true;
        }
        if (cuRet == CUresult::CUDA_ERROR_NOT_PERMITTED)
        {
            TLLM_LOG_WARNING("Try to creat fabric memory failed , setting imex channel may be required");
            return false;
        }
        TLLM_CU_CHECK(cuRet);

        return false;
    };
    static bool support = support_fun();
    return support;

#else
    return false;
#endif
}

// ============================================================================
// CacheTransBufferManager Implementation
// ============================================================================

size_t CacheTransBufferManager::computeTransferBufferSize(
    KVCacheManager::BaseKVCacheManager* cacheManager, std::optional<size_t> maxNumTokens, bool transferIndexerKCache)
{
    auto const dataType = getTransferDataType(cacheManager, transferIndexerKCache);

    auto const& blockManager = cacheManager->getBlockManager();
    auto const tokensPerBlock = blockManager.getTokensPerBlock();
    bool hasAttentionCachePool = false;
    for (SizeType32 poolIdx = 0; poolIdx < blockManager.getNumPools(); ++poolIdx)
    {
        hasAttentionCachePool |= isAttentionCachePool(blockManager, poolIdx);
    }
    size_t bufferSizeFromMaxNumToken = 0;

    if (maxNumTokens.has_value())
    {
        TLLM_CHECK(maxNumTokens.value() % tokensPerBlock == 0);
        auto const dataSize = common::getDTypeSize(dataType);
        SizeType32 indexerCacheByteSizePerTokenPerLayer = 0;
        if (transferIndexerKCache)
        {
            indexerCacheByteSizePerTokenPerLayer
                = cacheManager->getIndexerKCachePool()->getDimension<-1>() * dataSize / tokensPerBlock;
        }
        for (auto layerId = 0; layerId < blockManager.getNumLayers(); layerId++)
        {
            auto const poolIdx = blockManager.getLayerPoolIdx(layerId);
            auto const encodedWindowSize = blockManager.getPoolWindowSize(poolIdx);
            if (!transferIndexerKCache && hasAttentionCachePool
                && LinearAttentionMetadata::hasLinearCache(encodedWindowSize))
            {
                continue;
            }

            auto const windowSize = static_cast<size_t>(encodedWindowSize);
            auto alignedWindowSize = (windowSize + tokensPerBlock - 1) / tokensPerBlock * tokensPerBlock;
            auto validTokenNum = (alignedWindowSize < maxNumTokens.value() ? alignedWindowSize : maxNumTokens.value());
            if (common::getEnvKVCacheTransferAllBlocksForWindow())
            {
                validTokenNum = maxNumTokens.value();
            }
            validTokenNum += tokensPerBlock; // add one more block

            if (transferIndexerKCache)
            {
                bufferSizeFromMaxNumToken += validTokenNum * indexerCacheByteSizePerTokenPerLayer;
            }
            else
            {
                auto const primaryPool = blockManager.getPrimaryPool(poolIdx);
                auto const kvCacheByteSizePerTokenPerLayer
                    = primaryPool->getDimension<-1>() * primaryPool->getDimension<2>() * dataSize / tokensPerBlock;
                bufferSizeFromMaxNumToken += validTokenNum * kvCacheByteSizePerTokenPerLayer;
            }
        }
    }

    return maxNumTokens.has_value() ? bufferSizeFromMaxNumToken : common::getEnvMemSizeForKVCacheTransferBuffer();
}

CacheTransBufferManager::CacheTransBufferManager(
    KVCacheManager::BaseKVCacheManager* cacheManager, std::optional<size_t> maxNumTokens, bool transferIndexerKCache)
    : BaseTransBufferManager(computeTransferBufferSize(cacheManager, maxNumTokens, transferIndexerKCache),
        getTransferDataType(cacheManager, transferIndexerKCache), maxNumTokens)
    , mCacheManager{cacheManager}
    , mTransferIndexerKCache{transferIndexerKCache}
{
    // TODO: FP4 dataSize
    TLLM_CHECK(mCacheManager);
    TLLM_LOG_INFO("CacheTransBufferManager created for KV cache");
}

size_t CacheTransBufferManager::preAllocBufferSize(
    std::map<SizeType32, SizeType32> const& cacheSizeBytesPerTokenPerWindow, SizeType32 tokensPerBlock,
    std::optional<executor::CacheTransceiverConfig> const& cacheTransceiverConfig)
{
    if (!cacheTransceiverConfig.has_value())
    {
        return 0;
    }
    if (!cacheTransceiverConfig->getBackendType().has_value())
    {
        return 0;
    }
    auto maxNumTokens = cacheTransceiverConfig->getMaxTokensInBuffer();
    size_t transferBufferSize = common::getEnvMemSizeForKVCacheTransferBuffer();
    if (maxNumTokens.has_value())
    {
        transferBufferSize = 0;
        for (auto const& [windowSize, cacheSizeBytesPerToken] : cacheSizeBytesPerTokenPerWindow)
        {
            auto alignedWindowSize = (windowSize + tokensPerBlock - 1) / tokensPerBlock * tokensPerBlock;
            auto validTokenNum = (static_cast<size_t>(alignedWindowSize) < maxNumTokens.value()
                    ? static_cast<size_t>(alignedWindowSize)
                    : maxNumTokens.value());
            if (common::getEnvKVCacheTransferAllBlocksForWindow())
            {
                validTokenNum = maxNumTokens.value();
            }
            validTokenNum += tokensPerBlock; // add one more block
            transferBufferSize += validTokenNum * cacheSizeBytesPerToken;
        }
    }
    bool useFabricMemory = FabricMemory::supportFabricMemory()
        && (!(common::getEnvKVCacheTransferUseSyncBuffer() || common::getEnvKVCacheTransferUseAsyncBuffer()));
    if (useFabricMemory)
    {
        transferBufferSize = FabricMemory::getAlignedSize(transferBufferSize);
    }
    size_t recvBufferCount = common::getEnvRequestKVCacheConcurrent() ? common::getEnvKVCacheRecvBufferCount() : 1;
    size_t sendBufferCount = common::getEnvKVCacheSendMaxConcurrenceNum();
    size_t preAllocBufferSize = transferBufferSize * (recvBufferCount + sendBufferCount);
    return preAllocBufferSize;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
