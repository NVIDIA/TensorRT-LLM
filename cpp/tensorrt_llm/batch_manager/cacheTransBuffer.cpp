/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <NvInferRuntimeBase.h>
#include <mutex>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

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

    auto alingedSizeFun = []()
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
        return granularity;
    };
    static size_t granularity = alingedSizeFun();

    return (size + granularity - 1) / granularity * granularity;
}

bool FabricMemory::supportFbaricMemory()
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
        TLLM_LOG_DEBUG("FabricMemory::supportFbaricMemory fabric_handle_supported:%d", fabric_handle_supported);
        TLLM_LOG_DEBUG("FabricMemory::supportFbaricMemory gpu_direct_rdma_with_cuda_vmm_supported:%d",
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

CacheTransBufferManager::CacheTransBufferManager(
    KVCacheManager::BaseKVCacheManager* cacheManager, std::optional<size_t> maxNumTokens)
    : mCacheManager{cacheManager}
    , mBufferManager{std::make_shared<runtime::CudaStream>()}
{

    // TODO: FP4 dataSize
    TLLM_CHECK(mCacheManager);
    mDataType = mCacheManager->getPrimaryPool(0)->getDataType();

    auto tokensPerBlock = mCacheManager->getBlockManager().getTokensPerBlock();
    size_t bufferSizeFromMaxNumToken = 0;
    if (maxNumTokens.has_value())
    {
        TLLM_CHECK(maxNumTokens.value() % tokensPerBlock == 0);

        // Transmission always uses FP16 for mixed precision support
        size_t transmissionDataSize = common::getDTypeSize(nvinfer1::DataType::kHALF);
        size_t storageDataSize = common::getDTypeSize(mDataType);

        TLLM_LOG_INFO("=== Buffer Size Calculation Debug ===");
        TLLM_LOG_INFO("maxNumTokens: %ld", maxNumTokens.value());
        TLLM_LOG_INFO("tokensPerBlock: %ld", tokensPerBlock);
        
        // getBlockSize() returns volume of [numKvHeads, tokensPerBlock, sizePerHead] for ONE cache (K or V)
        // We need to multiply by KV factor (2 for SELF, 1 for SELFKONLY) to get both K and V
        size_t blockSizeInElements = mCacheManager->getBlockManager().getBlockSize(0);
        size_t kvFactor = (mCacheManager->getCacheType() == CacheType::kSELFKONLY ? 1 : 2);
        
        TLLM_LOG_INFO("getBlockSize(0): %ld elements (for ONE cache)", blockSizeInElements);
        TLLM_LOG_INFO("kvFactor: %ld (1=K only, 2=K+V)", kvFactor);
        TLLM_LOG_INFO("storageDataSize (mDataType): %ld bytes", storageDataSize);
        TLLM_LOG_INFO("transmissionDataSize (FP16): %ld bytes", transmissionDataSize);
        
        // Calculate bytes per token for both K and V in transmission format (FP16)
        size_t bytesPerTokenInStorage = (blockSizeInElements * kvFactor * storageDataSize) / tokensPerBlock;
        size_t bytesPerTokenInTransmission = (blockSizeInElements * kvFactor * transmissionDataSize) / tokensPerBlock;
        auto kvCacheByteSizePerTokenPerLayer = bytesPerTokenInTransmission;
        
        TLLM_LOG_INFO("bytesPerTokenInStorage: %ld", bytesPerTokenInStorage);
        TLLM_LOG_INFO("bytesPerTokenInTransmission: %ld", bytesPerTokenInTransmission);
        TLLM_LOG_INFO("kvCacheByteSizePerTokenPerLayer: %ld", kvCacheByteSizePerTokenPerLayer);
            
        for (auto layerId = 0; layerId < mCacheManager->getBlockManager().getNumLayers(); layerId++)
        {
            auto poolIdx = mCacheManager->getBlockManager().getLayerPoolIdx(layerId);
            auto windowSize = static_cast<size_t>(mCacheManager->getBlockManager().getPoolWindowSize(poolIdx));
            auto validTokenNum = (windowSize < maxNumTokens.value() ? windowSize : maxNumTokens.value());
            
            TLLM_LOG_INFO("Layer %d: poolIdx=%d, windowSize=%ld, validTokenNum=%ld", 
                layerId, poolIdx, windowSize, validTokenNum);
            TLLM_LOG_INFO("Layer %d: adding %ld bytes (validTokenNum=%ld * kvCacheBytesPerToken=%ld)", 
                layerId, validTokenNum * kvCacheByteSizePerTokenPerLayer, validTokenNum, kvCacheByteSizePerTokenPerLayer);
            
            bufferSizeFromMaxNumToken += validTokenNum * kvCacheByteSizePerTokenPerLayer;
        }

        TLLM_LOG_INFO("Total bufferSizeFromMaxNumToken: %ld bytes", bufferSizeFromMaxNumToken);
        
    }

    mTransferBufferSize
        = maxNumTokens.has_value() ? bufferSizeFromMaxNumToken : common::getEnvMemSizeForKVCacheTransferBuffer();

    TLLM_LOG_INFO("HERE HERE HERE HERE HERE --------- ");
    TLLM_LOG_INFO("mTransferBufferSize:%ld", mTransferBufferSize);
    TLLM_LOG_INFO("HERE HERE HERE HERE HERE --------- ");

    mOnlyUseDynamicBuffer = mTransferBufferSize == 0;
    mRecvBufferCount = common::getEnvRequestKVCacheConcurrent() ? common::getEnvKVCacheRecvBufferCount() : 1;
    mSendBufferCount = common::getEnvParallelCacheSend() ? common::getEnvKVCacheSendMaxConcurrenceNum() : 1;
    mUseFabricMemory = !(common::getEnvKVCacheTransferUseSyncBuffer() || common::getEnvKVCacheTransferUseAsyncBuffer())
        && FabricMemory::supportFbaricMemory();
    if (mUseFabricMemory)
    {
        mTransferBufferSize = FabricMemory::getAlignedSize(mTransferBufferSize);
    }
    mPreAllocBufferSize = mTransferBufferSize * (mRecvBufferCount + mSendBufferCount);
    TLLM_LOG_INFO(
        "CacheTransBufferManager: mMaxNumTokens:%ld, mRecvBufferCount:%ld, "
        "mSendBufferCount:%ld,mTransferBufferSize:%ld, mPreAllocBufferSize:%ld,mOnlyUseDynamicBuffer:%d "
        "mUseFabricMemory:%d mDataType:%d",
        maxNumTokens.has_value() ? maxNumTokens.value() : 0, mRecvBufferCount, mSendBufferCount, mTransferBufferSize,
        mPreAllocBufferSize, mOnlyUseDynamicBuffer, mUseFabricMemory, mDataType);

    allocateBuffer();
}

size_t CacheTransBufferManager::preAllocBufferSize(
    std::map<SizeType32, SizeType32> const& cacheSizeBytesPerTokenPerWindow,
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

    TLLM_LOG_INFO("CALLED preAllocBufferSize (we specified buffer size from cacheTransceiverConfig)");

    auto maxNumTokens = cacheTransceiverConfig->getMaxTokensInBuffer();
    size_t TransferBufferSize = common::getEnvMemSizeForKVCacheTransferBuffer();
    if (maxNumTokens.has_value())
    {
        TransferBufferSize = 0;
        for (auto const& [windowSize, cacheSizeBytesPerToken] : cacheSizeBytesPerTokenPerWindow)
        {
            auto validTokenNum
                = (static_cast<size_t>(windowSize) < maxNumTokens.value() ? static_cast<size_t>(windowSize)
                                                                          : maxNumTokens.value());

            size_t cacheBytes = cacheSizeBytesPerToken;
            if (cacheBytes > 0 && cacheBytes % common::getDTypeSize(nvinfer1::DataType::kFP8) == 0)
            {
                // If cache is stored as FP8 (1 byte) but will be transmitted as FP16 (2 bytes),
                // double the per-token bytes so the recv buffer is large enough.
                cacheBytes = cacheBytes / common::getDTypeSize(nvinfer1::DataType::kFP8)
                    * common::getDTypeSize(nvinfer1::DataType::kHALF);
            }

            TransferBufferSize += validTokenNum * cacheBytes;
        }

        TLLM_LOG_INFO("LOGGING THE TRANSFER BUFFER SIZE");
        TLLM_LOG_INFO("TransferBufferSize:%ld", TransferBufferSize);
        TLLM_LOG_INFO("LOGGING THE TRANSFER BUFFER SIZE");
    }
    bool useFabricMemory = FabricMemory::supportFbaricMemory()
        && (!(common::getEnvKVCacheTransferUseSyncBuffer() || common::getEnvKVCacheTransferUseAsyncBuffer()));
    if (useFabricMemory)
    {
        TransferBufferSize = FabricMemory::getAlignedSize(TransferBufferSize);
    }
    size_t RecvBufferCount = common::getEnvRequestKVCacheConcurrent() ? common::getEnvKVCacheRecvBufferCount() : 1;
    size_t SendBufferCount = common::getEnvParallelCacheSend() ? common::getEnvKVCacheSendMaxConcurrenceNum() : 1;
    size_t PreAllocBufferSize = TransferBufferSize * (RecvBufferCount + SendBufferCount);
    TLLM_LOG_INFO("LOGGING THE ALLOCATED BUFFER SIZE");
    TLLM_LOG_INFO("PreAllocBufferSize:%ld", PreAllocBufferSize);
    TLLM_LOG_INFO("LOGGING THE ALLOCATED BUFFER SIZE");
    return PreAllocBufferSize;
}

std::optional<int> CacheTransBufferManager::assignBufferIndexForSend()
{
    return assignBufferIndex(mConcurrenceSendResource, mSendBufferCount, mOnlyUseDynamicBuffer);
}

void CacheTransBufferManager::freeBufferIndexForSend(std::optional<int> bufferId)
{
    freeBufferIndex(mConcurrenceSendResource, bufferId, mSendBufferCount, mOnlyUseDynamicBuffer);
}

std::optional<int> CacheTransBufferManager::assignBufferIndexForRecv()
{
    return assignBufferIndex(mConcurrenceRecvResource, mRecvBufferCount, mOnlyUseDynamicBuffer);
}

void CacheTransBufferManager::freeBufferIndexForRecv(std::optional<int> bufferId)
{
    freeBufferIndex(mConcurrenceRecvResource, bufferId, mRecvBufferCount, mOnlyUseDynamicBuffer);
}

std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> CacheTransBufferManager::getOrAllocateSendBuffers(
    std::optional<int> bufferId, int targetNum, size_t targetBufferSize,
    runtime::BufferManager const& bufferManagerToUse)
{
    return getOrAllocateBuffers(bufferId, targetNum, targetBufferSize, bufferManagerToUse, mConcurrenceSendResource);
}

std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> CacheTransBufferManager::getOrAllocateRecvBuffers(
    std::optional<int> bufferId, int targetNum, size_t targetBufferSize,
    runtime::BufferManager const& bufferManagerToUse)
{
    return getOrAllocateBuffers(bufferId, targetNum, targetBufferSize, bufferManagerToUse, mConcurrenceRecvResource);
}

runtime::ITensor::SharedPtr CacheTransBufferManager::getSendBuffer(std::optional<int> bufferId)
{
    TLLM_CHECK(bufferId.has_value() || mOnlyUseDynamicBuffer);
    if (bufferId.has_value())
    {
        TLLM_CHECK(static_cast<size_t>(bufferId.value()) < mSendBufferCount);
        return mConcurrenceSendResource.mBuffers[bufferId.value()];
    }
    return nullptr;
}

runtime::ITensor::SharedPtr CacheTransBufferManager::getRecvBuffer(std::optional<int> bufferId)
{
    TLLM_CHECK(bufferId.has_value() || mOnlyUseDynamicBuffer);
    if (bufferId.has_value())
    {
        TLLM_CHECK(static_cast<size_t>(bufferId.value()) < mRecvBufferCount);
        // TLLM_CHECK(mConcurrenceRecvResource.mBufferIndexFlag[bufferId.value()] == 1);
        return mConcurrenceRecvResource.mBuffers[bufferId.value()];
    }
    return nullptr;
}

std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> CacheTransBufferManager::getOrAllocateBuffers(
    std::optional<int> bufferId, int targetNum, size_t targetBufferEleSize,
    runtime::BufferManager const& bufferManagerToUse, ConcurrenceResource& concurrenceResource)
{

    TLLM_LOG_INFO("CALLED getOrAllocateBuffers !!!! !!!!! !! ========= HERE HER EHRE RHERHEHR EHR E");

    TLLM_CHECK(bufferId.has_value() || mOnlyUseDynamicBuffer);
    std::vector<runtime::ITensor::SharedPtr> retSplitCaches;

// fpp16 -> format -> send -> unformat -> fp8


    // size_t bufferCoverTargetNum = std::min(
    //     static_cast<size_t>(targetNum), mTransferBufferSize / (targetBufferEleSize * common::getDTypeSize(mDataType)));

    size_t bufferCoverTargetNum = std::min(
        static_cast<size_t>(targetNum), mTransferBufferSize / (targetBufferEleSize * common::getDTypeSize(nvinfer1::DataType::kHALF)));


    TLLM_LOG_DEBUG("getOrAllocateBuffers bufferCoverTargetNum:%d", bufferCoverTargetNum);
    if (bufferCoverTargetNum < static_cast<size_t>(targetNum))
    {
        TLLM_LOG_WARNING(
            "CacheTransceiver getOrAllocateBuffers: bufferCoverTargetNum:%d < targetNum:%d, may use dynamic buffer, "
            "it's better to increase MaxTokensInBuffer in cacheTransceiverConfig, otherwise, the performance may "
            "be degraded",
            bufferCoverTargetNum, targetNum);
    }
    if (bufferId.has_value())
    {
        TLLM_CHECK(static_cast<size_t>(bufferId.value()) < concurrenceResource.mBuffers.size());
        TLLM_CHECK(concurrenceResource.mBufferIndexFlag[bufferId.value()] == 1);

        for (int i = 0; i < targetNum; i++)
        {
            if (static_cast<size_t>(i) < bufferCoverTargetNum)
            {
                auto slice = runtime::ITensor::slice(
                    concurrenceResource.mBuffers[bufferId.value()], i * targetBufferEleSize, targetBufferEleSize);
                retSplitCaches.push_back(std::move(slice));
            }
            else
            {
                // retSplitCaches.push_back(bufferManagerToUse.gpu(
                //     runtime::ITensor::makeShape({static_cast<int64_t>(targetBufferEleSize)}), mDataType));

                retSplitCaches.push_back(bufferManagerToUse.gpu(
                    runtime::ITensor::makeShape({static_cast<int64_t>(targetBufferEleSize)}), nvinfer1::DataType::kHALF));
            }
        }
    }
    else
    {
        for (int i = 0; i < targetNum; i++)
        {
            // retSplitCaches.push_back(bufferManagerToUse.gpu(
            //     runtime::ITensor::makeShape({static_cast<int64_t>(targetBufferEleSize)}), mDataType));

            retSplitCaches.push_back(bufferManagerToUse.gpu(
                runtime::ITensor::makeShape({static_cast<int64_t>(targetBufferEleSize)}), nvinfer1::DataType::kHALF));
        }
    }
    if (mOnlyUseDynamicBuffer)
    {
        bufferCoverTargetNum = targetNum;
    }
    return std::make_tuple(retSplitCaches, bufferCoverTargetNum, mOnlyUseDynamicBuffer);
}

void CacheTransBufferManager::allocateBuffer()
{
    if (mOnlyUseDynamicBuffer)
    {
        return;
    }

    TLLM_LOG_INFO("CALLED allocateBuffer");

    // mBufferEleSize = mTransferBufferSize / common::getDTypeSize(mDataType);

    mBufferEleSize = mTransferBufferSize / common::getDTypeSize(nvinfer1::DataType::kHALF);

    mConcurrenceSendResource.mBufferIndexFlag.resize(mSendBufferCount, 0);
    mConcurrenceRecvResource.mBufferIndexFlag.resize(mRecvBufferCount, 0);

    TLLM_LOG_INFO("mSendBufferCount:%ld", mSendBufferCount);
    TLLM_LOG_INFO("mRecvBufferCount:%ld", mRecvBufferCount);

    if (mUseFabricMemory)
    {
        TLLM_LOG_INFO("CALLED allocateBuffer WITH FABRIC MEMORY");
        mFabricMemory.reserve(mSendBufferCount + mRecvBufferCount);
        for (size_t i = 0; i < mSendBufferCount; i++)
        {
            mFabricMemory.emplace_back(std::make_unique<FabricMemory>(mTransferBufferSize));
            // Use FP16 for transmission buffers
            mConcurrenceSendResource.mBuffers[i] = runtime::ITensor::wrap(mFabricMemory.back()->getPtr(), nvinfer1::DataType::kHALF,
                runtime::ITensor::makeShape({static_cast<int64_t>(mBufferEleSize)}), mBufferEleSize);
        }
        for (size_t i = 0; i < mRecvBufferCount; i++)
        {
            mFabricMemory.emplace_back(std::make_unique<FabricMemory>(mTransferBufferSize));
            // Use FP16 for transmission buffers
            mConcurrenceRecvResource.mBuffers[i] = runtime::ITensor::wrap(mFabricMemory.back()->getPtr(), nvinfer1::DataType::kHALF,
                runtime::ITensor::makeShape({static_cast<int64_t>(mBufferEleSize)}), mBufferEleSize);
        }
    }
    else if (common::getEnvKVCacheTransferUseAsyncBuffer())
    {
        TLLM_LOG_INFO("CALLED allocateBuffer WITH ASYNC BUFFER");
        for (size_t i = 0; i < mSendBufferCount; i++)
        {
            // Use FP16 for transmission buffers
            mConcurrenceSendResource.mBuffers[i]
                = mBufferManager.gpu(runtime::ITensor::makeShape({static_cast<int64_t>(mBufferEleSize)}), nvinfer1::DataType::kHALF);
        }
        for (size_t i = 0; i < mRecvBufferCount; i++)
        {
            // Use FP16 for transmission buffers
            mConcurrenceRecvResource.mBuffers[i]
                = mBufferManager.gpu(runtime::ITensor::makeShape({static_cast<int64_t>(mBufferEleSize)}), nvinfer1::DataType::kHALF);
        }
        mBufferManager.getStream().synchronize();
    }
    else
    {
        TLLM_LOG_INFO("CALLED allocateBuffer WITH SYNC BUFFER");
        for (size_t i = 0; i < mSendBufferCount; i++)
        {
            // mConcurrenceSendResource.mBuffers[i] = mBufferManager.gpuSync(
            //     runtime::ITensor::makeShape({static_cast<int64_t>(mBufferEleSize)}), mDataType);

            mConcurrenceSendResource.mBuffers[i] = mBufferManager.gpuSync(
                runtime::ITensor::makeShape({static_cast<int64_t>(mBufferEleSize)}), nvinfer1::DataType::kHALF);

        }
        for (size_t i = 0; i < mRecvBufferCount; i++)
        {
            // mConcurrenceRecvResource.mBuffers[i] = mBufferManager.gpuSync(
            //     runtime::ITensor::makeShape({static_cast<int64_t>(mBufferEleSize)}), mDataType);

            mConcurrenceRecvResource.mBuffers[i] = mBufferManager.gpuSync(
                runtime::ITensor::makeShape({static_cast<int64_t>(mBufferEleSize)}), nvinfer1::DataType::kHALF);
        }
    }
}

std::optional<int> CacheTransBufferManager::assignBufferIndex(
    ConcurrenceResource& resource, size_t bufferCount, bool onlyUseDynamicBuffer)
{
    if (onlyUseDynamicBuffer)
    {
        return std::nullopt;
    }
    std::unique_lock lk(resource.mBuffersMutex);
    resource.mBuffersCV.wait(
        lk, [&resource, bufferCount]() { return static_cast<size_t>(resource.mConcurrence) < bufferCount; });
    int bufferId = -1;
    for (size_t i = 0; i < bufferCount; i++)
    {
        if (resource.mBufferIndexFlag[i] == 0)
        {
            bufferId = i;
            resource.mBufferIndexFlag[bufferId] = 1;
            resource.mConcurrence++;
            break;
        }
    }
    TLLM_CHECK_WITH_INFO(bufferId >= 0 && static_cast<size_t>(bufferId) < bufferCount,
        " assignBufferIndex: Buffer index already assigned");

    return bufferId;
}

void CacheTransBufferManager::freeBufferIndex(
    ConcurrenceResource& resource, std::optional<int> bufferId, size_t bufferCount, bool onlyUseDynamicBuffer)
{
    if (onlyUseDynamicBuffer)
    {
        return;
    }
    if (bufferId.has_value())
    {

        TLLM_CHECK(static_cast<size_t>(bufferId.value()) < bufferCount);
        {
            std::scoped_lock lk(resource.mBuffersMutex);
            resource.mBufferIndexFlag[bufferId.value()] = 0;
        }
        resource.mConcurrence--;
        resource.mBuffersCV.notify_one();
    }
}

size_t CacheTransBufferManager::getRecvBufferCount()
{
    return mRecvBufferCount;
}

size_t CacheTransBufferManager::getSendBufferCount()
{
    return mSendBufferCount;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
