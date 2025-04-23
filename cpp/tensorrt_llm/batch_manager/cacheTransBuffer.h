#pragma once

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

class CacheTransBufferManager
{
public:
    CacheTransBufferManager(
        KVCacheManager::BaseKVCacheManager* cacheManager, std::optional<size_t> maxNumTokens = std::nullopt);

    static size_t preAllocBufferSize(
        std::optional<size_t> maxNumTokens = std::nullopt, std::optional<size_t> kvCacheSizePerToken = std::nullopt);

    std::optional<int> assignBufferIndexForSend();
    void freeBufferIndexForSend(std::optional<int> bufferId);
    std::optional<int> assignBufferIndexForRecv();
    void freeBufferIndexForRecv(std::optional<int> bufferId);

    std::pair<std::vector<runtime::ITensor::SharedPtr>, size_t> getOrAllocateSendBuffers(std::optional<int> bufferId,
        int targetNum, size_t targetBufferSize, runtime::BufferManager const& bufferManagerToUse);

    std::pair<std::vector<runtime::ITensor::SharedPtr>, size_t> getOrAllocateRecvBuffers(std::optional<int> bufferId,
        int targetNum, size_t targetBufferSize, runtime::BufferManager const& bufferManagerToUse);

    runtime::ITensor::SharedPtr getSendBuffer(std::optional<int> bufferId);
    runtime::ITensor::SharedPtr getRecvBuffer(std::optional<int> bufferId);

private:
    struct ConcurrenceResource
    {
        std::unordered_map<int, runtime::ITensor::SharedPtr> mBuffers;
        std::vector<int> mBufferIndexFlag;
        std::mutex mBuffersMutex;
        std::condition_variable mBuffersCV;
        std::atomic<int> mConcurrence = 0;
    };

    std::pair<std::vector<runtime::ITensor::SharedPtr>, size_t> getOrAllocateBuffers(std::optional<int> bufferId,
        int targetNum, size_t targetBufferEleSize, runtime::BufferManager const& bufferManagerToUse,
        ConcurrenceResource& concurrenceResource);

    void allocateBuffer();
    std::optional<int> assignBufferIndex(ConcurrenceResource& resource, size_t bufferCount, bool onlyUseAsyncBuffer);
    void freeBufferIndex(
        ConcurrenceResource& resource, std::optional<int> bufferId, size_t bufferCount, bool onlyUseAsyncBuffer);

    size_t mPreAllocBufferSize;
    size_t mRecvBufferCount;
    size_t mSendBufferCount;
    size_t mTransferBufferSize;
    bool mOnlyUseAsyncBuffer;
    size_t mBufferEleSize;
    nvinfer1::DataType mDataType;
    ConcurrenceResource mConcurrenceSendResource;
    ConcurrenceResource mConcurrenceRecvResource;
    KVCacheManager::BaseKVCacheManager* mCacheManager;
    runtime::BufferManager mBufferManager;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
