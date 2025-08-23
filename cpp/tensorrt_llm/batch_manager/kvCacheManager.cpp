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

#include "tensorrt_llm/batch_manager/kvCacheManager.h"

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/evictionPolicy.h"
#include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <algorithm>
#include <map>
#include <optional>
#include <utility>

namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
namespace tle = tensorrt_llm::executor;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::batch_manager::eviction_policy;

namespace
{

//! \brief Split vector into list of blocks of given size.
//! \param vec vector to split
//! \param usableSize part of the vector that is processed
//! \param elementsPerBlock desired size of blocks
//! \param allowPartial whether to append a block smaller than `elementsPerBlock` at the end
//! \return list of blocks
template <typename T>
std::list<std::vector<T>> chopVectorIntoBlocks(
    std::vector<T> const& vec, SizeType32 usableSize, SizeType32 elementsPerBlock, bool allowPartial)
{
    TLLM_CHECK_WITH_INFO(
        usableSize <= static_cast<SizeType32>(vec.size()), "usableSize=%d > %ld=vec.size()", usableSize, vec.size());
    std::list<std::vector<T>> blockedVectors;
    auto const vecEnd = vec.begin() + usableSize;
    for (auto begin = vec.begin(); begin < vecEnd; begin += elementsPerBlock)
    {
        auto blockSize = std::min(elementsPerBlock, static_cast<SizeType32>(std::distance(begin, vecEnd)));
        auto end = begin + blockSize;
        if (blockSize == elementsPerBlock || allowPartial)
        {
            blockedVectors.emplace_back(begin, end);
        }
    }
    return blockedVectors;
}

std::vector<BlockKey> buildBlockKeys(
    std::list<VecUniqueTokens>& blockedUniqueTokens, tensorrt_llm::batch_manager::LlmRequest const& llmRequest)
{
    std::vector<BlockKey> blockKeys;
    for (auto& uniqueTokens : blockedUniqueTokens)
    {
        blockKeys.emplace_back(
            llmRequest.getInputTokensExtraIds().has_value(), llmRequest.getLoraTaskId(), std::move(uniqueTokens));
    }
    return blockKeys;
}

} // namespace

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

KVCacheBlock::KVCacheBlock(IdType blockId, tk::KVCacheIndex blockIdx)
    : mBlockId(blockId)
    , mMemoryPoolBlockIndex{blockIdx}
    , mRefCount(0)
    , mSchedulingRefCount(0)
    , mPrevBlock(nullptr)
    , mFreeBlockIterator(std::nullopt)
    , mIsFull{false}
    , mPriority{executor::KvCacheRetentionConfig::kDefaultRetentionPriority}
    , mDurationMs{std::nullopt}
    , mExpirationTime{std::nullopt}
    , mHash{0}
{
}

void KVCacheBlock::startScheduling()
{
    mSchedulingRefCount = mRefCount;
}

KVCacheBlock::IdType KVCacheBlock::getBlockId() const
{
    return mBlockId;
}

NextBlockMap KVCacheBlock::getNextBlocks() const
{
    return mNextBlocks;
}

tk::KVCacheIndex::UnderlyingType KVCacheBlock::getMemoryPoolBlockIndex() const
{
    return mMemoryPoolBlockIndex.get();
}

bool KVCacheBlock::isPrimary() const
{
    return mMemoryPoolBlockIndex.isPrimary();
}

void KVCacheBlock::swapMemoryPoolBlockOffset(std::shared_ptr<KVCacheBlock> otherBlock)
{
    std::swap(mMemoryPoolBlockIndex, otherBlock->mMemoryPoolBlockIndex);
}

void KVCacheBlock::incRefCount()
{
    mRefCount++;
}

void KVCacheBlock::decRefCount()
{
    TLLM_CHECK_WITH_INFO(hasRefs(), "Can't remove link from block that is not allocated");
    mRefCount--;
}

void KVCacheBlock::decSchedulingRefCount()
{
    TLLM_CHECK_WITH_INFO(hasSchedulingRefs(), "Can't remove link from block that is not allocated");
    mSchedulingRefCount--;
}

bool KVCacheBlock::hasRefs() const
{
    return mRefCount > 0;
}

bool KVCacheBlock::isShared() const
{
    // block is considered shared if ready for reuse
    return mRefCount > 1 || mPrevBlock != nullptr;
}

bool KVCacheBlock::hasSchedulingRefs() const
{
    return mSchedulingRefCount > 0;
}

void KVCacheBlock::setBlockKey(BlockKey const& blockKey, bool isFull)
{
    mBlockKey = blockKey;
    mIsFull = isFull;
}

BlockKey KVCacheBlock::getBlockKey()
{
    return mBlockKey;
}

void KVCacheBlock::setPriority(executor::RetentionPriority priority)
{
    mPriority = priority;
}

executor::RetentionPriority KVCacheBlock::getPriority() const
{
    return mPriority;
}

std::optional<std::chrono::milliseconds> KVCacheBlock::getDurationMs() const
{
    return mDurationMs;
}

void KVCacheBlock::setDurationMs(std::optional<std::chrono::milliseconds> durationMs)
{
    mDurationMs = durationMs;
}

void KVCacheBlock::setExpirationTime(std::optional<std::chrono::steady_clock::time_point::duration> expirationTime)
{
    mExpirationTime = expirationTime;
}

std::optional<std::chrono::steady_clock::time_point::duration> KVCacheBlock::getExpirationTime() const
{
    return mExpirationTime;
}

void KVCacheBlock::setHash(size_t hash)
{
    mHash = hash;
}

void KVCacheBlock::setHash()
{
    mHash = BlockKeyHasher()(mBlockKey, mPrevBlockInSeq ? mPrevBlockInSeq->getHash() : 0);
}

size_t KVCacheBlock::getHash() const
{
    return mHash;
}

VecUniqueTokens const& KVCacheBlock::getUniqueTokens() const
{
    return mBlockKey.uniqueTokens;
}

BlockPtr const& KVCacheBlock::getPrevBlock() const
{
    return mPrevBlock;
}

void KVCacheBlock::setPrevBlock(BlockPtr prevBlock)
{
    mPrevBlock = std::move(prevBlock);
}

BlockPtr const& KVCacheBlock::getPrevBlockInSeq() const
{
    return mPrevBlockInSeq;
}

void KVCacheBlock::setPrevBlockInSeq(BlockPtr prevBlock)
{
    mPrevBlockInSeq = std::move(prevBlock);
}

void KVCacheBlock::addNextBlock(BlockKey const& blockKey, BlockPtr block)
{
    if (mNextBlocks.find(blockKey) == mNextBlocks.end())
    {
        mNextBlocks[blockKey] = std::move(block);
    }
}

std::tuple<bool, SizeType32, BlockPtr> KVCacheBlock::findMatchingBlock(
    BlockKey const& blockKey, bool enablePartialReuse, bool copyOnPartialReuse) const
{
    if (blockKey.uniqueTokens.size() == 0 || mNextBlocks.size() == 0)
    {
        return {false, 0, nullptr};
    }
    auto itr = mNextBlocks.find(blockKey);
    if (itr == mNextBlocks.end())
    {
        if (enablePartialReuse)
        {
            SizeType32 bestNumMatched{0};
            BlockPtr bestBlock{nullptr};
            for (auto const& [key, block] : mNextBlocks)
            {
                if (copyOnPartialReuse || (!block->hasRefs() && block->isLeaf()))
                {
                    SizeType32 numMatched = key.partialMatch(blockKey);
                    if (numMatched > bestNumMatched)
                    {
                        bestNumMatched = numMatched;
                        bestBlock = block;
                    }
                }
            }
            if (bestNumMatched > 0)
            {
                return {true, bestNumMatched, bestBlock};
            }
        }
        return {false, 0, nullptr};
    }
    auto block = itr->second;
    return {!block->isFull(), static_cast<SizeType32>(blockKey.uniqueTokens.size()), block};
}

void KVCacheBlock::freeLeafBlock()
{
    // assure that this is a leaf block
    TLLM_CHECK(isLeaf());

    // free from previous block
    if (mPrevBlock != nullptr)
    {
        mPrevBlock->removeNextBlock(mBlockKey);
        mPrevBlock = nullptr;
    }
}

void KVCacheBlock::removeNextBlock(BlockKey const& blockKey)
{
    mNextBlocks.erase(blockKey);
}

bool KVCacheBlock::isFull() const
{
    return mIsFull;
}

bool KVCacheBlock::isLeaf() const
{
    return mNextBlocks.empty();
}

BlockManager::BlockManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool,
    SizeType32 maxNumSequences, std::shared_ptr<runtime::CudaStream> stream, bool onboardBlocks, CacheType cacheType,
    std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enableHashKey, bool enablePartialReuse,
    bool copyOnPartialReuse)
    : mNumPrimaryBlocks{blocksInPrimaryPool}
    , mNumSecondaryBlocks{blocksInSecondaryPool}
    , mOnboardBlocks(onboardBlocks)
    , mBufferManager{std::move(stream)}
    , mSizePerHead{sizePerHead}
    , mNumLayers{static_cast<SizeType32>(numKvHeadsPerLayer.size())}
    , mSchedulingNumFreeBlocks{0}
    , mTokensPerBlock{tokensPerBlock}
    , mCachedBlocksRoot{std::make_shared<KVCacheBlock>(KVCacheBlock::kCachedBlocksRootId, tk::KVCacheIndex{0})}
    , mCacheType{cacheType}
    , mEventManager(std::move(eventManager))
    , mTransferManager{std::make_shared<KVCacheTransferManager>(mBufferManager)}
    , mAllocTotalBlocks{0}
    , mAllocNewBlocks{0}
    , mReusedBlocks{0}
    , mReusedUniqueBlocks{0}
    , mMissedBlocks{0}
    , mReusedTokens{0.0}
    , mTotalInputTokens{0.0}
    , mEnableHashKey{enableHashKey}
    , mEnablePartialReuse{enablePartialReuse}
    , mCopyOnPartialReuse{copyOnPartialReuse}
{
    std::map<SizeType32, SizeType32> numLayersPerPool;

    mKVFactor = (cacheType == CacheType::kSELFKONLY) ? 1 : 2;

    // count how many layers should go in each pool
    mLayerIndexToPoolLayerIndex.reserve(mNumLayers);
    for (SizeType32 layerIdx = 0; layerIdx < mNumLayers; layerIdx++)
    {
        auto const numKvHeads = numKvHeadsPerLayer.at(layerIdx);
        auto search = numLayersPerPool.find(numKvHeads);
        numLayersPerPool[numKvHeads] = search == numLayersPerPool.end() ? 1 : search->second + 1;
        mLayerIndexToPoolLayerIndex.emplace_back(numLayersPerPool[numKvHeads] - 1);
    }

    for (auto const [numKvHeads, numLayers] : numLayersPerPool)
    {
        mPools.emplace_back(numLayers, numKvHeads, sizePerHead, tokensPerBlock, 1);
    }

    // assign each layer to its pool
    mLayerToPool.reserve(mNumLayers);
    for (SizeType32 layerIdx = 0; layerIdx < mNumLayers; layerIdx++)
    {
        auto poolPos = std::find_if(mPools.cbegin(), mPools.cend(),
            [numKvHeads = numKvHeadsPerLayer[layerIdx]](KVCacheBlockPool const& pool)
            { return numKvHeads == pool.numKvHeads; });
        mLayerToPool.emplace_back(poolPos - mPools.cbegin());
    }

    // Create free blocks
    mAllBlocksById.reserve(blocksInPrimaryPool + blocksInSecondaryPool);
    for (KVCacheBlock::IdType blockId = 0; blockId < blocksInPrimaryPool; ++blockId)
    {
        mAllBlocksById.emplace_back(std::make_shared<KVCacheBlock>(blockId, tk::KVCacheIndex{blockId, false}));
    }
    for (KVCacheBlock::IdType blockId = 0; blockId < blocksInSecondaryPool; ++blockId)
    {
        mAllBlocksById.emplace_back(
            std::make_shared<KVCacheBlock>(blocksInPrimaryPool + blockId, tk::KVCacheIndex{blockId, true}));
    }
    mAllocatedBlocksPerSeq.reserve(maxNumSequences);

    mEvictionPolicy = std::make_shared<LRUEvictionPolicy>();
    mEvictionPolicy->initialize(
        mAllBlocksById, {blocksInPrimaryPool, blocksInSecondaryPool}, secondaryOffloadMinPriority);
    if (mEventManager)
    {
        mEventManager->enqueueCreatedEvent({blocksInPrimaryPool, blocksInSecondaryPool});
    }
}

BlockManager::~BlockManager()
{
    float reusedUniqueBlocksPercentage = mReusedUniqueBlocks == 0 || mAllocTotalBlocks == 0
        ? 0
        : static_cast<float>(mReusedUniqueBlocks) / static_cast<float>(mAllocNewBlocks) * 100;
    float cacheHitRate = mReusedBlocks == 0
        ? 0
        : static_cast<float>(mReusedBlocks) / (static_cast<float>(mReusedBlocks + mMissedBlocks));
    TLLM_LOG_DEBUG("BlockManager - total allocated blocks:              %lu  ", mAllocTotalBlocks);
    TLLM_LOG_DEBUG("BlockManager - allocated new blocks:                %lu  ", mAllocNewBlocks);
    TLLM_LOG_DEBUG("BlockManager - missed blocks:                       %lu  ", mMissedBlocks);
    TLLM_LOG_DEBUG("BlockManager - reused blocks:                       %lu  ", mReusedBlocks);
    TLLM_LOG_DEBUG("BlockManager - reused unique blocks:                %lu  ", mReusedUniqueBlocks);
    TLLM_LOG_DEBUG("BlockManager - reused unique blocks percentage (%%): %.2f ", reusedUniqueBlocksPercentage);
    TLLM_LOG_DEBUG("BlockManager - cache hit rate:                      %.2f ", cacheHitRate);
    TLLM_LOG_DEBUG("BlockManager - reused tokens:                       %.0f ", mReusedTokens);
    TLLM_LOG_DEBUG(
        "BlockManager - reused tokens percentage (%%):        %.2f ", 100.0 * mReusedTokens / mTotalInputTokens);
}

bool BlockManager::verifyQueueIntegrity()
{
    return mEvictionPolicy->verifyQueueIntegrity();
}

void BlockManager::storeContextBlocks(GenerationRequest& sequence, LlmRequest const& llmRequest)
{
    constexpr int beamIdx = 0; // no need to consider more than one beam for input tokens
    auto const& cacheBlockIds = sequence.getCacheBlockIds();
    auto const& uniqueTokens = llmRequest.getUniqueTokens(beamIdx);

    auto blockedUniqueTokens
        = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, uniqueTokens.size() - 1, getTokensPerBlock(), false);
    auto blockKeys = buildBlockKeys(blockedUniqueTokens, llmRequest);
    storeBlocks(std::move(blockKeys), cacheBlockIds[beamIdx]);
}

void BlockManager::createBlockScalePools(SizeType32 quantBlockSize)
{
    auto num_pools = mPools.size();
    for (size_t i = 0; i < num_pools; ++i)
    {
        auto& kv_pool = mPools[i];
        TLLM_CHECK_WITH_INFO(kv_pool.blockSize % quantBlockSize == 0,
            "Cannot use FP4 quantization since kv_pool.blockSize is not divisible by FP4 quantBlockSize.");

        mPools.emplace_back(kv_pool.numLayers, kv_pool.numKvHeads, kv_pool.sizePerHead, kv_pool.tokensPerBlock,
            quantBlockSize,
            /*primaryPool=*/nullptr,
            /*secondaryPool=*/nullptr,
            /*containsBlockScales=*/true);
    }
}

// ===== hstu modification start =====
void BlockManager::allocatePools(nvinfer1::DataType dtype, bool useUvm, SizeType32 numReservedBlocks)
// ===== hstu modification end =====
{
    // TODO: make the block size configurable. Should we have an additional argument
    // to specify FP4 related parameters (scale dtypes, etc)? This can also be passed
    // in the constructor.
    constexpr SizeType32 kQuantBlockSizeNVFP4 = 16;
    constexpr nvinfer1::DataType kScaleDtypeNVFP4 = nvinfer1::DataType::kFP8;

#ifdef ENABLE_FP4
    if (dtype == nvinfer1::DataType::kFP4)
    {
        // It would be slightly better to move construction of these objects to the BlockManager
        // constructor for consistency. We would have to do a bit of refactoring to pass
        // the dtype in earlier.
        createBlockScalePools(kQuantBlockSizeNVFP4);
    }
#endif

    // Allocate a memory pool backing the blocks for each numKvHeads
    // TODO(oargov): allocate pools in a single buffer and split it, to avoid fragmentation
    for (auto& pool : mPools)
    {
        auto blockSize = pool.blockSize;
        auto poolDtype = pool.containsBlockScales ? kScaleDtypeNVFP4 : dtype;

#ifdef ENABLE_FP4
        auto const poolIsFP4 = poolDtype == nvinfer1::DataType::kFP4;
#else
        auto const poolIsFP4 = false;
#endif

        if (poolIsFP4)
        {
            TLLM_CHECK_WITH_INFO(blockSize % 2 == 0, "Block size must be divisible by 2 for FP4 KV cache.");
            // Divide by 2. We can't create FP4 buffers directly, so we'll have to create a uint8 buffer with
            // half the expected number of elements.
            blockSize /= 2;
            poolDtype = nvinfer1::DataType::kINT8;
        }

        // ===== hstu modification start =====
        nvinfer1::Dims const cacheShape = ITensor::makeShape({mNumPrimaryBlocks + numReservedBlocks, pool.numLayers, mKVFactor, blockSize});

        TLLM_LOG_DEBUG("[BlockManager] Allocating primary pool with %d blocks (%d blocks reserved) for %d layers with %d kv heads",
            mNumPrimaryBlocks + numReservedBlocks, numReservedBlocks, pool.numLayers, pool.numKvHeads);
        // ===== hstu modification end =====

        if (useUvm)
            pool.primaryPtr = BufferManager::managed(cacheShape, poolDtype);
        else
            pool.primaryPtr = mBufferManager.gpuSync(cacheShape, poolDtype);

        if (mNumSecondaryBlocks > 0)
        {
            nvinfer1::Dims const cacheShapeOffload
                = ITensor::makeShape({mNumSecondaryBlocks, pool.numLayers, mKVFactor, blockSize});
            TLLM_LOG_DEBUG("[BlockManager] Allocating secondary pool with %d blocks for %d layers with %d kv heads",
                mNumSecondaryBlocks, pool.numLayers, pool.numKvHeads);
            pool.secondaryPtr = BufferManager::pinned(cacheShapeOffload, poolDtype);
        }
    }
}

void BlockManager::releasePools()
{
    for (auto& pool : mPools)
    {
        if (pool.primaryPtr)
        {
            pool.primaryPtr->release();
        }
        if (pool.secondaryPtr)
        {
            pool.secondaryPtr->release();
        }
    }
    mBufferManager.getStream().synchronize();
    mBufferManager.memoryPoolTrimTo(0);
}

void BlockManager::startScheduling()
{
    mSchedulingNumFreeBlocks = mEvictionPolicy->getNumFreeBlocks(kPrimaryLevel);
    for (auto& [requestId, slotAllocatedBlocks] : mAllocatedBlocksPerSeq)
    {
        for (auto& allocatedBlock : slotAllocatedBlocks)
        {
            allocatedBlock->startScheduling();
        }
    }
}

void BlockManager::claimLeafBlock(BlockPtr block, std::optional<executor::RetentionPriority> priority,
    std::optional<std::chrono::milliseconds> durationMs)
{
    // The eviction policy needs blocks to still be linked to their old parents when they're reclaimed.
    // This is so it can check if the parent should be queued for eviction.
    mEvictionPolicy->claimBlock(block, priority, durationMs);
    block->freeLeafBlock();
}

BlockPtr BlockManager::getFreeBlock(
    executor::RetentionPriority priority, std::optional<std::chrono::milliseconds> durationMs)
{
    // eviction policy get free primary block
    auto [block, canOffload] = mEvictionPolicy->getFreeBlock(kPrimaryLevel);
    if (block->getUniqueTokens().empty())
    {
        ++mAllocNewBlocks;
    }
    ++mAllocTotalBlocks;
    if (!block->getUniqueTokens().empty() && canOffload && mEvictionPolicy->getNumFreeBlocks(kSecondaryLevel) > 0)
    {
        // If we're swapping a block to secondary memory, maintain the prior priority values.
        mEvictionPolicy->claimBlock(block);
        // Offload block in primary memory before repurposing
        auto offloadBlock = std::get<0>(mEvictionPolicy->getFreeBlock(kSecondaryLevel));
        mTransferManager->offload(block, offloadBlock, mPools);
        // swap linear block offsets (i.e. make block the offload block)
        block->swapMemoryPoolBlockOffset(offloadBlock);

        if (mEventManager && blockInRadixTree(block))
        {
            mEventManager->enqueueUpdatedEvent(
                tle::KVCacheUpdatedData(block->getHash()).cacheLevelUpdated(kPrimaryLevel, kSecondaryLevel));
        }
        mEvictionPolicy->releaseBlock(block); // append offload block to mFreeSecondaryBlocks queue
        block = offloadBlock;
    }

    if (mEventManager && blockInRadixTree(block))
    {
        mEventManager->enqueueRemovedEvent(block);
    }

    claimLeafBlock(block, priority, durationMs);

    return block;
}

// ===== hstu modification start =====
void BlockManager::evictBlocks(GenerationRequest& sequence, SizeType32 numBlocksToEvict)
{
    auto const requestId = sequence.getRequestId();
    // TODO: add checks
    auto& allocatedBlocks = mAllocatedBlocksPerSeq.at(requestId);
    for (SizeType32 idx = 0; idx < numBlocksToEvict; ++idx) {
        auto& block = allocatedBlocks[idx];
        block->decRefCount();
        mEvictionPolicy->releaseBlock(block);
        removeBlockFromHashMap(block);
    }
    allocatedBlocks.erase(allocatedBlocks.begin(), allocatedBlocks.begin() + numBlocksToEvict);
    sequence.evictBlocks(numBlocksToEvict, mTokensPerBlock);
}

void BlockManager::evictSequence(GenerationRequest& sequence)
{
    auto const requestId = sequence.getRequestId();
    markSequenceEvicted(requestId);
    releaseBlocks(sequence, {});
}

void BlockManager::markSequenceEvicted(LlmRequest::RequestIdType requestId)
{
    auto const tableIt = mSeqLRUTable.find(requestId);
    if (mSeqLRUTable.end() != tableIt) {
        mSeqLRUList.erase(tableIt->second);
        mSeqLRUTable.erase(tableIt);
    }
}

void BlockManager::markSequenceRetained(LlmRequest::RequestIdType requestId)
{
    auto const tableIt = mSeqLRUTable.find(requestId);
    if (mSeqLRUTable.end() != tableIt) {
        mSeqLRUList.erase(tableIt->second);
    }
    mSeqLRUList.push_front(requestId);
    mSeqLRUTable[requestId] = mSeqLRUList.begin();
}
// ===== hstu modification end =====

tk::KVCacheIndex BlockManager::getKOrVBlockIndex(
    KVCacheBlock::IdType blockId, SizeType32 fieldIdx, SizeType32 poolIdx) const
{
    TLLM_CHECK_WITH_INFO(poolIdx < getNumPools(), "Pool index %d is out of bounds", poolIdx);
    auto const& block = mAllBlocksById[blockId];
    auto const& pool = mPools.at(poolIdx);
    if (mCacheType == CacheType::kSELFKONLY)
    {
        fieldIdx = 0;
    }
    auto constexpr layerIdx = 0;
    return tk::KVCacheIndex{
        common::flat_index3(block->getMemoryPoolBlockIndex(), layerIdx, fieldIdx, pool.numLayers, mKVFactor)};
}

void KVCacheManager::setOffsets(tk::KVCacheIndex* offsetsPtr, nvinfer1::Dims const& offsetsShape, SizeType32 beamIdx,
    SizeType32 blockIdx, KVCacheBlock::IdType blockId) const
{
    auto constexpr kIdx = 0;
    auto constexpr vIdx = 1;

    auto const numPools = mBlockManager.getNumPools();

    for (SizeType32 poolIdx = 0; poolIdx < numPools; poolIdx++)
    {
        for (auto xIdx : {kIdx, vIdx})
        {
            auto const offsetIndex = tensorrt_llm::common::flat_index(offsetsShape.d, poolIdx, beamIdx, xIdx, blockIdx);
            offsetsPtr[offsetIndex] = mBlockManager.getKOrVBlockIndex(blockId, xIdx, poolIdx);
        }
    }
}

void BlockManager::addBlockToHashMap(BlockPtr block)
{
    if (!mEnableHashKey)
    {
        return;
    }
    auto range = mContextBlocksByHash.equal_range(block->getHash());
    for (auto it = range.first; it != range.second; ++it)
    {
        if (it->second == block)
        {
            // TODO: change to assert when reused block is added only once
            TLLM_LOG_TRACE(
                "Block %d by %zx exists", block->getBlockId(), block->getHash(), mContextBlocksByHash.size());
            return;
        }
    }
    TLLM_LOG_TRACE(
        "Add block %d by %zx, block n = %zu", block->getBlockId(), block->getHash(), mContextBlocksByHash.size());
    mContextBlocksByHash.emplace(block->getHash(), std::move(block));
}

void BlockManager::removeBlockFromHashMap(BlockPtr block)
{
    if (mContextBlocksByHash.empty() || block->getBlockKey().uniqueTokens.empty())
    {
        // Hash key not enabled / Empty block
        return;
    }
    auto range = mContextBlocksByHash.equal_range(block->getHash());
    TLLM_LOG_TRACE(
        "Remove block %d by %zx, block n = %zu", block->getBlockId(), block->getHash(), mContextBlocksByHash.size());
    for (auto it = range.first; it != range.second; ++it)
    {
        if (it->second == block)
        {
            mContextBlocksByHash.erase(it);
            return;
        }
    }
    // TODO: should be unreachable
    TLLM_LOG_DEBUG("Trying to remove block %d by %zx that is not in hash map", block->getBlockId(), block->getHash());
}

void BlockManager::onboardBlock(BlockPtr const& offloadBlock)
{
    if (mOnboardBlocks && !offloadBlock->isPrimary())
    {
        auto block = getFreeBlock();
        mTransferManager->onboard(offloadBlock, block, mPools);
        // swap linear block offsets (i.e. make block the offload block and vice versa)
        offloadBlock->swapMemoryPoolBlockOffset(block);

        if (mEventManager)
        {
            mEventManager->enqueueUpdatedEvent(
                tle::KVCacheUpdatedData(offloadBlock->getHash()).cacheLevelUpdated(kSecondaryLevel, kPrimaryLevel));
        }
        mEvictionPolicy->releaseBlock(block); // append block to offload queue
                                              // offloadBlock is now in primary memory pool
    }
}

void BlockManager::offloadBlock(BlockPtr const& block)
{
    if (mOnboardBlocks && block->isPrimary())
    {
        // Offload block in primary memory before repurposing
        auto offloadBlock = std::get<0>(mEvictionPolicy->getFreeBlock(kSecondaryLevel));
        // If we're swapping a block to secondary memory, maintain the prior priority values.
        mEvictionPolicy->claimBlock(offloadBlock);
        mTransferManager->offload(block, offloadBlock, mPools);
        // swap linear block offsets (i.e. make block the offload block)
        block->swapMemoryPoolBlockOffset(offloadBlock);

        if (mEventManager && blockInRadixTree(block))
        {
            mEventManager->enqueueUpdatedEvent(
                tle::KVCacheUpdatedData(block->getHash()).cacheLevelUpdated(kPrimaryLevel, kSecondaryLevel));
        }
        mEvictionPolicy->releaseBlock(offloadBlock); // append offloadBlock to mFreePrimaryBlocks queue
                                                     // block is now in secondary memory
    }
}

std::optional<BlockKey> BlockManager::findNewContextBlock(
    VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const
{
    auto blockedUniqueTokens
        = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, uniqueTokens.size(), mTokensPerBlock, false);
    auto blockKeys = buildBlockKeys(blockedUniqueTokens, llmRequest);
    BlockKey ret;
    ret.loraTaskId = llmRequest.getLoraTaskId();
    auto searchRoot = mCachedBlocksRoot;
    for (auto const& blockKey : blockKeys)
    {
        ret.uniqueTokens.insert(ret.uniqueTokens.end(), blockKey.uniqueTokens.begin(), blockKey.uniqueTokens.end());
        auto [partialMatch, numMatched, matchingBlock] = searchRoot != nullptr
            ? searchRoot->findMatchingBlock(blockKey, false, false)
            : std::make_tuple(false, 0, nullptr);
        if (matchingBlock == nullptr)
        {
            return ret;
        }
        searchRoot = std::move(matchingBlock);
    }
    return std::nullopt;
}

bool BlockManager::blockInRadixTree(BlockPtr const& block)
{
    return !block->getUniqueTokens().empty() && block->getPrevBlock() != nullptr;
}

SizeType32 BlockManager::loadOrAllocateBlocks(std::vector<BlockKey> const& blockKeys, SizeType32 numContextBlocks,
    GenerationRequest& sequence, std::vector<executor::RetentionPriorityAndDuration> const& perBlockRetentions)
{
    SizeType32 numMatchedTokens{0};
    auto searchRoot = mCachedBlocksRoot;

    // The last block cannot be shared between beams because it will be written to.
    // Make sure a unique block is allocated per beam.
    auto const beamWidth = sequence.getBeamWidth();
    SizeType32 numSharedContextBlocks = beamWidth > 1 ? numContextBlocks - 1 : numContextBlocks;

    auto blockItr = blockKeys.begin();
    for (int bi = 0; bi < numSharedContextBlocks; ++bi)
    {
        auto [partialMatch, numMatched, matchingBlock] = searchRoot != nullptr && blockItr != blockKeys.end()
            ? searchRoot->findMatchingBlock(*blockItr, mEnablePartialReuse, mCopyOnPartialReuse)
            : std::make_tuple(false, 0, nullptr);
        if (matchingBlock != nullptr)
        {
            KVCacheBlock::IdType matchingBlockId = matchingBlock->getBlockId();

            numMatchedTokens += numMatched > 0 ? numMatched : blockItr->uniqueTokens.size();
            if (perBlockRetentions[bi].retentionPriority.has_value()
                && matchingBlock->getPriority() != perBlockRetentions[bi].retentionPriority && mEventManager)
            {
                mEventManager->enqueueUpdatedEvent(
                    tle::KVCacheUpdatedData(matchingBlock->getHash())
                        .priorityUpdated(matchingBlock->getPriority(), *perBlockRetentions[bi].retentionPriority));
            }
            if (partialMatch)
            {
                if (matchingBlock->hasRefs() || !matchingBlock->isLeaf())
                {
                    // Somebody else is using block or it is not a leaf, copy reusable tokens
                    auto newBlock = getFreeBlock(matchingBlock->getPriority(), matchingBlock->getDurationMs());
                    mTransferManager->onboard(matchingBlock, newBlock, mPools, numMatched);
                    // TODO: (optional) Send out event
                    matchingBlock = newBlock;
                    TLLM_LOG_DEBUG(
                        "BlockManager::loadOrAllocateBlocks - Copied partially filled block %d", matchingBlockId);
                }
                else
                {
                    // Leaf block that nobody is using. Make block private and reuse
                    claimLeafBlock(
                        matchingBlock, perBlockRetentions[bi].retentionPriority, perBlockRetentions[bi].durationMs);
                    TLLM_LOG_DEBUG(
                        "BlockManager::loadOrAllocateBlocks - Reused partially filled block %d", matchingBlockId);
                    addBlockToHashMap(matchingBlock);
                }
                searchRoot = nullptr; // no matching needed for following blocks
            }
            else
            {
                // Recover block and reuse
                mEvictionPolicy->claimBlock(
                    matchingBlock, perBlockRetentions[bi].retentionPriority, perBlockRetentions[bi].durationMs);
                TLLM_LOG_DEBUG("BlockManager::loadOrAllocateBlocks - Matched full block %d", matchingBlockId);
                addBlockToHashMap(matchingBlock);
                searchRoot = matchingBlock;
            }
            onboardBlock(matchingBlock);
            addBlockToAllBeams(matchingBlock, sequence);
            // TODO: only add once for reused blocks
            ++mReusedBlocks;
            if (!reusedBlockIds.count(matchingBlockId))
            {
                reusedBlockIds.insert(matchingBlockId);
                ++mReusedUniqueBlocks;
            }
            ++blockItr;
        }
        else
        {
            // If we haven't set a priority, set it to the default priority level (low)
            auto freeBlock = getFreeBlock(perBlockRetentions[bi].retentionPriority.value_or(
                                              executor::KvCacheRetentionConfig::kDefaultRetentionPriority),
                perBlockRetentions[bi].durationMs);
            addBlockToAllBeams(freeBlock, sequence);
            TLLM_LOG_DEBUG("BlockManager::loadOrAllocateBlocks - No match, allocated new block %d for sequence %lu",
                freeBlock->getBlockId(), sequence.getRequestId());
            searchRoot = nullptr; // no matching needed for following blocks
            if (blockItr != blockKeys.end())
            {
                freeBlock->setBlockKey(
                    *blockItr, blockItr->uniqueTokens.size() == static_cast<size_t>(mTokensPerBlock));
                ++blockItr;
            }
            freeBlock->setHash();
            addBlockToHashMap(freeBlock);
            ++mMissedBlocks;
        }
    }

    // Allocate new blocks that cannot be shared by multiple beams.
    for (int bi = numSharedContextBlocks; bi < numContextBlocks; ++bi)
    {
        // TODO: Still look for match. Clone matching block or allocate fresh ones.
        // This work is described in JIRA task https://jirasw.nvidia.com/browse/TRTLLM-2069.
        for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
        {
            // If we haven't set a priority, set it to the default priority level (low)
            auto freeBlock = getFreeBlock(perBlockRetentions[bi].retentionPriority.value_or(
                                              executor::KvCacheRetentionConfig::kDefaultRetentionPriority),
                perBlockRetentions[bi].durationMs);
            addBlockToBeam(freeBlock, sequence, beamIdx);
            if (blockItr != blockKeys.end())
            {
                freeBlock->setBlockKey(
                    *blockItr, blockItr->uniqueTokens.size() == static_cast<size_t>(mTokensPerBlock));
                ++blockItr;
            }
            freeBlock->setHash();
            addBlockToHashMap(freeBlock);
            TLLM_LOG_DEBUG("BlockManager::loadOrAllocateBlocks - Beam %d. Allocated non-shared block %d for bi %d",
                beamIdx, freeBlock->getBlockId(), bi);
        }
        ++mMissedBlocks;
        if (blockItr != blockKeys.end())
        {
            ++blockItr;
        }
    }

    return numMatchedTokens;
}

void BlockManager::refreshBlocks()
{
    mEvictionPolicy->refresh();
    mTransferManager->syncTransfers();
}

void BlockManager::addSequence(
    GenerationRequest& sequence, SizeType32 inputLength, SizeType32 numContextBlocks, LlmRequest& llmRequest)
{
    auto const requestId = sequence.getRequestId();
    auto const [seqIt, emplaceDone] = mAllocatedBlocksPerSeq.emplace(requestId, std::vector<BlockPtr>{});
    TLLM_CHECK(emplaceDone);

    auto constexpr beamIdx = 0;
    auto const& uniqueTokens = (mCacheType == CacheType::kSELF || mCacheType == CacheType::kSELFKONLY)
        ? llmRequest.getUniqueTokens(beamIdx)
        : *(llmRequest.getEncoderUniqueTokens().value());

    // Ignore last token because it can't be recovered
    auto blockedUniqueTokens = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, inputLength - 1, mTokensPerBlock, true);
    // Add empty block if last token is separated
    if (inputLength % mTokensPerBlock == 1)
    {
        blockedUniqueTokens.emplace_back();
    }

    auto blockKeys = buildBlockKeys(blockedUniqueTokens, llmRequest);

    auto config = llmRequest.getKvCacheRetentionConfig();

    auto perBlockRetentions = config.value_or(executor::KvCacheRetentionConfig())
                                  .getPerBlockRetentionPriorityDuration(getTokensPerBlock(), inputLength);

    TLLM_CHECK(perBlockRetentions.size() == (size_t) numContextBlocks);

    auto const prepopulatedPromptLen = loadOrAllocateBlocks(blockKeys, numContextBlocks, sequence, perBlockRetentions);
    mReusedTokens += static_cast<double>(prepopulatedPromptLen);
    mTotalInputTokens += static_cast<double>(uniqueTokens.size());
    llmRequest.setPrepopulatedPromptLen(prepopulatedPromptLen, getTokensPerBlock());
    TLLM_LOG_DEBUG("addSequence: Request %lu, inputLength %d, prepopulatedPromptLen %d", llmRequest.mRequestId,
        inputLength, prepopulatedPromptLen);
}

void BlockManager::addSequence(GenerationRequest& sequence, SizeType32 numBlocks, SizeType32 unsharedBlockIdx)
{
    auto const requestId = sequence.getRequestId();
    auto const [seqIt, emplaceDone] = mAllocatedBlocksPerSeq.emplace(requestId, std::vector<BlockPtr>{});
    TLLM_CHECK(emplaceDone);

    // Allocate blocks
    for (SizeType32 bi = 0; bi < numBlocks; ++bi)
    {
        bool shareAmongBeams = bi != unsharedBlockIdx;
        allocateBlock(sequence, shareAmongBeams);
    }
}

void BlockManager::addBlockToBeam(BlockPtr& block, GenerationRequest& sequence, SizeType32 beamIdx)
{
    auto const requestId = sequence.getRequestId();
    block->incRefCount();
    if (sequence.getCacheBlockIds().at(beamIdx).size() == 0)
    {
        block->setPrevBlockInSeq(nullptr);
    }
    else
    {
        block->setPrevBlockInSeq(mAllBlocksById.at(sequence.getCacheBlockIds()[beamIdx].back()));
    }
    sequence.addCacheBlock(beamIdx, block->getBlockId());
    mAllocatedBlocksPerSeq.at(requestId).push_back(block);
}

void BlockManager::addBlockToAllBeams(BlockPtr& block, GenerationRequest& sequence)
{
    auto const beamWidth = sequence.getBeamWidth();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        addBlockToBeam(block, sequence, beamIdx);
    }
}

void BlockManager::allocateBlock(GenerationRequest& sequence, bool shareAmongBeams)
{
    auto const beamWidth = sequence.getBeamWidth();
    auto const requiredBlocks = shareAmongBeams ? 1 : beamWidth;

    TLLM_CHECK_WITH_INFO(hasFreeBlocks(requiredBlocks), "Can't allocate new blocks. No free blocks left.");

    if (shareAmongBeams)
    {
        // add same block to all beams
        auto block = getFreeBlock(sequence.getDecodeRetentionPriority(), sequence.getDecodeDurationMs());
        for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
        {
            addBlockToBeam(block, sequence, beamIdx);
        }
    }
    else
    {
        // add different block to each beam
        for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
        {
            auto block = getFreeBlock(sequence.getDecodeRetentionPriority(), sequence.getDecodeDurationMs());
            addBlockToBeam(block, sequence, beamIdx);
        }
    }
}

void BlockManager::storeBlocks(std::vector<BlockKey> blockKeys, std::vector<KVCacheBlock::IdType> const& blockIds)
{
    TLLM_LOG_DEBUG("BlockManager::storeBlocks - %zu blockKeys, %zu blockIds", blockKeys.size(), blockIds.size());

    auto searchRoot = mCachedBlocksRoot;
    bool needMatch = true;

    auto numBlocks = blockKeys.size();
    std::vector<BlockPtr> storedBlocks;
    for (std::size_t blockCnt = 0; blockCnt < numBlocks; ++blockCnt)
    {
        auto const bid = blockIds[blockCnt];
        TLLM_LOG_DEBUG("BlockManager::storeBlocks - Searching match for block %d", bid);
        auto& block = mAllBlocksById[bid];
        auto const& blockKey = blockKeys[blockCnt];

        auto [partialMatch, numMatched, matchedBlock]
            = needMatch ? searchRoot->findMatchingBlock(blockKey, false, false) : std::make_tuple(false, 0, nullptr);
        if (matchedBlock != nullptr)
        {
            // Found match
            TLLM_LOG_DEBUG("BlockManager::storeBlocks - Found matching block %d, traverse", matchedBlock->getBlockId());
            searchRoot = matchedBlock;
            // TODO possible optimization: if bid != matchedBlock->getBlockId(),
            // block can be freed and inserted at mFreePrimaryBlocks.begin()
        }
        else
        {
            // No match
            TLLM_LOG_DEBUG(
                "BlockManager::storeBlocks - No match, inserting block %d into search structure", block->getBlockId());
            needMatch = false; // no matching needed for following blocks
            block->setBlockKey(blockKey, static_cast<SizeType32>(blockKey.uniqueTokens.size()) == mTokensPerBlock);
            block->setPrevBlock(searchRoot);
            block->setPrevBlockInSeq(searchRoot);
            searchRoot->addNextBlock(blockKey, block);

            // Sanity check. The list of stored blocks should be connected.
            TLLM_CHECK(storedBlocks.empty() || block->getPrevBlock() == storedBlocks.back());

            storedBlocks.push_back(block);
            TLLM_CHECK(block->getPrevBlockInSeq() == nullptr
                || block->getPrevBlockInSeq()->getHash() == searchRoot->getHash());
            auto oldHash = block->getHash();
            auto newHash = BlockKeyHasher()(blockKey, searchRoot->getHash());
            if (oldHash != newHash)
            {
                TLLM_LOG_DEBUG("#%d block hash %zx -> %zx", block->getBlockId(), oldHash, newHash);
                removeBlockFromHashMap(block);
                block->setHash(newHash);
                addBlockToHashMap(block);
            }
            searchRoot = block;
        }
    }
    if (mEventManager)
    {
        mEventManager->enqueueStoredEvent(storedBlocks);
    }
}

void BlockManager::replaceSharedBlock(GenerationRequest& sequence, SizeType32 blockIdx)
{
    auto const requestId = sequence.getRequestId();
    auto const beamWidth = sequence.getBeamWidth();
    auto& allocatedBlocks = mAllocatedBlocksPerSeq.at(requestId);

    if (!allocatedBlocks.at((blockIdx + 1) * beamWidth - 1)->isShared())
    {
        return;
    }
    BlockKey blockKey = allocatedBlocks.at(blockIdx * beamWidth)->getBlockKey();
    bool isFull = allocatedBlocks.at(blockIdx * beamWidth)->isFull();

    // Free shared block
    for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto block = allocatedBlocks.at(blockIdx * beamWidth + beamIdx);
        block->decRefCount();
        if (!block->hasRefs())
        {
            mEvictionPolicy->releaseBlock(block);
            removeBlockFromHashMap(block);
        }
    }

    // Allocate new blocks
    TLLM_CHECK_WITH_INFO(hasFreeBlocks(beamWidth), "Can't allocate new blocks. No free blocks left.");
    for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto block = getFreeBlock();
        block->incRefCount();
        if (sequence.getCacheBlockIds().at(beamIdx).size() == 0)
        {
            block->setPrevBlockInSeq(nullptr);
        }
        else
        {
            block->setPrevBlockInSeq(mAllBlocksById.at(sequence.getCacheBlockIds()[beamIdx].back()));
        }
        block->setBlockKey(blockKey, isFull);
        block->setHash();
        sequence.changeCacheBlock(beamIdx, blockIdx, block->getBlockId());
        allocatedBlocks.at(blockIdx * beamWidth + beamIdx) = block;
    }
}

std::vector<KVCacheBlock::IdType> BlockManager::getNewlyAllocatedBlockIds(GenerationRequest const& sequence) const
{
    std::vector<KVCacheBlock::IdType> allocatedBlockIds;
    for (auto const& beamBlockIds : sequence.getCacheBlockIds())
    {
        for (auto const& blockId : beamBlockIds)
        {
            auto const& block = mAllBlocksById.at(blockId);
            if (!blockInRadixTree(block))
            {
                allocatedBlockIds.push_back(blockId);
            }
        }
    }
    return allocatedBlockIds;
}

void BlockManager::releaseLastBlock(GenerationRequest& sequence)
{
    auto const requestId = sequence.getRequestId();
    auto& allocatedBlocks = mAllocatedBlocksPerSeq.at(requestId);
    auto it = allocatedBlocks.rbegin();
    auto& block = *it;
    // Decrease ref count
    block->decRefCount();
    // If ref count is zero, move block to free blocks
    if (!block->hasRefs())
    {
        mEvictionPolicy->releaseBlock(block, true);
        removeBlockFromHashMap(block);
    }
    // Remove block from allocated blocks
    allocatedBlocks.pop_back();
    // Remove stored block ids in sequence
    sequence.removeLastBlock();
}

[[nodiscard]] SizeType32 BlockManager::getNumFreeBlocks() const noexcept
{
    return mEvictionPolicy->getNumFreeBlocks(kPrimaryLevel);
}

std::deque<tle::KVCacheEvent> BlockManager::getLatestEvents(std::optional<std::chrono::milliseconds> timeout) const
{
    return mEventManager ? mEventManager->getEvents(timeout) : std::deque<tle::KVCacheEvent>{};
}

void BlockManager::releaseBlocks(GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest)
{
    auto const requestId = sequence.getRequestId();

    // TODO: refactor this method in two: store blocks for reuse and just 'release blocks'. Only the caller
    // can know which blocks to store for reuse and which to just release.
    // When releasing the blocks for a sequence, we store those blocks for potential reuse only if:
    // - Block reuse is enabled.
    // - A request was provided to this function call to identify which tokens these blocks cover
    // - Beam search is NOT enabled <=> beam width == 1
    // - The sequence was not marked for use with cyclic kv-cache when it was added (when its context is too long to fit
    // the max attention window).
    // - The sequence did not switch to cyclic kv-cache during generation phase.
    bool const storeBlocksForReuse = sequence.getBeamWidth() == 1 && llmRequest.has_value() && !sequence.isCyclic();
    if (storeBlocksForReuse)
    {
        auto constexpr beamIdx = 0;
        auto const& uniqueTokens = llmRequest->getUniqueTokens(beamIdx);
        auto const& cacheBlockIds = sequence.getCacheBlockIds();

        // TODO: get the caller to mark tokens as filled / not filled, so that the kv-cache manager doesn't
        // have to guess. Only (length - 1) tokens of the sequence have their kv-state recorded in kv-cache. We assume
        // the last token's state is not filled yet.
        auto const usableSize = static_cast<runtime::SizeType32>(uniqueTokens.size()) - 1;
        auto blockedUniqueTokens = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, usableSize, mTokensPerBlock, true);
        auto blockKeys = buildBlockKeys(blockedUniqueTokens, *llmRequest);
        storeBlocks(std::move(blockKeys), cacheBlockIds[beamIdx]);
    }

    auto node = mAllocatedBlocksPerSeq.extract(requestId);
    auto& allocatedBlocks = node.mapped();
    for (auto it = allocatedBlocks.rbegin(); it != allocatedBlocks.rend(); ++it)
    {
        auto& block = *it;
        // Decrease ref count
        block->decRefCount();
        // If ref count is zero, move block to free blocks
        if (!block->hasRefs())
        {
            mEvictionPolicy->releaseBlock(block);
            removeBlockFromHashMap(block);
        }
    }
    // Remove stored block ids in sequence
    sequence.clearCacheBlocks();
}

void BlockManager::schedulingReleaseBlocks(RequestIdType requestId)
{
    for (auto& block : mAllocatedBlocksPerSeq.at(requestId))
    {
        // Decrease ref count
        block->decSchedulingRefCount();
        // If ref count is zero, move block to free blocks
        if (!block->hasSchedulingRefs())
        {
            mSchedulingNumFreeBlocks++;
        }
    }
}

KVCacheManager::KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool,
    SizeType32 maxNumSequences, SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
    SizeType32 temporaryAttentionWindow, SizeType32 sinkTokenLength, int64_t stream,
    std::optional<runtime::SizeType32> maxSequenceLength, bool enableBlockReuse, bool onboardBlocks,
    CacheType cacheType, bool enablePartialReuse, bool copyOnPartialReuse,
    // ===== hstu modification start =====
    SizeType32 reservedBlocksInPrimaryPool)
    // ===== hstu modification end =====
    : KVCacheManager(std::vector<SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, maxBeamWidth, maxAttentionWindowVec, temporaryAttentionWindow,
        sinkTokenLength, std::make_shared<runtime::CudaStream>(reinterpret_cast<cudaStream_t>(stream)),
        maxSequenceLength, enableBlockReuse, onboardBlocks, cacheType, std::nullopt, nullptr, false, enablePartialReuse,
        copyOnPartialReuse, reservedBlocksInPrimaryPool)
{
}

KVCacheManager::KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool,
    SizeType32 maxNumSequences, SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
    SizeType32 temporaryAttentionWindow, SizeType32 sinkTokenLength, int64_t stream,
    std::optional<runtime::SizeType32> maxSequenceLength, bool enableBlockReuse, bool onboardBlocks,
    CacheType cacheType, std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enablePartialReuse, bool copyOnPartialReuse,
    // ===== hstu modification start =====
    SizeType32 reservedBlocksInPrimaryPool)
    // ===== hstu modification end =====
    : KVCacheManager(numKvHeadsPerLayer, sizePerHead, tokensPerBlock, blocksInPrimaryPool, blocksInSecondaryPool,
        maxNumSequences, maxBeamWidth, maxAttentionWindowVec, temporaryAttentionWindow, sinkTokenLength,
        std::make_shared<runtime::CudaStream>(reinterpret_cast<cudaStream_t>(stream)), maxSequenceLength,
        enableBlockReuse, onboardBlocks, cacheType, secondaryOffloadMinPriority, std::move(eventManager), false,
        enablePartialReuse, copyOnPartialReuse, reservedBlocksInPrimaryPool)
{
}

KVCacheManager::KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool,
    SizeType32 maxNumSequences, SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
    SizeType32 temporaryAttentionWindow, SizeType32 sinkTokenLength, CudaStreamPtr stream,
    std::optional<runtime::SizeType32> maxSequenceLength, bool enableBlockReuse, bool onboardBlocks,
    CacheType cacheType, std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enableHashKey, bool enablePartialReuse,
    bool copyOnPartialReuse,
    // ===== hstu modification start =====
    SizeType32 reservedBlocksInPrimaryPool)
    : mMaxNumSequences(maxNumSequences)
    // ===== hstu modification end =====
    , mMaxBeamWidth(maxBeamWidth)
    , mMaxAttentionWindow(*std::max_element(maxAttentionWindowVec.begin(), maxAttentionWindowVec.end()))
    , mMinAttentionWindow(*std::min_element(maxAttentionWindowVec.begin(), maxAttentionWindowVec.end()))
    , mTemporaryAttentionWindow(temporaryAttentionWindow)
    , mTokensPerBlock(tokensPerBlock)
    , mSinkBubbleLength(BaseKVCacheManager::getSinkBubbleLength(sinkTokenLength, tokensPerBlock))
    , mSinkBlockTokenLength(mSinkBubbleLength + sinkTokenLength)
    , mBlockManager(numKvHeadsPerLayer, sizePerHead, tokensPerBlock, blocksInPrimaryPool, blocksInSecondaryPool,
          maxNumSequences, std::move(stream), onboardBlocks, cacheType, secondaryOffloadMinPriority,
          std::move(eventManager), enableHashKey, enablePartialReuse, copyOnPartialReuse)
    // disable block reuse for sink bubble since chopVectorIntoBlocks does not match KV cache blocks in this case
    , mEnableBlockReuse{mSinkBubbleLength > 0 ? false : enableBlockReuse}
    , mEnableHashKey{enableHashKey}
    // ===== hstu modification start =====
    , mReservedBlocksInPrimaryPool(reservedBlocksInPrimaryPool)
    // ===== hstu modification end =====
{
    // The sink tokens are stored in blocks separate from other tokens.
    // If the last block of sink tokens is only partially filled,
    // we fill that block with a "bubble" to reach the number of tokens per block.

    TLLM_CHECK(mSinkBlockTokenLength % tokensPerBlock == 0);

    mMaxTokenNum = mMaxAttentionWindow + mSinkBubbleLength;

    // If maxBeamWidth > 1, use one more block for each sequence in the paged kv cache to avoid dropping the needed
    // tokens, when enabling cyclic kv cache.
    mUseOneMoreBlock
        = maxSequenceLength.has_value() && maxSequenceLength.value() > mMaxAttentionWindow && maxBeamWidth > 1;
    TLLM_CHECK_WITH_INFO(!mUseOneMoreBlock || mTemporaryAttentionWindow == 0,
        "Can't support sliding window attention, paged context fmha, and beam search are used together.");
    if (mUseOneMoreBlock)
    {
        mMaxTokenNum += tokensPerBlock;
    }

    // Consider the mTemporaryAttentionWindow when allocating blocks.
    mMaxBlocksPerSeq = tc::ceilDiv(mMaxTokenNum + mTemporaryAttentionWindow, tokensPerBlock);

    TLLM_LOG_DEBUG("KV cache block reuse is %s", mEnableBlockReuse ? "enabled" : "disabled");
    TLLM_LOG_DEBUG("Max KV cache pages per sequence: %d", mMaxBlocksPerSeq);
    mSequences.reserve(maxNumSequences);
}

KVCacheManager::KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool,
    SizeType32 maxNumSequences, SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
    SizeType32 temporaryAttentionWindow, SizeType32 sinkTokenLength, CudaStreamPtr stream,
    std::optional<runtime::SizeType32> maxSequenceLength, bool enableBlockReuse, bool onboardBlocks,
    CacheType cacheType, std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enableHashKey, bool enablePartialReuse,
    bool copyOnPartialReuse,
    // ===== hstu modification start =====
    SizeType32 reservedBlocksInPrimaryPool)
    // ===== hstu modification end =====
    : KVCacheManager(std::vector<SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksInPrimaryPool,
        blocksInSecondaryPool, maxNumSequences, maxBeamWidth, maxAttentionWindowVec, temporaryAttentionWindow,
        sinkTokenLength, std::move(stream), maxSequenceLength, enableBlockReuse, onboardBlocks, cacheType,
        secondaryOffloadMinPriority, std::move(eventManager), enableHashKey, enablePartialReuse, copyOnPartialReuse,
        reservedBlocksInPrimaryPool)
{
}

void KVCacheManager::allocatePools(nvinfer1::DataType dtype, bool useUvm)
{
    // ===== hstu modification start =====
    mBlockManager.allocatePools(dtype, useUvm, mReservedBlocksInPrimaryPool);
    // ===== hstu modification end =====

    if (tc::Logger::getLogger()->getLevel() == tc::Logger::INFO)
    {
        uint64_t cacheSizeBytes = 0;
        auto const numPools = mBlockManager.getNumPools();
        for (SizeType32 poolIdx = 0; poolIdx < numPools; poolIdx++)
        {
            auto const cacheShape = mBlockManager.getPrimaryPool(poolIdx)->getShape();
            auto const cacheVolume = ITensor::volume(cacheShape);
#ifdef ENABLE_FP4
            auto const isFp4 = dtype == nvinfer1::DataType::kFP4;
#else
            auto const isFp4 = false;
#endif
            if (!isFp4)
            {
                cacheSizeBytes += cacheVolume * BufferDataType(dtype).getSize();
            }
            else
            {
                cacheSizeBytes += (cacheVolume * 4) / 8;
            }
        }

        TLLM_LOG_INFO("Number of tokens per block: %d.", mBlockManager.getTokensPerBlock());
        auto const maxNumTokens = mBlockManager.getNumPrimaryBlocks() * mBlockManager.getTokensPerBlock();
        TLLM_LOG_INFO("[MemUsageChange] Allocated %0.2f GiB for max tokens in paged KV cache (%d).",
            cacheSizeBytes / static_cast<double>(1 << 30), maxNumTokens);
    }

    auto const numPools = mBlockManager.getNumPools();
    auto const numKVPools = mBlockManager.getNumPools(/*include_block_scalar_pools=*/false);
    auto const numBlockScalePools = numPools - numKVPools;

    // Code in the attention kernels is cleaner if we can access the KV values and block scales separately.
    mBlockPoolPointers = BufferManager::cpu(ITensor::makeShape({numKVPools, 2}), TRTDataType<void*>::value);
    mBlockScalePoolPointers
        = BufferManager::cpu(ITensor::makeShape({numBlockScalePools, 2}), TRTDataType<void*>::value);

    auto poolPtrsRange = BufferRange<void*>(*mBlockPoolPointers);
    auto blockScalePtrsRange = BufferRange<void*>(*mBlockScalePoolPointers);
    SizeType32 kvPoolIdx = 0;
    SizeType32 blockScalePoolIdx = 0;

    for (SizeType32 poolIdx = 0; poolIdx < numPools; poolIdx++)
    {
        auto const isBlockScale = mBlockManager.containsBlockScales(poolIdx);
        auto& outIdx = isBlockScale ? blockScalePoolIdx : kvPoolIdx;
        auto& outRange = isBlockScale ? blockScalePtrsRange : poolPtrsRange;

        outRange[outIdx * 2] = mBlockManager.getPrimaryPool(poolIdx)->data();
        auto secondaryPool = mBlockManager.getSecondaryPool(poolIdx);
        outRange[outIdx * 2 + 1] = secondaryPool ? secondaryPool->data() : nullptr;
        ++outIdx;
    }

    auto const numLayers = mBlockManager.getNumLayers();
    mLayerToPoolMapping = BufferManager::cpu(ITensor::makeShape({numLayers, 2}), TRTDataType<SizeType32>::value);
    auto poolMappingRange = BufferRange<SizeType32>(*mLayerToPoolMapping);
    for (SizeType32 layerIdx = 0; layerIdx < numLayers; layerIdx++)
    {
        auto const indexOfPool = mBlockManager.getLayerPoolIdx(layerIdx);
        auto const layerIdxInCachePool = mBlockManager.getPoolLayerIdx(layerIdx);
        poolMappingRange[layerIdx * 2] = indexOfPool;
        poolMappingRange[layerIdx * 2 + 1] = layerIdxInCachePool;
    }
}

void KVCacheManager::releasePools()
{
    mBlockManager.releasePools();
}

void KVCacheManager::startScheduling()
{
    mBlockManager.startScheduling();
}

SizeType32 KVCacheManager::getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead) const
{
    SizeType32 numRequiredBlocks = 0;
    SizeType32 const numDraftTokens = req.getNumDraftTokens();
    SizeType32 const generatedTokens = req.getMaxNumGeneratedTokens();
    SizeType32 const maxTokensToAddToKVCache = req.mMaxNewTokens - generatedTokens + 1;
    SizeType32 const numTokensPerStep = std::min(numDraftTokens + 1, maxTokensToAddToKVCache);
    SizeType32 const numDraftTokensPerStep = std::min(numDraftTokens, maxTokensToAddToKVCache);
    if ((req.isContextInitState() && req.isFirstContextChunk()) || req.isDisaggGenerationInitState())
    {
        // Assumes shared among beam = True
        auto const promptCacheLen
            = std::min((isCrossKv() ? req.getEncoderOutputLen() : req.mPromptLen) + numDraftTokensPerStep,
                  mMaxAttentionWindow)
            + mSinkBubbleLength;
        auto const numSharedBlocks = promptCacheLen / getTokensPerBlock();
        auto const numUnSharedTokens = promptCacheLen % getTokensPerBlock();
        auto const numUnSharedBlocks
            = tc::ceilDiv(numUnSharedTokens, getTokensPerBlock()) * req.mSamplingConfig.beamWidth;
        numRequiredBlocks = numSharedBlocks + numUnSharedBlocks;
    }
    else if (req.isGenerationInProgressState())
    {
        if (isCrossKv())
        {
            return 0;
        }
        // Here we need to check if next token or the one after would require a new block
        // Because the active requests could be in flight, thus will only get their number
        // of generated tokens to be updated after scheduling
        auto const numPastTokens = req.mPromptLen + generatedTokens + mSinkBubbleLength - 1;
        auto const numNextTokens = numPastTokens + (twoStepsLookAhead ? 2 : 1) * numTokensPerStep;

        if (numNextTokens > mMaxTokenNum)
        {
            return 0;
        }

        auto const numPastBlocks = tc::ceilDiv(numPastTokens, getTokensPerBlock());
        auto const numNextBlocks = tc::ceilDiv(numNextTokens, getTokensPerBlock());
        numRequiredBlocks = (numNextBlocks - numPastBlocks) * req.mSamplingConfig.beamWidth;
    }
    return numRequiredBlocks;
}

SizeType32 KVCacheManager::getRemainingBlocksToCompletion(LlmRequest const& req) const
{
    if (isCrossKv())
    {
        if (req.isContextInitState() && req.getContextCurrentPosition() == 0)
        {
            return tc::ceilDiv(req.getEncoderOutputLen(), getTokensPerBlock());
        }

        return 0; // cross KV cache doesn't grow after the initial context phase
    }
    SizeType32 const numContextBlocks
        = (std::min(req.mPromptLen, mMaxAttentionWindow + mTemporaryAttentionWindow) + mSinkBubbleLength)
        / getTokensPerBlock();

    SizeType32 const numTotalBlocksPerBeam
        = tc::ceilDiv(std::min(req.mPromptLen + req.mMaxNewTokens, mMaxAttentionWindow + mTemporaryAttentionWindow)
                + mSinkBubbleLength,
            getTokensPerBlock());

    SizeType32 const numGenBlocksPerBeam = numTotalBlocksPerBeam - numContextBlocks;

    SizeType32 numAllocBlocksPerBeam = 0;
    {
        std::scoped_lock lck(mSequencesMtx);
        auto const seqIt = mSequences.find(req.mRequestId);
        if (seqIt != mSequences.end())
        {
            auto const& seq = seqIt->second;
            numAllocBlocksPerBeam = seq.getCacheBlockIds().at(0).size();
        }
    }

    if (numAllocBlocksPerBeam < numContextBlocks)
    {
        return numContextBlocks - numAllocBlocksPerBeam + numGenBlocksPerBeam * req.mSamplingConfig.beamWidth;
    }

    return (numTotalBlocksPerBeam - numAllocBlocksPerBeam) * req.mSamplingConfig.beamWidth;
}

void KVCacheManager::cacheBlockOffsets(GenerationRequest& sequence)
{
    auto const& cacheBlocks = sequence.getCacheBlockIds();
    auto& cacheBlocksTensor = sequence.getCacheBlockIndices();
    auto const beamWidth = sequence.getBeamWidth();

    auto* offsetsPtr = bufferCast<tk::KVCacheIndex>(cacheBlocksTensor);
    auto const& offsetsShape = cacheBlocksTensor.getShape();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto const& beamCacheBlock = cacheBlocks[beamIdx];
        for (SizeType32 blockIdx = 0; blockIdx < static_cast<SizeType32>(beamCacheBlock.size()); ++blockIdx)
        {
            auto const blockId = beamCacheBlock.at(blockIdx);
            setOffsets(offsetsPtr, offsetsShape, beamIdx, blockIdx, blockId);
        }
    }
}

void KVCacheManager::cacheNewBlockOffsets(GenerationRequest& sequence)
{
    auto const& cacheBlocks = sequence.getCacheBlockIds();
    auto& cacheBlocksTensor = sequence.getCacheBlockIndices();
    auto const beamWidth = sequence.getBeamWidth();

    auto* offsetsPtr = bufferCast<tk::KVCacheIndex>(cacheBlocksTensor);
    auto const& offsetsShape = cacheBlocksTensor.getShape();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto const& beamCacheBlock = cacheBlocks[beamIdx];
        auto const blockId = beamCacheBlock.back();
        auto const blockIdx = static_cast<SizeType32>(beamCacheBlock.size() - 1);
        setOffsets(offsetsPtr, offsetsShape, beamIdx, blockIdx, blockId);
    }
}

void KVCacheManager::updateNewBlockPointer(GenerationRequest& sequence, SizeType32 blockIdx)
{
    auto const& cacheBlocks = sequence.getCacheBlockIds();
    auto& cacheBlocksTensor = sequence.getCacheBlockIndices();
    auto const beamWidth = sequence.getBeamWidth();

    auto* offsetsPtr = bufferCast<tk::KVCacheIndex>(cacheBlocksTensor);
    auto const& offsetsShape = cacheBlocksTensor.getShape();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto const& beamCacheBlock = cacheBlocks[beamIdx];
        auto const blockId = beamCacheBlock.at(blockIdx);
        setOffsets(offsetsPtr, offsetsShape, beamIdx, blockIdx, blockId);
    }
}

void KVCacheManager::updateToken(GenerationRequest& sequence, bool addToken)
{
    auto currNumTokens = sequence.getNumTokens();

    if (addToken)
    {
        sequence.addNewTokens(1);
    }
    else
    {
        sequence.removeTokens(1);
    }

    auto newNumTokens = sequence.getNumTokens();

    if (!addToken)
    {
        std::swap(currNumTokens, newNumTokens);
    }

    SizeType32 const cyclicTokenNum = mMaxTokenNum - mSinkBlockTokenLength;
    SizeType32 const nextTokenIdxInCycle = (currNumTokens - mSinkBlockTokenLength) % cyclicTokenNum;
    SizeType32 const nextTokenIdxInCache = mSinkBlockTokenLength + nextTokenIdxInCycle;

    // (nextTokenIdxInCache - mSinkBlockTokenLength) % cyclicTokenNum == 0)
    // <=> nextTokenIdxInCycle == 0
    // <=> nextTokenIdxInCache == mSinkBlockTokenLength
    // => nextTokenIdxInCache % getTokensPerBlock() == 0

    // Check if require a new block
    if (nextTokenIdxInCache % getTokensPerBlock() == 0)
    {
        if (newNumTokens <= mMaxTokenNum)
        {
            if (addToken)
            {
                mBlockManager.allocateBlock(sequence);
                cacheNewBlockOffsets(sequence);
            }
            else
            {
                mBlockManager.releaseLastBlock(sequence);
            }
        }
        else if (sequence.getBeamWidth() > 1 || mEnableBlockReuse)
        {
            TLLM_CHECK_WITH_INFO(addToken, "Remove token is not supported with beam search");
            // Get next block index
            SizeType32 nextBlockIdx = nextTokenIdxInCache / getTokensPerBlock();
            // Replace the shared block with the unshared ones
            mBlockManager.replaceSharedBlock(sequence, nextBlockIdx);
            updateNewBlockPointer(sequence, nextBlockIdx);
        }
    }
}

// ====== hstu modification start =====
void KVCacheManager::appendTokens(GenerationRequest& sequence, SizeType32 numTokens)
{
    auto currNumTokens = sequence.getNumTokens();

    SizeType32 const cyclicTokenNum = mMaxTokenNum - mSinkBlockTokenLength;
    SizeType32 const nextTokenIdxInCycle = (currNumTokens - mSinkBlockTokenLength) % cyclicTokenNum;
    SizeType32 const nextTokenIdxInCache = mSinkBlockTokenLength + nextTokenIdxInCycle;

    SizeType32 numTokensOffset = nextTokenIdxInCache % getTokensPerBlock();
    SizeType32 const tokensIdxOffset = (numTokensOffset == 0) ? 0 : (getTokensPerBlock() - numTokensOffset);

    for (SizeType32 tokenIdx = tokensIdxOffset; tokenIdx < numTokens; tokenIdx += getTokensPerBlock()) {
        if (currNumTokens + tokenIdx < mMaxTokenNum) {
            mBlockManager.allocateBlock(sequence, true);
            cacheNewBlockOffsets(sequence);
        }
    }
    // cacheNewBlockOffsets(sequence);
    sequence.addNewTokens(numTokens);
}
// ====== hstu modification end =====

void KVCacheManager::addToken(RequestIdType requestId)
{
    auto& sequence = getSequence(requestId);
    updateToken(sequence, true);
}

std::optional<BlockKey> KVCacheManager::findNewContextBlock(
    VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const
{
    auto newContextBlockOpt = mBlockManager.findNewContextBlock(uniqueTokens, llmRequest);
    return newContextBlockOpt;
}

void KVCacheManager::addSequence(
    RequestIdType requestId, SizeType32 inputLength, SizeType32 beamWidth, OptionalRef<LlmRequest> llmRequest)
{
    // Need to add the bubble after the sink tokens to use even block size
    inputLength += mSinkBubbleLength;

    auto kvCacheRetentionConfig = llmRequest
        ? llmRequest->getKvCacheRetentionConfig().value_or(executor::KvCacheRetentionConfig())
        : executor::KvCacheRetentionConfig();

    auto const [seqIt, emplaceDone] = [&]
    {
        auto lck = std::scoped_lock(mSequencesMtx);
        return mSequences.emplace(requestId,
            GenerationRequest(requestId, inputLength, beamWidth, mMaxBlocksPerSeq,
                mMinAttentionWindow + mSinkBubbleLength, // When the request has started cycling, disable reuse.
                mBlockManager.getNumPools(), kvCacheRetentionConfig));
    }();
    TLLM_CHECK(emplaceDone);
    auto& sequence = seqIt->second;

    // Get the final token index in kv cache
    SizeType32 const finalTokenKVIdx
        = mSinkBlockTokenLength + ((inputLength - 1 - mSinkBlockTokenLength) % (mMaxTokenNum - mSinkBlockTokenLength));

    // Get block index that with shareAmongBeams=False.
    // For cross kv cache in encoder-decoder models, always shareAmongBeams=True.
    SizeType32 unsharedBlockIdx = -1;
    if ((!sequence.isCyclic() || beamWidth > 1 || finalTokenKVIdx % getTokensPerBlock() > 0) && !isCrossKv())
    {
        unsharedBlockIdx = ((finalTokenKVIdx + 1) % getTokensPerBlock() == 0)
            ? finalTokenKVIdx / getTokensPerBlock() + 1
            : finalTokenKVIdx / getTokensPerBlock();
    }

    // Consider the mTemporaryAttentionWindow when allocating blocks.
    inputLength = std::min(inputLength, mMaxTokenNum + mTemporaryAttentionWindow);
    auto const numContextBlocks = tc::ceilDiv(inputLength, getTokensPerBlock());

    // Get statistics for block allocations/reuse pre request.
    SizeType32 const numAllocTotalBlocksPreRequest = mBlockManager.getNumAllocTotalBlocks();
    SizeType32 const numAllocNewBlocksPreRequest = mBlockManager.getNumAllocNewBlocks();
    SizeType32 const numReusedBlocksPreRequest = mBlockManager.getNumReusedBlocks();
    SizeType32 const numMissedBlocksPreRequest = mBlockManager.getNumMissedBlocks();

    if (!sequence.isCyclic() && mEnableBlockReuse)
    {
        mBlockManager.addSequence(sequence, inputLength, numContextBlocks, *llmRequest);
    }
    else
    {
        if (!mEnableBlockReuse && llmRequest && llmRequest->getKvCacheRetentionConfig().has_value())
        {
            TLLM_LOG_WARNING(
                "Request %d has a retention configuration set, but block reuse is disabled. The retention config will "
                "have no effect.",
                llmRequest->mRequestId);
        }
        mBlockManager.addSequence(sequence, numContextBlocks, unsharedBlockIdx);
        if (mEnableHashKey && llmRequest.has_value() && beamWidth == 1)
        {
            constexpr SizeType32 beamIdx = 0;
            auto const& blockIds = sequence.getCacheBlockIds().at(beamIdx);
            auto const& uniqueTokens = llmRequest->getUniqueTokens(beamIdx);
            auto blockedUniqueTokens
                = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, uniqueTokens.size() - 1, getTokensPerBlock(), true);
            auto blockKeys = buildBlockKeys(blockedUniqueTokens, *llmRequest);
            auto tokensPerBlock = static_cast<size_t>(getTokensPerBlock());
            for (size_t i = 0; i < blockIds.size(); i++)
            {
                auto const& block = mBlockManager.getBlockById(blockIds[i]);
                if (i < blockKeys.size())
                {
                    block->setBlockKey(blockKeys[i], blockKeys[i].uniqueTokens.size() == tokensPerBlock);
                }
                else
                {
                    block->setBlockKey({}, false);
                }
                block->setHash();
                mBlockManager.addBlockToHashMap(block);
            }
        }
    }
    cacheBlockOffsets(sequence);

    if (llmRequest)
    {
        // Update statistics for block allocations/reuse per request.
        llmRequest->updateAllocTotalBlocksPerRequest(
            mBlockManager.getNumAllocTotalBlocks() - numAllocTotalBlocksPreRequest);
        llmRequest->updateAllocNewBlocksPerRequest(mBlockManager.getNumAllocNewBlocks() - numAllocNewBlocksPreRequest);
        llmRequest->updateReusedBlocksPerRequest(mBlockManager.getNumReusedBlocks() - numReusedBlocksPreRequest);
        llmRequest->updateMissedBlocksPerRequest(mBlockManager.getNumMissedBlocks() - numMissedBlocksPreRequest);
    }
}

void KVCacheManager::storeContextBlocks(LlmRequest const& llmRequest)
{
    auto const requestId = llmRequest.mRequestId;
    auto& sequence = getSequence(requestId);
    if (mEnableBlockReuse && !sequence.isCyclic())
    {
        mBlockManager.storeContextBlocks(sequence, llmRequest);
    }
}

void KVCacheManager::removeSequence(RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest)
{
    TLLM_LOG_TRACE("[%s]::%s start", isCrossKv() ? "CROSS" : "SELF", __PRETTY_FUNCTION__);
    auto sequenceNode = [this, requestId]
    {
        std::scoped_lock lock(mSequencesMtx);
        return mSequences.extract(requestId);
    }();
    if (!sequenceNode.empty())
    {
        // Free all blocks for this sequence
        if (mEnableBlockReuse)
        {
            mBlockManager.releaseBlocks(sequenceNode.mapped(), llmRequest);
        }
        else
        {
            mBlockManager.releaseBlocks(sequenceNode.mapped(), {});
        }
        // ===== hstu modification start =====
        mSeqCacheStartPos.erase(requestId);
        // ===== hstu modification end =====
    }
    TLLM_LOG_TRACE("[%s]::%s stop", isCrossKv() ? "CROSS" : "SELF", __PRETTY_FUNCTION__);
}

void KVCacheManager::schedulingRemoveSequence(RequestIdType requestId)
{
    // Mimic Free all blocks for this sequence
    mBlockManager.schedulingReleaseBlocks(requestId);
}

// ===== hstu modification start =====
void KVCacheManager::addSequenceWithEviction(
    RequestIdType requestId, SizeType32 start_pos, SizeType32 length, SizeType32 beamWidth, OptionalRef<LlmRequest> llmRequest)
{
    TLLM_CHECK_WITH_INFO(length <= mMaxBlocksPerSeq * mTokensPerBlock, 
        "Do not accept delta length %d, %d x %d.", length, mMaxBlocksPerSeq, mTokensPerBlock);
    mBlockManager.markSequenceRetained(requestId);

    auto const oldSeqIt = mSequences.find(requestId);
    if (mSequences.end() != oldSeqIt) {
        SizeType32 curLength = oldSeqIt->second.getNumTokens();
        TLLM_CHECK_WITH_INFO(start_pos == mSeqCacheStartPos[requestId] + curLength,
            "Currently, KVCache only supports continuous input sequence. "
            "Last seen position %d to %d -- current input position %d to %d ",
            mSeqCacheStartPos[requestId], mSeqCacheStartPos[requestId] + curLength,
            start_pos, start_pos + length);
    }
    if (mSequences.end() == oldSeqIt && (SizeType32)mSequences.size() == mMaxNumSequences) {
        auto requestIdToEvict = mBlockManager.getRequestIdToEvict();
        mSeqCacheStartPos.erase(requestIdToEvict);
        auto node = mSequences.extract(requestIdToEvict);
        mBlockManager.evictSequence(node.mapped());
    }

    SizeType32 curLength = (mSequences.end() == oldSeqIt) ? 0 : oldSeqIt->second.getNumTokens();
    SizeType32 newLength = curLength + length;
    SizeType32 curBlocks = (curLength + mTokensPerBlock - 1) / mTokensPerBlock;
    SizeType32 newBlocks = (newLength + mTokensPerBlock - 1) / mTokensPerBlock;

    // Default logic for sequence length exceeding limit.
    // Need to be avoided with input-side data length check.
    if (newBlocks > mMaxBlocksPerSeq) {
        SizeType32 numBlocksSelfEvict = newBlocks - mMaxBlocksPerSeq;
        if (numBlocksSelfEvict >= curBlocks) {
            auto node = mSequences.extract(requestId);
            mBlockManager.evictSequence(node.mapped());
            curLength = 0;
        }
        else {
            auto& sequence = mSequences.at(requestId);
            mBlockManager.evictBlocks(sequence, numBlocksSelfEvict);
            // cacheBlockOffsets(sequence);
            curLength = sequence.getNumTokens();
        }
        newLength = curLength + length;
        curBlocks = (curLength + mTokensPerBlock - 1) / mTokensPerBlock;
        newBlocks = (newLength + mTokensPerBlock - 1) / mTokensPerBlock;
        start_pos = start_pos + length - newLength;
    }

    SizeType32 numBlocksRequired = newBlocks - curBlocks;
    while (numBlocksRequired > getNumFreeBlocks()) {
        auto requestIdToEvict = mBlockManager.getRequestIdToEvict();
        // SizeType32 numBlocks = mSequences.at(requestIdToEvict).getCacheBlockIds().at(0).size();
        // if (numBlocks + getNumFreeBlocks() > numBlocksRequired) {
        //     SizeType32 numBlocksToEvict = numBlocksRequired - getNumFreeBlocks();
        //     auto& sequence = mSequences.at(requestIdToEvict);
        //     mBlockManager.evictBlocks(sequence, numBlocksToEvict);
        //     cacheBlockOffsets(sequence);
        //     break;
        // }
        mSeqCacheStartPos.erase(requestIdToEvict);
        auto node = mSequences.extract(requestIdToEvict);
        mBlockManager.evictSequence(node.mapped());
    }

    SizeType32 numAllocTotalBlocksPreRequest = mBlockManager.getNumAllocTotalBlocks();
    SizeType32 numAllocNewBlocksPreRequest = mBlockManager.getNumAllocNewBlocks();
    SizeType32 numReusedBlocksPreRequest = mBlockManager.getNumReusedBlocks();
    SizeType32 numMissedBlocksPreRequest = mBlockManager.getNumMissedBlocks();

    if (mSequences.end() == mSequences.find(requestId)) {
        auto kvCacheRetentionConfig = llmRequest
            ? llmRequest->getKvCacheRetentionConfig().value_or(executor::KvCacheRetentionConfig())
            : executor::KvCacheRetentionConfig();

        auto const [seqIt, emplaceSucess]= mSequences.emplace(requestId,
            GenerationRequest(requestId, newLength, beamWidth, mMaxBlocksPerSeq,
                mMinAttentionWindow, // When the request has started cycling, disable reuse.
                mBlockManager.getNumPools(), kvCacheRetentionConfig));
        TLLM_CHECK(emplaceSucess);
        auto& sequence = seqIt->second;
        mBlockManager.addSequence(sequence, numBlocksRequired, numBlocksRequired);
        cacheBlockOffsets(sequence);
        mSeqCacheStartPos[requestId] = start_pos;
    }
    else {
        auto& sequence = mSequences.at(requestId);
        appendTokens(sequence, newLength - curLength);
        cacheBlockOffsets(sequence);  // no need
    }

    if (llmRequest)
    {
        // Update statistics for block allocations/reuse per request.
        llmRequest->updateAllocTotalBlocksPerRequest(
            mBlockManager.getNumAllocTotalBlocks() - numAllocTotalBlocksPreRequest);
        llmRequest->updateAllocNewBlocksPerRequest(mBlockManager.getNumAllocNewBlocks() - numAllocNewBlocksPreRequest);
        llmRequest->updateReusedBlocksPerRequest(mBlockManager.getNumReusedBlocks() - numReusedBlocksPreRequest);
        llmRequest->updateMissedBlocksPerRequest(mBlockManager.getNumMissedBlocks() - numMissedBlocksPreRequest);
    }
}

void KVCacheManager::offloadSequence(RequestIdType requestId, std::optional<SizeType32> numTokens)
{
}

void KVCacheManager::evictAllSequences(void)
{
    for (auto pair: mSequences) {
        mSeqCacheStartPos.erase(pair.first);
        mBlockManager.evictSequence(pair.second);
    }
    mSequences.clear();
}

SizeType32 KVCacheManager::getNumTokensCached(RequestIdType requestId) const
{
    auto const seq_it = mSequences.find(requestId);
    if (mSequences.end() == seq_it) {
        return 0;
    }
    return seq_it->second.getNumTokens();
}

SizeType32 KVCacheManager::getCacheStartPos(RequestIdType requestId) const
{
    auto const seq_it = mSeqCacheStartPos.find(requestId);
    if (mSeqCacheStartPos.end() == seq_it) {
        return -1;
    }
    return seq_it->second;
}
// ===== hstu modification end =====

SizeType32 KVCacheManager::copyBlockOffsets(ITensor& output, SizeType32 outputSlotOffset, RequestIdType requestId) const
{
    auto const& sequence = getSequence(requestId);
    auto const& cacheBlocksTensor = sequence.getCacheBlockIndices();
    auto const beamWidth = sequence.getBeamWidth();

    auto* dstPtr = bufferCast<tk::KVCacheIndex>(output);
    auto const* srcPtr = bufferCast<tk::KVCacheIndex>(cacheBlocksTensor);
    auto const& dstShape = output.getShape();
    auto const& srcShape = cacheBlocksTensor.getShape();

    SizeType32 constexpr kIdx = 0;
    SizeType32 constexpr vIdx = 1;

    SizeType32 maxBlockCount{0};
    // Get page table for each KV cache pool
    auto const numPools = mBlockManager.getNumPools();

    for (SizeType32 poolIdx = 0; poolIdx < numPools; poolIdx++)
    {
        for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
        {
            auto const beamBlockCount = sequence.getCacheBlockIds()[beamIdx].size();
            auto const copyChunkSize = beamBlockCount * sizeof(tk::KVCacheIndex);
            for (auto xIdx : {kIdx, vIdx})
            {
                auto const srcIndex = tc::flat_index(srcShape.d, poolIdx, beamIdx, xIdx, 0);
                auto const dstIndex = tc::flat_index(dstShape.d, poolIdx, outputSlotOffset + beamIdx, xIdx, 0);
                std::memcpy(dstPtr + dstIndex, srcPtr + srcIndex, copyChunkSize);
            }
            maxBlockCount = std::max<SizeType32>(maxBlockCount, static_cast<SizeType32>(beamBlockCount));
        }
    }
    return maxBlockCount;
}

void KVCacheManager::getBlockOffsetsOfBatch(
    ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize, SizeType32 beamWidth) const
{
    // Get page table for each KV cache pool
    for (auto batchSlotIdx = 0; batchSlotIdx < batchSize; ++batchSlotIdx)
    {
        copyBlockOffsets(output, batchSlotIdx * beamWidth, firstBatchSlotIdx + batchSlotIdx);
    }
}

std::tuple<SizeType32, SizeType32> BaseKVCacheManager::calculateMaxNumBlocks(KvCacheConfig const& config,
    nvinfer1::DataType dtype, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    runtime::BufferManager const& bufferManager, SizeType32 kvFactor)
{
    auto const freeMemFraction = config.freeGpuMemoryFraction.value_or(KvCacheConfig::kDefaultGpuMemFraction);
    TLLM_CHECK_WITH_INFO(freeMemFraction < 1.0F,
        "Invalid freeMemFraction, freeMemFraction (%f) must be smaller than 1.0f", freeMemFraction);
    auto const cacheSizePerToken
        = kv_cache_manager::BaseKVCacheManager::calculateCacheSizePerToken(modelConfig, worldConfig, false, kvFactor);
    auto const cacheSizeBytesPerToken = cacheSizePerToken * BufferDataType(dtype).getSize();
    TLLM_CUDA_CHECK(::cudaDeviceSynchronize());
    auto const [freeMem, totalMem] = tc::getDeviceMemoryInfo(config.useUvm);
    auto maxTokens = static_cast<SizeType32>(freeMemFraction
        * static_cast<double>(freeMem + bufferManager.memoryPoolFree()) / static_cast<double>(cacheSizeBytesPerToken));
    TLLM_LOG_INFO("Memory usage when calculating max tokens in paged kv cache: total: %0.2f GiB, available: %0.2f GiB",
        (totalMem / static_cast<double>(1 << 30)),
        ((freeMem + bufferManager.memoryPoolFree()) / static_cast<double>(1 << 30)));

    // If user specified a number of tokens
    if (config.maxTokens.has_value())
    {
        // If user also specified a free gpu memory fraction, take the min
        if (config.freeGpuMemoryFraction.has_value())
        {
            maxTokens = std::min(config.maxTokens.value(), maxTokens);
            //  else use the number of tokens specified by user
            TLLM_LOG_WARNING(
                "Both freeGpuMemoryFraction (aka kv_cache_free_gpu_mem_fraction) "
                "and maxTokens (aka max_tokens_in_paged_kv_cache) "
                "are set (to %f and %ld, respectively). The smaller value will be used.",
                freeMemFraction, (int64_t) config.maxTokens.value());
        }
        else
        {
            TLLM_LOG_INFO("Maximum kv-cache token overridden by configuration as '%i'.", config.maxTokens.value());
            maxTokens = config.maxTokens.value();
        }
    }
    if (worldConfig.getSize() > 1)
    {
        TLLM_CHECK(worldConfig.validMpiConfig());
        // make sure all ranks use same value for maxTokens
        int64_t _maxTokensRank{maxTokens};
        int64_t _maxTokensWorld{0};
        COMM_SESSION.allreduce(&_maxTokensRank, &_maxTokensWorld, 1, mpi::MpiType::kINT64, mpi::MpiOp::MIN);
        maxTokens = static_cast<SizeType32>(_maxTokensWorld);
    }

    auto const tokensPerBlock = modelConfig.getTokensPerBlock();
    auto const blocksInPrimaryPool = tc::ceilDiv(maxTokens, tokensPerBlock);
    TLLM_LOG_INFO("Number of blocks in KV cache primary pool: %d", blocksInPrimaryPool);

    auto maxTokensSecondary = static_cast<SizeType32>(config.hostCacheSize.value_or(0) / cacheSizeBytesPerToken);
    auto const blocksInSecondaryPool = std::max(0, maxTokensSecondary / tokensPerBlock);
    TLLM_LOG_INFO("Number of blocks in KV cache secondary pool: %d, onboard blocks to primary memory before reuse: %s",
        blocksInSecondaryPool, config.onboardBlocks ? "true" : "false");

    return std::make_tuple(blocksInPrimaryPool, blocksInSecondaryPool);
}

void KVCacheManager::removeToken(RequestIdType requestId)
{
    auto& sequence = getSequence(requestId);
    auto const beamWidth = sequence.getBeamWidth();

    TLLM_CHECK_WITH_INFO(beamWidth == 1, "removeToken does not support beamWidth > 1");
    if (sequence.getNumTokens() == 0)
    {
        return;
    }
    updateToken(sequence, false);
}

void KVCacheManager::rewindKVCache(RequestIdType requestId, SizeType32 rewindLengths)
{
    for (SizeType32 si = 0; si < rewindLengths; ++si)
    {
        removeToken(requestId);
    }
}

GenerationRequest const& KVCacheManager::getSequence(RequestIdType requestId) const
{
    auto lck = std::scoped_lock(mSequencesMtx);
    return mSequences.at(requestId);
}

GenerationRequest& KVCacheManager::getSequence(RequestIdType requestId)
{
    auto lck = std::scoped_lock(mSequencesMtx);
    return mSequences.at(requestId);
}

SizeType32 BaseKVCacheManager::getSinkBubbleLength(SizeType32 sinkTokenLen, SizeType32 tokensPerBlock)
{
    auto const sinkTokensInLastBlock = sinkTokenLen % tokensPerBlock;
    auto const sinkBubbleLength = sinkTokensInLastBlock == 0 ? 0 : tokensPerBlock - sinkTokensInLastBlock;
    return sinkBubbleLength;
}

bool KVCacheManager::schedulingHasFreeBlocks(SizeType32 numRequired) const
{
    return mBlockManager.schedulingHasFreeBlocks(numRequired);
}

std::vector<std::vector<SizeType32>> const& KVCacheManager::getCacheBlockIds(RequestIdType requestId) const
{
    // ===== hstu modification start =====
    const static std::vector<std::vector<SizeType32>> empty(1);
    if (mSequences.end() == mSequences.find(requestId)) {
        return empty;
    }
    // ===== hstu modification end =====
    return getSequence(requestId).getCacheBlockIds();
}

std::vector<std::vector<std::vector<SizeType32>>> KVCacheManager::getBatchCacheBlockIds(
    std::vector<LlmRequest::RequestIdType> const& requestIds) const
{
    std::vector<std::vector<std::vector<SizeType32>>> result{};
    result.reserve(requestIds.size());
    for (auto const& requestId : requestIds)
    {
        auto const& sequence = getSequence(requestId);
        result.emplace_back(sequence.getCacheBlockIds());
    }
    return result;
}

std::vector<SizeType32> KVCacheManager::getNewlyAllocatedBlockIds(LlmRequest::RequestIdType requestId) const
{
    return mBlockManager.getNewlyAllocatedBlockIds(mSequences.at(requestId));
}

runtime::ITensor::SharedPtr KVCacheManager::getPrimaryPool(SizeType32 layer_idx) const
{
    return mBlockManager.getPrimaryPool(mBlockManager.getLayerPoolIdx(layer_idx));
}

SizeType32 KVCacheManager::getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const
{

    auto const blockRequirementsPerSequence = KVCacheManager::calculateMaxBlockRequirements(
        inputLength, outputLength, mSinkBlockTokenLength, mMaxAttentionWindow, mMaxBeamWidth, mTokensPerBlock);

    return mBlockManager.getNumPrimaryBlocks() / blockRequirementsPerSequence;
}

SizeType32 KVCacheManager::calculateMaxBlockRequirementsPerBeam(
    SizeType32 sequenceLength, SizeType32 sinkTokenLength, SizeType32 maxAttentionWindow, SizeType32 tokensPerBlock)
{
    auto const sinkBubbleLength = BaseKVCacheManager::getSinkBubbleLength(sinkTokenLength, tokensPerBlock);
    auto const actualSeqLen = std::min(sequenceLength, maxAttentionWindow);
    auto actualMaxTokenNum = actualSeqLen + sinkBubbleLength;
    return tc::ceilDiv(actualMaxTokenNum, tokensPerBlock);
}

SizeType32 KVCacheManager::calculateMaxBlockRequirements(SizeType32 inputLength, SizeType32 outputLength,
    SizeType32 sinkTokenLength, SizeType32 maxAttentionWindow, SizeType32 beamWidth, SizeType32 tokensPerBlock)
{
    // We split beam width > 1, as it introduces a lot of complexity.
    auto const wholeSequenceLength = inputLength + outputLength;
    if (beamWidth == 1)
    {
        return KVCacheManager::calculateMaxBlockRequirementsPerBeam(
            wholeSequenceLength, sinkTokenLength, maxAttentionWindow, tokensPerBlock);
    }

    // If the whole attention window can fit in the output, then we can simply multiply the cost of a sequence of length
    // max attention window by the beam width.
    if (maxAttentionWindow <= outputLength)
    {
        return KVCacheManager::calculateMaxBlockRequirementsPerBeam(
                   maxAttentionWindow, sinkTokenLength, maxAttentionWindow, tokensPerBlock)
            * beamWidth;
    }

    // Otherwise, we calculate how many tokens will be in output blocks.
    auto const effectiveAttentionWindow = std::min(maxAttentionWindow, wholeSequenceLength);
    auto const numContextTokensInAttentionWindow
        = effectiveAttentionWindow - outputLength; // This is positive because we handled the other case above.
    auto const sinkBubbleLength = BaseKVCacheManager::getSinkBubbleLength(sinkTokenLength, tokensPerBlock);
    auto const numContextBlocks = (numContextTokensInAttentionWindow + sinkBubbleLength) / tokensPerBlock;
    auto const leftoverContextToken = numContextTokensInAttentionWindow - numContextBlocks * tokensPerBlock;
    auto const numOutputBlocks = tc::ceilDiv(outputLength + leftoverContextToken, tokensPerBlock);
    return numContextBlocks + numOutputBlocks * beamWidth;
}

[[nodiscard]] SizeType32 KVCacheManager::calculateMaxAttentionWindow(SizeType32 inputLength, SizeType32 outputLength,
    SizeType32 sinkTokenLength, SizeType32 blockCapacity, SizeType32 beamWidth, SizeType32 tokensPerBlock)
{
    // The function that gives the number of blocks required given an attention window is only linear by part. It is
    // however, monotonically increasing in the attention window. Therefore, we need to find in which part of the
    // function we are. First, are we in a case where not even the entire output will fit?
    auto const outputBlockRequirements
        = calculateMaxBlockRequirements(0, outputLength, sinkTokenLength, outputLength, beamWidth, tokensPerBlock);
    if (outputBlockRequirements > blockCapacity)
    {
        return (blockCapacity / beamWidth) * tokensPerBlock;
    }

    // Otherwise, we need to determine how many context tokens we can fit on top of the output tokens. First, there are
    // a few context tokens we might be able to fit 'for free' because the output is not a multiple of the number of
    // tokens per block.
    auto const leftoverBlockCapacity = blockCapacity - outputBlockRequirements;
    return std::min(outputLength + leftoverBlockCapacity * tokensPerBlock, inputLength + outputLength);
}
} // namespace tensorrt_llm::batch_manager::kv_cache_manager
